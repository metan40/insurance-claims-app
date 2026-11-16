"""Vérification de justification par inférence textuelle (NLI) : mesure.

Problème mesuré le 13/11/2026 : sur des questions hors des documents, le modèle
cite un vrai extrait (la citation est vérifiée) mais sur un autre sujet, puis
conclut avec les mots de la question (« Non, votre vélo n'est pas couvert car
le local n'était pas fermé à clé »). Le contrôle actuel vérifie que l'extrait
existe, pas qu'il justifie la phrase.

Idée testée : un modèle d'inférence textuelle multilingue estime, pour chaque
phrase affichée h et chaque extrait vérifié p cité, la probabilité
P(implication | p, h) que p implique h. Le score d'une réponse est

    score = min sur les phrases h ( max sur les extraits p de P(implication | p, h) )

c'est-à-dire : la phrase la moins justifiée, par son meilleur extrait. Si
score < seuil, l'application ne conclut pas (elle montre les extraits).

Protocole, fixé avant toute mesure :

* questions hors des documents : partie « dev » / « test » écrite dans
  ``questions_hors_documents.csv`` ; bonne conduite = ne pas conclure ;
* questions avec référence (jeu tenu à part) : indice pair = dev, impair = test ;
  bonne conduite = réponse conforme (tous les faits obligatoires) ;
* les deux questions officielles (catégorie ``officielle_nuancee``) ne sont pas
  notées automatiquement : leur bonne réponse est un « je ne sais pas » étayé,
  jugé à la lecture ;
* le seuil est choisi sur dev seulement : il maximise
  (taux de refus correct hors documents) + (taux de réponses conformes) ;
* il est appliqué une seule fois sur test ; comparaison appariée avant/après
  (McNemar exact sur les mêmes passages) et intervalles de Wilson.

Usage : python evaluation/analyser_justification.py enregistrements.jsonl
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CONCLUT = ("answered", "partial")
SEUILS = [round(0.05 * k, 2) for k in range(0, 20)]


def phrases(text: str) -> list[str]:
    text = re.sub(r"\[S\d+\]", " ", text or "")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n = 0"
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.0%} [{max(0, centre - half):.0%} ; {min(1, centre + half):.0%}]"


def mcnemar_exact(b: int, c: int) -> float:
    """p-valeur bilatérale exacte : b et c = paires discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


class Juge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    def implication(self, premisses: list[str], hypothese: str) -> float:
        if not premisses:
            return 0.0
        batch = self.tokenizer(premisses, [hypothese] * len(premisses), truncation=True,
                               max_length=512, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**batch).logits, dim=-1)[:, self.entail]
        return float(probs.max())


def score(juge: Juge, record: dict) -> float | None:
    """None si l'application ne conclut pas déjà (rien à vérifier)."""
    if record["statut"] not in CONCLUT:
        return None
    hs = phrases(record.get("texte_du_modele_affiche", ""))
    if not hs:
        return None
    quotes = [q for q in record.get("extraits_verifies", []) if q.strip()]
    premisses = quotes + ([" ".join(quotes)] if len(quotes) > 1 else [])
    return min(juge.implication(premisses, h) for h in hs)


def bonne_conduite(record: dict, s: float | None, seuil: float | None) -> bool:
    bloque = seuil is not None and s is not None and s < seuil
    if record["jeu"] == "hors_documents":
        return record["statut"] not in CONCLUT or bloque
    return bool(record.get("reponse_conforme")) and not bloque


def partie(record: dict) -> str:
    if record["jeu"] == "hors_documents":
        return record["partie"]
    return "dev" if record["indice_question"] % 2 == 0 else "test"


def main(path: str) -> None:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    juge = Juge()
    for r in records:
        r["score"] = score(juge, r)

    notees = [r for r in records if r.get("categorie") != "officielle_nuancee"
              and (r["jeu"] == "hors_documents" or r.get("reponse_conforme") is not None)]

    def taux(rs, seuil):
        neg = [r for r in rs if r["jeu"] == "hors_documents"]
        pos = [r for r in rs if r["jeu"] != "hors_documents"]
        kn = sum(bonne_conduite(r, r["score"], seuil) for r in neg)
        kp = sum(bonne_conduite(r, r["score"], seuil) for r in pos)
        return kn, len(neg), kp, len(pos)

    dev = [r for r in notees if partie(r) == "dev"]
    test = [r for r in notees if partie(r) == "test"]

    print("## Choix du seuil sur dev\n")
    print("| seuil | refus corrects hors documents | réponses conformes |")
    print("|---|---|---|")
    best, best_value = None, -1.0
    for seuil in [None] + SEUILS:
        kn, nn, kp, np_ = taux(dev, seuil)
        value = kn / max(1, nn) + kp / max(1, np_)
        print(f"| {'aucun' if seuil is None else seuil} | {wilson(kn, nn)} | {wilson(kp, np_)} |")
        if seuil is not None and value > best_value + 1e-9:
            best, best_value = seuil, value
    print(f"\nSeuil retenu sur dev : **{best}**\n")

    print("## Test (une seule fois)\n")
    for jeu, nom in (("hors_documents", "Refus corrects hors documents"), ("heldout", "Réponses conformes")):
        rs = [r for r in test if (r["jeu"] == "hors_documents") == (jeu == "hors_documents")]
        avant = [bonne_conduite(r, r["score"], None) for r in rs]
        apres = [bonne_conduite(r, r["score"], best) for r in rs]
        b = sum(1 for x, y in zip(avant, apres) if not x and y)
        c = sum(1 for x, y in zip(avant, apres) if x and not y)
        print(f"- {nom} : avant {wilson(sum(avant), len(rs))} ; après {wilson(sum(apres), len(rs))} ; "
              f"gagnés {b}, perdus {c}, McNemar exact p = {mcnemar_exact(b, c):.4f}")

    print("\n## Questions officielles (à lire)\n")
    for r in records:
        if r.get("categorie") == "officielle_nuancee":
            s = r["score"]
            decision = "conclut" if r["statut"] in CONCLUT and not (s is not None and s < best) else "ne conclut pas"
            print(f"- {r['question'][:60]}… passage {r['passage']} : statut={r['statut']} "
                  f"score={s if s is None else round(s, 3)} -> {decision}")
            print(f"  réponse : {r['reponse_affichee'][:400]}")

    print("\n## Détail des scores (réponses qui concluent)\n")
    for r in records:
        if r["score"] is not None:
            print(f"- [{r['jeu']}/{partie(r)}] score={r['score']:.3f} conforme={r.get('reponse_conforme')} "
                  f"{r['question'][:55]} :: {r['texte_du_modele_affiche'][:160]}")


if __name__ == "__main__":
    main(sys.argv[1])
