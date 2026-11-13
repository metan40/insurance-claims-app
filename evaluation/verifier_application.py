"""Vérifie une application en marche (locale ou distante) par son API HTTP.

Le script pose les questions de ``questions_jury.csv`` comme le ferait un
utilisateur, plusieurs fois chacune, et affiche pour chaque passage :
le statut, le moteur de recherche utilisé, la source citée et vérifiée,
les faits obligatoires retrouvés et la réponse affichée.

Il sert à deux choses :

* prouver que l'application démarrée en local (Docker) fonctionne de bout en
  bout, sans le serveur distant ;
* choisir, sur mesure, les questions de démonstration : une réponse complète,
  une réponse partielle, une question hors des documents sans invention.

Uniquement la bibliothèque standard : aucun paquet à installer.

Exemples :
    python evaluation/verifier_application.py --url http://localhost:7860
    python evaluation/verifier_application.py --url https://metan40-insurance-claims-assistance.hf.space --repetitions 1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.question_parsing import normalize_text  # noqa: E402  (pur Python)

QUESTIONS = ROOT / "evaluation" / "questions_jury.csv"
REFERENCES = ROOT / "evaluation" / "reponses_reference.csv"
# Une question hors des documents est réussie si l'application ne conclut pas
# à partir de rien : réponse non trouvée ou simples extraits vérifiés.
STATUTS_SANS_CONCLUSION = {"not_found", "fallback"}


def get(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(url: str, question: str, timeout: float) -> dict:
    data = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(f"{url}/search", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(url: str, minutes: float, embeddings: bool) -> dict:
    deadline = time.time() + minutes * 60
    last = {}
    while time.time() < deadline:
        try:
            last = get(f"{url}/ready")
            checks = last.get("checks", {})
            if last.get("status") == "ready" and (not embeddings or checks.get("embedding_search_loaded")):
                return last
        except Exception as error:  # serveur pas encore démarré
            last = {"erreur": str(error)}
        time.sleep(15)
    raise SystemExit(f"Application non prête après {minutes} min : {last}")


def load_facts() -> dict[str, list[list[str]]]:
    with REFERENCES.open(encoding="utf-8") as handle:
        return {normalize_text(r["question"]): [[normalize_text(v) for v in g.split("|") if v.strip()]
                                                for g in r["faits_obligatoires"].split(";") if g.strip()]
                for r in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--attente-minutes", type=float, default=60)
    parser.add_argument("--sans-embeddings", action="store_true", help="ne pas attendre l'index bge-m3")
    parser.add_argument("--delai", type=float, default=600, help="délai maximal d'une réponse (secondes)")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    print("Version :", get(f"{url}/version"))
    ready = wait_until_ready(url, args.attente_minutes, not args.sans_embeddings)
    print("Prête :", json.dumps(ready, ensure_ascii=False))

    facts = load_facts()
    with QUESTIONS.open(encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))

    bilan: dict[str, Counter] = defaultdict(Counter)
    for item in questions:
        q, attendu, source = item["question"], item["comportement_attendu"], item["source_attendue"]
        groups = facts.get(normalize_text(q), [])
        print(f"\n=== [{attendu}] {q}")
        for passage in range(1, args.repetitions + 1):
            started = time.perf_counter()
            try:
                r = ask(url, q, args.delai)
            except Exception as error:
                print(f"  passage {passage} : ERREUR {error}")
                bilan[q]["erreur"] += 1
                continue
            seconds = round(time.perf_counter() - started, 1)
            answer = str(r.get("answer", ""))
            typed = r.get("typed_answer") or {}
            cited = {e.get("source_id") for e in typed.get("evidence") or [] if e.get("verified")}
            cited_titles = [s["title"] for s in r.get("results", []) if s.get("source_id") in cited]
            text = f" {normalize_text(answer)} "
            found = sum(1 for g in groups if any(f" {v}" in text for v in g))
            status = r.get("answer_status")
            if attendu == "absente":
                ok = status in STATUTS_SANS_CONCLUSION
            else:
                ok = bool(groups) and found == len(groups)
            bilan[q]["reussi" if ok else "echoue"] += 1
            bilan[q][status] += 1
            print(f"  passage {passage} : statut={status} moteur={r.get('retrieval_engine')} {seconds} s "
                  f"faits={found}/{len(groups)} bonne_source_citee={source in cited_titles} "
                  f"masquees={len(typed.get('unsupported_sentences') or [])} -> {'OK' if ok else 'NON'}")
            print("    réponse :", answer.replace("\n", " ")[:700])

    print("\nBILAN (réussites / passages)")
    for item in questions:
        c = bilan[item["question"]]
        n = sum(v for k, v in c.items() if k in ("reussi", "echoue", "erreur"))
        statuts = {k: v for k, v in c.items() if k not in ("reussi", "echoue")}
        print(f"  {c['reussi']}/{n}  [{item['comportement_attendu']}] {item['question'][:70]}  {statuts}")


if __name__ == "__main__":
    main()
