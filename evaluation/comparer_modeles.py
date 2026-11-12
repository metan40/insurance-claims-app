"""Tableau comparatif des mesures de ``mesurer_statuts.py``, avec intervalles de
confiance (Wilson, 95 %) et test de McNemar apparié entre modèles.

Usage : python evaluation/comparer_modeles.py <dossier contenant les statuts_*>

* Intervalle de Wilson : pour une proportion p̂ = k/n, z = 1,96,
  centre = (p̂ + z²/2n) / (1 + z²/n),
  demi-largeur = z·√(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n).
* McNemar exact : sur les mêmes questions (et le même passage), b = succès du
  modèle A seul, c = succès du modèle B seul ; p = 2·P(X ≤ min(b, c)) avec
  X ~ Binomiale(b + c, 1/2), plafonnée à 1.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

Z = 1.96


def wilson(k: int, n: int) -> str:
    if not n:
        return ""
    p = k / n
    denominator = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denominator
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denominator
    return f"{k}/{n} = {p:.0%} [{max(0, centre - half):.0%} ; {min(1, centre + half):.0%}]"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(folder: Path):
    summaries, details = [], {}
    for path in sorted(folder.rglob("statuts_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(summary)
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as handle:
                details[(summary["jeu"], summary["modele"])] = list(csv.DictReader(handle))
    return summaries, details


def main(folder: str) -> None:
    summaries, details = load(Path(folder))
    summaries.sort(key=lambda s: (s.get("jeu", ""), s.get("modele", "")))
    print("## Évaluation de l'assistant par modèle\n")
    print("Intervalles de confiance de Wilson à 95 %.\n")
    print("| Jeu | Modèle | Réponses conformes à la référence | Bonne source citée et vérifiée | "
          "Statuts | « Absente » à tort | Phrases masquées | Conditions ajoutées | Temps moyen / max (s) | Erreurs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        statuts = ", ".join(f"{k} {v}" for k, v in sorted((s.get("statuts") or {}).items()))
        print(f"| {s.get('jeu')} | {s.get('modele')} "
              f"| {wilson(s.get('reponses_conformes', 0), s.get('avec_reference', 0))} "
              f"| {wilson(s.get('bonne_source_citee_et_verifiee_n', 0), s.get('avec_source_attendue', 0))} "
              f"| {statuts} | {s.get('reponse_absente_alors_que_la_bonne_source_etait_lue')} "
              f"| {s.get('reponses_avec_phrases_masquees')} | {s.get('reponses_avec_condition_ajoutee')} "
              f"| {s.get('temps_moyen_secondes')} / {s.get('temps_max_secondes')} | {s.get('erreurs')} |")

    print("\n### Comparaisons appariées (McNemar exact, réponses conformes)\n")
    print("| Jeu | Modèle A | Modèle B | A seul réussit | B seul réussit | p |")
    print("|---|---|---|---|---|---|")
    jeux = sorted({jeu for jeu, _ in details})
    for jeu in jeux:
        models = sorted(m for j, m in details if j == jeu)
        for a, b in combinations(models, 2):
            key = lambda r: (r["question"], r.get("passage", "1"))
            ra = {key(r): r.get("reponse_conforme") == "True" for r in details[(jeu, a)] if r.get("reponse_conforme")}
            rb = {key(r): r.get("reponse_conforme") == "True" for r in details[(jeu, b)] if r.get("reponse_conforme")}
            common = ra.keys() & rb.keys()
            only_a = sum(1 for k in common if ra[k] and not rb[k])
            only_b = sum(1 for k in common if rb[k] and not ra[k])
            print(f"| {jeu} | {a} | {b} | {only_a} | {only_b} | {mcnemar(only_a, only_b):.3f} |")
    if not summaries:
        print("\nAucune mesure trouvée.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/results")
