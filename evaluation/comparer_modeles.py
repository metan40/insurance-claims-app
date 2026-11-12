"""Tableau comparatif des mesures de ``mesurer_statuts.py`` (une ligne par modèle et par jeu).

Usage : python evaluation/comparer_modeles.py <dossier contenant les statuts_*.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = [
    ("modele", "Modèle"),
    ("jeu", "Jeu"),
    ("questions", "Questions"),
    ("part_bonne_source_citee_et_verifiee", "Bonne source citée et vérifiée"),
    ("statuts", "Statuts"),
    ("reponse_absente_alors_que_la_bonne_source_etait_lue", "« Absente » à tort"),
    ("reponses_avec_phrases_masquees", "Réponses avec phrases masquées"),
    ("reponses_avec_condition_ajoutee", "Conditions ajoutées"),
    ("appels_avec_json_invalide", "JSON invalides"),
    ("temps_moyen_secondes", "Temps moyen (s)"),
    ("temps_max_secondes", "Temps max (s)"),
    ("erreurs", "Erreurs"),
]


def cell(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key} {count}" for key, count in sorted(value.items()))
    if isinstance(value, float) and value <= 1 and value >= 0:
        return f"{value:.0%}"
    return "" if value is None else str(value)


def main(folder: str) -> None:
    summaries = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(Path(folder).rglob("statuts_*.json"))]
    summaries.sort(key=lambda s: (s.get("jeu", ""), s.get("modele", "")))
    print("## Évaluation de l'assistant par modèle\n")
    print("| " + " | ".join(title for _, title in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for summary in summaries:
        print("| " + " | ".join(cell(summary.get(key)) for key, _ in COLUMNS) + " |")
    if not summaries:
        print("\nAucune mesure trouvée.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/results")
