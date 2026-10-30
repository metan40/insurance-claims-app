"""Évaluation reproductible de la recherche documentaire déployée.

Usage, depuis la racine du projet :

    python evaluation/evaluate_pipeline.py

Deux jeux de questions sont mesurés :

* ``titres`` : les 201 titres du corpus servent de questions et le document de
  la même ligne est la cible. Ce protocole est biaisé en faveur du signal
  lexical, qui compare justement la question aux titres (poids 3).
* ``reformulations`` : 40 questions réécrites avec un vocabulaire différent
  (``evaluation/paraphrase_questions.csv``). Elles mesurent la robustesse au
  changement de vocabulaire. Ce jeu a été rédigé pour le projet et n'a pas été
  validé par un expert métier ; une seule cible est retenue par question.

* ``tenu_a_part`` : 30 autres reformulations, figées *avant* la rédaction du
  dictionnaire de synonymes (``src/question_parsing.py``) et jamais utilisées
  pour l'ajuster. C'est la mesure la plus honnête de l'apport des synonymes.

Quatre classements sont comparés : similarité cosinus TF-IDF seule, score
lexical seul, pipeline hybride sans synonymes (version 7.6) et pipeline de
production avec analyse de la question (``HybridRetriever.rank``).
Les sorties sont écrites dans ``evaluation/results/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import DEFAULT_TOP_K, HybridRetriever  # noqa: E402

CORPUS_FILE = PROJECT_ROOT / "Data" / "insurance-fr.csv"
PARAPHRASE_FILE = PROJECT_ROOT / "evaluation" / "paraphrase_questions.csv"
HELDOUT_FILE = PROJECT_ROOT / "evaluation" / "heldout_questions.csv"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
CUTOFFS = (1, 3, 5)


def _load_reformulations(path: Path, titles: list[str]) -> list[tuple[str, int]]:
    frame = pd.read_csv(path)
    for row in frame.itertuples():
        if titles[row.expected_index] != row.expected_title:
            raise ValueError(f"Cible incohérente pour la question : {row.question}")
    return list(zip(frame["question"], frame["expected_index"].astype(int)))


def load_question_sets(titles: list[str]) -> dict[str, list[tuple[str, int]]]:
    return {
        "titres": [(title, index) for index, title in enumerate(titles)],
        "reformulations": _load_reformulations(PARAPHRASE_FILE, titles),
        "tenu_a_part": _load_reformulations(HELDOUT_FILE, titles),
    }


def ranking_methods(retriever: HybridRetriever, baseline: HybridRetriever):
    """Méthodes comparées ; ``baseline`` est le même pipeline sans synonymes."""
    def cosine_only(question: str) -> list[int]:
        return [int(i) for i in np.argsort(-baseline.cosine_scores(question), kind="stable")]

    def lexical_only(question: str) -> list[int]:
        scores = baseline.lexical_scores(question)
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    def hybrid_without_synonyms(question: str) -> list[int]:
        return [item.index for item in baseline.rank(question)]

    def hybrid(question: str) -> list[int]:
        return [item.index for item in retriever.rank(question)]

    return {
        "cosinus_tfidf": cosine_only,
        "lexical_seul": lexical_only,
        "hybride_sans_synonymes": hybrid_without_synonyms,
        "hybride_production": hybrid,
    }


def summarize(ranks: list[int | None]) -> dict[str, float]:
    count = len(ranks)
    summary = {
        f"recall@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / count, 4)
        for k in CUTOFFS
    }
    summary[f"mrr@{DEFAULT_TOP_K}"] = round(
        sum(1.0 / r for r in ranks if r is not None and r <= DEFAULT_TOP_K) / count, 4
    )
    summary["questions"] = count
    return summary


def main(write_results: bool = True, verbose: bool = True) -> dict:
    corpus = pd.read_csv(CORPUS_FILE)
    titles = corpus["title"].fillna("").astype(str).tolist()
    documents = corpus["content"].fillna("").astype(str).tolist()
    retriever = HybridRetriever(titles, documents)
    baseline = HybridRetriever(titles, documents, use_synonyms=False)

    rows, summary = [], {}
    for set_name, questions in load_question_sets(titles).items():
        summary[set_name] = {}
        for method_name, method in ranking_methods(retriever, baseline).items():
            ranks = []
            for question, target in questions:
                order = method(question)
                rank = order.index(target) + 1 if target in order else None
                ranks.append(rank)
                rows.append({
                    "jeu": set_name,
                    "methode": method_name,
                    "question": question,
                    "cible": titles[target],
                    "rang_cible": rank,
                    "premier_resultat": titles[order[0]],
                })
            summary[set_name][method_name] = summarize(ranks)

    if write_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "evaluation_par_question.csv", index=False)
        (RESULTS_DIR / "evaluation_synthese.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for set_name, methods in (summary.items() if verbose else ()):
        print(f"\nJeu « {set_name} »")
        for method_name, values in methods.items():
            print(
                f"  {method_name:20s} "
                + "  ".join(f"{key}={value}" for key, value in values.items())
            )
    return summary


if __name__ == "__main__":
    main()
