# ---------------------------------------------------------------------------
# Projet  : Insurance Claims Assistance (assistant documentaire et aide à l'analyse des sinistres)
# Fichier : evaluation/comparer_versions.py
# Auteur  : Thierry MBARGA
# Rôle    : Comparaison appariée de deux configurations sur les mêmes
#           questions
# Lancer  : python evaluation/comparer_versions.py  (options : --help)
# ---------------------------------------------------------------------------
"""Comparaison appariée de deux configurations sur les mêmes questions.

Entrées : deux fichiers ``statuts_heldout_*.csv`` produits par
``mesurer_statuts.py`` (mêmes questions, même nombre de passages).

Pour chaque question i, on compte les passages conformes dans chaque
configuration (k_i^A et k_i^B). Comparaison appariée par question :
* gagnée par B si k_i^B > k_i^A, perdue si k_i^B < k_i^A, égale sinon ;
* test du signe exact (binomial, p = 1/2) sur les questions non égales ;
la question est l'unité statistique, ce qui évite de traiter comme
indépendants les passages répétés d'une même question.

Usage :
    python evaluation/comparer_versions.py A.csv B.csv --noms "19.4.0" "19.5.0"
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict


def lire(path: str) -> "OrderedDict[str, list[bool]]":
    par_question: "OrderedDict[str, list[bool]]" = OrderedDict()
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            par_question.setdefault(row["question"], []).append(row.get("reponse_conforme") == "True")
    return par_question


def signe_exact(gagnees: int, perdues: int) -> float:
    n = gagnees + perdues
    if n == 0:
        return 1.0
    k = min(gagnees, perdues)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--noms", nargs=2, default=["A", "B"])
    args = parser.parse_args()
    a, b = lire(args.a), lire(args.b)
    na, nb = args.noms
    communes = [q for q in a if q in b]
    print(f"| question | {na} | {nb} |\n|---|---|---|")
    gagnees = perdues = 0
    total_a = total_b = passages_a = passages_b = 0
    for q in communes:
        ka, kb = sum(a[q]), sum(b[q])
        total_a, total_b = total_a + ka, total_b + kb
        passages_a, passages_b = passages_a + len(a[q]), passages_b + len(b[q])
        gagnees += kb > ka
        perdues += kb < ka
        print(f"| {q[:70]} | {ka}/{len(a[q])} | {kb}/{len(b[q])} |")
    print(f"\nQuestions communes : {len(communes)}")
    print(f"Passages conformes : {na} {total_a}/{passages_a} ; {nb} {total_b}/{passages_b}")
    print(f"Questions où {nb} fait mieux : {gagnees} ; moins bien : {perdues} ; "
          f"test du signe exact p = {signe_exact(gagnees, perdues):.4f}")


if __name__ == "__main__":
    main()
