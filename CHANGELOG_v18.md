# V18 — expérimentation SVD/LSA pour la recherche documentaire

## Ajouts

- Ajout de `src/compare_tfidf_svd_embeddings.py` pour comparer TF-IDF seul, TF-IDF + SVD tronquée, fusion RRF TF-IDF/SVD et, en option, embeddings et fusion SVD/embeddings.
- Ajout de `docs/evaluation_tfidf_svd_embeddings_v18.md` avec protocole, résultats exécutés, interprétation et limites.
- Ajout des fichiers de résultats `Data/tfidf_svd_embeddings_comparison.csv` et `Data/tfidf_svd_embeddings_comparison_summary.json`.
- L'application de production n'a pas été modifiée : l'expérience est isolée et ne change pas le comportement existant.

## Résultat préliminaire

Sur le proxy de 201 titres et documents correspondants, la fusion TF-IDF/SVD à 50 composantes donne Recall@5 = 65,67 % contre 63,68 % pour la configuration TF-IDF du même script. Ce gain (+1,99 point) est exploratoire, non validé par annotation humaine et ne suffit pas à justifier une modification de production.

La comparaison aux embeddings reste à exécuter avec `python src/compare_tfidf_svd_embeddings.py --with-embeddings`. Le rapport explicite les limites du protocole et ne revendique pas de résultat non mesuré.
