# V18 — comparaison TF-IDF, SVD/LSA et embeddings

## Objectif

Cette expérience ajoute une voie de recherche fondée sur TF-IDF puis SVD tronquée (LSA), afin de la comparer à TF-IDF seul et, lorsque les poids du modèle sont disponibles, aux embeddings déjà utilisés par le projet. Il s'agit d'un outil d'évaluation hors ligne : le pipeline de production n'est pas modifié.

## Reproductibilité

Depuis la racine du projet :

```bash
python src/compare_tfidf_svd_embeddings.py
python src/compare_tfidf_svd_embeddings.py --components 50,100,200
python src/compare_tfidf_svd_embeddings.py --with-embeddings --components 50,100,200
```

La dernière commande nécessite `sentence-transformers` et peut télécharger les poids de `paraphrase-multilingual-MiniLM-L12-v2` au premier lancement. Les fichiers générés sont `Data/tfidf_svd_embeddings_comparison.csv` (résultats par question et méthode) et `Data/tfidf_svd_embeddings_comparison_summary.json` (synthèse et limites).

## Résultats exécutés dans l'environnement disponible

Le test TF-IDF/SVD a été exécuté sur les 201 lignes de `Data/insurance-fr.csv`. La configuration TF-IDF de ce script utilise des unigrammes et bigrammes, accents normalisés et TF sous-linéaire ; elle n'est donc pas strictement identique à l'ancienne référence TF-IDF de V17.

| Méthode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 29,85 % | 54,23 % | 63,68 % | 0,4428 |
| TF-IDF + SVD (50 composantes) | 28,36 % | 51,74 % | 61,69 % | 0,4344 |
| Fusion RRF(TF-IDF, SVD-50) | 29,85 % | 52,24 % | 65,67 % | 0,4465 |
| TF-IDF + SVD (100 composantes) | 28,36 % | 54,23 % | 64,68 % | 0,4463 |
| Fusion RRF(TF-IDF, SVD-100) | 29,35 % | 53,23 % | 65,67 % | 0,4460 |
| TF-IDF + SVD (200 composantes) | 29,85 % | 55,22 % | 64,18 % | 0,4455 |
| Fusion RRF(TF-IDF, SVD-200) | 29,85 % | 54,73 % | 64,18 % | 0,4439 |

## Interprétation prudente

Sur ce proxy automatique, SVD seule n'améliore pas uniformément le classement par rapport à TF-IDF. La fusion RRF avec 50 composantes augmente Recall@5 de 63,68 % à 65,67 % (+1,99 points) et MRR de 0,4428 à 0,4465, mais ce résultat ne prouve pas une amélioration de pertinence métier. Les titres de chaque ligne sont utilisés comme requêtes et le document de cette même ligne est considéré comme la cible. Ce protocole facilite la comparaison reproductible, mais il ne remplace pas des questions réelles annotées par des personnes.

La comparaison aux embeddings n'a pas été exécutée dans cette campagne ; elle est disponible via `--with-embeddings` et doit être lancée dans un environnement disposant du paquet et des poids du modèle. Aucun résultat d'embeddings n'est donc revendiqué par cette version.

## Limites et suite

- La cible attendue issue de la même ligne est un proxy, pas une annotation humaine de pertinence.
- Le jeu est petit (201 documents) et les titres sont issus du même corpus.
- Le benchmark n'évalue ni la génération, ni l'appui réel des citations, ni la latence de bout en bout.
- Il faut annoter un jeu de questions métier indépendant, exécuter les embeddings dans le même protocole, puis comparer TF-IDF, SVD, embeddings et fusions sur ce jeu avant toute modification de la recherche en production.
- SVD est ajustée une fois sur la matrice documentaire ; les requêtes sont projetées avec la transformation apprise, sans recalculer l'espace latent par requête.
