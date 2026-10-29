# Expériences archivées (versions 4 à 18)

Ce dossier conserve les scripts et les résultats des expérimentations menées
avant la version 19 : comparaison embeddings / TF-IDF, analyse des échecs de
recherche, essais de SVD (LSA), rapports HTML de recherche.

- `scripts/` : scripts d'origine, déplacés depuis `src/` et la racine du dépôt
  sans modification de leur logique. Ils lisent et écrivent des fichiers sous
  `Data/` : pour les relancer, adapter les chemins vers `experiments/results/`.
  Certains nécessitent `sentence-transformers` :
  `pip install -r requirements-experiments.txt`.
- `results/` : fichiers produits par ces scripts (CSV, JSON, HTML).

Ces scripts ne sont ni exécutés par la CI ni inclus dans l'image Docker.
L'évaluation maintenue de la version en production est
`evaluation/evaluate_pipeline.py`.
