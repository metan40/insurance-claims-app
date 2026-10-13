# V17 — ajout d'une évaluation lexicale mesurée

## Ajouts

- Ajout de `docs/evaluation_retrieval_mesure_v16.md`, qui décrit le protocole de référence TF-IDF, les métriques obtenues et les limites d'interprétation.
- Ajout de `Data/tfidf_reference_evaluation_v16.csv`, qui conserve les résultats détaillés pour les 201 questions.
- Enrichissement du rapport d'évolution et du guide de livraison afin de séparer les mesures nouvellement exécutées des résultats historiques et des validations encore à réaliser.

## Résultats de la référence TF-IDF

- Recall@1 : 29,85 % (60/201).
- Recall@3 : 52,24 % (105/201).
- Recall@5 : 62,19 % (125/201).
- Recall@20 : 77,11 % (155/201).
- MRR : 0,4352.

## Limites explicites

Cette mesure est une référence lexicale indépendante. Elle ne rejoue pas le pipeline hybride de production et ne mesure pas l'effet du correctif V16. Le paquet `sentence_transformers` n'était pas disponible dans l'environnement de mesure, ce qui empêche la comparaison contrôlée du pipeline avec les poids réels. Aucune amélioration de la pertinence de production n'est revendiquée sur la base de cette seule expérience.

## Vérifications

La suite automatisée passe avec 42 tests. La compilation des modules Python réussit. L'archive de livraison est vérifiée après création.
