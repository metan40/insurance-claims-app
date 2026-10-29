# CHANGELOG — V16 : correction ciblée de la fusion hybride

## Problème identifié

L'audit de la V15 a mis en évidence que la fusion de rangs pouvait ajouter une contribution lexicale même lorsqu'un document ne partageait aucun terme significatif avec la question. Le simple fait de posséder un rang lexical ne suffit pas à établir une correspondance pertinente.

## Modification

- Ajout de `_rank_fusion_score` dans `src/app.py`.
- Suppression de la contribution lexicale lorsque le score lexical est nul, dans la sélection des candidats et dans le classement final.
- Alignement du champ `hybrid_rank_score` de l'API sur la formule réellement appliquée.
- Ajout de deux tests unitaires pour les cas avec et sans correspondance lexicale.
- Enrichissement du rapport général avec les équations, le raisonnement, la portée du correctif et les limites de validation.

## Résultats des vérifications

- `python -m pytest -q` : 42 tests réussis.
- `python -m compileall -q src tests` : compilation réussie.
- Le code applicatif passe de 726 à 754 lignes.
- Le fichier de tests passe de 497 à 521 lignes.

## Limites

Les tests vérifient la logique de fusion et utilisent des modèles simulés. Ils ne prouvent pas une amélioration de Recall@k, du MRR, de la fidélité des citations ou de la latence sur les modèles réels. Une comparaison contrôlée sur les 201 questions de référence et une recette avec Ollama restent nécessaires.

Le rapport technique détaillé est disponible dans `docs/audit_technique_v16_fusion_hybride.md`.
