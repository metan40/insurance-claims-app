# CHANGELOG — V15 : audit technique et enrichissement du rapport

## Ajouts

- Ajout de `docs/audit_technique_v14_et_plan_v15.md`, un rapport rédigé en phrases complètes qui décrit l'architecture RAG existante, les principes mathématiques du classement, les résultats historiques, les limites et le protocole de validation recommandé.
- Enrichissement de `docs/rapport_evolution_versions.md` avec une section V15 qui consigne le périmètre, les observations et les limites de l'audit.
- Enrichissement du guide de livraison avec un lien vers l'audit et une distinction entre version du paquet et version interne de l'application.
- Vérification de l'intégrité de l'archive source et exécution de la suite automatisée : 40 tests réussis.

## Comportement applicatif

Aucun comportement applicatif n'a été modifié dans cette étape. L'application conserve sa version interne 7.5.0. La V15 désigne le paquet enrichi de documentation et d'audit.

## Limites et suite recommandée

Les résultats Top 5 et Top 20 proviennent de fichiers d'évaluation historiques présents dans l'archive. Ils ne sont pas présentés comme une nouvelle évaluation de bout en bout. La prochaine évolution du moteur doit mesurer les effets sur un jeu de questions annotées, avec les mêmes données et les mêmes paramètres, puis examiner les régressions.
