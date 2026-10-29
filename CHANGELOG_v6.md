# Version 6 — rendre les citations et les scores plus transparents

## Constat

La V5 calculait les sources disponibles mais non citées, mais l'interface ne les affichait pas. Le libellé des scores ne rappelait pas explicitement qu'ils ne sont pas des probabilités de réponse correcte.

## Changements

- L'interface affiche les identifiants reconnus, les identifiants inconnus et les sources disponibles mais non citées.
- Le score du CrossEncoder est présenté comme un score brut, et non comme une probabilité.
- La similarité des embeddings est explicitement décrite comme une similarité cosinus utilisée pour la présélection.
- L'API ajoute `score_explanation` pour rendre ces significations explicites dans la réponse JSON.
- Les tests vérifient les libellés des scores et les informations de citations affichées par l'interface.

## Limites

Le contrôle vérifie la cohérence formelle des identifiants, pas la justification factuelle des phrases. Les tests automatisés utilisent des doubles ; les essais avec les vrais modèles et Ollama restent à effectuer.
