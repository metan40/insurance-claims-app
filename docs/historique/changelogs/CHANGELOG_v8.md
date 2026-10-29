# Journal des modifications — V7.3.0

## Objectif

Cette version améliore la sélection initiale des documents pour que les termes précis des questions d'assurance ne disparaissent pas avant le reclassement. Elle conserve la recherche par embeddings et le CrossEncoder, puis ajoute une sélection hybride qui tient compte des mots métier présents dans la question, le titre et le contenu.

## Modifications apportées

- `src/app.py` ajoute une normalisation lexicale qui ignore les accents et la ponctuation, puis calcule un score indicatif à partir des termes significatifs de la question.
- `src/app.py` fusionne les rangs sémantiques et lexicaux au moyen d'une Reciprocal Rank Fusion. Cette méthode combine les ordres de classement sans additionner directement des scores incompatibles.
- `src/app.py` réserve une partie du budget de candidats aux meilleurs résultats lexicaux, afin qu'un document métier exact reste disponible pour le CrossEncoder même lorsque les similarités sémantiques sont très proches.
- La réponse JSON expose `lexical_relevance_score` pour faciliter l'analyse des résultats.
- `static/index.html` affiche ce score avec une mention qui rappelle qu'il ne s'agit ni d'une probabilité ni d'une garantie de pertinence.
- `tests/test_app_validation.py` ajoute des tests de régression pour la sélection hybride et la comparaison des scores lexicaux.
- `docs/rapport_evolution_versions.md` documente le constat, la décision technique, la valeur métier, les limites et la prochaine validation réelle.
- `reports/rapport_evolution_versions_v8.html` fournit un rapport HTML autonome destiné à la consultation.

## Vérifications effectuées

La suite automatisée compte désormais 28 tests et les 28 tests réussissent dans l'environnement de vérification. La compilation syntaxique des fichiers Python réussit également. Les fichiers principaux modifiés ont été comparés à l'archive V7.2.1 ; leur nombre de lignes augmente et aucun de ces fichiers ne perd de lignes.

## Limites connues

Les tests automatisés utilisent des doubles pour les modèles et ne démontrent pas à eux seuls la pertinence des réponses avec les vrais modèles. Un score lexical élevé signifie que des termes se recoupent ; il ne prouve pas que le document couvre toutes les conditions de la question. Le CrossEncoder, la lecture des passages et la vérification des affirmations restent nécessaires.

## Vérification locale recommandée après livraison

Lancer l'application et demander : « Comment fonctionne la franchise absolue d'une assurance habitation ? ». Vérifier que le document de définition figure parmi les cinq sources, puis comparer la réponse aux exemples du corpus. La franchise absolue doit être distinguée de la franchise relative, et les exemples numériques doivent rester conformes au document. Relever ensuite les durées affichées pour comparer le temps de recherche et le temps de génération avec la version précédente.
