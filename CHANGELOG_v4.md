# Version 4 — traçabilité des sources et fiabilisation de la génération

## Pourquoi cette évolution ?

Dans la version précédente, l'interface affichait les documents récupérés, mais la réponse générée ne disposait pas d'un format d'identifiant de source explicite. De plus, `generate_answer` renvoyait directement le champ `response` ou un texte par défaut, ce qui pouvait masquer une réponse Ollama vide ou mal formée.

## Changements apportés

1. Chaque document présenté à la génération reçoit un identifiant stable pour la réponse courante (`S1` à `S5`). Le prompt demande au modèle de citer ces identifiants entre crochets et de ne pas en inventer.
2. L'interface affiche l'identifiant à côté du titre du document, afin de faciliter la vérification manuelle des citations.
3. L'API distingue maintenant le score du CrossEncoder (`reranker_score`) de la similarité des embeddings (`embedding_similarity`). Le champ historique `score` est conservé pour maintenir la compatibilité avec les consommateurs existants.
4. La réponse Ollama est contrôlée avant d'être renvoyée. Un JSON qui n'est pas un objet, un champ `response` absent, un contenu vide ou un contenu d'un type inattendu déclenche une erreur HTTP 502.
5. Les tests unitaires couvrent la validation des questions, l'inclusion des identifiants dans le prompt, les réponses Ollama non exploitables et une erreur de connexion.

## Hypothèses à vérifier

- Les identifiants de source doivent faciliter la vérification des affirmations de la réponse par rapport aux documents affichés.
- La séparation des scores doit rendre l'interface et l'analyse plus compréhensibles.
- La validation de la réponse Ollama doit éviter qu'une réponse vide ou mal formée soit présentée comme une réponse normale.

Ces points sont des hypothèses de conception. Les identifiants dans le prompt ne garantissent pas à eux seuls que le modèle cite correctement chaque affirmation ; les citations doivent être vérifiées pendant la campagne de tests de bout en bout.

## Vérifications automatisées

Les tests unitaires peuvent être exécutés depuis le dossier racine du projet avec :

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Ces tests utilisent des doubles pour les modèles et Ollama. Ils vérifient la logique locale, mais ne remplacent pas les tests d'intégration avec les vrais modèles chargés et un service Ollama actif.

## Éléments non modifiés

- Modèle d'embeddings : `paraphrase-multilingual-MiniLM-L12-v2`.
- Reranker : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Taille du premier ensemble de candidats : jusqu'à 20 documents.
- Nombre de documents transmis au générateur : jusqu'à 5.
- Modèle Ollama configuré : `llama3.2:3b`.

## État de validation

La validation complète de l'interface, de la recherche documentaire et de la génération avec Ollama reste une étape séparée. Les résultats de cette campagne devront être consignés après exécution dans l'environnement cible ; ils ne sont pas présumés réussis dans ce document.
