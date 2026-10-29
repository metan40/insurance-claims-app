# Version 5 — audit formel des citations et chemins robustes

## Constat

La V4 demandait au modèle de citer les sources avec des identifiants comme `[S1]`, mais l'application ne vérifiait pas si les identifiants écrits dans la réponse correspondaient réellement aux documents fournis. De plus, plusieurs chemins dépendaient du répertoire depuis lequel la commande était lancée. Ces points pouvaient compliquer la vérification des réponses et le lancement dans un autre contexte.

## Hypothèses de travail

1. Un contrôle des identifiants doit aider à repérer une citation absente ou inconnue.
2. Des chemins calculés à partir du dossier du projet doivent rendre le démarrage moins dépendant du répertoire courant.
3. L'affichage explicite des limites du contrôle doit éviter de confondre cohérence formelle des citations et justification factuelle.

Ces hypothèses motivent les changements ; elles ne constituent pas des résultats de validation de bout en bout.

## Modifications

- `src/app.py` : résolution des chemins de données et de l'interface à partir de `__file__` ; ajout de `audit_citations()` ; ajout de `citation_audit` à la réponse JSON.
- `static/index.html` : affichage du bilan formel des citations et rendu du message d'erreur avec `textContent`.
- `src/generate_retrieval_report.py` : résolution des chemins à partir du dossier du projet et enrichissement de la section de rapport consacrée à l'application.
- `tests/test_app_validation.py` : tests des identifiants valides, inconnus, absents et du périmètre limité de l'audit.
- `README_VERSION_5.md`, `VERSION.txt` et ce changelog : documentation de la version.

## Résultats mesurés sur le jeu d'évaluation existant

Le script de rapport a été exécuté avec les CSV présents dans le projet. Il analyse 201 questions :

- Recall@1 TF-IDF : 29,85 % ; embeddings : 37,81 %.
- Recall@5 TF-IDF : 62,19 % ; embeddings : 67,16 %.
- MRR TF-IDF : 0,4352 ; embeddings : 0,5051.
- La cible attendue est présente dans le Top-20 pour 167 questions sur 201 (83,08 %) et absente pour 34 (16,92 %).
- Après reranking, Hit Rate@5 : 76,12 % ; MRR : 0,6331.

Ces chiffres proviennent des fichiers d'évaluation déjà présents ; ils évaluent le retrieval et le reranking, pas la fidélité des réponses générées par Ollama.

## Vérifications de la V5

- Tests automatisés : exécutés localement avec des doubles pour les modèles et Ollama.
- Compilation syntaxique : à vérifier à chaque changement ; la validation réelle du modèle reste séparée.
- Rapport HTML : régénéré depuis les données CSV existantes.

## Limites restantes

L'audit repère uniquement les références qui correspondent au format `[S<number>]`. Il ne vérifie pas que le document cité soutient réellement la phrase, et une réponse sans citation peut encore contenir des informations exactes ou inexactes. La campagne de bout en bout devra inclure des questions couvertes par le corpus, des questions hors corpus, des citations inconnues et un test d'indisponibilité d'Ollama. Aucun résultat de ces tests réels n'est présumé à ce stade.
