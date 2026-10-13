# insurance_claims_app_v4

Cette version reprend le projet de la version 3 et ajoute une première traçabilité explicite des sources ainsi qu'une validation plus stricte de la réponse d'Ollama.

## Changements principaux

- La réponse générée est invitée à citer les documents par identifiants `[S1]`, `[S2]`, etc.
- Les identifiants correspondants sont affichés avec les titres des sources dans l'interface.
- L'API expose séparément le score du reranker et la similarité des embeddings. Le champ `score` reste disponible pour préserver la compatibilité.
- Une réponse Ollama vide ou mal formée entraîne une erreur HTTP 502 contrôlée.
- Les tests unitaires vérifient la validation des questions et plusieurs cas de réponse ou d'erreur Ollama.

## Lancer les tests unitaires

Depuis ce dossier, dans PowerShell :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Les tests automatisés utilisent des doubles pour les modèles et pour Ollama. Ils ne prouvent donc pas encore que toute l'application fonctionne avec les modèles réels.

## Démarrer l'application

Depuis ce dossier, une fois les dépendances installées et Ollama prêt :

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Puis ouvrir `http://127.0.0.1:8000/` dans le navigateur.

Consulter `docs/CHANGELOG_v4.md` pour la justification des changements et les vérifications restantes.
