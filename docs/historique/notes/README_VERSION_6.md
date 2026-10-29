# insurance_claims_app_v6

La V6 poursuit le travail de la V5 en améliorant la lisibilité du contrôle des citations et l'interprétation des scores, sans changer les modèles de recherche ni la stratégie de classement.

## Changements principaux

- L'interface montre les sources citées et reconnues, les identifiants inconnus et les sources disponibles mais non citées.
- Le score du CrossEncoder est indiqué comme score brut, et non comme probabilité.
- La similarité cosinus des embeddings est identifiée comme mesure de présélection, pas comme mesure de vérité.
- L'API renvoie des explications sur les deux scores dans `score_explanation`.
- Des tests supplémentaires couvrent les libellés de l'interface et les explications de l'API.

## Tests

Depuis la racine du projet :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Les tests automatisés remplacent les modèles ML et Ollama par des doubles. Ils vérifient des comportements contrôlés, mais ne démontrent pas la qualité réelle des réponses. Les tests de bout en bout avec les vrais modèles et un serveur Ollama actif restent à effectuer.

## Lancement local

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Puis ouvrir `http://127.0.0.1:8000/`.

## Documentation

- `docs/CHANGELOG_v6.md` : changements et limites.
- `docs/rapport_evolution_versions.md` : démarche, hypothèses et résultats.
- `docs/plan_tests_rag_et_partage_keyzan.md` : campagne à effectuer avant le partage.
