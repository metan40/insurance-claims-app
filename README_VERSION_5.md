# insurance_claims_app_v5

La version 5 poursuit les améliorations de la version 4 sans remplacer les modèles d'embeddings, le reranker ni le modèle Ollama configuré.

## Changements

- Les chemins du CSV et de l'interface sont construits à partir du dossier du projet ; le lancement ne dépend donc plus du répertoire courant pour ces deux fichiers.
- L'API ajoute `citation_audit`, un contrôle formel des références `[S1]`, `[S2]`, etc. Il distingue les identifiants reconnus, inconnus et les sources disponibles qui ne sont pas citées.
- L'interface affiche ce bilan et précise qu'il ne valide pas la véracité des affirmations.
- L'affichage des erreurs dans l'interface utilise `textContent` plutôt que d'injecter le message d'erreur dans du HTML.
- Le script de génération du rapport utilise lui aussi des chemins fondés sur le dossier du projet et son contenu a été enrichi pour expliquer les choix de la V5 et leurs limites.

## Tests automatisés

Depuis la racine du projet :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Les tests remplacent les modèles et l'appel à Ollama par des doubles. Ils vérifient la logique locale, pas la qualité réelle des réponses générées. La validation complète doit encore être faite avec les vrais modèles et un serveur Ollama actif.

## Lancement local

Depuis la racine du projet, après installation des dépendances et disponibilité d'Ollama :

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Puis ouvrir `http://127.0.0.1:8000/`.

## Documentation

- `docs/CHANGELOG_v5.md` : changements, motivation et limites.
- `docs/rapport_evolution_versions.md` : narration de la démarche et résultats de retrieval connus.
- `docs/plan_tests_rag_et_partage_keyzan.md` : scénarios à exécuter avant le partage.
- `Data/retrieval_report_2.html` et `reports/retrieval_report_2.html` : rapport d'évaluation du retrieval enrichi.
