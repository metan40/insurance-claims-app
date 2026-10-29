# insurance_claims_app_v7

La V7 prolonge la V6 en ajoutant des mesures de durée pour les principales étapes du pipeline RAG. L'objectif est de rendre les futures vérifications plus observables et de pouvoir relever des durées réelles pendant la campagne de tests, sans modifier les modèles ni la stratégie de recherche.

## Changements principaux

- `src/app.py` mesure la durée de l'encodage de la question, de la recherche initiale, du reranking, de la génération et de la requête totale.
- La réponse JSON de `/search` expose ces valeurs dans `pipeline_metrics`, ainsi que le nombre de documents du corpus, de candidats et de résultats renvoyés.
- `static/index.html` affiche les durées mesurées pour la requête en cours.
- Les tests vérifient la présence et le type des métriques, ainsi que leur affichage dans l'interface.
- Une réponse sans citation reconnue déclenche une seule tentative de révision ; si les références restent invalides, un message prudent remplace le texte non traçable.
- Le rapport d'évolution explique les hypothèses, les changements, les résultats des tests unitaires et les limites actuelles.

## Vérifications automatisées

Depuis la racine du projet :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

La suite contient 24 tests automatisés ; ils ont tous réussi lors de la vérification de cette archive (`24 passed`). Les tests remplacent les modèles ML et Ollama par des doubles. Ils vérifient le contrat de l'API dans des scénarios contrôlés. Ils ne fournissent pas de mesure de performance représentative des vrais modèles et ne démontrent pas la qualité des réponses générées.

## Lancement local

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Puis ouvrir `http://127.0.0.1:8000/`.

## Documentation

- `docs/CHANGELOG_v7.md` : changements, correctifs V7.1/V7.2 et limites.
- `docs/rapport_evolution_versions.md` : démarche, hypothèses, résultats et évolution des versions.
- `reports/rapport_evolution_versions_v7.html` : version HTML du rapport d'évolution.
- `docs/plan_tests_rag_et_partage_keyzan.md` : campagne de validation à effectuer avant le partage.


### Vérifier la version réellement démarrée

Après le démarrage de l’API, ouvrir `http://127.0.0.1:8001/version` (ou le port utilisé). La réponse attendue pour cette archive est `app_version: 7.2.0`. Si une autre version s’affiche ou si la route n’existe pas, le serveur pointe probablement vers un autre dossier ou une ancienne instance.

### Correctif V7.2 — rendre la seconde tentative autonome

La seconde requête Ollama contient désormais de nouveau la question et le texte des documents récupérés, car `/api/generate` ne conserve pas automatiquement le contexte du premier appel.


## Diagnostic ciblé de génération (V7.2.1)

Par défaut, les sorties brutes d'Ollama ne sont pas écrites dans la console. Pour un diagnostic local temporaire sous Windows PowerShell, exécuter depuis le dossier du projet :

```powershell
$env:RAG_DIAGNOSTICS = "1"
python -m uvicorn src.app:app --host 127.0.0.1 --port 8001
```

Le mode affiche la sortie brute et l'audit de chaque tentative ainsi que sa durée. Vérifier les journaux avant de les partager. Pour revenir au fonctionnement par défaut, arrêter le serveur et supprimer la variable avec `Remove-Item Env:RAG_DIAGNOSTICS`, puis relancer le serveur. Ce mode aide à diagnostiquer le défaut de citations ; il ne le corrige pas automatiquement.
