# Assistant Assurance — dossier de livraison et recette Keyzan

## Objet de cette version

Cette version prépare l'application à une recette contrôlée par Keyzan. Elle conserve le corpus, l'interface et les contrôles existants, et ajoute une fusion finale des rangs sémantiques et lexicaux, des paramètres Ollama configurables et des points de contrôle opérationnels. Ces améliorations techniques ne remplacent pas la validation métier des réponses.

## Prérequis

- Python 3.10 ou une version compatible avec les dépendances du projet.
- Les dépendances déclarées dans `requirements.txt`.
- Ollama installé et démarré sur l'hôte qui exécute le modèle.
- Le modèle configuré, par défaut `llama3.2:3b`, téléchargé dans Ollama.
- Une machine disposant de suffisamment de mémoire pour charger les modèles d'embeddings et de reranking.

## Démarrage local sous Windows PowerShell

Depuis le dossier racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2:3b
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Ouvrez ensuite `http://127.0.0.1:8000/`. La page `/docs` présente le contrat de l'API.

## Configuration Ollama

L'application lit `OLLAMA_URL`, `OLLAMA_MODEL` et `OLLAMA_TIMEOUT_SECONDS` dans l'environnement. `OLLAMA_URL` doit être l'URL complète du point d'accès `/api/generate`. Si Ollama fonctionne sur une autre machine, configurez cette URL vers un hôte joignable depuis le serveur API et protégez le réseau. Ne publiez pas directement le port Ollama sur Internet.

Exemple PowerShell :

```powershell
$env:OLLAMA_URL = "http://localhost:11434/api/generate"
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:OLLAMA_TIMEOUT_SECONDS = "120"
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

## Contrôles opérationnels

- `GET /health` confirme que le processus API répond.
- `GET /version` indique la version et le modèle configuré.
- `GET /ready` vérifie la présence des ressources et la disponibilité d'Ollama et du modèle. Un statut `503` signifie que l'application ne doit pas encore être présentée comme prête à répondre.
- `POST /search` attend un objet JSON comme `{"question":"Comment fonctionne la franchise absolue ?"}`.

Le contrôle `/ready` vérifie une disponibilité technique. Il ne prouve pas que les réponses sont exactes, que chaque citation justifie l'affirmation, ni que les temps de réponse sont acceptables.

## Sécurité et partage avec Keyzan

Pour la première recette, privilégiez un test sur la même machine ou un réseau privé contrôlé. Le serveur ne comporte pas de mécanisme d'authentification applicative ; ne l'exposez pas sur Internet et n'utilisez pas de données personnelles ou de dossiers de sinistres réels. Pour un partage à distance, utilisez un VPN ou un tunnel approuvé, limitez les personnes autorisées et faites valider le dispositif par l'administrateur réseau.

## Plan de recette

1. Démarrer l'application et confirmer que `/health` renvoie `status: ok`.
2. Confirmer que `/ready` renvoie `status: ready` lorsque le modèle Ollama est disponible.
3. Ouvrir la page et poser une question couverte par le corpus.
4. Poser la question « Comment fonctionne la franchise absolue d'une assurance habitation ? » et vérifier que le document définissant cette franchise apparaît parmi les sources.
5. Vérifier les faits de la réponse directement dans les extraits, sans se limiter aux identifiants de citation.
6. Poser une question hors corpus et confirmer que l'application exprime les limites des documents au lieu d'inventer une règle.
7. Envoyer une question vide et une question de plus de 2 000 caractères, puis vérifier le traitement contrôlé de l'entrée.
8. Arrêter Ollama, lancer une requête et confirmer que l'API retourne une erreur technique explicite plutôt qu'une réponse d'assurance inventée.
9. Répéter plusieurs questions et relever les durées d'embedding, de recherche, de reranking, de génération et le temps total.
10. Faire remplir la fiche de retour ci-dessous par Keyzan.

## Fiche de retour Keyzan

Pour chaque question, noter la question saisie, la réponse affichée, les sources proposées, la fidélité de chaque affirmation aux sources, les informations manquantes, la facilité d'utilisation et le temps d'attente. Classer le résultat comme `correct`, `partiel`, `hors sujet`, `non justifié` ou `erreur technique`. Ajouter une capture d'écran lorsqu'un problème est constaté.

## État réel des vérifications

La suite automatisée de la version de départ passe dans l'environnement de développement. Les tests unitaires utilisent des doubles pour les modèles et Ollama ; ils ne constituent donc pas un essai de bout en bout avec les modèles réels. Avant toute annonce de livraison validée, il reste nécessaire d'exécuter la recette ci-dessus dans l'environnement cible et de documenter les résultats réels.


## Extension d'aide à l'analyse des sinistres

La route `/claims` ouvre un tableau de bord complémentaire construit sur `Data/synthetic_claims.csv`. Les 120 dossiers de ce fichier sont entièrement fictifs et servent uniquement à démontrer le calcul d'indicateurs, le traitement des dossiers Pending et l'utilisation d'un modèle interprétable. Les routes `/api/claims/summary`, `/api/claims/cases`, `/api/claims/model` et `/api/claims/{claim_id}/analysis` exposent les indicateurs, les dossiers, les métriques d'évaluation et les analyses illustratives.

Le taux d'acceptation utilise uniquement les dossiers clos : les dossiers `Pending` restent hors du dénominateur. Le modèle de démonstration est une régression logistique entraînée sur les seuls dossiers acceptés ou refusés. Le tableau de bord affiche ses métriques de test et signale que les performances obtenues sur des données synthétiques ne permettent pas de conclure à sa validité sur des sinistres réels. Chaque score exige une revue humaine ; le système ne prend aucune décision automatique.


## Audit technique et plan d'amélioration

La livraison comprend également `docs/audit_technique_v14_et_plan_v15.md`. Ce document décrit le pipeline de récupération hybride, les scores utilisés, les limites de la validation formelle des citations et les résultats historiques contenus dans les fichiers CSV. Il distingue explicitement ces résultats historiques d'une nouvelle mesure de bout en bout.

La suite automatisée de la V14 a été exécutée pendant l'audit : 40 tests réussis. Cette vérification ne remplace pas une recette avec Ollama réel ni une validation humaine des réponses contractuelles. Pour les prochaines améliorations, il est recommandé d'utiliser un jeu de questions annotées et de suivre séparément les métriques de récupération, de fidélité factuelle et de latence.

La V15 enrichit la documentation et l'analyse sans modifier le pipeline applicatif. La version interne de l'application demeure 7.5.0 ; V15 désigne ici la nouvelle version du paquet documentaire et d'audit.

## Correctif V16 — fusion lexicale conditionnelle

La V16 corrige la fusion hybride afin qu'un document sans correspondance lexicale ne reçoive pas de contribution issue de son rang lexical arbitraire. Le classement sémantique et le CrossEncoder restent en place. Le détail mathématique, les tests et les limites de validation figurent dans `docs/rapport_evolution_versions.md` et `docs/CHANGELOG_v16.md`.

La suite automatisée compte 42 tests réussis dans l'environnement de développement et la compilation Python réussit. Ces résultats valident les comportements couverts par les tests, mais ils ne remplacent pas une comparaison du classement sur les 201 questions de référence ni une recette complète avec les modèles réels. Avant de déclarer la pertinence améliorée, il faut comparer les métriques de récupération avant et après et examiner les régressions.

Le changelog de cette évolution est disponible dans `docs/CHANGELOG_v16.md`.

Le rapport technique dédié à la correction de fusion hybride est disponible dans `docs/audit_technique_v16_fusion_hybride.md`.

## Mesure lexicale complémentaire de la V16

Une évaluation TF-IDF indépendante a été exécutée sur les 201 titres et contenus du corpus. Elle obtient un Recall@5 de 62,19 % et un MRR de 0,4352, selon le protocole détaillé dans `docs/evaluation_retrieval_mesure_v16.md`. Les résultats par question sont archivés dans `Data/tfidf_reference_evaluation_v16.csv`. Cette mesure sert de référence lexicale uniquement : elle ne constitue pas une mesure du pipeline hybride de production et ne prouve pas l'effet du correctif V16. La comparaison contrôlée du pipeline avec les modèles réels reste à effectuer.

## Expérience complémentaire V18 : TF-IDF + SVD/LSA

Le dépôt inclut désormais un benchmark hors ligne de TF-IDF, SVD/LSA et fusion de classements. Pour le lancer depuis la racine : `python src/compare_tfidf_svd_embeddings.py`. Pour ajouter la comparaison au modèle d'embeddings : `python src/compare_tfidf_svd_embeddings.py --with-embeddings` (poids du modèle nécessaires). Cette expérience ne change pas le pipeline de production. Les résultats préliminaires utilisent les titres du corpus comme requêtes et la ligne correspondante comme cible : ils sont utiles pour un premier contrôle reproductible, mais doivent être confirmés avec un jeu de questions métier annotées.
