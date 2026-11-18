# Guide de vérification pour l'examinateur

Ce guide permet de vérifier l'application en une dizaine de minutes, sans
lire le code. Chaque étape indique **quoi faire** et **le résultat attendu**.

- Application en ligne : https://huggingface.co/spaces/metan40/insurance_claims_assistance
- Code source : https://github.com/metan40/insurance-claims-app
- Accès direct à l'API du Space : `https://metan40-insurance-claims-assistance.hf.space`
  (ci-dessous noté `<URL>`)

> Le Space reste allumé en permanence. Après une mise à jour, le conteneur
> redémarre et recharge les modèles : comptez quelques minutes avant que
> `/ready` réponde `ready`. Une réponse prend de l'ordre d'une minute
> (serveur sans GPU).

## 1. Contrôles techniques (2 minutes)

| Étape | Action | Résultat attendu |
|---|---|---|
| 1 | Ouvrir `<URL>/health` | `{"status": "ok", "app_version": "19.5.0"}` |
| 2 | Ouvrir `<URL>/version` | version `19.5.0`, modèle `llama3.2:3b` |
| 3 | Ouvrir `<URL>/ready` | `"status": "ready"`, `retrieval_engine: embeddings` et toutes les vérifications à `true`. Une réponse 503 signifie qu'un modèle est encore en cours de chargement : patienter puis recharger. |
| 4 | Ouvrir `<URL>/docs` | documentation interactive listant `/search`, `/health`, `/ready`, `/version`, `/claims` et `/api/claims/...` |

## 2. Assistant documentaire RAG (5 minutes)

Ouvrir `<URL>/`, saisir chaque question, puis cliquer sur **Rechercher**.
Les taux indiqués ont été mesurés sur l'application en marche
(`evaluation/verifier_application.py`, 5 passages par question ; détail dans
`docs/decisions_mesurees.md`).

| Question à saisir | Comportement attendu | Mesuré |
|---|---|---|
| Que signifie louer un logement équipé de mobilier ? | réponse complète, document « Qu'est-ce qu'une location meublée ? » cité | 5/5 |
| La moisissure sur mes murs est-elle prise en charge ? | réponse partielle : la restriction de la source est affichée sous la réponse | 5/5 |
| Mon emplacement de stationnement doit-il être assuré ? | réponse partielle appuyée sur « L'assurance d'une place de stationnement » | 4/5 |
| Quel est le délai de carence de mon assurance décès ? | **limite connue** : la réponse n'est pas dans les documents, mais le modèle conclut souvent à partir d'une source sans rapport | 0/5 |
| Pendant mes vacances, une bougie laissée sur la table de jardin a fondu et abîmé la table : suis-je couvert ? | au mieux, statut « non vérifiable » : seuls les extraits vérifiés sont affichés | 2/5 |

Pour chaque réponse, vérifier :

1. **Statut** (pastille au-dessus de la réponse) : *réponse trouvée et
   justifiée*, *réponse partielle*, *réponse absente des documents*, ou
   *non vérifiable* (extraits affichés à la place). Ce statut est décidé par
   l'application, pas par le modèle.
2. **Étiquettes [S1], [S2]…** : elles renvoient aux sources listées sous la
   réponse ; chaque extrait cité est retrouvé mot pour mot dans sa source.
3. **« Condition indiquée par la source »** : restriction recopiée de la
   source quand la réponse l'avait omise.
4. **Sources** : similarité sémantique (bge-m3) entre la question et le
   document ; ce n'est pas une probabilité d'exactitude.
5. **Durées** : recherche, génération, total.

Lecture des limites : sur 28 passages de questions dont la réponse n'est pas
dans les documents, l'application ne s'abstient que 5 fois. Deux contrôles par
inférence textuelle ont été testés ; ils bloquaient la plupart de ces
conclusions mais aussi la moitié des bonnes réponses : ils n'ont pas été
intégrés (`CHANGELOG.md`, 19.5.0).

## 3. Tableau de bord des sinistres (2 minutes)

Ouvrir `<URL>/claims`.

| Vérification | Résultat attendu |
|---|---|
| Bandeau d'avertissement | « tous les dossiers affichés sont fictifs » |
| Indicateurs | 120 dossiers : 72 acceptés, 27 refusés, 21 en attente ; taux d'acceptation 72,73 % calculé sur les 99 dossiers clos (les Pending sont exclus) |
| Métriques du modèle | régression logistique évaluée sur 25 dossiers de test : exactitude 68 %, exactitude équilibrée 60,3 %, F1 77,8 %, ROC AUC 75,4 % |
| Bouton **Analyser** du dossier C00020 (catastrophe naturelle, en attente) | orientation « Compléter le dossier avant décision » ; point bloquant « garantie à vérifier » ; alerte d'ancienneté du dossier ; facteurs du score (la garantie à vérifier fait baisser le score) ; clauses couverture, qualification, indemnisation, franchise |
| Bouton **Analyser** du dossier C00001 (bris de glace) | probabilité ≈ 85 % ; orientation « Vérifier le contrat » car le corpus n'a pas de document général sur le bris de glace ; alerte montant élevé |
| Lien « Poser cette question à l'assistant » sous une clause | ouvre l'assistant avec la question préremplie et lance la recherche |

Les mêmes informations sont disponibles en JSON :
`<URL>/api/claims/summary`, `<URL>/api/claims/model`,
`<URL>/api/claims/C00001/analysis`,
`<URL>/api/claims/C00020/decision-support`.

## 4. Vérifier avec la ligne de commande (facultatif)

```bash
curl <URL>/health
curl <URL>/ready
curl -X POST <URL>/search \
     -H "Content-Type: application/json" \
     -d '{"question": "Comment fonctionne la franchise en assurance habitation ?"}'
```

Cas d'erreur à tester :

```bash
# Question vide : refusée avec le code 422
curl -X POST <URL>/search -H "Content-Type: application/json" -d '{"question": "   "}'
# Dossier inexistant : code 404
curl <URL>/api/claims/INCONNU/analysis
```

## 5. Rejouer les tests et l'évaluation (5 minutes, Python 3.11)

```bash
git clone https://github.com/metan40/insurance-claims-app
cd insurance-claims-app
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -v                       # suite complète, sans Ollama
python evaluation/evaluate_pipeline.py    # mesures Recall@k et MRR (recherche hybride)
```

Résultats attendus : tous les tests réussis (aucun ne nécessite Ollama ni
Internet), puis les tableaux de rappel décrits dans le rapport. Les mêmes
tests s'exécutent automatiquement sur GitHub à chaque modification (onglet
**Actions** du dépôt).

## 6. Lancer l'application localement (facultatif)

```bash
docker build -t insurance-claims .
docker volume create donnees
docker run -p 7860:7860 -v donnees:/data insurance-claims
# puis ouvrir http://localhost:7860 quand http://localhost:7860/ready répond « ready »
python evaluation/verifier_application.py --url http://localhost:7860
```

Le premier démarrage télécharge `llama3.2:3b` et `bge-m3` ; le volume
`donnees` les conserve pour les démarrages suivants. Mémoire nécessaire :
environ 8 Go libres.
