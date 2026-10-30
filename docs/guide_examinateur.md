# Guide de vérification pour l'examinateur

Ce guide permet de vérifier l'application en une dizaine de minutes, sans
lire le code. Chaque étape indique **quoi faire** et **le résultat attendu**.

- Application en ligne : https://huggingface.co/spaces/metan40/insurance_claims_assistance
- Code source : https://github.com/metan40/insurance-claims-app
- Accès direct à l'API du Space : `https://metan40-insurance-claims-assistance.hf.space`
  (ci-dessous noté `<URL>`)

> Le Space gratuit se met en veille après une période d'inactivité. Au premier
> accès, le conteneur redémarre et télécharge le modèle de langage : comptez
> quelques minutes avant que `/ready` réponde `ready`.

## 1. Contrôles techniques (2 minutes)

| Étape | Action | Résultat attendu |
|---|---|---|
| 1 | Ouvrir `<URL>/health` | `{"status": "ok", "app_version": "19.0.0"}` |
| 2 | Ouvrir `<URL>/version` | version `19.0.0` et modèle `llama3.2:1b` |
| 3 | Ouvrir `<URL>/ready` | `"status": "ready"` et toutes les vérifications à `true`. Une réponse 503 avec `ollama_model_available: false` signifie que le modèle est encore en cours de téléchargement : patienter puis recharger. |
| 4 | Ouvrir `<URL>/docs` | documentation interactive listant `/search`, `/health`, `/ready`, `/version`, `/claims` et `/api/claims/...` |

## 2. Assistant documentaire RAG (5 minutes)

Ouvrir `<URL>/`, saisir chaque question, puis cliquer sur **Rechercher**.

| Question à saisir | Source attendue parmi les premières | Ce qu'il faut observer |
|---|---|---|
| Comment fonctionne la franchise en assurance habitation ? | S1 « Comment fonctionne la franchise d’une assurance habitation ? » | réponse en français citant `[S1]` |
| Qui doit payer le serrurier, le propriétaire ou le locataire ? | S1 « Propriétaire ou locataire : qui doit payer le serrurier ? » | répartition propriétaire / locataire |
| Les panneaux solaires sont-ils couverts ? | S1 « Les panneaux solaires sont-ils couverts ? » | réponse appuyée sur la source |
| Quelle est la limite de garantie des surconsommations d'eau ? | S1 « Quelle est la limite de garantie des surconsommations d’eau » | montant ou plafond cité |
| Mes capteurs photovoltaïques sont-ils protégés ? | S1 « Les panneaux solaires sont-ils couverts ? » | « Termes métier ajoutés : panneaux solaires » : le synonyme a permis de retrouver la fiche |
| Mon aquarium s'est cassé, suis-je couvert ? | « Mon aquarium s'est cassé. Suis-je couvert ? » en S3 | exemple d'un cas où la bonne source n'est pas première |
| Quel est le cours de l'action Apple ? | aucune source pertinente | statut « Réponse absente des documents » : l'application refuse d'inventer |

Pour chaque réponse, vérifier :

1. **Statut** (pastille au-dessus de la réponse) : *réponse trouvée et
   justifiée*, *réponse partielle*, *réponse absente des documents*, ou
   *non vérifiable* (extraits affichés à la place). Ce statut est décidé par
   l'application, pas par le modèle.
2. **Extraits justificatifs** : chaque phrase citée par le modèle est marquée
   ✔ si l'application l'a retrouvée dans la source indiquée, ✘ sinon. La
   confiance affichée est celle déclarée par le modèle (non calibrée).
3. **Contrôle des références** : liste des sources citées, identifiants
   inconnus éventuels, sources disponibles mais non citées. Ce contrôle vérifie
   les identifiants `[S1]`…`[S5]`, pas la véracité du contenu.
4. **Durées observées** : vectorisation TF-IDF, classement hybride, génération,
   total. La génération représente l'essentiel du temps ; la ligne de
   confiance indique aussi combien de sources le modèle a lues (1 si la
   meilleure source suffisait, 3 sinon) et le nombre d'appels.
5. **Sources** : pour chaque source, la similarité cosinus TF-IDF, le score
   lexical et le score de fusion des rangs (aucun n'est une probabilité), puis
   l'extrait du document. L'examinateur peut lire l'extrait pour juger si la
   réponse est fidèle à la source.

Si aucun extrait du modèle n'est retrouvé dans les sources après deux
tentatives, l'application affiche volontairement les extraits des sources au
lieu d'une réponse non vérifiable.

## 3. Tableau de bord des sinistres (2 minutes)

Ouvrir `<URL>/claims`.

| Vérification | Résultat attendu |
|---|---|
| Bandeau d'avertissement | « tous les dossiers affichés sont fictifs » |
| Indicateurs | 120 dossiers : 72 acceptés, 27 refusés, 21 en attente ; taux d'acceptation 72,73 % calculé sur les 99 dossiers clos (les Pending sont exclus) |
| Métriques du modèle | régression logistique évaluée sur 25 dossiers de test : exactitude 68 %, exactitude équilibrée 60,3 %, F1 77,8 %, ROC AUC 75,4 % |
| Bouton **Analyser** du dossier C00020 (catastrophe naturelle, en attente) | orientation « Compléter le dossier avant décision » ; point bloquant « garantie à vérifier » ; alerte d'ancienneté du dossier ; facteurs du score (la garantie à vérifier fait baisser le score) ; clauses couverture, qualification, indemnisation, franchise |
| Bouton **Analyser** du dossier C00001 (bris de glace) | probabilité ≈ 85 % ; orientation « Vérifier le contrat » car le corpus n'a pas de fiche générale sur le bris de glace ; alerte montant élevé |
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
python evaluation/evaluate_pipeline.py    # mesures Recall@k et MRR
```

Résultats attendus : tous les tests réussis (aucun ne nécessite Ollama ni
Internet), puis les tableaux de rappel décrits dans le rapport. Les mêmes
tests s'exécutent automatiquement sur GitHub à chaque modification (onglet
**Actions** du dépôt).

## 6. Lancer l'application localement (facultatif)

```bash
docker build -t insurance-claims .
docker run -p 7860:7860 insurance-claims
# puis ouvrir http://localhost:7860
```

Le premier démarrage télécharge le modèle `llama3.2:1b` (environ 2 Go).
