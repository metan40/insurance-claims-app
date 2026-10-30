---
title: Insurance Claims Assistance
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Claims Assistance — version 19.0.0

Application d'**aide à l'analyse des sinistres** en assurance habitation. Elle
n'automatise aucune décision : elle rassemble pour le gestionnaire les
informations utiles et les sources qui permettent de les vérifier.

Deux briques complémentaires :

1. **Assistant documentaire RAG** (`/`) : la question est d'abord analysée
   (termes significatifs et **synonymes métier** : « nounou » → « baby-sitter »),
   puis rapprochée d'un corpus de 201 fiches d'assurance par une recherche
   hybride (TF-IDF + score lexical, fusion des rangs RRF). Les cinq meilleures
   fiches sont transmises à un LLM local (`llama3.2:1b` via Ollama) qui renvoie
   une **réponse typée** : réponse trouvée ou non, complète ou non, extraits
   justificatifs, confiance, réserves. L'application **vérifie elle-même** que
   chaque extrait figure dans la source citée et décide du statut affiché.
2. **Aide à la décision sur un dossier** (`/claims`) : 120 dossiers
   **fictifs**. Pour un dossier, l'application réunit en une vue le **score
   expliqué** (régression logistique et facteurs qui le font monter ou
   baisser), les **points à vérifier** (règles métier explicites), les
   **clauses du contrat** correspondant au type de sinistre et une
   **orientation** pour le gestionnaire (« compléter le dossier », « examiner
   les exclusions »…), jamais une décision automatique. Chaque clause renvoie
   à l'assistant RAG.

- Application en ligne : https://huggingface.co/spaces/metan40/insurance_claims_assistance
- **Examinateur : suivre [`docs/guide_examinateur.md`](docs/guide_examinateur.md)** (10 minutes).

## Vérifications rapides

| Route | Rôle |
|---|---|
| `GET /health` | l'API répond |
| `GET /version` | version et modèle configuré |
| `GET /ready` | fichiers, index de recherche, Ollama et modèle disponibles (503 sinon) |
| `GET /docs` | documentation interactive |
| `POST /search` | question → statut, réponse typée et extraits vérifiés, sources, durées |
| `GET /claims` | tableau de bord des sinistres fictifs |
| `GET /api/claims/{id}/decision-support` | aide à la décision d'un dossier |
| `GET /api/claims/summary`, `/api/claims/model`, `/api/claims/{id}/analysis` | données du tableau de bord |

`/health` ne garantit pas qu'Ollama est prêt : au premier démarrage, attendre
que `/ready` indique `status: ready` (téléchargement du modèle).

## Organisation du dépôt

```
src/                 code de l'application
  app.py             API FastAPI : routes, appel à Ollama
  question_parsing.py  analyse de la question, dictionnaire de synonymes métier
  retrieval.py       recherche hybride TF-IDF + lexicale (RRF)
  generation.py      réponse typée, vérification des extraits, boucle top-1 / top-3
  claim_decision.py  aide à la décision : règles métier, clauses, orientation
  claims_analytics.py  indicateurs et modèle des sinistres fictifs
static/              interface web (index.html, claims.html)
Data/                corpus documentaire et sinistres fictifs
tests/               tests automatisés (pytest), sans Ollama ni Internet
evaluation/          évaluation reproductible de la recherche (Recall@k, MRR)
experiments/         scripts et résultats des versions précédentes (archives)
docs/                guide examinateur, cadrage du projet, historique
Dockerfile, docker-entrypoint.sh   image du Space (Ollama + API)
VERSION              numéro de version unique
```

## Tester

```bash
pip install -r requirements-dev.txt
python -m pytest -v
python evaluation/evaluate_pipeline.py
```

Les tests s'exécutent aussi sur GitHub Actions à chaque push ; le Space
Hugging Face n'est mis à jour depuis `main` **que si les tests réussissent**.

## Démarrage local

Avec Docker (identique au Space) :

```bash
docker build -t insurance-claims .
docker run -p 7860:7860 insurance-claims
```

Sans Docker (Ollama installé et lancé séparément) :

```bash
pip install -r requirements.txt
ollama pull llama3.2:1b
uvicorn src.app:app --host 0.0.0.0 --port 7860
```

## Configuration

Variables d'environnement (Space : **Settings → Variables and secrets** ;
exemple dans `.env.example`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | serveur Ollama (celui du conteneur par défaut) |
| `OLLAMA_MODEL` | `llama3.2:1b` | modèle de génération ; `llama3.2:3b` rédige mieux mais deux fois plus lentement sur CPU |
| `OLLAMA_TIMEOUT_SECONDS` | `120` (180 dans le conteneur) | délai maximal d'un appel |
| `OLLAMA_NUM_CTX` | `2048` | taille de la fenêtre de contexte (plus petite = plus rapide) |
| `OLLAMA_NUM_PREDICT` | `280` | longueur maximale d'une réponse (en jetons) |
| `OLLAMA_KEEP_ALIVE` | `30m` | durée pendant laquelle le modèle reste chargé en mémoire |
| `RAG_DIAGNOSTICS` | `0` | `1` pour journaliser les sorties brutes du modèle (diagnostic local uniquement) |

Ne publiez pas le port d'un serveur Ollama externe sur Internet sans
protection et ne placez aucun jeton dans le code.

## Limites

- Le Space gratuit n'a pas de GPU : une réponse prend environ 2 minutes
  avec `llama3.2:1b` (environ 4 minutes avec `llama3.2:3b`).
- Les dossiers de sinistres sont **fictifs** : les performances du modèle ne
  décrivent aucun portefeuille réel et aucune décision ne doit en découler.
- La vérification confirme qu'un extrait existe dans la source, pas qu'il
  justifie entièrement la réponse : la lecture reste nécessaire.
- La recherche reste sensible au vocabulaire : sur 30 questions reformulées
  tenues à part, la bonne fiche figure dans les 5 premières dans 53 % des cas
  (40 % sans les synonymes). Voir le rapport et `evaluation/`.
