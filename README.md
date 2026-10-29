---
title: Insurance Claims Assistance
emoji: 🛡️
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

1. **Assistant documentaire RAG** (`/`) : une question en langage naturel est
   rapprochée d'un corpus de 201 fiches d'assurance par une recherche hybride
   (TF-IDF + score lexical, fusion des rangs RRF). Les cinq meilleures fiches
   sont transmises à un LLM local (`llama3.2:3b` via Ollama) qui doit répondre
   uniquement à partir de ces sources et les citer (`[S1]`…`[S5]`). Les
   citations sont contrôlées automatiquement.
2. **Tableau de bord sinistres** (`/claims`) : 120 dossiers **fictifs**,
   indicateurs descriptifs (les dossiers *Pending* sont exclus du taux
   d'acceptation) et régression logistique donnant une probabilité indicative
   d'acceptation, évaluée sur un jeu de test tenu à part.

- Application en ligne : https://huggingface.co/spaces/metan40/insurance_claims_assistance
- **Examinateur : suivre [`docs/guide_examinateur.md`](docs/guide_examinateur.md)** (10 minutes).

## Vérifications rapides

| Route | Rôle |
|---|---|
| `GET /health` | l'API répond |
| `GET /version` | version et modèle configuré |
| `GET /ready` | fichiers, index de recherche, Ollama et modèle disponibles (503 sinon) |
| `GET /docs` | documentation interactive |
| `POST /search` | question → réponse citée, sources, contrôle des citations, durées |
| `GET /claims` | tableau de bord des sinistres fictifs |
| `GET /api/claims/summary`, `/api/claims/model`, `/api/claims/{id}/analysis` | données du tableau de bord |

`/health` ne garantit pas qu'Ollama est prêt : au premier démarrage, attendre
que `/ready` indique `status: ready` (téléchargement du modèle).

## Organisation du dépôt

```
src/                 code de l'application
  app.py             API FastAPI : routes, génération Ollama, contrôle des citations
  retrieval.py       recherche hybride TF-IDF + lexicale (RRF)
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
ollama pull llama3.2:3b
uvicorn src.app:app --host 0.0.0.0 --port 7860
```

## Configuration

Variables d'environnement (Space : **Settings → Variables and secrets** ;
exemple dans `.env.example`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | serveur Ollama (celui du conteneur par défaut) |
| `OLLAMA_MODEL` | `llama3.2:3b` | modèle utilisé pour la génération |
| `OLLAMA_TIMEOUT_SECONDS` | `120` (180 dans le conteneur) | délai maximal d'un appel |
| `OLLAMA_NUM_PREDICT` | `400` | longueur maximale d'une réponse (en jetons) |
| `OLLAMA_KEEP_ALIVE` | `30m` | durée pendant laquelle le modèle reste chargé en mémoire |
| `RAG_DIAGNOSTICS` | `0` | `1` pour journaliser les sorties brutes du modèle (diagnostic local uniquement) |

Ne publiez pas le port d'un serveur Ollama externe sur Internet sans
protection et ne placez aucun jeton dans le code.

## Limites

- Les dossiers de sinistres sont **fictifs** : les performances du modèle ne
  décrivent aucun portefeuille réel et aucune décision ne doit en découler.
- Le contrôle des citations vérifie les identifiants de source, pas la
  fidélité du contenu : la lecture des extraits reste nécessaire.
- La recherche TF-IDF est sensible au vocabulaire : sur 40 questions
  reformulées, la bonne fiche figure dans les 5 premières dans 40 % des cas
  (voir le rapport et `evaluation/`).
