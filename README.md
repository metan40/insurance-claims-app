---
title: Insurance Claims Assistance
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Claims Assistance — version 19.5.0

Auteur : Thierry MBARGA

Application d'**aide à l'analyse des sinistres** en assurance habitation. Elle
n'automatise aucune décision : elle rassemble pour le gestionnaire les
informations utiles et les sources qui permettent de les vérifier.

Deux briques complémentaires :

1. **Assistant documentaire RAG** (`/`) : la question est d'abord analysée
   (termes significatifs et **synonymes métier** : « nounou » → « baby-sitter »),
   puis rapprochée d'un corpus de 201 documents d'assurance par une recherche
   sémantique (embeddings multilingues bge-m3 ; recherche hybride TF-IDF +
   lexicale en repli). Les meilleurs documents sont transmis à un LLM local
   (`llama3.2:3b` via Ollama) qui renvoie
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
docker volume create donnees
docker run -p 7860:7860 -v donnees:/data insurance-claims
```

Le volume `donnees` garde les modèles téléchargés (llama3.2:3b, bge-m3) et
l'index : seul le premier démarrage les télécharge. L'application est prête
quand <http://localhost:7860/ready> répond `ready`. Pour vérifier qu'elle
fonctionne de bout en bout, sans le serveur distant :

```bash
python evaluation/verifier_application.py --url http://localhost:7860
```

Le workflow GitHub « Vérification locale » refait exactement ces étapes sur
une machine vierge.

### Secours si le serveur distant est indisponible

1. Démarrer l'application en local (commandes ci-dessus) et attendre que
   <http://localhost:7860/ready> réponde `ready`.
2. Donner une adresse publique temporaire à l'application locale, sans
   ouvrir de port sur la box, avec Cloudflare Tunnel (gratuit, sans compte) :

   ```bash
   cloudflared tunnel --url http://localhost:7860
   ```

   La commande affiche une adresse `https://….trycloudflare.com`, ouvrable
   depuis n'importe quel ordinateur tant que la machine locale reste allumée.
3. Vérifier par cette adresse publique :

   ```bash
   python evaluation/verifier_application.py --url https://….trycloudflare.com --repetitions 1
   ```

Mémoire nécessaire : environ 8 Go libres (llama3.2:3b et bge-m3).

Sans Docker (Ollama installé et lancé séparément), avec les réglages du Space :

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
OLLAMA_MODEL=llama3.2:3b RETRIEVAL_ENGINE=embeddings uvicorn src.app:app --host 0.0.0.0 --port 7860
```

Sous Windows (PowerShell, Ollama pour Windows lancé) : un script fait les
mêmes étapes, environnement Python compris :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\lancer_local_windows.ps1
```

Sans `RETRIEVAL_ENGINE=embeddings`, l'application utilise la recherche
hybride TF-IDF de la version 19.4.0. Au premier démarrage, le modèle
d'embeddings bge-m3 (environ 2 Go) est téléchargé depuis Hugging Face.

## Configuration

Variables d'environnement (Space : **Settings → Variables and secrets** ;
exemple dans `.env.example`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | serveur Ollama (celui du conteneur par défaut) |
| `OLLAMA_MODEL` | `llama3.2:3b` dans le conteneur | modèle de génération ; choisi sur mesure (évaluation de bout en bout avec bge-m3) |
| `OLLAMA_TIMEOUT_SECONDS` | `120` (180 dans le conteneur) | délai maximal d'un appel |
| `OLLAMA_NUM_CTX` | `2048` | taille de la fenêtre de contexte (plus petite = plus rapide) |
| `OLLAMA_NUM_PREDICT` | `400` | longueur maximale d'une réponse (en jetons) |
| `OLLAMA_KEEP_ALIVE` | `30m` | durée pendant laquelle le modèle reste chargé en mémoire |
| `RAG_DIAGNOSTICS` | `0` | `1` pour journaliser les sorties brutes du modèle (diagnostic local uniquement) |

Ne publiez pas le port d'un serveur Ollama externe sur Internet sans
protection et ne placez aucun jeton dans le code.

## Limites

- Le Space n'a pas de GPU : une réponse prend de l'ordre de la minute ; la
  durée de chaque étape est renvoyée par `POST /search`.
- Les dossiers de sinistres sont **fictifs** : les performances du modèle ne
  décrivent aucun portefeuille réel et aucune décision ne doit en découler.
- La vérification confirme qu'un extrait existe dans la source, pas qu'il
  justifie entièrement la réponse : la lecture reste nécessaire.
- **Questions dont la réponse n'est pas dans les documents** : sur 28
  passages de test, l'application ne s'abstient que 5 fois (18 %) ; sinon
  elle cite un vrai extrait sur un autre sujet et conclut. Un contrôle par
  inférence textuelle bloquait 86 % de ces conclusions mais aussi 7 bonnes
  réponses sur 15 : il n'a pas été intégré (détail et chiffres dans le
  CHANGELOG, 19.5.0).
- Recherche : sur 30 questions reformulées tenues à part, le bon document
  figure parmi les 3 premiers dans 26 cas (87 %, IC de Wilson 95 %
  [70 % ; 95 %]) avec bge-m3, contre 13 (43 %) avec la recherche hybride.
  La qualité de la réponse reste limitée par le modèle de langage : 46 % de
  réponses conformes à la référence avec `llama3.2:3b`. Voir `evaluation/`.
