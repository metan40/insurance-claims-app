---
title: Insurance Claims Assistance
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Claims Assistance

Assistant assurance fondé sur une recherche documentaire RAG, avec API FastAPI et tableau de bord de démonstration.

## Configuration requise pour la génération

Cette version du conteneur démarre Ollama automatiquement et télécharge le modèle configuré au démarrage. Le premier démarrage peut donc être long ; attendez que `/ready` indique `status: ready` avant de tester une question. Le modèle et les modèles d'embeddings/reranking peuvent consommer plusieurs gigaoctets de mémoire.

Dans **Settings → Variables and secrets**, vous pouvez configurer :

- `OLLAMA_URL` : facultatif ; par défaut `http://127.0.0.1:11434/api/generate` pour l'Ollama lancé dans le conteneur. Ne le remplacez que si vous utilisez un serveur Ollama externe.
- `OLLAMA_MODEL` : nom exact du modèle installé sur ce serveur, par défaut `llama3.2:3b`.
- `OLLAMA_TIMEOUT_SECONDS` : délai maximal en secondes, par défaut `120`.

Si vous utilisez un serveur Ollama externe, il doit être joignable depuis le Space et protégé par un mécanisme adapté. Ne publiez pas directement le port Ollama sur Internet sans protection et ne placez aucun jeton ou mot de passe dans le code.

## Vérifications après déploiement

- `/health` : vérifie que l'API répond.
- `/version` : affiche la version et le modèle configuré.
- `/ready` : vérifie les ressources de l'application et l'accès au serveur Ollama.
- `/docs` : documentation interactive de l'API.
- `/` : interface principale.
- `/claims` : tableau de bord des dossiers synthétiques de démonstration.

Un `/health` fonctionnel ne garantit pas qu'Ollama est accessible. Si `/ready` indique `ollama_reachable: false`, vérifiez `OLLAMA_URL` et la connectivité réseau depuis le Space. Si le serveur est joignable mais que `ollama_model_available` vaut `false`, vérifiez `OLLAMA_MODEL` et l'installation du modèle sur le serveur Ollama.

## Démarrage local

```bash
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 7860
```

En local, si Ollama n'est pas déjà démarré, lancez-le puis installez le modèle configuré (`ollama pull llama3.2:3b` pour le modèle par défaut).

Les dossiers de sinistres fournis pour le tableau de bord sont synthétiques et destinés à la démonstration. Ne les utilisez pas pour prendre automatiquement des décisions sur des sinistres réels.
