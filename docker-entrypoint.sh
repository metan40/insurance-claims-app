#!/bin/sh
set -eu

# Dans ce conteneur, l'API FastAPI et Ollama partagent le même localhost.
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/api/generate}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
export OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-180}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

# Dossier des poids du modèle : /data si le stockage persistant est disponible
# (les poids survivent aux redémarrages), sinon le dossier personnel.
export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/ollama/models}"
if ! mkdir -p "$OLLAMA_MODELS" 2>/dev/null; then
    export OLLAMA_MODELS="${HOME:-/tmp}/.ollama/models"
    mkdir -p "$OLLAMA_MODELS"
fi
echo "Dossier des modèles Ollama : ${OLLAMA_MODELS}"

printf '%s\n' "Démarrage d'Ollama sur ${OLLAMA_HOST}..."
ollama serve &
OLLAMA_PID=$!

# Attendre que l'API Ollama réponde avant de lancer la préparation du modèle.
ready=0
for attempt in $(seq 1 60); do
    if curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "ERREUR : le processus Ollama s'est arrêté au démarrage." >&2
        exit 1
    fi
    sleep 2
done

if [ "$ready" -ne 1 ]; then
    echo "ERREUR : Ollama ne répond pas après 120 secondes." >&2
    exit 1
fi

# Le téléchargement est lancé en arrière-plan pour que le serveur HTTP démarre
# et expose /health et /ready pendant la préparation initiale du modèle.
(
    echo "Préparation du modèle Ollama ${OLLAMA_MODEL}..."
    ollama pull "$OLLAMA_MODEL"
    # Préchargement : la première question de l'utilisateur ne paie pas le
    # temps de chargement du modèle en mémoire.
    curl --silent --fail http://127.0.0.1:11434/api/generate \
        -d "{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"\", \"keep_alive\": \"${OLLAMA_KEEP_ALIVE:-30m}\"}" \
        >/dev/null || echo "Préchargement du modèle ignoré."
    echo "Modèle Ollama prêt : ${OLLAMA_MODEL}"
) &

exec uvicorn src.app:app --host 0.0.0.0 --port "${PORT:-7860}"
