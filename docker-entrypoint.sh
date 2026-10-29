#!/bin/sh
set -eu

# Dans ce conteneur, l'API FastAPI et Ollama partagent le même localhost.
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/api/generate}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-180}"

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
    echo "Modèle Ollama prêt : ${OLLAMA_MODEL}"
) &

exec uvicorn src.app:app --host 0.0.0.0 --port "${PORT:-7860}"
