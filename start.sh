#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/ollama/models}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/api/generate}"

# Le stockage /data du Space est prévu pour conserver les poids entre redémarrages.
mkdir -p "$OLLAMA_MODELS"

echo "[startup] Démarrage du serveur Ollama..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for i in $(seq 1 60); do
  if ollama list >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[startup] Ollama s'est arrêté. Logs :"
    cat /tmp/ollama.log || true
    exit 1
  fi
  sleep 2
done

if [ "$ready" -ne 1 ]; then
  echo "[startup] Ollama n'est pas devenu disponible. Logs :"
  cat /tmp/ollama.log || true
  exit 1
fi

echo "[startup] Vérification/téléchargement du modèle ${MODEL}..."
if ! ollama show "$MODEL" >/dev/null 2>&1; then
  ollama pull "$MODEL"
fi

echo "[startup] Modèle prêt. Démarrage de FastAPI sur le port 7860..."
exec uvicorn src.app:app --host 0.0.0.0 --port 7860 --proxy-headers
