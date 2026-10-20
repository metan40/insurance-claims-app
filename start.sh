
#!/usr/bin/env bash
set -e

export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="/data/ollama-models"

mkdir -p "$OLLAMA_MODELS"

echo "Démarrage d'Ollama..."
ollama serve &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Attente du serveur Ollama..."
OLLAMA_READY=0

for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
    OLLAMA_READY=1
    break
  fi
  sleep 2
done

if [ "$OLLAMA_READY" -ne 1 ]; then
  echo "ERREUR : Ollama n'a pas démarré."
  exit 1
fi

echo "Téléchargement ou vérification du modèle..."
ollama pull llama3.2:3b

echo "Démarrage de FastAPI..."
exec uvicorn src.app:app --host 0.0.0.0 --port 7860
