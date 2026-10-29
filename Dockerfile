# Image du Space Hugging Face « Insurance Claims Assistance ».
# Un seul conteneur exécute Ollama (LLM local) et l'API FastAPI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODEL=llama3.2:3b

WORKDIR /app

# curl sert au script de démarrage ; zstd peut être exigé par l'installateur Ollama.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libgomp1 zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# Dépendances Python installées avant le code pour profiter du cache Docker.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY VERSION docker-entrypoint.sh ./
COPY src ./src
COPY static ./static
COPY Data ./Data
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 7860
CMD ["/app/docker-entrypoint.sh"]
