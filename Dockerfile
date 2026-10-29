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

# Hugging Face exécute les Spaces Docker avec l'utilisateur 1000 :
# on crée cet utilisateur et on lui donne les fichiers de l'application.
RUN useradd -m -u 1000 user

COPY --chown=user VERSION docker-entrypoint.sh ./
COPY --chown=user src ./src
COPY --chown=user static ./static
COPY --chown=user Data ./Data
RUN chmod +x /app/docker-entrypoint.sh

# Les poids du modèle sont conservés dans /data (stockage persistant du
# Space) lorsqu'il existe ; sinon le script se replie sur le dossier personnel.
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    OLLAMA_MODELS=/data/ollama/models

USER user
EXPOSE 7860
CMD ["/app/docker-entrypoint.sh"]
