FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl bash \
    && rm -rf /var/lib/apt/lists/*

# Installer le binaire Ollama (le modèle sera téléchargé au démarrage).
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY --chown=user . /app
COPY --chown=user start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/data/ollama/models \
    OLLAMA_URL=http://127.0.0.1:11434/api/generate \
    OLLAMA_MODEL=llama3.2:3b \
    OLLAMA_TIMEOUT_SECONDS=120

USER user
EXPOSE 7860

CMD ["/app/start.sh"]
