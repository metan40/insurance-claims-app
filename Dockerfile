
FROM python:3.11-slim

USER root
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copier le projet et le script de démarrage
COPY . /app
RUN chmod +x /app/start.sh

# Créer un utilisateur non privilégié
RUN useradd -m -u 1000 user \
    && mkdir -p /data/ollama-models \
    && chown -R user:user /app /home/user

ENV HOME=/home/user
ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/data/ollama-models
ENV OLLAMA_MODEL=llama3.2:3b
ENV PATH="/home/user/.local/bin:$PATH"

USER user

EXPOSE 7860

CMD ["/app/start.sh"]
