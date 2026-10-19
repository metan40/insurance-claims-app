
FROM python:3.11

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY --chown=user . /app

ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:$PATH"

USER user

EXPOSE 7860

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
