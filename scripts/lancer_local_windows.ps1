# Lance l'application en local sous Windows, sans Docker, avec les mêmes
# réglages que le Space (19.5.0) : recherche bge-m3, modèle llama3.2:3b.
# Prérequis : Python 3.10 ou plus, Ollama pour Windows installé et lancé.
# Usage, dans PowerShell, depuis le dossier du dépôt :
#     powershell -ExecutionPolicy Bypass -File scripts\lancer_local_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "1/4 Vérification d'Ollama…"
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null }
catch { Write-Host "Ollama ne répond pas : lancez l'application Ollama, puis relancez ce script."; exit 1 }

Write-Host "2/4 Modèle de langage llama3.2:3b (téléchargé une seule fois, environ 2 Go)…"
ollama pull llama3.2:3b

Write-Host "3/4 Environnement Python (.venv) et dépendances…"
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "4/4 Démarrage sur http://localhost:7860 (prête quand /ready répond 'ready')…"
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:RETRIEVAL_ENGINE = "embeddings"
$env:OLLAMA_TIMEOUT_SECONDS = "300"
& .\.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 7860
