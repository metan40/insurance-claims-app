"""Configuration commune : l'application est importée sans démarrer Ollama.

Le chargement de ``src.app`` indexe le corpus (TF-IDF) et entraîne le modèle
de démonstration des sinistres ; aucun appel réseau n'est effectué à l'import.
Les appels à Ollama sont remplacés par des doubles dans chaque test.
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    """Réponse HTTP minimale imitant ``requests.Response``."""

    def __init__(self, payload, ok=True):
        self.payload = payload
        self.ok = ok

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.exceptions.HTTPError("erreur simulée")

    def json(self):
        return self.payload


@pytest.fixture(scope="session")
def app_module():
    from src import app
    return app


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT
