"""Tests des routes de l'API.

Les fonctions de route sont appelées directement ; un second groupe de tests
passe par le client HTTP de FastAPI (``TestClient``) pour vérifier la
sérialisation JSON et les codes de statut tels qu'un navigateur les verrait.
"""
from pathlib import Path

import pytest
from fastapi import HTTPException

from tests.conftest import FakeResponse


@pytest.fixture
def cited_generation(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module, "generate_answer", lambda question, results: "Réponse de test [S1]."
    )


# ---------- Modèle de requête ----------

def test_question_is_trimmed(app_module):
    assert app_module.Question(question="  Mes droits ?  ").question == "Mes droits ?"


@pytest.mark.parametrize("value", ["   ", "x" * 2001])
def test_invalid_questions_are_rejected(app_module, value):
    with pytest.raises(Exception):
        app_module.Question(question=value)


def test_question_at_limit_is_accepted(app_module):
    assert len(app_module.Question(question="x" * 2000).question) == 2000


# ---------- Version, santé, disponibilité ----------

def test_single_version_source(app_module, project_root):
    expected = (project_root / "VERSION").read_text(encoding="utf-8").strip()
    assert app_module.APP_VERSION == expected
    assert app_module.version()["app_version"] == expected
    assert app_module.health() == {"status": "ok", "app_version": expected}
    assert app_module.app.version == expected


def test_int_setting_parser(app_module):
    assert app_module.read_int_setting("30", 120, 5) == 30
    assert app_module.read_int_setting("1", 120, 5) == 5
    assert app_module.read_int_setting("abc", 120, 5) == 120
    assert app_module.read_int_setting(None, 120, 5) == 120


def test_tags_url_is_derived_from_generate_url(app_module):
    assert (
        app_module.ollama_tags_url("http://127.0.0.1:11434/api/generate")
        == "http://127.0.0.1:11434/api/tags"
    )


@pytest.mark.parametrize(
    "configured, installed, expected",
    [
        ("llama3.2:3b", {"llama3.2:3b"}, True),
        ("llama3.2", {"llama3.2:latest"}, True),
        ("llama3.2:3b", {"llama3.2:1b"}, False),
        ("llama3.2:3b", set(), False),
    ],
)
def test_model_installation_check_is_exact(app_module, configured, installed, expected):
    assert app_module.model_is_installed(configured, installed) is expected


def test_ready_when_ollama_and_model_are_available(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **k: FakeResponse({"models": [{"name": app_module.OLLAMA_MODEL}]}),
    )
    response = app_module.readiness()
    assert response["status"] == "ready"
    assert all(response["checks"].values())
    assert "ne valide pas la qualité métier" in response["note"]


def test_not_ready_when_ollama_is_unreachable(app_module, monkeypatch):
    def unavailable(*args, **kwargs):
        raise app_module.requests.exceptions.ConnectionError("Ollama arrêté")

    monkeypatch.setattr(app_module.requests, "get", unavailable)
    with pytest.raises(HTTPException) as error:
        app_module.readiness()
    assert error.value.status_code == 503
    assert error.value.detail["checks"]["ollama_reachable"] is False


def test_not_ready_while_model_is_still_downloading(app_module, monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *a, **k: FakeResponse({"models": []}))
    with pytest.raises(HTTPException) as error:
        app_module.readiness()
    checks = error.value.detail["checks"]
    assert checks["ollama_reachable"] is True
    assert checks["ollama_model_available"] is False


# ---------- Recherche ----------

def test_search_response_structure(app_module, cited_generation):
    response = app_module.search(app_module.Question(question="Que couvre la garantie vol ?"))
    assert response["app_version"] == app_module.APP_VERSION
    assert response["answer"].endswith("[S1].")
    assert [r["source_id"] for r in response["results"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert response["citation_audit"]["valid_source_ids"] == ["S1"]
    first = response["results"][0]
    for key in ("title", "content", "tfidf_cosine_similarity",
                "lexical_relevance_score", "hybrid_rank_score"):
        assert key in first
    assert len(first["content"]) <= app_module.CONTEXT_CHARACTERS_PER_SOURCE
    for key in ("tfidf_cosine_similarity", "lexical_relevance_score", "hybrid_rank_score"):
        assert "probabilité" in response["score_explanation"][key]


def test_search_results_are_sorted_by_fusion_score(app_module, cited_generation):
    response = app_module.search(app_module.Question(question="vol de bijoux"))
    scores = [r["hybrid_rank_score"] for r in response["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_pipeline_metrics(app_module, cited_generation):
    metrics = app_module.search(app_module.Question(question="Question métriques ?"))[
        "pipeline_metrics"
    ]
    assert metrics["document_count"] == len(app_module.documents)
    assert metrics["candidate_count"] == min(20, len(app_module.documents))
    assert metrics["returned_count"] == 5
    for key in ("vectorization_ms", "ranking_ms", "generation_ms", "total_ms"):
        assert metrics[key] >= 0
    assert "varient" in metrics["note"]


def test_ollama_payload_is_tuned_for_speed(app_module, monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse({"response": "Oui [S1]."})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    app_module.generate_answer("Q ?", [{"source_id": "S1", "title": "T", "content": "C"}])
    assert captured["keep_alive"] == app_module.OLLAMA_KEEP_ALIVE
    assert captured["options"]["num_predict"] == app_module.OLLAMA_NUM_PREDICT


# ---------- Chemins et pages ----------

def test_project_paths_are_absolute_and_exist(app_module):
    for path in (app_module.PROJECT_ROOT, app_module.DATA_FILE, app_module.INDEX_FILE,
                 app_module.CLAIMS_INDEX_FILE):
        assert path.is_absolute() and path.exists()


def test_home_serves_index_from_any_working_directory(app_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = app_module.home()
    assert Path(response.path).resolve() == app_module.INDEX_FILE.resolve()


def test_all_routes_are_registered(app_module):
    paths = {route.path for route in app_module.app.routes}
    for expected in ("/", "/health", "/ready", "/version", "/search", "/claims",
                     "/api/claims/summary", "/api/claims/cases", "/api/claims/model",
                     "/api/claims/{claim_id}/analysis"):
        assert expected in paths


# ---------- Tests HTTP de bout en bout (sans Ollama) ----------

@pytest.fixture
def client(app_module):
    pytest.importorskip("httpx")
    testclient = pytest.importorskip("fastapi.testclient")
    return testclient.TestClient(app_module.app)


def test_http_health_and_version(client, app_module):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/version").json()["app_version"] == app_module.APP_VERSION


def test_http_pages_are_served(client):
    for path in ("/", "/claims"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


def test_http_search_end_to_end(client, cited_generation):
    response = client.post("/search", json={"question": "Ma serrure est cassée"})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 5


def test_http_search_validation_errors(client):
    assert client.post("/search", json={"question": "   "}).status_code == 422
    assert client.post("/search", json={}).status_code == 422


def test_http_search_returns_502_when_ollama_is_down(client, app_module, monkeypatch):
    def unavailable(*args, **kwargs):
        raise app_module.requests.exceptions.ConnectionError("Ollama arrêté")

    monkeypatch.setattr(app_module.requests, "post", unavailable)
    response = client.post("/search", json={"question": "franchise"})
    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]


def test_http_claims_routes(client):
    assert client.get("/api/claims/summary").status_code == 200
    assert client.get("/api/claims/cases?limit=5").status_code == 200
    assert client.get("/api/claims/cases?limit=0").status_code == 422
    assert client.get("/api/claims/C00001/analysis").status_code == 200
    assert client.get("/api/claims/INCONNU/analysis").status_code == 404
