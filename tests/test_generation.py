"""Tests de la génération (Ollama simulé) et du contrôle des citations."""
import pytest
from fastapi import HTTPException

from tests.conftest import FakeResponse

SOURCE = [{"rank": 1, "source_id": "S1", "title": "Garanties", "content": "Le contrat couvre X."}]


# ---------- Contrôle des citations ----------

def test_audit_recognizes_available_sources(app_module):
    result = app_module.audit_citations(
        "La garantie est décrite [S1] et les exclusions dans [S2].",
        [{"source_id": "S1"}, {"source_id": "S2"}],
    )
    assert result["citations_found"] is True
    assert result["valid_source_ids"] == ["S1", "S2"]
    assert result["unknown_source_ids"] == []
    assert result["format_check_passed"] is True


def test_audit_flags_unknown_source_id(app_module):
    result = app_module.audit_citations("Règle dans [S7].", [{"source_id": "S1"}])
    assert result["unknown_source_ids"] == ["S7"]
    assert result["format_check_passed"] is False


def test_audit_reports_missing_citations(app_module):
    result = app_module.audit_citations("Pas de référence.", [{"source_id": "S1"}])
    assert result["citations_found"] is False
    assert result["available_but_uncited_source_ids"] == ["S1"]
    assert result["format_check_passed"] is False


def test_audit_sorts_ids_numerically(app_module):
    result = app_module.audit_citations(
        "[S10] puis [S2]", [{"source_id": "S2"}, {"source_id": "S10"}]
    )
    assert result["valid_source_ids"] == ["S2", "S10"]


def test_audit_does_not_claim_factual_validation(app_module):
    result = app_module.audit_citations("Information [S1].", [{"source_id": "S1"}])
    assert "reste à vérifier" in result["scope"]


def test_audit_tolerates_non_text_answer(app_module):
    assert app_module.audit_citations(None, SOURCE)["citations_found"] is False


# ---------- Construction des consignes ----------

def test_prompt_contains_sources_and_grounding_rules(app_module):
    prompt = app_module.build_prompt("Que couvre le contrat ?", app_module.build_context(SOURCE))
    assert "SOURCE [S1]" in prompt
    assert "N'invente jamais d'identifiant de source" in prompt
    assert "uniquement" in prompt and "Que couvre le contrat ?" in prompt


def test_revision_prompt_repeats_question_sources_and_allowed_ids(app_module):
    prompt = app_module.build_revision_prompt(
        "Que couvre la garantie ?", app_module.build_context(SOURCE), "Réponse sans source", ["S1"]
    )
    for expected in ("QUESTION ORIGINALE", "SOURCE [S1]", "[S1]", "RÉPONSE PRÉCÉDENTE À CORRIGER"):
        assert expected in prompt


# ---------- Appels à Ollama simulés ----------

def test_generate_answer_returns_cited_text(app_module, monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse({"response": "La garantie est décrite dans le contrat [S1]."})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    answer = app_module.generate_answer("Que couvre le contrat ?", SOURCE)

    assert answer == "La garantie est décrite dans le contrat [S1]."
    assert captured["payload"]["model"] == app_module.OLLAMA_MODEL
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == pytest.approx(0.1)
    assert captured["timeout"] == app_module.OLLAMA_TIMEOUT_SECONDS


@pytest.mark.parametrize("payload", [{}, {"response": "   "}, {"response": None}, []])
def test_generate_answer_rejects_unusable_payload(app_module, monkeypatch, payload):
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: FakeResponse(payload))
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", [])
    assert error.value.status_code == 502


def test_generate_answer_maps_connection_error_to_502(app_module, monkeypatch):
    def fail(*args, **kwargs):
        raise app_module.requests.exceptions.ConnectionError("Ollama arrêté")

    monkeypatch.setattr(app_module.requests, "post", fail)
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", [])
    assert error.value.status_code == 502


def test_generate_answer_maps_http_error_to_502(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module.requests, "post", lambda *a, **k: FakeResponse({}, ok=False)
    )
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", SOURCE)
    assert error.value.status_code == 502


def test_generate_answer_retries_when_citations_are_missing(app_module, monkeypatch):
    prompts = []
    responses = [
        {"response": "La garantie couvre les événements indiqués."},
        {"response": "La garantie couvre les événements décrits [S1]."},
    ]

    def fake_post(url, json, timeout):
        prompts.append(json["prompt"])
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    answer = app_module.generate_answer("Que couvre la garantie ?", SOURCE)

    assert answer.endswith("[S1].")
    assert len(prompts) == 2
    assert "RÉPONSE PRÉCÉDENTE À CORRIGER" in prompts[1]


def test_generate_answer_does_not_retry_when_first_answer_is_cited(app_module, monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(1)
        return FakeResponse({"response": "Réponse [S1]."})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    app_module.generate_answer("Question ?", SOURCE)
    assert len(calls) == 1


def test_generate_answer_retries_when_citation_is_invented(app_module, monkeypatch):
    responses = [{"response": "Selon [S9], oui."}, {"response": "Selon [S1], oui."}]
    monkeypatch.setattr(
        app_module.requests, "post", lambda *a, **k: FakeResponse(responses.pop(0))
    )
    assert app_module.generate_answer("Question ?", SOURCE) == "Selon [S1], oui."


def test_safe_fallback_shows_source_excerpts_not_uncited_text(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "post",
        lambda *a, **k: FakeResponse({"response": "La garantie s'applique dans plusieurs cas."}),
    )
    answer = app_module.generate_answer(
        "Que couvre la garantie ?",
        [{"rank": 1, "source_id": "S1", "title": "Conditions de garantie",
          "content": "La garantie couvre les dommages accidentels décrits au contrat."}],
    )
    assert "suffisamment traçable" in answer
    assert "Conditions de garantie** [S1]" in answer
    assert "La garantie s'applique dans plusieurs cas." not in answer


def test_safe_fallback_truncates_long_excerpts(app_module):
    long_text = "mot " * 1000
    fallback = app_module.build_safe_fallback(
        [{"source_id": "S1", "title": "Long", "content": long_text}]
    )
    excerpt = fallback.split("[S1] : ", 1)[1]
    assert len(excerpt) <= app_module.FALLBACK_EXCERPT_CHARACTERS + 1
    assert excerpt.endswith("…")


def test_safe_fallback_without_usable_source(app_module):
    assert "Aucun extrait exploitable" in app_module.build_safe_fallback([])


def test_diagnostic_mode_logs_raw_answers_audits_and_durations(app_module, monkeypatch, capsys):
    monkeypatch.setenv("RAG_DIAGNOSTICS", "1")
    responses = [
        {"response": "Première sortie sans référence."},
        {"response": "Seconde sortie toujours sans référence."},
    ]
    monkeypatch.setattr(
        app_module.requests, "post", lambda *a, **k: FakeResponse(responses.pop(0))
    )
    answer = app_module.generate_answer("Question ?", [{"source_id": "S1", "title": "T", "content": "C"}])
    output = capsys.readouterr().out

    assert "suffisamment traçable" in answer
    assert "tentative 1 — durée Ollama" in output
    assert "tentative 2 / révision — durée Ollama" in output
    assert "Première sortie sans référence." in output
    assert output.count("audit :") == 2


def test_diagnostic_mode_is_silent_by_default(app_module, monkeypatch, capsys):
    monkeypatch.delenv("RAG_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(
        app_module.requests, "post", lambda *a, **k: FakeResponse({"response": "Oui [S1]."})
    )
    app_module.generate_answer("Question ?", SOURCE)
    assert "RAG_DIAGNOSTICS" not in capsys.readouterr().out
