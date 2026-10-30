"""Tests de la génération typée (Ollama simulé), de la vérification des extraits
et du contrôle des citations."""
import json

import pytest
from fastapi import HTTPException

from src import generation
from tests.conftest import FakeResponse

CONTENT = "La garantie vol couvre les biens mobiliers dérobés lors d'une effraction constatée."
SOURCES = [{"rank": 1, "source_id": "S1", "title": "Garantie vol", "content": CONTENT}]


def model_output(answer_found=True, complete=True, answer="Le vol avec effraction est couvert.",
                 evidence=None, confidence=0.8, caveats=""):
    if evidence is None:
        evidence = [{"source_id": "S1", "quote": "couvre les biens mobiliers dérobés lors d'une effraction"}]
    return json.dumps({
        "answer_found": answer_found, "complete_answer_found": complete, "answer": answer,
        "evidence": evidence, "confidence": confidence, "caveats": caveats,
    }, ensure_ascii=False)


def scripted(outputs):
    """Faux modèle renvoyant successivement les sorties prévues."""
    prompts = []

    def call(prompt, label):
        prompts.append(prompt)
        return outputs.pop(0)

    return call, prompts


# ---------- Vérification des extraits ----------

def test_quote_found_verbatim_is_verified():
    assert generation.quote_is_in_source("biens mobiliers dérobés", CONTENT)


def test_quote_check_ignores_case_accents_and_punctuation():
    assert generation.quote_is_in_source("BIENS MOBILIERS DEROBES, lors d'une effraction", CONTENT)


def test_quote_with_small_variation_is_tolerated():
    quote = "La garantie vol couvre les biens mobiliers dérobés lors de une effraction constatée"
    assert generation.quote_is_in_source(quote, CONTENT)


@pytest.mark.parametrize("quote", [
    "la garantie couvre aussi les catastrophes naturelles",
    "vol",
    "",
])
def test_invented_or_too_short_quote_is_rejected(quote):
    assert not generation.quote_is_in_source(quote, CONTENT)


def test_evidence_with_unknown_source_is_not_verified():
    typed = generation.parse_typed_answer(model_output(
        evidence=[{"source_id": "S9", "quote": "couvre les biens mobiliers dérobés"}]
    ))
    checked = generation.verify_answer(typed, SOURCES)
    assert checked[0]["source_known"] is False and checked[0]["verified"] is False


@pytest.mark.parametrize("raw", ["pas du json", "[]", '{"answer": "x"}',
                                 model_output(confidence=1.7)])
def test_invalid_model_output_is_rejected(raw):
    assert generation.parse_typed_answer(raw) is None


# ---------- Décision du statut ----------

THREE_SOURCES = SOURCES + [
    {"rank": 2, "source_id": "S2", "title": "Exclusions", "content": "Le vol sans effraction ni violence n'est pas garanti par le contrat."},
    {"rank": 3, "source_id": "S3", "title": "Plafonds", "content": "Les bijoux sont indemnisés dans la limite du plafond prévu aux conditions particulières."},
]


def test_complete_verified_answer_at_top1_stops_the_loop():
    call, prompts = scripted([model_output()])
    result = generation.generate_typed_answer("Le vol est-il couvert ?", THREE_SOURCES, call)
    assert result["status"] == "answered"
    assert result["attempts"] == 1 and result["sources_read"] == 1
    assert "SOURCE [S1]" in prompts[0] and "SOURCE [S2]" not in prompts[0]
    assert result["answer"].endswith("[S1]")
    assert result["typed_answer"]["evidence"][0]["verified"] is True


def test_partial_answer_at_top1_reads_three_sources():
    second = model_output(evidence=[
        {"source_id": "S1", "quote": "couvre les biens mobiliers dérobés lors d'une effraction"},
        {"source_id": "S2", "quote": "le vol sans effraction ni violence n'est pas garanti"},
    ])
    call, prompts = scripted([model_output(complete=False), second])
    result = generation.generate_typed_answer("Le vol est-il couvert ?", THREE_SOURCES, call)
    assert result["status"] == "answered"
    assert result["attempts"] == 2 and result["sources_read"] == 3
    assert "SOURCE [S3]" in prompts[1]
    assert result["answer"].endswith("[S1] [S2]")


def test_partial_answer_is_kept_when_three_sources_do_not_complete_it():
    call, _ = scripted([model_output(complete=False), model_output(complete=False)])
    assert generation.generate_typed_answer("Q ?", THREE_SOURCES, call)["status"] == "partial"


def test_not_found_at_top1_is_checked_against_three_sources():
    not_found = model_output(answer_found=False, complete=False, evidence=[], confidence=0.0,
                             answer="Les documents ne permettent pas de répondre.")
    call, prompts = scripted([not_found, not_found])
    result = generation.generate_typed_answer("Cours de l'action Apple ?", THREE_SOURCES, call)
    assert result["status"] == "not_found"
    assert len(prompts) == 2
    assert "ne permettent pas" in result["answer"]


def test_single_source_needs_a_single_call():
    call, prompts = scripted([model_output(complete=False)])
    result = generation.generate_typed_answer("Q ?", SOURCES, call)
    assert result["status"] == "partial" and len(prompts) == 1


def test_unverifiable_quote_at_top1_is_retried_with_three_sources():
    bad = model_output(evidence=[{"source_id": "S1", "quote": "texte inventé qui n'existe nulle part"}])
    call, prompts = scripted([bad, model_output()])
    result = generation.generate_typed_answer("Q ?", THREE_SOURCES, call)
    assert result["status"] == "answered" and result["attempts"] == 2


def test_invalid_json_at_top1_is_retried():
    call, prompts = scripted(["réponse en texte libre", model_output()])
    assert generation.generate_typed_answer("Q ?", THREE_SOURCES, call)["status"] == "answered"
    assert len(prompts) == 2


def test_fallback_after_two_unverifiable_answers():
    bad = model_output(evidence=[])
    call, prompts = scripted([bad, bad])
    result = generation.generate_typed_answer("Q ?", THREE_SOURCES, call)
    assert result["status"] == "fallback"
    assert result["typed_answer"] is None
    assert "suffisamment traçable" in result["answer"]
    assert "Garantie vol** [S1]" in result["answer"]
    assert "Le vol avec effraction est couvert." not in result["answer"]


def test_declared_confidence_does_not_override_verification():
    bad = model_output(confidence=1.0, evidence=[{"source_id": "S1", "quote": "réponse sûre mais inventée de toutes pièces"}])
    call, _ = scripted([bad, bad])
    assert generation.generate_typed_answer("Q ?", THREE_SOURCES, call)["status"] == "fallback"


def test_stage_sources_are_truncated_for_speed():
    long_sources = [{"source_id": f"S{i}", "title": "T", "content": "mot " * 1000} for i in range(1, 6)]
    stage = generation.stage_sources(long_sources, 3, 900)
    assert len(stage) == 3
    assert all(len(source["content"]) <= 901 for source in stage)
    assert len(long_sources[0]["content"]) == 4000  # l'original n'est pas modifié


def test_fallback_truncates_long_excerpts():
    text = generation.build_fallback_text([{"source_id": "S1", "title": "Long", "content": "mot " * 1000}])
    excerpt = text.split("[S1] : ", 1)[1]
    assert len(excerpt) <= generation.FALLBACK_EXCERPT_CHARACTERS + 1 and excerpt.endswith("…")


def test_fallback_without_usable_source():
    assert "Aucun extrait exploitable" in generation.build_fallback_text([])


def test_prompt_requires_typed_grounded_answer():
    prompt = generation.build_prompt("Que couvre le contrat ?", generation.build_context(SOURCES))
    for expected in ("SOURCE [S1]", "answer_found", "MOT POUR MOT", "N'invente jamais"):
        assert expected in prompt


def test_schema_exposes_sufficiency_indicators():
    properties = generation.ANSWER_SCHEMA["properties"]
    for field in ("answer_found", "complete_answer_found", "answer", "evidence", "confidence", "caveats"):
        assert field in properties


# ---------- Intégration avec l'API (Ollama simulé) ----------

def test_api_sends_schema_and_speed_settings(app_module, monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse({"response": model_output()})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    result = app_module.generate_answer("Le vol est-il couvert ?", SOURCES)
    assert result["status"] == "answered"
    assert captured["format"] == generation.ANSWER_SCHEMA
    assert captured["stream"] is False
    assert captured["keep_alive"] == app_module.OLLAMA_KEEP_ALIVE
    assert captured["options"]["num_predict"] == app_module.OLLAMA_NUM_PREDICT
    assert captured["options"]["num_ctx"] == app_module.OLLAMA_NUM_CTX
    assert captured["options"]["temperature"] == pytest.approx(0.1)


@pytest.mark.parametrize("payload", [{}, {"response": "   "}, {"response": None}, []])
def test_api_maps_unusable_payload_to_502(app_module, monkeypatch, payload):
    monkeypatch.setattr(app_module.requests, "post", lambda *a, **k: FakeResponse(payload))
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", SOURCES)
    assert error.value.status_code == 502


@pytest.mark.parametrize("failure", ["connection", "http"])
def test_api_maps_ollama_errors_to_502(app_module, monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == "connection":
            raise app_module.requests.exceptions.ConnectionError("Ollama arrêté")
        return FakeResponse({}, ok=False)

    monkeypatch.setattr(app_module.requests, "post", fail)
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", SOURCES)
    assert error.value.status_code == 502


def test_diagnostic_mode_logs_raw_outputs_audits_and_durations(app_module, monkeypatch, capsys):
    monkeypatch.setenv("RAG_DIAGNOSTICS", "1")
    outputs = ["première sortie illisible", "seconde sortie illisible"]
    monkeypatch.setattr(app_module.requests, "post",
                        lambda *a, **k: FakeResponse({"response": outputs.pop(0)}))
    result = app_module.generate_answer("Q ?", THREE_SOURCES)
    output = capsys.readouterr().out
    assert result["status"] == "fallback"
    assert "tentative 1 (top-1) — durée Ollama" in output
    assert "tentative 2 (top-3) — durée Ollama" in output
    assert "première sortie illisible" in output
    assert output.count("audit :") == 2


def test_diagnostic_mode_is_silent_by_default(app_module, monkeypatch, capsys):
    monkeypatch.delenv("RAG_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(app_module.requests, "post",
                        lambda *a, **k: FakeResponse({"response": model_output()}))
    app_module.generate_answer("Q ?", SOURCES)
    assert "RAG_DIAGNOSTICS" not in capsys.readouterr().out


# ---------- Contrôle des citations sur le texte affiché ----------

def test_audit_recognizes_available_sources(app_module):
    result = app_module.audit_citations("Décrit [S1] et [S2].", [{"source_id": "S1"}, {"source_id": "S2"}])
    assert result["valid_source_ids"] == ["S1", "S2"] and result["format_check_passed"] is True


def test_audit_flags_unknown_and_missing_citations(app_module):
    unknown = app_module.audit_citations("Dans [S7].", [{"source_id": "S1"}])
    assert unknown["unknown_source_ids"] == ["S7"] and unknown["format_check_passed"] is False
    missing = app_module.audit_citations("Sans référence.", [{"source_id": "S1"}])
    assert missing["available_but_uncited_source_ids"] == ["S1"]
    assert "reste à vérifier" in missing["scope"]


def test_audit_sorts_ids_numerically(app_module):
    result = app_module.audit_citations("[S10] puis [S2]", [{"source_id": "S2"}, {"source_id": "S10"}])
    assert result["valid_source_ids"] == ["S2", "S10"]


# ---------- Incohérences du modèle (observées avec llama3.2:1b) ----------

def test_not_found_with_verified_quote_is_downgraded_to_partial():
    contradictory = model_output(answer_found=False, complete=False, confidence=0.5)
    call, _ = scripted([contradictory, contradictory])
    result = generation.generate_typed_answer("Q ?", THREE_SOURCES, call)
    assert result["status"] == "partial"
    assert result["typed_answer"]["model_indicators_inconsistent"] is True
    assert result["answer"].endswith("[S1]")


def test_consistent_answer_is_not_flagged():
    call, _ = scripted([model_output()])
    result = generation.generate_typed_answer("Q ?", THREE_SOURCES, call)
    assert result["typed_answer"]["model_indicators_inconsistent"] is False


def test_html_entities_from_the_model_are_decoded():
    escaped = model_output(
        answer="L&#39;effraction est couverte.",
        evidence=[{"source_id": "S1", "quote": "couvre les biens mobiliers dérobés lors d&#39;une effraction"}],
    )
    call, _ = scripted([escaped])
    result = generation.generate_typed_answer("Q ?", THREE_SOURCES, call)
    assert result["status"] == "answered"
    assert "&#39;" not in result["answer"] and "L'effraction" in result["answer"]
    assert "&#39;" not in result["typed_answer"]["evidence"][0]["quote"]
