"""Tests unitaires de l'API sans charger les modèles ML ni démarrer Ollama."""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException


def load_app_module(change_directory=True):
    """Charge app.py avec des doubles pour éviter le chargement des modèles."""
    project_root = Path(__file__).resolve().parents[1]
    if change_directory:
        os.chdir(project_root)

    fake_st = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return [[1.0] for _ in texts]

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return [1.0 for _ in pairs]

    fake_st.SentenceTransformer = FakeSentenceTransformer
    fake_st.CrossEncoder = FakeCrossEncoder
    old = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_st
    try:
        spec = importlib.util.spec_from_file_location(
            "app_v4_test", project_root / "src" / "app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = old


@pytest.fixture(scope="module")
def app_module():
    return load_app_module()


def test_question_is_trimmed(app_module):
    value = app_module.Question(question="  Quels sont mes droits ?  ")
    assert value.question == "Quels sont mes droits ?"


def test_empty_question_is_rejected(app_module):
    with pytest.raises(Exception):
        app_module.Question(question="   ")


def test_question_over_2000_characters_is_rejected(app_module):
    with pytest.raises(Exception):
        app_module.Question(question="x" * 2001)


def test_question_at_limit_is_accepted(app_module):
    value = app_module.Question(question="x" * 2000)
    assert len(value.question) == 2000


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_generate_answer_returns_text_and_includes_source_ids(app_module, monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"response": "La garantie est décrite dans le contrat [S1]."})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    answer = app_module.generate_answer(
        "Que couvre le contrat ?",
        [{"rank": 1, "source_id": "S1", "title": "Garanties", "content": "Le contrat couvre X."}],
    )

    assert answer == "La garantie est décrite dans le contrat [S1]."
    assert "SOURCE [S1]" in captured["payload"]["prompt"]
    assert "N'invente jamais d'identifiant de source" in captured["payload"]["prompt"]


@pytest.mark.parametrize("payload", [{}, {"response": "   "}, {"response": None}, []])
def test_generate_answer_rejects_unusable_ollama_payload(app_module, monkeypatch, payload):
    monkeypatch.setattr(
        app_module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(payload),
    )
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", [])
    assert error.value.status_code == 502


def test_generate_answer_retries_when_citations_are_missing(app_module, monkeypatch):
    captured_prompts = []
    responses = [
        {"response": "La garantie couvre les événements indiqués dans le contrat."},
        {"response": "La garantie couvre les événements décrits dans le contrat [S1]."},
    ]

    def fake_post(url, json, timeout):
        captured_prompts.append(json["prompt"])
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    answer = app_module.generate_answer(
        "Que couvre la garantie ?",
        [{"rank": 1, "source_id": "S1", "title": "Garanties", "content": "La garantie couvre les événements décrits."}],
    )

    assert answer.endswith("[S1].")
    assert len(captured_prompts) == 2
    assert "RÉPONSE PRÉCÉDENTE À CORRIGER" in captured_prompts[1]
    assert "[S1]" in captured_prompts[1]
    assert "QUESTION ORIGINALE" in captured_prompts[1]
    assert "SOURCE [S1]" in captured_prompts[1]
    assert "La garantie couvre les événements décrits." in captured_prompts[1]


def test_generate_answer_uses_safe_fallback_if_retry_still_has_no_citations(app_module, monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["prompt"])
        return FakeResponse({"response": "Cette garantie s'applique dans plusieurs cas."})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    answer = app_module.generate_answer(
        "Que couvre la garantie ?",
        [{"rank": 1, "source_id": "S1", "title": "Garanties", "content": "Texte de garantie."}],
    )

    assert len(calls) == 2
    assert "suffisamment traçable" in answer
    assert "Cette garantie s'applique" not in answer


def test_generate_answer_maps_connection_error_to_502(app_module, monkeypatch):
    def fail_post(*args, **kwargs):
        raise app_module.requests.exceptions.ConnectionError("test local")

    monkeypatch.setattr(app_module.requests, "post", fail_post)
    with pytest.raises(HTTPException) as error:
        app_module.generate_answer("Question ?", [])
    assert error.value.status_code == 502


def test_audit_citations_recognizes_available_sources(app_module):
    result = app_module.audit_citations(
        "La garantie est décrite [S1] et les exclusions dans [S2].",
        [{"source_id": "S1"}, {"source_id": "S2"}],
    )
    assert result["citations_found"] is True
    assert result["valid_source_ids"] == ["S1", "S2"]
    assert result["unknown_source_ids"] == []
    assert result["format_check_passed"] is True


def test_audit_citations_flags_unknown_source_id(app_module):
    result = app_module.audit_citations(
        "Cette règle apparaît dans [S7].",
        [{"source_id": "S1"}, {"source_id": "S2"}],
    )
    assert result["unknown_source_ids"] == ["S7"]
    assert result["format_check_passed"] is False


def test_audit_citations_reports_missing_citations(app_module):
    result = app_module.audit_citations(
        "Les garanties sont détaillées dans le document.",
        [{"source_id": "S1"}],
    )
    assert result["citations_found"] is False
    assert result["available_but_uncited_source_ids"] == ["S1"]
    assert result["format_check_passed"] is False


def test_audit_citations_does_not_claim_factual_validation(app_module):
    result = app_module.audit_citations("Information [S1].", [{"source_id": "S1"}])
    assert "ne prouve pas" in result["scope"] or "reste à vérifier" in result["scope"]


def test_search_response_includes_citation_audit(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "generate_answer",
        lambda question, results: "La réponse est décrite dans le document [S1].",
    )
    response = app_module.search(app_module.Question(question="Question de test ?"))

    assert response["answer"].endswith("[S1].")
    assert response["citation_audit"]["valid_source_ids"] == ["S1"]
    assert response["citation_audit"]["unknown_source_ids"] == []
    assert response["results"][0]["source_id"] == "S1"
    assert "reranker_score" in response["results"][0]
    assert "embedding_similarity" in response["results"][0]
    assert "pas une probabilité" in response["score_explanation"]["reranker_score"]
    assert "similarité cosinus" in response["score_explanation"]["embedding_similarity"].lower()


def test_project_paths_are_absolute(app_module):
    assert app_module.PROJECT_ROOT.is_absolute()
    assert app_module.DATA_FILE.is_absolute()
    assert app_module.INDEX_FILE.is_absolute()


def test_frontend_displays_uncited_sources_and_score_limitations(app_module):
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "static" / "index.html").read_text(encoding="utf-8")
    assert "available_but_uncited_source_ids" in html
    assert "source(s) disponible(s) mais non citée(s)" in html
    assert "pas une probabilité" in html
    assert "Similarité cosinus des embeddings" in html


def test_search_response_includes_pipeline_metrics(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "generate_answer",
        lambda question, results: "Réponse de test [S1].",
    )
    response = app_module.search(app_module.Question(question="Question métriques ?"))
    metrics = response["pipeline_metrics"]

    assert metrics["document_count"] == len(app_module.documents)
    assert metrics["candidate_count"] == min(20, len(app_module.documents))
    assert metrics["returned_count"] == len(response["results"])
    for key in ("embedding_ms", "retrieval_ms", "reranking_ms", "generation_ms", "total_ms"):
        assert isinstance(metrics[key], (int, float))
        assert metrics[key] >= 0
    assert "varient" in metrics["note"] or "varient" in metrics["note"].lower()


def test_frontend_displays_pipeline_timing_metrics(app_module):
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "static" / "index.html").read_text(encoding="utf-8")
    assert "pipeline_metrics" in html
    assert "Durées observées (ms)" in html
    assert "génération" in html


def test_project_files_resolve_from_another_working_directory(tmp_path, monkeypatch):
    """Les chemins du projet doivent dépendre du fichier app.py, pas du cwd."""
    monkeypatch.chdir(tmp_path)
    original_cwd = Path.cwd()
    module = load_app_module(change_directory=False)

    assert Path.cwd() == original_cwd
    assert module.DATA_FILE.is_file()
    assert module.INDEX_FILE.is_file()
    response = module.home()
    assert Path(response.path).resolve() == module.INDEX_FILE.resolve()


def test_version_endpoint_identifies_the_running_build(app_module):
    response = app_module.version()
    assert response["app_version"] == "7.5.0"
    assert response["ollama_model"] == app_module.OLLAMA_MODEL


def test_search_response_includes_app_version(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "generate_answer",
        lambda question, results: "Réponse de test [S1].",
    )
    response = app_module.search(app_module.Question(question="Question version ?"))
    assert response["app_version"] == "7.5.0"


def test_diagnostic_mode_logs_both_raw_answers_audits_and_call_durations(app_module, monkeypatch, capsys):
    monkeypatch.setenv("RAG_DIAGNOSTICS", "1")
    responses = [
        {"response": "Première sortie sans référence."},
        {"response": "Seconde sortie toujours sans référence."},
    ]

    def fake_post(url, json, timeout):
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    answer = app_module.generate_answer(
        "Question diagnostique ?",
        [{"source_id": "S1", "title": "Titre", "content": "Contenu"}],
    )

    output = capsys.readouterr().out
    assert "suffisamment traçable" in answer
    assert "tentative 1 — durée Ollama" in output
    assert "tentative 2 / révision — durée Ollama" in output
    assert "Première sortie sans référence." in output
    assert "Seconde sortie toujours sans référence." in output
    assert output.count("audit :") == 2
    assert "format_check_passed': False" in output


def test_generate_answer_fallback_shows_cited_source_excerpts(app_module, monkeypatch):
    """Le repli doit rester utile et traçable si Ollama ignore encore les citations."""
    monkeypatch.setattr(
        app_module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {"response": "La garantie s'applique dans plusieurs cas."}
        ),
    )
    answer = app_module.generate_answer(
        "Que couvre la garantie ?",
        [
            {
                "rank": 1,
                "source_id": "S1",
                "title": "Conditions de garantie",
                "content": "La garantie couvre les dommages accidentels décrits au contrat.",
            }
        ],
    )

    assert "suffisamment traçable" in answer
    assert "Conditions de garantie** [S1]" in answer
    assert "La garantie couvre les dommages accidentels" in answer
    assert "La garantie s'applique dans plusieurs cas." not in answer


def test_hybrid_candidate_selection_prioritizes_exact_insurance_concepts(app_module):
    """Le document définissant la franchise absolue doit rester candidat."""
    import numpy as np

    question = "Comment fonctionne la franchise absolue d'une assurance habitation ?"
    similarities = np.ones(len(app_module.documents), dtype=float)
    candidates, lexical_scores = app_module._select_candidate_indices(
        question, similarities, candidate_limit=20
    )
    target_index = next(
        index for index, title in enumerate(app_module.titles)
        if "franchise" in title.lower()
    )

    assert len(candidates) == 20
    assert target_index in candidates[:5]
    assert lexical_scores[target_index] > 0


def test_lexical_relevance_score_rewards_exact_title_terms(app_module):
    exact_score = app_module._lexical_relevance_score(
        "franchise absolue assurance habitation",
        "Comment fonctionne la franchise absolue ?",
        "L'assureur déduit la franchise de l'indemnisation.",
    )
    unrelated_score = app_module._lexical_relevance_score(
        "franchise absolue assurance habitation",
        "Attestation d'assurance habitation",
        "Le locataire remet un justificatif au propriétaire.",
    )

    assert exact_score > unrelated_score


def test_health_endpoint_reports_process_availability(app_module):
    response = app_module.health()
    assert response["status"] == "ok"
    assert response["app_version"] == "7.5.0"


def test_readiness_endpoint_checks_ollama_and_local_resources(app_module, monkeypatch):
    class FakeTagsResponse:
        ok = True

        def json(self):
            return {"models": [{"name": "llama3.2:3b"}]}

    monkeypatch.setattr(app_module.requests, "get", lambda *args, **kwargs: FakeTagsResponse())
    response = app_module.readiness()
    assert response["status"] == "ready"
    assert all(response["checks"].values())
    assert "ne valide pas la qualité métier" in response["note"]


def test_readiness_endpoint_returns_503_when_ollama_is_unavailable(app_module, monkeypatch):
    def unavailable(*args, **kwargs):
        raise app_module.requests.exceptions.ConnectionError("Ollama arrêté")

    monkeypatch.setattr(app_module.requests, "get", unavailable)
    with pytest.raises(HTTPException) as error:
        app_module.readiness()
    assert error.value.status_code == 503
    assert error.value.detail["status"] == "not_ready"


def test_final_hybrid_ranking_keeps_exact_insurance_topic_near_the_top(app_module, monkeypatch):
    """La fusion finale doit valoriser un titre qui reprend le concept demandé."""
    monkeypatch.setattr(
        app_module,
        "generate_answer",
        lambda question, results: "La franchise absolue est déduite de l'indemnisation [S1].",
    )
    response = app_module.search(
        app_module.Question(question="Comment fonctionne la franchise absolue d'une assurance habitation ?")
    )
    titles = [result["title"].lower() for result in response["results"]]
    assert any("franchise" in title for title in titles[:3])
    assert all("hybrid_rank_score" in result for result in response["results"])
    assert "hybrid_rank_score" in response["score_explanation"]


def test_frontend_and_delivery_guide_explain_operational_checks(app_module):
    project_root = Path(__file__).resolve().parents[1]
    guide = (project_root / "README_LIVRAISON_KEYZAN.md").read_text(encoding="utf-8")
    assert "GET /health" in guide
    assert "GET /ready" in guide
    assert "ne l'exposez pas sur Internet" in guide
    assert "fiche de retour" in guide.lower()


def test_frontend_explains_final_hybrid_rank_score(app_module):
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "static" / "index.html").read_text(encoding="utf-8")
    assert "hybrid_rank_score" in html
    assert "Score de fusion des rangs" in html
    assert "comme une probabilité de pertinence" in html


def test_claims_summary_excludes_pending_from_acceptance_rate(app_module):
    summary = app_module.get_claims_summary()
    assert summary["total_cases"] == 120
    assert summary["accepted_cases"] + summary["rejected_cases"] == summary["closed_cases"]
    assert summary["pending_excluded_from_acceptance_rate"] is True
    expected = summary["accepted_cases"] / summary["closed_cases"]
    assert summary["acceptance_rate"] == round(expected, 4)
    assert "fictif" in summary["warning"].lower()


def test_claims_cases_are_fictitious_and_json_safe(app_module):
    cases = app_module.get_claims_cases(12)
    assert len(cases) == 12
    assert all(case["jeu_de_donnees"] == "FICTIF_DEMONSTRATION" for case in cases)
    assert all(case["age_sinistre_jours"] >= 0 for case in cases)
    assert all(case["delai_traitement_jours"] is None or isinstance(case["delai_traitement_jours"], (int, float)) for case in cases)


def test_claims_model_excludes_pending_and_disclaims_real_world_validity(app_module):
    info = app_module.get_claims_model_info()
    assert info["pending_used_for_training"] is False
    assert info["closed_cases_count"] == info["training_cases_count"] + info["test_cases_count"]
    assert "synthétiques" in info["warning"].lower()
    assert "validation humaine" in info["decision_policy"].lower()


def test_claim_analysis_is_explanatory_not_an_automatic_decision(app_module):
    result = app_module.get_claim_analysis("C00001")
    assert result is not None
    assert 0 <= result["estimated_acceptance_probability"] <= 1
    assert result["human_review_required"] is True
    assert "ne constitue ni une décision" in result["warning"]
    assert app_module.get_claim_analysis("C99999") is None


def test_claims_dashboard_and_api_routes_are_registered(app_module):
    route_paths = {route.path for route in app_module.app.routes}
    assert "/claims" in route_paths
    assert "/api/claims/summary" in route_paths
    assert "/api/claims/cases" in route_paths
    assert "/api/claims/model" in route_paths
    assert "/api/claims/{claim_id}/analysis" in route_paths


def test_claims_dashboard_marks_data_as_synthetic_and_links_back_to_rag(app_module):
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "static" / "claims.html").read_text(encoding="utf-8")
    assert "tous les dossiers affichés sont fictifs" in html
    assert "Pending" in html
    assert "décision" in html.lower()
    home = (project_root / "static" / "index.html").read_text(encoding="utf-8")
    assert 'href="/claims"' in home



def test_rank_fusion_does_not_reward_lexical_rank_without_term_overlap(app_module):
    semantic_only = app_module._rank_fusion_score(
        primary_rank=1,
        lexical_rank=1,
        lexical_score=0.0,
        fusion_constant=60.0,
    )
    expected = 1.0 / 61.0
    assert semantic_only == pytest.approx(expected)


def test_rank_fusion_uses_lexical_rank_when_query_terms_match(app_module):
    fused = app_module._rank_fusion_score(
        primary_rank=2,
        lexical_rank=3,
        lexical_score=1.25,
        fusion_constant=60.0,
    )
    expected = 1.0 / 62.0 + 1.0 / 63.0
    assert fused == pytest.approx(expected)
    assert fused > 1.0 / 62.0
