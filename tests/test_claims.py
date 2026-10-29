"""Tests du module d'aide à l'analyse des sinistres (données fictives)."""
import pandas as pd

from src import claims_analytics


def test_dataset_is_fully_synthetic_and_well_formed():
    frame = pd.read_csv(claims_analytics.CLAIMS_FILE)
    assert len(frame) == 120
    assert frame["claim_id"].is_unique
    assert set(frame["jeu_de_donnees"]) == {"FICTIF_DEMONSTRATION"}
    assert set(frame["statut"]) <= {"Accepted", "Rejected", "Pending"}
    for column in claims_analytics.FEATURE_COLUMNS:
        assert column in frame.columns


def test_summary_excludes_pending_from_acceptance_rate():
    summary = claims_analytics.get_claims_summary()
    assert summary["total_cases"] == 120
    assert summary["accepted_cases"] + summary["rejected_cases"] == summary["closed_cases"]
    assert summary["closed_cases"] + summary["pending_cases"] == summary["total_cases"]
    assert summary["acceptance_rate"] == round(
        summary["accepted_cases"] / summary["closed_cases"], 4
    )
    assert summary["pending_excluded_from_acceptance_rate"] is True
    assert "fictif" in summary["warning"].lower()


def test_cases_are_json_safe_and_bounded():
    cases = claims_analytics.get_claims_cases(12)
    assert len(cases) == 12
    assert all(case["age_sinistre_jours"] >= 0 for case in cases)
    assert all(
        case["delai_traitement_jours"] is None
        or isinstance(case["delai_traitement_jours"], (int, float))
        for case in cases
    )
    assert len(claims_analytics.get_claims_cases(10_000)) == 120


def test_model_is_trained_on_closed_cases_only_and_evaluated_on_held_out_data():
    info = claims_analytics.get_claims_model_info()
    assert info["pending_used_for_training"] is False
    assert info["closed_cases_count"] == info["training_cases_count"] + info["test_cases_count"]
    assert info["test_cases_count"] > 0
    for metric in ("accuracy", "balanced_accuracy", "f1_score", "roc_auc"):
        assert 0.0 <= info[metric] <= 1.0
    assert "validation humaine" in info["decision_policy"].lower()
    assert "synthétiques" in info["warning"].lower()


def test_model_training_is_reproducible():
    _, first, _ = claims_analytics._build_model()
    _, second, _ = claims_analytics._build_model()
    assert first == second


def test_claim_analysis_is_explanatory_not_a_decision():
    result = claims_analytics.get_claim_analysis("C00001")
    assert 0.0 <= result["estimated_acceptance_probability"] <= 1.0
    assert result["human_review_required"] is True
    assert "ne constitue ni une décision" in result["warning"]
    assert claims_analytics.get_claim_analysis("C99999") is None
