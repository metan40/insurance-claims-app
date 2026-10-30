"""Tests de l'aide à la décision : score expliqué, règles métier, clauses, orientation."""
import pandas as pd
import pytest

from src import claim_decision as cd
from src import claims_analytics as ca

BASE_CLAIM = {
    "claim_id": "X1", "type_sinistre": "Dégâts des eaux", "garantie": "Oui",
    "dossier_complet": "Oui", "montant_demande": 1500.0, "nb_sinistres_precedents": 0,
    "statut": "Accepted", "age_sinistre_jours": 10,
}


def test_every_mapped_clause_exists_in_corpus():
    titles = set(pd.read_csv("Data/insurance-fr.csv")["title"])
    mapped = [t for clauses in cd.CLAUSES_BY_CLAIM_TYPE.values() for _, t in clauses]
    mapped += [t for _, t in cd.COMMON_CLAUSES]
    assert set(mapped) <= titles


def test_every_claim_type_of_the_portfolio_is_mapped():
    types = set(pd.read_csv("Data/synthetic_claims.csv")["type_sinistre"])
    assert types <= set(cd.CLAUSES_BY_CLAIM_TYPE)


def test_types_without_dedicated_fiche_are_flagged_as_corpus_gaps():
    for claim_type, clauses in cd.CLAUSES_BY_CLAIM_TYPE.items():
        if not clauses:
            assert claim_type in cd.CORPUS_GAPS


def test_clean_claim_has_no_check():
    assert cd.checks_for_claim(BASE_CLAIM, amount_threshold=6000) == []


@pytest.mark.parametrize("change, level, text", [
    ({"dossier_complet": "Non"}, "bloquant", "pièces manquantes"),
    ({"garantie": "À vérifier"}, "bloquant", "Garantie à vérifier"),
    ({"montant_demande": 9000.0}, "attention", "Montant élevé"),
    ({"nb_sinistres_precedents": 4}, "attention", "sinistres précédents"),
    ({"statut": "Pending", "age_sinistre_jours": 45}, "attention", "à relancer"),
    ({"statut": "Pending", "age_sinistre_jours": 120}, "attention", "escalader"),
])
def test_business_rules(change, level, text):
    checks = cd.checks_for_claim({**BASE_CLAIM, **change}, amount_threshold=6000)
    assert len(checks) == 1
    assert checks[0]["level"] == level and text in checks[0]["message"]


def test_recent_pending_claim_is_not_flagged():
    claim = {**BASE_CLAIM, "statut": "Pending", "age_sinistre_jours": 5}
    assert cd.checks_for_claim(claim, amount_threshold=6000) == []


@pytest.mark.parametrize("probability, checks, claim_type, expected", [
    (0.9, [{"level": "bloquant", "message": ""}], "Vol", "completer"),
    (0.9, [], "Incendie", "verifier_contrat"),
    (0.9, [], "Vol", "favorable"),
    (0.2, [], "Vol", "defavorable"),
    (0.55, [], "Vol", "approfondir"),
    (0.9, [{"level": "attention", "message": ""}], "Vol", "approfondir"),
])
def test_orientation(probability, checks, claim_type, expected):
    assert cd.orientation(probability, checks, claim_type)["code"] == expected


def test_score_explanation_names_readable_factors():
    factors = ca.explain_claim("C00001")
    assert factors and all(f["contribution"] != 0 for f in factors)
    assert any(f["factor"].startswith("Garantie = ") for f in factors)
    positives = [f for f in factors if f["contribution"] > 0]
    assert positives == sorted(positives, key=lambda f: -f["contribution"])
    assert ca.explain_claim("INCONNU") == []


def test_high_amount_threshold_is_the_90th_percentile():
    amounts = pd.read_csv("Data/synthetic_claims.csv")["montant_demande"]
    assert ca.high_amount_threshold() == pytest.approx(amounts.quantile(0.9))


def test_decision_support_endpoint(app_module):
    result = app_module.claim_decision_support("C00020")
    assert result["claim_id"] == "C00020"
    assert result["human_review_required"] is True
    assert result["orientation"]["code"] == "completer"
    assert any(c["level"] == "bloquant" for c in result["checks"])
    headings = [c["heading"] for c in result["contract_clauses"]]
    assert "Couverture" in headings and "Franchise" in headings
    assert all(c["excerpt"] for c in result["contract_clauses"])


def test_decision_support_unknown_claim_returns_404(app_module):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        app_module.claim_decision_support("INCONNU")
    assert error.value.status_code == 404
