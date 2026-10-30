"""Tests unitaires et de non-régression de la recherche documentaire."""
import numpy as np
import pytest

from src.retrieval import (
    HybridRetriever,
    lexical_relevance_score,
    normalize_tokens,
    query_terms,
    rank_fusion_score,
)


# ---------- Normalisation et score lexical ----------

def test_normalize_tokens_removes_accents_and_punctuation():
    assert normalize_tokens("Dégât des EAUX, franchise !") == ["degat", "des", "eaux", "franchise"]


def test_query_terms_drop_stopwords_short_and_generic_terms():
    terms = query_terms("Comment fonctionne la franchise de mon assurance habitation ?")
    assert terms == ["franchise"]


def test_query_terms_are_unique_and_keep_order():
    assert query_terms("vol vol bijoux vol") == ["vol", "bijoux"]


def test_lexical_score_is_zero_without_meaningful_terms():
    assert lexical_relevance_score("le la les", "Titre", "Contenu") == 0.0


def test_lexical_score_rewards_exact_title_terms():
    exact = lexical_relevance_score(
        "franchise absolue assurance habitation",
        "Comment fonctionne la franchise absolue ?",
        "L'assureur déduit la franchise de l'indemnisation.",
    )
    unrelated = lexical_relevance_score(
        "franchise absolue assurance habitation",
        "Attestation d'assurance habitation",
        "Le locataire remet un justificatif au propriétaire.",
    )
    assert exact > unrelated


def test_lexical_score_title_match_weighs_three_times_content_match():
    in_title = lexical_relevance_score("serrure", "Problème de serrure", "texte")
    in_content = lexical_relevance_score("serrure", "Titre", "une serrure cassée")
    assert in_title == pytest.approx(3.0)
    assert in_content == pytest.approx(1.0)


# ---------- Fusion des rangs (RRF) ----------

def test_rank_fusion_does_not_reward_lexical_rank_without_term_overlap():
    assert rank_fusion_score(1, 1, 0.0, 60.0) == pytest.approx(1.0 / 61.0)


def test_rank_fusion_uses_lexical_rank_when_query_terms_match():
    fused = rank_fusion_score(2, 3, 1.25, 60.0)
    assert fused == pytest.approx(1.0 / 62.0 + 1.0 / 63.0)


def test_rank_fusion_decreases_when_rank_gets_worse():
    assert rank_fusion_score(1, 1, 1.0) > rank_fusion_score(10, 1, 1.0) > rank_fusion_score(10, 10, 1.0)


# ---------- Comportement du retriever ----------

@pytest.fixture(scope="module")
def tiny_retriever():
    titles = ["Franchise habitation", "Vol de bijoux", "Dégât des eaux", "Serrure cassée"]
    documents = [
        "La franchise reste à la charge de l'assuré après indemnisation.",
        "Le vol de bijoux est couvert selon les plafonds du contrat.",
        "Une fuite d'eau provoque un dégât des eaux dans le logement.",
        "Le serrurier intervient en cas de serrure cassée.",
    ]
    return HybridRetriever(titles, documents)


def test_retriever_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        HybridRetriever(["a"], ["x", "y"])
    with pytest.raises(ValueError):
        HybridRetriever(["a"], ["   "])


def test_retriever_puts_obvious_document_first(tiny_retriever):
    assert tiny_retriever.rank("vol de bijoux")[0].index == 1
    assert tiny_retriever.rank("fuite d'eau")[0].index == 2


def test_retriever_respects_candidate_limit_and_unique_documents(tiny_retriever):
    ranked = tiny_retriever.rank("franchise", candidate_limit=3)
    assert len(ranked) == 3
    assert len({item.index for item in ranked}) == 3
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_retriever_scores_are_finite_and_cosine_bounded(tiny_retriever):
    for item in tiny_retriever.rank("serrure franchise"):
        assert 0.0 <= item.cosine_similarity <= 1.0 + 1e-9
        assert np.isfinite(item.fusion_score)


def test_retriever_is_deterministic(tiny_retriever):
    first = [item.index for item in tiny_retriever.rank("dégât des eaux franchise")]
    second = [item.index for item in tiny_retriever.rank("dégât des eaux franchise")]
    assert first == second


def test_retriever_handles_question_without_known_terms(tiny_retriever):
    ranked = tiny_retriever.rank("zzzz qwerty")
    assert len(ranked) == 4
    assert all(item.lexical_score == 0.0 for item in ranked)


# ---------- Non-régression sur le corpus réel ----------

def test_candidate_pool_keeps_exact_insurance_concept(app_module):
    question = "Comment fonctionne la franchise absolue d'une assurance habitation ?"
    candidates = app_module.retriever.select_candidates(
        question,
        np.ones(len(app_module.documents)),
        app_module.retriever.lexical_scores(question),
        candidate_limit=20,
    )
    target = next(i for i, title in enumerate(app_module.titles) if "franchise" in title.lower())
    assert len(candidates) == 20
    assert target in candidates[:5]


def test_final_ranking_keeps_franchise_topic_near_the_top(app_module):
    ranked = app_module.retriever.rank(
        "Comment fonctionne la franchise absolue d'une assurance habitation ?"
    )
    top_titles = [app_module.titles[item.index].lower() for item in ranked[:3]]
    assert any("franchise" in title for title in top_titles)


def test_retrieval_quality_does_not_regress():
    """Seuils planchers mesurés en V19 (voir evaluation/evaluate_pipeline.py)."""
    from evaluation.evaluate_pipeline import main

    summary = main(write_results=False, verbose=False)
    production = {name: values["hybride_production"] for name, values in summary.items()}
    assert production["titres"]["recall@5"] >= 0.85
    assert production["reformulations"]["recall@5"] >= 0.60
    assert production["tenu_a_part"]["recall@5"] >= 0.50
    # Le pipeline doit rester meilleur que la similarité TF-IDF seule. Les
    # synonymes doivent améliorer les deux jeux de reformulations ; sur les
    # titres, une perte d'une question au plus (0,5 point) est tolérée.
    for name, values in summary.items():
        assert values["hybride_production"]["recall@5"] > values["cosinus_tfidf"]["recall@5"]
    for name in ("reformulations", "tenu_a_part"):
        assert (
            summary[name]["hybride_production"]["recall@5"]
            > summary[name]["hybride_sans_synonymes"]["recall@5"]
        )
    assert (
        summary["titres"]["hybride_sans_synonymes"]["recall@5"]
        - summary["titres"]["hybride_production"]["recall@5"]
    ) <= 0.006  # une question sur 201, valeurs arrondies à 4 décimales


# ---------- Analyse de la question (synonymes métier) ----------

def test_parser_maps_everyday_words_to_business_terms():
    from src.question_parsing import parse_question

    parsed = parse_question("La nounou qui garde mes enfants est-elle couverte ?")
    assert "baby sitter" in parsed.expansions
    assert ["nounou", "baby sitter"] in parsed.concepts


def test_parser_merges_multiword_variant_into_one_concept():
    from src.question_parsing import parse_question

    parsed = parse_question("Mon bac à poissons est fissuré")
    concept = next(c for c in parsed.concepts if "aquarium" in c)
    assert "bac" in concept and "poissons" in concept
    assert not any(c == ["poissons"] for c in parsed.concepts)


def test_parser_does_not_expand_terms_already_used_by_the_corpus():
    from src.question_parsing import parse_question

    assert parse_question("Comment fonctionne la franchise ?").expansions == []


def test_parser_can_be_disabled():
    from src.question_parsing import parse_question

    assert parse_question("La nounou est-elle couverte ?", use_synonyms=False).expansions == []


def test_synonym_concept_counts_once_in_lexical_score():
    from src.question_parsing import parse_question

    parsed = parse_question("nounou")
    with_variant = lexical_relevance_score("nounou", "Garde par une baby-sitter", "", parsed)
    assert with_variant == pytest.approx(3.0)


def test_synonyms_bring_the_right_document_for_everyday_wording(app_module):
    question = "Mes capteurs photovoltaïques sont-ils protégés ?"
    target = next(i for i, t in enumerate(app_module.titles) if "panneaux solaires" in t.lower())
    with_synonyms = [item.index for item in app_module.retriever.rank(question)[:5]]
    assert target in with_synonyms
