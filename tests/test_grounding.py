"""Contrôle d'ancrage : une phrase de la réponse qui ne vient pas des sources
vérifiées n'est jamais affichée (cas réel du vélo volé dans la cave, 12/11/2026)."""
import json

from src import generation
from src.grounding import check_grounding

QUESTION = "On m'a volé mon vélo dans ma cave, suis-je couvert ?"
S2_CONTENT = ("Vous bénéficiez de la garantie vol et vandalisme intérieur dans le cadre "
              "des formules n°2 et 3 (voir conditions), mais pas n°1")
SOURCES = [
    {"rank": 1, "source_id": "S1", "title": "Vol dans une dépendance",
     "content": "Les objets à risque de vol (voir définition) situés dans les dépendances "
                "et locaux attenants sans communication directe avec l’habitation "
                "principale ne sont pas assurés"},
    {"rank": 2, "source_id": "S2", "title": "Est-ce qu'en cas de vol je suis couvert ou pas ?",
     "content": S2_CONTENT},
    {"rank": 3, "source_id": "S3", "title": "Je déménage par moi même. Suis-je couvert ?",
     "content": "si vous transportez vous-même votre mobilier dans votre véhicule, celui-ci "
                "est couvert en cas de dommages matériels consécutifs à un accident."},
]
S2_VERIFIED = [{"source_id": "S2", "quote": S2_CONTENT, "verified": True}]


def output(answer, quote=S2_CONTENT, source_id="S2", complete=True):
    return json.dumps({
        "evidence": [{"source_id": source_id, "quote": quote}], "answer": answer,
        "caveats": "", "answer_found": True, "complete_answer_found": complete,
        "confidence": 0.5,
    }, ensure_ascii=False)


def scripted(outputs):
    iterator = iter(outputs)
    return lambda prompt, label: next(iterator)


def test_answer_mixing_an_uncited_source_is_rejected():
    report = check_grounding(
        "Je suis couvert, car je transportais mon vélo dans un véhicule assuré.",
        QUESTION, S2_VERIFIED, SOURCES)
    assert report["supported"] == []
    assert {"transportais", "vehicule"} <= set(report["unsupported"][0]["novel_terms"])


def test_answer_drawn_from_the_verified_source_is_kept():
    answer = ("Le vol de votre vélo est couvert par la garantie vol et vandalisme intérieur "
              "si vous avez souscrit la formule 2 ou 3, mais pas la formule 1.")
    report = check_grounding(answer, QUESTION, S2_VERIFIED, SOURCES)
    assert report["unsupported"] == []
    assert report["supported"] == [answer]


def test_only_unsupported_sentences_are_masked():
    report = check_grounding(
        "Le vol est couvert dans les formules n°2 et 3. "
        "Votre véhicule transporte le mobilier en cas d'accident.",
        QUESTION, S2_VERIFIED, SOURCES)
    assert report["supported"] == ["Le vol est couvert dans les formules n°2 et 3."]
    assert len(report["unsupported"]) == 1


def test_invented_synthesis_is_never_displayed():
    invented = "Je suis couvert, car je transportais mon vélo dans un véhicule assuré."
    result = generation.generate_typed_answer(
        QUESTION, SOURCES, scripted([output(invented, quote=SOURCES[0]["content"], source_id="S1"),
                                     output(invented)]))
    assert result["status"] == "fallback"
    assert "transportais" not in result["answer"]
    assert "garantie vol et vandalisme" in result["answer"]
    assert result["typed_answer"]["answer"] == ""
    assert result["typed_answer"]["model_unverified_text"] == invented


def test_partially_grounded_answer_is_downgraded_and_masked():
    answer = ("Le vol est couvert dans les formules n°2 et 3. "
              "Votre véhicule transporte le mobilier en cas d'accident.")
    result = generation.generate_typed_answer(
        QUESTION, SOURCES, scripted([output(answer, quote=SOURCES[0]["content"], source_id="S1"),
                                     output(answer)]))
    assert result["status"] == "partial"
    assert "formules n°2 et 3" in result["answer"]
    assert "véhicule" not in result["answer"]
    assert "n’a (ont) pas été affichée(s)" in result["answer"]


def test_grounded_answer_is_delivered_unchanged():
    answer = "Le vol est couvert dans les formules n°2 et 3, mais pas dans la formule 1."
    result = generation.generate_typed_answer(
        QUESTION, SOURCES[1:2], scripted([output(answer)]))
    assert result["status"] == "answered"
    assert result["answer"].startswith(answer)
    assert result["typed_answer"]["unsupported_sentences"] == []
