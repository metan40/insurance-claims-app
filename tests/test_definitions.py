"""Renvois « (voir définition) » : la fiche de définition est jointe à la source."""
from src.definitions import attach_definitions, build_definition_index, referenced_definitions

TITLES = [
    "On a volé des objets à risque de vol dans une dépendance, suis-je couvert ?",
    "Que sont les objets à risque de vol ?",
    "Qu’est-ce qu’une location meublée ?",
    "Est-ce qu'en cas de vol je suis couvert ou pas ?",
]
DOCUMENTS = [
    "Les objets à risque de vol (voir définition) situés dans les dépendances ne sont pas assurés",
    "Sont considérés comme objets à risque de vol : les bijoux et objets en or, argent, platine.",
    "Une location meublée est un logement loué avec les meubles.",
    "Vous bénéficiez de la garantie vol dans le cadre des formules n°2 et 3, mais pas n°1",
]


def test_definition_titles_are_indexed():
    index = build_definition_index(TITLES)
    assert index["objet a risque de vol"] == 1
    assert index["location meublee"] == 2


def test_reference_finds_the_definition_sheet():
    index = build_definition_index(TITLES)
    assert referenced_definitions(DOCUMENTS[0], index) == [1]
    assert referenced_definitions(DOCUMENTS[3], index) == []


def test_definition_is_appended_to_the_referring_source_only():
    index = build_definition_index(TITLES)
    results = [{"source_id": "S1", "title": TITLES[0], "content": DOCUMENTS[0]},
               {"source_id": "S2", "title": TITLES[3], "content": DOCUMENTS[3]}]
    expanded = attach_definitions(results, TITLES, DOCUMENTS, index)
    assert "bijoux et objets en or" in expanded[0]["content"]
    assert "Définition jointe automatiquement" in expanded[0]["content"]
    assert expanded[0]["attached_definitions"] == [TITLES[1]]
    assert expanded[1] == results[1]
    assert results[0]["content"] == DOCUMENTS[0]  # l'original n'est pas modifié


def test_real_corpus_links_the_theft_sheets_to_their_definition():
    import csv
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "Data" / "insurance-fr.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    titles = [row["title"] for row in rows]
    documents = [row["content"] for row in rows]
    index = build_definition_index(titles)
    referring = [i for i, doc in enumerate(documents) if "(voir définition)" in doc]
    assert referring
    for position in referring:
        found = referenced_definitions(documents[position], index)
        assert found and titles[found[0]].startswith("Que sont les objets à risque de vol")
