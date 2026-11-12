"""Renvois « (voir définition) » : la définition est jointe à la source.

Cas réel (12/11/2026) : pour un vélo volé dans la cave, la source retenue dit
« Les objets à risque de vol (voir définition) situés dans les dépendances…
ne sont pas assurés ». La réponse dépend de la définition (un vélo est-il un
objet à risque de vol ?), mais la fiche « Que sont les objets à risque de
vol ? » n'était pas parmi les sources lues : le modèle ne pouvait pas conclure.

Quand une source contient « <terme> (voir définition) » et que le corpus
possède une fiche dont le titre définit ce terme (« Que sont les… »,
« Qu'est-ce qu'un… »), le texte de cette fiche est ajouté à la fin de la
source, sous une mention explicite. Le classement n'est pas modifié : seules
les informations transmises au modèle et affichées sont complétées.
"""
from __future__ import annotations

import re
from typing import Sequence

from src.question_parsing import normalize_text

REFERENCE = re.compile(r"([^.;:()\n]{3,80}?)\s*\(\s*(?:voir|cf\.?)\s+d[ée]finitions?\s*\)", re.IGNORECASE)
DEFINITION_TITLE = re.compile(
    r"^\s*(que sont|qu.est.ce qu.(une|un|on|il|elle)?|quelle est la definition d.?)\s*",
)
LEADING_ARTICLES = ("les ", "le ", "la ", "l ", "un ", "une ", "des ", "vos ", "votre ")
DEFINITION_LABEL = "Définition jointe automatiquement (fiche « {title} »)"


def _term_key(text: str) -> str:
    """Terme normalisé sans article initial ni pluriel final simple."""
    key = normalize_text(text)
    for article in LEADING_ARTICLES:
        if key.startswith(article):
            key = key[len(article):]
    return " ".join(word[:-1] if len(word) > 3 and word.endswith("s") else word for word in key.split())


def build_definition_index(titles: Sequence[str]) -> dict[str, int]:
    """Terme défini -> indice de la fiche de définition."""
    index = {}
    for position, title in enumerate(titles):
        normalized = normalize_text(title)
        match = DEFINITION_TITLE.match(normalized)
        if match:
            term = _term_key(normalized[match.end():])
            if term:
                index.setdefault(term, position)
    return index


def referenced_definitions(content: str, index: dict[str, int]) -> list[int]:
    """Fiches de définition auxquelles renvoie un texte (ordre d'apparition)."""
    found = []
    for match in REFERENCE.finditer(content):
        words = _term_key(match.group(1)).split()
        # On cherche le plus long suffixe du groupe nominal qui est un terme défini.
        for start in range(len(words)):
            position = index.get(" ".join(words[start:]))
            if position is not None:
                if position not in found:
                    found.append(position)
                break
    return found


def attach_definitions(results: list[dict], titles: Sequence[str], documents: Sequence[str],
                       index: dict[str, int], characters: int = 900) -> list[dict]:
    """Ajoute à chaque source le texte des définitions auxquelles elle renvoie."""
    expanded = []
    for result in results:
        content = str(result.get("content", ""))
        additions = []
        for position in referenced_definitions(content, index):
            if titles[position] == result.get("title"):
                continue  # la source est elle-même la fiche de définition
            text = str(documents[position])
            if len(text) > characters:
                text = text[:characters].rsplit(" ", 1)[0] + "…"
            additions.append(f"{DEFINITION_LABEL.format(title=titles[position].strip())} : {text}")
        if additions:
            result = {**result, "content": content + "\n\n" + "\n\n".join(additions),
                      "attached_definitions": [titles[p] for p in referenced_definitions(content, index)]}
        expanded.append(result)
    return expanded
