"""Contrôle d'ancrage : chaque phrase de la réponse doit venir des sources.

La vérification des extraits (``generation.verify_answer``) prouve que les
phrases recopiées par le modèle existent dans la source citée. Elle ne prouve
pas que la *réponse rédigée* en découle. Cas réel (Space, 12/11/2026) :

    Question : vélo volé dans la cave.
    Réponse  : « Je suis couvert, car je transportais mon vélo dans un
               véhicule assuré. [S2] »

L'extrait de S2 (« garantie vol et vandalisme intérieur… ») était vérifié,
mais « transportais » et « véhicule » venaient de S3 (déménagement), une source
que le modèle n'a pas citée. Le modèle avait mélangé deux sources.

Règle appliquée ici : un mot significatif de la réponse doit apparaître dans
la question, dans le contenu d'une source dont un extrait a été vérifié, ou
dans un petit vocabulaire de rédaction (couvert, garantie, formule…). Une
phrase qui contient au moins ``MAX_NOVEL_TERMS`` mots absents de ces trois
réservoirs est jugée non étayée et n'est pas affichée.

La comparaison se fait sur les ``PREFIX_LENGTH`` premières lettres du mot
normalisé (sans accents) : « volé », « vol » et « vols » se rejoignent, de même
que « garantie » et « garanties », sans dictionnaire de conjugaison.

Limite assumée : ce contrôle est lexical. Il détecte une information apportée
d'ailleurs (un mot nouveau), pas une négation inversée (« n'est pas couvert »
au lieu de « est couvert ») construite avec les mêmes mots.
"""
from __future__ import annotations

import re

from src.question_parsing import FRENCH_QUERY_STOPWORDS, normalize_text

PREFIX_LENGTH = 5
MIN_TERM_LENGTH = 4
MAX_NOVEL_TERMS = 2

# Mots de rédaction que le modèle emploie légitimement sans les recopier
# (normalisés, sans accents). Seuls leurs préfixes comptent.
DRAFTING_TERMS = (
    "couvert", "couverture", "couvre", "assure", "assureur", "assurance",
    "garantie", "contrat", "formule", "condition", "conditions", "souscrit",
    "sinistre", "indemnisation", "indemnise", "rembourse", "charge",
    "exclusion", "exclu", "depend", "selon", "uniquement", "seulement",
    "egalement", "cependant", "toutefois", "ainsi", "alors", "votre", "vous",
    "etes", "pouvez", "devez", "sont", "sera", "serait", "peuvent", "pourrait",
    "avez", "avoir", "etre", "cette", "celle", "celui", "ceux", "dont", "comme",
    "bien", "biens", "aussi", "donc", "mais", "lorsque", "quand", "sous",
    "non", "oui", "tout", "tous", "toute", "toutes", "votre", "notre", "leur",
    "cas", "prise", "pris", "precise", "indique", "mentionne", "document",
    "source", "sources", "reponse", "question", "information",
)

STOPWORDS = FRENCH_QUERY_STOPWORDS | {
    "je", "j", "tu", "il", "ils", "elles", "nous", "on", "me", "te", "lui",
    "mes", "tes", "nos", "ma", "ta", "ne", "ni", "si", "car", "y", "l", "d",
    "n", "s", "c", "m", "t", "qu", "suis", "es", "sont", "ete", "etait", "avait",
    "plus", "tres", "sans", "entre", "chez", "vers", "apres", "avant",
}

_DRAFTING_PREFIXES = frozenset(term[:PREFIX_LENGTH] for term in DRAFTING_TERMS)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[S\d+\]")


def significant_prefixes(text: str) -> list[str]:
    """Préfixes des mots significatifs d'un texte (ordre conservé)."""
    return [
        token[:PREFIX_LENGTH]
        for token in normalize_text(text).split()
        if len(token) >= MIN_TERM_LENGTH and token not in STOPWORDS and not token.isdigit()
    ]


def split_sentences(text: str) -> list[str]:
    """Découpe une réponse en phrases (les citations [S1] restent attachées)."""
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]


def novel_terms(sentence: str, allowed: set[str]) -> list[str]:
    """Mots de la phrase absents des réservoirs autorisés (sous forme de préfixes)."""
    words = [
        token for token in normalize_text(_CITATION.sub(" ", sentence)).split()
        if len(token) >= MIN_TERM_LENGTH and token not in STOPWORDS and not token.isdigit()
    ]
    return [
        word for word in words
        if word[:PREFIX_LENGTH] not in allowed and word[:PREFIX_LENGTH] not in _DRAFTING_PREFIXES
    ]


def check_grounding(answer: str, question: str, checked: list[dict], sources) -> dict:
    """Sépare les phrases étayées des phrases non étayées.

    ``checked`` est la liste produite par ``verify_answer`` ; seules les
    sources dont au moins un extrait a été vérifié servent de référence.
    """
    verified_ids = {item["source_id"] for item in checked if item.get("verified")}
    reference = [question]
    reference += [item["quote"] for item in checked if item.get("verified")]
    reference += [
        f"{source.get('title', '')} {source.get('content', '')}"
        for source in sources
        if str(source.get("source_id")) in verified_ids
    ]
    allowed = set(significant_prefixes(" ".join(reference)))

    supported, unsupported = [], []
    for sentence in split_sentences(answer):
        novel = novel_terms(sentence, allowed)
        if len(novel) >= MAX_NOVEL_TERMS:
            unsupported.append({"sentence": sentence, "novel_terms": novel})
        else:
            supported.append(sentence)
    return {"supported": supported, "unsupported": unsupported}
