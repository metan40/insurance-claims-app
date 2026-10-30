"""Génération typée et vérifiée de la réponse.

Le modèle ne renvoie pas un texte libre mais un objet JSON conforme au
schéma ``TypedAnswer`` (sortie structurée d'Ollama, paramètre ``format``) :

* ``answer_found`` / ``complete_answer_found`` : indicateurs de suffisance ;
* ``answer`` : la réponse rédigée ;
* ``evidence`` : les extraits recopiés des sources qui la justifient ;
* ``confidence`` et ``caveats`` : confiance déclarée et réserves.

L'application ne fait pas confiance à ces déclarations : elle **vérifie**
elle-même que chaque extrait existe bien dans la source citée. La décision de
livrer la réponse repose sur cette vérification, pas sur la confiance
déclarée par le modèle (un score que les petits modèles calibrent mal).

Statuts possibles :

* ``answered``    réponse trouvée, complète, appuyée par au moins un extrait vérifié ;
* ``partial``     réponse partielle, appuyée par au moins un extrait vérifié ;
* ``not_found``   le modèle déclare que les sources ne contiennent pas la réponse ;
* ``fallback``    après une révision, aucun extrait vérifiable : l'application
                  affiche les sources au lieu d'une réponse non justifiée.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from src.question_parsing import normalize_text

MIN_QUOTE_TOKENS = 3
QUOTE_MATCH_RATIO = 0.8
FALLBACK_EXCERPT_CHARACTERS = 700
FALLBACK_SOURCES = 3


class Evidence(BaseModel):
    source_id: str = Field(description="Identifiant de la source, par exemple S1")
    quote: str = Field(description="Phrase recopiée mot pour mot depuis cette source")


class TypedAnswer(BaseModel):
    answer_found: bool = Field(description="Les sources contiennent-elles la réponse ?")
    complete_answer_found: bool = Field(description="La réponse est-elle complète ?")
    answer: str = Field(description="Réponse en français, claire et concise")
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: str = Field(default="", description="Réserves ou limites, chaîne vide sinon")


ANSWER_SCHEMA = TypedAnswer.model_json_schema()


# ------------------------------------------------------------------
# Consignes
# ------------------------------------------------------------------

def build_context(results) -> str:
    """Assemble les sources numérotées transmises au LLM."""
    parts = []
    for position, result in enumerate(results, start=1):
        source_id = result.get("source_id", f"S{result.get('rank', position)}")
        parts.append(
            f"SOURCE [{source_id}]\n"
            f"TITRE : {result['title']}\n"
            f"CONTENU : {result['content']}"
        )
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Consigne principale : réponse typée fondée uniquement sur les sources."""
    return f"""
Tu es un assistant spécialisé dans l'assurance habitation.

Réponds à la question uniquement à partir des documents fournis, sous la forme
d'un objet JSON respectant le schéma demandé.

Règles :
- answer_found : false si aucun document ne contient la réponse. Dans ce cas,
  answer explique brièvement que les documents ne permettent pas de répondre,
  et evidence est une liste vide. N'invente jamais de réponse.
- complete_answer_found : false si les documents ne répondent qu'en partie.
- answer : réponse en français, claire et concise, sans information absente
  des documents.
- evidence : pour chaque information importante, recopie MOT POUR MOT une
  phrase courte du document qui la justifie, avec l'identifiant exact de sa
  source (S1, S2...). N'invente jamais d'identifiant.
- confidence : nombre entre 0 et 1.
- caveats : réserves utiles (conditions, exclusions), ou chaîne vide.

QUESTION :
{question}

DOCUMENTS FOURNIS :
{context}
"""


def build_revision_prompt(question: str, context: str, previous: str, available_ids) -> str:
    """Seconde tentative quand aucun extrait de la première n'a pu être vérifié."""
    allowed = ", ".join(available_ids) or "(aucune source disponible)"
    return f"""
Ta réponse précédente n'a pas pu être vérifiée : ses extraits ne figurent pas
mot pour mot dans les sources citées. Corrige-la en respectant le même schéma
JSON. La question et les documents sont reproduits car chaque appel est
indépendant.

QUESTION ORIGINALE :
{question}

DOCUMENTS DE RÉFÉRENCE :
{context}

RÈGLES DE RÉVISION :
- Chaque extrait de evidence doit être COPIÉ EXACTEMENT depuis le CONTENU de la
  source indiquée, sans reformulation.
- Les seuls identifiants autorisés sont : {allowed}.
- Si les documents ne permettent pas de répondre, mets answer_found à false.

RÉPONSE PRÉCÉDENTE À CORRIGER :
{previous}
"""


# ------------------------------------------------------------------
# Lecture et vérification
# ------------------------------------------------------------------

def parse_typed_answer(raw_text: str) -> TypedAnswer | None:
    """Convertit la sortie du modèle en TypedAnswer ; None si elle est invalide."""
    try:
        return TypedAnswer.model_validate(json.loads(raw_text))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def quote_is_in_source(quote: str, source_text: str) -> bool:
    """Vrai si l'extrait figure dans la source, à la casse, aux accents et à la
    ponctuation près. De petites variations sont tolérées : au moins 80 % des
    mots de l'extrait doivent se retrouver dans l'ordre dans la source, dont un
    passage d'un seul tenant couvrant au moins 40 % de l'extrait."""
    quote_tokens = normalize_text(quote).split()
    source_tokens = normalize_text(source_text).split()
    if len(quote_tokens) < MIN_QUOTE_TOKENS or not source_tokens:
        return False
    if f" {' '.join(quote_tokens)} " in f" {' '.join(source_tokens)} ":
        return True
    matcher = SequenceMatcher(None, quote_tokens, source_tokens, autojunk=False)
    blocks = matcher.get_matching_blocks()
    matched = sum(block.size for block in blocks)
    longest = max((block.size for block in blocks), default=0)
    return (
        matched / len(quote_tokens) >= QUOTE_MATCH_RATIO
        and longest >= max(MIN_QUOTE_TOKENS, 0.4 * len(quote_tokens))
    )


def verify_answer(typed: TypedAnswer, results) -> list[dict]:
    """Vérifie chaque extrait : source existante et texte présent dans la source."""
    contents = {str(r.get("source_id")): str(r.get("content", "")) for r in results}
    checked = []
    for item in typed.evidence:
        source_known = item.source_id in contents
        checked.append({
            "source_id": item.source_id,
            "quote": item.quote,
            "source_known": source_known,
            "verified": source_known and quote_is_in_source(item.quote, contents[item.source_id]),
        })
    return checked


def decide_status(typed: TypedAnswer | None, checked: list[dict]) -> str | None:
    """Statut livrable, ou None si une révision est nécessaire."""
    if typed is None:
        return None
    if not typed.answer_found:
        return "not_found"
    if not any(item["verified"] for item in checked):
        return None
    return "answered" if typed.complete_answer_found else "partial"


def build_fallback_text(results) -> str:
    """Repli : extraits des premières sources, sans synthèse non justifiée."""
    excerpts = []
    for result in results[:FALLBACK_SOURCES]:
        source_id = str(result.get("source_id", "")).strip()
        title = str(result.get("title", "Document récupéré")).strip()
        content = str(result.get("content", "")).strip()
        if not source_id or not content:
            continue
        if len(content) > FALLBACK_EXCERPT_CHARACTERS:
            content = content[:FALLBACK_EXCERPT_CHARACTERS].rsplit(" ", 1)[0] + "…"
        excerpts.append(f"- **{title}** [{source_id}] : {content}")

    message = (
        "Je ne peux pas formuler une synthèse suffisamment "
        "traçable : le modèle n’a pas fourni d’extraits "
        "vérifiables dans les sources. Pour éviter d’affirmer un élément sans "
        "preuve, voici les extraits des documents récupérés, "
        "avec leurs identifiants de source. Leur pertinence "
        "doit être vérifiée au regard de votre question."
    )
    if excerpts:
        return message + "\n\n" + "\n\n".join(excerpts)
    return message + (
        "\n\nAucun extrait exploitable avec identifiant "
        "de source n’est disponible pour cette requête."
    )


def display_text(typed: TypedAnswer, checked: list[dict]) -> str:
    """Texte affiché : la réponse suivie des identifiants des extraits vérifiés."""
    verified_ids = sorted(
        {item["source_id"] for item in checked if item["verified"]},
        key=lambda value: int(value[1:]) if value[1:].isdigit() else 0,
    )
    suffix = "".join(f" [{source_id}]" for source_id in verified_ids)
    return typed.answer.strip() + suffix


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def generate_typed_answer(
    question: str,
    results,
    call_model: Callable[[str, str], str],
    log: Callable[[str], None] = lambda message: None,
) -> dict:
    """Une tentative, une révision si nécessaire, puis repli.

    ``call_model(prompt, label)`` renvoie le texte brut du modèle ; les erreurs
    réseau qu'il lève sont propagées à l'appelant.
    """
    context = build_context(results)
    available_ids = [str(r["source_id"]) for r in results if r.get("source_id")]
    prompts = [("tentative 1", build_prompt(question, context))]
    attempts = 0
    raw = ""

    while prompts:
        label, prompt = prompts.pop(0)
        attempts += 1
        raw = call_model(prompt, label)
        typed = parse_typed_answer(raw)
        checked = verify_answer(typed, results) if typed else []
        status = decide_status(typed, checked)
        log(
            f"{label} — audit : json_valide={typed is not None}, "
            f"extraits_verifies={sum(c['verified'] for c in checked)}/{len(checked)}, "
            f"statut={status}"
        )
        if status is not None:
            text = (
                display_text(typed, checked)
                if status != "not_found"
                else typed.answer.strip()
                or "Les documents consultés ne permettent pas de répondre à cette question."
            )
            return {
                "status": status,
                "answer": text,
                "typed_answer": {**typed.model_dump(exclude={"evidence"}), "evidence": checked},
                "attempts": attempts,
            }
        if attempts == 1:
            prompts.append((
                "tentative 2 / révision",
                build_revision_prompt(question, context, raw, available_ids),
            ))

    return {
        "status": "fallback",
        "answer": build_fallback_text(results),
        "typed_answer": None,
        "attempts": attempts,
    }
