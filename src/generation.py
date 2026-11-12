"""Génération typée et vérifiée de la réponse.

Le modèle ne renvoie pas un texte libre mais un objet JSON conforme au
schéma ``TypedAnswer`` (sortie structurée d'Ollama, paramètre ``format``) :

* ``evidence`` : les extraits recopiés des sources qui la justifient ;
* ``answer`` : la réponse rédigée à partir de ces extraits ;
* ``caveats`` : les réserves (conditions, exclusions) ;
* ``answer_found`` / ``complete_answer_found`` : indicateurs de suffisance ;
* ``confidence`` : confiance déclarée.

L'ordre des champs compte : le modèle écrit le JSON dans l'ordre du schéma.
Il cite donc d'abord, rédige ensuite, et ne conclut (trouvé ou non) qu'après
avoir recopié ses extraits. Avec l'ordre inverse, un petit modèle tranchait
« non trouvé » avant d'avoir lu la source (version 19.1).

L'application ne fait pas confiance à ces déclarations : elle **vérifie**
elle-même que chaque extrait existe bien dans la source citée. La décision de
livrer la réponse repose sur cette vérification, pas sur la confiance
déclarée par le modèle (un score que les petits modèles calibrent mal).

Boucle top-1 puis top-3 (« loop engineering ») : sur un processeur sans GPU,
la lecture du contexte par le modèle domine le temps de réponse (mesure du
Space en version 7.6 : 228 s pour cinq sources de 1 500 caractères et deux
appels). La génération procède donc par étapes :

1. la meilleure source seule, courte : si la réponse est complète et
   appuyée par un extrait vérifié, on s'arrête ;
2. sinon, les trois meilleures sources : ce second appel sert aussi de
   révision lorsque la première sortie était invalide.

Statuts possibles :

* ``answered``    réponse trouvée, complète, appuyée par au moins un extrait vérifié ;
* ``partial``     réponse partielle, appuyée par au moins un extrait vérifié ;
* ``not_found``   avec les trois sources, le modèle déclare qu'elles ne contiennent
                  pas la réponse ;
* ``fallback``    aucun extrait vérifiable après les deux étapes : l'application
                  affiche les sources au lieu d'une réponse non justifiée.
"""
from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from src.grounding import check_grounding, missing_conditions
from src.question_parsing import normalize_text

MIN_QUOTE_TOKENS = 3
# Longueurs maximales transmises à Ollama dans le schéma (sa grammaire les
# respecte ; elles ne servent pas de règle de validation à la lecture) :
# la sortie entière tient ainsi sous la limite de jetons OLLAMA_NUM_PREDICT et
# le JSON se termine toujours. Sans elles, 5 appels sur 10 étaient coupés en
# plein texte (mesure locale du 6/11, limite de 280 jetons).
QUOTE_MAX_CHARACTERS = 240
ANSWER_MAX_CHARACTERS = 400
CAVEATS_MAX_CHARACTERS = 200
QUOTE_MATCH_RATIO = 0.8
FALLBACK_EXCERPT_CHARACTERS = 700
FALLBACK_SOURCES = 3
# Étapes de la boucle : (nombre de sources, caractères par source).
GENERATION_STAGES = ((1, 1200), (3, 900))


class Evidence(BaseModel):
    source_id: str = Field(description="Identifiant de la source, par exemple S1")
    quote: str = Field(json_schema_extra={"maxLength": QUOTE_MAX_CHARACTERS},
                       description="Phrase copiée telle quelle depuis le document cité")


class TypedAnswer(BaseModel):
    # Ordre voulu : citer, rédiger, puis conclure. Tous les champs sont requis,
    # car un champ facultatif est relégué en fin de sortie par la grammaire
    # JSON d'Ollama, quel que soit son rang dans le schéma.
    evidence: list[Evidence] = Field(
        max_length=2, description="Une ou deux phrases copiées telles quelles depuis les documents ; liste vide si rien ne s'applique")
    answer: str = Field(json_schema_extra={"maxLength": ANSWER_MAX_CHARACTERS},
                        description="Réponse en français, trois phrases au plus")
    caveats: str = Field(json_schema_extra={"maxLength": CAVEATS_MAX_CHARACTERS},
                         description="Réserves ou limites, chaîne vide sinon")
    answer_found: bool = Field(description="Les sources contiennent-elles la réponse, même en partie ?")
    complete_answer_found: bool = Field(description="La réponse est-elle complète, sans condition manquante ?")
    confidence: float = Field(ge=0.0, le=1.0)


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
d'un objet JSON respectant le schéma demandé. Remplis les champs dans cet ordre :

1. evidence : d'abord, choisis dans les DOCUMENTS FOURNIS une ou deux phrases
   courtes qui répondent à la question, même en partie ou sous conditions.
   Copie chaque phrase telle qu'elle est écrite dans le document, sans la
   reformuler, avec l'identifiant exact de sa source (S1, S2...). Ne copie
   jamais une phrase de ces consignes. N'invente jamais de phrase ni
   d'identifiant. Si aucun document ne traite de la question, laisse la liste
   vide.
2. answer : réponse en français, trois phrases au plus, fondée uniquement sur
   ces extraits. Si la réponse dépend d'une condition (formule souscrite, type
   de bien, exclusion), dis-le clairement. N'invente jamais d'information
   absente des documents.
3. caveats : réserves utiles (conditions, exclusions), ou chaîne vide.
4. answer_found : true si au moins un extrait recopié répond à la question,
   même en partie ou sous conditions ; false seulement si evidence est vide.
5. complete_answer_found : true si les extraits répondent entièrement ; false
   si la réponse dépend de conditions ou d'informations absentes des documents.
6. confidence : nombre entre 0 et 1.

QUESTION :
{question}

DOCUMENTS FOURNIS :
{context}
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
        quote = html.unescape(item.quote)  # « L&#39;assurance » -> « L'assurance »
        verified = source_known and quote_is_in_source(quote, contents[item.source_id])
        checked.append({
            "source_id": item.source_id,
            "quote": quote,
            "source_known": source_known,
            "verified": verified,
            # Incident observé avec llama3.2:1b : le modèle recopiait la consigne
            # (« MOT POUR MOT ») au lieu d'une phrase du document.
            "copied_from_instructions": not verified and copies_instructions(quote),
        })
    return checked


INSTRUCTION_TEXT = normalize_text(build_prompt("", ""))


def copies_instructions(quote: str) -> bool:
    """Vrai si l'extrait reprend un passage de la consigne plutôt que d'un document."""
    tokens = normalize_text(quote).split()
    return bool(tokens) and f" {' '.join(tokens)} " in f" {INSTRUCTION_TEXT} "


def is_inconsistent(typed: TypedAnswer | None, checked: list[dict]) -> bool:
    """Le modèle déclare « non trouvé » tout en fournissant des extraits vérifiés."""
    return bool(typed) and not typed.answer_found and any(item["verified"] for item in checked)


def decide_status(typed: TypedAnswer | None, checked: list[dict]) -> str | None:
    """Statut livrable, ou None si une nouvelle étape est nécessaire.

    C'est l'application qui tranche : si le modèle déclare « non trouvé » alors
    que des extraits ont été retrouvés dans les sources (incohérence observée
    avec llama3.2:1b), la réponse est traitée comme partielle, et
    l'incohérence est signalée à l'utilisateur.
    """
    if typed is None:
        return None
    if is_inconsistent(typed, checked):
        return "partial"
    if not typed.answer_found:
        return "not_found"
    if not any(item["verified"] for item in checked):
        return None
    return "answered" if typed.complete_answer_found else "partial"


NOT_FOUND_MESSAGE = (
    "Le modèle n’a pas trouvé de réponse justifiée dans les documents consultés. "
    "L’application n’affiche donc pas de synthèse : voici les extraits des sources "
    "les plus proches de votre question, à lire pour juger vous-même."
)


def build_fallback_text(results, message: str | None = None) -> str:
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

    message = message or (
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
    return complete_sentences(typed.answer) + suffix


def complete_sentences(text: str) -> str:
    """Coupe un texte interrompu par la limite de longueur à sa dernière phrase
    complète (« … pour déterm » devient « … suite à une chute. »)."""
    text = text.strip()
    if not text or text[-1] in ".!?»)":
        return text
    cut = max(text.rfind(mark) for mark in (". ", "! ", "? "))
    return text[: cut + 1] if cut > 0 else text + "…"


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def stage_sources(results, count: int, characters: int) -> list[dict]:
    """Sources transmises à une étape : les ``count`` premières, tronquées."""
    selected = []
    for result in results[:count]:
        content = str(result.get("content", ""))
        if len(content) > characters:
            content = content[:characters].rsplit(" ", 1)[0] + "…"
        selected.append({**result, "content": content})
    return selected


INCONSISTENT_MESSAGE = (
    "Le modèle a déclaré ne pas avoir trouvé de réponse, tout en citant des "
    "passages qui figurent bien dans les sources. Sa conclusion n’étant pas "
    "fiable, l’application n’affiche pas sa synthèse : voici les extraits "
    "vérifiés, à lire pour juger vous-même."
)


def inconsistent_text(checked: list[dict]) -> str:
    """Texte affiché quand le modèle se contredit : seulement les extraits vérifiés."""
    lines = [INCONSISTENT_MESSAGE, ""]
    for item in checked:
        if item["verified"]:
            lines.append(f"- « {item['quote'].strip()} » [{item['source_id']}]")
    return "\n".join(lines)


UNGROUNDED_MESSAGE = (
    "La synthèse rédigée par le modèle contenait des affirmations qui ne "
    "figurent pas dans les sources qu’il a citées. Pour ne rien affirmer sans "
    "preuve, l’application ne l’affiche pas : voici les extraits vérifiés, à "
    "lire pour juger vous-même."
)
MASKED_NOTE = (
    "{count} phrase(s) de la synthèse du modèle ne figurai(en)t pas dans les "
    "sources citées et n’a (ont) pas été affichée(s)."
)


CONDITION_LINE = "Condition indiquée par la source : « {quote} » [{source_id}]"


def ungrounded_text(checked: list[dict]) -> str:
    """Texte affiché quand aucune phrase de la synthèse n'est étayée."""
    lines = [UNGROUNDED_MESSAGE, ""]
    for item in checked:
        if item["verified"]:
            lines.append(f"- « {item['quote'].strip()} » [{item['source_id']}]")
    return "\n".join(lines)


def grounded_display_text(grounding: dict, checked: list[dict]) -> str:
    """Seulement les phrases étayées, suivies des identifiants vérifiés."""
    kept = " ".join(_CITATION_IN_TEXT.sub("", s).strip() for s in grounding["supported"])
    text = display_text_from(kept, checked)
    if grounding["unsupported"]:
        text += "\n\n" + MASKED_NOTE.format(count=len(grounding["unsupported"]))
    return text


_CITATION_IN_TEXT = re.compile(r"\s*\[S\d+\]")


def display_text_from(answer: str, checked: list[dict]) -> str:
    verified_ids = sorted(
        {item["source_id"] for item in checked if item["verified"]},
        key=lambda value: int(value[1:]) if value[1:].isdigit() else 0,
    )
    return complete_sentences(answer) + "".join(f" [{source_id}]" for source_id in verified_ids)


def apply_grounding(status, typed, checked, question, sources):
    """Contrôle d'ancrage d'une réponse livrable.

    Renvoie (statut, rapport). Le statut devient ``None`` (nouvelle étape) si
    aucune phrase n'est étayée, ``partial`` si une partie a été masquée.
    """
    if status not in ("answered", "partial") or typed is None or is_inconsistent(typed, checked):
        return status, None
    grounding = check_grounding(typed.answer, question, checked, sources)
    if not grounding["unsupported"]:
        return status, grounding
    if not grounding["supported"]:
        return None, grounding
    return "partial", grounding


def _deliver(status, typed, checked, results, attempts, sources_read, grounding=None) -> dict:
    """Assemble la réponse livrée pour un statut donné."""
    inconsistent = is_inconsistent(typed, checked)
    if status == "not_found":
        # Le texte éventuel du modèle n'est appuyé par aucun extrait vérifié ;
        # l'application ne l'affiche pas et montre les sources les plus proches.
        text = build_fallback_text(results, NOT_FOUND_MESSAGE)
    elif inconsistent:
        text = inconsistent_text(checked)
    elif grounding is not None and not grounding["supported"]:
        text = ungrounded_text(checked)
    elif grounding is not None and grounding["unsupported"]:
        text = grounded_display_text(grounding, checked)
    else:
        text = display_text(typed, checked)
    omitted = []
    if status in ("answered", "partial") and not inconsistent and not (
            grounding is not None and not grounding["supported"]):
        shown = " ".join(grounding["supported"]) if grounding is not None else typed.answer
        omitted = missing_conditions(shown, checked)
        if omitted:
            # La conclusion du modèle omet une condition écrite dans la source :
            # on l'affiche telle quelle, et la réponse n'est plus « complète ».
            text += "\n\n" + "\n".join(
                CONDITION_LINE.format(quote=item["quote"].strip(), source_id=item["source_id"])
                for item in omitted
            )
            status = "partial"
    typed_dump = typed.model_dump(exclude={"evidence"})
    typed_dump["answer"] = html.unescape(typed_dump["answer"])
    typed_dump["caveats"] = html.unescape(typed_dump["caveats"])
    typed_dump["model_indicators_inconsistent"] = inconsistent
    # Les réserves sont rédigées par le modèle et ne sont pas vérifiées : elles
    # restent dans l'audit mais ne sont pas livrées comme une information sûre
    # (mesure du 6/11 : « les dommages causés par les enfants ne sont pas
    # couverts », alors que la source dit l'inverse).
    typed_dump["model_unverified_caveats"] = typed_dump["caveats"]
    typed_dump["caveats"] = ""
    typed_dump["unsupported_sentences"] = grounding["unsupported"] if grounding else []
    typed_dump["conditions_added_from_sources"] = [item["source_id"] for item in omitted]
    if status == "not_found" or inconsistent or (grounding is not None and grounding["unsupported"]):
        # Conservé pour l'audit, mais retiré des champs affichés.
        typed_dump["model_unverified_text"] = typed_dump["answer"]
        typed_dump["answer"] = (
            " ".join(grounding["supported"])
            if grounding is not None and not (status == "not_found" or inconsistent) else ""
        )
        typed_dump["caveats"] = ""
    return {
        "status": status,
        "answer": html.unescape(text),
        "typed_answer": {**typed_dump, "evidence": checked},
        "attempts": attempts,
        "sources_read": sources_read,
    }


def generate_typed_answer(
    question: str,
    results,
    call_model: Callable[[str, str], str],
    log: Callable[[str], None] = lambda message: None,
    stages=GENERATION_STAGES,
) -> dict:
    """Boucle top-1 puis top-3, puis repli.

    ``call_model(prompt, label)`` renvoie le texte brut du modèle ; les erreurs
    réseau qu'il lève sont propagées à l'appelant.
    """
    # On ne garde que les étapes qui apportent des sources supplémentaires.
    effective, seen = [], 0
    for count, characters in stages:
        available = min(count, len(results))
        if available > seen:
            effective.append((available, characters))
            seen = available

    attempts = 0
    best_partial = None  # réponse partielle vérifiée d'une étape précédente
    ungrounded = None    # dernière synthèse entièrement non étayée (extraits vérifiés)
    for position, (count, characters) in enumerate(effective, start=1):
        sources = stage_sources(results, count, characters)
        label = f"tentative {position} (top-{len(sources)})"
        attempts += 1
        raw = call_model(build_prompt(question, build_context(sources)), label)
        typed = parse_typed_answer(raw)
        checked = verify_answer(typed, sources) if typed else []
        status = decide_status(typed, checked)
        # Contrôle d'ancrage (src/grounding.py) : une synthèse dont aucune
        # phrase ne vient des sources vérifiées est traitée comme invalide.
        status, grounding = apply_grounding(status, typed, checked, question, sources)
        if grounding is not None and not grounding["supported"]:
            ungrounded = (typed, checked, grounding, len(sources))
        is_last = position == len(effective)
        log(
            f"{label} — audit : json_valide={typed is not None}, "
            f"extraits_verifies={sum(c['verified'] for c in checked)}/{len(checked)}, "
            f"extraits_copies_de_la_consigne={sum(c['copied_from_instructions'] for c in checked)}, "
            f"phrases_non_etayees={len(grounding['unsupported']) if grounding else 0}, "
            f"statut={status}"
        )
        # À l'étape 1, seule une réponse complète et justifiée arrête la boucle :
        # une réponse partielle ou « non trouvée » peut être complétée par les
        # sources suivantes.
        if status == "answered":
            return _deliver(status, typed, checked, results, attempts, len(sources), grounding)
        if status == "partial" and best_partial is None:
            best_partial = (typed, checked, attempts, len(sources), grounding)
        if is_last:
            # Une réponse partielle vérifiée obtenue plus tôt vaut mieux qu'une
            # dernière sortie illisible ou « non trouvée » (mesure du 6/11 :
            # sorties de l'étape 2 coupées par la limite de longueur).
            if status in (None, "not_found") and best_partial is not None:
                typed_b, checked_b, _, read_b, grounding_b = best_partial
                log(f"{label} — on garde la réponse partielle vérifiée de l'étape précédente")
                return _deliver("partial", typed_b, checked_b, results, attempts, read_b, grounding_b)
            if status is not None:
                return _deliver(status, typed, checked, results, attempts, len(sources), grounding)

    if ungrounded is not None:
        # Des extraits ont été vérifiés, mais la synthèse n'en découlait pas :
        # on montre les extraits vérifiés, jamais la synthèse.
        typed_u, checked_u, grounding_u, read_u = ungrounded
        delivered = _deliver("partial", typed_u, checked_u, results, attempts, read_u, grounding_u)
        delivered["status"] = "fallback"
        return delivered

    return {
        "status": "fallback",
        "answer": build_fallback_text(results),
        "typed_answer": None,
        "attempts": attempts,
        "sources_read": seen,
    }
