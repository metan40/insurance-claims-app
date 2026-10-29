"""API FastAPI de l'assistant assurance (RAG) et du tableau de bord sinistres.

Chaîne de traitement d'une question (``POST /search``) :

1. recherche documentaire hybride TF-IDF + lexicale (module ``src.retrieval``) ;
2. génération d'une réponse par un LLM local servi par Ollama, à partir des
   cinq documents retenus uniquement ;
3. contrôle formel des citations ``[S1]``…``[S5]`` ; si le modèle ne cite pas
   correctement ses sources, une seconde tentative est demandée, puis, en
   dernier recours, l'API renvoie les extraits des sources plutôt qu'une
   réponse non traçable.

Les routes ``/api/claims/*`` exposent le module d'aide à l'analyse des
sinistres fictifs (``src.claims_analytics``).
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, StringConstraints

from src.claims_analytics import (
    CLAIMS_FILE,
    get_claim_analysis,
    get_claims_cases,
    get_claims_model_info,
    get_claims_summary,
)
from src.retrieval import DEFAULT_CANDIDATE_LIMIT, DEFAULT_TOP_K, HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("insurance_rag")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "Data" / "insurance-fr.csv"
INDEX_FILE = PROJECT_ROOT / "static" / "index.html"
CLAIMS_INDEX_FILE = PROJECT_ROOT / "static" / "claims.html"
VERSION_FILE = PROJECT_ROOT / "VERSION"


def read_app_version() -> str:
    """Lit la version unique du projet (fichier VERSION à la racine)."""
    try:
        version_text = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "inconnue"
    return version_text or "inconnue"


def read_int_setting(raw_value: str | None, default: int, minimum: int) -> int:
    """Convertit une variable d'environnement en entier borné (défaut si invalide)."""
    try:
        return max(minimum, int(raw_value if raw_value is not None else default))
    except (TypeError, ValueError):
        return default


APP_VERSION = read_app_version()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
OLLAMA_TIMEOUT_SECONDS = read_int_setting(os.getenv("OLLAMA_TIMEOUT_SECONDS"), default=120, minimum=5)
# Rapidité : borne la longueur des réponses et garde le modèle chargé en
# mémoire entre deux questions (évite un rechargement de plusieurs secondes).
OLLAMA_NUM_PREDICT = read_int_setting(os.getenv("OLLAMA_NUM_PREDICT"), default=400, minimum=50)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
FALLBACK_EXCERPT_CHARACTERS = 700
CONTEXT_CHARACTERS_PER_SOURCE = 1500

logger.info("Insurance Claims Assistance — version %s", APP_VERSION)
logger.info("Modèle Ollama configuré : %s", OLLAMA_MODEL)


# ============================================================
# CHARGEMENT DU CORPUS ET INDEXATION
# ============================================================

if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Fichier de données introuvable : {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
missing_columns = {"content", "title"} - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Colonnes manquantes dans {DATA_FILE.name} : {', '.join(sorted(missing_columns))}"
    )

documents = df["content"].fillna("").astype(str).tolist()
titles = df["title"].fillna("").astype(str).tolist()

retriever = HybridRetriever(titles, documents)
logger.info(
    "Recherche prête : %s documents, %s termes TF-IDF.",
    len(retriever),
    retriever.vocabulary_size,
)


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Insurance Claims Assistance API",
    description=(
        "Assistant assurance : recherche documentaire hybride (TF-IDF + lexicale), "
        "génération par LLM local avec citations contrôlées, et aide à l'analyse "
        "de sinistres fictifs."
    ),
    version=APP_VERSION,
)

QuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class Question(BaseModel):
    """Corps de la requête POST /search."""

    question: QuestionText


@app.get("/")
def home():
    """Sert l'interface principale de questions-réponses."""
    if not INDEX_FILE.is_file():
        raise HTTPException(status_code=503, detail="La page d'accueil est indisponible.")
    return FileResponse(INDEX_FILE)


@app.get("/version")
def version():
    """Expose la version de l'application et le modèle configuré."""
    return {"app_version": APP_VERSION, "ollama_model": OLLAMA_MODEL}


@app.get("/health")
def health():
    """Vérifie seulement que le processus API répond."""
    return {"status": "ok", "app_version": APP_VERSION}


def ollama_tags_url(generate_url: str) -> str:
    """Déduit l'URL /api/tags à partir de l'URL /api/generate configurée."""
    return generate_url.rsplit("/api/generate", 1)[0].rstrip("/") + "/api/tags"


def model_is_installed(configured_model: str, installed_names: set[str]) -> bool:
    """Compare le modèle configuré aux modèles installés.

    Ollama ajoute l'étiquette ``:latest`` lorsqu'aucune n'est précisée ; une
    autre étiquette (par exemple ``llama3.2:1b`` au lieu de ``llama3.2:3b``)
    n'est pas considérée comme équivalente.
    """
    def canonical(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return canonical(configured_model) in {canonical(name) for name in installed_names}


@app.get("/ready")
def readiness():
    """Vérifie les fichiers, l'index de recherche et la disponibilité d'Ollama."""
    checks = {
        "dataset_available": DATA_FILE.is_file() and not df.empty,
        "interface_available": INDEX_FILE.is_file(),
        "claims_dashboard_available": CLAIMS_INDEX_FILE.is_file(),
        "claims_dataset_available": CLAIMS_FILE.is_file(),
        "text_search_loaded": (
            retriever.matrix.shape[0] == len(documents) and retriever.vocabulary_size > 0
        ),
        "ollama_reachable": False,
        "ollama_model_available": False,
    }

    try:
        response = requests.get(
            ollama_tags_url(OLLAMA_URL), timeout=min(5, OLLAMA_TIMEOUT_SECONDS)
        )
        checks["ollama_reachable"] = bool(response.ok)
        if response.ok:
            try:
                installed = {
                    str(item.get("name", ""))
                    for item in response.json().get("models", [])
                    if isinstance(item, dict)
                }
                checks["ollama_model_available"] = model_is_installed(OLLAMA_MODEL, installed)
            except (ValueError, AttributeError, TypeError):
                checks["ollama_model_available"] = False
    except requests.exceptions.RequestException:
        pass

    all_ready = all(checks.values())
    result = {
        "status": "ready" if all_ready else "not_ready",
        "app_version": APP_VERSION,
        "ollama_model": OLLAMA_MODEL,
        "checks": checks,
        "note": (
            "Ce contrôle vérifie la disponibilité technique ; "
            "il ne valide pas la qualité métier des réponses."
        ),
    }
    if not all_ready:
        raise HTTPException(status_code=503, detail=result)
    return result


# ============================================================
# TABLEAU DE BORD DES SINISTRES (DONNÉES FICTIVES)
# ============================================================

@app.get("/claims")
def claims_dashboard():
    """Sert le tableau de bord des sinistres fictifs."""
    if not CLAIMS_INDEX_FILE.is_file():
        raise HTTPException(
            status_code=503, detail="Le tableau de bord des sinistres est indisponible."
        )
    return FileResponse(CLAIMS_INDEX_FILE)


@app.get("/api/claims/summary")
def claims_summary():
    """Indicateurs descriptifs (les Pending sont exclus du taux d'acceptation)."""
    return get_claims_summary()


@app.get("/api/claims/cases")
def claims_cases(limit: int = 120):
    """Extrait borné des dossiers fictifs."""
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=422, detail="Le paramètre limit doit être compris entre 1 et 500."
        )
    return {"cases": get_claims_cases(limit), "dataset_label": "FICTIF_DEMONSTRATION"}


@app.get("/api/claims/model")
def claims_model_info():
    """Métriques du modèle de démonstration."""
    return get_claims_model_info()


@app.get("/api/claims/{claim_id}/analysis")
def claim_analysis(claim_id: str):
    """Score exploratoire et facteurs observables d'un dossier fictif."""
    result = get_claim_analysis(claim_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Aucun dossier de démonstration ne correspond à cet identifiant.",
        )
    return result


# ============================================================
# CONTRÔLE DES RÉFÉRENCES AUX SOURCES
# ============================================================

_SOURCE_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def _source_number(source_id: str) -> int:
    return int(source_id[1:])


def audit_citations(answer, results):
    """Vérifie la cohérence formelle des identifiants de source cités."""
    answer_text = answer if isinstance(answer, str) else ""
    available_ids = {
        str(result.get("source_id")) for result in results if result.get("source_id")
    }
    cited_ids = set(_SOURCE_CITATION_PATTERN.findall(answer_text))

    return {
        "citations_found": bool(cited_ids),
        "valid_source_ids": sorted(cited_ids & available_ids, key=_source_number),
        "unknown_source_ids": sorted(cited_ids - available_ids, key=_source_number),
        "available_but_uncited_source_ids": sorted(
            available_ids - cited_ids, key=_source_number
        ),
        "format_check_passed": bool(cited_ids) and not (cited_ids - available_ids),
        "scope": (
            "Contrôle des identifiants uniquement ; "
            "la pertinence factuelle reste à vérifier."
        ),
    }


# ============================================================
# GÉNÉRATION AVEC OLLAMA
# ============================================================

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
    """Consigne principale : répondre uniquement à partir des sources, en les citant."""
    return f"""
Tu es un assistant spécialisé dans l'assurance.

Tu dois répondre à la question de l'utilisateur uniquement
à partir des informations présentes dans les documents fournis.

Si les documents ne permettent pas de répondre avec certitude,
dis-le clairement.

Ne crée aucune information qui n'est pas présente dans les documents.

Lorsque tu avances une information factuelle tirée d'un document, indique la source
correspondante avec son identifiant exact entre crochets, par exemple [S1].
N'invente jamais d'identifiant de source. Si les documents ne justifient pas une
affirmation, ne la présente pas comme un fait établi.

Réponds en français de manière claire, concise et professionnelle.

QUESTION :
{question}

DOCUMENTS FOURNIS :
{context}

REPONSE :
"""


def build_revision_prompt(question: str, context: str, previous_answer: str, available_ids) -> str:
    """Consigne de révision lorsque la première réponse ne cite pas correctement."""
    allowed = ", ".join(f"[{source_id}]" for source_id in available_ids)
    return f"""
Tu dois corriger la réponse précédente en utilisant exclusivement la question
et les documents reproduits ci-dessous. Ces éléments sont inclus dans cette
requête car chaque appel à Ollama est indépendant.

QUESTION ORIGINALE :
{question}

DOCUMENTS DE RÉFÉRENCE :
{context}

RÈGLES DE RÉVISION :
- Réponds en français, de façon claire et concise.
- Chaque affirmation factuelle doit être accompagnée d'au moins une citation exacte.
- Les seuls identifiants autorisés sont :
{allowed or '(aucune source disponible)'}.
- N'invente aucun identifiant ni aucune information.
- Si les documents ne permettent pas de répondre, dis-le explicitement.
- Retourne uniquement la réponse finale, sans commentaire sur ces consignes.

RÉPONSE PRÉCÉDENTE À CORRIGER :
{previous_answer}

RÉPONSE FINALE :
"""


def build_safe_fallback(results) -> str:
    """Réponse de repli : extraits des sources, sans synthèse non traçable."""
    excerpts = []
    for result in results[:3]:
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
        "traçable : le modèle n’a pas fourni de citations "
        "vérifiables. Pour éviter d’affirmer un élément sans "
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


def diagnostics_enabled() -> bool:
    return os.getenv("RAG_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "oui"}


def call_ollama(prompt: str, attempt_label: str) -> str:
    """Appelle Ollama (non streamé) et renvoie le texte de la réponse."""
    started = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0.1, "num_predict": OLLAMA_NUM_PREDICT},
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Le JSON retourné par Ollama n'est pas un objet.")
    answer_text = data.get("response")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("Le champ 'response' est absent ou vide.")
    answer_text = answer_text.strip()

    if diagnostics_enabled():
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        print(f"[RAG_DIAGNOSTICS] {attempt_label} — durée Ollama : {duration_ms} ms", flush=True)
        print(
            f"[RAG_DIAGNOSTICS] {attempt_label} — sortie brute début >>>\n"
            f"{answer_text}\n<<< sortie brute fin",
            flush=True,
        )
    return answer_text


def generate_answer(question, results):
    """Génère une réponse citée à partir des documents récupérés.

    Lève HTTPException(502) si Ollama est injoignable ou répond de façon
    inexploitable.
    """
    context = build_context(results)

    def log_audit(label, audit):
        if diagnostics_enabled():
            print(f"[RAG_DIAGNOSTICS] {label} — audit : {audit}", flush=True)

    try:
        answer = call_ollama(build_prompt(question, context), "tentative 1")
        first_audit = audit_citations(answer, results)
        log_audit("tentative 1", first_audit)
        if first_audit["format_check_passed"]:
            return answer

        available_ids = [str(r["source_id"]) for r in results if r.get("source_id")]
        revised = call_ollama(
            build_revision_prompt(question, context, answer, available_ids),
            "tentative 2 / révision",
        )
        revised_audit = audit_citations(revised, results)
        log_audit("tentative 2 / révision", revised_audit)
        if revised_audit["format_check_passed"]:
            return revised

        logger.warning(
            "Réponse Ollama écartée : citations absentes ou inconnues après révision."
        )
        return build_safe_fallback(results)

    except requests.exceptions.RequestException as error:
        logger.error("Erreur Ollama : %s", error)
        raise HTTPException(
            status_code=502,
            detail=(
                "Le service de génération de réponse est indisponible. "
                "Vérifiez qu'Ollama est lancé et que le modèle est disponible."
            ),
        ) from error
    except (ValueError, TypeError) as error:
        logger.error("Réponse Ollama invalide : %s", error)
        raise HTTPException(
            status_code=502,
            detail="Le service de génération a renvoyé une réponse invalide.",
        ) from error


# ============================================================
# RECHERCHE + GÉNÉRATION
# ============================================================

SCORE_EXPLANATION = {
    "tfidf_cosine_similarity": (
        "Similarité cosinus TF-IDF entre la question et le document ; "
        "ce n'est pas une probabilité ni une mesure de vérité."
    ),
    "lexical_relevance_score": (
        "Score lexical indicatif basé sur les termes de la question présents "
        "dans le titre et le contenu ; ce n'est pas une probabilité."
    ),
    "hybrid_rank_score": (
        "Score de fusion des rangs (RRF) textuel et lexical ; il sert au "
        "classement final et ne mesure ni une probabilité ni la véracité."
    ),
}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


@app.post("/search")
def search(data: Question):
    """Recherche les documents pertinents puis génère une réponse citée."""
    request_started = time.perf_counter()
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="La question ne peut pas être vide.")

    vectorization_started = time.perf_counter()
    cosine = retriever.cosine_scores(question)
    vectorization_ms = _elapsed_ms(vectorization_started)

    ranking_started = time.perf_counter()
    ranked = retriever.rank(question, candidate_limit=DEFAULT_CANDIDATE_LIMIT, cosine=cosine)
    ranking_ms = _elapsed_ms(ranking_started)

    results = [
        {
            "rank": item.rank,
            "source_id": f"S{item.rank}",
            "title": titles[item.index],
            "tfidf_cosine_similarity": item.cosine_similarity,
            "lexical_relevance_score": item.lexical_score,
            "hybrid_rank_score": item.fusion_score,
            "content": documents[item.index][:CONTEXT_CHARACTERS_PER_SOURCE],
        }
        for item in ranked[:DEFAULT_TOP_K]
    ]

    generation_started = time.perf_counter()
    answer = generate_answer(question, results)
    generation_ms = _elapsed_ms(generation_started)

    return {
        "app_version": APP_VERSION,
        "question": question,
        "answer": answer,
        "results": results,
        "citation_audit": audit_citations(answer, results),
        "pipeline_metrics": {
            "document_count": len(documents),
            "candidate_count": len(ranked),
            "returned_count": len(results),
            "vectorization_ms": vectorization_ms,
            "ranking_ms": ranking_ms,
            "generation_ms": generation_ms,
            "total_ms": _elapsed_ms(request_started),
            "note": (
                "Durées observées pendant cette requête ; elles varient "
                "selon la machine, la charge et les modèles."
            ),
        },
        "score_explanation": SCORE_EXPLANATION,
    }
