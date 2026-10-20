
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, constr

from src.claims_analytics import (
    get_claim_analysis,
    get_claims_cases,
    get_claims_model_info,
    get_claims_summary,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "Data" / "insurance-fr.csv"
INDEX_FILE = PROJECT_ROOT / "static" / "index.html"
CLAIMS_INDEX_FILE = PROJECT_ROOT / "static" / "claims.html"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
).strip()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

try:
    OLLAMA_TIMEOUT_SECONDS = max(
        5,
        int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
    )
except (TypeError, ValueError):
    OLLAMA_TIMEOUT_SECONDS = 120

APP_VERSION = "7.6.0"

print(f"Insurance Claims App — version {APP_VERSION}")
print(f"Modèle Ollama configuré : {OLLAMA_MODEL}")


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

print("Chargement des données...")

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Fichier de données introuvable : {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = {"content", "title"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Colonnes manquantes dans {DATA_FILE.name} : "
        f"{', '.join(sorted(missing_columns))}"
    )

documents = df["content"].fillna("").astype(str).tolist()
titles = df["title"].fillna("").astype(str).tolist()

if not documents or not any(document.strip() for document in documents):
    raise ValueError("Le fichier de données ne contient aucun document exploitable.")



# ============================================================
# RECHERCHE TEXTUELLE LEGERE
# ============================================================

print("Préparation de la recherche textuelle...")

vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    max_features=20000,
)

document_embeddings = vectorizer.fit_transform(documents)

# Aucun CrossEncoder n'est chargé : le classement repose sur TF-IDF,
# la similarité cosinus et les indices lexicaux.
reranker = None

print(
    f"Recherche textuelle prête : "
    f"{len(documents)} documents, "
    f"{len(vectorizer.vocabulary_)} termes."
)


# ============================================================
# SELECTION HYBRIDE DES CANDIDATS
# ============================================================

_FRENCH_QUERY_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "est", "la", "le", "les", "leur", "leurs",
    "mais", "mon", "ou", "par", "pas", "pour", "qu", "que", "quel",
    "quelle", "quelles", "quels", "qui", "sa", "se", "ses", "son",
    "sur", "un", "une", "vos", "votre", "vous", "comment", "fonctionne",
    "assurance", "assurances", "habitation", "contrat", "contrats",
}


def _normalize_search_text(value):
    """Normalise les accents et la ponctuation."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", without_accents.lower())


def _lexical_relevance_score(question, title, content):
    """Calcule un score lexical indicatif."""
    query_tokens = [
        token
        for token in _normalize_search_text(question)
        if token not in _FRENCH_QUERY_STOPWORDS and len(token) > 2
    ]

    unique_query_tokens = list(dict.fromkeys(query_tokens))

    if not unique_query_tokens:
        return 0.0

    title_tokens = set(_normalize_search_text(title))
    content_tokens = set(_normalize_search_text(content))

    title_matches = sum(
        token in title_tokens for token in unique_query_tokens
    )
    content_matches = sum(
        token in content_tokens for token in unique_query_tokens
    )

    phrase_bonus = 0.0
    normalized_title = " ".join(_normalize_search_text(title))
    normalized_content = " ".join(_normalize_search_text(content))

    for position in range(len(unique_query_tokens) - 1):
        phrase = " ".join(unique_query_tokens[position:position + 2])

        if phrase in normalized_title:
            phrase_bonus += 2.0
        elif phrase in normalized_content:
            phrase_bonus += 1.0

    coverage = (
        title_matches * 3.0 + content_matches
    ) / len(unique_query_tokens)

    return round(coverage + phrase_bonus, 6)


def _rank_fusion_score(
    primary_rank,
    lexical_rank,
    lexical_score,
    fusion_constant=60.0,
):
    """Combine les rangs textuel et lexical."""
    primary_contribution = 1.0 / (fusion_constant + primary_rank)

    lexical_contribution = (
        1.0 / (fusion_constant + lexical_rank)
        if lexical_score > 0
        else 0.0
    )

    return primary_contribution + lexical_contribution


def _select_candidate_indices(question, similarities, candidate_limit=20):
    """Sélectionne les candidats par fusion des rangs et score lexical."""
    semantic_order = [
        int(index) for index in similarities.argsort()[::-1]
    ]

    lexical_scores = [
        _lexical_relevance_score(
            question,
            titles[index],
            documents[index],
        )
        for index in range(len(documents))
    ]

    lexical_order = sorted(
        range(len(documents)),
        key=lambda index: (
            lexical_scores[index],
            len(_normalize_search_text(titles[index])),
        ),
        reverse=True,
    )

    semantic_rank = {
        index: rank
        for rank, index in enumerate(semantic_order, start=1)
    }

    lexical_rank = {
        index: rank
        for rank, index in enumerate(lexical_order, start=1)
    }

    fusion_constant = 60.0

    fused_order = sorted(
        range(len(documents)),
        key=lambda index: (
            _rank_fusion_score(
                semantic_rank[index],
                lexical_rank[index],
                lexical_scores[index],
                fusion_constant,
            ),
            lexical_scores[index],
        ),
        reverse=True,
    )

    lexical_quota = min(5, candidate_limit, len(documents))

    strong_lexical_candidates = [
        index
        for index in lexical_order
        if lexical_scores[index] > 0
    ][:lexical_quota]

    selected = list(strong_lexical_candidates)

    selected.extend(
        index
        for index in fused_order
        if index not in selected
    )

    return selected[:min(candidate_limit, len(documents))], lexical_scores


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Insurance RAG API",
    description=(
        "Assistant assurance avec recherche textuelle, "
        "classement hybride et génération de réponses."
    ),
    version=APP_VERSION,
)


@app.get("/version")
def version():
    """Expose la version de l’application."""
    return {
        "app_version": APP_VERSION,
        "ollama_model": OLLAMA_MODEL,
    }


@app.get("/claims")
def claims_dashboard():
    """Sert le tableau de bord des sinistres fictifs."""
    if not CLAIMS_INDEX_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail="Le tableau de bord des sinistres est indisponible.",
        )

    return FileResponse(CLAIMS_INDEX_FILE)


@app.get("/api/claims/summary")
def claims_summary():
    """Expose les indicateurs descriptifs des sinistres fictifs."""
    return get_claims_summary()


@app.get("/api/claims/cases")
def claims_cases(limit: int = 120):
    """Expose un extrait borné des dossiers fictifs."""
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=422,
            detail="Le paramètre limit doit être compris entre 1 et 500.",
        )

    return {
        "cases": get_claims_cases(limit),
        "dataset_label": "FICTIF_DEMONSTRATION",
    }


@app.get("/api/claims/model")
def claims_model_info():
    """Expose les métriques du modèle de démonstration."""
    return get_claims_model_info()


@app.get("/api/claims/{claim_id}/analysis")
def claim_analysis(claim_id: str):
    """Analyse un dossier fictif."""
    result = get_claim_analysis(claim_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucun dossier de démonstration ne correspond "
                "à cet identifiant."
            ),
        )

    return result


@app.get("/health")
def health():
    """Vérifie que le processus API répond."""
    return {
        "status": "ok",
        "app_version": APP_VERSION,
    }


@app.get("/ready")
def readiness():
    """Vérifie les fichiers, la recherche et la disponibilité d'Ollama."""
    checks = {
        "dataset_available": DATA_FILE.is_file() and not df.empty,
        "interface_available": INDEX_FILE.is_file(),
        "claims_dashboard_available": CLAIMS_INDEX_FILE.is_file(),
        "claims_dataset_available": (
            PROJECT_ROOT / "Data" / "synthetic_claims.csv"
        ).is_file(),
        "text_search_loaded": (
            document_embeddings is not None
            and document_embeddings.shape[0] == len(documents)
            and bool(vectorizer.vocabulary_)
        ),
    }

    ollama_url = (
        OLLAMA_URL.rsplit("/api/generate", 1)[0].rstrip("/")
        + "/api/tags"
    )

    try:
        ollama_response = requests.get(
            ollama_url,
            timeout=min(5, OLLAMA_TIMEOUT_SECONDS),
        )

        checks["ollama_reachable"] = ollama_response.ok

        if ollama_response.ok:
            try:
                model_names = {
                    item.get("name", "")
                    for item in ollama_response.json().get("models", [])
                    if isinstance(item, dict)
                }

                checks["ollama_model_available"] = (
                    OLLAMA_MODEL in model_names
                    or any(
                        name.split(":", 1)[0]
                        == OLLAMA_MODEL.split(":", 1)[0]
                        for name in model_names
                    )
                )
            except (ValueError, AttributeError, TypeError):
                checks["ollama_model_available"] = False
        else:
            checks["ollama_model_available"] = False

    except requests.exceptions.RequestException:
        checks["ollama_reachable"] = False
        checks["ollama_model_available"] = False

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
# MODELE DE REQUETE
# ============================================================

class Question(BaseModel):
    question: constr(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    )


# ============================================================
# PAGE D'ACCUEIL
# ============================================================

@app.get("/")
def home():
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail="La page d'accueil est indisponible.",
        )

    return FileResponse(INDEX_FILE)


# ============================================================
# CONTROLE DES REFERENCES AUX SOURCES
# ============================================================

_SOURCE_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def audit_citations(answer, results):
    """Vérifie la cohérence formelle des identifiants de source."""
    answer_text = answer if isinstance(answer, str) else ""

    available_ids = {
        str(result.get("source_id"))
        for result in results
        if result.get("source_id")
    }

    cited_ids = set(_SOURCE_CITATION_PATTERN.findall(answer_text))

    valid_ids = sorted(
        cited_ids & available_ids,
        key=lambda value: int(value[1:]),
    )

    invalid_ids = sorted(
        cited_ids - available_ids,
        key=lambda value: int(value[1:]),
    )

    uncited_ids = sorted(
        available_ids - cited_ids,
        key=lambda value: int(value[1:]),
    )

    return {
        "citations_found": bool(cited_ids),
        "valid_source_ids": valid_ids,
        "unknown_source_ids": invalid_ids,
        "available_but_uncited_source_ids": uncited_ids,
        "format_check_passed": bool(cited_ids) and not invalid_ids,
        "scope": (
            "Contrôle des identifiants uniquement ; "
            "la pertinence factuelle reste à vérifier."
        ),
    }


# ============================================================
# GENERATION AVEC OLLAMA
# ============================================================

def generate_answer(question, results):
    """Génère une réponse à partir des documents récupérés."""

    diagnostics_enabled = (
        os.getenv("RAG_DIAGNOSTICS", "0").strip().lower()
        in {"1", "true", "yes", "oui"}
    )

    def diagnostic(message):
        if diagnostics_enabled:
            print(f"[RAG_DIAGNOSTICS] {message}", flush=True)

    context_parts = []

    for result in results:
        source_id = result.get(
            "source_id",
            f"S{result.get('rank', len(context_parts) + 1)}",
        )

        context_parts.append(
            f"SOURCE [{source_id}]\n"
            f"TITRE : {result['title']}\n"
            f"CONTENU : {result['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
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

    def call_ollama(current_prompt, attempt_label):
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": current_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        call_started = time.perf_counter()

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError(
                "Le JSON retourné par Ollama n'est pas un objet."
            )

        answer_text = data.get("response")

        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError(
                "Le champ 'response' est absent ou vide."
            )

        answer_text = answer_text.strip()

        call_ms = round(
            (time.perf_counter() - call_started) * 1000,
            2,
        )

        diagnostic(
            f"{attempt_label} — durée Ollama : {call_ms} ms"
        )

        diagnostic(
            f"{attempt_label} — sortie brute début >>>\n"
            f"{answer_text}\n<<< sortie brute fin"
        )

        return answer_text

    try:
        answer = call_ollama(prompt, "tentative 1")

        first_audit = audit_citations(answer, results)
        diagnostic(f"tentative 1 — audit : {first_audit}")

        if not first_audit["format_check_passed"]:
            available_ids = [
                str(result["source_id"])
                for result in results
                if result.get("source_id")
            ]

            retry_prompt = f"""
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
{', '.join(f'[{source_id}]' for source_id in available_ids) or '(aucune source disponible)'}.
- N'invente aucun identifiant ni aucune information.
- Si les documents ne permettent pas de répondre, dis-le explicitement.
- Retourne uniquement la réponse finale, sans commentaire sur ces consignes.

RÉPONSE PRÉCÉDENTE À CORRIGER :
{answer}

RÉPONSE FINALE :
"""

            revised_answer = call_ollama(
                retry_prompt,
                "tentative 2 / révision",
            )

            revised_audit = audit_citations(revised_answer, results)

            diagnostic(
                f"tentative 2 / révision — audit : {revised_audit}"
            )

            if revised_audit["format_check_passed"]:
                return revised_answer

            print(
                "Réponse Ollama écartée : citations absentes ou "
                "inconnues après une tentative de correction."
            )

            source_excerpts = []

            for result in results[:3]:
                fallback_source_id = str(
                    result.get("source_id", "")
                ).strip()

                fallback_title = str(
                    result.get("title", "Document récupéré")
                ).strip()

                fallback_content = str(
                    result.get("content", "")
                ).strip()

                if not fallback_source_id or not fallback_content:
                    continue

                excerpt_limit = 700

                if len(fallback_content) > excerpt_limit:
                    fallback_content = (
                        fallback_content[:excerpt_limit]
                        .rsplit(" ", 1)[0]
                        + "…"
                    )

                source_excerpts.append(
                    f"- **{fallback_title}** "
                    f"[{fallback_source_id}] : {fallback_content}"
                )

            safe_fallback = (
                "Je ne peux pas formuler une synthèse suffisamment "
                "traçable : le modèle n’a pas fourni de citations "
                "vérifiables. Pour éviter d’affirmer un élément sans "
                "preuve, voici les extraits des documents récupérés, "
                "avec leurs identifiants de source. Leur pertinence "
                "doit être vérifiée au regard de votre question."
            )

            if source_excerpts:
                safe_fallback += "\n\n" + "\n\n".join(source_excerpts)
            else:
                safe_fallback += (
                    "\n\nAucun extrait exploitable avec identifiant "
                    "de source n’est disponible pour cette requête."
                )

            return safe_fallback

        return answer

    except requests.exceptions.RequestException as error:
        print(f"Erreur Ollama : {error}")

        raise HTTPException(
            status_code=502,
            detail=(
                "Le service de génération de réponse est indisponible. "
                "Vérifiez qu'Ollama est lancé et que le modèle est disponible."
            ),
        ) from error

    except (ValueError, TypeError) as error:
        print(f"Réponse Ollama invalide : {error}")

        raise HTTPException(
            status_code=502,
            detail="Le service de génération a renvoyé une réponse invalide.",
        ) from error


# ============================================================
# RECHERCHE
# ============================================================

@app.post("/search")
def search(data: Question):
    request_started = time.perf_counter()
    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="La question ne peut pas être vide.",
        )

    # --------------------------------------------------------
    # 1. Transformation de la question
    # --------------------------------------------------------

    embedding_started = time.perf_counter()

    question_embedding = vectorizer.transform([question])

    embedding_ms = (
        time.perf_counter() - embedding_started
    ) * 1000

    # --------------------------------------------------------
    # 2. Recherche initiale par similarité cosinus
    # --------------------------------------------------------

    retrieval_started = time.perf_counter()

    similarities = cosine_similarity(
        question_embedding,
        document_embeddings,
    )[0]

    # --------------------------------------------------------
    # 3. Sélection hybride des candidats
    # --------------------------------------------------------

    candidate_indices, lexical_scores = _select_candidate_indices(
        question,
        similarities,
        candidate_limit=20,
    )

    retrieval_ms = (
        time.perf_counter() - retrieval_started
    ) * 1000

    # --------------------------------------------------------
    # 4. Classement final sans CrossEncoder
    # --------------------------------------------------------

    reranking_started = time.perf_counter()

    reranked = sorted(
        (
            (index, float(similarities[index]))
            for index in candidate_indices
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    reranking_ms = (
        time.perf_counter() - reranking_started
    ) * 1000

    # --------------------------------------------------------
    # 5. Fusion des rangs textuels et lexicaux
    # --------------------------------------------------------

    primary_rank = {
        index: rank
        for rank, (index, _) in enumerate(reranked, start=1)
    }

    lexical_order = sorted(
        range(len(documents)),
        key=lambda index: lexical_scores[index],
        reverse=True,
    )

    lexical_rank = {
        index: rank
        for rank, index in enumerate(lexical_order, start=1)
    }

    fusion_constant = 60.0

    fused_reranked = sorted(
        reranked,
        key=lambda item: (
            _rank_fusion_score(
                primary_rank[item[0]],
                lexical_rank[item[0]],
                lexical_scores[item[0]],
                fusion_constant,
            ),
            item[1],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # 6. Sélection des cinq meilleurs documents
    # --------------------------------------------------------

    results = []

    for rank, (index, text_score) in enumerate(
        fused_reranked[:5],
        start=1,
    ):
        text_score = float(text_score)
        similarity = float(similarities[index])

        results.append({
            "rank": rank,
            "source_id": f"S{rank}",
            "title": titles[index],
            # Noms conservés pour compatibilité avec l'interface existante.
            # Ce score n'est plus un score de CrossEncoder.
            "score": text_score,
            "reranker_score": text_score,
            "embedding_similarity": similarity,
            "lexical_relevance_score": float(lexical_scores[index]),
            "hybrid_rank_score": round(
                _rank_fusion_score(
                    primary_rank[index],
                    lexical_rank[index],
                    lexical_scores[index],
                    fusion_constant,
                ),
                8,
            ),
            "content": documents[index][:1500],
        })

    # --------------------------------------------------------
    # 7. Génération de la réponse avec Ollama
    # --------------------------------------------------------

    print("Génération de la réponse avec Ollama...")

    generation_started = time.perf_counter()

    answer = generate_answer(
        question,
        results,
    )

    generation_ms = (
        time.perf_counter() - generation_started
    ) * 1000

    citation_audit = audit_citations(answer, results)

    total_ms = (
        time.perf_counter() - request_started
    ) * 1000

    # --------------------------------------------------------
    # 8. Réponse JSON
    # --------------------------------------------------------

    return {
        "app_version": APP_VERSION,
        "question": question,
        "answer": answer,
        "results": results,
        "citation_audit": citation_audit,
        "pipeline_metrics": {
            "document_count": len(documents),
            "candidate_count": len(candidate_indices),
            "returned_count": len(results),
            "embedding_ms": round(embedding_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "reranking_ms": round(reranking_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
            "note": (
                "Durées observées pendant cette requête ; elles varient "
                "selon la machine, la charge et les modèles."
            ),
        },
        "score_explanation": {
            "reranker_score": (
                "Ancien nom conservé pour compatibilité : score de "
                "similarité textuelle utilisé dans cette version, "
                "pas un score de CrossEncoder ni une probabilité."
            ),
            "embedding_similarity": (
                "Similarité cosinus TF-IDF utilisée pour classer "
                "les documents ; ce n'est pas une mesure de vérité."
            ),
            "lexical_relevance_score": (
                "Score lexical indicatif basé sur les termes de la "
                "question présents dans le titre et le contenu ; "
                "ce n'est pas une probabilité."
            ),
            "hybrid_rank_score": (
                "Score de fusion des rangs textuel et lexical ; "
                "il sert au classement final et ne mesure ni une "
                "probabilité ni la véracité."
            ),
        },
    }
