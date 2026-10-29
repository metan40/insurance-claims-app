import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder
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

# Les paramètres d'exécution sont configurables sans modifier le code.
# OLLAMA_URL doit désigner le point d'accès complet /api/generate.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
try:
    OLLAMA_TIMEOUT_SECONDS = max(5, int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")))
except (TypeError, ValueError):
    OLLAMA_TIMEOUT_SECONDS = 120
APP_VERSION = "7.5.0"
print(f"Insurance Claims App — version {APP_VERSION}")
print(f"Modèle Ollama configuré : {OLLAMA_MODEL}")


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

print("Chargement des données...")

df = pd.read_csv(DATA_FILE)

documents = df["content"].fillna("").tolist()
titles = df["title"].fillna("").tolist()


# ============================================================
# MODELE D'EMBEDDINGS
# ============================================================

print("Chargement du modèle d'embeddings...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Calcul des embeddings des documents...")

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# MODELE DE RERANKING
# ============================================================

print("Chargement du modèle de reranking...")

reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)



# ============================================================
# SELECTION HYBRIDE DES CANDIDATS (EMBEDDINGS + INDICES LEXICAUX)
# ============================================================

# Les embeddings retrouvent des formulations proches, mais ils peuvent écarter
# un document qui contient exactement une notion métier importante. Le signal
# lexical complète donc la recherche sémantique sans remplacer le CrossEncoder.
_FRENCH_QUERY_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "est", "la", "le", "les", "leur", "leurs",
    "mais", "mon", "ou", "par", "pas", "pour", "qu", "que", "quel",
    "quelle", "quelles", "quels", "qui", "sa", "se", "ses", "son",
    "sur", "un", "une", "vos", "votre", "vous", "comment", "fonctionne",
    "assurance", "assurances", "habitation", "contrat", "contrats",
}

def _normalize_search_text(value):
    """Normalise les accents et la ponctuation pour comparer les termes métier."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", without_accents.lower())

def _lexical_relevance_score(question, title, content):
    """Calcule un signal lexical explicable, distinct d'un score de probabilité."""
    query_tokens = [
        token for token in _normalize_search_text(question)
        if token not in _FRENCH_QUERY_STOPWORDS and len(token) > 2
    ]
    unique_query_tokens = list(dict.fromkeys(query_tokens))
    if not unique_query_tokens:
        return 0.0

    title_tokens = set(_normalize_search_text(title))
    content_tokens = set(_normalize_search_text(content))
    title_matches = sum(token in title_tokens for token in unique_query_tokens)
    content_matches = sum(token in content_tokens for token in unique_query_tokens)
    phrase_bonus = 0.0
    normalized_title = " ".join(_normalize_search_text(title))
    normalized_content = " ".join(_normalize_search_text(content))
    for position in range(len(unique_query_tokens) - 1):
        phrase = " ".join(unique_query_tokens[position:position + 2])
        if phrase in normalized_title:
            phrase_bonus += 2.0
        elif phrase in normalized_content:
            phrase_bonus += 1.0

    coverage = (title_matches * 3.0 + content_matches) / len(unique_query_tokens)
    return round(coverage + phrase_bonus, 6)

def _rank_fusion_score(primary_rank, lexical_rank, lexical_score, fusion_constant=60.0):
    """Fusionne les rangs sans donner de bonus lexical aux documents sans recouvrement.

    Un rang lexical existe mécaniquement pour chaque document, même lorsque tous
    ses termes sont absents de la question. Dans ce cas, le rang seul ne constitue
    pas une preuve de pertinence et ne doit pas contribuer au score de fusion.
    """
    primary_contribution = 1.0 / (fusion_constant + primary_rank)
    lexical_contribution = (
        1.0 / (fusion_constant + lexical_rank)
        if lexical_score > 0
        else 0.0
    )
    return primary_contribution + lexical_contribution


def _select_candidate_indices(question, similarities, candidate_limit=20):
    """Fusionne les rangs sémantiques et lexicaux pour éviter les oublis métier."""
    semantic_order = [int(index) for index in similarities.argsort()[::-1]]
    lexical_scores = [
        _lexical_relevance_score(question, titles[index], documents[index])
        for index in range(len(documents))
    ]
    lexical_order = sorted(
        range(len(documents)),
        key=lambda index: (lexical_scores[index], len(_normalize_search_text(titles[index]))),
        reverse=True,
    )

    # Le Reciprocal Rank Fusion combine les rangs sans mélanger des échelles
    # incompatibles (similarité cosinus et score lexical). Les documents sans
    # correspondance lexicale restent éligibles grâce à leur rang sémantique.
    semantic_rank = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
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

    # Réserver une partie du budget aux meilleurs documents lexicaux empêche
    # qu'un bon résultat métier disparaisse lorsque les rangs sémantiques sont
    # très proches ou ex æquo. Le CrossEncoder décidera ensuite du classement final.
    lexical_quota = min(5, candidate_limit, len(documents))
    strong_lexical_candidates = [
        index for index in lexical_order if lexical_scores[index] > 0
    ][:lexical_quota]
    selected = list(strong_lexical_candidates)
    selected.extend(
        index for index in fused_order
        if index not in selected
    )
    return selected[:min(candidate_limit, len(documents))], lexical_scores


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Insurance RAG API",
    description="Assistant assurance basé sur RAG, embeddings, reranking et Llama",
    version=APP_VERSION,
)


@app.get("/version")
def version():
    """Expose la version de l’application pour vérifier l’instance réellement lancée."""
    return {"app_version": APP_VERSION, "ollama_model": OLLAMA_MODEL}

@app.get("/claims")
def claims_dashboard():
    """Sert le tableau de bord analytique de démonstration des sinistres fictifs."""
    if not CLAIMS_INDEX_FILE.is_file():
        raise HTTPException(status_code=503, detail="Le tableau de bord des sinistres est indisponible.")
    return FileResponse(CLAIMS_INDEX_FILE)


@app.get("/api/claims/summary")
def claims_summary():
    """Expose les indicateurs descriptifs du jeu de données fictif."""
    return get_claims_summary()


@app.get("/api/claims/cases")
def claims_cases(limit: int = 120):
    """Expose un extrait borné des dossiers fictifs pour la revue humaine."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="Le paramètre limit doit être compris entre 1 et 500.")
    return {"cases": get_claims_cases(limit), "dataset_label": "FICTIF_DEMONSTRATION"}


@app.get("/api/claims/model")
def claims_model_info():
    """Expose les métriques et limites du modèle de démonstration."""
    return get_claims_model_info()


@app.get("/api/claims/{claim_id}/analysis")
def claim_analysis(claim_id: str):
    """Fournit un score indicatif et les facteurs visibles d'un dossier fictif."""
    result = get_claim_analysis(claim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Aucun dossier de démonstration ne correspond à cet identifiant.")
    return result


@app.get("/health")
def health():
    """Vérifie que le processus API répond, sans déclencher de génération."""
    return {"status": "ok", "app_version": APP_VERSION}


@app.get("/ready")
def readiness():
    """Vérifie les ressources locales et la disponibilité du service Ollama."""
    checks = {
        "dataset_available": DATA_FILE.is_file() and not df.empty,
        "interface_available": INDEX_FILE.is_file(),
        "claims_dashboard_available": CLAIMS_INDEX_FILE.is_file(),
        "claims_dataset_available": (PROJECT_ROOT / "Data" / "synthetic_claims.csv").is_file(),
        "embeddings_loaded": document_embeddings is not None and len(document_embeddings) == len(documents),
        "reranker_loaded": reranker is not None,
    }
    ollama_url = OLLAMA_URL.rsplit("/api/generate", 1)[0].rstrip("/") + "/api/tags"
    try:
        ollama_response = requests.get(ollama_url, timeout=min(5, OLLAMA_TIMEOUT_SECONDS))
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
                    or any(name.split(":", 1)[0] == OLLAMA_MODEL.split(":", 1)[0] for name in model_names)
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
        "note": "Ce contrôle vérifie la disponibilité technique ; il ne valide pas la qualité métier des réponses.",
    }
    if not all_ready:
        raise HTTPException(status_code=503, detail=result)
    return result


# ============================================================
# MODELE DE REQUETE
# ============================================================

class Question(BaseModel):
    # Les espaces de début/fin sont retirés avant validation.
    # Une question vide ou trop longue est rejetée automatiquement par FastAPI.
    question: constr(strip_whitespace=True, min_length=1, max_length=2000)


# ============================================================
# PAGE D'ACCUEIL
# ============================================================

@app.get("/")
def home():

    return FileResponse(INDEX_FILE)


# ============================================================
# GENERATION AVEC LLAMA
# ============================================================

def generate_answer(question, results):
    # Les sorties brutes peuvent contenir des éléments issus de la question et du
    # corpus. Leur journalisation complète est donc désactivée par défaut et ne
    # s'active que pour un diagnostic local explicite.
    diagnostics_enabled = os.getenv("RAG_DIAGNOSTICS", "0").strip().lower() in {
        "1", "true", "yes", "oui"
    }

    def diagnostic(message):
        if diagnostics_enabled:
            print(f"[RAG_DIAGNOSTICS] {message}", flush=True)


    # --------------------------------------------------------
    # Construction du contexte
    # --------------------------------------------------------

    context_parts = []

    for result in results:
        # L'identifiant permet à la réponse de référencer explicitement une source.
        source_id = result.get("source_id", f"S{result.get('rank', len(context_parts) + 1)}")
        context_parts.append(
            f"SOURCE [{source_id}]\n"
            f"TITRE : {result['title']}\n"
            f"CONTENU : {result['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Appel Ollama et contrôle formel des citations
    # --------------------------------------------------------

    def call_ollama(current_prompt, attempt_label):
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": current_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        call_started = time.perf_counter()
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()

        # Une réponse HTTP 200 ne garantit pas que le modèle a renvoyé du texte.
        if not isinstance(data, dict):
            raise ValueError("Le JSON retourné par Ollama n'est pas un objet.")

        answer_text = data.get("response")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError("Le champ 'response' est absent ou vide.")
        answer_text = answer_text.strip()
        call_ms = round((time.perf_counter() - call_started) * 1000, 2)
        diagnostic(f"{attempt_label} — durée Ollama : {call_ms} ms")
        diagnostic(f"{attempt_label} — sortie brute début >>>\n{answer_text}\n<<< sortie brute fin")
        return answer_text

    try:
        answer = call_ollama(prompt, "tentative 1")

        # Le modèle peut ignorer la consigne de citation. Dans ce cas, on lui
        # demande une seule révision, en lui rappelant les identifiants permis.
        first_audit = audit_citations(answer, results)
        diagnostic(f"tentative 1 — audit : {first_audit}")
        if not first_audit["format_check_passed"]:
            available_ids = [
                str(result["source_id"])
                for result in results
                if result.get("source_id")
            ]
            # L'API /api/generate d'Ollama est sans état entre deux appels :
            # le modèle ne voit pas automatiquement le prompt du premier appel.
            # On réinjecte donc la question et les documents originaux dans la
            # seconde requête, ainsi que la réponse à corriger.
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
            revised_answer = call_ollama(retry_prompt, "tentative 2 / révision")
            revised_audit = audit_citations(revised_answer, results)
            diagnostic(f"tentative 2 / révision — audit : {revised_audit}")
            if revised_audit["format_check_passed"]:
                return revised_answer

            # Repli prudent : ne pas afficher comme réponse d'assurance un texte
            # dont les références restent absentes ou incohérentes après révision.
            print(
                "Réponse Ollama écartée : citations absentes ou inconnues "
                "après une tentative de correction."
            )
            # Repli additionnel et traçable : si le modèle échoue encore à citer,
            # présenter des extraits réellement récupérés plutôt qu'une réponse
            # générée sans références. Les extraits restent attribués à leur source.
            source_excerpts = []
            for result in results[:3]:
                fallback_source_id = str(result.get("source_id", "")).strip()
                fallback_title = str(result.get("title", "Document récupéré")).strip()
                fallback_content = str(result.get("content", "")).strip()
                if not fallback_source_id or not fallback_content:
                    continue
                excerpt_limit = 700
                if len(fallback_content) > excerpt_limit:
                    fallback_content = fallback_content[:excerpt_limit].rsplit(" ", 1)[0] + "…"
                source_excerpts.append(
                    f"- **{fallback_title}** [{fallback_source_id}] : {fallback_content}"
                )

            safe_fallback = (
                "Je ne peux pas formuler une synthèse suffisamment traçable : "
                "le modèle n’a pas fourni de citations vérifiables. Pour éviter "
                "d’affirmer un élément sans preuve, voici les extraits des documents "
                "récupérés, avec leurs identifiants de source. Leur pertinence doit "
                "être vérifiée au regard de votre question."
            )
            if source_excerpts:
                safe_fallback += "\n\n" + "\n\n".join(source_excerpts)
            else:
                safe_fallback += (
                    "\n\nAucun extrait exploitable avec identifiant de source "
                    "n’est disponible pour cette requête."
                )
            return safe_fallback

        return answer

    except requests.exceptions.RequestException as e:

        # Le détail reste dans les logs ; l'API renvoie un statut explicite.
        print(f"Erreur Ollama : {e}")
        raise HTTPException(
            status_code=502,
            detail="Le service de génération de réponse est indisponible. Vérifiez qu'Ollama est lancé et que le modèle est disponible."
        ) from e

    except (ValueError, TypeError) as e:

        # Couvre notamment une réponse JSON invalide ou un champ inattendu.
        print(f"Réponse Ollama invalide : {e}")
        raise HTTPException(
            status_code=502,
            detail="Le service de génération a renvoyé une réponse invalide."
        ) from e

# ============================================================
# CONTROLE DES REFERENCES AUX SOURCES
# ============================================================

_SOURCE_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def audit_citations(answer, results):
    """Repère les identifiants de source cités et ceux qui n'existent pas.

    Ce contrôle vérifie uniquement la cohérence formelle des identifiants.
    Il ne prouve pas que chaque affirmation est réellement justifiée par la source.
    """
    answer_text = answer if isinstance(answer, str) else ""
    available_ids = {
        str(result.get("source_id"))
        for result in results
        if result.get("source_id")
    }
    cited_ids = set(_SOURCE_CITATION_PATTERN.findall(answer_text))
    valid_ids = sorted(cited_ids & available_ids, key=lambda value: int(value[1:]))
    invalid_ids = sorted(cited_ids - available_ids, key=lambda value: int(value[1:]))
    uncited_ids = sorted(available_ids - cited_ids, key=lambda value: int(value[1:]))

    return {
        "citations_found": bool(cited_ids),
        "valid_source_ids": valid_ids,
        "unknown_source_ids": invalid_ids,
        "available_but_uncited_source_ids": uncited_ids,
        "format_check_passed": bool(cited_ids) and not invalid_ids,
        "scope": "Contrôle des identifiants uniquement ; la pertinence factuelle reste à vérifier.",
    }


# ============================================================
# RECHERCHE
# ============================================================

@app.post("/search")
def search(data: Question):

    request_started = time.perf_counter()
    question = data.question.strip()

    # --------------------------------------------------------
    # 1. Vérification
    # --------------------------------------------------------

    # Le modèle Question rejette déjà les valeurs vides ou trop longues.
    # Cette garde conserve une protection si search est appelé directement.
    if not question:
        raise HTTPException(
            status_code=422,
            detail="La question ne peut pas être vide."
        )

    # --------------------------------------------------------
    # 2. Embedding de la question
    # --------------------------------------------------------

    embedding_started = time.perf_counter()
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )
    embedding_ms = (time.perf_counter() - embedding_started) * 1000

    # --------------------------------------------------------
    # 3. Recherche initiale
    # --------------------------------------------------------

    retrieval_started = time.perf_counter()
    similarities = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    # --------------------------------------------------------
    # 4. Sélection hybride des candidats
    # --------------------------------------------------------

    # La fusion des rangs conserve la recherche sémantique et ajoute un signal
    # lexical afin de mieux retrouver les notions exactes, notamment les termes
    # de contrat comme « franchise absolue », « plafond » ou « exclusion ».
    candidate_indices, lexical_scores = _select_candidate_indices(
        question, similarities, candidate_limit=20
    )
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    pairs = [
        [question, documents[index]]
        for index in candidate_indices
    ]

    # --------------------------------------------------------
    # 5. Reranking
    # --------------------------------------------------------

    reranking_started = time.perf_counter()
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_indices, scores),
        key=lambda x: x[1],
        reverse=True
    )
    reranking_ms = (time.perf_counter() - reranking_started) * 1000

    # Fusion finale des rangs : le CrossEncoder reste le signal principal,
    # tandis que le rang lexical protège les notions métier explicites.
    # Cette fusion évite de comparer directement des scores de natures différentes.
    reranker_rank = {index: rank for rank, (index, _) in enumerate(reranked, start=1)}
    lexical_order = sorted(
        range(len(documents)),
        key=lambda index: lexical_scores[index],
        reverse=True,
    )
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
    fusion_constant = 60.0
    fused_reranked = sorted(
        reranked,
        key=lambda item: (
            _rank_fusion_score(
                reranker_rank[item[0]],
                lexical_rank[item[0]],
                lexical_scores[item[0]],
                fusion_constant,
            ),
            item[1],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # 6. Sélection des 5 meilleurs documents
    # --------------------------------------------------------

    results = []

    for rank, (index, reranker_score) in enumerate(
        fused_reranked[:5],
        start=1
    ):
        # Les deux scores restent distincts : la similarité d'embedding sert à
        # la présélection, tandis que le score du CrossEncoder sert au reranking.
        embedding_similarity = float(similarities[index])
        reranker_score = float(reranker_score)

        results.append({
            "rank": rank,
            "source_id": f"S{rank}",
            "title": titles[index],
            # Conservé pour compatibilité avec la version précédente de l'interface.
            "score": reranker_score,
            "reranker_score": reranker_score,
            "embedding_similarity": embedding_similarity,
            "lexical_relevance_score": float(lexical_scores[index]),
            "hybrid_rank_score": round(
                _rank_fusion_score(
                    reranker_rank[index],
                    lexical_rank[index],
                    lexical_scores[index],
                    fusion_constant,
                ),
                8,
            ),
            "content": documents[index][:1500]
        })

    # --------------------------------------------------------
    # 7. Génération de la réponse LLM
    # --------------------------------------------------------

    print("Génération de la réponse avec Llama...")

    generation_started = time.perf_counter()
    answer = generate_answer(
        question,
        results
    )
    generation_ms = (time.perf_counter() - generation_started) * 1000
    citation_audit = audit_citations(answer, results)
    total_ms = (time.perf_counter() - request_started) * 1000

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
            "note": "Durées observées pendant cette requête ; elles varient selon la machine, la charge et les modèles.",
        },
        "score_explanation": {
            "reranker_score": "Score brut du CrossEncoder utilisé pour classer les documents ; ce n'est pas une probabilité.",
            "embedding_similarity": "Similarité cosinus utilisée pour présélectionner les documents ; ce n'est pas une mesure de vérité.",
            "lexical_relevance_score": "Score lexical indicatif basé sur les termes de la question présents dans le titre et le contenu ; ce n'est pas une probabilité.",
            "hybrid_rank_score": "Score de fusion des rangs CrossEncoder et lexical ; il sert au classement final et ne mesure ni une probabilité ni la véracité."
        }
    }
