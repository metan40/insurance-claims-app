"""Compare TF-IDF, TF-IDF+SVD (LSA), embeddings, and rank fusion.

Usage from project root:
    python src/compare_tfidf_svd_embeddings.py
    python src/compare_tfidf_svd_embeddings.py --with-embeddings

Important: the bundled dataset's title is used as the query and the document
from the same row is treated as the expected target. This is a reproducible
proxy evaluation, not human relevance annotation and not an end-to-end RAG test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "insurance-fr.csv"
OUT_CSV = ROOT / "Data" / "tfidf_svd_embeddings_comparison.csv"
OUT_JSON = ROOT / "Data" / "tfidf_svd_embeddings_comparison_summary.json"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60


def ranking(scores: np.ndarray) -> List[int]:
    # Stable tie handling makes results reproducible.
    return np.argsort(-scores, kind="stable").tolist()


def reciprocal_rank_fusion(rankings: List[List[int]], n_docs: int, k: int = RRF_K) -> List[int]:
    scores = np.zeros(n_docs, dtype=float)
    for ranked in rankings:
        for position, doc_idx in enumerate(ranked, start=1):
            scores[doc_idx] += 1.0 / (k + position)
    return ranking(scores)


def metrics(ranked_indices: List[int], expected_idx: int) -> Dict[str, float]:
    rank = ranked_indices.index(expected_idx) + 1 if expected_idx in ranked_indices else len(ranked_indices) + 1
    return {
        "expected_rank": rank,
        "recall_at_1": float(rank <= 1),
        "recall_at_3": float(rank <= 3),
        "recall_at_5": float(rank <= 5),
        "recall_at_20": float(rank <= 20),
        "reciprocal_rank": 1.0 / rank,
    }


def aggregate(rows: pd.DataFrame) -> Dict[str, float]:
    return {
        "questions": int(len(rows)),
        "Recall@1_pct": round(100 * rows["recall_at_1"].mean(), 2),
        "Recall@3_pct": round(100 * rows["recall_at_3"].mean(), 2),
        "Recall@5_pct": round(100 * rows["recall_at_5"].mean(), 2),
        "Recall@20_pct": round(100 * rows["recall_at_20"].mean(), 2),
        "MRR": round(float(rows["reciprocal_rank"].mean()), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-embeddings", action="store_true", help="also run the SentenceTransformer baseline (may download model weights)")
    parser.add_argument("--components", default="50,100,200", help="comma-separated SVD dimensions to test")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_FILE}")
    df = pd.read_csv(DATA_FILE).fillna("")
    if not {"title", "content"}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes 'title' et 'content'.")
    questions = df["title"].astype(str).tolist()
    documents = df["content"].astype(str).tolist()
    n_docs = len(documents)
    if n_docs < 3:
        raise ValueError("Le corpus doit contenir au moins trois documents.")

    print(f"Corpus : {n_docs} documents/questions")
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
    doc_tfidf = vectorizer.fit_transform(documents)
    query_tfidf = vectorizer.transform(questions)
    tfidf_scores = cosine_similarity(query_tfidf, doc_tfidf)
    tfidf_rankings = [ranking(row) for row in tfidf_scores]

    result_rows = []
    for i, ranked in enumerate(tfidf_rankings):
        m = metrics(ranked, i)
        result_rows.append({"question_index": i, "question": questions[i], "method": "TF-IDF", **m})

    summary: Dict[str, Dict[str, float]] = {"TF-IDF": aggregate(pd.DataFrame([r for r in result_rows if r["method"] == "TF-IDF"]))}
    dims = []
    for item in args.components.split(","):
        try:
            dim = int(item.strip())
        except ValueError:
            raise ValueError(f"Dimension SVD invalide : {item}")
        if dim > 0 and dim < min(doc_tfidf.shape) and dim not in dims:
            dims.append(dim)
    if not dims:
        raise ValueError(f"Aucune dimension SVD valide. Matrice TF-IDF : {doc_tfidf.shape}; choisissez une dimension positive inférieure à {min(doc_tfidf.shape)}.")

    svd_rankings_by_dim = {}
    for dim in dims:
        print(f"Calcul SVD, composantes={dim}...")
        svd = TruncatedSVD(n_components=dim, random_state=42, n_iter=7)
        doc_latent = normalize(svd.fit_transform(doc_tfidf))
        query_latent = normalize(svd.transform(query_tfidf))
        scores = cosine_similarity(query_latent, doc_latent)
        method = f"TF-IDF+SVD-{dim}"
        ranks = [ranking(row) for row in scores]
        svd_rankings_by_dim[dim] = ranks
        for i, ranked in enumerate(ranks):
            result_rows.append({"question_index": i, "question": questions[i], "method": method, **metrics(ranked, i)})
        summary[method] = aggregate(pd.DataFrame([r for r in result_rows if r["method"] == method]))
        summary[method]["explained_variance_ratio_sum_pct"] = round(float(svd.explained_variance_ratio_.sum() * 100), 2)
        fused = [reciprocal_rank_fusion([tfidf_rankings[i], ranks[i]], n_docs) for i in range(n_docs)]
        fused_method = f"RRF(TF-IDF,SVD-{dim})"
        for i, ranked in enumerate(fused):
            result_rows.append({"question_index": i, "question": questions[i], "method": fused_method, **metrics(ranked, i)})
        summary[fused_method] = aggregate(pd.DataFrame([r for r in result_rows if r["method"] == fused_method]))

    if args.with_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers n'est pas installé. Installez les dépendances puis relancez.") from exc
        print(f"Encodage embeddings avec {EMBEDDING_MODEL} (téléchargement possible au premier lancement)...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        doc_embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True)
        query_embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)
        emb_scores = cosine_similarity(query_embeddings, doc_embeddings)
        emb_rankings = [ranking(row) for row in emb_scores]
        for i, ranked in enumerate(emb_rankings):
            result_rows.append({"question_index": i, "question": questions[i], "method": "Embeddings", **metrics(ranked, i)})
        summary["Embeddings"] = aggregate(pd.DataFrame([r for r in result_rows if r["method"] == "Embeddings"]))
        for dim in dims:
            svd_rankings = svd_rankings_by_dim[dim]
            fused = [reciprocal_rank_fusion([svd_rankings[i], emb_rankings[i]], n_docs) for i in range(n_docs)]
            fused_method = f"RRF(SVD-{dim},Embeddings)"
            for i, ranked in enumerate(fused):
                result_rows.append({"question_index": i, "question": questions[i], "method": fused_method, **metrics(ranked, i)})
            summary[fused_method] = aggregate(pd.DataFrame([r for r in result_rows if r["method"] == fused_method]))

    detail = pd.DataFrame(result_rows)
    detail.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "n_documents": n_docs,
        "query_target_protocol": "Chaque titre de ligne sert de requête et le document de la même ligne est la cible attendue.",
        "limitations": [
            "La cible de même ligne est un proxy automatique, pas une annotation humaine de pertinence.",
            "Les titres sont issus du même corpus; cette évaluation ne représente pas nécessairement des questions utilisateur réelles.",
            "Les résultats ne mesurent ni la qualité des réponses générées ni la fidélité des citations.",
            "Les résultats SVD dépendent du nombre de composantes et du corpus; variance expliquée et scores doivent être interprétés ensemble."
        ],
        "results": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRésultats (pourcentage sauf MRR et variance) :")
    for method, values in summary.items():
        print(f"- {method}: Recall@1={values['Recall@1_pct']}%, Recall@3={values['Recall@3_pct']}%, Recall@5={values['Recall@5_pct']}%, MRR={values['MRR']}")
    print(f"\nDétails : {OUT_CSV.relative_to(ROOT)}\nSynthèse : {OUT_JSON.relative_to(ROOT)}")
    print("ATTENTION : ces résultats utilisent la cible de même ligne comme proxy; ils ne constituent pas une validation métier annotée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
