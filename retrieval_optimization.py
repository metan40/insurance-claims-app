import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "Data/insurance-fr.csv"
OUTPUT_FILE = "Data/retrieval_optimization_detailed.csv"

TOP_K_CANDIDATES = 20

df = pd.read_csv(DATA_FILE)

documents = df["content"].fillna("").tolist()
questions = df["title"].fillna("").tolist()

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

rows = []

for i, question in enumerate(questions):

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    embedding_rank = list(ranked_indices).index(i) + 1

    candidate_indices = ranked_indices[:TOP_K_CANDIDATES]

    candidate_pairs = [
        [question, documents[index]]
        for index in candidate_indices
    ]

    reranker_scores = reranker.predict(candidate_pairs)

    reranked = sorted(
        zip(candidate_indices, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_indices = [
        index for index, score in reranked
    ]

    if i in candidate_indices:

        optimized_rank = reranked_indices.index(i) + 1

        if optimized_rank < embedding_rank:
            situation = "Amélioration"
        elif optimized_rank > embedding_rank:
            situation = "Dégradation"
        else:
            situation = "Même rang"

    else:

        optimized_rank = None
        situation = "Absent du Top-K"

    rows.append({
        "question": question,
        "expected_document": i,
        "expected_title": df.iloc[i]["title"],
        "embedding_rank": embedding_rank,
        "optimized_rank": optimized_rank,
        "embedding_top5": embedding_rank <= 5,
        "optimized_top5": (
            optimized_rank is not None
            and optimized_rank <= 5
        ),
        "retrieved_in_top20": i in candidate_indices,
        "situation": situation
    })

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

def recall_at_k(ranks, k):
    return sum(
        rank is not None and rank <= k
        for rank in ranks
    ) / len(ranks) * 100

def mrr(ranks):
    return sum(
        1 / rank if rank is not None else 0
        for rank in ranks
    ) / len(ranks)

embedding_ranks = results["embedding_rank"].tolist()
optimized_ranks = results["optimized_rank"].tolist()

print("\n======================================")
print("RETRIEVAL OPTIMIZATION")
print("======================================")

print(f"\nNombre de questions : {len(results)}")
print(f"Nombre de candidats : {TOP_K_CANDIDATES}")

print("\n--------------------------------------")
print("EMBEDDINGS")
print("--------------------------------------")

print(
    f"Recall@1 : "
    f"{recall_at_k(embedding_ranks, 1):.2f} %"
)

print(
    f"Recall@3 : "
    f"{recall_at_k(embedding_ranks, 3):.2f} %"
)

print(
    f"Recall@5 : "
    f"{recall_at_k(embedding_ranks, 5):.2f} %"
)

print(
    f"MRR      : "
    f"{mrr(embedding_ranks):.4f}"
)

print("\n--------------------------------------")
print("EMBEDDINGS + RERANKING")
print("--------------------------------------")

print(
    f"Recall@1 : "
    f"{recall_at_k(optimized_ranks, 1):.2f} %"
)

print(
    f"Recall@3 : "
    f"{recall_at_k(optimized_ranks, 3):.2f} %"
)

print(
    f"Recall@5 : "
    f"{recall_at_k(optimized_ranks, 5):.2f} %"
)

print(
    f"MRR      : "
    f"{mrr(optimized_ranks):.4f}"
)

print("\n--------------------------------------")
print("COUVERTURE DU TOP-20")
print("--------------------------------------")

retrieved_count = results["retrieved_in_top20"].sum()

print(
    f"Document attendu présent dans le Top-20 : "
    f"{retrieved_count}/{len(results)} "
    f"({retrieved_count / len(results) * 100:.2f} %)"
)

print("\n--------------------------------------")
print("EFFET DU RERANKING")
print("--------------------------------------")

print(
    results["situation"]
    .value_counts()
    .to_string()
)

print("\n--------------------------------------")
print("TOP-5")
print("--------------------------------------")

print(
    pd.crosstab(
        results["embedding_top5"],
        results["optimized_top5"],
        rownames=["Embeddings"],
        colnames=["Réévaluation"]
    )
)

print("\n======================================")
print(f"Fichier créé : {OUTPUT_FILE}")
print("========================================")