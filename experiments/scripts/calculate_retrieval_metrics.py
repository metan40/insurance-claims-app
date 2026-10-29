import pandas as pd

FILE = "Data/retrieval_optimization_detailed.csv"

df = pd.read_csv(FILE)

df["embedding_rank"] = pd.to_numeric(
    df["embedding_rank"],
    errors="coerce"
)

df["optimized_rank"] = pd.to_numeric(
    df["optimized_rank"],
    errors="coerce"
)

def recall_at_k(ranks, k):
    ranks = ranks.dropna()
    return (ranks <= k).mean() * 100

def mrr(ranks):
    ranks = ranks.dropna()
    return (1 / ranks).mean()

embedding_ranks = df["embedding_rank"]
optimized_ranks = df["optimized_rank"]

print()
print("=" * 60)
print("METRIQUES RETRIEVAL")
print("=" * 60)

print()
print("EMBEDDINGS")

print(f"Recall@1 : {recall_at_k(embedding_ranks, 1):.2f} %")
print(f"Recall@3 : {recall_at_k(embedding_ranks, 3):.2f} %")
print(f"Recall@5 : {recall_at_k(embedding_ranks, 5):.2f} %")
print(f"MRR      : {mrr(embedding_ranks):.4f}")

print()
print("EMBEDDINGS + RERANKER")

print(f"Recall@1 : {recall_at_k(optimized_ranks, 1):.2f} %")
print(f"Recall@3 : {recall_at_k(optimized_ranks, 3):.2f} %")
print(f"Recall@5 : {recall_at_k(optimized_ranks, 5):.2f} %")
print(f"MRR      : {mrr(optimized_ranks):.4f}")

print()
print("=" * 60)
print("AMELIORATION")
print("=" * 60)

for k in [1, 3, 5]:

    before = recall_at_k(embedding_ranks, k)
    after = recall_at_k(optimized_ranks, k)

    print(
        f"Recall@{k} : "
        f"{after - before:+.2f} points"
    )

mrr_before = mrr(embedding_ranks)
mrr_after = mrr(optimized_ranks)

print(
    f"MRR       : "
    f"{mrr_after - mrr_before:+.4f}"
)

print()
print("=" * 60)
