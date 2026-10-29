import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Data/insurance-fr.csv")

documents = df["content"].fillna("").tolist()
questions = df["title"].fillna("").tolist()

# ==============================
# TF-IDF
# ==============================

tfidf_vectorizer = TfidfVectorizer(lowercase=True)

tfidf_document_vectors = tfidf_vectorizer.fit_transform(documents)

tfidf_ranks = []

for i, question in enumerate(questions):

    question_vector = tfidf_vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        tfidf_document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    rank = list(ranked_indices).index(i) + 1

    tfidf_ranks.append(rank)

# ==============================
# EMBEDDINGS
# ==============================

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

embedding_document_vectors = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

embedding_ranks = []

for i, question in enumerate(questions):

    question_vector = model.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        question_vector,
        embedding_document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    rank = list(ranked_indices).index(i) + 1

    embedding_ranks.append(rank)

# ==============================
# CREATION DU TABLEAU
# ==============================

results = pd.DataFrame({
    "question": questions,
    "expected_document": range(len(df)),
    "tfidf_rank": tfidf_ranks,
    "embedding_rank": embedding_ranks
})

# ==============================
# METRIQUES
# ==============================

def recall_at_k(ranks, k):
    return sum(rank <= k for rank in ranks) / len(ranks) * 100

def mrr(ranks):
    return sum(1 / rank for rank in ranks) / len(ranks)

print("\n======================================")
print("COMPARAISON TF-IDF / EMBEDDINGS")
print("======================================")

print(f"\nNombre de questions : {len(df)}")

print("\nTF-IDF")
print(f"Recall@1 : {recall_at_k(tfidf_ranks, 1):.2f} %")
print(f"Recall@3 : {recall_at_k(tfidf_ranks, 3):.2f} %")
print(f"Recall@5 : {recall_at_k(tfidf_ranks, 5):.2f} %")
print(f"MRR      : {mrr(tfidf_ranks):.4f}")

print("\nEMBEDDINGS")
print(f"Recall@1 : {recall_at_k(embedding_ranks, 1):.2f} %")
print(f"Recall@3 : {recall_at_k(embedding_ranks, 3):.2f} %")
print(f"Recall@5 : {recall_at_k(embedding_ranks, 5):.2f} %")
print(f"MRR      : {mrr(embedding_ranks):.4f}")

# ==============================
# SAUVEGARDE
# ==============================

results.to_csv(
    "Data/retrieval_comparison.csv",
    index=False,
    encoding="utf-8"
)

print("\nFichier créé : Data/retrieval_comparison.csv")
