import pandas as pd

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "Data/insurance-fr.csv"

EMBEDDING_MODEL = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

TOP_K_CANDIDATES = 20


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

print("\nChargement des données...")

df = pd.read_csv(DATA_FILE)

documents = df["content"].fillna("").tolist()
questions = df["title"].fillna("").tolist()


print(f"Nombre de documents : {len(documents)}")


# ============================================================
# MODELE D'EMBEDDINGS
# ============================================================

print("\nChargement du modèle d'embeddings...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# EMBEDDINGS DES DOCUMENTS
# ============================================================

print("\nCalcul des embeddings des documents...")

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# MODELE DE RERANKING
# ============================================================

print("\nChargement du CrossEncoder...")

reranker = CrossEncoder(
    RERANKER_MODEL
)


# ============================================================
# STOCKAGE DES RANGS
# ============================================================

embedding_ranks = []
reranker_ranks = []


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluation du retrieval...\n")


for i, question in enumerate(questions):

    print(
        f"Question {i + 1}/{len(questions)}",
        end="\r"
    )

    # --------------------------------------------------------
    # 1. Embedding de la question
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # 2. Similarité embeddings
    # --------------------------------------------------------

    similarities = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    # --------------------------------------------------------
    # 3. Rang obtenu avec les embeddings seuls
    # --------------------------------------------------------

    embedding_rank = (
        list(ranked_indices).index(i) + 1
    )

    embedding_ranks.append(
        embedding_rank
    )

    # --------------------------------------------------------
    # 4. Sélection des Top-20
    # --------------------------------------------------------

    candidate_indices = ranked_indices[
        :TOP_K_CANDIDATES
    ]

    # --------------------------------------------------------
    # 5. Construction des paires pour CrossEncoder
    # --------------------------------------------------------

    pairs = [
        [question, documents[index]]
        for index in candidate_indices
    ]

    # --------------------------------------------------------
    # 6. Reranking
    # --------------------------------------------------------

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_indices, scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_indices = [
        index
        for index, score in reranked
    ]

    # --------------------------------------------------------
    # 7. Rang après reranking
    # --------------------------------------------------------

    if i in reranked_indices:

        reranker_rank = (
            reranked_indices.index(i) + 1
        )

    else:

        # Le document attendu n'était pas dans le Top-20
        # initial. Il ne peut donc pas être récupéré
        # par le reranker.

        reranker_rank = TOP_K_CANDIDATES + 1

    reranker_ranks.append(
        reranker_rank
    )


# ============================================================
# METRIQUES
# ============================================================

def recall_at_k(ranks, k):

    return (
        sum(rank <= k for rank in ranks)
        / len(ranks)
        * 100
    )


def mrr(ranks):

    return (
        sum(1 / rank for rank in ranks)
        / len(ranks)
    )


# ============================================================
# RESULTATS
# ============================================================

total = len(df)


print("\n\n======================================")
print("EVALUATION DU RETRIEVAL")
print("======================================")

print(
    f"\nNombre total de questions : {total}"
)


# ============================================================
# EMBEDDINGS
# ============================================================

print("\n--------------------------------------")
print("EMBEDDINGS SEULS")
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


# ============================================================
# EMBEDDINGS + RERANKER
# ============================================================

print("\n--------------------------------------")
print("EMBEDDINGS + CROSSENCODER")
print("--------------------------------------")

print(
    f"Recall@1 : "
    f"{recall_at_k(reranker_ranks, 1):.2f} %"
)

print(
    f"Recall@3 : "
    f"{recall_at_k(reranker_ranks, 3):.2f} %"
)

print(
    f"Recall@5 : "
    f"{recall_at_k(reranker_ranks, 5):.2f} %"
)

print(
    f"MRR      : "
    f"{mrr(reranker_ranks):.4f}"
)


# ============================================================
# AMELIORATION
# ============================================================

print("\n--------------------------------------")
print("IMPACT DU RERANKING")
print("--------------------------------------")

for k in [1, 3, 5]:

    embedding_score = recall_at_k(
        embedding_ranks,
        k
    )

    reranker_score = recall_at_k(
        reranker_ranks,
        k
    )

    improvement = (
        reranker_score
        - embedding_score
    )

    print(
        f"Recall@{k} : "
        f"{improvement:+.2f} points"
    )


print(
    f"MRR : "
    f"{mrr(reranker_ranks) - mrr(embedding_ranks):+.4f}"
)


print("\n======================================")
print("FIN DE L'EVALUATION")
print("======================================")