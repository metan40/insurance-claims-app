
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "Data/insurance-fr.csv"
OUTPUT_FILE = "Data/retrieval_failures.csv"

TOP_K = 10

df = pd.read_csv(DATA_FILE)

documents = df["content"].fillna("").tolist()
titles = df["title"].fillna("").tolist()

print("Chargement du modèle...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Calcul des embeddings...")

document_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

rows = []

for i, question in enumerate(titles):

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    rank = list(ranked_indices).index(i) + 1

    if rank <= TOP_K:
        continue

    top5_indices = ranked_indices[:5]

    top5_titles = [
        titles[index]
        for index in top5_indices
    ]

    top5_scores = [
        round(float(similarities[index]), 4)
        for index in top5_indices
    ]

    rows.append({
        "question": question,
        "expected_document": i,
        "expected_title": titles[i],
        "embedding_rank": rank,
        "expected_score": round(
            float(similarities[i]),
            4
        ),
        "top1_title": top5_titles[0],
        "top1_score": top5_scores[0],
        "top2_title": top5_titles[1],
        "top2_score": top5_scores[1],
        "top3_title": top5_titles[2],
        "top3_score": top5_scores[2],
        "top4_title": top5_titles[3],
        "top4_score": top5_scores[3],
        "top5_title": top5_titles[4],
        "top5_score": top5_scores[4]
    })

results = pd.DataFrame(rows)

results = results.sort_values(
    by="embedding_rank"
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 60)
print("ANALYSE DES ECHECS DU RETRIEVAL")
print("=" * 60)

print()
print(f"Nombre total de questions : {len(df)}")

print(
    f"Questions hors Top-{TOP_K} : "
    f"{len(results)}"
)

print()
print(
    f"Fichier créé : {OUTPUT_FILE}"
)

print()
print("Premiers cas à examiner :")

print(
    results[
        [
            "question",
            "expected_title",
            "embedding_rank",
            "expected_score",
            "top1_title",
            "top1_score"
        ]
    ].head(10).to_string(index=False)
)
