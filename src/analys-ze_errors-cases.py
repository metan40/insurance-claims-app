
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "Data/insurance-fr.csv"

df = pd.read_csv(DATA_FILE)

documents = df["content"].fillna("").tolist()
titles = df["title"].fillna("").tolist()

# ============================================================
# QUESTIONS A ANALYSER
# ============================================================

questions = [
    "Assurance location meublée : comment ça se passe ?",
    "L’assurance enfant incluse dans l’assurance habitation",
    "Les exclusions de la garantie dégât des eaux",
    "Changer d’assurance habitation : quelles sont les informations nécessaires ?",
    "Que couvre la responsabilité civile de l’assurance habitation ?"
]

# ============================================================
# TF-IDF
# ============================================================

tfidf_vectorizer = TfidfVectorizer(lowercase=True)

tfidf_document_vectors = tfidf_vectorizer.fit_transform(documents)

# ============================================================
# EMBEDDINGS
# ============================================================

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

embedding_document_vectors = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ============================================================
# ANALYSE
# ============================================================

for question in questions:

    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    # Document attendu
    expected_indices = [
        i for i, title in enumerate(titles)
        if title == question
    ]

    if not expected_indices:
        print("\nDocument attendu introuvable.")
        continue

    expected_index = expected_indices[0]

    print("\n")
    print("-" * 80)
    print("DOCUMENT ATTENDU")
    print("-" * 80)

    print(f"Index : {expected_index}")
    print(f"Titre : {titles[expected_index]}")
    print()
    print(documents[expected_index][:1500])

    # ========================================================
    # TF-IDF
    # ========================================================

    question_vector = tfidf_vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        tfidf_document_vectors
    )[0]

    tfidf_indices = similarities.argsort()[::-1][:5]

    print("\n")
    print("=" * 80)
    print("TOP-5 TF-IDF")
    print("=" * 80)

    for position, index in enumerate(tfidf_indices, start=1):

        print("\n")
        print(f"RANG {position}")
        print(f"Index : {index}")
        print(f"Score : {similarities[index]:.4f}")
        print(f"Titre : {titles[index]}")
        print()
        print(documents[index][:800])

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        question_embedding,
        embedding_document_vectors
    )[0]

    embedding_indices = similarities.argsort()[::-1][:5]

    print("\n")
    print("=" * 80)
    print("TOP-5 EMBEDDINGS")
    print("=" * 80)

    for position, index in enumerate(embedding_indices, start=1):

        print("\n")
        print(f"RANG {position}")
        print(f"Index : {index}")
        print(f"Score : {similarities[index]:.4f}")
        print(f"Titre : {titles[index]}")
        print()
        print(documents[index][:800])
