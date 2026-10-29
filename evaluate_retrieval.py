import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Charger le dataset
df = pd.read_csv("Data/insurance-fr.csv")

# 2. Récupérer les contenus
documents = df["content"].fillna("")

# 3. Construire la représentation TF-IDF
vectorizer = TfidfVectorizer(lowercase=True)
document_vectors = vectorizer.fit_transform(documents)


# Compteurs
top1 = 0
top3 = 0
top5 = 0


# 4. Tester chaque question
for i in range(len(df)):

    question = df.iloc[i]["title"]

    # Transformer la question
    question_vector = vectorizer.transform([question])

    # Calculer la similarité avec tous les documents
    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    # Classer les documents du plus proche au moins proche
    ranked_indices = similarities.argsort()[::-1]

    # Positions des 3 premiers niveaux
    top1_indices = ranked_indices[:1]
    top3_indices = ranked_indices[:3]
    top5_indices = ranked_indices[:5]

    # Le document attendu est celui de la même ligne
    if i in top1_indices:
        top1 += 1

    if i in top3_indices:
        top3 += 1

    if i in top5_indices:
        top5 += 1


# 5. Afficher les résultats
total = len(df)

print("\n======================================")
print("ÉVALUATION DU RETRIEVAL")
print("======================================")

print(f"\nNombre total de questions : {total}")

print(
    f"Top-1 : {top1}/{total} "
    f"({top1 / total * 100:.2f} %)"
)

print(
    f"Top-3 : {top3}/{total} "
    f"({top3 / total * 100:.2f} %)"
)

print(
    f"Top-5 : {top5}/{total} "
    f"({top5 / total * 100:.2f} %)"
)
