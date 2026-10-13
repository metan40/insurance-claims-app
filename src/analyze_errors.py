import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Charger le dataset
df = pd.read_csv("Data/insurance-fr.csv")

documents = df["content"].fillna("")

# 2. TF-IDF
vectorizer = TfidfVectorizer(lowercase=True)
document_vectors = vectorizer.fit_transform(documents)

print("\n======================================")
print("ANALYSE DES ERREURS")
print("======================================")

erreurs = 0

# 3. Tester chaque question
for i in range(len(df)):

    question = df.iloc[i]["title"]

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    top5 = ranked_indices[:5]

    # Le document attendu n'est pas dans le Top-5
    if i not in top5:

        erreurs += 1

        print("\n--------------------------------------")
        print(f"Question : {question}")
        print(
            f"Document attendu : "
            f"{i + 1} - {df.iloc[i]['title']}"
        )

        print("\nDocuments retrouvés :")

        for position, index in enumerate(top5, start=1):

            print(
                f"{position}. "
                f"{df.iloc[index]['title']} "
                f"(score = {similarities[index]:.4f})"
            )

            print(
                "   Contenu :",
                df.iloc[index]["content"][:300]
                .replace("\n", " ")
            )

        # Afficher seulement les 10 premières erreurs
        if erreurs >= 10:
            break


print("\n======================================")
print(f"Nombre d'erreurs affichées : {erreurs}")
print("======================================")
