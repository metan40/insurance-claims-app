
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "Data/insurance-fr.csv"
OUTPUT_FILE = Path("Data/error_cases_report.html")

df = pd.read_csv(DATA_FILE)

documents = df["content"].fillna("").tolist()
titles = df["title"].fillna("").tolist()

questions = [
    "Assurance location meublée : comment ça se passe ?",
    "L’assurance enfant incluse dans l’assurance habitation",
    "Les exclusions de la garantie dégât des eaux",
    "Changer d’assurance habitation : quelles sont les informations nécessaires ?",
    "Que couvre la responsabilité civile de l’assurance habitation ?"
]

tfidf_vectorizer = TfidfVectorizer(lowercase=True)
tfidf_document_vectors = tfidf_vectorizer.fit_transform(documents)

print("Chargement du modèle d'embeddings...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Calcul des embeddings...")

embedding_document_vectors = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

def escape_html(text):

    if pd.isna(text):
        return ""

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

html_cases = ""

for case_number, question in enumerate(questions, start=1):

    expected_indices = [
        i for i, title in enumerate(titles)
        if title == question
    ]

    if not expected_indices:
        continue

    expected_index = expected_indices[0]

    # TF-IDF
    question_vector = tfidf_vectorizer.transform([question])

    tfidf_similarities = cosine_similarity(
        question_vector,
        tfidf_document_vectors
    )[0]

    tfidf_indices = tfidf_similarities.argsort()[::-1][:5]

    # Embeddings
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    embedding_similarities = cosine_similarity(
        question_embedding,
        embedding_document_vectors
    )[0]

    embedding_indices = embedding_similarities.argsort()[::-1][:5]

    html_cases += f"""
    <section class="case">

        <h2>Cas {case_number}</h2>

        <div class="question">
            <h3>Question</h3>
            <p>{escape_html(question)}</p>
        </div>

        <div class="expected">
            <h3>Document attendu</h3>

            <p><strong>Index :</strong> {expected_index}</p>

            <p>
                <strong>Titre :</strong>
                {escape_html(titles[expected_index])}
            </p>

            <div class="content">
                {escape_html(documents[expected_index][:2000])}
            </div>
        </div>

        <div class="method">

            <h3>Top-5 TF-IDF</h3>

            <table>

                <tr>
                    <th>Rang</th>
                    <th>Index</th>
                    <th>Score</th>
                    <th>Titre</th>
                    <th>Contenu</th>
                </tr>
    """

    for position, index in enumerate(tfidf_indices, start=1):

        css_class = ""

        if index == expected_index:
            css_class = "expected-row"

        html_cases += f"""
                <tr class="{css_class}">

                    <td>{position}</td>
                    <td>{index}</td>
                    <td>{tfidf_similarities[index]:.4f}</td>

                    <td>
                        {escape_html(titles[index])}
                    </td>

                    <td>
                        {escape_html(documents[index][:1000])}
                    </td>

                </tr>
        """

    html_cases += """
            </table>

        </div>

        <div class="method">

            <h3>Top-5 Embeddings</h3>

            <table>

                <tr>
                    <th>Rang</th>
                    <th>Index</th>
                    <th>Score</th>
                    <th>Titre</th>
                    <th>Contenu</th>
                </tr>
    """

    for position, index in enumerate(
        embedding_indices,
        start=1
    ):

        css_class = ""

        if index == expected_index:
            css_class = "expected-row"

        html_cases += f"""
                <tr class="{css_class}">

                    <td>{position}</td>
                    <td>{index}</td>
                    <td>{embedding_similarities[index]:.4f}</td>

                    <td>
                        {escape_html(titles[index])}
                    </td>

                    <td>
                        {escape_html(documents[index][:1000])}
                    </td>

                </tr>
        """

    html_cases += """
            </table>

        </div>

    </section>
    """

html = f"""
<!DOCTYPE html>

<html lang="fr">

<head>

<meta charset="UTF-8">

<title>
Analyse des erreurs de retrieval
</title>

<style>

body {{
    font-family: Arial, Helvetica, sans-serif;
    background: #f5f7fa;
    color: #222;
    margin: 40px;
}}

h1 {{
    color: #17365d;
}}

h2 {{
    color: #244a73;
    border-bottom: 2px solid #244a73;
    padding-bottom: 8px;
}}

h3 {{
    color: #333;
}}

.case {{
    background: white;
    padding: 25px;
    margin-bottom: 40px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}}

.question {{
    background: #eaf2f8;
    padding: 15px;
    border-left: 5px solid #3498db;
}}

.expected {{
    background: #fff8dc;
    padding: 15px;
    margin-top: 20px;
    border-left: 5px solid #f1c40f;
}}

.method {{
    margin-top: 25px;
}}

.content {{
    white-space: pre-wrap;
    background: #fafafa;
    padding: 12px;
    margin-top: 10px;
    border: 1px solid #ddd;
    max-height: 350px;
    overflow-y: auto;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
}}

th {{
    background: #17365d;
    color: white;
    padding: 10px;
    text-align: left;
}}

td {{
    border: 1px solid #ddd;
    padding: 8px;
    vertical-align: top;
}}

tr:nth-child(even) {{
    background: #f8f8f8;
}}

.expected-row {{
    background: #d5f5e3 !important;
    font-weight: bold;
}}

.note {{
    background: #fff3cd;
    padding: 15px;
    border-left: 5px solid #ffc107;
    margin-bottom: 30px;
}}

</style>

</head>

<body>

<h1>
Analyse détaillée des erreurs de retrieval
</h1>

<p>
Dataset :
<strong>insurance-fr.csv</strong>
</p>

<p>
Méthodes comparées :
<strong>TF-IDF</strong> et
<strong>Embeddings</strong>
</p>

<div class="note">

<strong>Objectif :</strong>
examiner les documents réellement récupérés
afin de comprendre pourquoi certaines questions
ne retrouvent pas le document attendu dans le Top-5.

<p>
Les interprétations ne sont pas encore définitives.
Elles seront révisées après examen des cas.
</p>

</div>

{html_cases}

</body>

</html>
"""

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE.write_text(
    html,
    encoding="utf-8"
)

print()
print("=" * 60)
print("RAPPORT TERMINÉ")
print("=" * 60)

print()
print(f"Fichier créé : {OUTPUT_FILE}")

print()
print("Ouvre ce fichier dans ton navigateur.")
