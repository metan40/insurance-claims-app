
import pandas as pd
from pathlib import Path

RESULTS_FILE = "retrieval_comparison.csv"
DATA_FILE = "Data/insurance-fr.csv"

OUTPUT_DIR = Path("Data")
OUTPUT_FILE = OUTPUT_DIR / "common_failures.csv"

results = pd.read_csv(RESULTS_FILE)
data = pd.read_csv(DATA_FILE)

common_failures = results[
    (results["tfidf_rank"] > 5) &
    (results["embedding_rank"] > 5)
].copy()

common_failures["expected_title"] = common_failures[
    "expected_document"
].apply(
    lambda i: data.iloc[int(i)]["title"]
)

common_failures = common_failures[
    [
        "question",
        "expected_document",
        "expected_title",
        "tfidf_rank",
        "embedding_rank"
    ]
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

common_failures.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("======================================")
print("ERREURS COMMUNES")
print("======================================")

print(
    f"\nNombre de questions où "
    f"TF-IDF et embeddings échouent au Top-5 : "
    f"{len(common_failures)}"
)

print(f"\nFichier créé : {OUTPUT_FILE}")
