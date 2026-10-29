import pandas as pd
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = Path("Data/insurance-fr.csv")
COMPARISON_FILE = Path("Data/retrieval_comparison.csv")

OUTPUT_DIR = Path("Data")
OUTPUT_ERRORS = OUTPUT_DIR / "retrieval_error_analysis.csv"
OUTPUT_FAILURES = OUTPUT_DIR / "retrieval_both_failures.csv"

# ============================================================
# 2. CHARGEMENT DES DONNEES
# ============================================================

dataset = pd.read_csv(DATA_FILE)
comparison = pd.read_csv(COMPARISON_FILE)

if len(dataset) != len(comparison):
    raise ValueError(
        f"Nombre de lignes différent : "
        f"dataset={len(dataset)}, comparison={len(comparison)}"
    )

required_columns = {
    "question",
    "tfidf_rank",
    "embedding_rank"
}

missing = required_columns - set(comparison.columns)

if missing:
    raise ValueError(
        f"Colonnes manquantes dans le fichier comparatif : {missing}"
    )

# Association des documents aux résultats.
# Cette association suppose que les lignes sont dans le même ordre.
comparison["document_title"] = dataset["title"].fillna("")
comparison["document_content"] = dataset["content"].fillna("")

# ============================================================
# 3. IDENTIFICATION DE LA MEILLEURE METHODE
# ============================================================

def determine_method(row):
    if row["embedding_rank"] < row["tfidf_rank"]:
        return "Embeddings"
    elif row["tfidf_rank"] < row["embedding_rank"]:
        return "TF-IDF"
    return "Égalité"

comparison["meilleure_methode"] = comparison.apply(
    determine_method,
    axis=1
)

comparison["tfidf_top5"] = comparison["tfidf_rank"] <= 5
comparison["embeddings_top5"] = comparison["embedding_rank"] <= 5

# ============================================================
# 4. CAS OU LES METHODES SONT EN DESACCORD SUR LE TOP-5
# ============================================================

errors = comparison[
    comparison["tfidf_top5"] != comparison["embeddings_top5"]
].copy()

errors.to_csv(
    OUTPUT_ERRORS,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 5. CAS OU LES DEUX METHODES ECHOUENT AU TOP-5
# ============================================================

failures = comparison[
    (comparison["tfidf_rank"] > 5)
    & (comparison["embedding_rank"] > 5)
].copy()

failures.to_csv(
    OUTPUT_FAILURES,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 6. AFFICHAGE DES RESULTATS
# ============================================================

print("=" * 65)
print("ANALYSE DES ERREURS DU RETRIEVAL")
print("=" * 65)

print(f"\nNombre total de questions : {len(comparison)}")

print(
    f"Cas où les méthodes divergent sur le Top-5 : "
    f"{len(errors)}"
)

print(
    f"Cas où les deux méthodes échouent au Top-5 : "
    f"{len(failures)}"
)

print("\n--- Cas de désaccord entre les méthodes ---")

columns = [
    "question",
    "tfidf_rank",
    "embedding_rank",
    "meilleure_methode",
    "document_title"
]

print(errors[columns].to_string(index=False))

print("\n--- Cas où les deux méthodes échouent ---")

print(failures[columns].to_string(index=False))

print("\n--- Fichiers créés ---")
print(f"Désaccords : {OUTPUT_ERRORS}")
print(f"Échecs communs : {OUTPUT_FAILURES}")