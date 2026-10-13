import pandas as pd

file = "Data/insurance-fr.csv"

df = pd.read_csv(file)

print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", len(df))

print("\nColonnes :")
print(df.columns.tolist())

print("\nPremières lignes :")
print(df.head().to_string())
print("\nShape :", df.shape)

print("\nTypes :")
print(df.dtypes)

print("\nPremière ligne :")
print(df.iloc[0])

print("\nNombre de valeurs manquantes :")
print(df.isna().sum())
print("\nTitres des documents :")

for i, title in enumerate(df["title"], start=1):
    print(f"{i}. {title}")
print("\n--- DEUXIÈME EXEMPLE ---")

print("TITLE :")
print(df.iloc[1]["title"])

print("\nCONTENT :")
print(df.iloc[1]["content"])


