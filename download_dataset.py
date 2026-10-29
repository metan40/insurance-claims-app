import requests
from pathlib import Path

URL = "https://huggingface.co/datasets/zelros/insurance-fr/resolve/main/data.csv"
OUTPUT = Path("data/insurance-fr.csv")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Téléchargement du dataset...")
    print(f"Source : {URL}")

    response = requests.get(URL)

    if response.status_code == 200:
        OUTPUT.write_bytes(response.content)

        print("Téléchargement terminé.")
        print(f"Fichier enregistré ici : {OUTPUT.resolve()}")
    else:
        print(f"Erreur lors du téléchargement : {response.status_code}")


if __name__ == "__main__":
    main()
