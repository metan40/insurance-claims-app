"""Expérience : ajouter un classement par embeddings multilingues à la recherche.

Question posée : un troisième classement (similarité d'embeddings entre la
question et la fiche), fusionné par RRF avec le classement de production,
améliore-t-il la recherche sur des questions reformulées ?

Protocole (fixé avant de lancer l'expérience) :

1. Jeu de **développement** : les 40 reformulations
   (``paraphrase_questions.csv``). Toutes les configurations y sont mesurées et
   **une seule** est retenue : la meilleure en Recall@3, puis en MRR@5.
2. Jeu **tenu à part** : les 30 questions de ``heldout_questions.csv``,
   jamais utilisées pour régler quoi que ce soit. La configuration retenue y est
   comparée à la production par un test de McNemar exact apparié sur
   « bonne fiche dans le top 3 » (les 3 sources lues par le modèle).
   Les autres configurations sont affichées à titre indicatif seulement.

Fusion : score(d) = 1/(k + r_prod(d)) + w / (k + r_emb(d)), k = 60, où r_prod
est le rang dans le pool de production (20 candidats, absent sinon) et r_emb
le rang par similarité cosinus des embeddings sur tout le corpus.

Usage : python evaluation/experience_embeddings.py  (nécessite sentence-transformers)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import DEFAULT_CANDIDATE_LIMIT, RRF_CONSTANT, HybridRetriever  # noqa: E402

MODELS = {
    # nom court : (identifiant Hugging Face, préfixe question, préfixe document)
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "minilm-l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}
WEIGHTS = (0.5, 1.0, 2.0)
DOCUMENT_CHARACTERS = 1000
RESULTS = ROOT / "evaluation" / "results"


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r is not None and r <= 5) / n
    return {"recall@1": round(hit(1), 3), "recall@3": round(hit(3), 3),
            "recall@5": round(hit(5), 3), "mrr@5": round(mrr, 3)}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def rank_of(order: list[int], target: int) -> int | None:
    return order.index(target) + 1 if target in order else None


def main() -> None:
    from sentence_transformers import SentenceTransformer

    corpus = pd.read_csv(ROOT / "Data" / "insurance-fr.csv")
    titles = corpus["title"].fillna("").astype(str).tolist()
    documents = corpus["content"].fillna("").astype(str).tolist()
    retriever = HybridRetriever(titles, documents)
    sets = {
        "dev": pd.read_csv(ROOT / "evaluation" / "paraphrase_questions.csv"),
        "tenu_a_part": pd.read_csv(ROOT / "evaluation" / "heldout_questions.csv"),
    }
    production = {
        name: [[item.index for item in retriever.rank(q, candidate_limit=DEFAULT_CANDIDATE_LIMIT)]
               for q in frame["question"]]
        for name, frame in sets.items()
    }

    table = []   # une ligne par (jeu, configuration)
    orders = {}  # (jeu, configuration) -> classements
    for name, frame in sets.items():
        orders[(name, "production")] = production[name]
        targets = frame["expected_index"].tolist()
        table.append({"jeu": name, "configuration": "production",
                      **metrics([rank_of(o, t) for o, t in zip(production[name], targets)])})

    for short, (model_id, query_prefix, doc_prefix) in MODELS.items():
        print(f"Encodage avec {model_id}…", flush=True)
        model = SentenceTransformer(model_id)
        passages = [f"{doc_prefix}{t}\n{d[:DOCUMENT_CHARACTERS]}" for t, d in zip(titles, documents)]
        doc_vectors = model.encode(passages, normalize_embeddings=True, batch_size=16)
        for name, frame in sets.items():
            targets = frame["expected_index"].tolist()
            q_vectors = model.encode([f"{query_prefix}{q}" for q in frame["question"]], normalize_embeddings=True)
            similarity = q_vectors @ doc_vectors.T
            emb_orders = [list(np.argsort(-row)) for row in similarity]
            orders[(name, f"{short} seul")] = emb_orders
            table.append({"jeu": name, "configuration": f"{short} seul",
                          **metrics([rank_of(o, t) for o, t in zip(emb_orders, targets)])})
            for w in WEIGHTS:
                fused = []
                for prod, emb in zip(production[name], emb_orders):
                    score = {}
                    for r, d in enumerate(prod, start=1):
                        score[d] = score.get(d, 0.0) + 1 / (RRF_CONSTANT + r)
                    for r, d in enumerate(emb[:DEFAULT_CANDIDATE_LIMIT], start=1):
                        score[d] = score.get(d, 0.0) + w / (RRF_CONSTANT + r)
                    fused.append([d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])])
                config = f"production + {short} (w={w})"
                orders[(name, config)] = fused
                table.append({"jeu": name, "configuration": config,
                              **metrics([rank_of(o, t) for o, t in zip(fused, targets)])})

    df = pd.DataFrame(table)
    dev = df[df["jeu"] == "dev"].sort_values(["recall@3", "mrr@5"], ascending=False)
    chosen = dev.iloc[0]["configuration"]

    held_targets = sets["tenu_a_part"]["expected_index"].tolist()
    in_top3 = lambda config: [
        (rank_of(o, t) or 99) <= 3 for o, t in zip(orders[("tenu_a_part", config)], held_targets)]
    prod_hits, chosen_hits = in_top3("production"), in_top3(chosen)
    only_prod = sum(1 for a, b in zip(prod_hits, chosen_hits) if a and not b)
    only_chosen = sum(1 for a, b in zip(prod_hits, chosen_hits) if b and not a)
    verdict = {
        "configuration_retenue_sur_dev": chosen,
        "tenu_a_part_production_top3": sum(prod_hits),
        "tenu_a_part_retenue_top3": sum(chosen_hits),
        "questions": len(held_targets),
        "seule_production_trouve": only_prod,
        "seule_retenue_trouve": only_chosen,
        "mcnemar_p": round(mcnemar(only_prod, only_chosen), 4),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "experience_embeddings.csv", index=False)
    (RESULTS / "experience_embeddings_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n## Expérience : embeddings multilingues dans la fusion\n")
    for name in ("dev", "tenu_a_part"):
        print(f"\n### Jeu {name}\n")
        print(df[df["jeu"] == name].sort_values("recall@3", ascending=False).to_markdown(index=False))
    print("\n### Verdict (configuration choisie sur dev, testée une fois sur le jeu tenu à part)\n")
    print("```json\n" + json.dumps(verdict, ensure_ascii=False, indent=2) + "\n```")


if __name__ == "__main__":
    main()
