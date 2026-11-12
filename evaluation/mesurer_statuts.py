"""Mesure des statuts de l'assistant RAG (étape 1 : état actuel, sans rien modifier).

Pour chaque question, le script passe par la même fonction que l'application
(``src.app.search``) : même recherche, même prompt, mêmes réglages Ollama.
Il enregistre, appel par appel, ce que le modèle a répondu :

* ``json_valide``         : la sortie respecte-t-elle le schéma ?
* ``answer_found``        : le modèle déclare-t-il avoir trouvé la réponse ?
* ``extraits_cites``      : nombre d'extraits recopiés par le modèle ;
* ``extraits_verifies``   : combien sont retrouvés dans la source citée ;

puis le statut final affiché (answered, partial, not_found, fallback).

Pour les questions dont la bonne source est connue (jeu ``heldout``), il dit
aussi si cette source faisait partie des documents lus par le modèle. Cela
sépare deux causes d'une « réponse absente » :

* le bon document n'a pas été lu   -> problème de recherche ;
* le bon document a été lu         -> le modèle a refusé à tort.

Prérequis : Ollama lancé, modèle installé (``ollama pull llama3.2:1b``).

Exemples :
    python evaluation/mesurer_statuts.py --jeu demo
    python evaluation/mesurer_statuts.py --jeu heldout --etiquette avant
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.app as application  # noqa: E402  (charge le corpus et l'index, comme au démarrage)
from src.generation import parse_typed_answer  # noqa: E402

if application.embedding_retriever is not None:
    # Moteur « embeddings » : on attend l'index avant de mesurer.
    print("Préparation de la recherche par embeddings…", flush=True)
    if not application.embedding_retriever.wait(3600):
        raise SystemExit(f"Embeddings indisponibles : {application.embedding_retriever.error}")

DEMO_QUESTIONS = [
    "Ma bague a été dérobée, comment suis-je indemnisé ?",
    "Mon fils a cassé la télévision d'un ami, suis-je couvert ?",
    "Quel est le montant de la franchise en cas de catastrophe naturelle ?",
    "Si la nounou de mes enfants cause un dommage, mon assurance la couvre-t-elle ?",
    "Mon vélo a été volé dans la cave, suis-je couvert ?",
]

RESULTS_DIR = ROOT / "evaluation" / "results"
REFERENCE_FILE = ROOT / "evaluation" / "reponses_reference.csv"


def load_references() -> dict[str, list[list[str]]]:
    """Faits obligatoires par question : groupes séparés par « ; », variantes par « | »."""
    from src.question_parsing import normalize_text
    if not REFERENCE_FILE.exists():
        return {}
    with REFERENCE_FILE.open(encoding="utf-8") as handle:
        return {
            normalize_text(row["question"]): [
                [normalize_text(v) for v in group.split("|") if v.strip()]
                for group in row["faits_obligatoires"].split(";") if group.strip()
            ]
            for row in csv.DictReader(handle)
        }


def facts_present(answer: str, groups: list[list[str]]) -> tuple[int, int]:
    """(groupes de faits retrouvés dans la réponse affichée, groupes attendus)."""
    from src.question_parsing import normalize_text
    text = f" {normalize_text(answer)} "
    found = sum(1 for group in groups if any(f" {variant}" in text for variant in group))
    return found, len(groups)


def load_questions(jeu: str) -> list[dict]:
    if jeu == "demo":
        sources = {}
        if REFERENCE_FILE.exists():
            with REFERENCE_FILE.open(encoding="utf-8") as handle:
                sources = {row["question"]: row["source_attendue"] for row in csv.DictReader(handle)}
        return [{"question": q, "expected_title": sources.get(q, "")} for q in DEMO_QUESTIONS]
    path = ROOT / "evaluation" / ("heldout_questions.csv" if jeu == "heldout" else "paraphrase_questions.csv")
    with path.open(encoding="utf-8") as handle:
        return [{"question": row["question"], "expected_title": row["expected_title"]} for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jeu", choices=["demo", "heldout", "dev"], default="demo")
    parser.add_argument("--etiquette", default="avant", help="nom de la mesure, par exemple avant ou apres")
    parser.add_argument("--limite", type=int, default=0, help="ne traiter que les N premières questions")
    parser.add_argument("--repetitions", type=int, default=1, help="passages par question (variabilité du modèle)")
    args = parser.parse_args()

    questions = load_questions(args.jeu)
    if args.limite:
        questions = questions[: args.limite]
    questions = [dict(item, passage=k) for item in questions for k in range(1, args.repetitions + 1)]
    references = load_references()
    from src.question_parsing import normalize_text as _norm

    # On enregistre chaque appel au modèle sans changer son comportement.
    calls: list[dict] = []
    original_call = application.call_ollama

    def recording_call(prompt: str, label: str) -> str:
        started = time.perf_counter()
        raw = original_call(prompt, label)
        calls.append({"label": label, "raw": raw, "seconds": round(time.perf_counter() - started, 1)})
        return raw

    application.call_ollama = recording_call

    rows = []
    for number, item in enumerate(questions, start=1):
        calls.clear()
        started = time.perf_counter()
        try:
            response = application.search(application.Question(question=item["question"]))
        except Exception as error:  # Ollama arrêté, délai dépassé…
            print(f"[{number}] ERREUR : {error}")
            rows.append({"question": item["question"], "statut": "erreur", "detail": str(error)})
            continue
        total = round(time.perf_counter() - started, 1)

        titles_read = [source["title"] for source in response["results"]][: response.get("sources_read_by_model") or 5]
        typed = response.get("typed_answer") or {}
        evidence = typed.get("evidence") or []
        expected_ids = {source["source_id"] for source in response["results"]
                        if item["expected_title"] and source["title"] == item["expected_title"]}
        row = {
            "question": item["question"],
            "statut": response["answer_status"],
            "appels": len(calls),
            "secondes": total,
            "bonne_source_attendue": item["expected_title"],
            "bonne_source_lue": (item["expected_title"] in titles_read) if item["expected_title"] else "",
            "extraits_verifies_final": sum(1 for e in evidence if e.get("verified")),
            # Qualité : la réponse affichée s'appuie-t-elle sur la bonne source ?
            "bonne_source_citee_et_verifiee": (
                any(e.get("verified") and e.get("source_id") in expected_ids for e in evidence)
                if item["expected_title"] else ""),
            "phrases_non_etayees_masquees": len(typed.get("unsupported_sentences") or []),
            "conditions_ajoutees_depuis_les_sources": len(typed.get("conditions_added_from_sources") or []),
            "reponse_affichee": str(response.get("answer", "")).replace("\n", " ")[:500],
            "passage": item.get("passage", 1),
        }
        groups = references.get(_norm(item["question"]))
        if groups:
            found, expected = facts_present(str(response.get("answer", "")), groups)
            row["faits_retrouves"] = f"{found}/{expected}"
            row["reponse_conforme"] = found == expected
        for index, call in enumerate(calls, start=1):
            parsed = parse_typed_answer(call["raw"])
            row[f"appel{index}_json_valide"] = parsed is not None
            row[f"appel{index}_answer_found"] = parsed.answer_found if parsed else ""
            row[f"appel{index}_extraits_cites"] = len(parsed.evidence) if parsed else ""
            row[f"appel{index}_secondes"] = call["seconds"]
            row[f"appel{index}_sortie_brute"] = call["raw"].replace("\n", " ")[:600]
        rows.append(row)
        print(f"[{number}/{len(questions)}] {row['statut']:<9} appels={row['appels']} "
              f"{total:>6} s  bonne source lue={row['bonne_source_lue']}  {item['question'][:60]}")

    application.call_ollama = original_call

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"statuts_{args.jeu}_{args.etiquette}.csv"
    columns = sorted({key for row in rows for key in row}, key=lambda k: (k.startswith("appel"), k))
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Synthèse
    done = [r for r in rows if r["statut"] != "erreur"]
    counts = Counter(r["statut"] for r in rows)
    refused = [r for r in done if r["statut"] == "not_found"]
    refused_with_source = [r for r in refused if r.get("bonne_source_lue") is True]
    summary = {
        "jeu": args.jeu,
        "etiquette": args.etiquette,
        "modele": application.OLLAMA_MODEL,
        "moteur_recherche": application.active_retrieval_engine(),
        "questions": len(rows),
        "statuts": dict(counts),
        "part_reponse_absente": round(len(refused) / len(done), 3) if done else None,
        "reponse_absente_alors_que_la_bonne_source_etait_lue": len(refused_with_source),
        "appels_avec_answer_found_false": sum(
            1 for r in done for k, v in r.items() if k.endswith("_answer_found") and v is False),
        "appels_avec_json_invalide": sum(
            1 for r in done for k, v in r.items() if k.endswith("_json_valide") and v is False),
        "temps_moyen_secondes": round(sum(r["secondes"] for r in done) / len(done), 1) if done else None,
        "temps_max_secondes": max((r["secondes"] for r in done), default=None),
        "part_bonne_source_citee_et_verifiee": (
            round(sum(1 for r in done if r.get("bonne_source_citee_et_verifiee") is True)
                  / max(1, sum(1 for r in done if r.get("bonne_source_attendue"))), 3)),
        "reponses_avec_phrases_masquees": sum(1 for r in done if r.get("phrases_non_etayees_masquees")),
        "reponses_avec_condition_ajoutee": sum(1 for r in done if r.get("conditions_ajoutees_depuis_les_sources")),
        "erreurs": len(rows) - len(done),
        "avec_reference": sum(1 for r in done if "reponse_conforme" in r),
        "reponses_conformes": sum(1 for r in done if r.get("reponse_conforme") is True),
        "bonne_source_citee_et_verifiee_n": sum(1 for r in done if r.get("bonne_source_citee_et_verifiee") is True),
        "avec_source_attendue": sum(1 for r in done if r.get("bonne_source_attendue")),
        "repetitions": args.repetitions,
    }
    summary["part_reponses_conformes"] = (
        round(summary["reponses_conformes"] / summary["avec_reference"], 3) if summary["avec_reference"] else None)
    (RESULTS_DIR / f"statuts_{args.jeu}_{args.etiquette}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSYNTHÈSE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDétail par question : {out}")


if __name__ == "__main__":
    main()
