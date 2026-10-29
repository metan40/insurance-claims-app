import pandas as pd
from pathlib import Path
from html import escape

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "Data" / "retrieval_comparison.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data"
OUTPUT_FILE = OUTPUT_DIR / "retrieval_report_2.html"

# ============================================================
# 1. CHARGEMENT ET VALIDATION DES RESULTATS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}\n"
        "Exécute d'abord retrieval_comparison.py."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "question",
    "expected_document",
    "tfidf_rank",
    "embedding_rank",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Colonnes manquantes dans le CSV : {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("Le fichier de résultats ne contient aucune question.")

for column in ["tfidf_rank", "embedding_rank"]:
    df[column] = pd.to_numeric(df[column], errors="raise")

    if df[column].isna().any() or (df[column] < 1).any():
        raise ValueError(
            f"La colonne {column} contient des rangs invalides."
        )

total = len(df)

# ============================================================
# 2. CALCUL DES METRIQUES
# ============================================================

def recall_at_k(ranks, k):
    return (ranks <= k).mean() * 100


def mrr(ranks):
    return (1 / ranks).mean()


tfidf_ranks = df["tfidf_rank"]
embedding_ranks = df["embedding_rank"]

metrics = {
    "TF-IDF": {
        "Recall@1": recall_at_k(tfidf_ranks, 1),
        "Recall@3": recall_at_k(tfidf_ranks, 3),
        "Recall@5": recall_at_k(tfidf_ranks, 5),
        "MRR": mrr(tfidf_ranks),
    },
    "Embeddings": {
        "Recall@1": recall_at_k(embedding_ranks, 1),
        "Recall@3": recall_at_k(embedding_ranks, 3),
        "Recall@5": recall_at_k(embedding_ranks, 5),
        "MRR": mrr(embedding_ranks),
    },
}

recall1_gain = (
    metrics["Embeddings"]["Recall@1"]
    - metrics["TF-IDF"]["Recall@1"]
)

recall3_gain = (
    metrics["Embeddings"]["Recall@3"]
    - metrics["TF-IDF"]["Recall@3"]
)

recall5_gain = (
    metrics["Embeddings"]["Recall@5"]
    - metrics["TF-IDF"]["Recall@5"]
)

mrr_gain = (
    metrics["Embeddings"]["MRR"]
    - metrics["TF-IDF"]["MRR"]
)

# ============================================================
# 3. COMPARAISON DES CLASSEMENTS
# ============================================================

embedding_better = (
    df["embedding_rank"] < df["tfidf_rank"]
).sum()

tfidf_better = (
    df["tfidf_rank"] < df["embedding_rank"]
).sum()

same_rank = (
    df["tfidf_rank"] == df["embedding_rank"]
).sum()

both_top5 = (
    (df["tfidf_rank"] <= 5)
    & (df["embedding_rank"] <= 5)
).sum()

tfidf_only_top5 = (
    (df["tfidf_rank"] <= 5)
    & (df["embedding_rank"] > 5)
).sum()

embedding_only_top5 = (
    (df["embedding_rank"] <= 5)
    & (df["tfidf_rank"] > 5)
).sum()

both_fail_top5 = (
    (df["tfidf_rank"] > 5)
    & (df["embedding_rank"] > 5)
).sum()

# ============================================================
# 4. SYNTHESE DES RESULTATS
# ============================================================

summary_html = f"""
<p>
L'évaluation porte sur <strong>{total} questions</strong> du jeu de
données <strong>insurance-fr.csv</strong>. Elle compare une recherche
lexicale TF-IDF à une recherche sémantique par embeddings multilingues.
</p>

<p>
<strong>Recall@1 :</strong> les embeddings obtiennent
{metrics["Embeddings"]["Recall@1"]:.2f} %, contre
{metrics["TF-IDF"]["Recall@1"]:.2f} % pour TF-IDF,
soit un écart de {recall1_gain:+.2f} points de pourcentage.
</p>

<p>
<strong>Recall@3 :</strong> les embeddings obtiennent
{metrics["Embeddings"]["Recall@3"]:.2f} %, contre
{metrics["TF-IDF"]["Recall@3"]:.2f} % pour TF-IDF,
soit un écart de {recall3_gain:+.2f} points.
</p>

<p>
<strong>Recall@5 :</strong> les embeddings obtiennent
{metrics["Embeddings"]["Recall@5"]:.2f} %, contre
{metrics["TF-IDF"]["Recall@5"]:.2f} % pour TF-IDF,
soit un écart de {recall5_gain:+.2f} points.
</p>

<p>
<strong>MRR :</strong> le score passe de
{metrics["TF-IDF"]["MRR"]:.4f} pour TF-IDF à
{metrics["Embeddings"]["MRR"]:.4f} pour les embeddings,
soit un écart de {mrr_gain:+.4f}.
</p>

<p>
Sur l'ensemble des questions, les embeddings obtiennent un meilleur
rang dans {embedding_better} cas, TF-IDF dans {tfidf_better} cas,
et les deux méthodes obtiennent le même rang dans {same_rank} cas.
</p>

<p>
Pour {both_fail_top5} questions, aucune méthode ne classe le document
attendu dans les cinq premiers résultats. Ces cas représentent
{both_fail_top5 / total * 100:.2f} % des questions évaluées.
</p>

<p>
<strong>Conclusion :</strong> sur ce jeu de données et selon les
métriques mesurées, les embeddings présentent les meilleurs résultats
globaux. TF-IDF reste toutefois plus performant pour certaines questions.
L'analyse des erreurs permettra d'identifier les limites de chaque
méthode et les possibilités d'amélioration.
</p>
"""

# ============================================================
# 5. TABLEAU DES METRIQUES
# ============================================================

metric_rows = ""

for method in ["TF-IDF", "Embeddings"]:
    values = metrics[method]

    metric_rows += f"""
    <tr>
        <td>{method}</td>
        <td>{values["Recall@1"]:.2f} %</td>
        <td>{values["Recall@3"]:.2f} %</td>
        <td>{values["Recall@5"]:.2f} %</td>
        <td>{values["MRR"]:.4f}</td>
    </tr>
    """

metrics_table = f"""
<table>
    <thead>
        <tr>
            <th>Méthode</th>
            <th>Recall@1</th>
            <th>Recall@3</th>
            <th>Recall@5</th>
            <th>MRR</th>
        </tr>
    </thead>
    <tbody>
        {metric_rows}
    </tbody>
</table>
"""

# ============================================================
# 6. TABLEAU DE COMPARAISON
# ============================================================

comparison_items = [
    ("Embeddings meilleur", embedding_better),
    ("TF-IDF meilleur", tfidf_better),
    ("Même rang", same_rank),
    ("Les deux dans le Top-5", both_top5),
    ("TF-IDF dans le Top-5 uniquement", tfidf_only_top5),
    ("Embeddings dans le Top-5 uniquement", embedding_only_top5),
    ("Aucune méthode dans le Top-5", both_fail_top5),
]

comparison_rows = ""

for label, count in comparison_items:
    comparison_rows += f"""
    <tr>
        <td>{label}</td>
        <td>{count}</td>
        <td>{count / total * 100:.2f} %</td>
    </tr>
    """

comparison_table = f"""
<table>
    <thead>
        <tr>
            <th>Situation</th>
            <th>Nombre de questions</th>
            <th>Pourcentage</th>
        </tr>
    </thead>
    <tbody>
        {comparison_rows}
    </tbody>
</table>
"""

# ============================================================
# 7. TABLEAU DETAILLE DES QUESTIONS
# ============================================================

details = df.copy()

details["meilleure_methode"] = details.apply(
    lambda row:
        "Embeddings"
        if row["embedding_rank"] < row["tfidf_rank"]
        else "TF-IDF"
        if row["tfidf_rank"] < row["embedding_rank"]
        else "Égalité",
    axis=1,
)

details["TF-IDF Top-5"] = details["tfidf_rank"] <= 5
details["Embeddings Top-5"] = details["embedding_rank"] <= 5

details_html = details.to_html(
    index=False,
    classes="data-table",
    border=0,
    escape=True,
    na_rep="",
)

# ============================================================
# 8. CAS OU LES METHODES SONT EN DESACCORD
# ============================================================

interesting = df[
    (
        (df["tfidf_rank"] <= 5)
        & (df["embedding_rank"] > 5)
    )
    |
    (
        (df["embedding_rank"] <= 5)
        & (df["tfidf_rank"] > 5)
    )
].copy()

interesting["meilleure_methode"] = interesting.apply(
    lambda row:
        "Embeddings"
        if row["embedding_rank"] < row["tfidf_rank"]
        else "TF-IDF",
    axis=1,
)

interesting_html = interesting.to_html(
    index=False,
    classes="data-table",
    border=0,
    escape=True,
    na_rep="",
)

# ============================================================
# 9. ANALYSE DU RERANKING ET CONTROLES COMPLEMENTAIRES
# ============================================================

# Je charge le fichier détaillé du reranking lorsqu'il est disponible.
# Cette analyse reste séparée de la comparaison TF-IDF / embeddings afin
# de ne pas mélanger les métriques calculées sur des étapes différentes.
RERANKING_FILE = PROJECT_ROOT / "Data" / "retrieval_optimization_detailed.csv"
reranking_available = RERANKING_FILE.exists()
reranking_df = pd.DataFrame()
reranking_metrics = {}
reranking_summary_html = ""
reranking_details_html = ""
missing_targets_html = ""

if reranking_available:
    reranking_df = pd.read_csv(RERANKING_FILE)
    reranking_required_columns = {
        "question",
        "expected_document",
        "embedding_rank",
        "optimized_rank",
        "retrieved_in_top20",
    }
    reranking_missing_columns = (
        reranking_required_columns - set(reranking_df.columns)
    )

    if reranking_missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le fichier de reranking : "
            f"{sorted(reranking_missing_columns)}"
        )

    if reranking_df.empty:
        raise ValueError(
            "Le fichier détaillé du reranking ne contient aucune question."
        )

    reranking_total = len(reranking_df)
    reranking_df["optimized_rank"] = pd.to_numeric(
        reranking_df["optimized_rank"], errors="coerce"
    )
    reranking_df["embedding_rank"] = pd.to_numeric(
        reranking_df["embedding_rank"], errors="raise"
    )

    if reranking_df["embedding_rank"].isna().any():
        raise ValueError(
            "Le fichier de reranking contient des rangs d'embeddings manquants."
        )

    if (reranking_df["embedding_rank"] < 1).any():
        raise ValueError(
            "Les rangs d'embeddings doivent être supérieurs ou égaux à 1."
        )

    valid_optimized_ranks = reranking_df["optimized_rank"].dropna()
    if (valid_optimized_ranks < 1).any():
        raise ValueError(
            "Les rangs optimisés présents doivent être supérieurs ou égaux à 1."
        )

    # Une cible absente du classement initial reste dans le dénominateur.
    # Son rang optimisé est manquant et compte donc comme un échec au Top-5.
    reranking_df["optimized_hit_at_5"] = (
        reranking_df["optimized_rank"].notna()
        & (reranking_df["optimized_rank"] <= 5)
    )
    reranking_df["embedding_hit_at_5"] = (
        reranking_df["embedding_rank"] <= 5
    )
    reranking_df["target_in_top20"] = reranking_df[
        "retrieved_in_top20"
    ].astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "oui"]
    )

    targets_in_top20 = int(reranking_df["target_in_top20"].sum())
    targets_missing_top20 = reranking_total - targets_in_top20
    embedding_hit5_reranking = (
        reranking_df["embedding_hit_at_5"].mean() * 100
    )
    optimized_hit5 = reranking_df["optimized_hit_at_5"].mean() * 100
    reranking_gain_hit5 = optimized_hit5 - embedding_hit5_reranking

    # Le MRR optimisé attribue zéro à une cible absente du classement.
    reciprocal_optimized_rank = reranking_df["optimized_rank"].apply(
        lambda rank: 1 / rank if pd.notna(rank) and rank >= 1 else 0
    )
    optimized_mrr = reciprocal_optimized_rank.mean()

    reranking_metrics = {
        "total": reranking_total,
        "targets_in_top20": targets_in_top20,
        "targets_missing_top20": targets_missing_top20,
        "coverage_pct": targets_in_top20 / reranking_total * 100,
        "missing_pct": targets_missing_top20 / reranking_total * 100,
        "embedding_hit5": embedding_hit5_reranking,
        "optimized_hit5": optimized_hit5,
        "gain_hit5": reranking_gain_hit5,
        "optimized_mrr": optimized_mrr,
    }

    reranking_summary_html = f"""
    <p>
        Dans ce projet, j'ai aussi étudié l'effet du reranking sur les
        documents proposés par la recherche initiale. Le reranking réorganise
        les documents déjà récupérés ; il ne peut pas faire apparaître une
        cible qui n'a pas été récupérée à l'étape précédente.
    </p>
    <p>
        Le fichier détaillé contient {reranking_total} questions. Pour
        {targets_in_top20} questions ({targets_in_top20 / reranking_total * 100:.2f} %),
        la cible attendue est présente dans le Top-20 initial. Pour les
        {targets_missing_top20} autres questions
        ({targets_missing_top20 / reranking_total * 100:.2f} %), la cible
        est absente du Top-20.
    </p>
    <p>
        Le Hit Rate@5 des embeddings est de {embedding_hit5_reranking:.2f} %
        dans ce fichier détaillé. Après reranking, il atteint
        {optimized_hit5:.2f} %, soit une évolution de
        {reranking_gain_hit5:+.2f} points de pourcentage. Je conserve les
        {reranking_total} questions dans le calcul, y compris celles pour
        lesquelles la cible est absente du Top-20 initial.
    </p>
    <p>
        Le MRR après reranking atteint {optimized_mrr:.4f}. Pour ce calcul,
        j'attribue une contribution nulle aux questions dont la cible ne
        possède pas de rang optimisé. Cette règle évite de gonfler le score
        en supprimant les cas où le système n'a pas retrouvé le document.
    </p>
    <p>
        Ces résultats décrivent le jeu de questions utilisé dans ce projet.
        Il ne faudrait pas supposer qu'ils garantissent les mêmes performances
        sur toutes les questions réelles. Une évaluation complémentaire avec
        des questions variées permettrait de vérifier si les tendances se
        maintiennent dans d'autres situations.
    </p>
    """

    reranking_metric_rows = "".join([
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("Questions évaluées", f"{reranking_total}"),
            ("Cibles présentes dans le Top-20", f"{targets_in_top20} / {reranking_total} ({targets_in_top20 / reranking_total * 100:.2f} %)"),
            ("Cibles absentes du Top-20", f"{targets_missing_top20} / {reranking_total} ({targets_missing_top20 / reranking_total * 100:.2f} %)"),
            ("Hit Rate@5 des embeddings", f"{embedding_hit5_reranking:.2f} %"),
            ("Hit Rate@5 après reranking", f"{optimized_hit5:.2f} %"),
            ("Évolution du Hit Rate@5", f"{reranking_gain_hit5:+.2f} points"),
            ("MRR après reranking", f"{optimized_mrr:.4f}"),
        ]
    ])
    reranking_metrics_table = f"""
    <table>
        <thead><tr><th>Indicateur</th><th>Résultat</th></tr></thead>
        <tbody>{reranking_metric_rows}</tbody>
    </table>
    """

    # Je présente toutes les questions dont la cible n'était pas dans le Top-20.
    # Ce tableau aide à identifier les cas que le reranking ne pouvait pas corriger.
    missing_targets = reranking_df.loc[
        ~reranking_df["target_in_top20"],
        ["question", "expected_document", "embedding_rank", "optimized_rank"],
    ].copy()
    missing_targets = missing_targets.rename(columns={
        "question": "Question",
        "expected_document": "Document attendu",
        "embedding_rank": "Rang embeddings",
        "optimized_rank": "Rang après reranking",
    })
    missing_targets_html = missing_targets.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
        na_rep="Absent",
    )

    reranking_details = reranking_df.copy()
    reranking_details = reranking_details.rename(columns={
        "question": "Question",
        "expected_document": "Document attendu",
        "expected_title": "Titre attendu",
        "embedding_rank": "Rang embeddings",
        "optimized_rank": "Rang après reranking",
        "retrieved_in_top20": "Cible présente dans le Top-20",
        "situation": "Situation",
    })
    reranking_details_html = reranking_details.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
        na_rep="Absent",
    )
else:
    reranking_summary_html = """
    <p>
        Je n'ai pas trouvé le fichier Data/retrieval_optimization_detailed.csv.
        Le rapport présente donc uniquement la comparaison TF-IDF / embeddings.
        Pour analyser le reranking, il faut d'abord générer le fichier détaillé.
    </p>
    """
    reranking_metrics_table = ""
    reranking_details_html = ""
    missing_targets_html = ""

# ============================================================
# 10. EXPLICATION DES METRIQUES ET METHODE D'EVALUATION
# ============================================================

methodology_html = f"""
<p>
    Dans cette évaluation, chaque question est associée à un document attendu.
    Je compare la position de ce document dans les résultats de TF-IDF et dans
    ceux des embeddings. Un rang plus petit signifie que le document attendu
    apparaît plus tôt dans la liste des résultats.
</p>
<p>
    Le Hit Rate@1 mesure la proportion de questions pour lesquelles le document
    attendu arrive en première position. Le Hit Rate@5 mesure la proportion de
    questions pour lesquelles le document attendu figure parmi les cinq premiers
    résultats. Le MRR tient aussi compte de la position exacte du premier document
    pertinent : les rangs les plus proches de la première position contribuent
    davantage au score.
</p>
<p>
    Je dois interpréter ces indicateurs ensemble. Un meilleur Hit Rate@5 indique
    que davantage de documents attendus apparaissent dans les cinq premières
    positions, mais il ne garantit pas que chaque question obtient un bon résultat.
    Le tableau détaillé permet donc d'examiner les différences question par question.
</p>
<p>
    Les résultats dépendent des questions et des documents présents dans les CSV.
    Je ne peux pas en déduire automatiquement que la même méthode sera toujours
    meilleure dans toutes les situations. Il faudrait tester le système avec un
    ensemble de questions plus large et représentatif pour confirmer sa robustesse.
</p>
"""

# ============================================================
# 11. CONTROLES DE COHERENCE POUR LE RAPPORT
# ============================================================

# Ces contrôles servent à signaler les différences de taille entre les deux
# fichiers d'évaluation. Ils ne remplacent pas l'inspection manuelle des cas.
consistency_notes = []
if reranking_available:
    if len(reranking_df) != total:
        consistency_notes.append(
            "Les fichiers de comparaison et de reranking ne contiennent pas "
            "le même nombre de questions. Les métriques restent calculées "
            "séparément sur chaque fichier."
        )
    else:
        same_questions = (
            df["question"].astype(str).reset_index(drop=True).equals(
                reranking_df["question"].astype(str).reset_index(drop=True)
            )
        )
        if not same_questions:
            consistency_notes.append(
                "Les questions ne sont pas dans le même ordre ou ne sont pas "
                "identiques entre les fichiers. Il faut vérifier leur alignement "
                "avant de comparer les métriques ligne par ligne."
            )
        else:
            consistency_notes.append(
                "Les deux fichiers contiennent le même nombre de questions et "
                "les libellés de questions correspondent dans le même ordre."
            )
else:
    consistency_notes.append(
        "Le contrôle croisé du reranking n'a pas été effectué car le fichier "
        "détaillé n'est pas disponible."
    )

consistency_html = "".join(
    f"<li>{escape(note)}</li>" for note in consistency_notes
)
consistency_html = f"<ul>{consistency_html}</ul>"

# ============================================================
# 12. GENERATION DU RAPPORT HTML
# ============================================================

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Rapport d'évaluation du retrieval</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 20px;
            color: #222;
            background: #f7f8fa;
            line-height: 1.6;
        }}

        h1 {{
            color: #17365d;
            border-bottom: 3px solid #17365d;
            padding-bottom: 12px;
        }}

        h2 {{
            color: #244a73;
            margin-top: 38px;
        }}

        p {{
            line-height: 1.7;
        }}

        .metadata {{
            background: white;
            padding: 16px 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}

        .summary {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }}

        .table-container {{
            width: 100%;
            overflow-x: auto;
            margin: 15px 0 25px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            margin-top: 15px;
        }}

        th {{
            background: #17365d;
            color: white;
            padding: 10px;
            text-align: left;
            white-space: nowrap;
        }}

        td {{
            border: 1px solid #ddd;
            padding: 8px;
            vertical-align: top;
        }}

        tbody tr:nth-child(even) {{
            background: #f0f3f7;
        }}

        .data-table {{
            font-size: 12px;
        }}

        .data-table td:first-child {{
            min-width: 250px;
        }}

        footer {{
            margin-top: 40px;
            padding: 18px 0;
            border-top: 1px solid #ccc;
            color: #555;
            font-size: 13px;
        }}

        @media print {{
            body {{
                background: white;
                max-width: none;
                margin: 10mm;
                padding: 0;
            }}

            .table-container {{
                overflow: visible;
            }}

            table {{
                font-size: 10px;
            }}

            h2 {{
                break-after: avoid;
            }}
        }}
    </style>
</head>

<body>

    <h1>Rapport d'évaluation du système de recherche documentaire</h1>

    <div class="metadata">
        <p><strong>Jeu de données :</strong> insurance-fr.csv</p>
        <p><strong>Nombre de questions :</strong> {total}</p>
        <p><strong>Méthodes comparées :</strong> TF-IDF et embeddings multilingues</p>
    </div>

    <section class="hypotheses">
        <h2>Hypothèses de départ concernant la mini-application RAG</h2>
        <p>
            Dans ce projet, je souhaite vérifier si la combinaison de la recherche sémantique,
            du reranking et d'un modèle de langage permet de fournir des réponses pertinentes
            aux questions portant sur les documents d'assurance. Je considère cette idée comme
            une hypothèse de départ, et non comme un résultat déjà démontré.
        </p>
        <p>
            Je pars de l'hypothèse que le reranking permet de mieux positionner les documents
            les plus pertinents et que cette amélioration peut contribuer à la qualité des
            réponses générées par le LLM. Les métriques de recherche permettent d'étudier le
            classement des documents, mais elles ne suffisent pas à elles seules pour juger
            de la qualité des réponses finales.
        </p>
        <p>
            Je souhaite également vérifier que le contexte fourni au LLM l'aide à produire des
            réponses justifiées par les documents récupérés. Lorsque les documents ne donnent
            pas assez d'informations pour répondre, l'application devrait signaler cette limite
            au lieu de présenter une supposition comme un fait établi.
        </p>
        <p>
            Enfin, je veux vérifier que la mini-application fonctionne dans des conditions
            normales d'utilisation et qu'une autre personne, notamment Keyzan, peut l'utiliser
            pour tester des questions, contrôler les sources et signaler les problèmes rencontrés.
            Je confirmerai ou nuancerai ces hypothèses à partir des tests réellement effectués.
        </p>
    </section>

    <h2>1. Performances globales</h2>

    <div class="table-container">
        {metrics_table}
    </div>

    <h2>2. Comparaison des classements</h2>

    <div class="table-container">
        {comparison_table}
    </div>

    <h2>3. Synthèse et conclusion</h2>

    <div class="summary">
        {summary_html}
    </div>

    <h2>4. Résultats détaillés par question</h2>

    <p>
        Le tableau suivant présente le rang du document attendu pour chaque
        question avec les deux méthodes, ainsi que la méthode ayant obtenu
        le meilleur rang.
    </p>

    <div class="table-container">
        {details_html}
    </div>

    <h2>5. Questions pour lesquelles les méthodes divergent</h2>

    <p>
        Ces questions sont particulièrement utiles pour étudier les différences
        entre la recherche lexicale et la recherche sémantique.
    </p>

    <div class="table-container">
        {interesting_html}
    </div>

    <h2>6. Comment j'interprète les métriques</h2>

    <div class="summary">
        {methodology_html}
    </div>

    <h2>7. Analyse du reranking</h2>

    <div class="summary">
        {reranking_summary_html}
    </div>

    <div class="table-container">
        {reranking_metrics_table}
    </div>

    <h2>8. Questions dont la cible est absente du Top-20</h2>

    <p>
        J'ai isolé ces questions pour comprendre les limites de la récupération
        initiale. Le reranking ne peut pas classer en première position un document
        qui n'a pas été fourni par cette première étape.
    </p>

    <div class="table-container">
        {missing_targets_html}
    </div>

    <h2>9. Détail des résultats du reranking</h2>

    <p>
        Ce tableau reprend les rangs initiaux et optimisés enregistrés dans le CSV.
        Les rangs optimisés absents sont affichés comme « Absent » et ne sont pas
        retirés du calcul des métriques.
    </p>

    <div class="table-container">
        {reranking_details_html}
    </div>

    <h2>10. Contrôles de cohérence des données</h2>

    <p>
        J'ai vérifié la disponibilité des fichiers et la cohérence du nombre de
        questions. Les remarques ci-dessous indiquent les contrôles automatiques
        effectués pendant la génération du rapport.
    </p>
    {consistency_html}

    <h2>11. Évaluation de l'application RAG avec le LLM</h2>

    <div class="summary">
        <p>
            Après l'évaluation de la recherche documentaire, je dois aussi vérifier
            le fonctionnement de l'application complète. Une bonne position des
            documents dans le classement ne garantit pas que le modèle de langage
            formule une réponse exacte, complète et fidèle aux sources. Je vais donc
            tester séparément la recherche, le reranking, la génération avec Ollama,
            l'affichage des sources et le comportement de l'interface.
        </p>
        <p>
            Dans le code actuel, l'application charge le fichier
            <code>Data/insurance-fr.csv</code>, calcule les embeddings au démarrage,
            récupère jusqu'à 20 candidats, les réordonne avec un CrossEncoder, puis
            transmet les cinq premiers documents à <code>llama3.2:3b</code> via Ollama.
            Le contexte transmis au LLM contient le titre et un extrait de chaque
            document, limité à 1 500 caractères. L'API expose une page d'accueil sur
            <code>/</code> et un point d'accès de recherche sur <code>POST /search</code>.
        </p>
        <p>
            Je n'ai pas encore considéré les tests fonctionnels de bout en bout comme
            réussis. Les scénarios ci-dessous constituent le plan que je dois exécuter
            avant de partager une version de test avec Keyzan. Je noterai les résultats
            réels après chaque exécution, au lieu de supposer que l'application fonctionne
            simplement parce que le serveur démarre.
        </p>
    </div>

    <h3>11.1 Scénarios de test à exécuter</h3>
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr><th>Test</th><th>Procédure</th><th>Résultat attendu</th><th>État</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>Démarrage de l'API</td>
                    <td>Je démarre FastAPI depuis la racine du projet et j'attends le chargement des modèles.</td>
                    <td>Le serveur démarre sans exception et les fichiers de données sont trouvés.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Ouverture de l'interface</td>
                    <td>J'ouvre la route <code>/</code> dans le navigateur.</td>
                    <td>La page <code>static/index.html</code> s'affiche et permet de saisir une question.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Question d'assurance connue</td>
                    <td>Je pose une question dont la réponse figure clairement dans le corpus.</td>
                    <td>L'API renvoie une réponse non vide et jusqu'à cinq documents avec leur titre, leur rang et leur score.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Question vide</td>
                    <td>J'envoie une chaîne vide ou composée d'espaces à <code>POST /search</code>.</td>
                    <td>L'API ne lance pas inutilement la recherche ni la génération et renvoie une réponse contrôlée.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Question hors corpus</td>
                    <td>Je pose une question qui n'est pas couverte par les documents.</td>
                    <td>Le modèle indique clairement que les sources ne permettent pas de répondre et n'invente pas de garanties.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Fidélité aux sources</td>
                    <td>Je compare chaque affirmation importante de la réponse aux extraits retournés.</td>
                    <td>Les affirmations sont justifiées par les documents et les incertitudes sont signalées.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Indisponibilité d'Ollama</td>
                    <td>Je teste le comportement lorsque le service Ollama est arrêté ou ne répond pas.</td>
                    <td>L'application présente une erreur compréhensible sans laisser croire qu'une réponse valide a été générée.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Latence et délai d'attente</td>
                    <td>Je mesure le temps de démarrage, le temps de recherche et le temps total de réponse.</td>
                    <td>Les temps sont documentés et les dépassements du délai de 120 secondes sont traités proprement.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Affichage des sources</td>
                    <td>Je vérifie dans l'interface que les documents renvoyés par l'API sont présentés clairement.</td>
                    <td>L'utilisateur peut identifier les documents utilisés et distinguer la réponse générée des sources récupérées.</td>
                    <td>À exécuter</td>
                </tr>
                <tr>
                    <td>Essais par Keyzan</td>
                    <td>Je fournis une URL de test, un jeu de questions et une fiche de retour.</td>
                    <td>Keyzan peut exécuter les mêmes scénarios et rapporter les erreurs, les réponses douteuses et les problèmes d'interface.</td>
                    <td>Après préparation du déploiement</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h3>11.2 Points techniques à vérifier avant le déploiement</h3>
    <ul>
        <li><strong>Accès à Ollama :</strong> le code utilise actuellement <code>http://localhost:11434/api/generate</code>. Si l'application tourne sur une autre machine, cette adresse désignera la machine qui héberge l'application, pas l'ordinateur de Keyzan. Je devrai donc héberger Ollama avec l'application ou rendre son adresse configurable.</li>
        <li><strong>Installation des modèles :</strong> l'environnement de test devra disposer des dépendances Python, des modèles Sentence Transformers et CrossEncoder, ainsi que du modèle Ollama <code>llama3.2:3b</code>. Je vérifierai ces éléments avant de communiquer l'URL de test.</li>
        <li><strong>Chemins des fichiers :</strong> dans la version 5, le CSV et la page HTML sont localisés à partir du dossier du projet. Je vérifierai ce comportement en lançant l’application depuis plusieurs répertoires.</li>
        <li><strong>Protection de l'accès :</strong> avant d'exposer l'application sur Internet, je devrai décider qui peut y accéder et ajouter les protections nécessaires, notamment une authentification ou une restriction d'accès, ainsi qu'une limitation des requêtes si nécessaire.</li>
        <li><strong>Gestion des erreurs :</strong> je vérifierai que les erreurs de connexion au LLM sont présentées comme des erreurs techniques et ne sont pas confondues avec une réponse d'assurance normale.</li>
        <li><strong>Confidentialité :</strong> je n'utiliserai pas de données personnelles réelles pendant les premiers essais partagés. Je privilégierai des questions de test et des données adaptées à cette phase.</li>
    </ul>

    <h3>11.3 Conditions avant de partager la version de test</h3>
    <ol>
        <li>Je confirmerai que l'API démarre sur un environnement propre et que la page d'accueil est accessible.</li>
        <li>Je testerai plusieurs questions couvertes par le corpus et plusieurs questions auxquelles les documents ne permettent pas de répondre.</li>
        <li>Je contrôlerai manuellement les réponses du LLM et les documents affichés pour repérer les affirmations non justifiées.</li>
        <li>Je vérifierai que l'application reste compréhensible lorsque le LLM est indisponible ou qu'une requête prend trop de temps.</li>
        <li>Je préparerai ensuite un accès de test, des consignes courtes et un tableau permettant à Keyzan de noter les questions utilisées, les réponses obtenues et les problèmes observés.</li>
    </ol>
    <p>
        Je considérerai le déploiement comme prêt pour les essais externes uniquement
        lorsque les vérifications essentielles auront été exécutées et que les limites
        connues auront été expliquées à Keyzan. Pour l'instant, cette section décrit le
        plan de validation et les points de vigilance ; elle ne signifie pas que l'application
        a déjà été déployée ni que tous les tests ont réussi.
    </p>

    <h2>12. Présentation de la mini-application « Assistant Assurance »</h2>

    <div class="summary">
        <p>
            J'ai développé une mini-application web pour rendre le système RAG utilisable
            depuis une interface simple. L'utilisateur peut saisir une question en français,
            lancer la recherche et consulter la réponse générée par le modèle. L'objectif est
            de réunir dans un même parcours la recherche documentaire, le reclassement des
            documents et la génération d'une réponse à partir des informations retrouvées.
        </p>
        <p>
            L'interface se trouve dans <code>static/index.html</code> et l'API est définie
            dans <code>src/app.py</code>. La page envoie la question à la route
            <code>POST /search</code>. L'API calcule l'embedding de la question, sélectionne
            jusqu'à 20 candidats, applique le CrossEncoder pour les réordonner, puis transmet
            les cinq premiers documents à <code>llama3.2:3b</code> via Ollama. L'API renvoie
            ensuite la réponse et les documents retenus à l'interface.
        </p>
        <p>
            Dans l'interface actuelle, je présente la réponse dans un bloc distinct et j'affiche
            ensuite les sources avec leur titre, leur rang, leur score de pertinence et un extrait
            du contenu. Cette présentation aide l'utilisateur à consulter les documents utilisés.
            Elle ne constitue toutefois pas encore une citation précise de chaque phrase de la
            réponse : il faudra vérifier manuellement que les affirmations importantes sont bien
            justifiées par les extraits affichés.
        </p>
    </div>

    <h3>12.1 Parcours d'une question dans la mini-application</h3>
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr><th>Étape</th><th>Ce que fait l'application</th><th>Ce que voit l'utilisateur</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>1. Saisie</td>
                    <td>La page vérifie que la zone de texte contient une question.</td>
                    <td>L'utilisateur saisit sa question dans le champ prévu à cet effet.</td>
                </tr>
                <tr>
                    <td>2. Recherche sémantique</td>
                    <td>L'API compare l'embedding de la question aux embeddings des documents du corpus.</td>
                    <td>L'interface indique que la recherche des documents pertinents est en cours.</td>
                </tr>
                <tr>
                    <td>3. Reranking</td>
                    <td>Le CrossEncoder réordonne les 20 candidats initiaux et l'application conserve les cinq premiers.</td>
                    <td>Cette étape reste interne ; les documents retenus sont présentés dans la section des sources.</td>
                </tr>
                <tr>
                    <td>4. Génération</td>
                    <td>L'application transmet la question et les cinq documents au modèle Llama via Ollama.</td>
                    <td>L'utilisateur reçoit une réponse rédigée en français.</td>
                </tr>
                <tr>
                    <td>5. Consultation des sources</td>
                    <td>L'API renvoie le titre, le rang, le score et un extrait de chaque document retenu.</td>
                    <td>L'utilisateur peut consulter les documents présentés sous la réponse.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h3>12.2 Éléments déjà présents dans le code</h3>
    <ul>
        <li><strong>Une interface web :</strong> la page affiche le titre « Assistant Assurance », un champ de saisie et un bouton de recherche.</li>
        <li><strong>Un contrôle de saisie :</strong> l'interface signale à l'utilisateur lorsqu'il lance une recherche sans question.</li>
        <li><strong>Un indicateur de chargement :</strong> la page affiche un message pendant la recherche et désactive temporairement le bouton.</li>
        <li><strong>Une réponse séparée des sources :</strong> l'interface présente d'abord la réponse générée, puis les documents récupérés.</li>
        <li><strong>Des informations sur les sources :</strong> chaque document affiché comprend son titre, son rang, son score de pertinence et le contenu renvoyé par l'API.</li>
        <li><strong>Une gestion des erreurs côté interface :</strong> la page affiche un message lorsque la requête échoue ou que le serveur renvoie une erreur HTTP.</li>
    </ul>

    <h3>12.3 Limites actuelles et améliorations à tester</h3>
    <ul>
        <li><strong>Justification des réponses :</strong> l'application affiche les documents sources, mais elle ne relie pas encore chaque affirmation de la réponse à un passage précis. Je vérifierai donc manuellement la fidélité de la réponse aux documents.</li>
        <li><strong>Questions sans réponse :</strong> le prompt demande au modèle de signaler lorsque les documents ne permettent pas de répondre, mais ce comportement doit être testé avec des questions réellement absentes du corpus.</li>
        <li><strong>Erreurs d'Ollama :</strong> la fonction de génération renvoie actuellement un message d'erreur sous forme de texte. Je vérifierai que l'interface distingue clairement cette erreur technique d'une réponse d'assurance normale.</li>
        <li><strong>Accès à distance :</strong> l'adresse actuelle d'Ollama utilise <code>localhost</code>. Pour les essais de Keyzan, le service devra être accessible depuis la machine qui exécute l'API et l'accès à l'application devra être préparé de manière contrôlée.</li>
        <li><strong>Chemins des fichiers :</strong> l'application utilise des chemins relatifs pour le CSV et la page HTML. Je vérifierai le lancement depuis la racine du projet afin d'éviter les erreurs de fichiers introuvables.</li>
        <li><strong>Protection de l'accès :</strong> avant de rendre la mini-application accessible à d'autres personnes, je définirai les restrictions d'accès nécessaires et j'éviterai d'utiliser des données personnelles réelles pour les premiers essais.</li>
    </ul>

    <h3>12.4 Critères de validation de la mini-application</h3>
    <p>
        Je considérerai que la mini-application fonctionne correctement lorsque je pourrai
        démarrer l'API, ouvrir la page dans un navigateur, envoyer une question couverte par
        le corpus, obtenir une réponse non vide et retrouver les documents présentés comme
        sources. Je testerai également une question vide, une question hors corpus et une
        indisponibilité d'Ollama. Je noterai le résultat réel de chaque test dans une fiche
        de validation et je ne marquerai un test comme réussi qu'après l'avoir exécuté.
    </p>
    <p>
        Cette mini-application constitue la partie concrète du projet : elle permet de tester
        le retrieval évalué dans les sections précédentes avec une génération de réponse et
        une interface utilisable. La validation complète reste nécessaire avant de considérer
        le projet terminé et avant de transmettre un accès de test à Keyzan.
    </p>


    <h3>12.5 Évolution du fichier <code>app.py</code></h3>
    <p>
        Dans la version 4, j'ai renforcé la validation de la question, le contrôle de la réponse
        renvoyée par Ollama et la traçabilité des documents présentés à l'utilisateur. La version
        5 prolonge ce travail sans modifier les modèles de recherche : elle rend les chemins des
        fichiers indépendants du répertoire courant, ajoute un contrôle formel des identifiants de
        source cités par la réponse et affiche le résultat de ce contrôle dans l'interface.
    </p>
    <div class="table-container">
        <table class="data-table">
            <thead><tr><th>Élément</th><th>État dans la version 5</th><th>Limite ou vérification restante</th></tr></thead>
            <tbody>
                <tr><td>Chemins des fichiers</td><td>Le CSV et la page HTML sont localisés à partir du dossier du projet, et non du répertoire courant du terminal.</td><td>Vérifier le démarrage depuis la racine du projet et depuis un autre répertoire.</td></tr>
                <tr><td>Recherche initiale</td><td>La similarité des embeddings sert à présélectionner jusqu'à 20 candidats.</td><td>Mesurer la qualité des candidats avec le corpus et les modèles réels.</td></tr>
                <tr><td>Reranking</td><td>Le CrossEncoder reclasse les candidats ; l'API renvoie séparément son score et la similarité des embeddings.</td><td>Vérifier que l'interface affiche les valeurs correspondant au classement retourné.</td></tr>
                <tr><td>Références aux sources</td><td>La réponse peut citer les documents avec des identifiants comme [S1]. L'API détecte les identifiants reconnus, inconnus et les sources disponibles qui ne sont pas citées.</td><td>Ce contrôle est formel : il ne démontre pas qu'une affirmation est réellement justifiée par le document cité.</td></tr>
                <tr><td>Génération</td><td>Une réponse Ollama vide ou mal formée entraîne une erreur HTTP contrôlée.</td><td>Vérifier les délais d'attente, l'indisponibilité d'Ollama et la génération avec le modèle réel.</td></tr>
                <tr><td>Interface</td><td>L'interface affiche les sources, les deux scores et le bilan formel des identifiants cités.</td><td>Vérifier la lisibilité et le comportement sur des réponses réelles, notamment quand une citation est absente ou inconnue.</td></tr>
            </tbody>
        </table>
    </div>
    <p>
        La suite de tests automatisés utilise des doubles pour les modèles et pour Ollama. Ces tests
        permettent de vérifier certaines règles de manière reproductible, mais ils ne remplacent pas
        une validation de bout en bout. Je ne considérerai donc pas la version comme validée avant
        d'avoir exécuté les scénarios avec les vrais modèles et un service Ollama actif.
    </p>

    <h3>12.6 Démarche suivie pour la version 5</h3>
    <p>
        J'ai retenu deux améliorations ciblées. Premièrement, les chemins construits à partir du
        dossier du projet doivent limiter les erreurs de fichiers introuvables lorsque le lancement
        est effectué depuis un autre répertoire. Deuxièmement, le contrôle des identifiants de source
        doit aider à repérer une citation absente ou un identifiant inventé. Je traite ces deux points
        comme des hypothèses de conception à vérifier, et non comme une preuve de qualité des réponses.
    </p>
    <p>
        Le contrôle de citations recherche les identifiants au format [S1], [S2], etc. Il distingue les
        identifiants présents dans les documents fournis, les identifiants inconnus et les documents
        disponibles qui ne sont pas cités. Il ne vérifie ni le lien sémantique entre une phrase et un
        document, ni la vérité de l'affirmation. Cette validation de fond restera une tâche de la campagne
        de tests manuels, à partir de questions dont la réponse attendue est connue.
    </p>
    <p>
        Les tests automatisés de cette version couvrent la validation des questions, plusieurs erreurs
        de génération, la détection des identifiants de source et les cas où aucune citation n'est
        présente. Les tests réussis sont consignés séparément des scénarios de bout en bout qui restent
        à exécuter. Les résultats de cette campagne seront ajoutés au rapport après leur observation.
    </p>

    <footer>
        Rapport généré automatiquement à partir des résultats enregistrés
        dans Data/retrieval_comparison.csv.
    </footer>

</body>
</html>
"""

# ============================================================
# 10. SAUVEGARDE
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE.write_text(
    html,
    encoding="utf-8",
)

print("======================================")
print("RAPPORT D'EVALUATION DU RETRIEVAL")
print("======================================")
print()
print(f"Questions analysées : {total}")
print(f"Recall@1 TF-IDF : {metrics['TF-IDF']['Recall@1']:.2f} %")
print(f"Recall@1 Embeddings : {metrics['Embeddings']['Recall@1']:.2f} %")
print(f"Recall@5 TF-IDF : {metrics['TF-IDF']['Recall@5']:.2f} %")
print(f"Recall@5 Embeddings : {metrics['Embeddings']['Recall@5']:.2f} %")
print(f"MRR TF-IDF : {metrics['TF-IDF']['MRR']:.4f}")
print(f"MRR Embeddings : {metrics['Embeddings']['MRR']:.4f}")
print()
if reranking_available:
    print(f"Cibles présentes dans le Top-20 : {reranking_metrics['targets_in_top20']}/{reranking_metrics['total']} ({reranking_metrics['coverage_pct']:.2f} %)")
    print(f"Cibles absentes du Top-20 : {reranking_metrics['targets_missing_top20']}/{reranking_metrics['total']} ({reranking_metrics['missing_pct']:.2f} %)")
    print(f"Hit Rate@5 après reranking : {reranking_metrics['optimized_hit5']:.2f} %")
    print(f"MRR après reranking : {reranking_metrics['optimized_mrr']:.4f}")
print(f"Rapport créé : {OUTPUT_FILE}")