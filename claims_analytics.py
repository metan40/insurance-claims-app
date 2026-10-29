"""Analytique de démonstration pour l'aide à l'analyse des sinistres.

Le jeu de données livré est intégralement fictif. Le modèle illustre un flux
reproductible et ne doit pas servir à prendre une décision réelle d'indemnisation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_FILE = PROJECT_ROOT / "Data" / "synthetic_claims.csv"
DATASET_LABEL = "FICTIF_DEMONSTRATION"
TARGET_COLUMN = "statut"
CATEGORICAL_FEATURES = ["type_sinistre", "garantie", "canal_declaration", "dossier_complet"]
NUMERIC_FEATURES = ["montant_demande", "anciennete_client_mois", "nb_sinistres_precedents"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _load_claims() -> pd.DataFrame:
    """Charge le corpus fictif et calcule l'âge des dossiers à la date du jour."""
    frame = pd.read_csv(CLAIMS_FILE, keep_default_na=False)
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["date_declaration"] = pd.to_datetime(frame["date_declaration"], errors="coerce")
    frame["age_sinistre_jours"] = (pd.Timestamp(date.today()) - frame["date_declaration"]).dt.days
    frame["age_sinistre_jours"] = frame["age_sinistre_jours"].fillna(0).clip(lower=0).astype(int)
    frame["delai_traitement_jours"] = pd.to_numeric(frame["delai_traitement_jours"], errors="coerce")
    return frame


def _build_model() -> tuple[Pipeline, dict[str, Any], int]:
    """Évalue un modèle simple sur les seuls dossiers clos, puis l'entraîne sur eux."""
    frame = _load_claims()
    closed = frame[frame[TARGET_COLUMN].isin(["Accepted", "Rejected"])].copy()
    if len(closed) < 20 or closed[TARGET_COLUMN].nunique() < 2:
        raise RuntimeError("Le jeu de démonstration ne contient pas assez de dossiers clos pour entraîner le modèle.")

    X = closed[FEATURE_COLUMNS]
    y = (closed[TARGET_COLUMN] == "Accepted").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    evaluation_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    evaluation_model.fit(X_train, y_train)
    predicted = evaluation_model.predict(X_test)
    probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    metrics = {
        "evaluation_method": "Découpage stratifié 75/25, random_state=42, dossiers clos uniquement",
        "closed_cases_count": int(len(closed)),
        "training_cases_count": int(len(X_train)),
        "test_cases_count": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predicted)), 4),
        "f1_score": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4) if y_test.nunique() == 2 else None,
        "data_scope": "Données entièrement fictives, démonstration technique uniquement",
        "decision_policy": "Aucune décision automatique ; la validation humaine reste obligatoire.",
    }

    # Après l'évaluation tenue à part, le modèle de démonstration est réentraîné
    # sur tous les dossiers clos afin d'illustrer une probabilité sur les dossiers en cours.
    final_model = Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ]
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    final_model.fit(X, y)
    return final_model, metrics, len(closed)


_MODEL, _MODEL_METRICS, _CLOSED_CASES = _build_model()


def get_claims_summary() -> dict[str, Any]:
    """Retourne des indicateurs et exclut Pending du taux d'acceptation."""
    frame = _load_claims()
    accepted = int((frame["statut"] == "Accepted").sum())
    rejected = int((frame["statut"] == "Rejected").sum())
    pending = int((frame["statut"] == "Pending").sum())
    closed_count = accepted + rejected
    acceptance_rate = accepted / closed_count if closed_count else None
    closed_delays = frame.loc[frame["statut"].isin(["Accepted", "Rejected"]), "delai_traitement_jours"].dropna()
    return {
        "dataset_label": DATASET_LABEL,
        "total_cases": int(len(frame)),
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "pending_cases": pending,
        "closed_cases": closed_count,
        "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
        "pending_rate": round(pending / len(frame), 4) if len(frame) else None,
        "mean_closed_processing_days": round(float(closed_delays.mean()), 2) if not closed_delays.empty else None,
        "median_closed_processing_days": round(float(closed_delays.median()), 2) if not closed_delays.empty else None,
        "pending_excluded_from_acceptance_rate": True,
        "warning": "Indicateurs calculés sur un jeu entièrement fictif ; ils ne décrivent aucun portefeuille réel.",
    }


def get_claims_cases(limit: int = 120) -> list[dict[str, Any]]:
    """Retourne les dossiers de démonstration avec l'âge calculé des Pending."""
    frame = _load_claims().sort_values("claim_id")
    selected = frame.head(max(1, min(int(limit), 500)))
    records = selected.copy()
    records["date_declaration"] = records["date_declaration"].dt.strftime("%Y-%m-%d")
    records = records.astype(object).where(pd.notna(records), None)
    return records.to_dict(orient="records")


def get_claims_model_info() -> dict[str, Any]:
    """Retourne les métriques de démonstration sans revendiquer une validité métier."""
    return {
        **_MODEL_METRICS,
        "model": "Régression logistique avec prétraitement catégoriel et standardisation",
        "feature_columns": FEATURE_COLUMNS,
        "pending_used_for_training": False,
        "warning": "Les performances sur données synthétiques ne permettent pas de conclure à la performance sur de vrais sinistres.",
    }


def get_claim_analysis(claim_id: str) -> dict[str, Any] | None:
    """Fournit un score illustratif et les facteurs observables d'un dossier."""
    frame = _load_claims()
    matched = frame[frame["claim_id"].astype(str) == str(claim_id)]
    if matched.empty:
        return None
    row = matched.iloc[0]
    features = row[FEATURE_COLUMNS].to_frame().T
    probability = float(_MODEL.predict_proba(features)[0, 1])
    return {
        "claim_id": str(row["claim_id"]),
        "status": str(row["statut"]),
        "estimated_acceptance_probability": round(probability, 4),
        "observed_factors": {
            "guarantee_status": str(row["garantie"]),
            "claim_file_complete": str(row["dossier_complet"]),
            "claim_amount": float(row["montant_demande"]),
            "previous_claims": int(row["nb_sinistres_precedents"]),
            "claim_age_days": int(row["age_sinistre_jours"]),
        },
        "human_review_required": True,
        "warning": "Score exploratoire issu de données fictives ; il ne constitue ni une décision ni une recommandation d'indemnisation.",
    }
