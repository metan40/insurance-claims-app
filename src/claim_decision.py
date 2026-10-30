"""Aide à la décision sur un dossier : relier le score, le dossier et le contrat.

Pour un dossier, l'application rassemble en une seule vue ce dont le
gestionnaire a besoin pour décider :

1. le **score expliqué** : probabilité indicative et facteurs qui la
   font monter ou baisser (contributions de la régression logistique) ;
2. les **points à vérifier**, issus de règles métier explicites et lisibles
   (dossier incomplet, garantie à vérifier, montant élevé, ancienneté du
   Pending, historique chargé) ;
3. les **clauses contractuelles** pertinentes pour ce type de sinistre :
   couverture, exclusions, indemnisation, franchise ;
4. une **orientation** pour le gestionnaire, jamais une décision automatique.

Choix de conception : les clauses sont choisies par une table de
correspondance « type de sinistre → fiches », établie à la main à partir des
titres du corpus, plutôt que par une recherche libre. Les essais de recherche
libre ramenaient des fiches voisines mais fausses (par exemple « assurance de
garage » pour un bris de glace). Une table explicite est déterministe,
vérifiable par un expert métier, et permet de signaler honnêtement les
**trous du corpus** : il n'existe aucune fiche dédiée à l'incendie ni au bris
de glace en général.
"""
from __future__ import annotations

from typing import Any, Sequence

# Type de sinistre -> liste de (rubrique, titre exact de la fiche dans le corpus).
CLAUSES_BY_CLAIM_TYPE: dict[str, list[tuple[str, str]]] = {
    "Dégâts des eaux": [
        ("Couverture", "Que couvre la garantie dégâts des eaux de l’assurance habitation ?"),
        ("Exclusions", "Les exclusions de la garantie dégât des eaux"),
        ("Indemnisation", "Comment être indemnisé après un dégât des eaux ?"),
    ],
    "Vol": [
        ("Couverture", "Est-ce qu'en cas de vol je suis couvert ou pas ?"),
        ("Exclusions", "En cas de vol par une personne de ma famille, suis-je couvert ?"),
        ("Points d’attention", "Que sont les objets à risque de vol ?"),
    ],
    "Catastrophe naturelle": [
        ("Couverture", "Les dégâts des eaux liés à une catastrophe naturelle"),
        ("Qualification", "Quelle est la différence entre inondation et catastrophes naturelles ?"),
        ("Indemnisation", "Comment percevoir les indemnisations en cas d’inondation ?"),
    ],
    "Responsabilité civile": [
        ("Couverture", "Que couvre la responsabilité civile de l’assurance habitation ?"),
        ("Exclusions", "Que ne couvre pas l’assurance responsabilité civile ?"),
        ("Démarche", "Comment bénéficier de votre assurance responsabilité civile ?"),
    ],
    "Bris de glace": [
        ("Exclusions", "J'étais en train de changer mes fenêtres, et pendant les travaux une vitre s'est brisée. Suis-je couvert ?"),
        ("Cas particulier", "L'écran de mon téléviseur s'est brisé. Suis-je couvert ?"),
    ],
    "Incendie": [],
}
COMMON_CLAUSES = [
    ("Franchise", "Comment fonctionne la franchise d’une assurance habitation ?"),
    ("Exclusions générales", "Ce que le contrat ne garantit jamais"),
]
CORPUS_GAPS = {
    "Incendie": "Le corpus ne contient aucune fiche dédiée à la garantie incendie : "
                "vérifier directement les conditions générales du contrat.",
    "Bris de glace": "Le corpus ne contient pas de fiche générale sur la garantie bris de "
                     "glace, seulement des cas particuliers : vérifier les conditions générales.",
}
EXCERPT_CHARACTERS = 320

# Seuils des règles métier (explicites pour pouvoir être discutés et ajustés).
HIGH_AMOUNT_QUANTILE = 0.90
PENDING_WATCH_DAYS = 30
PENDING_ALERT_DAYS = 90
HEAVY_HISTORY = 3
FAVOURABLE_PROBABILITY = 0.70
UNFAVOURABLE_PROBABILITY = 0.40


def excerpt(text: str, limit: int = EXCERPT_CHARACTERS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def contract_clauses(claim_type: str, titles: Sequence[str], documents: Sequence[str]) -> list[dict]:
    """Fiches contractuelles à lire pour ce type de sinistre."""
    index_by_title = {title: index for index, title in enumerate(titles)}
    clauses = []
    for heading, title in CLAUSES_BY_CLAIM_TYPE.get(claim_type, []) + COMMON_CLAUSES:
        index = index_by_title.get(title)
        if index is None:
            continue
        clauses.append({
            "heading": heading,
            "title": title,
            "excerpt": excerpt(documents[index]),
            "question_for_assistant": title,
        })
    return clauses


def checks_for_claim(claim: dict[str, Any], amount_threshold: float) -> list[dict]:
    """Points à vérifier, avec leur niveau : « bloquant » ou « attention »."""
    checks = []
    if str(claim["dossier_complet"]) != "Oui":
        checks.append({"level": "bloquant", "message":
                       "Dossier incomplet : demander les pièces manquantes avant toute décision."})
    if str(claim["garantie"]) != "Oui":
        checks.append({"level": "bloquant", "message":
                       "Garantie à vérifier : confirmer l’applicabilité dans le contrat "
                       "(voir les clauses de couverture et d’exclusion ci-dessous)."})
    if float(claim["montant_demande"]) > amount_threshold:
        checks.append({"level": "attention", "message":
                       f"Montant élevé (au-dessus de {amount_threshold:,.0f} €, soit 90 % des "
                       "dossiers) : contrôle renforcé ou expertise.".replace(",", " ")})
    if int(claim["nb_sinistres_precedents"]) >= HEAVY_HISTORY:
        checks.append({"level": "attention", "message":
                       f"{int(claim['nb_sinistres_precedents'])} sinistres précédents : "
                       "vérifier la cohérence avec l’historique."})
    if str(claim["statut"]) == "Pending":
        age = int(claim["age_sinistre_jours"])
        if age > PENDING_ALERT_DAYS:
            checks.append({"level": "attention", "message":
                           f"En attente depuis {age} jours : anomalie de traitement probable, à escalader."})
        elif age > PENDING_WATCH_DAYS:
            checks.append({"level": "attention", "message":
                           f"En attente depuis {age} jours : à relancer."})
    return checks


def orientation(probability: float, checks: list[dict], claim_type: str) -> dict:
    """Orientation proposée au gestionnaire (pas une décision)."""
    if any(check["level"] == "bloquant" for check in checks):
        return {"code": "completer", "label": "Compléter le dossier avant décision",
                "reason": "Au moins un point bloquant doit être levé ; le score n’est pas exploitable seul."}
    if claim_type in CORPUS_GAPS:
        return {"code": "verifier_contrat", "label": "Vérifier le contrat avant décision",
                "reason": CORPUS_GAPS[claim_type]}
    if probability >= FAVOURABLE_PROBABILITY and not checks:
        return {"code": "favorable", "label": "Éléments favorables : valider après lecture des clauses",
                "reason": "Dossier complet, garantie applicable, aucun signal d’attention, score élevé."}
    if probability < UNFAVOURABLE_PROBABILITY:
        return {"code": "defavorable", "label": "Éléments défavorables : examiner les exclusions",
                "reason": "Le score est bas : identifier la clause ou le facteur qui justifierait un refus."}
    return {"code": "approfondir", "label": "Analyse approfondie recommandée",
            "reason": "Signaux mitigés : le score et les points d’attention ne permettent pas de trancher."}


def build_decision_support(
    analysis: dict, claim: dict, explanation: list[dict], amount_threshold: float,
    titles: Sequence[str], documents: Sequence[str],
) -> dict:
    """Assemble la vue d'aide à la décision d'un dossier."""
    claim_type = str(claim["type_sinistre"])
    checks = checks_for_claim(claim, amount_threshold)
    return {
        "claim_id": analysis["claim_id"],
        "claim_type": claim_type,
        "status": analysis["status"],
        "amount": float(claim["montant_demande"]),
        "estimated_acceptance_probability": analysis["estimated_acceptance_probability"],
        "score_factors": explanation,
        "checks": checks,
        "contract_clauses": contract_clauses(claim_type, titles, documents),
        "corpus_gap": CORPUS_GAPS.get(claim_type),
        "orientation": orientation(analysis["estimated_acceptance_probability"], checks, claim_type),
        "human_review_required": True,
        "warning": (
            "Aide à la décision sur données fictives : l’orientation résume les éléments "
            "disponibles, elle ne remplace ni la lecture du contrat ni la décision du gestionnaire."
        ),
    }
