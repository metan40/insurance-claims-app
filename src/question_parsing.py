"""Analyse de la question avant la recherche (« question parsing »).

Problème mesuré : TF-IDF et le score lexical ne reconnaissent que les mots
identiques. Lorsqu'un assuré écrit « nounou » et que la fiche dit
« baby-sitter », la bonne fiche n'est pas retrouvée.

Réponse : un dictionnaire métier relie le vocabulaire courant d'un assuré aux
termes employés par les fiches. Il a été rédigé à partir de la connaissance
du domaine (assurance habitation), sans consulter les questions du jeu tenu
à part ``evaluation/heldout_questions.csv``, figé avant sa rédaction.

La question est ensuite représentée comme une liste de *concepts* : chaque
concept est satisfait si l'une de ses formes apparaît dans un document. Une
synonymie n'augmente donc pas le nombre de concepts à couvrir, ce qui évite
de diluer le score lexical.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Mots outils et termes trop généraux pour ce corpus (presque toutes les
# fiches parlent d'« assurance habitation »).
FRENCH_QUERY_STOPWORDS = frozenset({
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "est", "la", "le", "les", "leur", "leurs",
    "mais", "mon", "ou", "par", "pas", "pour", "qu", "que", "quel",
    "quelle", "quelles", "quels", "qui", "sa", "se", "ses", "son",
    "sur", "un", "une", "vos", "votre", "vous", "comment", "fonctionne",
    "assurance", "assurances", "habitation", "contrat", "contrats",
})

# Terme employé par les fiches  ->  formulations courantes d'un assuré.
# Les formes sont écrites sans accents ; la comparaison se fait sur le texte
# normalisé (minuscules, sans accents ni ponctuation).
# Protocole : le jeu de 40 reformulations a servi de jeu de développement ;
# les formulations propres au jeu tenu à part en ont été volontairement
# exclues afin que sa mesure reste indépendante.
EXPERT_SYNONYMS: dict[str, tuple[str, ...]] = {
    # Personnes
    "baby sitter": ("nounou", "nourrice", "garde d enfants", "assistante maternelle"),
    "locataire": ("occupant", "preneur", "je loue", "en location"),
    "proprietaire": ("bailleur", "loueur", "proprio"),
    "colocataire": ("coloc", "colocation", "colocs"),
    "enfant": ("fils", "fille", "enfants", "gamin", "ado", "adolescent"),
    "etudiant": ("etudes", "fac", "universite", "eleve", "eleves"),
    # Événements
    "degat des eaux": ("fuite", "fuite d eau", "infiltration", "inondation interieure", "debordement", "canalisation percee"),
    "inondation": ("crue", "montee des eaux", "eaux debordent"),
    "catastrophe naturelle": ("catastrophes naturelles", "cat nat", "seisme", "tremblement de terre"),
    "vol": ("vole", "volee", "voles", "derobe", "derobee", "cambriolage", "cambriole", "disparu"),
    "vandalisme": ("degradation", "tag", "tags", "saccage"),
    "dommage electrique": ("court circuit", "surtension", "foudre", "grille", "panne electrique"),
    "bris de glace": ("vitre cassee", "vitre brisee", "fenetre cassee", "miroir casse"),
    "humidite": ("moisissure", "moisissures", "condensation", "salpetre"),
    "evenements climatiques": ("tempete", "grele", "neige", "orage", "intemperies"),
    "refoulement des egouts": ("reflux", "remontee d egout", "canalisations d evacuation", "egout"),
    "surconsommations d eau": ("facture d eau", "consommation d eau", "compteur d eau"),
    "sinistre": ("accident", "incident", "dommage", "dommages"),
    # Biens
    "bijoux": ("bague", "bagues", "collier", "colliers", "bracelet", "montre", "joaillerie"),
    "objet de valeur": ("objets precieux", "objet precieux"),
    "billets de banque": ("argent liquide", "especes", "liquide", "cash"),
    "panneaux solaires": ("photovoltaique", "photovoltaiques", "capteurs solaires"),
    "piscine": ("bassin", "bassin de baignade", "jacuzzi", "spa"),
    "aquarium": ("bac a poissons", "poissons"),
    "television": ("televiseur", "tele", "tv", "ecran plat"),
    "electromenager": ("frigo", "refrigerateur", "lave linge", "machine a laver", "four", "lave vaisselle"),
    "garage": ("box", "box de voiture", "parking couvert"),
    "place de stationnement": ("place de parking",),
    "clotures": ("cloture", "portail", "haie", "haies", "grillage"),
    "dependances": ("dependance", "abri de jardin", "cabanon", "remise", "cave", "grenier"),
    "tondeuse": ("materiel de jardinage", "outil de jardin", "debroussailleuse"),
    "eoliennes": ("eolienne",),
    "serrure": ("porte bloquee", "cle perdue", "cles perdues", "serrurier"),
    "biens professionnels": ("materiel professionnel", "ordinateur professionnel"),
    "boite aux lettres": ("boite aux lettre",),
    "colis": ("paquet", "livraison"),
    # Logement
    "residence secondaire": ("maison de vacances", "residence de vacances"),
    "location saisonniere": ("location de vacances", "location courte duree", "airbnb", "plateforme de location"),
    "location meublee": ("meuble", "meublee"),
    "logement de fonction": ("appartement de fonction", "maison de fonction"),
    "copropriete": ("immeuble", "syndic", "parties communes"),
    "pieces principales": ("nombre de pieces", "combien de pieces"),
    "teletravail": ("travaille depuis chez moi", "travail a domicile", "travaille a la maison", "bosse depuis chez moi"),
    "demenagement": ("demenage", "demenager", "changement de logement"),
    # Contrat
    "franchise": ("reste a ma charge", "reste a charge", "somme qui reste"),
    "plafond": ("limite de garantie", "montant maximum", "jusqu a quel montant"),
    "indemnisation": ("rembourse", "remboursement", "rembourser", "dedommage", "dedommagement", "indemnise"),
    "obligatoire": ("oblige", "obligee", "force", "forcee", "obligation", "imposee"),
    "attestation": ("justificatif", "certificat", "preuve d assurance"),
    "exclusions": ("pas pris en charge", "pas couvert", "non couvert", "ne couvre pas"),
    "garanties": ("couverture", "protection", "protege", "protegee"),
    "telesurveillance": ("alarme reliee", "centre de surveillance", "surveillance a distance", "alarme connectee"),
    "responsabilite civile": ("dommages causes aux autres", "dommages causes a autrui", "rc"),
    "resilier": ("changer d assureur", "changer d assurance", "resiliation"),
}


def normalize_text(value: object) -> str:
    """Minuscules, sans accents, ponctuation remplacée par des espaces."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", without_accents.lower()))


def normalize_tokens(value: object) -> list[str]:
    return normalize_text(value).split()


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Vrai si la phrase normalisée apparaît en mots entiers."""
    return f" {phrase} " in f" {normalized_text} "


@dataclass
class ParsedQuestion:
    """Question analysée : concepts à retrouver et vocabulaire ajouté."""

    original: str
    concepts: list[list[str]] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    leading_terms: list[str] = field(default_factory=list)

    @property
    def expanded_text(self) -> str:
        """Texte utilisé par TF-IDF : la question plus les termes métier ajoutés."""
        return " ".join([self.original, *self.expansions]).strip()


def parse_question(question: str, use_synonyms: bool = True) -> ParsedQuestion:
    """Transforme la question en concepts, enrichis par le dictionnaire métier."""
    normalized = normalize_text(question)
    terms = [
        token for token in normalized.split()
        if token not in FRENCH_QUERY_STOPWORDS and len(token) > 2
    ]
    terms = list(dict.fromkeys(terms))
    parsed = ParsedQuestion(
        original=question, concepts=[[term] for term in terms], leading_terms=terms
    )
    if not use_synonyms:
        return parsed

    for canonical, variants in EXPERT_SYNONYMS.items():
        if contains_phrase(normalized, canonical):
            continue  # la question emploie déjà le terme des fiches
        variant = next((v for v in variants if contains_phrase(normalized, v)), None)
        if variant is None:
            continue
        parsed.expansions.append(canonical)
        covered = [
            index for index, concept in enumerate(parsed.concepts)
            if concept[0] in variant.split()
        ]
        if covered:
            # La formulation de l'assuré et le terme des fiches forment un
            # seul concept : l'un ou l'autre suffit.
            keep = covered[0]
            parsed.concepts[keep] = parsed.concepts[keep] + [canonical]
            for index in reversed(covered[1:]):
                parsed.concepts[keep] += parsed.concepts[index]
                del parsed.concepts[index]
        else:
            parsed.concepts.append([canonical])
    return parsed
