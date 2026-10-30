"""Recherche documentaire hybride du RAG assurance.

Ce module isole l'étape de *retrieval* de l'API afin de pouvoir la tester et
l'évaluer indépendamment de la génération par le LLM.

Pipeline (comportement identique à la version 7.6 déployée, vérifié question
par question) :

1. Représentation TF-IDF (unigrammes + bigrammes, accents normalisés) des
   contenus, puis similarité cosinus entre la question et chaque document.
2. Score lexical indicatif : part des termes significatifs de la question
   présents dans le titre (poids 3) et dans le contenu (poids 1), avec un bonus
   lorsque deux termes consécutifs apparaissent ensemble.
3. Fusion réciproque des rangs (RRF, k = 60) :

       H(d) = 1 / (k + r_cos(d)) + 1[s_lex(d) > 0] / (k + r_lex(d))

   La contribution lexicale n'est ajoutée que si au moins un terme est partagé.
4. Étape A (pool) : les meilleurs documents lexicaux (au plus
   ``lexical_quota``) sont retenus d'office, puis le pool de 20 candidats est
   complété dans l'ordre de la fusion calculée sur tout le corpus.
5. Étape B (classement final) : la même formule RRF est réappliquée au pool,
   avec cette fois le rang cosinus *à l'intérieur du pool*. Les cinq premiers
   documents sont transmis au LLM.

Les scores produits sont des scores de classement : ce ne sont ni des
probabilités ni des mesures de véracité.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RRF_CONSTANT = 60.0
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_TOP_K = 5
DEFAULT_LEXICAL_QUOTA = 5

from src.question_parsing import (  # noqa: E402  (réexportés pour les tests)
    FRENCH_QUERY_STOPWORDS,
    ParsedQuestion,
    contains_phrase,
    normalize_text,
    normalize_tokens,
    parse_question,
)


def query_terms(question: str) -> list[str]:
    """Termes significatifs et uniques de la question, dans leur ordre d'apparition."""
    return parse_question(question, use_synonyms=False).leading_terms


def _concept_in(concept: list[str], tokens: set[str], normalized: str) -> bool:
    return any(
        (form in tokens) if " " not in form else contains_phrase(normalized, form)
        for form in concept
    )


def lexical_relevance_score(
    question: str, title: str, content: str, parsed: ParsedQuestion | None = None
) -> float:
    """Score lexical indicatif (couverture pondérée des concepts + bonus de bigrammes).

    Un concept est couvert si l'une de ses formes (terme de la question ou
    synonyme métier) apparaît. Sans ``parsed``, la question est analysée sans
    dictionnaire de synonymes.
    """
    parsed = parsed or parse_question(question, use_synonyms=False)
    if not parsed.concepts:
        return 0.0

    normalized_title = normalize_text(title)
    normalized_content = normalize_text(content)
    title_tokens = set(normalized_title.split())
    content_tokens = set(normalized_content.split())
    title_matches = sum(
        _concept_in(concept, title_tokens, normalized_title) for concept in parsed.concepts
    )
    content_matches = sum(
        _concept_in(concept, content_tokens, normalized_content) for concept in parsed.concepts
    )

    terms = parsed.leading_terms
    phrase_bonus = 0.0
    for position in range(len(terms) - 1):
        phrase = " ".join(terms[position:position + 2])
        if phrase in normalized_title:
            phrase_bonus += 2.0
        elif phrase in normalized_content:
            phrase_bonus += 1.0

    coverage = (title_matches * 3.0 + content_matches) / len(parsed.concepts)
    return round(coverage + phrase_bonus, 6)


def rank_fusion_score(
    primary_rank: int,
    lexical_rank: int,
    lexical_score: float,
    fusion_constant: float = RRF_CONSTANT,
) -> float:
    """RRF à deux listes ; le rang lexical ne compte que si le score lexical est positif."""
    primary_contribution = 1.0 / (fusion_constant + primary_rank)
    lexical_contribution = (
        1.0 / (fusion_constant + lexical_rank) if lexical_score > 0 else 0.0
    )
    return primary_contribution + lexical_contribution


@dataclass(frozen=True)
class RankedDocument:
    """Document classé, avec les signaux qui expliquent sa position."""

    index: int
    rank: int
    cosine_similarity: float
    lexical_score: float
    cosine_rank: int
    lexical_rank: int
    fusion_score: float


class HybridRetriever:
    """Index TF-IDF + signal lexical, fusionnés par RRF."""

    def __init__(
        self,
        titles: Sequence[str],
        documents: Sequence[str],
        *,
        fusion_constant: float = RRF_CONSTANT,
        lexical_quota: int = DEFAULT_LEXICAL_QUOTA,
        use_synonyms: bool = True,
    ) -> None:
        if len(titles) != len(documents):
            raise ValueError("titles et documents doivent avoir la même longueur.")
        if not documents or not any(str(doc).strip() for doc in documents):
            raise ValueError("Le corpus ne contient aucun document exploitable.")

        self.titles = [str(title) for title in titles]
        self.documents = [str(document) for document in documents]
        self.fusion_constant = fusion_constant
        self.lexical_quota = lexical_quota
        self.use_synonyms = use_synonyms
        # Titres et contenus normalisés une seule fois (gain de temps par requête).
        self._title_tokens_count = [len(normalize_tokens(title)) for title in self.titles]

        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def parse(self, question: str) -> ParsedQuestion:
        """Analyse la question (avec ou sans dictionnaire de synonymes)."""
        return parse_question(question, use_synonyms=self.use_synonyms)

    def cosine_scores(self, question: str, parsed: ParsedQuestion | None = None) -> np.ndarray:
        """Similarité cosinus TF-IDF entre la question (enrichie) et chaque document."""
        parsed = parsed or self.parse(question)
        return cosine_similarity(
            self.vectorizer.transform([parsed.expanded_text]), self.matrix
        )[0]

    def lexical_scores(self, question: str, parsed: ParsedQuestion | None = None) -> list[float]:
        parsed = parsed or self.parse(question)
        return [
            lexical_relevance_score(question, title, document, parsed)
            for title, document in zip(self.titles, self.documents)
        ]

    def select_candidates(
        self,
        question: str,
        cosine: np.ndarray,
        lexical: Sequence[float],
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> list[int]:
        """Étape A — constitue le pool de candidats sur tout le corpus.

        Les meilleurs documents lexicaux (quota) sont retenus d'office, puis le
        pool est complété dans l'ordre de la fusion RRF (rang cosinus global,
        rang lexical global).
        """
        count = len(self.documents)
        cosine_order = [int(index) for index in np.asarray(cosine).argsort()[::-1]]
        lexical_order = sorted(
            range(count),
            key=lambda i: (lexical[i], self._title_tokens_count[i]),
            reverse=True,
        )
        cosine_rank = {index: rank for rank, index in enumerate(cosine_order, start=1)}
        lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
        fused_order = sorted(
            range(count),
            key=lambda i: (
                rank_fusion_score(
                    cosine_rank[i], lexical_rank[i], lexical[i], self.fusion_constant
                ),
                lexical[i],
            ),
            reverse=True,
        )

        limit = min(candidate_limit, count)
        quota = min(self.lexical_quota, limit)
        selected = [index for index in lexical_order if lexical[index] > 0][:quota]
        selected.extend(index for index in fused_order if index not in selected)
        return selected[:limit]

    def rank(
        self,
        question: str,
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        cosine: np.ndarray | None = None,
        parsed: ParsedQuestion | None = None,
    ) -> list[RankedDocument]:
        """Retourne le pool de candidats dans l'ordre final de classement.

        Étape B — classement final du pool : le rang principal est le rang
        cosinus *à l'intérieur du pool* ; il est fusionné (RRF) avec le rang
        lexical calculé sur tout le corpus. À fusion égale, la similarité
        cosinus départage.
        """
        count = len(self.documents)
        parsed = parsed or self.parse(question)
        cosine = (
            self.cosine_scores(question, parsed)
            if cosine is None
            else np.asarray(cosine, dtype=float)
        )
        lexical = self.lexical_scores(question, parsed)
        candidates = self.select_candidates(question, cosine, lexical, candidate_limit)

        pool_by_cosine = sorted(candidates, key=lambda i: float(cosine[i]), reverse=True)
        pool_rank = {index: rank for rank, index in enumerate(pool_by_cosine, start=1)}
        lexical_order = sorted(range(count), key=lambda i: lexical[i], reverse=True)
        lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
        fusion = {
            index: rank_fusion_score(
                pool_rank[index], lexical_rank[index], lexical[index], self.fusion_constant
            )
            for index in candidates
        }
        final_order = sorted(
            pool_by_cosine, key=lambda i: (fusion[i], float(cosine[i])), reverse=True
        )

        return [
            RankedDocument(
                index=index,
                rank=position,
                cosine_similarity=float(cosine[index]),
                lexical_score=float(lexical[index]),
                cosine_rank=pool_rank[index],
                lexical_rank=lexical_rank[index],
                fusion_score=round(fusion[index], 8),
            )
            for position, index in enumerate(final_order, start=1)
        ]
