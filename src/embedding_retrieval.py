"""Recherche par embeddings multilingues (bge-m3).

Pourquoi ce module (expérience du 12/11/2026, ``evaluation/experience_embeddings.py``) :
sur les 30 questions tenues à part, jamais utilisées pour régler le système,
la bonne fiche figure parmi les 3 premières sources

* 13 fois sur 30 avec la recherche hybride TF-IDF + lexicale (43 %) ;
* 26 fois sur 30 avec bge-m3 seul (87 %), test de McNemar exact p = 0,001.

La configuration (bge-m3 seul, sans fusion avec TF-IDF) a été choisie sur les
40 reformulations de développement, puis testée une seule fois sur le jeu
tenu à part.

Les fiches sont encodées une fois (titre + 1 000 premiers caractères) et le
résultat est mis en cache sur disque, indexé par le modèle et le contenu du
corpus. Le chargement se fait dans un fil d'exécution séparé pour que le
serveur réponde à /health pendant la préparation ; tant que l'index n'est pas
prêt, l'application utilise la recherche hybride et le signale.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DOCUMENT_CHARACTERS = 1000


class EmbeddingRetriever:
    def __init__(self, titles: Sequence[str], documents: Sequence[str], *,
                 model_id: str = DEFAULT_MODEL, cache_dir: str | os.PathLike | None = None) -> None:
        self.model_id = model_id
        self.passages = [f"{t}\n{d[:DOCUMENT_CHARACTERS]}" for t, d in zip(titles, documents)]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        self.doc_vectors: np.ndarray | None = None
        self.error: str | None = None
        self._ready = threading.Event()
        self._done = threading.Event()  # chargement terminé, réussi ou non

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def _cache_file(self) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(("\x00".join(self.passages) + self.model_id).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"embeddings_{digest}.npy"

    def load(self) -> None:
        """Charge le modèle et l'index (bloquant)."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_id)
            cache = self._cache_file()
            if cache is not None and cache.is_file():
                vectors = np.load(cache)
            else:
                vectors = self.model.encode(self.passages, normalize_embeddings=True, batch_size=16)
                if cache is not None:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache, vectors)
                    except OSError as error:
                        logger.warning("Cache des embeddings non écrit : %s", error)
            self.doc_vectors = np.asarray(vectors, dtype=np.float32)
            self._ready.set()
            logger.info("Recherche par embeddings prête : %s (%s fiches).", self.model_id, len(self.passages))
        except Exception as error:  # modèle indisponible, mémoire, réseau…
            self.error = f"{type(error).__name__}: {error}"
            logger.error("Recherche par embeddings indisponible : %s", self.error)
        finally:
            self._done.set()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.load, name="embeddings", daemon=True)
        thread.start()
        return thread

    def wait(self, timeout: float | None = None) -> bool:
        """Attend la fin du chargement ; vrai si l'index est prêt."""
        self._done.wait(timeout)
        return self.ready

    def similarities(self, question: str) -> np.ndarray:
        query = self.model.encode([question], normalize_embeddings=True)[0]
        return self.doc_vectors @ query

    def rank(self, question: str, k: int) -> list[tuple[int, float]]:
        """Les k fiches les plus proches : (indice, similarité cosinus)."""
        scores = self.similarities(question)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]
