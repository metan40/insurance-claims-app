"""Recherche par embeddings : classement, repli sur la recherche hybride, API.

Le vrai modèle (bge-m3, 2 Go) n'est jamais chargé ici : un encodeur factice
renvoie des vecteurs fixés, ce qui teste la logique sans réseau ni GPU.
"""
import numpy as np

from src.embedding_retrieval import EmbeddingRetriever


class FakeEncoder:
    """Vecteur de la question : composante 0 si elle parle de vélo, 1 sinon."""

    def encode(self, texts, normalize_embeddings=True, batch_size=16):
        return np.array([[1.0, 0.0] if "vélo" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


def ready_retriever(titles, documents, vectors):
    retriever = EmbeddingRetriever(titles, documents)
    retriever.model = FakeEncoder()
    retriever.doc_vectors = np.asarray(vectors, dtype=np.float32)
    retriever._ready.set()
    return retriever


def test_rank_orders_by_cosine_similarity():
    retriever = ready_retriever(["A", "B", "C"], ["a", "b", "c"], [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    ranked = retriever.rank("Mon vélo a été volé", 2)
    assert [index for index, _ in ranked] == [1, 2]
    assert ranked[0][1] > ranked[1][1]


def test_load_failure_is_recorded_not_raised(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    retriever = EmbeddingRetriever(["A"], ["a"])
    retriever.load()
    assert not retriever.ready and "absent" in retriever.error


def test_search_uses_embeddings_when_ready(app_module, monkeypatch):
    vectors = np.zeros((len(app_module.documents), 2), dtype=np.float32)
    vectors[:, 1] = 1.0
    target = next(i for i, t in enumerate(app_module.titles) if t.startswith("Est-ce qu'en cas de vol"))
    vectors[target] = [1.0, 0.0]
    monkeypatch.setattr(app_module, "embedding_retriever",
                        ready_retriever(app_module.titles, app_module.documents, vectors))
    monkeypatch.setattr(app_module, "generate_answer", lambda question, results: {
        "status": "answered", "answer": "Test [S1]", "typed_answer": None, "attempts": 1})
    response = app_module.search(app_module.Question(question="Mon vélo a été volé dans la cave"))
    assert response["retrieval_engine"] == "embeddings"
    assert response["results"][0]["title"] == app_module.titles[target]
    assert "embedding_similarity" in response["results"][0]


def test_search_falls_back_to_hybrid_while_embeddings_load(app_module, monkeypatch):
    loading = EmbeddingRetriever(app_module.titles, app_module.documents)  # jamais prêt
    monkeypatch.setattr(app_module, "embedding_retriever", loading)
    monkeypatch.setattr(app_module, "generate_answer", lambda question, results: {
        "status": "answered", "answer": "Test [S1]", "typed_answer": None, "attempts": 1})
    response = app_module.search(app_module.Question(question="Mon vélo a été volé dans la cave"))
    assert response["retrieval_engine"] == "hybrid (embeddings en préparation)"
    assert "hybrid_rank_score" in response["results"][0]
