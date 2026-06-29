from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.schemas import RetrievalDocument
from src.utils import stable_seed

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


def _mock_embedding(text: str, dims: int = 64) -> list[float]:
    rng = np.random.default_rng(stable_seed(text))
    vector = rng.random(dims)
    norm = np.linalg.norm(vector) or 1.0
    return (vector / norm).tolist()


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalRetriever:
    def __init__(self, query_embedder: Callable[[str], list[float]] | None = None):
        self.documents: list[RetrievalDocument] = []
        self._faiss_index = None
        self._embeddings = np.empty((0, 64), dtype="float32")
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._query_embedder = query_embedder
        self.mode = "keyword"

    def add_documents(self, documents: list[RetrievalDocument], embeddings: list[list[float]] | None = None) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [doc.text for doc in self.documents]
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

        if embeddings and faiss is not None:
            matrix = np.array(embeddings, dtype="float32")
            if matrix.ndim != 2 or matrix.shape[0] != len(documents):
                self.mode = "keyword"
                return
            matrix = _normalize_matrix(matrix)
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            elif self._embeddings.size and matrix.shape[1] != self._embeddings.shape[1]:
                self.mode = "keyword"
                return
            self._faiss_index.add(matrix)
            if self._embeddings.size == 0:
                self._embeddings = matrix
            else:
                self._embeddings = np.vstack([self._embeddings, matrix])
            self.mode = "vector"
            return
        self.mode = "keyword"

    def query(self, text: str, top_k: int = 5) -> list[RetrievalDocument]:
        if not self.documents:
            return []
        if self.mode == "vector" and self._faiss_index is not None and self._query_embedder is not None:
            try:
                query_vector = np.array([self._query_embedder(text)], dtype="float32")
                query_vector = _normalize_matrix(query_vector)
                scores, indices = self._faiss_index.search(query_vector, min(top_k, len(self.documents)))
                results = [self.documents[index] for index in indices[0] if index >= 0]
                if results:
                    return results
            except Exception:
                pass
        if self._vectorizer is None or self._tfidf_matrix is None:
            return self._keyword_fallback(text, top_k)
        query_matrix = self._vectorizer.transform([text])
        scores = (self._tfidf_matrix @ query_matrix.T).toarray().ravel()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [self.documents[index] for index in ranked if scores[index] > 0]

    def _keyword_fallback(self, text: str, top_k: int) -> list[RetrievalDocument]:
        query_terms = Counter(text.lower().split())
        scored: list[tuple[float, RetrievalDocument]] = []
        for doc in self.documents:
            doc_terms = Counter(doc.text.lower().split())
            overlap = sum(min(query_terms[word], doc_terms[word]) for word in query_terms)
            length_penalty = math.log(len(doc.text.split()) + 1)
            scored.append((overlap / max(length_penalty, 1.0), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]
