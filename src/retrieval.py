from __future__ import annotations

import math
from collections import Counter

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


class LocalRetriever:
    def __init__(self):
        self.documents: list[RetrievalDocument] = []
        self._faiss_index = None
        self._embeddings = np.empty((0, 64), dtype="float32")
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self.mode = "keyword"

    def add_documents(self, documents: list[RetrievalDocument], embeddings: list[list[float]] | None = None) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        if embeddings and faiss is not None:
            matrix = np.array(embeddings, dtype="float32")
            if matrix.ndim != 2:
                return
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            self._faiss_index.add(matrix)
            self._embeddings = matrix
            self.mode = "vector"
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [doc.text for doc in self.documents]
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self.mode = "keyword"

    def query(self, text: str, top_k: int = 5) -> list[RetrievalDocument]:
        if not self.documents:
            return []
        if self.mode == "vector" and self._faiss_index is not None:
            query_vector = np.array([_mock_embedding(text, dims=self._embeddings.shape[1])], dtype="float32")
            scores, indices = self._faiss_index.search(query_vector, min(top_k, len(self.documents)))
            return [self.documents[index] for index in indices[0] if index >= 0]
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
