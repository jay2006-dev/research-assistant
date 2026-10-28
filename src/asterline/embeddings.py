"""Embedding abstractions and implementations for Asterline knowledge base.

Provides:
- Base protocol and Chroma adapter.
- Deterministic offline FakeEmbeddingFunction for routine unit and integration testing.
- LocalSentenceTransformerEmbedding using sentence-transformers/all-MiniLM-L6-v2.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Protocol, Sequence


class EmbeddingFunction(Protocol):
    """Protocol for embedding generation."""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a sequence of document texts."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        ...


class ChromaEmbeddingAdapter:
    """Chroma-compatible embedding function wrapper conforming to chromadb.EmbeddingFunction."""

    def __init__(self, embedder: EmbeddingFunction):
        self._embedder = embedder

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        return self._embedder.embed_documents(input)

    def embed_query(self, input: Sequence[str]) -> List[List[float]]:
        return self._embedder.embed_documents(input)

    def name(self) -> str:
        return getattr(self._embedder, "name", "custom_embedding_adapter")


class FakeEmbeddingFunction:
    """Deterministic, offline, zero-network embedding generator.

    Produces normalized 384-dimensional unit vectors derived deterministically
    from cryptographic hashes of input text. Identical texts produce identical
    vectors; different texts produce distinct, stable vectors.
    """

    DIMENSION: int = 384
    name: str = "fake_deterministic_384d"

    def _generate_vector(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.DIMENSION

        # Deterministic pseudo-random vector based on SHA-256 rounds
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = []
        for i in range(self.DIMENSION):
            byte_val = h[i % len(h)]
            # Mix in index position
            val = ((byte_val ^ (i * 31 % 256)) - 128) / 128.0
            raw.append(val)

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            return [x / norm for x in raw]
        return [1.0 / math.sqrt(self.DIMENSION)] * self.DIMENSION

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)


class LocalSentenceTransformerEmbedding:
    """Local SentenceTransformer embedding model wrapper.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run 'uv add sentence-transformers' to install it."
                ) from err
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
