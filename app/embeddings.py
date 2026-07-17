"""
Pluggable embedding providers.

The rest of the app talks only to the EmbeddingProvider ABC, never to a
specific SDK. Two built-in backends:

  - OpenAIEmbedding  (default)  — text-embedding-3-small, 1536 dims.
  - LocalEmbedding              — sentence-transformers, fully offline.

A `FakeEmbedding` is provided for tests / offline CI (deterministic hash).

Select via settings.embedding_provider ("openai" | "local"). To add a new
backend, subclass EmbeddingProvider and register it in `build_embedder`.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from app.config import settings


class EmbeddingProvider(ABC):
    """Single-purpose interface: turn text into fixed-size float vectors."""

    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors (fixed)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ── OpenAI ───────────────────────────────────────────────────────────────────

_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingProvider):
    name = "openai"

    def __init__(self, model: str, api_key: str | None) -> None:
        self._model = model
        self._dim = _OPENAI_DIMS.get(model, 1536)
        self._api_key = api_key
        self._client = None

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        # OpenAI rejects empty strings; substitute a single space.
        clean = [t if t.strip() else " " for t in texts]
        resp = client.embeddings.create(model=self._model, input=clean)
        return [d.embedding for d in resp.data]


# ── Local (sentence-transformers) ────────────────────────────────────────────

class LocalEmbedding(EmbeddingProvider):
    name = "local"

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model = None
        self._dim_cached: int | None = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dim_cached = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        if self._dim_cached is None:
            self._ensure_model()
        return int(self._dim_cached)  # type: ignore[arg-type]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=False)
        return [v.tolist() for v in vecs]


# ── Fake (deterministic, offline; tests/CI) ──────────────────────────────────

class FakeEmbedding(EmbeddingProvider):
    """Deterministic hash embedding — no network, for tests only."""

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for token in text.lower().split():
                h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
                vec[h % self._dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


# ── Factory ──────────────────────────────────────────────────────────────────

def build_embedder() -> EmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedding(settings.openai_embedding_model, settings.openai_api_key)
    if provider == "local":
        return LocalEmbedding(settings.local_embedding_model)
    if provider == "fake":
        return FakeEmbedding()
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
