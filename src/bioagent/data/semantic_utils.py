#!/usr/bin/env python3
"""
Shared semantic embedding helpers.

This module centralizes embedding model constants and lazy loading so
search/ingestion code can use one implementation pattern.
"""

from __future__ import annotations

from typing import Iterable

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_MODEL: SentenceTransformer | None = None


def normalize_semantic_text(value: str | None) -> str:
    """Normalize text for embedding lookups."""
    if not value:
        return ""
    return " ".join(value.lower().strip().split())


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


def encode_texts(values: Iterable[str]) -> list[list[float]]:
    """Encode text values into vector embeddings."""
    model = get_embedding_model()
    encoded = model.encode(list(values), show_progress_bar=False)
    return [row.tolist() for row in encoded]


def encode_query_vector(value: str) -> list[float]:
    """Encode a single query string as an embedding vector."""
    model = get_embedding_model()
    return model.encode(normalize_semantic_text(value), show_progress_bar=False).tolist()
