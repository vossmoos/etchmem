"""Content-addressed identity helpers (SHA-256)."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable


def content_hash(text: str) -> str:
    """Deterministic id for a single raw signal (idempotent intake)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_set_hash(source_ids: Iterable[str]) -> str:
    """Order-independent id for a set of source ids."""
    combined = "|".join(sorted(set(source_ids)))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def claim_hash(entity_id: str, prop: str, value_norm: str, polarity: str) -> str:
    """
    Identity of a claim = (entity, property, normalized value, polarity).

    Two signals asserting the same thing produce the SAME claim id, so they
    are recorded as corroboration (counted), not duplicated.
    """
    key = f"{entity_id}|{prop}|{value_norm}|{polarity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
