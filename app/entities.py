"""
Entity resolution — find-or-create canonical entities.

Order:
  1. normalized-name match against stored aliases (deterministic);
  2. embedding fuzzy match against existing entities (>= entity_sim_threshold);
  3. otherwise create a new entity.

Runs during extraction; the authoritative cross-batch merge also runs here
because the worker is single-process and owns all writes to the right DB.
"""
from __future__ import annotations

import time

from app.config import settings
from app.embeddings import EmbeddingProvider
from app.stores import Entity, RightStore
from app.text import normalize_entity_name, slugify


class EntityResolver:
    def __init__(self, store: RightStore, embedder: EmbeddingProvider) -> None:
        self._store = store
        self._embed = embedder

    def resolve(self, name: str, entity_type: str, scope: str | None) -> Entity:
        norm = normalize_entity_name(name, entity_type)

        # 1. exact normalized-alias match
        existing = self._store.find_entity_by_alias(norm)
        if existing:
            return self._maybe_add_alias(existing, norm)

        # 2. embedding fuzzy match
        vec = self._embed.embed_one(norm)
        for entity, sim in self._store.search_entities(vec, top_k=3):
            if sim >= settings.entity_sim_threshold:
                return self._maybe_add_alias(entity, norm)

        # 3. create new
        now = time.time()
        entity = Entity(
            id=slugify(f"{entity_type}:{norm}"),
            name=name,
            type=entity_type,
            aliases=[norm],
            scope=scope,
            created_at=now,
            updated_at=now,
            embedding=vec,
        )
        self._store.upsert_entity(entity)
        return entity

    def _maybe_add_alias(self, entity: Entity, norm: str) -> Entity:
        if norm not in entity.aliases:
            entity.aliases.append(norm)
            entity.updated_at = time.time()
            self._store.upsert_entity(entity)
        return entity
