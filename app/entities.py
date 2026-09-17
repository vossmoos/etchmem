"""
Entity resolution — find-or-create canonical entities.

Order:
  0. declared-identifier match — when an `entities:` extension gives this
     subject type an `identifier_pattern` and the surface name carries one,
     that identifier IS the identity. Resolution is string equality and the
     fuzzy step is skipped entirely;
  1. normalized-name match against stored aliases (deterministic);
  2. embedding fuzzy match against existing entities (>= sim threshold),
     unless the type is declared `match: exact`;
  3. otherwise create a new entity.

Step 0 exists because embedding similarity is the wrong tool for catalogue
identifiers. "MSM-0808" and "MSM-0809" are different products and nearly
identical strings; any threshold loose enough to unify real name variants is
also loose enough to merge two distinct article numbers, and a merged entity
cross-contaminates two histories into confident, plausible, wrong answers.
Declared identifiers remove the guess.

`ignore: true` types resolve to None — the caller drops the claim. Use it for
identifiers that address a record rather than describe a thing (ticket numbers,
message ids), which would otherwise accumulate as thousands of junk entities.

Runs during extraction; the authoritative cross-batch merge also runs here
because the worker is single-process and owns all writes to the right DB.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config import settings
from app.embeddings import EmbeddingProvider
from app.stores import Entity, RightStore
from app.text import normalize_entity_name, slugify

if TYPE_CHECKING:
    from app.ext import ExtRegistry


@dataclass
class ResolutionStats:
    """How each subject was identified during one tick.

    Worth surfacing rather than keeping internal: `by_fuzzy` climbing on a type
    you declared with an identifier means the declaration is not matching real
    names, and that goes wrong silently. `ambiguous` counts names carrying two
    identifiers, which is a prompt or pattern problem, not a data problem.
    """
    by_key: int = 0          # declared identifier, string equality
    by_alias: int = 0        # exact normalized-name match
    by_fuzzy: int = 0        # embedding similarity — the guess
    created: int = 0         # new entity
    ignored: int = 0         # declared `ignore: true`
    ambiguous: int = 0       # two identifiers in one name, no safe key
    unmatched: int = 0       # pattern declared but name carries no identifier

    def reset(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)

    def to_dict(self) -> dict[str, int]:
        return {f"entities_{k}": v for k, v in self.__dict__.items()}


class EntityResolver:
    def __init__(
        self,
        store: RightStore,
        embedder: EmbeddingProvider,
        registry: "ExtRegistry | None" = None,
    ) -> None:
        self._store = store
        self._embed = embedder
        if registry is None:
            from app.ext import load_extensions
            registry = load_extensions()
        self._registry = registry
        self.stats = ResolutionStats()

    def resolve(self, name: str, entity_type: str, scope: str | None) -> Entity | None:
        """Canonical entity for this surface name, or None when the declared
        subject type is ignored."""
        if self._registry.is_ignored(entity_type):
            self.stats.ignored += 1
            return None

        key, reason = self._registry.canonical_key_detail(entity_type, name)
        if reason == "ambiguous":
            self.stats.ambiguous += 1
        elif reason == "no_match":
            self.stats.unmatched += 1
        norm = key or normalize_entity_name(name, entity_type)

        # 1. exact normalized-alias match
        existing = self._store.find_entity_by_alias(norm)
        if existing:
            if key is not None:
                self.stats.by_key += 1
            else:
                self.stats.by_alias += 1
            return self._maybe_add_alias(existing, norm)

        # 2. embedding fuzzy match — skipped for declared identifiers and for
        #    types declared `match: exact`, where similarity must not decide.
        vec = self._embed.embed_one(norm)
        if key is None and not self._registry.is_exact_match(entity_type):
            threshold = self._registry.sim_threshold(
                entity_type, settings.entity_sim_threshold)
            for entity, sim in self._store.search_entities(vec, top_k=3):
                if sim >= threshold:
                    self.stats.by_fuzzy += 1
                    return self._maybe_add_alias(entity, norm)

        # 3. create new
        self.stats.created += 1
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

    def resolve_key(self, key: str, entity_type: str, scope: str | None) -> Entity | None:
        """Resolve a subject from an already-extracted identifier key.

        Used when one claim named several subjects and the registry split it:
        the keys are known, so the pattern must not be re-applied (the key is
        lowercase and would no longer match an upper-case pattern).
        """
        if self._registry.is_ignored(entity_type):
            self.stats.ignored += 1
            return None
        existing = self._store.find_entity_by_alias(key)
        if existing:
            self.stats.by_key += 1
            return self._maybe_add_alias(existing, key)
        self.stats.created += 1
        now = time.time()
        entity = Entity(
            id=slugify(f"{entity_type}:{key}"), name=key.upper(), type=entity_type,
            aliases=[key], scope=scope, created_at=now, updated_at=now,
            embedding=self._embed.embed_one(key),
        )
        self._store.upsert_entity(entity)
        return entity

    def _maybe_add_alias(self, entity: Entity, norm: str) -> Entity:
        if norm not in entity.aliases:
            entity.aliases.append(norm)
            entity.updated_at = time.time()
            self._store.upsert_entity(entity)
        return entity
