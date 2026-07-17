"""
Service layer — wires stores, embedder, agents and the pipeline, and implements
the operations the API exposes: remember, recall, sleep, export, stats, history.

Holds the in-process Pipeline; the WorkerLoop (started by main.lifespan) calls
pipeline.run_once() on a cadence, and POST /sleep calls it on demand.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any

from app.agents import (
    ClaimExtractor, ConflictResolver, build_claim_extractor, build_conflict_resolver,
)
from app.config import settings
from app.embeddings import EmbeddingProvider, build_embedder
from app.hashing import content_hash
from app.schemas import EtchOut, RecallResult
from app.stores import S_BATCHED, S_EXTRACTED, S_NEW, Signal, Stores
from app.worker import Pipeline


class MemoryService:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        extractor: ClaimExtractor | None = None,
        resolver: ConflictResolver | None = None,
    ) -> None:
        self.embedder = embedder or build_embedder()
        self.stores = Stores(settings.data_dir, self.embedder.dim)
        # Agents are built lazily so the server boots without an LLM key
        # (only extract/fold need them).
        self._extractor = extractor
        self._resolver = resolver
        self._pipeline: Pipeline | None = None
        if extractor is not None and resolver is not None:
            self._pipeline = Pipeline(self.stores, self.embedder, extractor, resolver)

    # ── remember ────────────────────────────────────────────────────────────

    def remember(self, data, source, scope, extract_mode="deferred", metadata=None):
        sig_id = content_hash(data)
        now = time.time()
        ttl = settings.signal_ttl_seconds
        sig = Signal(
            id=sig_id, content=data, source=source, scope=scope,
            metadata=metadata or {}, embedding=self.embedder.embed_one(data),
            created_at=now, expires_at=(now + ttl if ttl > 0 else 0.0),
            status=S_NEW, extract_mode=extract_mode, canonical_id=None,
        )
        stored = self.stores.left.add_signal(sig)
        return sig_id, stored

    # ── recall ──────────────────────────────────────────────────────────────

    def recall(self, query, scope=None, source=None, top_k=5,
               include_signals=True, as_of=None) -> list[RecallResult]:
        if settings.claims_anonymization:
            # Raw signals keep the original (non-anonymized) text for
            # provenance; never surface them through retrieval.
            include_signals = False
        qvec = self.embedder.embed_one(query)
        as_of_ts = self._parse_as_of(as_of)

        results: dict[str, RecallResult] = {}
        for h in self.stores.right.search_etches(qvec, top_k=top_k, scope=scope):
            content, status, value, confidence, version = (
                h.content, h.row["status"], h.row["current_value"],
                h.row["confidence"], h.row["version"])
            updated_at = h.row.get("updated_at")
            if as_of_ts is not None:
                snap = self.stores.right.version_as_of(h.id, as_of_ts)
                if snap is None:
                    continue  # etch didn't exist yet at that time
                content, status, confidence, version = (
                    snap["narrative"], snap["status"], snap["confidence"], snap["version"])
                value = snap["current_value"]
                updated_at = snap["created_at"]
            results[h.id] = RecallResult(
                id=h.id, content=content, score=h.similarity, origin="etch",
                entity_name=h.row.get("entity_name"), property=h.row.get("property"),
                value=value, status=status, confidence=confidence, version=version,
                scope=h.row.get("scope"), source=h.row.get("source"),
                created_at=h.row.get("created_at", 0.0), updated_at=updated_at)

        if include_signals and as_of_ts is None:
            for h in self.stores.left.search_signals(qvec, top_k=top_k, scope=scope, source=source):
                results[h.id] = RecallResult(
                    id=h.id, content=h.content,
                    score=settings.recall_signal_weight * h.similarity, origin="signal",
                    source=h.row.get("source"), scope=h.row.get("scope"),
                    created_at=h.row.get("created_at", 0.0))

        return sorted(results.values(), key=lambda r: r.score, reverse=True)[:top_k]

    # ── sleep (manual tick) ──────────────────────────────────────────────────

    def sleep(self) -> dict[str, int]:
        return self._get_pipeline().run_once().to_dict()

    # ── export ────────────────────────────────────────────────────────────────

    def export(self) -> tuple[str, list[EtchOut]]:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_dir = os.path.join(settings.data_dir, "export", ts)
        os.makedirs(export_dir, exist_ok=True)
        out: list[EtchOut] = []
        for e in self.stores.right.all_etches():
            eo = EtchOut(
                id=e.id, entity_name=e.entity_name, property=e.property,
                current_value=e.current_value, status=e.status, confidence=e.confidence,
                narrative=e.narrative, version=e.version, scope=e.scope, source=e.source,
                claim_ids=e.claim_ids, source_ids=e.source_ids,
                created_at=e.created_at, updated_at=e.updated_at)
            with open(os.path.join(export_dir, f"{e.id.replace('::', '__')}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(eo.model_dump(), fh, indent=2, ensure_ascii=False)
            out.append(eo)
        return export_dir, out

    # ── history ────────────────────────────────────────────────────────────────

    def history(self, etch_id: str) -> list[dict[str, Any]]:
        return self.stores.right.versions(etch_id)

    # ── stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        left = self.stores.left
        return {
            "signals_total": left.count_signals(),
            "signals_new": left.count_signals(S_NEW),
            "signals_batched": left.count_signals(S_BATCHED),
            "signals_extracted": left.count_signals(S_EXTRACTED),
            "claims": left.count_claims(),
            "entities": self.stores.right.count_entities(),
            "etches": self.stores.right.count_etches(),
            "contested": self.stores.right.count_contested(),
            "scopes": left.scopes(),
        }

    # ── internals ────────────────────────────────────────────────────────────

    def get_pipeline(self) -> Pipeline:
        return self._get_pipeline()

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._extractor = self._extractor or build_claim_extractor()
            self._resolver = self._resolver or build_conflict_resolver()
            self._pipeline = Pipeline(self.stores, self.embedder, self._extractor, self._resolver)
        return self._pipeline

    @staticmethod
    def _parse_as_of(as_of: str | None) -> float | None:
        if not as_of:
            return None
        try:
            return datetime.datetime.fromisoformat(as_of.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return float(as_of)
            except Exception:
                return None
