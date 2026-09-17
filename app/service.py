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
from app.schemas import (
    AssociateNode, AssociateResponse, ClaimOut, DossierResponse, EdgeOut,
    KnowledgeResponse, EntityOut, EtchOut, RecallResult, SignalOut, VersionOut,
)
from app.stores import S_BATCHED, S_EXTRACTED, S_NEW, Signal, Stores
from app.worker import Pipeline


def parse_timestamp(value: str | float | None) -> float | None:
    """ISO-8601 (with or without 'Z') or epoch seconds → epoch float, else None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


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

    def remember(self, data, source, scope, extract_mode="deferred",
                 metadata=None, occurred_at=None):
        """Deposit a raw signal.

        `occurred_at` declares when the content actually happened (ISO-8601 or
        epoch seconds); it defaults to ingest time. Depositing a historical
        archive without it leaves every claim carrying today's timestamp, which
        collapses the gate's recency policy — a 2009 answer then competes with
        a 2022 answer as an equal.

        Returns (signal_id, stored, occurred_at) where occurred_at is 0.0 when
        nothing parseable was declared.
        """
        sig_id = content_hash(data)
        now = time.time()
        ttl = settings.signal_ttl_seconds
        declared = parse_timestamp(occurred_at) or 0.0
        sig = Signal(
            id=sig_id, content=data, source=source, scope=scope,
            metadata=metadata or {}, embedding=self.embedder.embed_one(data),
            created_at=now, occurred_at=declared,
            expires_at=(now + ttl if ttl > 0 else 0.0),
            status=S_NEW, extract_mode=extract_mode, canonical_id=None,
        )
        stored = self.stores.left.add_signal(sig)
        return sig_id, stored, declared

    # ── recall ──────────────────────────────────────────────────────────────

    def recall(self, query, scope=None, source=None, top_k=5,
               include_signals=True, as_of=None,
               as_of_basis: str = "ingest") -> list[RecallResult]:
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
                snap = self.stores.right.version_as_of(h.id, as_of_ts, as_of_basis)
                if snap is None:
                    continue  # etch didn't exist yet at that time
                content, status, confidence, version = (
                    snap["narrative"], snap["status"], snap["confidence"], snap["version"])
                value = snap["current_value"]
                updated_at = snap["created_at"]
            results[h.id] = RecallResult(
                id=h.id, content=content, score=h.similarity, origin="etch",
                entity_id=h.row.get("entity_id"),
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
                value_entity_id=e.value_entity_id,
                claim_ids=e.claim_ids, source_ids=e.source_ids,
                created_at=e.created_at, updated_at=e.updated_at)
            with open(os.path.join(export_dir, f"{e.id.replace('::', '__')}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(eo.model_dump(), fh, indent=2, ensure_ascii=False)
            out.append(eo)
        return export_dir, out

    # ── know ─────────────────────────────────────────────────────────

    def know(self, ref: str, as_of: str | None = None,
             entity_type: str | None = None,
             as_of_basis: str = "ingest") -> KnowledgeResponse | None:
        """Every belief about one subject. `ref` is an entity id or a name.

        Exhaustive by construction: recall ranks by embedding distance and
        returns top_k, which is the wrong tool for "what do we know about X" —
        a fact that embeds poorly against the query would simply be missing,
        and the caller could not tell the difference between absent and unranked.
        """
        entity = self._resolve_ref(ref, entity_type)
        if entity is None:
            return None
        as_of_ts = self._parse_as_of(as_of)
        out: list[EtchOut] = []
        contested = 0
        for e in self.stores.right.etches_for_entity(entity.id):
            value, status, confidence, version = (
                e.current_value, e.status, e.confidence, e.version)
            narrative, updated_at = e.narrative, e.updated_at
            if as_of_ts is not None:
                snap = self.stores.right.version_as_of(e.id, as_of_ts, as_of_basis)
                if snap is None:
                    continue           # belief did not exist yet
                value, status = snap["current_value"], snap["status"]
                confidence, version = snap["confidence"], snap["version"]
                narrative, updated_at = snap["narrative"], snap["created_at"]
            if status == "contested":
                contested += 1
            out.append(EtchOut(
                id=e.id, entity_name=e.entity_name, property=e.property,
                current_value=value, status=status, confidence=confidence,
                narrative=narrative, version=version, scope=e.scope,
                source=e.source, value_entity_id=e.value_entity_id,
                claim_ids=e.claim_ids, source_ids=e.source_ids,
                created_at=e.created_at, updated_at=updated_at))
        return KnowledgeResponse(
            entity=EntityOut(id=entity.id, name=entity.name, type=entity.type,
                             aliases=entity.aliases, scope=entity.scope),
            etches=out, count=len(out), as_of=as_of, as_of_basis=as_of_basis,
            contested=contested)

    def _resolve_ref(self, ref: str, entity_type: str | None):
        """An entity id, or a surface name resolved the way ingestion resolved it."""
        found = self.stores.right.get_entity(ref)
        if found is not None:
            return found
        from app.ext import load_extensions
        from app.text import normalize_entity_name
        registry = load_extensions()
        # Try the declared identifier first, then the plain normalized name, so
        # "MSM 0808" finds the entity that ingestion keyed as "msm-0808".
        candidates: list[str] = []
        types = [entity_type] if entity_type else [e.type for e in registry.entities] + [None]
        for t in types:
            key = registry.canonical_key(t, ref) if t else None
            if key:
                candidates.append(key)
        candidates.append(normalize_entity_name(ref, entity_type))
        for norm in candidates:
            hit = self.stores.right.find_entity_by_alias(norm)
            if hit is not None:
                return hit
        return None

    # ── associate ────────────────────────────────────────────────────────────

    def associate(self, query: str, scope: str | None = None, top_k: int = 5,
                  hops: int = 1, as_of: str | None = None,
                  min_score: float = 0.15,
                  as_of_basis: str = "ingest") -> AssociateResponse:
        """What the memory holds for a phrase.

        Not a lookup of a thing you already named — the caller may not know
        which product, or may not be asking about products at all. Three steps:

          1. the beliefs whose narratives match the phrase (ranked, semantic);
          2. every belief held about each subject behind them (exhaustive);
          3. the nodes those subjects are connected to, followed through
             declared relations in BOTH directions.

        Step 3 is what a similarity search cannot do. "Bathroom leaks" finds
        the fault; the seal that fixes it is connected to that module by a
        relation, not by wording, and no amount of embedding gets you there.

        `min_score` drops seeds that do not actually match. `recall` returns
        `top_k` rows whatever their similarity, which is fine when a human reads
        a ranked list — but here a zero-scoring belief would seed the graph and
        drag its whole neighbourhood in, so the answer would stop meaning
        anything.
        """
        seeds = [h for h in self.recall(query, scope=scope, top_k=top_k,
                                        include_signals=False, as_of=as_of,
                                        as_of_basis=as_of_basis)
                 if h.score >= min_score]
        nodes: dict[str, AssociateNode] = {}
        edges: list[EdgeOut] = []

        def add(entity_id: str, hop: int, score: float, via: str | None) -> None:
            ent = self.stores.right.get_entity(entity_id)
            if ent is None:
                return
            node = nodes.get(entity_id)
            if node is not None:
                if score > node.score:
                    node.score = score
                node.hops = min(node.hops, hop)
                return
            facts = self.know(entity_id, as_of=as_of, as_of_basis=as_of_basis)
            nodes[entity_id] = AssociateNode(
                entity=EntityOut(id=ent.id, name=ent.name, type=ent.type,
                                 aliases=ent.aliases, scope=ent.scope),
                etches=facts.etches if facts else [], hops=hop, score=score,
                reached_via=via)

        for hit in seeds:
            if hit.entity_id:
                add(hit.entity_id, 0, hit.score, None)

        if hops >= 1:
            for entity_id in list(nodes):
                for e in self.stores.right.etches_for_entity(entity_id):
                    if not e.value_entity_id:
                        continue
                    target = self.stores.right.get_entity(e.value_entity_id)
                    if target is None:
                        continue
                    edges.append(EdgeOut(
                        etch_id=e.id, from_entity=entity_id, property=e.property,
                        to_entity=target.id, to_name=target.name, status=e.status,
                        confidence=e.confidence, direction="out"))
                    add(target.id, 1, nodes[entity_id].score, e.property)
                for e in self.stores.right.etches_referencing(entity_id):
                    src = self.stores.right.get_entity(e.entity_id)
                    if src is None:
                        continue
                    edges.append(EdgeOut(
                        etch_id=e.id, from_entity=src.id, property=e.property,
                        to_entity=entity_id, to_name=nodes[entity_id].entity.name,
                        status=e.status, confidence=e.confidence, direction="in"))
                    add(src.id, 1, nodes[entity_id].score, e.property)

        ordered = sorted(nodes.values(), key=lambda n: (n.hops, -n.score))
        contested = sum(1 for n in ordered for e in n.etches if e.status == "contested")
        return AssociateResponse(query=query, seeds=seeds, nodes=ordered,
                                 edges=edges, as_of=as_of, contested=contested)

    # ── history ────────────────────────────────────────────────────────────────

    def history(self, etch_id: str) -> list[dict[str, Any]]:
        return self.stores.right.versions(etch_id)

    # ── dossier ────────────────────────────────────────────────────────────────

    def dossier(self, etch_id: str) -> DossierResponse | None:
        """Full provenance for one belief: etch + versions + claims + signals."""
        e = self.stores.right.get_etch(etch_id)
        if e is None:
            return None
        etch = EtchOut(
            id=e.id, entity_name=e.entity_name, property=e.property,
            current_value=e.current_value, status=e.status, confidence=e.confidence,
            narrative=e.narrative, version=e.version, scope=e.scope, source=e.source,
            value_entity_id=e.value_entity_id,
            claim_ids=e.claim_ids, source_ids=e.source_ids,
            created_at=e.created_at, updated_at=e.updated_at)
        versions = [VersionOut(**v) for v in self.stores.right.versions(etch_id)]
        claims = [
            ClaimOut(
                id=c.id, entity_name=c.entity_name, property=c.property,
                value=c.value, polarity=c.polarity,
                corroboration_count=c.corroboration_count, confidence=c.confidence,
                sources=c.sources, evidence_signal_ids=c.evidence_signal_ids,
                scope=c.scope, event_time=c.event_time, ingest_time=c.ingest_time,
                created_at=c.created_at)
            for c in self.stores.left.claims_by_ids(e.claim_ids)]
        signals: list[SignalOut] = []
        signals_omitted = False
        if settings.claims_anonymization:
            # Raw signals keep the original (non-anonymized) text — never surface.
            signals_omitted = True
        else:
            sig_ids = sorted({sid for c in claims for sid in c.evidence_signal_ids}
                             | set(e.source_ids))
            signals = [
                SignalOut(id=s.id, content=s.content, source=s.source,
                          scope=s.scope, created_at=s.created_at,
                          occurred_at=s.event_at)
                for s in self.stores.left.signals_by_ids(list(sig_ids))]
        return DossierResponse(etch=etch, versions=versions, claims=claims,
                               signals=signals, signals_omitted=signals_omitted)

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
        return parse_timestamp(as_of)
