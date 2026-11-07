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
import uuid
from typing import Any

from app.agents import (
    ClaimExtractor, ConflictResolver, build_claim_extractor, build_conflict_resolver,
)
from app.config import settings
from app.embeddings import EmbeddingProvider, build_embedder
from app.hashing import content_hash
from app.quotes import QuoteRequest, quotes_for
from app.text import normalize_value
from app.schemas import (
    AssociateNode, AssociateResponse, ClaimOut, DossierResponse, EdgeOut,
    EtchListResponse, KnowledgeResponse, EntityOut, EtchOut, ListedEtchOut, Quote, RecallResult,
    ContextResponse, ContextSignalOut, ContextSubjectOut, ForgetResponse, SignalOut,
    SignalRecordResponse, VersionOut,
)
from app.stores import C_NEW, ERASED, S_BATCHED, S_EXTRACTED, S_NEW, Etch, Signal, Stores
from app.worker import Pipeline, TickSummary


ETCHES_DEFAULT_LIMIT = 500
ETCHES_MAX_LIMIT = 2000


def _as_list(value: str | list[str] | None) -> list[str] | None:
    """A filter given as one name, a comma-separated string or a list → a list, or None."""
    if value is None:
        return None
    items = [value] if isinstance(value, str) else value
    out = [part.strip() for v in items for part in str(v).split(",") if part.strip()]
    return out or None


def _scopes(value: str | list[str] | None) -> str | list[str] | None:
    """A scope filter: None/"" = every scope, "a" or "a,b" or ["a", "b"] = any of
    them, [] = none."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else parts
    return [p.strip() for v in value for p in str(v).split(",") if p.strip()]


def _in_scope(scope: str | None, allowed: str | list[str] | None) -> bool:
    if allowed is None:
        return True
    if isinstance(allowed, str):
        return scope == allowed
    return scope in allowed


def _writers_of(source: str | None) -> set[str]:
    """The writers behind a belief: its `source`, "hubspot, mossy" -> {hubspot, mossy}."""
    return {w.strip() for w in (source or "").split(",") if w.strip()}


def _readable(source: str | None, writers: list[str] | None) -> bool:
    """None = no filter. Otherwise EVERY writer behind the belief must be allowed, so
    knowledge built partly from a source the reader may not see is not shown at all."""
    if writers is None:
        return True
    return _writers_of(source) <= set(writers)


def _reduce(belief, shown: set[str]) -> None:
    """A belief seen through some of its writers only (see MemoryService._views): keep
    the value those writers state, drop what other writers added (wording, evidence,
    other values, their names)."""
    value = belief.current_value if hasattr(belief, "current_value") else belief.value
    plain = f"{belief.entity_name} — {(belief.property or '').replace('_', ' ')}: {value}."
    belief.source = ", ".join(sorted(shown))
    belief.details, belief.evidence, belief.values = [], None, []
    if hasattr(belief, "narrative"):
        belief.narrative = plain
        belief.claim_ids, belief.source_ids = [], []
    else:
        belief.content = plain


def _etch_out(e: Etch, **override) -> EtchOut:
    fields = dict(
        id=e.id, entity_name=e.entity_name, property=e.property,
        current_value=e.current_value, status=e.status, confidence=e.confidence,
        narrative=e.narrative, version=e.version, scope=e.scope, source=e.source,
        value_entity_id=e.value_entity_id, details=e.details,
        evidence=e.evidence or None, values=e.values,
        claim_ids=e.claim_ids, source_ids=e.source_ids,
        created_at=e.created_at, updated_at=e.updated_at)
    fields.update(override)
    return EtchOut(**fields)


def _claim_out(c) -> ClaimOut:
    return ClaimOut(
        id=c.id, entity_name=c.entity_name, property=c.property,
        value=c.value, polarity=c.polarity, detail=c.detail,
        corroboration_count=c.corroboration_count, confidence=c.confidence,
        sources=c.sources, evidence_signal_ids=c.evidence_signal_ids,
        scope=c.scope, event_time=c.event_time,
        ingest_time=c.ingest_time, created_at=c.created_at)


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
        self.stores = Stores(settings.data_dir, self.embedder.dim,
                             database_url=settings.database_url or None,
                             schema=settings.database_schema or None)
        # Agents are built lazily so the server boots without an LLM key
        # (only extract/fold need them).
        self._extractor = extractor
        self._resolver = resolver
        self._pipeline: Pipeline | None = None
        if extractor is not None and resolver is not None:
            self._pipeline = Pipeline(self.stores, self.embedder, extractor, resolver)

    # ── remember ────────────────────────────────────────────────────────────

    def remember(self, data, source, scope, extract_mode="deferred",
                 metadata=None, occurred_at=None, context=None, context_parent=None,
                 source_record_id=None, actor=None):
        """Deposit a raw signal.

        `occurred_at` declares when the content actually happened (ISO-8601 or
        epoch seconds); it defaults to ingest time. Depositing a historical
        archive without it leaves every claim carrying today's timestamp, which
        collapses the gate's recency policy — a 2009 answer then competes with
        a 2022 answer as an equal.

        `context` / `context_parent` say where it was said (a thread and its
        channel, a ticket's comments, a meeting), as the client addresses it. The
        same words said in another conversation are another signal.

        `source_record_id` and `actor` are recorded as given: where the signal
        came from in its source system, and who deposited it.

        Returns (signal_id, stored, occurred_at) where occurred_at is 0.0 when
        nothing parseable was declared. A signal that was erased before is not
        stored again (`was_erased` tells it apart from a duplicate).

        A signal's identity is its writer plus its text: the same words deposited
        again by the same writer are a duplicate, but two writers depositing the same
        words (two agents shown the same ticket) are two signals, each in its writer's
        memory. Ids from before 0.4 had no writer in them; an erasure recorded under
        such an id still refuses the text.
        """
        payload = data if not context else f"{context}\n{data}"
        sig_id = content_hash(f"{source}\x00{payload}")
        for erased_id in (sig_id, content_hash(payload)):
            if self.stores.right.is_erased(erased_id):
                return erased_id, False, parse_timestamp(occurred_at) or 0.0
        now = time.time()
        ttl = settings.signal_ttl_seconds
        declared = parse_timestamp(occurred_at) or 0.0
        sig = Signal(
            id=sig_id, content=data, source=source, scope=scope,
            metadata=metadata or {}, embedding=self.embedder.embed_one(data),
            created_at=now, occurred_at=declared,
            expires_at=(now + ttl if ttl > 0 else 0.0),
            status=S_NEW, extract_mode=extract_mode, canonical_id=None,
            context_id=context or None, context_parent=context_parent or None,
            source_record_id=source_record_id or None, actor=dict(actor or {}),
        )
        stored = self.stores.left.add_signal(sig)
        return sig_id, stored, declared

    def was_erased(self, signal_id: str) -> bool:
        return self.stores.right.is_erased(signal_id)

    # ── recall ──────────────────────────────────────────────────────────────

    def recall(self, query, scope=None, source=None, top_k=5,
               include_signals=True, as_of=None,
               as_of_basis: str = "ingest", quotes: str = "full",
               writers: list[str] | None = None) -> list[RecallResult]:
        if settings.claims_anonymization:
            # Raw signals keep the original (non-anonymized) text for
            # provenance; never surface them through retrieval.
            include_signals = False
        scope = _scopes(scope)
        qvec = self.embedder.embed_one(query)
        as_of_ts = self._parse_as_of(as_of)

        writers = _as_list(writers) if writers is not None else None
        results: dict[str, RecallResult] = {}
        # With a writer filter, rank a wider pool so filtered-out beliefs do not eat top_k.
        pool = top_k if writers is None else min(top_k * 5, 200)
        candidates = self.stores.right.search_etches(qvec, top_k=pool, scope=scope)
        partial = [] if writers is None else [
            e for e in (self.stores.right.get_etch(h.id) for h in candidates
                        if not _readable(h.row.get("source"), writers)
                        and _writers_of(h.row.get("source")) & set(writers)) if e is not None]
        views = self._views(partial, writers)
        hits = [h for h in candidates
                if _readable(h.row.get("source"), writers) or h.id in views][:top_k]
        for h in hits:
            content, status, value, confidence, version = (
                h.content, h.row["status"], h.row["current_value"],
                h.row["confidence"], h.row["version"])
            details = h.row.get("details") or []
            evidence = h.row.get("evidence") or None
            values = h.row.get("values") or []
            updated_at = h.row.get("updated_at")
            if as_of_ts is not None:
                snap = self.stores.right.version_as_of(h.id, as_of_ts, as_of_basis)
                if snap is None:
                    continue  # etch didn't exist yet at that time
                content, status, confidence, version = (
                    snap["narrative"], snap["status"], snap["confidence"], snap["version"])
                value = snap["current_value"]
                details = snap.get("details") or []
                evidence = snap.get("evidence") or None
                values = snap.get("values") or []
                updated_at = snap["created_at"]
            score = h.similarity
            if values and not any(v.get("active", True) for v in values):
                # Retracted beliefs stay findable, at much lower priority.
                score *= min(max(settings.retraction_weight, 0.0), 1.0)
            results[h.id] = RecallResult(
                id=h.id, content=content, score=score, origin="etch",
                entity_id=h.row.get("entity_id"),
                entity_name=h.row.get("entity_name"), property=h.row.get("property"),
                value=value, status=status, confidence=confidence, version=version,
                scope=h.row.get("scope"), source=h.row.get("source"),
                details=details, evidence=evidence, values=values,
                created_at=h.row.get("created_at", 0.0), updated_at=updated_at)
            if h.id in views:
                _reduce(results[h.id], views[h.id])

        self._attach_quotes(list(results.values()), as_of_ts, as_of_basis,
                            full=quotes == "full", writers=writers)

        if include_signals and as_of_ts is None:
            for h in self.stores.left.search_signals(qvec, top_k=top_k, scope=scope, source=source):
                if writers is not None and h.row.get("source") not in writers:
                    continue
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
            eo = _etch_out(e)
            with open(os.path.join(export_dir, f"{e.id.replace('::', '__')}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(eo.model_dump(), fh, indent=2, ensure_ascii=False)
            out.append(eo)
        return export_dir, out

    # ── etches (filtered list) ───────────────────────────────────────────────

    def etches(self, *, scope: str | list[str] | None = None,
               property: str | list[str] | None = None,
               entity_type: str | list[str] | None = None,
               status: str | None = None,
               min_confidence: float | None = None,
               limit: int = ETCHES_DEFAULT_LIMIT, offset: int = 0,
               writers: list[str] | None = None) -> EtchListResponse:
        """Every belief matching the filters, newest first, one page at a time.

        For "all incorrect_assumption beliefs in scope X": exhaustive where
        `recall` ranks and cuts at top_k, and across subjects where `know`
        covers one. Unlike `export` it filters and writes nothing to disk.
        """
        limit = max(1, min(int(limit), ETCHES_MAX_LIMIT))
        offset = max(0, int(offset))
        filters = dict(scope=_scopes(scope), properties=_as_list(property),
                       entity_types=_as_list(entity_type), status=status or None,
                       min_confidence=min_confidence)
        if writers is None:
            rows, total = self.stores.right.list_etches(**filters, limit=limit, offset=offset)
        else:
            # Filter by writer, then page: totals count only what the reader may see.
            allowed = _as_list(writers) or []
            every, _ = self.stores.right.list_etches(**filters, limit=ETCHES_MAX_LIMIT * 10, offset=0)
            views = self._views([e for e, _ in every if not _readable(e.source, allowed)], allowed)
            every = [(e, t) for e, t in every if _readable(e.source, allowed) or e.id in views]
            total, rows = len(every), every[offset:offset + limit]
        out = [ListedEtchOut(
            id=e.id, entity_name=e.entity_name, entity_type=etype, property=e.property,
            current_value=e.current_value, status=e.status, confidence=e.confidence,
            narrative=e.narrative, version=e.version, scope=e.scope, source=e.source,
            value_entity_id=e.value_entity_id, details=e.details,
            evidence=e.evidence or None, values=e.values,
            claim_ids=e.claim_ids, source_ids=e.source_ids,
            created_at=e.created_at, updated_at=e.updated_at) for e, etype in rows]
        if writers is not None:
            for item in out:
                if item.id in views:
                    _reduce(item, views[item.id])
        return EtchListResponse(etches=out, count=len(out), total=total,
                                limit=limit, offset=offset)

    # ── know ─────────────────────────────────────────────────────────

    def know(self, ref: str, as_of: str | None = None,
             entity_type: str | None = None,
             as_of_basis: str = "ingest",
             scope: str | list[str] | None = None,
             quotes: str = "full",
             writers: list[str] | None = None) -> KnowledgeResponse | None:
        """Every belief about one subject. `ref` is an entity id or a name.

        Exhaustive by construction: recall ranks by embedding distance and
        returns top_k, which is the wrong tool for "what do we know about X" —
        a fact that embeds poorly against the query would simply be missing,
        and the caller could not tell the difference between absent and unranked.
        """
        subjects = self._resolve_refs(ref, entity_type, scope)
        if not subjects:
            return None
        entity = subjects[0]
        as_of_ts = self._parse_as_of(as_of)
        out: list[EtchOut] = []
        contested = 0
        writers = _as_list(writers) if writers is not None else None
        held = [e for subj in subjects
                for e in self.stores.right.etches_for_entity(subj.id, scope=_scopes(scope))]
        views = self._views([e for e in held if not _readable(e.source, writers)], writers)
        beliefs = [e for e in held if _readable(e.source, writers) or e.id in views]
        for e in beliefs:
            value, status, confidence, version = (
                e.current_value, e.status, e.confidence, e.version)
            narrative, updated_at, details = e.narrative, e.updated_at, e.details
            evidence = e.evidence or None
            values = e.values
            if as_of_ts is not None:
                snap = self.stores.right.version_as_of(e.id, as_of_ts, as_of_basis)
                if snap is None:
                    continue           # belief did not exist yet
                value, status = snap["current_value"], snap["status"]
                confidence, version = snap["confidence"], snap["version"]
                narrative, updated_at = snap["narrative"], snap["created_at"]
                details = snap.get("details") or []
                evidence = snap.get("evidence") or None
                values = snap.get("values") or []
            if status == "contested":
                contested += 1
            out.append(_etch_out(
                e, current_value=value, status=status, confidence=confidence,
                narrative=narrative, version=version, details=details,
                evidence=evidence, values=values, updated_at=updated_at))
            if e.id in views:
                _reduce(out[-1], views[e.id])
        self._attach_quotes(out, as_of_ts, as_of_basis, full=quotes == "full",
                            claim_ids={e.id: e.claim_ids for e in beliefs}, writers=writers)
        return KnowledgeResponse(
            entity=EntityOut(id=entity.id, name=entity.name, type=entity.type,
                             aliases=entity.aliases, scope=entity.scope),
            etches=out, count=len(out), as_of=as_of, as_of_basis=as_of_basis,
            contested=contested)

    def _attach_quotes(self, beliefs: list, as_of_ts: float | None, as_of_basis: str,
                       claim_ids: dict[str, list[str]] | None = None,
                       full: bool = True, writers: list[str] | None = None) -> None:
        """Set `quote` on each belief (EtchOut or etch RecallResult) in place:
        a reference with the source excerpt, or (`full=False`) with the excerpt
        only where the belief has no wording."""
        if settings.quote_chars <= 0 or not beliefs:
            return
        if claim_ids is None:
            claim_ids = {}
            for b in beliefs:
                e = self.stores.right.get_etch(b.id)
                claim_ids[b.id] = e.claim_ids if e else []
        requests = [QuoteRequest(
            etch_id=b.id, entity_name=b.entity_name or "", property=b.property or "",
            value=(b.current_value if hasattr(b, "current_value") else b.value) or "",
            claim_ids=claim_ids.get(b.id, []), has_wording=bool(b.details))
            for b in beliefs]
        found = quotes_for(self.stores, requests, as_of=as_of_ts,
                           as_of_basis=as_of_basis, full=full, writers=writers)
        for b in beliefs:
            if b.id in found:
                b.quote = Quote(**found[b.id])

    def _views(self, etches: list[Etch], writers: list[str] | None) -> dict[str, set[str]]:
        """For beliefs NOT all of whose writers are allowed: which allowed writers state
        their current value. A belief is shown (reduced to that value) only if some allowed
        writer states it itself, e.g. two agents shown the same ticket each see it, while
        a belief that only another agent's notes support stays hidden."""
        if writers is None or not etches:
            return {}
        allowed = set(writers)
        claims = {c.id: c for c in self.stores.left.claims_by_ids(
            sorted({cid for e in etches for cid in e.claim_ids}))}
        out: dict[str, set[str]] = {}
        for e in etches:
            wanted = {normalize_value(v) for v in e.current_value.split(", ")}
            wanted.add(normalize_value(e.current_value))
            backers = {src for cid in e.claim_ids
                       if (c := claims.get(cid)) is not None and c.polarity == "asserted"
                       and (c.value_norm in wanted or normalize_value(c.value) in wanted)
                       for src in c.sources}
            if backers & allowed:
                out[e.id] = backers & allowed
        return out

    def _resolve_ref(self, ref: str, entity_type: str | None, scope=None):
        """An entity id, or a surface name resolved the way ingestion resolved it."""
        found = self._resolve_refs(ref, entity_type, scope)
        return found[0] if found else None

    def _resolve_refs(self, ref: str, entity_type: str | None, scope=None) -> list:
        """Every entity `ref` names: the one with that id, or — for a surface
        name — one per scope it was seen in (subjects are resolved per scope),
        limited to `scope` when given."""
        scope = _scopes(scope)
        found = self.stores.right.get_entity(ref)
        if found is not None:
            return [found] if _in_scope(found.scope, scope) else []
        from app.ext import load_extensions
        from app.text import normalize_entity_name, normalize_value
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
            hits = self.stores.right.find_entities_by_alias(norm, scope)
            if entity_type:
                hits = [h for h in hits if h.type == entity_type]
            if hits:
                return hits
        return []

    # ── associate ────────────────────────────────────────────────────────────

    def associate(self, query: str, scope: str | list[str] | None = None, top_k: int = 5,
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
        scope = _scopes(scope)
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
            facts = self.know(entity_id, as_of=as_of, as_of_basis=as_of_basis, scope=scope)
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
                for e in self.stores.right.etches_for_entity(entity_id, scope=scope):
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
                    if not _in_scope(e.scope, scope):
                        continue
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

    def history(self, etch_id: str, writers: list[str] | None = None) -> list[dict[str, Any]] | None:
        if writers is not None:
            e = self.stores.right.get_etch(etch_id)
            if e is None or not _readable(e.source, _as_list(writers)):
                return None
        return self.stores.right.versions(etch_id)

    # ── dossier ────────────────────────────────────────────────────────────────

    def dossier(self, etch_id: str, writers: list[str] | None = None) -> DossierResponse | None:
        """Full provenance for one belief: etch + versions + claims + signals. With
        `writers`, a belief that rests on any other writer is reported as unknown."""
        e = self.stores.right.get_etch(etch_id)
        if e is None or not _readable(e.source, _as_list(writers) if writers is not None else None):
            return None
        etch = _etch_out(e)
        versions = [VersionOut(**v) for v in self.stores.right.versions(etch_id)]
        claims = [_claim_out(c) for c in self.stores.left.claims_by_ids(e.claim_ids)]
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
                          scope=s.scope, source_record_id=s.source_record_id,
                          actor=s.actor, created_at=s.created_at,
                          occurred_at=s.event_at)
                for s in self.stores.left.signals_by_ids(list(sig_ids))]
        return DossierResponse(etch=etch, versions=versions, claims=claims,
                               signals=signals, signals_omitted=signals_omitted)

    def signal(self, signal_id: str) -> SignalRecordResponse | None:
        """One signal and what came of it. A duplicate reports the claims of its
        cluster's representative, since those claims carry it as evidence too."""
        sig = self.stores.left.get_signal(signal_id)
        if sig is None:
            return None
        claims = [_claim_out(c) for c in self.stores.left.claims_for_signal(signal_id)]
        hidden = settings.claims_anonymization
        return SignalRecordResponse(
            id=sig.id, content=None if hidden else sig.content, source=sig.source,
            scope=sig.scope, status=sig.status, canonical_id=sig.canonical_id,
            context=sig.context_id, context_parent=sig.context_parent,
            source_record_id=sig.source_record_id, actor=sig.actor,
            metadata=sig.metadata or {}, created_at=sig.created_at,
            occurred_at=sig.event_at, claims=claims)

    def context(self, context_id: str, *, limit: int = 50) -> ContextResponse:
        """One conversation as recorded: what it has been about (focus, most recent
        first) and its signals in the order they were said."""
        focus = [ContextSubjectOut(entity_id=e.id, name=e.name, entity_type=e.type,
                                   mentions=n, last_seen_at=last)
                 for e, n, last in self.stores.right.context_focus(context_id, limit=20)]
        hidden = settings.claims_anonymization
        sigs = self.stores.left.signals_in_context(context_id=context_id, before=float("inf"),
                                                   limit=limit)
        if not sigs:
            sigs = self.stores.left.signals_in_context(context_parent=context_id,
                                                       before=float("inf"), limit=limit)
        signals = [ContextSignalOut(id=s.id, content=None if hidden else s.content,
                                    source=s.source, status=s.status, context=s.context_id,
                                    occurred_at=s.event_at)
                   for s in sigs]
        return ContextResponse(id=context_id, focus=focus, signals=signals)

    # ── forget (erasure) ─────────────────────────────────────────────────────

    def forget(self, *, entity: str | None = None, entity_type: str | None = None,
               source_record_ids: list[str] | None = None,
               signal_ids: list[str] | None = None,
               reason: str | None = None) -> ForgetResponse:
        """Erase personal data from every layer and report what changed.

        1. Choose the raw signals to delete: the ones given, the ones from the
           given source records, and — for a subject — every signal that cites
           it as evidence or names it in its text.
        2. Claims lose those signals as evidence. A claim with no evidence left,
           or one about the subject or pointing at it, is deleted.
        3. Beliefs: the subject's own, and any left with no claims, are deleted
           with all their versions. A belief that lost only some evidence keeps
           its timeline, but every stored version is blanked to "[erased]" (when
           it changed stays, what it said goes), its head is blanked at once,
           and it is re-consolidated from the claims that remain.
        4. A tombstone records the request: counts and signal ids (content
           hashes) only, so a backup restore can re-apply it and the same signal
           is refused if it is deposited again.
        """
        left, right = self.stores.left, self.stores.right
        erasure_id = uuid.uuid4().hex
        note = f"{ERASED} erasure {erasure_id}"
        report = ForgetResponse(erasure_id=erasure_id, reason=reason)

        # A name may stand for one entity per scope: erasure covers all of them.
        subjects = self._resolve_refs(entity, entity_type) if entity else []
        subject_ids = {s.id for s in subjects}
        report.entity_found = bool(subjects)

        doomed: set[str] = set(signal_ids or [])
        if source_record_ids:
            doomed |= set(left.signal_ids_for_records(list(source_record_ids)))
        subject_claims = {c.id: c for subj in subjects for c in left.claims_about(subj.id)}
        subject_claims = list(subject_claims.values())
        for c in subject_claims:
            doomed |= set(c.evidence_signal_ids)
        names = [n for subj in subjects for n in (subj.name, *subj.aliases)]
        # The name as given: a person is often only ever a value inside someone else's
        # belief ("primary contact = Petra Lang"), never a subject of their own, and
        # must be erased all the same.
        if entity and len(entity.strip()) >= 3:
            names.append(entity.strip())
        if names:
            by_name = set(left.signal_ids_mentioning(names))
            report.signals_matched_by_name = len(by_name - doomed)
            doomed |= by_name
        doomed_signals = left.signals_by_ids(sorted(doomed))
        doomed = {s.id for s in doomed_signals}
        doomed_sources = {s.source for s in doomed_signals}

        # Claims: drop the subject's, strip erased evidence from the rest.
        delete_claims = {c.id for c in subject_claims}
        affected_pairs: set[tuple[str, str]] = {(c.entity_id, c.property) for c in subject_claims}
        for c in left.claims_citing(sorted(doomed)):
            affected_pairs.add((c.entity_id, c.property))
            if c.id in delete_claims:
                continue
            remaining = [sid for sid in c.evidence_signal_ids if sid not in doomed]
            if not remaining:
                delete_claims.add(c.id)
                continue
            kept = left.signals_by_ids(remaining)
            # Sources of the signals still there, plus recorded sources no erased
            # signal came from (their signals may have expired by TTL).
            sources = sorted({s.source for s in kept}
                             | {src for src in c.sources if src not in doomed_sources})
            left.replace_claim_evidence(c.id, remaining, sources)
            report.claims_updated += 1
        report.claims_deleted = left.delete_claims(sorted(delete_claims))
        report.signals_deleted = left.delete_signals(sorted(doomed))

        # Beliefs.
        refold = False
        for entity_id, prop in sorted(affected_pairs):
            etch_id = f"{entity_id}::{prop}"
            remaining = left.claims_for_pair(entity_id, prop)
            head = right.get_etch(etch_id)
            if entity_id in subject_ids or not remaining:
                if head is not None:
                    report.versions_deleted += right.delete_etch(etch_id)
                    report.etches_deleted += 1
                continue
            if head is not None:
                report.versions_redacted += right.redact_versions(etch_id, note)
                head.current_value, head.narrative = ERASED, note
                head.details, head.values, head.evidence = [], [], {}
                head.embedding = None            # out of recall until re-folded
                head.claim_ids = [c.id for c in remaining]
                head.source_ids = sorted({sid for c in remaining for sid in c.evidence_signal_ids})
                right.upsert_etch(head)
                report.etches_refolded += 1
            left.set_claims_status([c.id for c in remaining], C_NEW)
            refold = True

        for subj in subjects:
            for e in right.etches_for_entity(subj.id):      # any not reached via claims
                report.versions_deleted += right.delete_etch(e.id)
                report.etches_deleted += 1
            right.delete_entity(subj.id)

        if refold:
            report.refold = "pending"
            try:
                self._get_pipeline().fold_pass(TickSummary())
                report.refold = "done"
            except Exception:  # no model reachable now: the worker folds later
                pass

        if not doomed and not subjects and not report.claims_deleted:
            report.note = ("nothing matched: no subject, no signal text and no source "
                           "record named this; nothing was erased")

        right.add_erasure(erasure_id, reason, sorted(doomed),
                          report.model_dump(exclude={"erasure_id", "reason"}))
        return report

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
            "sources_without_trust": self._sources_without_trust(),
        }

    def _sources_without_trust(self) -> list[str]:
        from app.ext import load_extensions
        declared = {**load_extensions().source_trust, **settings.source_trust}
        return [s for s in self.stores.left.sources() if s not in declared]

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
