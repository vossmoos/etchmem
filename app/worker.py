"""
The pipeline + the in-process background worker (no external broker).

Three passes, driven by the signal/claim status lifecycle which IS the queue:

  batch_pass   : new signals      → dedup-grouped, canonical_id set → batched
  extract_pass : batched signals  → claims (LLM stage 2) → extracted
  fold_pass    : new claims       → etches + versions (gate + LLM stage 3)

`run_once()` runs one full tick and is what POST /sleep calls. `WorkerLoop`
runs `run_once()` forever on a cadence inside the API process.

Crash-safety: a row's status advances only after its work commits, so a crash
mid-tick just reprocesses cleanly; hashes make every step idempotent.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from app.agents import ClaimExtractor, ConflictResolver, ExtractionResult
from app.anonymize import ANON_ENTITY_LABELS, scrub
from app.config import settings
from app.dedup import group_duplicates
from app.entities import EntityResolver
from app.embeddings import EmbeddingProvider
from app.gate import ROUTE_CONTESTED, route_and_resolve
from app.hashing import claim_hash
from app.stores import (
    C_NEW, C_CONSOLIDATED, S_BATCHED, S_EXTRACTED, S_NEW,
    Claim, Etch, Stores,
)
from app.text import normalize_value

log = logging.getLogger("etchmem.worker")


@dataclass
class TickSummary:
    batched: int = 0
    extracted_signals: int = 0
    claims_written: int = 0
    pairs_folded: int = 0
    etches_formed: int = 0
    etches_updated: int = 0
    contested: int = 0
    claims_dropped: int = 0          # subjects declared `ignore: true` in ext/
    subject_retries: int = 0         # signals re-read after a compound entity_name
    relations_resolved: int = 0      # claim values resolved to an entity (edges)
    # How subjects were identified this tick — see entities.ResolutionStats.
    entities: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _parse_event_time(iso: str | None, default: float) -> float:
    if not iso:
        return default
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return default


def resolve_event_time(extracted_iso: str | None, signal) -> float:
    """When did this claim's fact become true?

    Three sources, most specific first:
      1. a date the extractor read out of the signal text — a 2014 record may
         state that something became true in 2009, and that is the truth the
         gate's recency policy needs;
      2. `occurred_at` declared by the caller on the signal;
      3. ingest time.

    Set ETCHMEM_OCCURRED_AT_PRECEDENCE=declared to invert 1 and 2 whenever the
    signal carries a declared time: for a bulk backfill the record date is
    authoritative, and an extractor misreading a date corrupts the timeline
    silently, which is worse than losing the finer-grained date.
    """
    declared = signal.occurred_at or 0.0
    if declared and settings.occurred_at_precedence == "declared":
        return declared
    return _parse_event_time(extracted_iso, signal.event_at)


def _with_retry(fn):
    """Run an LLM call with exponential backoff + jitter (429 gets longer waits)."""
    last: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt >= settings.llm_max_retries:
                break
            msg = str(e).lower()
            rate_limited = "429" in msg or "rate limit" in msg or "rate_limit" in msg
            base = 5.0 if rate_limited else 1.5
            delay = min(base * (2 ** attempt), 45.0) * (0.5 + random.random())
            log.warning("LLM call failed (%s); retry %d/%d in %.1fs",
                        e, attempt + 1, settings.llm_max_retries, delay)
            time.sleep(delay)
    raise last  # type: ignore[misc]


class Pipeline:
    def __init__(
        self,
        stores: Stores,
        embedder: EmbeddingProvider,
        extractor: ClaimExtractor,
        resolver: ConflictResolver,
    ) -> None:
        self._s = stores
        self._embed = embedder
        self._extractor = extractor
        self._resolver = resolver
        from app.ext import load_extensions
        self._registry = load_extensions()
        self._entities = EntityResolver(stores.right, embedder, self._registry)

    # ── one full tick ──────────────────────────────────────────────────────

    def run_once(self) -> TickSummary:
        summary = TickSummary()
        self.batch_pass(summary)
        self.extract_pass(summary)
        self.fold_pass(summary)
        self._s.left.expire()
        return summary

    # ── Stage 1: batch / dedup ───────────────────────────────────────────────

    def batch_pass(self, summary: TickSummary) -> None:
        new = self._s.left.signals_by_status(S_NEW)
        if not new:
            return
        for group in group_duplicates(new, settings.signal_dedup_distance,
                                      settings.dedup_max_time_gap_seconds):
            # Earliest event, not earliest insertion: in a backfill the load
            # order says nothing about which record came first.
            canonical = min(group, key=lambda s: (s.event_at, s.created_at))
            for sig in group:
                self._s.left.set_canonical(sig.id, canonical.id, S_BATCHED)
            summary.batched += len(group)

    # ── Stage 2: extract claims ──────────────────────────────────────────────

    def extract_pass(self, summary: TickSummary) -> None:
        batched = self._s.left.signals_by_status(S_BATCHED)
        if not batched:
            return
        self._entities.stats.reset()
        representatives = [s for s in batched if s.canonical_id == s.id]

        now = time.time()
        eligible_all = (
            len(representatives) >= settings.extract_min_batch
            or any(s.extract_mode == "immediate" for s in representatives)
            or any((now - s.created_at) >= settings.extract_max_wait_seconds
                   for s in representatives)
        )
        if not eligible_all:
            return

        # ── LLM fan-out: batched prompts, bounded concurrency, retries ─────
        # Only the network calls run in threads; all DuckDB writes happen
        # sequentially below (single-writer constraint).
        size = max(1, settings.extract_llm_batch_size)
        chunks = [representatives[i:i + size] for i in range(0, len(representatives), size)]
        chunk_results: list[list[ExtractionResult] | None] = [None] * len(chunks)

        def _run_chunk(chunk):
            return _with_retry(lambda: self._extractor.extract_batch([s.content for s in chunk]))

        if len(chunks) == 1:
            try:
                chunk_results[0] = _run_chunk(chunks[0])
            except Exception:
                log.exception("extraction failed; signals stay 'batched' and retry next tick")
        else:
            with ThreadPoolExecutor(max_workers=max(1, settings.extract_concurrency)) as pool:
                futures = {pool.submit(_run_chunk, c): i for i, c in enumerate(chunks)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        chunk_results[i] = fut.result()
                    except Exception:
                        log.exception("extraction chunk %d failed; its signals stay "
                                      "'batched' and retry next tick", i)

        now = time.time()
        reps_with_results = [
            (rep, result)
            for chunk, results in zip(chunks, chunk_results) if results is not None
            for rep, result in zip(chunk, results)
        ]

        for rep, result in reps_with_results:
            members = self._s.left.signals_with_canonical(rep.id)
            sources = sorted({m.source for m in members})
            evidence_ids = sorted({m.id for m in members})

            claims = self._repair_compound_subjects(rep, result.claims, summary)
            for ec in claims:
                subjects = self._subjects_for(ec, rep.scope)
                if not subjects:
                    # Ignored subject type, or a name we could not attach a fact
                    # to safely. Unattached beats attached to the wrong thing.
                    summary.claims_dropped += 1
                    continue
                for entity in subjects:
                    self._write_claim(ec, entity, rep, sources, evidence_ids,
                                      now, summary)

            self._s.left.set_signal_status([m.id for m in members], S_EXTRACTED)
            summary.extracted_signals += len(members)

        summary.entities = self._entities.stats.to_dict()

    def _compound(self, ec) -> bool:
        """Does this claim name more than one subject? A count, not a judgement."""
        return self._registry.subject_count(ec.entity_type, ec.entity_name) > 1

    def _repair_compound_subjects(self, rep, claims, summary):
        """Send a signal back to the model when it named several subjects at once.

        Whether "A und B" is a list and "A für B" is a relation is a reading of
        the original sentence. Inspecting the mangled entity_name with a word
        list would decide that question with strictly less information than the
        model already had, in one language at a time, and would need extending
        forever. So the model is asked again, with the sentence and with what it
        got wrong.

        One retry. If the answer is still compound, the claim is dropped and
        counted: unattached beats attached to the wrong subject.
        """
        bad = [c for c in claims if self._compound(c)]
        if not bad or not settings.subject_retry_enabled:
            return claims
        try:
            summary.subject_retries += 1
            multi = sorted(n for n, sp in self._registry.by_name.items()
                           if sp.distributive)
            fixed = _with_retry(lambda: self._extractor.re_extract_subjects(
                rep.content, [c.entity_name for c in bad], multi_subject_properties=multi))
        except Exception:
            log.exception("subject re-extraction failed; dropping compound claims")
            return [c for c in claims if not self._compound(c)]
        good = [c for c in claims if not self._compound(c)]
        return good + [c for c in fixed.claims if not self._compound(c)]

    def _subjects_for(self, ec, scope):
        """The single subject one claim is about, or nothing.

        There is no distribution step here by design. Whether a compound name
        is a list ("A und B both leak") or a relation ("A for B") is a reading
        of the original sentence, so the model decides it by emitting one claim
        per subject. A name still carrying several identifiers after the retry
        is malformed, and a malformed claim is dropped: unattached beats
        attached to the wrong subject.
        """
        if self._compound(ec):
            return []
        one = self._entities.resolve(ec.entity_name, ec.entity_type, scope)
        return [one] if one is not None else []

    def _resolve_object(self, ec, scope):
        """Resolve the VALUE of a declared relation into an entity.

        Two things follow from this. The edge becomes followable in both
        directions, and the value stops being a spelling: "NT 2405" and
        "NT-2405" normalise to one entity id, so the gate sees one value rather
        than a conflict between two.
        """
        rel_type = self._registry.relation_type(ec.property)
        if not rel_type:
            return None
        if self._registry.subject_count(rel_type, ec.value) > 1:
            return None          # several identifiers in one value: no safe edge
        return self._entities.resolve(ec.value, rel_type, scope)

    def _write_claim(self, ec, entity, rep, sources, evidence_ids, now, summary) -> None:
        display_name, value = entity.name, ec.value
        obj = self._resolve_object(ec, rep.scope)
        if settings.claims_anonymization:
            label = ANON_ENTITY_LABELS.get(entity.type)
            if label:
                display_name = self._s.right.get_or_assign_anon_token(entity.id, label)
            value = scrub(value)
        # A resolved object IS the normalised value: equality is entity identity.
        value_norm = obj.id if obj is not None else normalize_value(value)
        if obj is not None:
            summary.relations_resolved += 1
        cid = claim_hash(entity.id, ec.property, value_norm, ec.polarity)
        claim = Claim(
            id=cid,
            entity_id=entity.id,
            entity_name=display_name,
            property=ec.property,
            value=value,
            value_norm=value_norm,
            polarity=ec.polarity,
            event_time=resolve_event_time(ec.event_time, rep),
            ingest_time=now,
            sources=sources,
            evidence_signal_ids=evidence_ids,
            corroboration_count=len(evidence_ids),
            confidence=ec.confidence,
            value_entity_id=obj.id if obj is not None else None,
            scope=rep.scope,
            status=C_NEW,
            created_at=now,
            updated_at=now,
        )
        self._s.left.upsert_claim(claim)
        summary.claims_written += 1

    # ── Stage 3: fold claims → etches ────────────────────────────────────────

    def fold_pass(self, summary: TickSummary) -> None:
        for entity_id, prop in self._s.left.pairs_with_new_claims():
            claims = self._s.left.claims_for_pair(entity_id, prop)
            if not claims:
                continue
            summary.pairs_folded += 1
            entity_name = claims[0].entity_name
            scope = claims[0].scope
            decision = route_and_resolve(prop, claims)

            if decision.route == ROUTE_CONTESTED:
                res = self._resolver.resolve(entity_name, prop, decision.competing)
                value, status = res.current_value, res.status
                confidence, narrative = res.confidence, res.narrative
                if settings.claims_anonymization:  # safety net over LLM output
                    value, narrative = scrub(value), scrub(narrative)
                summary.contested += 1
            else:
                value, status = decision.value, decision.status
                confidence = decision.confidence
                narrative = f"{entity_name} — {prop.replace('_', ' ')}: {value}."

            etch_id = f"{entity_id}::{prop}"
            existing = self._s.right.get_etch(etch_id)
            now = time.time()

            if existing is None:
                version, created_at, changed = 1, now, True
            elif existing.current_value != value or existing.status != status:
                version, created_at, changed = existing.version + 1, existing.created_at, True
            else:
                version, created_at, changed = existing.version, existing.created_at, False

            object_ids = {c.value_entity_id for c in claims if c.value_entity_id}
            value_entity_id = object_ids.pop() if len(object_ids) == 1 else None
            source_ids = sorted({sid for c in claims for sid in c.evidence_signal_ids})
            sources = sorted({s for c in claims for s in c.sources})
            new_claim_ids = [c.id for c in claims if c.status == C_NEW]

            etch = Etch(
                id=etch_id, entity_id=entity_id, entity_name=entity_name, property=prop,
                current_value=value, status=status, confidence=confidence,
                narrative=narrative, version=version, scope=scope,
                source=", ".join(sources) if sources else None,
                claim_ids=[c.id for c in claims], source_ids=source_ids,
                created_at=created_at, updated_at=now,
                embedding=self._embed.embed_one(narrative),
                value_entity_id=value_entity_id,
            )
            self._s.right.upsert_etch(etch)

            if changed:
                # When the newest fact behind this belief became true. Claims
                # carry occurred_at / an extracted date, so this is the real
                # timeline even when the records arrived late or out of order.
                event_at = max((c.event_time for c in claims if c.event_time), default=now)
                self._s.right.add_version(etch_id, version, value, status, confidence,
                                          narrative, new_claim_ids, now, event_at)
                if existing is None:
                    summary.etches_formed += 1
                else:
                    summary.etches_updated += 1

            self._s.left.set_claims_status([c.id for c in claims], C_CONSOLIDATED)


# ── Background loop ──────────────────────────────────────────────────────────

class WorkerLoop:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        log.info("etchmem worker loop started (interval=%ss)", settings.worker_interval_seconds)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._pipeline.run_once)
            except Exception:  # never let the loop die
                log.exception("worker tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=settings.worker_interval_seconds)
            except asyncio.TimeoutError:
                pass
