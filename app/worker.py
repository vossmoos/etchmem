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
from dataclasses import dataclass, field, replace
from datetime import datetime

from app.agents import (ClaimExtractor, ConflictResolver, ConversationView, ExtractionResult,
                        Resolution, SignalInput, SubjectCandidate)
from app.anonymize import ANON_ENTITY_LABELS, scrub
from app.config import settings
from app.dedup import group_duplicates
from app.entities import EntityResolver
from app.embeddings import EmbeddingProvider
from app.gate import ROUTE_CONTESTED, route_and_resolve, select_details
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
    claims_unsupported: int = 0      # claims the verify step found the signal not to state
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


def _ago(seconds: float) -> str:
    """'5 min earlier', '3 h earlier', '2 days earlier'."""
    seconds = max(0.0, seconds)
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))} min earlier"
    if seconds < 2 * 86400:
        return f"{int(seconds // 3600)} h earlier"
    return f"{int(seconds // 86400)} days earlier"


def _align_labels(result: ExtractionResult, checks, labels) -> ExtractionResult:
    """Use the existing label the verify step named, when it is one that exists for
    that subject and property; otherwise keep the extracted value."""
    claims = []
    for c, check, known in zip(result.claims, checks, labels):
        if check.same_as and check.same_as in known and check.same_as != c.value:
            c = c.model_copy(update={"value": check.same_as})
        claims.append(c)
    return ExtractionResult(claims=claims)


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

        # ── LLM fan-out: bounded concurrency, retries ─────────────────────
        # Signals without a conversation are batched as before. Signals said in a
        # conversation form one lane per conversation and are read in the order
        # they were said, each seeing the ones before it. Lanes and batches run in
        # parallel; all writes happen sequentially below.
        size = max(1, settings.extract_llm_batch_size)
        loose: dict[str, list] = {}
        lanes: dict[tuple, list] = {}
        for s in representatives:
            conv = s.context_parent or s.context_id
            if conv:
                lanes.setdefault((s.scope, conv), []).append(s)
            else:
                loose.setdefault(s.scope, []).append(s)
        units: list[tuple[list, bool]] = [
            (group[i:i + size], False) for group in loose.values()
            for i in range(0, len(group), size)]
        units += [(sorted(group, key=lambda x: (x.event_at, x.created_at)), True)
                  for group in lanes.values()]
        unit_results: list[list | None] = [None] * len(units)

        def _run_unit(i):
            sigs, conversational = units[i]
            if not conversational:
                return self._read(sigs, None, None)
            out, earlier = [], []
            for sig in sigs:
                view, focus = self._conversation(sig, earlier)
                (row,) = self._read([sig], [view], [focus])
                out.append(row)
                earlier.append((sig, row[1]))
            return out

        if len(units) == 1:
            try:
                unit_results[0] = _run_unit(0)
            except Exception:
                log.exception("extraction failed; signals stay 'batched' and retry next tick")
        else:
            with ThreadPoolExecutor(max_workers=max(1, settings.extract_concurrency)) as pool:
                futures = {pool.submit(_run_unit, i): i for i in range(len(units))}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        unit_results[i] = fut.result()
                    except Exception:
                        log.exception("extraction unit %d failed; its signals stay "
                                      "'batched' and retry next tick", i)

        now = time.time()
        reps_with_results = [
            (rep, *row)
            for (sigs, _), rows in zip(units, unit_results) if rows is not None
            for rep, row in zip(sigs, rows)]

        for rep, result, resolution, keep, view in reps_with_results:
            members = self._s.left.signals_with_canonical(rep.id)
            sources = sorted({m.source for m in members})
            evidence_ids = sorted({m.id for m in members})

            kept = []
            for j, ec in enumerate(result.claims):
                if keep is not None and not (j < len(keep) and keep[j]):
                    log.info("dropping unsupported claim %s · %s (signal %s)",
                             ec.entity_name, ec.property, rep.id[:12])
                    summary.claims_unsupported += 1
                    continue
                kept.append(ec)
            if resolution is not None or view is not None:
                # What the signal was found to be about, in which conversation, and
                # what it brought to mind: part of the record.
                meta = dict(rep.metadata or {})
                if resolution is not None:
                    meta["subjects"] = [r.model_dump() for r in resolution.subjects]
                    meta["associations"] = [a.model_dump() for a in resolution.associations]
                if view is not None:
                    meta["focus"] = list(view.focus)
                self._s.left.update_signal_metadata(rep.id, meta)

            touched: list[str] = []
            claims = self._repair_compound_subjects(rep, kept, summary)
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
                    touched.append(entity.id)
            if resolution is not None:
                for r in resolution.subjects:
                    found = self._entities.find(r.name, r.entity_type)
                    if found is not None:
                        touched.append(found.id)
            for conv in (rep.context_id, rep.context_parent):
                self._s.right.touch_context(conv, touched, rep.event_at)

            self._s.left.set_signal_status([m.id for m in members], S_EXTRACTED)
            summary.extracted_signals += len(members)

        summary.entities = self._entities.stats.to_dict()

    def _read(self, sigs, views, focus) -> list[tuple]:
        """Resolve, extract and verify a few signals in one go.
        Returns per signal: (result, resolution | None, keep | None, view | None)."""
        cands = [self._candidates(s, focus[k] if focus else []) for k, s in enumerate(sigs)]
        inputs = [SignalInput(text=s.content, sent_at=s.event_at, source=s.source) for s in sigs]
        candidates = [[c for c, _ in per] for per in cands]
        known = {(c.name, prop): vals for per in cands for c, by_prop in per
                 for prop, vals in by_prop.items()}
        if views:
            raw = _with_retry(lambda: self._extractor.resolve_subjects(inputs, candidates, views))
        else:
            raw = _with_retry(lambda: self._extractor.resolve_subjects(inputs, candidates))
        resolved = None if raw is None else [
            r if isinstance(r, Resolution) else Resolution(subjects=list(r)) for r in raw]
        if resolved is None:
            # No resolve step (stubs, simple backends): the candidates are hints.
            names = list(dict.fromkeys(c.name for per in candidates for c in per))
            labels = [f"{name} · {prop}: " + ", ".join(vals)
                      for (name, prop), vals in known.items() if name in names]
        else:
            # Extraction sees the message and its subjects only. Existing labels
            # are matched afterwards, by the verify step: shown to the extractor
            # they get restated instead of the message's own facts.
            inputs = [replace(inp, subjects=tuple(f"{r.name} ({r.entity_type})" for r in rs.subjects))
                      for inp, rs in zip(inputs, resolved)]
            names, labels = [], []
        results = _with_retry(lambda: self._extractor.extract_signals(
            inputs, names or None, labels or None))
        checks = None
        if settings.verify_claims:
            claim_labels = [[known.get((c.entity_name, c.property), []) for c in r.claims]
                            for r in results]
            checks = _with_retry(lambda: self._extractor.verify_claims(
                inputs, results, claim_labels))
            if checks is not None:
                results = [_align_labels(r, row, lab)
                           for r, row, lab in zip(results, checks, claim_labels)]
        keep = None if checks is None else [[c.supported for c in row] for row in checks]
        return [(results[k], resolved[k] if resolved is not None else None,
                 keep[k] if keep is not None else None, views[k] if views else None)
                for k in range(len(sigs))]

    def _conversation(self, sig, earlier) -> tuple[ConversationView, list]:
        """What the conversation this signal was said in has been about, and its
        last messages, as of when it was said. `earlier`: (signal, resolution) read
        just before it in the same lane, not yet written.

        Returns the view and the focus subjects as Entities (or (name, type) for
        subjects only this lane has seen so far).
        """
        window = max(0.0, settings.context_window_seconds)
        at = sig.event_at
        left, right = self._s.left, self._s.right
        recent = left.signals_in_context(context_id=sig.context_id, before=at, limit=4) \
            if sig.context_id else []
        if sig.context_parent:
            # The last messages around the thread, however old: how long ago they were
            # said is shown, and the model judges whether they still matter.
            ids = {x.id for x in recent}
            around = [x for x in left.signals_in_context(
                context_parent=sig.context_parent, before=at, limit=3)
                if x.id not in ids]
            recent = sorted(recent + around, key=lambda x: x.event_at)
        focus: list = []
        seen: set = set()
        for prev, res in reversed(earlier):          # newest first
            same = (sig.context_id and prev.context_id == sig.context_id) or (
                sig.context_parent and prev.context_parent == sig.context_parent
                and at - prev.event_at <= window)
            if not same or res is None:
                continue
            for r in res.subjects:
                ent = self._entities.find(r.name, r.entity_type)
                key = ent.id if ent else (r.name, r.entity_type)
                if key not in seen:
                    seen.add(key)
                    focus.append(ent or (r.name, r.entity_type))
        for conv, since in ((sig.context_id, 0.0), (sig.context_parent, at - window)):
            for ent, _n, _last in right.context_focus(conv, since=since, before=at):
                if ent.id not in seen:
                    seen.add(ent.id)
                    focus.append(ent)
        focus = focus[:5]
        names = tuple(f"{f.name} ({f.type})" if hasattr(f, "id") else f"{f[0]} ({f[1]})"
                      for f in focus)
        lines = tuple(f"[{_ago(at - x.event_at)}] " + x.content.replace("\n", " ")[:300]
                      for x in recent[-5:])
        return ConversationView(focus=names, recent=lines), focus

    def _candidates(self, sig, focus=()) -> list[tuple[SubjectCandidate, dict[str, list[str]]]]:
        """Subjects offered to the resolve step, with what is known about them and
        their labels (for reusing a label, or retracting it):

          focus    what the signal's conversation has been about
          linked   tied to a focus subject by a declared relation, either way
                   (the ticket that tracks a feature)
          similar  beliefs and names that look like the signal (embedding search)

        Nothing here decides; the resolve step reads the signal and chooses.
        """
        n = max(0, settings.subject_candidates)
        if n == 0:
            return []
        right = self._s.right
        label_props = set(settings.multi_value_set) | {
            name for name, spec in self._registry.by_name.items() if spec.detail}
        per_label = max(0, settings.extract_known_labels)
        out: list[tuple[SubjectCandidate, dict[str, list[str]]]] = []
        seen: set = set()

        def add(ent, role) -> None:
            if ent is None or ent.id in seen or self._registry.is_ignored(ent.type):
                return
            if ent.scope != sig.scope:
                return
            seen.add(ent.id)
            etches = right.etches_for_entity(ent.id)
            # What identifies the work first: its title, then what it is about.
            first = ["title", "decision_state", "work_state", "tracked_in"] + sorted(label_props)
            ranked = sorted(etches, key=lambda e: first.index(e.property)
                            if e.property in first else len(first))
            facts = tuple(f"{e.property}: {e.current_value[:80]}" for e in ranked[:6])
            labels: dict[str, list[str]] = {}
            for e in etches:
                if e.property not in label_props or per_label == 0:
                    continue
                vals = [v["value"] for v in e.values] or \
                    [x.strip() for x in (e.current_value or "").split(",") if x.strip()]
                if vals:
                    labels[e.property] = vals[:per_label]
            out.append((SubjectCandidate(ent.name, ent.type, facts, role), labels))

        focus_entities = [f for f in focus if hasattr(f, "id")]
        for f in focus:
            if hasattr(f, "id"):
                add(f, "focus")
            elif (f[0], f[1]) not in seen:
                seen.add((f[0], f[1]))
                out.append((SubjectCandidate(f[0], f[1], (), "focus"), {}))
        for ent in focus_entities:
            for e in right.etches_for_entity(ent.id):
                if e.value_entity_id:
                    add(right.get_entity(e.value_entity_id), "linked")
            for e in right.etches_referencing(ent.id):
                add(right.get_entity(e.entity_id), "linked")
        fixed = len(out)
        if sig.embedding:
            # Earlier messages like this one, and what they turned out to be about.
            for other, _sim in self._s.left.similar_signals(sig.embedding, scope=sig.scope,
                                                            top_k=n, exclude=sig.id):
                for c in self._s.left.claims_for_signal(other.id):
                    if len(out) - fixed >= n:
                        break
                    add(right.get_entity(c.entity_id), "similar")
            for hit in right.search_etches(sig.embedding, top_k=n * 3, scope=sig.scope):
                if len(out) - fixed >= n:
                    break
                add(right.get_entity(hit.row.get("entity_id")), "similar")
            for ent, _sim in right.search_entities(sig.embedding, top_k=n * 4):
                if len(out) - fixed >= n:
                    break
                add(ent, "similar")
        return out

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
        # The full wording rides next to the label, only where the property
        # declares it; it never takes part in the claim's identity.
        detail = (ec.detail or "").strip() or None
        if detail and not self._registry.has_detail(ec.property):
            detail = None
        if detail and settings.claims_anonymization:
            detail = scrub(detail)
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
            detail=detail,
            scope=rep.scope,
            status=C_NEW,
            created_at=now,
            updated_at=now,
        )
        self._s.left.upsert_claim(claim)
        summary.claims_written += 1

    # ── Stage 3: fold claims → etches ────────────────────────────────────────

    def source_trust(self) -> dict[str, float]:
        """Declared trust per source: the ext `sources:` block, env on top."""
        return {**self._registry.source_trust, **settings.source_trust}

    def _signal_sources(self, claims) -> dict[str, str]:
        """signal id → source for the evidence behind these claims.

        Lets the gate tell ten signals from one source (one witness, repeated)
        from ten sources. Signals purged by TTL are simply absent; the gate
        still counts each claim's recorded sources once.
        """
        ids = sorted({sid for c in claims for sid in c.evidence_signal_ids})
        return {s.id: s.source for s in self._s.left.signals_by_ids(ids)}

    def fold_pass(self, summary: TickSummary) -> None:
        for entity_id, prop in self._s.left.pairs_with_new_claims():
            claims = self._s.left.claims_for_pair(entity_id, prop)
            if not claims:
                continue
            summary.pairs_folded += 1
            entity_name = claims[0].entity_name
            scope = claims[0].scope
            decision = route_and_resolve(
                prop, claims, trust=self.source_trust(),
                signal_sources=self._signal_sources(claims))

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

            evidence = decision.evidence
            values = decision.values
            details = select_details(prop, claims, value)
            if details:
                # The narrative is what recall embeds and returns, so it carries
                # the wording, not just the labels.
                narrative = (f"{entity_name} — {prop.replace('_', ' ')}: "
                             + "; ".join(f"{d['value']} — {d['detail']}" for d in details) + ".")
            if decision.policy == "retracted":
                narrative += " No longer valid: retracted."

            etch_id = f"{entity_id}::{prop}"
            existing = self._s.right.get_etch(etch_id)
            now = time.time()

            if existing is None:
                version, created_at, changed = 1, now, True
            elif (existing.current_value != value or existing.status != status
                  or existing.details != details
                  or (existing.values and _active(existing.values) != _active(values))):
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
                value_entity_id=value_entity_id, details=details,
                evidence=evidence, values=values,
            )
            self._s.right.upsert_etch(etch)

            if changed:
                # When the newest fact behind this belief became true. Claims
                # carry occurred_at / an extracted date, so this is the real
                # timeline even when the records arrived late or out of order.
                event_at = max((c.event_time for c in claims if c.event_time), default=now)
                self._s.right.add_version(etch_id, version, value, status, confidence,
                                          narrative, new_claim_ids, now, event_at, details,
                                          evidence, values)
                if existing is None:
                    summary.etches_formed += 1
                else:
                    summary.etches_updated += 1

            self._s.left.set_claims_status([c.id for c in claims], C_CONSOLIDATED)


def _active(values: list[dict]) -> set[tuple[str, bool]]:
    """Which values are in force — what a new version must be written for."""
    return {(v.get("value", ""), bool(v.get("active", True))) for v in values}


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
