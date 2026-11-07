"""
Quotes: where a belief comes from.

A belief is short by design ("Acme · billing_issue = double-charged invoice",
plus its wording). Its `quote` names the newest signal behind its current
value: who deposited it (the agent and the person it acted for), in which
source record, and when. So an agent can cite a fact, judge how fresh it is,
and open the source record for its current state, without a dossier call.

By default (recall/know `quotes="full"`) every quote carries an excerpt of the
original text around the fact. `quotes="ref"` keeps the excerpt only where the
belief has no wording of its own (a plain value such as a status or a tier),
since elsewhere it mostly repeats that wording: shorter replies for agents that
read many beliefs.

Two queries per call, whatever the number of beliefs: their claims, then those
claims' signals. With claims anonymization on, the excerpt and the actor are
left out: raw signals keep personal data and are never surfaced on retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.stores import Signal, Stores
from app.text import normalize_value

_SPACE = re.compile(r"\s+")


@dataclass
class QuoteRequest:
    """One belief to quote, as it stands (now, or at `as_of`)."""
    etch_id: str
    entity_name: str
    property: str
    value: str
    claim_ids: list[str] = field(default_factory=list)
    has_wording: bool = False           # the belief carries `details` already


def excerpt(text: str, anchors: list[str], limit: int) -> str:
    """At most `limit` characters of `text`, around the first anchor found."""
    flat = _SPACE.sub(" ", text or "").strip()
    if limit <= 0 or len(flat) <= limit:
        return flat
    low = flat.lower()
    pos = next((p for p in (low.find(a.lower()) for a in anchors if a and len(a) > 2)
                if p >= 0), 0)
    start = max(0, pos - limit // 3)
    if start:
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < pos else start
    end = start + limit
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        end = space if space > start + limit // 2 else end
    return ("…" if start else "") + flat[start:end].strip() + ("…" if end < len(flat) else "")


def quotes_for(stores: Stores, requests: list[QuoteRequest], *,
               as_of: float | None = None, as_of_basis: str = "ingest",
               full: bool = True, writers: list[str] | None = None) -> dict[str, dict]:
    """etch id → quote (a dict shaped like schemas.Quote), for beliefs that have one."""
    limit = settings.quote_chars
    if limit <= 0 or not requests:
        return {}
    claims = {c.id: c for c in stores.left.claims_by_ids(
        sorted({cid for r in requests for cid in r.claim_ids}))}
    chosen: dict[str, list[str]] = {}
    for r in requests:
        mine = [claims[cid] for cid in r.claim_ids if cid in claims]
        asserted = [c for c in mine if c.polarity == "asserted"] or mine
        wanted = {normalize_value(v) for v in r.value.split(", ")} | {normalize_value(r.value)}
        current = [c for c in asserted
                   if c.value_norm in wanted or normalize_value(c.value) in wanted] or asserted
        chosen[r.etch_id] = [sid for c in current for sid in c.evidence_signal_ids]
    signals = {s.id: s for s in stores.left.signals_by_ids(
        sorted({sid for ids in chosen.values() for sid in ids}))}

    def when(s: Signal) -> float:
        return s.event_at if as_of_basis == "event" else s.created_at

    anonymous = settings.claims_anonymization
    out: dict[str, dict] = {}
    for r in requests:
        candidates = [signals[sid] for sid in chosen[r.etch_id] if sid in signals
                      and (writers is None or signals[sid].source in writers)]
        if as_of is not None:
            candidates = [s for s in candidates if when(s) <= as_of]
        if not candidates:
            continue
        s = max(candidates, key=lambda x: (x.event_at, x.created_at))
        anchors = [r.value, *r.value.split(", "), r.entity_name]
        with_text = not anonymous and (full or not r.has_wording)
        out[r.etch_id] = {
            "signal_id": s.id,
            "excerpt": excerpt(s.content, anchors, limit) if with_text else None,
            "source": s.source,
            "source_record_id": s.source_record_id,
            "actor": {} if anonymous else dict(s.actor or {}),
            "occurred_at": s.event_at,
        }
    return out
