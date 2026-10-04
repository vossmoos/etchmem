"""
The deterministic routing gate.

Given all claims for one (entity, property), decide — cheaply, with no LLM —
whether the fold is:

  AGREE     all asserted claims share one value          → settle, no LLM
  POLICY    differing values, but recency / source-trust / cardinality
            resolves it                                  → settle, no LLM
  CONTESTED genuine disagreement policy can't break       → escalate to LLM

Only CONTESTED reaches the top-tier model. The gate also computes a
deterministic confidence so settled etches never need the LLM.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.agents import CompetingClaim
from app.config import settings
from app.stores import Claim
from app.text import normalize_value

ROUTE_AGREE = "agree"
ROUTE_POLICY = "policy"
ROUTE_CONTESTED = "contested"


@dataclass
class GateDecision:
    route: str
    value: str
    status: str                              # "settled" | "contested"
    confidence: float
    policy: str = ""                         # which rule fired (for audit)
    competing: list[CompetingClaim] = field(default_factory=list)
    # What the confidence rests on: {"signals", "sources": [{source, trust,
    # signals}], "max_trust"} for the evidence behind the chosen value(s).
    evidence: dict = field(default_factory=dict)


def _strength(counts: dict[str, int], trust: dict[str, float]) -> float:
    """How strongly a set of signals supports a claim, in 0..1.

    Each source is one witness whose reliability is its declared trust. Several
    signals from the SAME source are not several witnesses: the k-th repeat
    counts `decay^(k-1)` of the first, so repetition helps and saturates. Then
    the independent witnesses are combined as independent evidence (noisy-OR):
    two 0.5 sources give 0.75, a single 0.9 source gives 0.9.
    """
    decay = min(max(settings.evidence_repeat_decay, 0.0), 0.99)
    miss = 1.0
    for source, n in counts.items():
        t = min(max(trust.get(source, settings.default_source_trust), 0.0), 1.0)
        effective = (1.0 - decay ** n) / (1.0 - decay)
        miss *= (1.0 - t) ** effective
    return 1.0 - miss


def _evidence(signal_ids: set[str], claims: list[Claim], signal_sources: dict[str, str],
              trust: dict[str, float]) -> tuple[float, dict]:
    """(strength, breakdown) for the signals behind one value, or the whole union.

    `signal_ids` are counted once however many claims cite them: one deposit
    that yields two labels is ONE witness, not two. A source with no surviving
    signal (expired by TTL) still counts once, so confidence does not collapse
    when old signals are purged.
    """
    counts: Counter[str] = Counter(
        signal_sources[sid] for sid in signal_ids if sid in signal_sources)
    for c in claims:
        for s in c.sources:
            counts.setdefault(s, 1)
    breakdown = {
        "signals": int(sum(counts.values())),
        "sources": [{"source": s, "trust": round(trust.get(s, settings.default_source_trust), 3),
                     "signals": n} for s, n in sorted(counts.items())],
        "max_trust": round(max((trust.get(s, settings.default_source_trust) for s in counts),
                               default=0.0), 3),
    }
    return _strength(dict(counts), trust), breakdown


def _confidence(chosen_corro: int, total_corro: int, strength: float,
                penalty: float = 1.0) -> float:
    agreement = (chosen_corro / total_corro) if total_corro else 0.5
    return max(0.05, min(0.99, agreement * strength * penalty))


def route_and_resolve(prop: str, claims: list[Claim], *,
                      trust: dict[str, float] | None = None,
                      signal_sources: dict[str, str] | None = None) -> GateDecision:
    """Route one (entity, property) fold.

    `trust`          source → 0..1 (ext `sources:` block, env override on top).
    `signal_sources` signal id → source, so repeats from one source can be told
                     from independent witnesses. Without it every distinct
                     source counts once.
    """
    trust = settings.source_trust if trust is None else trust
    signal_sources = signal_sources or {}
    asserted = [c for c in claims if c.polarity == "asserted"]
    if not asserted:
        return GateDecision(ROUTE_POLICY, "unknown", "settled", 0.1, "no_assertion")

    # Aggregate by normalized value.
    by_value: dict[str, dict] = {}
    for c in asserted:
        g = by_value.setdefault(c.value_norm, {"value": c.value, "corro": 0, "claims": [],
                                               "sources": set(), "signals": set(),
                                               "event_time": 0.0})
        g["corro"] += max(1, c.corroboration_count)
        g["claims"].append(c)
        g["sources"].update(c.sources)
        g["signals"].update(c.evidence_signal_ids)
        g["event_time"] = max(g["event_time"], c.event_time or 0.0)

    total_corro = sum(g["corro"] for g in by_value.values())

    def judge(group_list, corro, penalty=1.0):
        """(confidence, evidence) for the signals behind one or several groups."""
        ids = set().union(*(g["signals"] for g in group_list))
        cl = [c for g in group_list for c in g["claims"]]
        strength, ev = _evidence(ids, cl, signal_sources, trust)
        return _confidence(corro, total_corro, strength, penalty), ev

    # ── Multi-valued property → union, never a conflict ────────────────────
    if prop in settings.multi_value_set:
        values = sorted(g["value"] for g in by_value.values())
        conf, ev = judge(list(by_value.values()), total_corro)
        return GateDecision(ROUTE_POLICY, ", ".join(values), "settled", conf,
                            "cardinality_union", evidence=ev)

    # ── Single distinct value → AGREE ──────────────────────────────────────
    if len(by_value) == 1:
        g = next(iter(by_value.values()))
        conf, ev = judge([g], g["corro"])
        return GateDecision(ROUTE_AGREE, g["value"], "settled", conf, "unanimous",
                            evidence=ev)

    # ── Differing values → try deterministic policies ──────────────────────
    # 1) recency (state-machine: latest event_time wins, if unique)
    if any(g["event_time"] > 0 for g in by_value.values()):
        ranked = sorted(by_value.items(), key=lambda kv: kv[1]["event_time"], reverse=True)
        (top_v, top_g), (_, runner_g) = ranked[0], ranked[1]
        if top_g["event_time"] > runner_g["event_time"]:
            conf, ev = judge([top_g], top_g["corro"], penalty=0.85)
            return GateDecision(ROUTE_POLICY, top_g["value"], "settled", conf,
                                "recency", evidence=ev)

    # 2) source trust gap
    if trust:
        def value_trust(g) -> float:
            return max((trust.get(s, settings.default_source_trust) for s in g["sources"]),
                       default=settings.default_source_trust)
        ranked = sorted(by_value.items(), key=lambda kv: value_trust(kv[1]), reverse=True)
        (top_v, top_g), (_, runner_g) = ranked[0], ranked[1]
        if value_trust(top_g) - value_trust(runner_g) >= settings.trust_gap:
            conf, ev = judge([top_g], top_g["corro"], penalty=0.9)
            return GateDecision(ROUTE_POLICY, top_g["value"], "settled", conf,
                                "source_trust", evidence=ev)

    # ── Otherwise: genuine conflict → escalate ─────────────────────────────
    competing = [
        CompetingClaim(value=g["value"], polarity="asserted",
                       sources=sorted(g["sources"]), corroboration_count=g["corro"],
                       event_time=g["event_time"] or None)
        for g in by_value.values()
    ]
    # Fallback value/confidence if the LLM is unavailable: most-corroborated.
    top = max(by_value.values(), key=lambda g: g["corro"])
    conf, ev = judge([top], top["corro"], penalty=0.5)
    return GateDecision(ROUTE_CONTESTED, top["value"], "contested", conf,
                        "ambiguous", competing, evidence=ev)


def select_details(prop: str, claims: list[Claim], value: str) -> list[dict]:
    """The full wording behind the etch's label(s): [{"value", "detail"}].

    Identity never looked at `detail`, so this runs after the gate decided. A
    multi-value property carries one entry per asserted label (in the order the
    union lists them); any other property carries the wording of the chosen
    value. Empty when no claim has a detail, so properties without
    `detail: true` are untouched.
    """
    asserted = [c for c in claims if c.polarity == "asserted"]
    if not any(c.detail for c in asserted):
        return []
    if prop in settings.multi_value_set:
        picked = sorted(asserted, key=lambda c: c.value)
    else:
        wanted = normalize_value(value)
        matching = [c for c in asserted if c.value_norm == wanted or c.value == value]
        picked = sorted(matching, key=lambda c: c.event_time or 0.0, reverse=True)[:1]
    return [{"value": c.value, "detail": c.detail or ""} for c in picked if c.detail]
