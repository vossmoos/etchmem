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
    # Every value ever asserted, heaviest first, with retraction state.
    values: list[dict] = field(default_factory=list)


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


def _confidence(chosen_weight: float, total_weight: float, strength: float,
                penalty: float = 1.0) -> float:
    agreement = (chosen_weight / total_weight) if total_weight else 0.5
    return max(0.05, min(0.99, agreement * strength * penalty))


_DAY = 86400.0


def _age_weight(event_time: float, ref_time: float) -> float:
    """How much a claim still counts, by its age relative to the newest claim.

    Exponential decay with a configurable half-life. Measured from the newest
    claim in the fold, not from "now": a belief nobody touches keeps its
    confidence instead of eroding with the calendar.
    """
    half_life = settings.agreement_half_life_days
    if half_life <= 0 or not event_time or not ref_time:
        return 1.0
    age_days = max(0.0, ref_time - event_time) / _DAY
    return 0.5 ** (age_days / half_life)


def _odds(t: float) -> float:
    t = min(max(t, 0.01), 0.99)
    return t / (1.0 - t)


def trust_is_comparable(newer: float, older: float) -> bool:
    """May a newer, less trusted statement override an older, more trusted one?

    Trust is compared as odds, the scale on which people judge probabilities
    roughly linearly. A newer source at least as trusted always qualifies; a
    less trusted one only while the odds ratio stays within
    `trust_same_odds_ratio` (default 3, the conventional "barely worth
    mentioning" line for a Bayes factor).
    """
    if newer >= older:
        return True
    return _odds(older) / _odds(newer) <= settings.trust_same_odds_ratio


def route_and_resolve(prop: str, claims: list[Claim], *,
                      trust: dict[str, float] | None = None,
                      signal_sources: dict[str, str] | None = None) -> GateDecision:
    """Route one (entity, property) fold.

    `trust`          source → 0..1 (ext `sources:` block, env override on top).
    `signal_sources` signal id → source, so repeats from one source can be told
                     from independent witnesses. Without it every distinct
                     source counts once.

    Three time-aware rules shape the outcome (all configurable):
      retraction  a value whose newest negation is newer than its newest
                  assertion stays known but drops to `retraction_weight`;
      half-life   agreement weighs each claim by its age relative to the
                  newest claim, so a state machine's history does not read as
                  disagreement;
      trust       a newer value wins on recency only if its source is about as
                  trusted as the older value's; otherwise it is contested.
    """
    trust = settings.source_trust if trust is None else trust
    signal_sources = signal_sources or {}
    asserted = [c for c in claims if c.polarity == "asserted"]
    if not asserted:
        return GateDecision(ROUTE_POLICY, "unknown", "settled", 0.1, "no_assertion")

    ref_time = max((c.event_time or 0.0) for c in claims)
    negated_at: dict[str, float] = {}
    for c in claims:
        if c.polarity == "negated":
            negated_at[c.value_norm] = max(negated_at.get(c.value_norm, 0.0),
                                           c.event_time or 0.0)

    # Aggregate by normalized value.
    by_value: dict[str, dict] = {}
    for c in asserted:
        g = by_value.setdefault(c.value_norm, {"value": c.value, "corro": 0, "claims": [],
                                               "sources": set(), "signals": set(),
                                               "event_time": 0.0, "weight": 0.0})
        n = max(1, c.corroboration_count)
        g["corro"] += n
        g["weight"] += n * _age_weight(c.event_time or 0.0, ref_time)
        g["claims"].append(c)
        g["sources"].update(c.sources)
        g["signals"].update(c.evidence_signal_ids)
        g["event_time"] = max(g["event_time"], c.event_time or 0.0)

    rw = min(max(settings.retraction_weight, 0.0), 1.0)
    for norm, g in by_value.items():
        retracted = negated_at.get(norm, 0.0)
        g["retracted_at"] = retracted if retracted > g["event_time"] else None
        if g["retracted_at"] is not None:
            g["weight"] *= rw

    total_weight = sum(g["weight"] for g in by_value.values()) or 1.0
    active = {k: g for k, g in by_value.items() if g["retracted_at"] is None}

    def value_trust(g) -> float:
        return max((trust.get(s, settings.default_source_trust) for s in g["sources"]),
                   default=settings.default_source_trust)

    values_out = sorted(
        ({"value": g["value"], "weight": round(g["weight"] / total_weight, 4),
          "active": g["retracted_at"] is None,
          "last_asserted_at": g["event_time"] or None,
          "retracted_at": g["retracted_at"]} for g in by_value.values()),
        key=lambda v: (-v["weight"], v["value"]))

    def judge(group_list, weight, penalty=1.0, total=None):
        """(confidence, evidence) for the signals behind one or several groups."""
        ids = set().union(*(g["signals"] for g in group_list))
        cl = [c for g in group_list for c in g["claims"]]
        strength, ev = _evidence(ids, cl, signal_sources, trust)
        return _confidence(weight, total_weight if total is None else total,
                           strength, penalty), ev

    def decide(route, value, status, conf, policy, ev, competing=None):
        return GateDecision(route, value, status, conf, policy, competing or [],
                            evidence=ev, values=values_out)

    # ── Everything retracted: keep the last value, at retraction weight ────
    if not active:
        latest = max(by_value.values(), key=lambda g: g["event_time"])
        if prop in settings.multi_value_set:
            groups = list(by_value.values())
            value = ", ".join(sorted(g["value"] for g in groups))
        else:
            groups, value = [latest], latest["value"]
        conf, ev = judge(groups, 1.0, penalty=rw, total=1.0)
        return decide(ROUTE_POLICY, value, "settled", conf, "retracted", ev)

    # ── Multi-valued property → union of active values, never a conflict ───
    if prop in settings.multi_value_set:
        groups = list(active.values())
        conf, ev = judge(groups, 1.0, total=1.0)
        return decide(ROUTE_POLICY, ", ".join(sorted(g["value"] for g in groups)),
                      "settled", conf, "cardinality_union", ev)

    # ── Single active value → AGREE ────────────────────────────────────────
    if len(active) == 1:
        g = next(iter(active.values()))
        conf, ev = judge([g], g["weight"])
        return decide(ROUTE_AGREE, g["value"], "settled", conf,
                      "unanimous" if len(by_value) == 1 else "retraction", ev)

    # ── Differing active values → try deterministic policies ───────────────
    # 1) recency (latest event_time wins, if unique AND its source is about as
    #    trusted as the strongest older value's — otherwise it is a dispute)
    escalate = False
    if any(g["event_time"] > 0 for g in active.values()):
        ranked = sorted(active.values(), key=lambda g: g["event_time"], reverse=True)
        top_g, runner_g = ranked[0], ranked[1]
        if top_g["event_time"] > runner_g["event_time"]:
            older_trust = max(value_trust(g) for g in ranked[1:])
            if trust_is_comparable(value_trust(top_g), older_trust):
                conf, ev = judge([top_g], top_g["weight"], penalty=0.85)
                return decide(ROUTE_POLICY, top_g["value"], "settled", conf,
                              "recency", ev)
            escalate = True

    # 2) source trust gap (only when time could not order the values)
    if trust and not escalate:
        ranked = sorted(active.values(), key=value_trust, reverse=True)
        top_g, runner_g = ranked[0], ranked[1]
        if value_trust(top_g) - value_trust(runner_g) >= settings.trust_gap:
            conf, ev = judge([top_g], top_g["weight"], penalty=0.9)
            return decide(ROUTE_POLICY, top_g["value"], "settled", conf,
                          "source_trust", ev)

    # ── Otherwise: genuine conflict → escalate ─────────────────────────────
    competing = [
        CompetingClaim(value=g["value"], polarity="asserted",
                       sources=sorted(g["sources"]), corroboration_count=g["corro"],
                       event_time=g["event_time"] or None)
        for g in active.values()
    ]
    # Fallback value/confidence if the LLM is unavailable: the heaviest value.
    top = max(active.values(), key=lambda g: g["weight"])
    conf, ev = judge([top], top["weight"], penalty=0.5)
    return decide(ROUTE_CONTESTED, top["value"], "contested", conf,
                  "trust_dispute" if escalate else "ambiguous", ev, competing)


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
        # Only the labels the union kept: a retracted label's wording lives on
        # in the version history, not in the current belief.
        kept = {normalize_value(v) for v in value.split(", ")}
        asserted = [c for c in asserted if c.value_norm in kept
                    or normalize_value(c.value) in kept] or asserted
    if prop in settings.multi_value_set:
        picked = sorted(asserted, key=lambda c: c.value)
    else:
        wanted = normalize_value(value)
        matching = [c for c in asserted if c.value_norm == wanted or c.value == value]
        picked = sorted(matching, key=lambda c: c.event_time or 0.0, reverse=True)[:1]
    return [{"value": c.value, "detail": c.detail or ""} for c in picked if c.detail]
