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

from dataclasses import dataclass, field

from app.agents import CompetingClaim
from app.config import settings
from app.stores import Claim

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


def _confidence(chosen_corro: int, total_corro: int, n_sources: int, penalty: float = 1.0) -> float:
    agreement = (chosen_corro / total_corro) if total_corro else 0.5
    src_factor = 1.0 - 1.0 / (1.0 + n_sources)
    return max(0.05, min(0.99, agreement * src_factor * penalty))


def route_and_resolve(prop: str, claims: list[Claim]) -> GateDecision:
    asserted = [c for c in claims if c.polarity == "asserted"]
    if not asserted:
        return GateDecision(ROUTE_POLICY, "unknown", "settled", 0.1, "no_assertion")

    # Aggregate by normalized value.
    by_value: dict[str, dict] = {}
    for c in asserted:
        g = by_value.setdefault(c.value_norm, {"value": c.value, "corro": 0,
                                               "sources": set(), "event_time": 0.0})
        g["corro"] += max(1, c.corroboration_count)
        g["sources"].update(c.sources)
        g["event_time"] = max(g["event_time"], c.event_time or 0.0)

    total_corro = sum(g["corro"] for g in by_value.values())

    # ── Multi-valued property → union, never a conflict ────────────────────
    if prop in settings.multi_value_set:
        values = sorted(g["value"] for g in by_value.values())
        all_sources = set().union(*(g["sources"] for g in by_value.values()))
        return GateDecision(ROUTE_POLICY, ", ".join(values), "settled",
                            _confidence(total_corro, total_corro, len(all_sources)),
                            "cardinality_union")

    # ── Single distinct value → AGREE ──────────────────────────────────────
    if len(by_value) == 1:
        g = next(iter(by_value.values()))
        return GateDecision(ROUTE_AGREE, g["value"], "settled",
                            _confidence(g["corro"], total_corro, len(g["sources"])),
                            "unanimous")

    # ── Differing values → try deterministic policies ──────────────────────
    # 1) recency (state-machine: latest event_time wins, if unique)
    if any(g["event_time"] > 0 for g in by_value.values()):
        ranked = sorted(by_value.items(), key=lambda kv: kv[1]["event_time"], reverse=True)
        (top_v, top_g), (_, runner_g) = ranked[0], ranked[1]
        if top_g["event_time"] > runner_g["event_time"]:
            return GateDecision(ROUTE_POLICY, top_g["value"], "settled",
                                _confidence(top_g["corro"], total_corro,
                                            len(top_g["sources"]), penalty=0.85),
                                "recency")

    # 2) source trust gap
    trust = settings.source_trust
    if trust:
        def value_trust(g) -> float:
            return max((trust.get(s, 0.5) for s in g["sources"]), default=0.5)
        ranked = sorted(by_value.items(), key=lambda kv: value_trust(kv[1]), reverse=True)
        (top_v, top_g), (_, runner_g) = ranked[0], ranked[1]
        if value_trust(top_g) - value_trust(runner_g) >= settings.trust_gap:
            return GateDecision(ROUTE_POLICY, top_g["value"], "settled",
                                _confidence(top_g["corro"], total_corro,
                                            len(top_g["sources"]), penalty=0.9),
                                "source_trust")

    # ── Otherwise: genuine conflict → escalate ─────────────────────────────
    competing = [
        CompetingClaim(value=g["value"], polarity="asserted",
                       sources=sorted(g["sources"]), corroboration_count=g["corro"],
                       event_time=g["event_time"] or None)
        for g in by_value.values()
    ]
    # Fallback value/confidence if the LLM is unavailable: most-corroborated.
    top = max(by_value.values(), key=lambda g: g["corro"])
    return GateDecision(ROUTE_CONTESTED, top["value"], "contested",
                        _confidence(top["corro"], total_corro, len(top["sources"]),
                                    penalty=0.5),
                        "ambiguous", competing)
