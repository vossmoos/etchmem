"""Offline tests for 0.0.10 reconsolidation rules and extraction context.

  retraction   a negated value is kept but deprioritised, not deleted
  half-life    agreement weighs claims by age, so history is not disagreement
  trust        a newer value beats an older one only from a comparable source
  context      the extractor sees each signal's send time and source, plus the
               subject names and labels it should reuse

No network, no LLM. Run:  pytest -q   (from etchmem-server/)
"""
from __future__ import annotations

import os
import textwrap

os.environ["EMBEDDING_PROVIDER"] = "fake"

import pytest                                          # noqa: E402

from app import config as _cfg                         # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolution, ConflictResolver, ExtractedClaim,
    ExtractionResult, SignalInput, context_hint, signal_header,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.gate import route_and_resolve, trust_is_comparable  # noqa: E402
from app.service import MemoryService                  # noqa: E402
from app.stores import Claim                           # noqa: E402
from app.text import normalize_value                   # noqa: E402

DAY = 86400.0
S = _cfg.settings


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Pin every setting these tests depend on, and restore it afterwards."""
    for name, value in {"retraction_weight": 0.1, "agreement_half_life_days": 14.0,
                        "trust_same_odds_ratio": 3.0, "default_source_trust": 0.5,
                        "multi_value_properties": "", "trust_gap": 0.3,
                        "occurred_at_precedence": "extracted",
                        "extract_known_entities": 50, "extract_known_labels": 10,
                        "source_trust_json": "{}", "extract_min_batch": 1}.items():
        monkeypatch.setattr(S, name, value)


def claim(value, day, source="s", polarity="asserted", n=1, sig=None):
    return Claim(id=f"{value}-{day}-{source}-{polarity}", entity_id="e", entity_name="E",
                 property="p", value=value, value_norm=normalize_value(value),
                 polarity=polarity, event_time=1_800_000_000 + day * DAY,
                 sources=[source], evidence_signal_ids=[sig or f"{value}{day}{source}"],
                 corroboration_count=n)


# ── retraction ──────────────────────────────────────────────────────────────


def test_multi_value_retraction_keeps_value_at_low_weight(monkeypatch):
    monkeypatch.setattr(S, "multi_value_properties", "p")
    d = route_and_resolve("p", [claim("csv export", 1), claim("parquet export", 2),
                                claim("csv export", 5, polarity="negated")])
    assert d.value == "parquet export"                       # current = active only
    v = {x["value"]: x for x in d.values}
    assert v["csv export"]["active"] is False and v["csv export"]["retracted_at"]
    assert v["parquet export"]["active"] is True
    assert v["csv export"]["weight"] < v["parquet export"]["weight"] / 5
    assert d.values[0]["value"] == "parquet export"         # heaviest first


def test_reasserting_after_retraction_reactivates():
    d = route_and_resolve("p", [claim("on", 1), claim("on", 3, polarity="negated"),
                                claim("on", 6)])
    assert d.value == "on" and d.values[0]["active"] is True and d.policy == "unanimous"


def test_single_value_all_retracted_keeps_value_with_low_confidence():
    before = route_and_resolve("p", [claim("gold tier", 1)])
    d = route_and_resolve("p", [claim("gold tier", 1),
                                claim("gold tier", 3, polarity="negated")])
    assert d.value == "gold tier" and d.status == "settled" and d.policy == "retracted"
    assert d.confidence == pytest.approx(max(0.05, before.confidence * 0.1))


def test_retraction_weight_one_disables_it(monkeypatch):
    monkeypatch.setattr(S, "retraction_weight", 1.0)
    d = route_and_resolve("p", [claim("a", 1), claim("a", 3, polarity="negated")])
    assert d.values[0]["weight"] == 1.0


def test_older_negation_does_not_retract():
    d = route_and_resolve("p", [claim("a", 1, polarity="negated"), claim("a", 3)])
    assert d.values[0]["active"] is True and d.policy == "unanimous"


# ── half-life ───────────────────────────────────────────────────────────────


def _states():
    return [claim("proposed", 0, "x"), claim("agreed", 4, "x"), claim("dropped", 30, "x")]


def test_half_life_lifts_confidence_of_a_state_that_moved(monkeypatch):
    monkeypatch.setattr(S, "agreement_half_life_days", 0)
    flat = route_and_resolve("p", _states())
    monkeypatch.setattr(S, "agreement_half_life_days", 14)
    decayed = route_and_resolve("p", _states())
    assert flat.value == decayed.value == "dropped"
    assert flat.policy == decayed.policy == "recency"
    assert decayed.confidence > flat.confidence * 1.8


def test_half_life_is_relative_to_newest_claim_not_now():
    # Shifting the whole history in time changes nothing.
    a = route_and_resolve("p", _states())
    shifted = [claim(c.value, (c.event_time - 1_800_000_000) / DAY + 365, "x")
               for c in _states()]
    assert route_and_resolve("p", shifted).confidence == pytest.approx(a.confidence)


# ── trust ───────────────────────────────────────────────────────────────────


def test_trust_comparability_uses_odds_ratio():
    assert trust_is_comparable(0.7, 0.85)            # OR 2.43
    assert not trust_is_comparable(0.6, 0.85)        # OR 3.78
    assert trust_is_comparable(0.95, 0.5)            # more trusted always may
    assert not trust_is_comparable(0.85, 0.95)       # OR 3.35


def test_less_trusted_newer_value_is_contested_not_recency():
    trust = {"po": 0.85, "dev": 0.6, "call": 0.7}
    d = route_and_resolve("p", [claim("postgres", 1, "po"), claim("duckdb", 3, "dev")],
                          trust=trust)
    assert d.status == "contested" and d.policy == "trust_dispute"
    assert {c.value for c in d.competing} == {"postgres", "duckdb"}
    ok = route_and_resolve("p", [claim("postgres", 1, "po"), claim("duckdb", 3, "call")],
                           trust=trust)
    assert ok.policy == "recency" and ok.value == "duckdb"


def test_huge_odds_ratio_restores_pure_recency(monkeypatch):
    monkeypatch.setattr(S, "trust_same_odds_ratio", 1e9)
    d = route_and_resolve("p", [claim("a", 1, "po"), claim("b", 3, "dev")],
                          trust={"po": 0.95, "dev": 0.2})
    assert d.policy == "recency" and d.value == "b"


# ── extraction context ──────────────────────────────────────────────────────


def test_signal_header_and_hint_format():
    ts = 1_792_159_500.0                              # 2026-10-16 (Fri) 14:05 UTC
    h = signal_header(SignalInput("x", ts, "slack.team"))
    assert h == "[sent 2026-10-16 (Fri) 14:05 UTC · source slack.team]\n"
    assert signal_header(SignalInput("x", 0.0, "s")) == ""
    hint = context_hint(["Acme"], ["Acme · plan: gold tier"])
    assert "Known entities" in hint and "- Acme · plan: gold tier" in hint


class Recording(ClaimExtractor):
    """Text format 'entity|property|value|polarity'; records what it was given."""

    def __init__(self):
        self.calls = []

    def extract(self, text, known_entities=None):
        raise AssertionError("pipeline must call extract_signals")

    def extract_signals(self, signals, known_entities=None, known_labels=None):
        self.calls.append((signals, known_entities, known_labels))
        out = []
        for s in signals:
            ent, prop, val, pol = s.text.split("|")
            out.append(ExtractionResult(claims=[ExtractedClaim(
                entity_name=ent, entity_type="company", property=prop, value=val,
                polarity=pol)]))
        return out


class Res(ConflictResolver):
    def resolve(self, e, p, c):
        return ConflictResolution(current_value=c[0].value, status="contested",
                                  narrative="n", confidence=0.5)


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "t.yaml").write_text(textwrap.dedent("""
        domain: t
        properties:
          - name: feature
            detail: true
    """), encoding="utf-8")
    monkeypatch.setattr(S, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(S, "ext_dir", str(ext))
    monkeypatch.setattr(S, "multi_value_properties", "feature")
    rec = Recording()
    service = MemoryService(embedder=FakeEmbedding(), extractor=rec, resolver=Res())
    service.rec = rec
    return service


def test_pipeline_passes_send_time_source_and_context(svc):
    svc.remember("Acme|feature|csv export|asserted", "crm", "t", occurred_at="2026-09-01")
    svc.sleep()
    signals, names, labels = svc.rec.calls[0]
    assert signals[0].source == "crm" and signals[0].sent_at > 0
    assert names is None and labels is None                   # empty memory, no hint

    svc.remember("Acme|feature|csv export|negated", "crm", "t", occurred_at="2026-09-20")
    svc.sleep()
    _, names, labels = svc.rec.calls[1]
    assert names == ["Acme"]
    assert labels == ["Acme · feature: csv export"]


def test_retraction_end_to_end_versions_and_recall(svc):
    svc.remember("Acme|feature|csv export|asserted", "crm", "t", occurred_at="2026-09-01")
    svc.remember("Acme|feature|parquet export|asserted", "crm", "t", occurred_at="2026-09-02")
    svc.sleep()
    svc.remember("Acme|feature|csv export|negated", "crm", "t", occurred_at="2026-09-20")
    svc.sleep()
    e = next(x for x in svc.know("Acme").etches if x.property == "feature")
    assert e.current_value == "parquet export" and e.version == 2
    assert {(v.value, v.active) for v in e.values} == {("parquet export", True),
                                                        ("csv export", False)}
    hist = svc.history(e.id)
    assert hist[0]["current_value"] == "csv export, parquet export"   # it once was


def test_all_retracted_belief_sinks_in_recall(svc):
    svc.remember("Acme|plan|gold tier|asserted", "crm", "t", occurred_at="2026-09-01")
    svc.remember("Beta|plan|gold tier|asserted", "crm", "t", occurred_at="2026-09-01")
    svc.sleep()
    svc.remember("Acme|plan|gold tier|negated", "crm", "t", occurred_at="2026-09-10")
    svc.sleep()
    hits = {h.entity_name: h for h in svc.recall("gold tier", include_signals=False)}
    assert hits["Acme"].score < hits["Beta"].score
    assert hits["Acme"].values[0].active is False
    acme = next(x for x in svc.know("Acme").etches if x.property == "plan")
    assert acme.current_value == "gold tier" and acme.version == 2
    assert "retracted" in acme.narrative


def test_remember_accepts_epoch_seconds_over_http(tmp_path, monkeypatch):
    """The SDK sends occurred_at as a float; the API must accept it, not only ISO strings."""
    from fastapi.testclient import TestClient
    import app.main as main
    monkeypatch.setattr(S, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(S, "worker_enabled", False)
    monkeypatch.setattr(main, "MemoryService",
                        lambda: MemoryService(embedder=FakeEmbedding(), extractor=Recording(),
                                              resolver=Res()))
    with TestClient(main.app) as client:
        for value in (1_792_000_000.5, "1792000000", "2026-10-16T14:05:00Z"):
            r = client.post("/remember", json={"data": f"x {value}", "source": "s",
                                               "scope": "t", "occurred_at": value})
            assert r.status_code == 202, r.text
            assert r.json()["occurred_at_declared"] is True
