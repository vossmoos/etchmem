"""
Offline end-to-end tests for the full cascade:
  remember → batch(dedup) → extract(claims) → fold(gate/resolve) → etch + versions

Uses a deterministic FakeEmbedding and stub agents — no network, no LLM key.
Run:  pytest -q   (from etchmem-server/)
"""
from __future__ import annotations

import os
import tempfile

os.environ["EMBEDDING_PROVIDER"] = "fake"

from app import config as _config                      # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolver, ConflictResolution,
    ExtractedClaim, ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.service import MemoryService                  # noqa: E402
from app.text import normalize_entity_name, slugify    # noqa: E402


# ── stubs ────────────────────────────────────────────────────────────────────

class StubExtractor(ClaimExtractor):
    def extract(self, signal_text: str, known_entities=None) -> ExtractionResult:
        t = signal_text.lower()
        if "dao" in t:
            ent = "Dao Corp"
        elif "acme" in t:
            ent = "Acme Corp"
        elif "globex" in t:
            ent = "Globex Corp"
        else:
            return ExtractionResult(claims=[])
        if "broke" in t or "broken" in t:
            val = "broken"
        elif "cancel" in t:
            val = "cancelled"
        elif "signed" in t or "sign" in t:
            val = "signed"
        else:
            return ExtractionResult(claims=[])
        event_time = "2026-06-24T00:00:00" if "@same" in t else None
        return ExtractionResult(claims=[ExtractedClaim(
            entity_name=ent, entity_type="company", property="contract_status",
            value=val, event_time=event_time, confidence=0.8)])


class StubResolver(ConflictResolver):
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, entity_name, prop, claims) -> ConflictResolution:
        self.calls += 1
        return ConflictResolution(
            current_value=claims[0].value, status="contested",
            narrative=f"{entity_name} {prop} is disputed.", confidence=0.4)


def _service(resolver=None) -> MemoryService:
    _config.settings.data_dir = tempfile.mkdtemp(prefix="etchmem-test-")
    return MemoryService(
        embedder=FakeEmbedding(), extractor=StubExtractor(),
        resolver=resolver or StubResolver())


def _etch_id(name: str) -> str:
    return f"{slugify('company:' + normalize_entity_name(name, 'company'))}::contract_status"


# ── tests ────────────────────────────────────────────────────────────────────

def test_forms_etch_no_contamination():
    svc = _service()
    svc.remember("Acme Corp signed the enterprise contract", "agent-33", "sales")
    svc.remember("Dao Corp signed the enterprise contract", "agent-34", "sales")
    summary = svc.sleep()

    assert summary["etches_formed"] == 2          # Acme and Dao kept separate
    st = svc.stats()
    assert st["entities"] == 2
    assert st["etches"] == 2

    acme = svc.stores.right.get_etch(_etch_id("Acme Corp"))
    dao = svc.stores.right.get_etch(_etch_id("Dao Corp"))
    assert acme.entity_name == "Acme Corp" and acme.current_value == "signed"
    assert dao.entity_name == "Dao Corp" and dao.current_value == "signed"
    assert acme.id != dao.id                       # no merged blob


def test_corroboration_raises_confidence_same_entity():
    svc = _service()
    svc.remember("Acme Corp signed the enterprise contract", "agent-33", "sales")
    svc.sleep()
    c1 = svc.stores.right.get_etch(_etch_id("Acme Corp")).confidence

    # Different source, different wording, SAME entity + value → corroboration.
    svc.remember("ACME Corporation signed the contract", "agent-99", "sales")
    svc.sleep()
    etch = svc.stores.right.get_etch(_etch_id("Acme Corp"))

    assert svc.stats()["entities"] == 1            # Acme == ACME Corporation
    assert svc.stats()["claims"] == 1             # one claim, corroborated
    assert etch.confidence > c1                    # more sources → higher confidence


def test_recency_supersede_creates_version():
    svc = _service()
    svc.remember("Acme Corp signed the enterprise contract", "agent-33", "sales")
    svc.sleep()
    svc.remember("Acme Corp broke the contract", "agent-33", "sales")
    svc.sleep()

    etch = svc.stores.right.get_etch(_etch_id("Acme Corp"))
    assert etch.current_value == "broken"         # recency policy resolved it
    assert etch.status == "settled"
    assert etch.version == 2

    versions = svc.history(_etch_id("Acme Corp"))
    assert [v["current_value"] for v in versions] == ["signed", "broken"]


def test_genuine_conflict_is_contested_and_calls_resolver():
    resolver = StubResolver()
    svc = _service(resolver=resolver)
    # Same forced event_time → recency can't break the tie → CONTESTED.
    svc.remember("Globex signed the contract @same", "agent-1", "sales")
    svc.remember("Globex cancelled the contract @same", "agent-2", "sales")
    svc.sleep()

    etch = svc.stores.right.get_etch(_etch_id("Globex Corp"))
    assert etch.status == "contested"
    assert resolver.calls == 1                     # top-tier model invoked once
    assert svc.stats()["contested"] == 1


def test_time_travel_recall():
    svc = _service()
    svc.remember("Acme Corp signed the enterprise contract", "agent-33", "sales")
    svc.sleep()
    svc.remember("Acme Corp broke the contract", "agent-33", "sales")
    svc.sleep()

    versions = svc.history(_etch_id("Acme Corp"))
    v1c, v2c = versions[0]["created_at"], versions[1]["created_at"]
    mid = (v1c + v2c) / 2.0

    # HEAD recall → current belief.
    head = [r for r in svc.recall("acme contract status", top_k=5) if r.origin == "etch"]
    assert head and head[0].value == "broken"

    # Time-travel recall → belief as of `mid` (before the "broke" signal).
    past = [r for r in svc.recall("acme contract status", top_k=5, as_of=str(mid))
            if r.origin == "etch"]
    assert past and past[0].value == "signed" and past[0].version == 1
