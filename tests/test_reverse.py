"""A replaced fact retires the beliefs that state it from the other side.

"Acme · account_manager = Lisa" and "Lisa · account_manager_for = Acme" come from
the same record. When Acme's account manager becomes Tom, the reverse belief must
stop answering "Lisa". Offline: fake embeddings and a scripted extractor.
"""
from __future__ import annotations

import os

os.environ["EMBEDDING_PROVIDER"] = "fake"

import pytest                                          # noqa: E402

from app import config as _cfg                         # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolution, ConflictResolver, ExtractedClaim, ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.service import MemoryService                  # noqa: E402


class Scripted(ClaimExtractor):
    """Each line of a signal is 'entity|type|property|value'."""

    def extract(self, text, known_entities=None):
        claims = []
        for line in text.splitlines():
            entity, etype, prop, value = (p.strip() for p in line.split("|"))
            claims.append(ExtractedClaim(entity_name=entity, entity_type=etype,
                                         property=prop, value=value))
        return ExtractionResult(claims=claims)

    def extract_batch(self, texts, known_entities=None):
        return [self.extract(t) for t in texts]


class Res(ConflictResolver):
    def resolve(self, e, p, c):
        return ConflictResolution(current_value=c[-1].value, status="settled",
                                  narrative="n", confidence=0.9)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr(_cfg.settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(_cfg.settings, "ext_dir", str(tmp_path / "no-ext"))
    monkeypatch.setattr(_cfg.settings, "extract_min_batch", 1)
    return MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())


DEAL = ("Acme Corp|company|account_manager|Lisa Brandt\n"
        "Lisa Brandt|person|account_manager_for|Acme Corp\n"
        "Lisa Brandt|person|role|account manager")
HANDOVER = "Acme Corp|company|account_manager|Tom Krause"


def _remember(svc, text, when, record):
    svc.remember(text, "hubspot", "sales", occurred_at=when, source_record_id=record)
    svc.sleep()


def _belief(svc, subject, prop):
    return next((e for e in svc.know(subject, scope="sales").etches if e.property == prop), None)


def test_a_replaced_fact_retires_its_reverse_belief(svc):
    _remember(svc, DEAL, "2026-09-15", "hubspot:deal:881")
    assert _belief(svc, "Lisa Brandt", "account_manager_for").current_value == "Acme Corp"
    _remember(svc, HANDOVER, "2026-11-07", "chat:1")

    assert _belief(svc, "Acme Corp", "account_manager").current_value == "Tom Krause"
    reverse = _belief(svc, "Lisa Brandt", "account_manager_for")
    assert "No longer valid" in reverse.narrative
    assert reverse.values and not any(v.active for v in reverse.values)
    # unrelated beliefs about Lisa from the same record stay as they are
    assert "No longer valid" not in _belief(svc, "Lisa Brandt", "role").narrative
    # recall ranks the retired belief below the current one
    hits = svc.recall("Acme Corp account manager", scope="sales", include_signals=False, top_k=10)
    order = [(h.entity_name, h.property) for h in hits]
    assert order.index(("Acme Corp", "account_manager")) < order.index(("Lisa Brandt", "account_manager_for"))
    # history keeps what was believed on 1 November
    then = svc.know("Lisa Brandt", scope="sales", as_of="2026-11-01", as_of_basis="event")
    assert any(e.property == "account_manager_for" and e.current_value == "Acme Corp"
               and "No longer valid" not in e.narrative for e in then.etches)


def test_a_reverse_belief_with_its_own_evidence_is_kept(svc):
    _remember(svc, DEAL, "2026-09-15", "hubspot:deal:881")
    # a separate record says the reverse fact on its own
    _remember(svc, "Lisa Brandt|person|account_manager_for|Acme Corp", "2026-10-01", "slack:1")
    _remember(svc, HANDOVER, "2026-11-07", "chat:1")
    reverse = _belief(svc, "Lisa Brandt", "account_manager_for")
    assert "No longer valid" not in reverse.narrative          # not ours to retire


def test_the_same_value_again_retires_nothing(svc):
    _remember(svc, DEAL, "2026-09-15", "hubspot:deal:881")
    _remember(svc, "Acme Corp|company|account_manager|Lisa Brandt", "2026-10-01", "slack:2")
    assert "No longer valid" not in _belief(svc, "Lisa Brandt", "account_manager_for").narrative


def test_the_extractor_is_told_to_state_a_relation_once():
    from app.agents import claim_system
    text = claim_system()
    assert "ONE DIRECTION PER RELATION" in text and "account_manager_for" in text
