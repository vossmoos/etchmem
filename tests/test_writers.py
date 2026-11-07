"""Reads filtered by writer: a belief is shown only if every writer behind it is allowed.

Offline: fake embeddings and a scripted extractor.
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

    def extract(self, text, known_entities=None, known_labels=None):
        return ExtractionResult(claims=[
            ExtractedClaim(entity_name=e, entity_type=t, property=p, value=v)
            for e, t, p, v in (line.split("|") for line in text.splitlines())])

    def extract_batch(self, texts, known_entities=None, known_labels=None):
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
    s = MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())
    s.remember("Acme Corp|company|tier|gold", "public-bot", "sales")
    s.remember("Acme Corp|company|margin|22%", "internal-bot", "sales")
    s.remember("Acme Corp|company|account_manager|Lisa Brandt", "hubspot", "sales")
    # same fact, different text (identical text would be one signal)
    s.remember("Acme Corp|company|account_manager|Lisa Brandt\nAcme Corp|company|renewal|due",
               "internal-bot", "sales")
    s.sleep()
    return s


def _props(facts):
    return {e.property for e in facts.etches}


def test_no_filter_shows_everything(svc):
    assert _props(svc.know("Acme Corp")) == {"tier", "margin", "account_manager", "renewal"}


def test_a_public_reader_never_sees_internal_notes(svc):
    public = ["public-bot", "hubspot"]
    facts = {e.property: e for e in svc.know("Acme Corp", writers=public).etches}
    # margin and renewal rest only on internal-bot: hidden
    assert set(facts) == {"tier", "account_manager"}
    # account_manager rests on hubspot AND internal-bot: shown, as HubSpot states it,
    # without anything the internal notes added
    shared = facts["account_manager"]
    assert shared.current_value == "Lisa Brandt" and shared.source == "hubspot"
    assert shared.details == [] and shared.evidence is None and shared.claim_ids == []
    assert "internal" not in shared.narrative
    assert shared.quote is None or shared.quote.source == "hubspot"     # never internal-bot's text
    assert facts["tier"].source == "public-bot"
    hits = svc.recall("Acme", writers=public, include_signals=True, top_k=10)
    assert {h.property for h in hits if h.origin == "etch"} == {"tier", "account_manager"}
    assert all(h.source in public for h in hits)
    listed = svc.etches(writers=public)
    assert listed.total == 2 and {e.source for e in listed.etches} == {"public-bot", "hubspot"}


def test_dossier_and_history_of_an_unreadable_belief_are_unknown(svc):
    assert svc.dossier("company_acme::margin", writers=["public-bot"]) is None
    assert svc.history("company_acme::margin", writers=["public-bot"]) is None
    assert svc.dossier("company_acme::margin", writers=["internal-bot"]) is not None
    assert svc.dossier("company_acme::margin") is not None


def test_a_reader_of_every_writer_sees_mixed_beliefs(svc):
    every = ["public-bot", "internal-bot", "hubspot"]
    assert _props(svc.know("Acme Corp", writers=every)) == {"tier", "margin", "account_manager", "renewal"}


def test_writers_reach_the_http_api(svc, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main
    monkeypatch.setattr(main, "get_service", lambda: svc)
    client = TestClient(main.app)       # no lifespan: the test's own service, not a new one
    if True:
        r = client.get("/v1/entity/Acme Corp/etches", params={"writers": ["public-bot"]})
        assert {e["property"] for e in r.json()["etches"]} == {"tier"}   # no hubspot: no account_manager
        r = client.post("/v1/recall", json={"query": "Acme", "writers": ["public-bot"],
                                            "include_signals": False})
        assert {h["property"] for h in r.json()["results"]} == {"tier"}
        assert client.get("/v1/etch/company_acme::margin/dossier",
                          params={"writers": "public-bot"}).status_code == 404


# ── the same words from two writers ──────────────────────────────────────────


def test_two_writers_depositing_the_same_text_are_two_signals(svc):
    a, stored_a, _ = svc.remember("Acme Corp|company|tier|platinum", "agent-a", "support")
    b, stored_b, _ = svc.remember("Acme Corp|company|tier|platinum", "agent-b", "support")
    again, stored_again, _ = svc.remember("Acme Corp|company|tier|platinum", "agent-a", "support")
    assert stored_a and stored_b and a != b
    assert not stored_again and again == a                    # same writer, same words: a duplicate


def test_an_erasure_under_a_pre_0_4_id_still_refuses_the_text(svc):
    from app.hashing import content_hash
    text = "Acme Corp|company|contact|Maria Muster"
    svc.stores.right.add_erasure("old", "DSR", [content_hash(text)], {})
    sig_id, stored, _ = svc.remember(text, "agent-a", "support")
    assert not stored and svc.was_erased(sig_id)


def test_what_two_agents_were_shown_reaches_each_of_them(svc):
    svc.remember("Acme Corp|company|tier|platinum", "agent-a", "support")
    svc.remember("Acme Corp|company|tier|platinum", "agent-b", "support")
    svc.sleep()
    for writer in ("agent-a", "agent-b"):
        assert any(e.property == "tier" and e.current_value == "platinum"
                   for e in svc.know("Acme Corp", writers=[writer]).etches), writer
