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
    assert _props(svc.know("Acme Corp", writers=public)) == {"tier"}
    # account_manager rests on hubspot AND internal-bot: hidden, not half-shown
    hits = svc.recall("Acme", writers=public, include_signals=True, top_k=10)
    assert {h.property for h in hits if h.origin == "etch"} == {"tier"}
    assert all(h.source in public for h in hits if h.origin == "signal")
    listed = svc.etches(writers=public)
    assert listed.total == 1 and listed.etches[0].property == "tier"


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
        assert {e["property"] for e in r.json()["etches"]} == {"tier"}
        r = client.post("/v1/recall", json={"query": "Acme", "writers": ["public-bot"],
                                            "include_signals": False})
        assert {h["property"] for h in r.json()["results"]} == {"tier"}
        assert client.get("/v1/etch/company_acme::margin/dossier",
                          params={"writers": "public-bot"}).status_code == 404
