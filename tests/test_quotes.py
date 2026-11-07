"""Quotes: each belief from recall and know carries the newest signal behind it.

Offline: fake embeddings and a scripted extractor. Run:  pytest -q
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
from app.quotes import excerpt                         # noqa: E402
from app.service import MemoryService                  # noqa: E402

TICKET = (
    "Ticket 123 from Acme. Subject: Invoice wrong. Status open, priority high. "
    "The customer writes that the October invoice was charged twice and asks for a "
    "credit note before the end of the month, otherwise they will escalate to their CFO."
)


class Scripted(ClaimExtractor):
    """Every signal → Acme.billing_issue; the value is the text before ' :: ' if present."""

    def extract(self, text, known_entities=None):
        value = text.split(" :: ")[0] if " :: " in text else "double-charged invoice"
        return ExtractionResult(claims=[
            ExtractedClaim(entity_name="Acme", entity_type="company", property="billing_issue",
                           value=value, detail="The October invoice was charged twice."),
            ExtractedClaim(entity_name="Acme", entity_type="company", property="priority",
                           value="high"),
        ])

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


def _remember(svc, text, when, record="zendesk:ticket:123"):
    svc.remember(text, "zendesk", "support", occurred_at=when, source_record_id=record,
                 actor={"agent": "mossy", "user": "maria@acme.test"})
    svc.sleep()


def _belief(svc, prop="billing_issue"):
    return next(e for e in svc.know("Acme", scope="support").etches if e.property == prop)


def test_quotes_ref_leaves_out_the_excerpt_where_a_wording_exists(svc):
    _remember(svc, TICKET, "2026-11-07T10:00:00Z")
    belief = next(e for e in svc.know("Acme", scope="support", quotes="ref").etches
                  if e.property == "billing_issue")
    assert belief.details                                       # it has a wording
    quote = belief.quote
    assert quote.source == "zendesk" and quote.source_record_id == "zendesk:ticket:123"
    assert quote.actor == {"agent": "mossy", "user": "maria@acme.test"}
    assert quote.excerpt is None                                # would repeat the wording
    hit = next(h for h in svc.recall("October invoice", scope="support", include_signals=False,
                                     quotes="ref") if h.property == "billing_issue")
    assert hit.quote and hit.quote.signal_id == quote.signal_id and hit.quote.excerpt is None


def test_quotes_ref_keeps_the_excerpt_on_a_plain_value(svc):
    _remember(svc, TICKET, "2026-11-07T10:00:00Z")
    plain = next(e for e in svc.know("Acme", scope="support", quotes="ref").etches
                 if e.property == "priority")
    assert not plain.details and plain.current_value == "high"
    assert "priority high" in plain.quote.excerpt and len(plain.quote.excerpt) <= 302


def test_by_default_every_quote_carries_the_excerpt(svc):
    _remember(svc, TICKET, "2026-11-07T10:00:00Z")
    issue = _belief(svc)
    assert issue.details and "charged twice" in issue.quote.excerpt
    assert len(issue.quote.excerpt) <= 302
    hit = next(h for h in svc.recall("October invoice", scope="support", include_signals=False)
               if h.property == "billing_issue")
    assert "charged twice" in hit.quote.excerpt


def test_the_newest_signal_for_the_current_value_is_quoted(svc):
    _remember(svc, "first report :: double-charged invoice", "2026-11-01T09:00:00Z", "zendesk:ticket:1")
    _remember(svc, "second report :: double-charged invoice", "2026-11-05T09:00:00Z", "zendesk:ticket:2")
    quote = _belief(svc).quote
    assert quote.source_record_id == "zendesk:ticket:2"
    assert quote.excerpt.startswith("second report")


def test_time_travel_quotes_what_was_known_then(svc):
    _remember(svc, "first report :: double-charged invoice", "2026-11-01T09:00:00Z", "zendesk:ticket:1")
    first = _belief(svc)
    _remember(svc, "second report :: double-charged invoice", "2026-11-05T09:00:00Z", "zendesk:ticket:2")
    then = svc.know("Acme", scope="support", as_of="2026-11-03T00:00:00Z", as_of_basis="event")
    old = next(e for e in then.etches if e.property == "billing_issue")
    assert old.quote.source_record_id == "zendesk:ticket:1"
    assert first.quote.source_record_id == "zendesk:ticket:1"


def test_anonymization_leaves_out_text_and_author(svc, monkeypatch):
    _remember(svc, TICKET, "2026-11-07T10:00:00Z")
    monkeypatch.setattr(_cfg.settings, "claims_anonymization", True)
    quote = _belief(svc).quote                                  # would carry an excerpt
    assert quote.excerpt is None and quote.actor == {}
    assert quote.source_record_id == "zendesk:ticket:123"       # still traceable


def test_quotes_can_be_turned_off(svc, monkeypatch):
    _remember(svc, TICKET, "2026-11-07T10:00:00Z")
    monkeypatch.setattr(_cfg.settings, "quote_chars", 0)
    assert _belief(svc).quote is None


def test_excerpt_centres_on_the_fact_and_cuts_on_words():
    text = "Intro words. " * 40 + "The October invoice was charged twice. " + "Tail words. " * 40
    out = excerpt(text, ["charged twice"], 120)
    assert "charged twice" in out and out.startswith("…") and out.endswith("…")
    assert len(out) <= 122
    assert excerpt("short text", ["x"], 300) == "short text"
