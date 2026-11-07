"""Erasing a name that is only a value.

Offline: fake embeddings and scripted extractors.
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
        claims = []
        for line in text.splitlines():
            if line.startswith("["):
                continue
            parts = [p.strip() for p in line.split("|")]
            claims.append(ExtractedClaim(entity_name=parts[0], entity_type=parts[1],
                                         property=parts[2], value=parts[3]))
        return ExtractionResult(claims=claims)

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
    return MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())


def test_a_person_who_is_only_a_value_is_erased_by_name(svc):
    svc.remember("Acme Corp|company|primary_support_contact|Petra Lang", "zendesk", "support",
                 source_record_id="zendesk:org:42")
    svc.remember("Acme Corp|company|account_manager|Lisa Brandt", "hubspot", "support",
                 source_record_id="hubspot:deal:1")
    svc.sleep()
    assert svc.know("Petra Lang") is None                       # never a subject
    report = svc.forget(entity="Petra Lang", reason="test")
    assert report.entity_found is False
    assert report.signals_deleted == 1 and report.signals_matched_by_name == 1
    assert report.note is None
    facts = svc.know("Acme Corp")
    props = {e.property: e.current_value for e in facts.etches}
    assert "primary_support_contact" not in props or "Petra" not in props["primary_support_contact"]
    assert props["account_manager"] == "Lisa Brandt"            # the rest stays
    # the erased record cannot be deposited again
    again = svc.remember("Acme Corp|company|primary_support_contact|Petra Lang", "zendesk",
                         "support", source_record_id="zendesk:org:42")
    assert again[1] is False and svc.was_erased(again[0])


def test_erasing_an_unknown_name_says_nothing_matched(svc):
    svc.remember("Acme Corp|company|account_manager|Lisa Brandt", "hubspot", "support")
    svc.sleep()
    report = svc.forget(entity="Nobody Atall", reason="test")
    assert report.signals_deleted == 0 and "nothing matched" in report.note


def test_records_are_not_subjects_in_the_prompt():
    from app.agents import claim_system
    assert "SUBJECTS ARE THINGS, NOT RECORDS" in claim_system()
