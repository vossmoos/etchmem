"""
Offline tests for claims anonymization (ETCHMEM_CLAIMS_ANONYMIZATION).

Person/company subject names → consistent numbered tokens; regex safety net
scrubs cards/IBANs/emails/phones from values; recall excludes raw signals.
No network, no LLM key.
"""
from __future__ import annotations

import os
import tempfile

os.environ["EMBEDDING_PROVIDER"] = "fake"

import pytest                                          # noqa: E402

from app import config as _config                      # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolver, ConflictResolution,
    ExtractedClaim, ExtractionResult,
)
from app.anonymize import scrub                        # noqa: E402
from app.embeddings import FakeEmbedding               # noqa: E402
from app.service import MemoryService                  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_anonymization():
    yield
    _config.settings.claims_anonymization = False


class PIIExtractor(ClaimExtractor):
    def extract(self, signal_text: str, known_entities=None) -> ExtractionResult:
        t = signal_text.lower()
        claims = []
        if "john" in t:
            claims.append(ExtractedClaim(
                entity_name="John Smith", entity_type="person",
                property="role", value="cfo", confidence=0.9))
        if "maria" in t:
            claims.append(ExtractedClaim(
                entity_name="Maria Lopez", entity_type="person",
                property="role", value="ceo", confidence=0.9))
        if "acme" in t:
            claims.append(ExtractedClaim(
                entity_name="Acme Corp", entity_type="company",
                property="billing_card", value="4111 1111 1111 1111",
                confidence=0.9))
        return ExtractionResult(claims=claims)


class StubResolver(ConflictResolver):
    def resolve(self, entity_name, prop, claims) -> ConflictResolution:
        return ConflictResolution(
            current_value=claims[0].value, status="contested",
            narrative=f"{entity_name} {prop} is disputed.", confidence=0.4)


def _service(anonymize: bool = True) -> MemoryService:
    _config.settings.data_dir = tempfile.mkdtemp(prefix="etchmem-anon-")
    _config.settings.claims_anonymization = anonymize
    return MemoryService(
        embedder=FakeEmbedding(), extractor=PIIExtractor(), resolver=StubResolver())


# ── tests ────────────────────────────────────────────────────────────────────

def test_person_tokens_numbered_and_consistent():
    svc = _service()
    svc.remember("John Smith is the new CFO", "agent-1", "hr")
    svc.sleep()
    svc.remember("Maria Lopez is the CEO", "agent-2", "hr")
    svc.sleep()
    # Same person again, different wording/source → same token, same etch.
    svc.remember("John Smith was appointed CFO", "agent-3", "hr")
    svc.sleep()

    etches = svc.stores.right.all_etches()
    by_name = {e.entity_name: e for e in etches if e.property == "role"}
    assert set(by_name) == {"[PERSON_1]", "[PERSON_2]"}
    assert by_name["[PERSON_1]"].current_value == "cfo"     # John stayed PERSON_1
    assert "[PERSON_1]" in by_name["[PERSON_1]"].narrative  # narrative uses token
    assert svc.stats()["entities"] == 2                     # resolution intact


def test_company_token_and_card_scrubbed():
    svc = _service()
    svc.remember("Acme card on file", "agent-1", "billing")
    svc.sleep()

    (etch,) = svc.stores.right.all_etches()
    assert etch.entity_name == "[COMPANY_1]"
    assert etch.current_value == "[BANK_CARD]"
    assert "4111" not in etch.narrative


def test_recall_excludes_raw_signals():
    svc = _service()
    svc.remember("John Smith is the new CFO", "agent-1", "hr")
    svc.sleep()

    results = svc.recall("who is the cfo", top_k=10, include_signals=True)
    assert results
    assert all(r.origin == "etch" for r in results)
    assert all("John" not in r.content for r in results)


def test_disabled_keeps_real_names():
    svc = _service(anonymize=False)
    svc.remember("John Smith is the new CFO", "agent-1", "hr")
    svc.sleep()

    (etch,) = svc.stores.right.all_etches()
    assert etch.entity_name == "John Smith"


def test_scrub_regex_safety_net():
    assert scrub("card 4111 1111 1111 1111 ok") == "card [BANK_CARD] ok"
    assert scrub("iban DE89370400440532013000") == "iban [IBAN]"
    assert scrub("mail john@acme.com") == "mail [EMAIL]"
    assert scrub("call +49 170 1234567") == "call [PHONE]"
    assert scrub("plain text stays") == "plain text stays"
