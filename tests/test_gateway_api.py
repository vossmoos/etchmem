"""
Offline tests for what a gateway in front of etchmem relies on: the API key,
the /v1 paths, deposit metadata (source record, actor), reads over several
scopes, and erasure (`forget`).

No network, no LLM key. Runs on DuckDB, or on Postgres with
ETCHMEM_TEST_DATABASE_URL set (see conftest.py).
"""
from __future__ import annotations

import os
import tempfile

os.environ["EMBEDDING_PROVIDER"] = "fake"

from fastapi.testclient import TestClient              # noqa: E402

from app import config as _config                      # noqa: E402
from app import main as app_main                       # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolution, ConflictResolver, ExtractedClaim,
    ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.service import MemoryService                  # noqa: E402
from app.stores import ERASED                          # noqa: E402


class Scripted(ClaimExtractor):
    """Signal text "<subject>/<type>|<property>|<value> #<anything>" → one claim."""

    def extract(self, signal_text: str, known_entities=None) -> ExtractionResult:
        head = signal_text.split("#")[0].strip()
        subject, prop, value = (p.strip() for p in head.split("|"))
        name, etype = subject.split("/")
        return ExtractionResult(claims=[ExtractedClaim(
            entity_name=name, entity_type=etype, property=prop, value=value,
            confidence=0.9)])

    def extract_batch(self, texts, known_entities=None):
        return [self.extract(t) for t in texts]


class Res(ConflictResolver):
    def resolve(self, entity_name, prop, claims) -> ConflictResolution:
        return ConflictResolution(current_value=claims[0].value, status="settled",
                                  narrative="n", confidence=0.9)


def _svc() -> MemoryService:
    _config.settings.data_dir = tempfile.mkdtemp(prefix="etchmem-gateway-")
    _config.settings.extract_min_batch = 1
    svc = MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())
    app_main._service = svc
    return svc


def _deposit(svc, text, scope="sales", **kw):
    return svc.remember(data=text, source=kw.pop("source", "crm"), scope=scope,
                        extract_mode="immediate", **kw)[0]


# ── API key and /v1 ──────────────────────────────────────────────────────────

def test_api_key_guards_rest_and_mcp_but_not_health():
    _svc()
    client = TestClient(app_main.app)
    _config.settings.api_key = "k-old, k-new"
    try:
        assert client.get("/v1/stats").status_code == 401
        assert client.get("/stats").status_code == 401
        assert client.get("/v1/stats", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.get("/v1/stats", headers={"Authorization": "Bearer k-new"}).status_code == 200
        assert client.get("/v1/stats", headers={"X-API-Key": "k-old"}).status_code == 200
        assert client.post("/v1/mcp/", json={}).status_code == 401
        health = client.get("/v1/health")
        assert health.status_code == 200 and health.json()["auth"] is True
        assert client.get("/health").status_code == 200
        assert client.get("/ui").status_code == 200       # the page; its API calls need the key
    finally:
        _config.settings.api_key = ""
    assert client.get("/v1/stats").status_code == 200


def test_v1_and_legacy_paths_answer_the_same():
    svc = _svc()
    _deposit(svc, "Acme/company|tier|gold #1")
    svc.sleep()
    client = TestClient(app_main.app)
    v1 = client.get("/v1/etches").json()
    legacy = client.get("/etches").json()
    assert v1 == legacy and v1["total"] == 1
    assert client.get("/v1/health").json()["database"] in ("duckdb", "postgres")


# ── deposit metadata ─────────────────────────────────────────────────────────

def test_metadata_is_recorded_and_beliefs_carry_no_labels():
    svc = _svc()
    sid = _deposit(svc, "Acme/company|contact_email|cfo@acme.example #1",
                   source_record_id="zendesk:ticket:7",
                   actor={"agent": "support-bot", "user": "anna@acme.com"})
    _deposit(svc, "Acme/company|contact_email|cfo@acme.example #2", source="mail")
    svc.sleep()

    rec = svc.signal(sid)
    assert rec.source_record_id == "zendesk:ticket:7"
    assert rec.actor == {"agent": "support-bot", "user": "anna@acme.com"}
    assert "taints" not in rec.model_dump()

    client = TestClient(app_main.app)
    listed = client.get("/v1/etches").json()["etches"][0]
    assert "taints" not in listed and listed["source"] == "crm, mail"
    # a client still sending the old field is accepted, and the field is ignored
    r = client.post("/v1/remember", json={"data": "Acme/company|tier|gold", "source": "x",
                                          "scope": "sales", "taints": ["pii"]})
    assert r.status_code == 202


# ── several scopes ───────────────────────────────────────────────────────────

def test_reads_accept_several_scopes():
    svc = _svc()
    _deposit(svc, "Acme/company|tier|gold #s", scope="sales")
    _deposit(svc, "Acme/company|open_tickets|3 #p", scope="support")
    _deposit(svc, "Acme/company|salary_band|E5 #h", scope="hr")
    svc.sleep()

    def scopes(results):
        return {r.scope for r in results}

    both = svc.recall("Acme", scope=["sales", "support"], top_k=10, include_signals=False)
    assert scopes(both) == {"sales", "support"}
    assert scopes(svc.recall("Acme", scope="sales,support", top_k=10,
                             include_signals=False)) == {"sales", "support"}
    assert svc.recall("Acme", scope=[], top_k=10, include_signals=False) == []

    assert {e.scope for e in svc.etches(scope=["hr", "sales"]).etches} == {"hr", "sales"}
    client = TestClient(app_main.app)
    rest = client.get("/v1/etches", params=[("scope", "support"), ("scope", "sales")]).json()
    assert {e["scope"] for e in rest["etches"]} == {"sales", "support"}

    # Subjects are resolved per scope: "Acme" is one entity in each scope, so
    # no belief folds evidence from two scopes, and `know` by name covers the
    # scopes asked for.
    ids = {r.entity_id for r in svc.recall("Acme", top_k=10, include_signals=False)}
    assert len(ids) == 3
    assert {e.scope for e in svc.know("Acme").etches} == {"hr", "sales", "support"}
    assert {e.scope for e in svc.know("Acme", scope=["sales", "support"]).etches} == {
        "sales", "support"}
    assert svc.know("Acme", scope="finance") is None


def test_the_same_fact_in_two_scopes_stays_two_beliefs():
    svc = _svc()
    _deposit(svc, "Acme/company|tier|gold #s", scope="sales")
    _deposit(svc, "Acme/company|tier|platinum #h", scope="hr")
    svc.sleep()
    beliefs = svc.etches().etches
    assert {(e.scope, e.current_value) for e in beliefs} == {("sales", "gold"), ("hr", "platinum")}
    assert all(e.status == "settled" for e in beliefs)    # never contested across scopes


# ── forget ───────────────────────────────────────────────────────────────────

def test_forget_a_source_record_refolds_what_remains_and_blanks_history():
    svc = _svc()
    erased = _deposit(svc, "Acme/company|tier|gold #from-ticket",
                      source="zendesk", source_record_id="zendesk:ticket:7")
    svc.sleep()
    _deposit(svc, "Acme/company|tier|gold #from-crm", source="crm")
    svc.sleep()
    etch_id = svc.know("Acme").etches[0].id
    assert len(svc.history(etch_id)) >= 1

    report = svc.forget(source_record_ids=["zendesk:ticket:7"], reason="Art. 17 #1")
    assert report.signals_deleted == 1 and report.claims_updated == 1
    assert report.claims_deleted == 0 and report.etches_deleted == 0
    assert report.etches_refolded == 1 and report.refold == "done"
    assert report.versions_redacted >= 1

    assert svc.signal(erased) is None
    head = svc.know("Acme").etches[0]
    assert head.current_value == "gold"                  # still held, by the CRM signal
    claim = svc.dossier(etch_id).claims[0]
    assert erased not in claim.evidence_signal_ids and claim.sources == ["crm"]
    versions = svc.history(etch_id)
    assert versions[-1]["current_value"] == "gold"       # new version after the erasure
    assert any(v["current_value"] == ERASED for v in versions[:-1])

    # The same signal cannot come back, and the tombstone holds no text.
    again = svc.remember(data="Acme/company|tier|gold #from-ticket", source="zendesk",
                         scope="sales")
    assert again[1] is False and svc.was_erased(again[0])
    tomb = svc.stores.right.erasures()
    assert len(tomb) == 1 and tomb[0]["reason"] == "Art. 17 #1"
    assert "gold" not in str(tomb)


def test_forget_a_subject_removes_it_everywhere():
    svc = _svc()
    _deposit(svc, "Maria Muster/person|role|CTO #1")
    _deposit(svc, "Maria Muster/person|phone|+49 30 1234567 #2")
    mention = _deposit(svc, "Acme/company|tier|gold # call with Maria Muster")
    unrelated = _deposit(svc, "Globex/company|tier|silver #x")
    svc.sleep()
    pending = svc.remember(data="Maria Muster said she prefers e-mail",
                           source="crm", scope="sales")[0]          # not extracted yet
    assert svc.know("Maria Muster") is not None

    client = TestClient(app_main.app)
    resp = client.post("/v1/forget", json={"entity": "Maria Muster", "reason": "DSR-9"})
    assert resp.status_code == 200
    report = resp.json()
    assert report["entity_found"] is True
    # role, phone, and the Acme belief whose only evidence named her
    assert report["etches_deleted"] == 3 and report["versions_deleted"] >= 3
    assert report["signals_deleted"] >= 4

    assert svc.know("Maria Muster") is None
    for sid in (mention, pending):
        assert svc.signal(sid) is None                   # raw text named her
    assert svc.signal(unrelated) is not None
    assert {e.entity_name for e in svc.etches().etches} == {"Globex"}
    assert "Maria" not in str(client.get("/v1/erasures").json())
    assert client.get("/v1/erasures").json()["erasures"][0]["reason"] == "DSR-9"


def test_forget_needs_a_selector():
    _svc()
    client = TestClient(app_main.app)
    assert client.post("/v1/forget", json={}).status_code == 422


def test_a_store_built_for_another_embedding_size_refuses_to_start():
    import pytest

    from app.stores import Stores

    folder = tempfile.mkdtemp(prefix="etchmem-dim-")
    url, schema = _config.settings.database_url or None, _config.settings.database_schema or None
    Stores(folder, 8, database_url=url, schema=schema)
    with pytest.raises(RuntimeError, match="8-dimension embeddings"):
        Stores(folder, 16, database_url=url, schema=schema)

