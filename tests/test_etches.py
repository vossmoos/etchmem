"""
Tests for the filtered belief list (`etches`): exhaustive within its filters,
scoped, paged, and the same answer over REST and MCP. No network, no LLM key.
"""
from __future__ import annotations

import os
import tempfile

os.environ["EMBEDDING_PROVIDER"] = "fake"

from fastapi.testclient import TestClient              # noqa: E402

from app import config as _config                      # noqa: E402
from app import main as app_main                       # noqa: E402
from app import mcp_server                             # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolver, ConflictResolution, ExtractedClaim,
    ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.service import MemoryService                  # noqa: E402

# signal text -> (entity, entity_type, property, value)
FACTS = {
    "rm-1": ("Cast Radar", "product", "incorrect_assumption", "MCP server is free"),
    "rm-2": ("Kroger", "company", "incorrect_assumption", "Gemini CX inference on K8s"),
    "rm-3": ("COHORT:IC/UTILITIES/NA", "cohort", "avoid_pattern", "read-only agent claim"),
    "rm-4": ("Kroger", "company", "worked_pattern", "short question"),
    "demo-1": ("Acme", "company", "incorrect_assumption", "runs on GKE"),
}


class Scripted(ClaimExtractor):
    def extract(self, signal_text: str, known_entities=None) -> ExtractionResult:
        name, etype, prop, value = FACTS[signal_text.strip()]
        return ExtractionResult(claims=[ExtractedClaim(
            entity_name=name, entity_type=etype, property=prop, value=value,
            confidence=0.9)])


class Res(ConflictResolver):
    def resolve(self, entity_name, prop, claims) -> ConflictResolution:
        return ConflictResolution(current_value=claims[0].value, status="settled",
                                  narrative="n", confidence=0.9)


def _svc() -> MemoryService:
    _config.settings.data_dir = tempfile.mkdtemp(prefix="etchmem-etches-")
    _config.settings.extract_min_batch = 1
    svc = MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())
    for text in FACTS:
        scope = "demo" if text.startswith("demo") else "rm"
        svc.remember(data=text, source="test", scope=scope, extract_mode="immediate")
    svc.sleep()
    app_main._service = svc
    return svc


def _values(resp) -> set[str]:
    return {e.current_value for e in resp.etches}


def test_filters_by_scope_and_property_across_subjects():
    svc = _svc()
    out = svc.etches(scope="rm", property="incorrect_assumption")
    assert _values(out) == {"MCP server is free", "Gemini CX inference on K8s"}
    assert out.total == 2 and out.count == 2
    assert {e.scope for e in out.etches} == {"rm"}


def test_property_list_and_comma_string_are_the_same_filter():
    svc = _svc()
    want = {"MCP server is free", "Gemini CX inference on K8s", "read-only agent claim"}
    assert _values(svc.etches(scope="rm",
                              property=["incorrect_assumption", "avoid_pattern"])) == want
    assert _values(svc.etches(scope="rm",
                              property="incorrect_assumption, avoid_pattern")) == want


def test_entity_type_filter_and_type_is_returned():
    svc = _svc()
    out = svc.etches(scope="rm", entity_type="company")
    assert _values(out) == {"Gemini CX inference on K8s", "short question"}
    assert {e.entity_type for e in out.etches} == {"company"}


def test_no_filters_returns_everything():
    assert _svc().etches().total == len(FACTS)


def test_paging_covers_every_row_once():
    svc = _svc()
    seen, offset = [], 0
    while True:
        page = svc.etches(limit=2, offset=offset)
        seen += [e.id for e in page.etches]
        offset += page.count
        if offset >= page.total:
            break
    assert len(seen) == len(set(seen)) == len(FACTS)


def test_limit_is_clamped():
    out = _svc().etches(limit=0)
    assert out.limit == 1 and out.count == 1


def test_min_confidence_and_status():
    svc = _svc()
    assert svc.etches(scope="rm", min_confidence=0.95).total == 0
    assert svc.etches(scope="rm", status="settled").total == 4
    assert svc.etches(scope="rm", status="contested").total == 0


def test_mcp_tool_and_rest_endpoint_agree():
    _svc()
    tool = mcp_server.etches(scope="rm", property=["incorrect_assumption"])
    rest = TestClient(app_main.app).get(
        "/etches", params={"scope": "rm", "property": "incorrect_assumption"}).json()
    assert tool["total"] == rest["total"] == 2
    assert {e["id"] for e in tool["etches"]} == {e["id"] for e in rest["etches"]}
    rest_multi = TestClient(app_main.app).get(
        "/etches", params=[("scope", "rm"), ("property", "incorrect_assumption"),
                           ("property", "avoid_pattern")]).json()
    assert rest_multi["total"] == 3


def test_writes_nothing_to_disk():
    svc = _svc()
    svc.etches(scope="rm")
    assert not os.path.exists(os.path.join(_config.settings.data_dir, "export"))
