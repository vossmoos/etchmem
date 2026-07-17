"""
Tests for the MCP interface: tools registered, mounted at /mcp, and the tool
functions drive the same service layer as REST. No network, no LLM key.
"""
from __future__ import annotations

import os
import tempfile

os.environ["EMBEDDING_PROVIDER"] = "fake"

from app import config as _config                      # noqa: E402
from app import main as app_main                       # noqa: E402
from app import mcp_server                             # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolver, ConflictResolution, ExtractedClaim,
    ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.service import MemoryService                  # noqa: E402

EXPECTED_TOOLS = {"remember", "recall", "sleep", "export", "stats", "etch_history"}


class StubExtractor(ClaimExtractor):
    def extract(self, signal_text: str, known_entities=None) -> ExtractionResult:
        if "acme" not in signal_text.lower():
            return ExtractionResult(claims=[])
        return ExtractionResult(claims=[ExtractedClaim(
            entity_name="Acme Corp", entity_type="company",
            property="contract_status", value="signed", confidence=0.8)])


class StubResolver(ConflictResolver):
    def resolve(self, entity_name, prop, claims) -> ConflictResolution:
        return ConflictResolution(current_value=claims[0].value, status="contested",
                                  narrative="disputed", confidence=0.4)


def _install_service() -> MemoryService:
    _config.settings.data_dir = tempfile.mkdtemp(prefix="etchmem-mcp-")
    svc = MemoryService(embedder=FakeEmbedding(), extractor=StubExtractor(),
                        resolver=StubResolver())
    app_main._service = svc
    return svc


# ── tests ────────────────────────────────────────────────────────────────────

def test_all_tools_registered():
    names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
    assert EXPECTED_TOOLS <= names


def test_mcp_mounted_on_fastapi_app():
    assert any(getattr(r, "path", None) == "/mcp" for r in app_main.app.routes)


def test_tools_share_service_with_rest():
    _install_service()

    out = mcp_server.remember(data="Acme Corp signed the contract",
                              source="agent-1", scope="sales")
    assert out["stored"] is True and out["status"] == "new"

    tick = mcp_server.sleep()
    assert tick["etches_formed"] == 1

    st = mcp_server.stats()
    assert st["etches"] == 1 and st["claims"] == 1

    hits = mcp_server.recall(query="acme contract", top_k=5)
    assert any(h["origin"] == "etch" and h["value"] == "signed" for h in hits)

    etch_id = next(h["id"] for h in hits if h["origin"] == "etch")
    hist = mcp_server.etch_history(etch_id)
    assert [v["version"] for v in hist["versions"]] == [1]

    exp = mcp_server.export()
    assert exp["count"] == 1 and os.path.isdir(exp["export_dir"])
