"""
MCP interface — the same operations as the REST API, exposed as MCP *tools*.

Mounted into the FastAPI app at /mcp (streamable HTTP, stateless, JSON
responses), so one process serves both REST and MCP. Point any MCP client at:

    http://<host>:<port>/mcp

Tools:
  remember      — deposit a raw signal
  recall        — semantic recall over etches (+ time-travel via as_of)
  sleep         — run one worker tick now (batch → extract → fold)
  export        — dump all etches to JSON files
  stats         — queue depths + counts
  etch_history  — version timeline of one etch
"""
from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "etchmem",
    instructions=(
        "etchmem is a knowledge consolidation engine: deposit raw experience "
        "with `remember`, let the pipeline fold it into consolidated, versioned "
        "beliefs (etches), and query them with `recall`. Use `sleep` to force "
        "a consolidation tick, `stats` for queue depths, `etch_history` for "
        "the version timeline of one belief."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",       # final path is where main.py mounts it: /mcp
)


def _svc():
    from app.main import get_service  # lazy: avoids a circular import

    return get_service()


@mcp.tool()
def remember(
    data: str,
    source: str,
    scope: str,
    extract_mode: Literal["immediate", "deferred"] = "deferred",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deposit a raw signal (call note, tool output, email, decision...).
    `source` = who produced it (e.g. 'agent-33'); `scope` = domain tag
    (e.g. 'sales'). 'immediate' extract_mode skips batching for urgent facts."""
    sig_id, stored = _svc().remember(
        data=data, source=source, scope=scope,
        extract_mode=extract_mode, metadata=metadata)
    return {"id": sig_id, "stored": stored,
            "status": "new" if stored else "duplicate"}


@mcp.tool()
def recall(
    query: str,
    scope: str | None = None,
    source: str | None = None,
    top_k: int = 5,
    include_signals: bool = True,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic recall over consolidated beliefs (etches), optionally blended
    with fresh raw signals. Pass an ISO-8601 `as_of` for time-travel: what did
    the system believe at that moment?"""
    results = _svc().recall(
        query=query, scope=scope, source=source, top_k=top_k,
        include_signals=include_signals, as_of=as_of)
    return [r.model_dump() for r in results]


@mcp.tool()
def sleep() -> dict[str, int]:
    """Run one full pipeline tick now: batch (dedup) → extract (claims) →
    fold (etches). Returns counts of work done."""
    return _svc().sleep()


@mcp.tool()
def export() -> dict[str, Any]:
    """Export all consolidated etches to JSON files; returns the export
    directory and the etches themselves."""
    export_dir, etches = _svc().export()
    return {"export_dir": export_dir, "count": len(etches),
            "etches": [e.model_dump() for e in etches]}


@mcp.tool()
def stats() -> dict[str, Any]:
    """Queue depths and counts: signals by status, claims, entities, etches,
    contested beliefs, known scopes."""
    return _svc().stats()


@mcp.tool()
def etch_history(etch_id: str) -> dict[str, Any]:
    """Version timeline of one etch (id format: '<entity_id>::<property>') —
    every belief change with narrative, confidence and triggering claims."""
    return {"etch_id": etch_id, "versions": _svc().history(etch_id)}
