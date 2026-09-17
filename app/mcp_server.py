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
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Deposit a raw signal (call note, tool output, email, decision...).
    `source` = who produced it (e.g. 'agent-33'); `scope` = domain tag
    (e.g. 'sales'). 'immediate' extract_mode skips batching for urgent facts.
    `occurred_at` = when the fact actually happened (ISO-8601 or epoch
    seconds), defaulting to now — always set it for historical records, or
    they all arrive dated today and cannot be ordered against each other."""
    sig_id, stored, parsed = _svc().remember(
        data=data, source=source, scope=scope,
        extract_mode=extract_mode, metadata=metadata, occurred_at=occurred_at)
    return {"id": sig_id, "stored": stored,
            "status": "new" if stored else "duplicate",
            "occurred_at": parsed, "occurred_at_declared": bool(parsed)}


@mcp.tool()
def recall(
    query: str,
    scope: str | None = None,
    source: str | None = None,
    top_k: int = 5,
    include_signals: bool = True,
    as_of: str | None = None,
    as_of_basis: Literal["ingest", "event"] = "ingest",
) -> list[dict[str, Any]]:
    """Semantic recall over consolidated beliefs (etches), optionally blended
    with fresh raw signals. Pass an ISO-8601 `as_of` for time-travel: what did
    the system believe at that moment?"""
    results = _svc().recall(
        query=query, scope=scope, source=source, top_k=top_k,
        include_signals=include_signals, as_of=as_of, as_of_basis=as_of_basis)
    return [r.model_dump() for r in results]


@mcp.tool()
def associate(
    query: str,
    scope: str | None = None,
    top_k: int = 5,
    hops: int = 1,
    as_of: str | None = None,
    min_score: float = 0.15,
    as_of_basis: Literal["ingest", "event"] = "ingest",
) -> dict[str, Any]:
    """What the memory holds for a phrase — beliefs, their subjects, and the
    connected nodes.

    Use when you do NOT already know which entity you are asking about.
    `recall` ranks beliefs by wording; this also returns every fact about each
    matched subject and follows declared relations one hop, so a fault reached
    by wording leads to the part that fixes it, which wording alone never
    would."""
    return _svc().associate(query=query, scope=scope, top_k=top_k, hops=hops,
                            as_of=as_of, min_score=min_score,
                            as_of_basis=as_of_basis).model_dump()


@mcp.tool()
def know(
    ref: str,
    as_of: str | None = None,
    entity_type: str | None = None,
    as_of_basis: Literal["ingest", "event"] = "ingest",
) -> dict[str, Any]:
    """EVERY belief about one subject — exhaustive, unlike `recall`.

    recall retrieves by cue and can miss; `know` is direct access to everything
    held about a subject.

    Use this when the question is "what do we know about X". `recall` ranks by
    embedding distance and returns top_k, so a fact can be missing without the
    caller being able to tell. `ref` is an entity id ('product_msm_0808') or a
    surface name ('MSM-0808'). `as_of` (ISO-8601) gives the beliefs as they
    stood then."""
    facts = _svc().know(ref, as_of=as_of, entity_type=entity_type,
                        as_of_basis=as_of_basis)
    if facts is None:
        return {"error": f"no entity matching {ref!r}", "etches": []}
    return facts.model_dump()


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
