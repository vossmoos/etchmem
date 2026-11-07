"""
FastAPI application for etchmem-server.

The API lives under /v1 (the unprefixed paths below still answer, as before):

  POST /v1/remember          — deposit a raw signal (source + scope; extract_mode)
  POST /v1/recall            — semantic recall over etches (+ time-travel via as_of)
  POST /v1/associate         — beliefs for a phrase, their subjects, connected nodes
  GET  /v1/entity/{ref}/etches — every belief about one subject
  GET  /v1/etches            — every etch matching filters (scope, property, entity_type...)
  POST /v1/sleep             — run one worker tick now (batch → extract → fold)
  POST /v1/export            — dump all etches to JSON files
  POST /v1/forget            — erase a subject, source records or signals; returns a report
  GET  /v1/erasures          — erasure tombstones (no personal data)
  GET  /v1/stats             — queue depths + counts
  GET  /v1/health            — liveness + active config (never needs a key)
  GET  /v1/etch/{id}/history — version timeline of one etch
  GET  /v1/etch/{id}/dossier — full provenance: etch + versions + claims + signals
  GET  /ui                   — Etchmem Beliefs Explorer (single-page report UI)

An MCP interface exposing the same reads and `remember` as tools is mounted at
/v1/mcp and /mcp (streamable HTTP). Erasure is REST-only on purpose: an agent
should not be able to delete memory by mistake.

When ETCHMEM_API_KEY is set, every call except /health (and the /ui page) needs
`Authorization: Bearer <key>` or `X-API-Key: <key>`, on REST and MCP alike.

A single in-process WorkerLoop runs the pipeline on a cadence (no broker).
"""
from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query

from app import __version__
from app.config import settings
from app.schemas import (
    AssociateRequest, AssociateResponse, ContextResponse, DossierResponse,
    EtchListResponse, ExportResponse, ForgetRequest, ForgetResponse, HealthResponse,
    HistoryResponse, KnowledgeResponse, RecallRequest, RecallResponse, RememberRequest,
    RememberResponse, SignalRecordResponse, SleepResponse, StatsResponse, VersionOut,
)
from app.mcp_server import mcp as mcp_server
from app.service import MemoryService
from app.worker import WorkerLoop

logging.basicConfig(level=logging.INFO)

_service: MemoryService | None = None
_worker: WorkerLoop | None = None

API_PREFIX = "/v1"
# Probes, and the Explorer page itself (it holds no data; its API calls carry the key).
_OPEN_PATHS = {"/health", f"{API_PREFIX}/health", "/ui"}


def get_service() -> MemoryService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service, _worker
    _service = MemoryService()
    if settings.worker_enabled:
        _worker = WorkerLoop(_service.get_pipeline())
        _worker.start()
    async with mcp_server.session_manager.run():
        yield
    if _worker:
        await _worker.stop()


class ApiKeyGuard:
    """Pure ASGI middleware: when keys are configured, every HTTP request but
    /health must carry one. Covers the mounted MCP app too. Read per request,
    so changing ETCHMEM_API_KEY in tests or at runtime takes effect at once."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        keys = settings.api_keys
        if scope["type"] != "http" or not keys or scope.get("path") in _OPEN_PATHS:
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        given = headers.get("x-api-key", "")
        auth = headers.get("authorization", "")
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        if given and any(hmac.compare_digest(given, k) for k in keys):
            return await self.app(scope, receive, send)
        body = b'{"detail":"missing or invalid API key"}'
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})


app = FastAPI(
    title="etchmem-server",
    version=__version__,
    summary="Enterprise memory OS: raw signals → claims → consolidated etches.",
    lifespan=lifespan,
)
app.add_middleware(ApiKeyGuard)

router = APIRouter()


@router.post("/remember", response_model=RememberResponse, status_code=202)
def remember(req: RememberRequest) -> RememberResponse:
    svc = get_service()
    sig_id, stored, occurred_at = svc.remember(
        data=req.data, source=req.source, scope=req.scope,
        extract_mode=req.extract_mode, metadata=req.metadata,
        occurred_at=req.occurred_at, context=req.context, context_parent=req.context_parent,
        source_record_id=req.source_record_id, actor=req.actor)
    if stored:
        status, message = "new", "Accepted."
    elif svc.was_erased(sig_id):
        status, message = "erased", "Refused: this signal was erased before."
    else:
        status, message = "duplicate", "Already present (idempotent)."
    return RememberResponse(
        id=sig_id, stored=stored, status=status, message=message,
        occurred_at=occurred_at, occurred_at_declared=bool(occurred_at))


@router.post("/recall", response_model=RecallResponse)
def recall(req: RecallRequest) -> RecallResponse:
    results = get_service().recall(
        query=req.query, scope=req.scope, source=req.source, top_k=req.top_k,
        include_signals=req.include_signals, as_of=req.as_of,
        as_of_basis=req.as_of_basis, quotes=req.quotes, writers=req.writers)
    return RecallResponse(query=req.query, as_of=req.as_of, results=results)


@router.post("/associate", response_model=AssociateResponse)
def associate(req: AssociateRequest) -> AssociateResponse:
    """What the memory holds for a phrase: matching beliefs, their subjects'
    full facts, and the nodes those subjects are connected to."""
    return get_service().associate(
        query=req.query, scope=req.scope, top_k=req.top_k, hops=req.hops,
        as_of=req.as_of, min_score=req.min_score, as_of_basis=req.as_of_basis)


@router.get("/entity/{ref}/etches", response_model=KnowledgeResponse)
def know(ref: str, as_of: str | None = None,
         entity_type: str | None = None,
         as_of_basis: str = "ingest",
         scope: list[str] | None = Query(default=None),
         quotes: Literal["ref", "full"] = "full",
         writers: list[str] | None = Query(default=None)) -> KnowledgeResponse:
    """Every belief about one subject — exhaustive, not similarity-ranked.

    `ref` is an entity id (`product_msm_0808`) or a surface name (`MSM-0808`,
    `MSM 0808`), resolved the same way ingestion resolved it. `as_of` gives the
    beliefs as they stood at that time. `scope` repeats or takes a
    comma-separated list: only beliefs in those scopes. Every belief's quote
    carries the source excerpt (default `quotes=full`); `quotes=ref` keeps it only
    where a belief has no wording.
    """
    facts = get_service().know(ref, as_of=as_of, entity_type=entity_type,
                               as_of_basis=as_of_basis, scope=scope, quotes=quotes,
                               writers=writers)
    if facts is None:
        raise HTTPException(status_code=404, detail=f"no entity matching {ref!r}")
    return facts


@router.get("/etches", response_model=EtchListResponse)
def etches(scope: list[str] | None = Query(default=None),
           property: list[str] | None = Query(default=None),
           entity_type: list[str] | None = Query(default=None),
           status: str | None = None,
           min_confidence: float | None = None,
           limit: int = 500, offset: int = 0,
           writers: list[str] | None = Query(default=None)) -> EtchListResponse:
    """Every belief matching the filters — exhaustive, not similarity-ranked.

    `scope`, `property` and `entity_type` repeat (`?property=a&property=b`) or
    take a comma-separated list. Newest first; page with `offset` until
    `offset + count >= total`.
    """
    return get_service().etches(scope=scope, property=property, entity_type=entity_type,
                                status=status, min_confidence=min_confidence,
                                limit=limit, offset=offset, writers=writers)


@router.post("/sleep", response_model=SleepResponse)
def sleep() -> SleepResponse:
    try:
        return SleepResponse(**get_service().sleep())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"sleep failed: {exc}") from exc


@router.post("/export", response_model=ExportResponse)
def export() -> ExportResponse:
    export_dir, etches = get_service().export()
    return ExportResponse(export_dir=export_dir, count=len(etches), etches=etches)


@router.post("/forget", response_model=ForgetResponse)
def forget(req: ForgetRequest) -> ForgetResponse:
    """Erase a subject, source records or signals from every layer (GDPR Art. 17).

    Returns the erasure report, which is also kept as a tombstone without
    personal data. Beliefs that lost only part of their evidence are blanked
    at once and re-consolidated from what remains."""
    if not (req.entity or req.source_record_ids or req.signal_ids):
        raise HTTPException(status_code=422,
                            detail="give entity, source_record_ids or signal_ids")
    return get_service().forget(
        entity=req.entity, entity_type=req.entity_type,
        source_record_ids=req.source_record_ids, signal_ids=req.signal_ids,
        reason=req.reason)


@router.get("/erasures")
def erasures() -> dict:
    """Every erasure performed: id, time, reason and counts. No personal data."""
    return {"erasures": get_service().stores.right.erasures()}


@router.get("/etch/{etch_id:path}/history", response_model=HistoryResponse)
def history(etch_id: str, writers: list[str] | None = Query(default=None)) -> HistoryResponse:
    versions = get_service().history(etch_id, writers=writers)
    if versions is None:
        raise HTTPException(status_code=404, detail=f"Unknown etch: {etch_id}")
    return HistoryResponse(etch_id=etch_id, versions=[VersionOut(**v) for v in versions])


@router.get("/etch/{etch_id:path}/dossier", response_model=DossierResponse)
def dossier(etch_id: str, writers: list[str] | None = Query(default=None)) -> DossierResponse:
    d = get_service().dossier(etch_id, writers=writers)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Unknown etch: {etch_id}")
    return d


@router.get("/signal/{signal_id}", response_model=SignalRecordResponse)
def signal(signal_id: str) -> SignalRecordResponse:
    s = get_service().signal(signal_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Unknown signal: {signal_id}")
    return s


@router.get("/context/{context_id:path}", response_model=ContextResponse)
def context(context_id: str) -> ContextResponse:
    return get_service().context(context_id)


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**get_service().stats())


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = get_service()
    return HealthResponse(
        status="ok", version=__version__,
        embedding_provider=svc.embedder.name, embedding_dim=svc.embedder.dim,
        claim_model=settings.claim_model, etch_model=settings.etch_model,
        worker_enabled=settings.worker_enabled,
        claims_anonymization=settings.claims_anonymization,
        database=svc.stores.engine, auth=bool(settings.api_keys))


# The versioned API, and the original unprefixed paths for existing clients.
app.include_router(router, prefix=API_PREFIX)
app.include_router(router, include_in_schema=False)


# ── Etchmem Beliefs Explorer (single static page; print-to-PDF for export) ──
if settings.ui_enabled:
    from pathlib import Path

    from fastapi.responses import FileResponse

    _UI_PAGE = Path(__file__).parent / "static" / "explorer.html"

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(_UI_PAGE, media_type="text/html")


# MCP interface (tools over streamable HTTP) — one process, both protocols.
_mcp_app = mcp_server.streamable_http_app()
app.mount(f"{API_PREFIX}/mcp", _mcp_app)
app.mount("/mcp", _mcp_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
