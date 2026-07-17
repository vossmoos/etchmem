"""
FastAPI application for etchmem-server.

Endpoints:
  POST /remember          — deposit a raw signal (source + scope; extract_mode)
  POST /recall            — semantic recall over etches (+ time-travel via as_of)
  POST /sleep             — run one worker tick now (batch → extract → fold)
  POST /export            — dump all etches to JSON files
  GET  /stats             — queue depths + counts
  GET  /health            — liveness + active config
  GET  /etch/{id}/history — version timeline of one etch

An MCP interface exposing the same operations as tools (remember, recall,
sleep, export, stats, etch_history) is mounted at /mcp (streamable HTTP).

A single in-process WorkerLoop runs the pipeline on a cadence (no broker).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import __version__
from app.config import settings
from app.schemas import (
    ExportResponse, HealthResponse, HistoryResponse, RecallRequest, RecallResponse,
    RememberRequest, RememberResponse, SleepResponse, StatsResponse, VersionOut,
)
from app.mcp_server import mcp as mcp_server
from app.service import MemoryService
from app.worker import WorkerLoop

logging.basicConfig(level=logging.INFO)

_service: MemoryService | None = None
_worker: WorkerLoop | None = None


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


app = FastAPI(
    title="etchmem-server",
    version=__version__,
    summary="Enterprise memory OS: raw signals → claims → consolidated etches.",
    lifespan=lifespan,
)


@app.post("/remember", response_model=RememberResponse, status_code=202)
def remember(req: RememberRequest) -> RememberResponse:
    sig_id, stored = get_service().remember(
        data=req.data, source=req.source, scope=req.scope,
        extract_mode=req.extract_mode, metadata=req.metadata)
    return RememberResponse(
        id=sig_id, stored=stored, status="new" if stored else "duplicate",
        message="Accepted." if stored else "Already present (idempotent).")


@app.post("/recall", response_model=RecallResponse)
def recall(req: RecallRequest) -> RecallResponse:
    results = get_service().recall(
        query=req.query, scope=req.scope, source=req.source, top_k=req.top_k,
        include_signals=req.include_signals, as_of=req.as_of)
    return RecallResponse(query=req.query, as_of=req.as_of, results=results)


@app.post("/sleep", response_model=SleepResponse)
def sleep() -> SleepResponse:
    try:
        return SleepResponse(**get_service().sleep())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"sleep failed: {exc}") from exc


@app.post("/export", response_model=ExportResponse)
def export() -> ExportResponse:
    export_dir, etches = get_service().export()
    return ExportResponse(export_dir=export_dir, count=len(etches), etches=etches)


@app.get("/etch/{etch_id:path}/history", response_model=HistoryResponse)
def history(etch_id: str) -> HistoryResponse:
    versions = get_service().history(etch_id)
    return HistoryResponse(etch_id=etch_id, versions=[VersionOut(**v) for v in versions])


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**get_service().stats())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = get_service()
    return HealthResponse(
        status="ok", version=__version__,
        embedding_provider=svc.embedder.name, embedding_dim=svc.embedder.dim,
        claim_model=settings.claim_model, etch_model=settings.etch_model,
        worker_enabled=settings.worker_enabled,
        claims_anonymization=settings.claims_anonymization)


# MCP interface (tools over streamable HTTP) — one process, both protocols.
app.mount("/mcp", mcp_server.streamable_http_app())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
