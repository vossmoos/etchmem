"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── remember ───────────────────────────────────────────────────────────────

class RememberRequest(BaseModel):
    data: str = Field(..., description="Raw text signal to store.")
    source: str = Field(..., description="Who produced this signal, e.g. 'agent-33'.")
    scope: str = Field(..., description="Tag / domain of the signal, e.g. 'sales'.")
    extract_mode: Literal["immediate", "deferred"] = Field(
        "deferred",
        description="'immediate' = extract on the next worker tick (urgent); "
                    "'deferred' = batch and extract later (cheap default).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RememberResponse(BaseModel):
    id: str
    stored: bool
    status: str
    message: str


# ── recall ─────────────────────────────────────────────────────────────────

class RecallRequest(BaseModel):
    query: str
    scope: str | None = None
    source: str | None = None
    top_k: int = Field(5, ge=1, le=50)
    include_signals: bool = Field(True, description="Blend fresh raw signals from the left DB.")
    as_of: str | None = Field(
        None,
        description="ISO-8601 time for time-travel recall. Returns the etch "
                    "version current as of that ingest time (HEAD candidates only).",
    )


class RecallResult(BaseModel):
    id: str
    content: str
    score: float
    origin: str                     # "etch" | "signal"
    entity_name: str | None = None
    property: str | None = None
    value: str | None = None
    status: str | None = None
    confidence: float | None = None
    version: int | None = None
    scope: str | None = None
    source: str | None = None
    created_at: float = 0.0
    updated_at: float | None = None


class RecallResponse(BaseModel):
    query: str
    as_of: str | None = None
    results: list[RecallResult]


# ── sleep (manual worker tick) ───────────────────────────────────────────────

class SleepResponse(BaseModel):
    batched: int
    extracted_signals: int
    claims_written: int
    pairs_folded: int
    etches_formed: int
    etches_updated: int
    contested: int


# ── export ───────────────────────────────────────────────────────────────────

class EtchOut(BaseModel):
    id: str
    entity_name: str
    property: str
    current_value: str
    status: str
    confidence: float
    narrative: str
    version: int
    scope: str | None = None
    source: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


class ExportResponse(BaseModel):
    export_dir: str
    count: int
    etches: list[EtchOut]


# ── history ───────────────────────────────────────────────────────────────────

class VersionOut(BaseModel):
    version: int
    current_value: str
    status: str
    confidence: float
    narrative: str
    triggered_by: list[str] = Field(default_factory=list)
    created_at: float


class HistoryResponse(BaseModel):
    etch_id: str
    versions: list[VersionOut]


# ── stats / health ───────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    signals_total: int
    signals_new: int
    signals_batched: int
    signals_extracted: int
    claims: int
    entities: int
    etches: int
    contested: int
    scopes: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    embedding_dim: int
    claim_model: str
    etch_model: str
    worker_enabled: bool
