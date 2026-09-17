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
    occurred_at: str | None = Field(
        None,
        description="When the fact actually happened — ISO-8601 (e.g. "
                    "'2014-03-17' or '2014-03-17T09:20:00Z') or epoch seconds. "
                    "Defaults to ingest time. Set it when depositing historical "
                    "records: without it every signal in a backfill carries "
                    "today's date and the recency policy cannot order them.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RememberResponse(BaseModel):
    id: str
    stored: bool
    status: str
    message: str
    occurred_at: float = Field(
        0.0, description="Event time as parsed and stored, epoch seconds. Echoed "
                         "so a bulk load can verify the date was understood.")
    occurred_at_declared: bool = Field(
        False, description="False means occurred_at was absent or unparseable "
                           "and ingest time was used.")


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
                    "version current as of that time (HEAD candidates only).",
    )
    as_of_basis: Literal["ingest", "event"] = Field(
        "ingest",
        description="Which clock `as_of` uses. 'ingest' (default) = what we "
                    "BELIEVED then. 'event' = what was TRUE then, by the dates "
                    "on the underlying facts. They differ whenever a source "
                    "reported late.")



class RecallResult(BaseModel):
    id: str
    content: str
    score: float
    origin: str                     # "etch" | "signal"
    entity_id: str | None = None    # the subject's slug — feed it to know()
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
    claims_dropped: int = 0
    subject_retries: int = 0
    relations_resolved: int = 0
    entities: dict[str, int] = Field(
        default_factory=dict,
        description="How subjects were identified: entities_by_key / by_alias / "
                    "by_fuzzy / created / ignored / ambiguous / unmatched. "
                    "by_fuzzy rising on a type you declared with an "
                    "identifier_pattern means the pattern is not matching.")


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
    value_entity_id: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


class ExportResponse(BaseModel):
    export_dir: str
    count: int
    etches: list[EtchOut]


# ── know ─────────────────────────────────────────────────────────────

class EntityOut(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    scope: str | None = None


class KnowledgeResponse(BaseModel):
    """Everything believed about one subject — exhaustive, not ranked."""
    entity: EntityOut
    etches: list[EtchOut] = Field(default_factory=list)
    count: int = 0
    as_of: str | None = None
    as_of_basis: str = "ingest"
    contested: int = 0


# ── associate ────────────────────────────────────────────────────────────────

class AssociateRequest(BaseModel):
    query: str
    scope: str | None = None
    top_k: int = Field(5, ge=1, le=50)
    hops: int = Field(1, ge=0, le=1, description="1 follows declared relations.")
    as_of_basis: Literal["ingest", "event"] = Field(
        "ingest",
        description="Which clock `as_of` uses. 'ingest' (default) = what we "
                    "BELIEVED then. 'event' = what was TRUE then, by the dates "
                    "on the underlying facts. They differ whenever a source "
                    "reported late.")

    min_score: float = Field(
        0.15, ge=0.0, le=1.0,
        description="Drop seeds below this similarity. recall returns top_k "
                    "whatever the score; a zero-scoring seed would drag its "
                    "whole neighbourhood into the answer.")
    as_of: str | None = None


class EdgeOut(BaseModel):
    """A relation between two entities, as one belief."""
    etch_id: str
    from_entity: str
    property: str
    to_entity: str
    to_name: str
    status: str
    confidence: float
    direction: str                    # "out" (from a seed) | "in" (toward a seed)


class AssociateNode(BaseModel):
    entity: EntityOut
    etches: list[EtchOut] = Field(default_factory=list)
    hops: int = 0                     # 0 = matched the query, 1 = reached by an edge
    score: float = 0.0                # best seed score that reached this node
    reached_via: str | None = None    # the property that led here


class AssociateResponse(BaseModel):
    """What the memory holds for a phrase: the beliefs that match, the subjects
    behind them, and the nodes those subjects are connected to."""
    query: str
    seeds: list[RecallResult] = Field(default_factory=list)
    nodes: list[AssociateNode] = Field(default_factory=list)
    edges: list[EdgeOut] = Field(default_factory=list)
    as_of: str | None = None
    contested: int = 0


# ── history ───────────────────────────────────────────────────────────────────

class VersionOut(BaseModel):
    version: int
    current_value: str
    status: str
    confidence: float
    narrative: str
    triggered_by: list[str] = Field(default_factory=list)
    created_at: float                  # when we learned it
    event_at: float = 0.0              # when the fact behind it became true


class HistoryResponse(BaseModel):
    etch_id: str
    versions: list[VersionOut]


# ── dossier ──────────────────────────────────────────────────────────────────

class ClaimOut(BaseModel):
    id: str
    entity_name: str
    property: str
    value: str
    polarity: str
    corroboration_count: int
    confidence: float
    sources: list[str] = Field(default_factory=list)
    evidence_signal_ids: list[str] = Field(default_factory=list)
    scope: str | None = None
    event_time: float = 0.0
    ingest_time: float = 0.0
    created_at: float = 0.0


class SignalOut(BaseModel):
    id: str
    content: str
    source: str
    scope: str
    created_at: float = 0.0
    occurred_at: float = 0.0


class DossierResponse(BaseModel):
    """Full provenance view of one belief: the etch, its version timeline,
    the claims that formed it, and the raw signals behind those claims."""
    etch: EtchOut
    versions: list[VersionOut]
    claims: list[ClaimOut]
    signals: list[SignalOut] = Field(default_factory=list)
    signals_omitted: bool = Field(
        False, description="True when claims_anonymization hides raw signals.")


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
    claims_anonymization: bool = False
