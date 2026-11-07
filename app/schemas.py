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
    occurred_at: str | float | None = Field(
        None,
        description="When the fact actually happened — ISO-8601 (e.g. "
                    "'2014-03-17' or '2014-03-17T09:20:00Z') or epoch seconds. "
                    "Defaults to ingest time. Set it when depositing historical "
                    "records: without it every signal in a backfill carries "
                    "today's date and the recency policy cannot order them.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    context: str | None = Field(
        None, description="The conversation it was said in, as you address it: a "
                          "Slack thread, a ticket's comments, a meeting. Read with its "
                          "earlier messages; same words in another context = another signal.")
    context_parent: str | None = Field(
        None, description="The conversation around it, e.g. the channel of a thread.")
    source_record_id: str | None = Field(
        None, description="The record this came from in its source system, e.g. "
                          "'zendesk:ticket:123'. Lets an erasure there be followed here "
                          "with `forget(source_record_ids=[...])`.")
    actor: dict[str, Any] = Field(
        default_factory=dict,
        description="Who deposited it, e.g. {\"agent\": \"support-bot\", \"user\": "
                    "\"anna@acme.com\"}. Recorded for provenance; never interpreted.")


class RememberResponse(BaseModel):
    id: str
    stored: bool
    status: str = Field(..., description="new | duplicate | erased (this exact "
                                       "signal was erased before and is refused)")
    message: str
    occurred_at: float = Field(
        0.0, description="Event time as parsed and stored, epoch seconds. Echoed "
                         "so a bulk load can verify the date was understood.")
    occurred_at_declared: bool = Field(
        False, description="False means occurred_at was absent or unparseable "
                           "and ingest time was used.")


# ── recall ─────────────────────────────────────────────────────────────────

Scopes = str | list[str] | None
_SCOPE_HELP = ("One scope, or a list: any of them. Omit for every scope. An empty "
               "list matches nothing.")


_QUOTES_HELP = ("Each belief's `quote`: 'full' (default) = author, source record, time "
                "and the original excerpt; 'ref' = the excerpt only for beliefs without "
                "a wording, for shorter replies.")


_WRITERS_HELP = ("Only beliefs ALL of whose writers (deposit sources) are in this list; "
                 "omit for no filter. A gateway sets it from the caller's permissions.")


class RecallRequest(BaseModel):
    query: str
    scope: Scopes = Field(None, description=_SCOPE_HELP)
    writers: list[str] | None = Field(None, description=_WRITERS_HELP)
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
    quotes: Literal["ref", "full"] = Field("full", description=_QUOTES_HELP)



class DetailItem(BaseModel):
    """One labelled fact: `value` is the short label, `detail` the author's full
    wording (every property by default; see ETCHMEM_DETAIL_DEFAULT)."""
    value: str
    detail: str


class Quote(BaseModel):
    """The newest signal behind a belief's current value: what was actually
    written, by whom, where and when. The dossier has every signal in full."""
    signal_id: str
    excerpt: str | None = Field(None, description="The signal's text around the fact (with "
                                "quotes='ref' only for beliefs without a wording); "
                                "never when claims anonymization is on.")
    source: str
    source_record_id: str | None = None
    actor: dict[str, Any] = Field(default_factory=dict,
                                  description="The depositing agent and the person it acted for.")
    occurred_at: float = 0.0


class ValueWeight(BaseModel):
    """One value ever asserted for a belief. A retracted value is kept, with
    `active=False` and a much lower weight: no longer valid, but once it was."""
    value: str
    weight: float = 0.0                 # share of the belief's total weight, 0..1
    active: bool = True
    last_asserted_at: float | None = None
    retracted_at: float | None = None


class SourceEvidence(BaseModel):
    source: str
    trust: float                    # declared trust at fold time (0..1)
    signals: int


class EvidenceOut(BaseModel):
    """What a confidence rests on, so the number can be explained or re-judged."""
    signals: int = 0
    sources: list[SourceEvidence] = Field(default_factory=list)
    max_trust: float = 0.0


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
    details: list[DetailItem] = Field(default_factory=list)
    evidence: EvidenceOut | None = None
    quote: Quote | None = None
    values: list[ValueWeight] = Field(default_factory=list)
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
    reverse_retired: int = 0
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
    details: list[DetailItem] = Field(default_factory=list)
    evidence: EvidenceOut | None = None
    quote: Quote | None = None
    values: list[ValueWeight] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


class ExportResponse(BaseModel):
    export_dir: str
    count: int
    etches: list[EtchOut]


# ── etches (filtered list) ───────────────────────────────────────────────────

class ListedEtchOut(EtchOut):
    entity_type: str | None = None


class EtchListResponse(BaseModel):
    """Beliefs matching plain filters — exhaustive within them, not ranked.

    `total` counts every match; `etches` is the page at `offset`. Page on until
    `offset + count >= total`.
    """
    etches: list[ListedEtchOut] = Field(default_factory=list)
    count: int = 0
    total: int = 0
    limit: int
    offset: int = 0


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
    scope: Scopes = Field(None, description=_SCOPE_HELP)
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
    details: list[DetailItem] = Field(default_factory=list)
    evidence: EvidenceOut | None = None
    values: list[ValueWeight] = Field(default_factory=list)
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
    detail: str | None = None
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
    source_record_id: str | None = None
    actor: dict = Field(default_factory=dict)
    created_at: float = 0.0
    occurred_at: float = 0.0


class SignalRecordResponse(BaseModel):
    """One signal as etchmem recorded it: what it said, how far it got through
    the pipeline, what it was found to be about, and the claims taken from it."""
    id: str
    content: str | None = Field(None, description="Withheld when claims anonymization is on.")
    source: str
    scope: str
    status: str = Field(..., description="new | batched | extracted")
    canonical_id: str | None = Field(None, description="The duplicate cluster's representative.")
    context: str | None = None
    context_parent: str | None = None
    source_record_id: str | None = None
    actor: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict,
                           description="As deposited, plus once resolved: `subjects` (what it "
                                       "is about), `associations` (related knowledge it "
                                       "brought to mind, not claims) and `focus` (what its "
                                       "conversation was about when it was said).")
    created_at: float = 0.0
    occurred_at: float = 0.0
    claims: list[ClaimOut] = Field(default_factory=list)


class ContextSubjectOut(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    mentions: int
    last_seen_at: float


class ContextSignalOut(BaseModel):
    id: str
    content: str | None = None
    source: str
    status: str
    context: str | None = None
    occurred_at: float = 0.0


class ContextResponse(BaseModel):
    """One conversation: what it has been about, and its signals in order."""
    id: str
    focus: list[ContextSubjectOut] = Field(default_factory=list)
    signals: list[ContextSignalOut] = Field(default_factory=list)


class DossierResponse(BaseModel):
    """Full provenance view of one belief: the etch, its version timeline,
    the claims that formed it, and the raw signals behind those claims."""
    etch: EtchOut
    versions: list[VersionOut]
    claims: list[ClaimOut]
    signals: list[SignalOut] = Field(default_factory=list)
    signals_omitted: bool = Field(
        False, description="True when claims_anonymization hides raw signals.")


# ── forget (erasure) ─────────────────────────────────────────────────────────

class ForgetRequest(BaseModel):
    """What to erase. Give at least one selector; they add up."""
    entity: str | None = Field(
        None, description="A subject to erase entirely (entity id or name, resolved "
                          "like `know`): its beliefs, the claims about it or pointing at "
                          "it, and every raw signal that cites it or names it.")
    entity_type: str | None = Field(None, description="Narrows how `entity` is resolved.")
    source_record_ids: list[str] = Field(
        default_factory=list,
        description="Erase the signals deposited with these `source_record_id`s, e.g. "
                    "after the record was deleted in its source system.")
    signal_ids: list[str] = Field(default_factory=list,
                                  description="Erase these signals (ids from `remember`).")
    reason: str | None = Field(
        None, description="Kept with the tombstone, e.g. 'GDPR Art. 17 request #123'. "
                          "Do not put personal data here.")


class ForgetResponse(BaseModel):
    """The erasure report: what was removed and which beliefs changed. Kept as
    a tombstone without personal data (counts and signal ids only)."""
    erasure_id: str
    reason: str | None = None
    entity_found: bool = False
    signals_matched_by_name: int = Field(
        0, description="Signals deleted because their text contains the name, whether or "
                       "not memory made the name a subject.")
    note: str | None = Field(None, description="Set when nothing matched.")
    signals_deleted: int = 0
    claims_deleted: int = 0
    claims_updated: int = 0
    etches_deleted: int = 0
    etches_refolded: int = 0
    versions_redacted: int = 0
    versions_deleted: int = 0
    refold: Literal["done", "pending", "none"] = Field(
        "none", description="'pending' = beliefs that lost evidence are blanked now and "
                            "re-consolidated on the next worker tick.")


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
    sources_without_trust: list[str] = Field(
        default_factory=list,
        description="Deposit sources seen but not declared in a `sources:` block "
                    "(or ETCHMEM_SOURCE_TRUST_JSON). They get the default trust.")


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    embedding_dim: int
    claim_model: str
    etch_model: str
    worker_enabled: bool
    claims_anonymization: bool = False
    database: str = "duckdb"
    auth: bool = False
