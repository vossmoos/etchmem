"""
Runtime configuration, loaded from environment / .env.

A single Settings instance (`settings`) is imported across the app.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM cascade (Pydantic AI model strings; swap freely) ───────────────
    # Stage 2: claim extraction (cheap/mini model).
    claim_model: str = "openai:gpt-5-mini"
    # Stage 3: conflict resolution + narrative (most capable model).
    etch_model: str = "openai:gpt-5.5"
    openai_api_key: str | None = None

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedding_provider: str = "openai"          # "openai" | "local" | "fake"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Storage ────────────────────────────────────────────────────────────
    data_dir: str = "./data"

    # ── Extensions (declarative claim/etch vocabulary) ─────────────────────
    # Folder of YAML files declaring extra *properties* the extractor should
    # look for (e.g. sales_intent). Additive vocabulary only — never overrides
    # the core (entity, property, value) triple. Missing folder = no extensions.
    ext_dir: str = "./ext"

    # ── Stage 1: signal dedup (batch) ──────────────────────────────────────
    # Cosine DISTANCE below which two raw signals are treated as duplicates
    # and collapsed onto one canonical representative (provenance preserved).
    signal_dedup_distance: float = 0.08

    # ── Entity resolution ──────────────────────────────────────────────────
    # Cosine SIMILARITY above which two entity names are treated as the same
    # entity (fuzzy fallback after normalized-name match).
    entity_sim_threshold: float = 0.86

    # ── Consolidation / fold ───────────────────────────────────────────────
    # Properties that may hold multiple simultaneous values (union, never a
    # conflict). Comma-separated env, e.g. "tech_stack,integrations".
    multi_value_properties: str = ""
    # Optional source-trust weights as JSON, e.g. {"crm":1.0,"agent-33":0.6}.
    # A gap larger than `trust_gap` lets the gate resolve a conflict by trust.
    source_trust_json: str = "{}"
    trust_gap: float = 0.3

    # ── Privacy: claims anonymization ──────────────────────────────────────
    # When true, personal data is anonymized as signals are folded into
    # claims/etches: person/company names become consistent numbered tokens
    # ([PERSON_1], [COMPANY_2]); addresses, bank cards, IBANs, emails and
    # phone numbers become generic tokens ([ADDRESS], [BANK_CARD], ...).
    # Recall then returns etches only (raw signals, which keep the original
    # text for provenance, are never surfaced).
    claims_anonymization: bool = False

    # ── Recall ─────────────────────────────────────────────────────────────
    recall_signal_weight: float = 0.6           # left-DB freshness enrichment

    # ── Worker (in-process; no external broker) ────────────────────────────
    worker_enabled: bool = True
    worker_interval_seconds: float = 5.0        # base tick
    extract_min_batch: int = 1                  # min batched signals to extract
    extract_max_wait_seconds: float = 30.0      # ...or extract anyway after this

    # ── Stage-2 throughput (batched prompts + bounded parallelism) ─────────
    extract_llm_batch_size: int = 6             # signals per extraction LLM call
    extract_concurrency: int = 4                # parallel extraction LLM calls
    llm_max_retries: int = 4                    # backoff retries on 429/transient

    # ── Signal TTL ─────────────────────────────────────────────────────────
    signal_ttl_seconds: int = 0                 # 0 = never expire

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def multi_value_set(self) -> set[str]:
        return {p.strip() for p in self.multi_value_properties.split(",") if p.strip()}

    @property
    def source_trust(self) -> dict[str, float]:
        import json

        try:
            return {k: float(v) for k, v in json.loads(self.source_trust_json).items()}
        except Exception:
            return {}


_ENV_ALIASES = {
    "claim_model": "ETCHMEM_CLAIM_MODEL",
    "etch_model": "ETCHMEM_ETCH_MODEL",
    "openai_api_key": "OPENAI_API_KEY",
    "embedding_provider": "EMBEDDING_PROVIDER",
    "openai_embedding_model": "OPENAI_EMBEDDING_MODEL",
    "local_embedding_model": "LOCAL_EMBEDDING_MODEL",
    "data_dir": "ETCHMEM_DATA_DIR",
    "ext_dir": "ETCHMEM_EXT_DIR",
    "signal_dedup_distance": "ETCHMEM_SIGNAL_DEDUP_DISTANCE",
    "entity_sim_threshold": "ETCHMEM_ENTITY_SIM_THRESHOLD",
    "multi_value_properties": "ETCHMEM_MULTI_VALUE_PROPERTIES",
    "source_trust_json": "ETCHMEM_SOURCE_TRUST_JSON",
    "trust_gap": "ETCHMEM_TRUST_GAP",
    "claims_anonymization": "ETCHMEM_CLAIMS_ANONYMIZATION",
    "worker_enabled": "ETCHMEM_WORKER_ENABLED",
    "worker_interval_seconds": "ETCHMEM_WORKER_INTERVAL_SECONDS",
    "extract_min_batch": "ETCHMEM_EXTRACT_MIN_BATCH",
    "extract_max_wait_seconds": "ETCHMEM_EXTRACT_MAX_WAIT_SECONDS",
    "extract_llm_batch_size": "ETCHMEM_EXTRACT_LLM_BATCH_SIZE",
    "extract_concurrency": "ETCHMEM_EXTRACT_CONCURRENCY",
    "llm_max_retries": "ETCHMEM_LLM_MAX_RETRIES",
    "signal_ttl_seconds": "ETCHMEM_SIGNAL_TTL_SECONDS",
    "host": "ETCHMEM_HOST",
    "port": "ETCHMEM_PORT",
}


@lru_cache
def get_settings() -> Settings:
    import os

    overrides: dict[str, str] = {}
    for field, env_name in _ENV_ALIASES.items():
        val = os.environ.get(env_name)
        if val is not None:
            overrides[field] = val
    return Settings(**overrides)


settings = get_settings()
