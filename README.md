# etchmem-server

**The enterprise memory operating system for AI employees.**

A REST service that turns a stream of raw agent signals into a clean, typed,
versioned knowledge base — backed by **DuckDB** (two databases) and a
**Pydantic AI** model cascade for consolidation. It is not RAG: instead of
returning documents, it maintains *beliefs* — one consolidated statement
("etch") per entity-property, with confidence, provenance, conflict status,
and a full version history.

See [TODO.md](TODO.md) for the product vision and planned future capabilities.

## Pipeline

```
 remember()          batch (dedup)        extract (LLM #2)        fold (gate + LLM #3)
┌────────┐  signals  ┌──────────────┐  claims  ┌──────────┐  etches  ┌──────────────┐
│ agents │ ────────► │  LEFT duckdb │ ───────► │  claims  │ ───────► │ RIGHT duckdb │ ◄── recall
└────────┘           │  signals     │          │ (typed)  │          │ entities     │     (semantic +
                     └──────────────┘          └──────────┘          │ etches+vers. │      time-travel)
                       status: new → batched → extracted → consolidated
```

- **Signal** — a raw deposit (`source`, `scope`, text, embedding). Immutable.
- **Claim** — one atomic typed assertion `(entity, property, value, polarity)`
  extracted from a signal. Append-only; identical claims merge as
  *corroboration* (counted, never discarded) so confidence reflects how many
  distinct sources agree.
- **Etch** — the current belief for one `(entity, property)`: `current_value`,
  `status` (settled/contested), `confidence`, a generated `narrative`, an
  embedding for recall, and a `version`. The etch is a *fold over claims*.
- **etch_versions** — an immutable snapshot per belief change → time-travel.

Entities are canonical (registry + aliases + fuzzy match), so "Acme Corp" and
"ACME Corporation" resolve to one entity and never contaminate each other.

## The consolidation cascade (cheap → expensive)

1. **Stage 1 — dedup (no LLM).** Embeddings group near-duplicate signals onto a
   canonical representative, preserving every source for corroboration.
2. **Stage 2 — claim extraction (mini model).** `claim_agent` turns a signal
   into typed claims, resolving/creating the canonical entity.
3. **Routing gate (no LLM).** For each `(entity, property)` the gate inspects
   the claims and decides: **AGREE** (one value), **POLICY** (recency /
   source-trust / multi-value cardinality resolves it), or **CONTESTED**.
4. **Stage 3 — conflict resolution (top model).** Only CONTESTED folds reach
   `etch_agent`, which resolves the disagreement and writes the narrative.

Each model is a Pydantic-AI model string, swapped via one env var
(`ETCHMEM_CLAIM_MODEL`, `ETCHMEM_ETCH_MODEL`).

## Runtime: one process, no broker

The API process also runs an in-process async **worker loop**. The signal/claim
`status` column *is* the queue. `POST /remember` writes a signal and returns
`202` immediately; the worker drains `new → batched → extracted → consolidated`
on a cadence. `POST /sleep` runs one tick on demand. Only one process touches
DuckDB, sidestepping its single-writer constraint. (When volume demands it, the
worker lifts out into its own container behind a real broker + Postgres/pgvector
— see TODO.md.)

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /remember` | Deposit a raw signal (`source`, `scope`, `extract_mode`). |
| `POST /recall` | Semantic recall over etches; `as_of` for time-travel. |
| `POST /sleep` | Run one worker tick now (batch → extract → fold). |
| `POST /export` | Dump all etches to JSON files. |
| `GET /etch/{id}/history` | Version timeline of one etch. |
| `GET /stats` | Queue depths + counts (signals/claims/entities/etches/contested). |
| `GET /health` | Liveness + active config. |

Interactive docs at `http://localhost:8000/docs`.

### Examples

```bash
# Deposit raw signals (source + scope on every signal)
curl -X POST localhost:8000/remember -H 'content-type: application/json' -d '{
  "data": "Acme Corp signed the enterprise contract today",
  "source": "agent-33", "scope": "sales", "extract_mode": "immediate"
}'

# Run a consolidation tick now (or let the worker do it on its cadence)
curl -X POST localhost:8000/sleep

# Recall the current belief
curl -X POST localhost:8000/recall -H 'content-type: application/json' \
  -d '{"query": "what is Acme'\''s contract status?", "top_k": 5}'

# Time-travel: what did we believe as of a given time?
curl -X POST localhost:8000/recall -H 'content-type: application/json' \
  -d '{"query": "Acme contract status", "as_of": "2026-06-24T01:30:00Z"}'
```

## Run with Docker

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up --build
```

DuckDB files persist in the `etchmem-data` volume at `/data`.

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export ETCHMEM_DATA_DIR=./data
uvicorn app.main:app --reload
```

## Configuration

Via environment / `.env` (see `.env.example`). Highlights:

- `ETCHMEM_CLAIM_MODEL` / `ETCHMEM_ETCH_MODEL` — cascade model strings.
- `EMBEDDING_PROVIDER` — `openai` (default), `local`, or `fake` (tests).
- `ETCHMEM_SIGNAL_DEDUP_DISTANCE`, `ETCHMEM_ENTITY_SIM_THRESHOLD` — dedup and
  entity-merge thresholds.
- `ETCHMEM_MULTI_VALUE_PROPERTIES` — properties that union instead of conflict.
- `ETCHMEM_SOURCE_TRUST_JSON` / `ETCHMEM_TRUST_GAP` — trust-based resolution.
- `ETCHMEM_WORKER_ENABLED`, `ETCHMEM_WORKER_INTERVAL_SECONDS`,
  `ETCHMEM_EXTRACT_MIN_BATCH`, `ETCHMEM_EXTRACT_MAX_WAIT_SECONDS` — worker cadence.

## Architecture

```
app/
  config.py     env-driven settings (+ cascade models, worker cadence)
  schemas.py    request/response models
  embeddings.py pluggable EmbeddingProvider (openai | local | fake)
  text.py       entity-name / value normalization, slugs
  hashing.py    content + claim hashing (corroboration & idempotency)
  stores.py     LeftStore (signals, claims) + RightStore (entities, etches, versions)
  dedup.py      Stage 1 — embedding dedup
  agents.py     Stage 2 claim_agent + Stage 3 conflict resolver (Pydantic AI)
  entities.py   entity resolution (find-or-create, alias, fuzzy)
  gate.py       deterministic routing gate + resolution policy + confidence
  worker.py     Pipeline (batch/extract/fold) + in-process WorkerLoop
  service.py    wires everything; remember/recall/sleep/export/stats/history
  main.py       FastAPI app + routes + worker lifespan
tests/
  test_integration.py  offline end-to-end (fake embedder + stub agents)
```

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q       # fully offline; no API key needed
```

Covered: entity-boundary (no contamination), corroboration raising confidence,
recency supersession + versioning, genuine conflict → contested (LLM invoked),
and time-travel recall.
