# etchmem-server

**The Knowledge Consolidation Engine for AI agents.**

Your agents generate experience all day and forget it by the next session.
etchmem turns that stream of raw signals into a clean, typed, versioned
knowledge base: **one consolidated belief per fact**, with confidence,
provenance, conflict status, and full version history — queryable at any
point in time.

It's **more advanced than RAG**. RAG retrieves document chunks and leaves your
agent to guess which of five contradicting versions is true. etchmem
*consolidates*: it extracts claims, resolves entities, counts corroboration,
settles or flags conflicts, and maintains the current belief — with an audit
trail from every answer back to its sources — and does it transparently.

|                               | Custom RAG                  | etchmem                              |
|-------------------------------|-----------------------------|--------------------------------------|
| Returns                       | document chunks             | consolidated beliefs (memory etches) |
| Contradicting sources         | all returned, agent guesses | resolved or flagged *contested*      |
| "Why does the agent think that?" | unanswerable             | narrative + full provenance chain    |
| Knowledge changes over time   | stale chunks accumulate     | versioned, superseded, time-travel   |
| Confidence                    | similarity score            | corroboration-based confidence       |
| Duplicate entities            | "Acme" ≠ "ACME Corp"        | one canonical entity                 |

Keep your RAG for document search. Use etchmem for what your system *believes*
and which patterns it knows.

Backed by **DuckDB** (two databases) and a **Pydantic AI** model cascade.
One process, one `docker compose up`, a REST API **and an MCP interface**
(same operations as tools at `/mcp`).

See [TODO.md](TODO.md) for the product vision and planned capabilities.

## Pipeline

```
 remember()          batch (dedup)        extract (LLM #2)        fold (gate + LLM #3)
┌────────┐  signals  ┌──────────────┐  claims  ┌──────────┐  etches  ┌──────────────┐
│ agents │ ────────► │  LEFT duckdb │ ───────► │  claims  │ ───────► │ RIGHT duckdb │ ◄── recall
└────────┘           │  signals     │          │ (typed)  │          │ entities     │     (semantic +
                     └──────────────┘          └──────────┘          │ etches+vers. │      time-travel)
                       status: new → batched → extracted → consolidated
```

### Core concepts

- **Signal** — a raw deposit (`source`, `scope`, text, embedding). Immutable.
  Call notes, tool outputs, emails, decisions — nothing to pre-structure.
- **Claim** — one atomic typed assertion `(entity, property, value, polarity)`
  extracted from a signal. Append-only; identical claims merge as
  *corroboration* (counted, never discarded), so confidence reflects how many
  distinct sources agree.
- **Etch** — the current belief for one `(entity, property)`: `current_value`,
  `status` (settled / contested), `confidence`, a one-sentence generated
  `narrative` explaining the belief, an embedding for recall, and a `version`.
  The etch is a *fold over claims*.
- **etch_versions** — an immutable snapshot per belief change. This is what
  makes time-travel and incident forensics possible: replay exactly what the
  system believed when a decision was made.

Entities are canonical (registry + aliases + fuzzy match), so "Acme Corp" and
"ACME Corporation" resolve to one entity and never contaminate each other.

## The consolidation cascade (cheap → expensive)

Consolidation cost stays flat because expensive models only see genuinely
hard cases:

1. **Stage 1 — dedup (no LLM).** Embeddings group near-duplicate signals onto
   a canonical representative, preserving every source for corroboration.
2. **Stage 2 — claim extraction (mini model).** `claim_agent` turns a signal
   into typed claims, resolving/creating the canonical entity.
3. **Routing gate (no LLM).** For each `(entity, property)` the gate inspects
   the claims and decides: **AGREE** (one value), **POLICY** (recency /
   source-trust / multi-value cardinality resolves it), or **CONTESTED**.
4. **Stage 3 — conflict resolution (top model).** Only CONTESTED folds reach
   `etch_agent`, which resolves the disagreement and writes the narrative.
   When a conflict can't be settled, the belief is marked *contested* — your
   agents know what they know and how well they know it.

Each model is a Pydantic-AI model string, swapped via one env var
(`ETCHMEM_CLAIM_MODEL`, `ETCHMEM_ETCH_MODEL`). No vendor lock-in.

## Explainability by construction

Every belief answers "why do you think that?" without extra tooling:

```
belief (narrative, confidence, status)
  └── claims (typed assertions, corroboration counts)
        └── signals (raw deposits: which source, which scope, when)
```

`GET /etch/{id}/history` returns the full version timeline —
what changed, when, and what triggered it.

## Runtime: one process, no broker

The API process also runs an in-process async **worker loop**. The
signal/claim `status` column *is* the queue. `POST /remember` writes a signal
and returns `202` immediately; the worker drains
`new → batched → extracted → consolidated` on a cadence. `POST /sleep` runs
one tick on demand. Only one process touches DuckDB, sidestepping its
single-writer constraint. When volume demands it, the worker lifts out into
its own container behind a real broker + Postgres/pgvector — see TODO.md.

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /remember` | Deposit a raw signal (`source`, `scope`, `extract_mode`). |
| `POST /recall` | Semantic recall over beliefs; `as_of` for time-travel. |
| `POST /sleep` | Run one worker tick now (batch → extract → fold). |
| `POST /export` | Dump all etches to JSON files. |
| `GET /etch/{id}/history` | Version timeline of one belief. |
| `GET /stats` | Queue depths + counts (signals/claims/entities/etches/contested). |
| `GET /health` | Liveness + active config. |

REST, any language, any agent framework. Interactive docs at
`http://localhost:8000/docs`.

## MCP interface

The same operations are exposed as **MCP tools** — `remember`, `recall`,
`sleep`, `export`, `stats`, `etch_history` — over streamable HTTP at
`http://localhost:8000/mcp` (stateless, JSON responses). One process serves
both protocols, so Claude Code / Claude Desktop / any MCP client can use
etchmem as a memory backend directly:

```json
{
  "mcpServers": {
    "etchmem": { "type": "http", "url": "http://localhost:8000/mcp" }
  }
}
```

## Export: your knowledge as training data

`POST /export` (or the `export` MCP tool) dumps every consolidated etch to a
timestamped directory of JSON files — one file per belief, carrying the
current value, status, confidence, narrative, version and full provenance
(claim ids + source signal ids). Because etches are already deduplicated,
entity-resolved, conflict-settled and confidence-scored, the export is a
clean, typed dataset rather than a document dump: use it to build fine-tuning
corpora or evaluation sets from what your agents actually learned, feed
distilled domain knowledge into your own models, or snapshot the knowledge
base for backup and offline analysis. Combined with claims anonymization
(below), the exported JSON is free of personal data — ready to leave the
production boundary.

## Claims anonymization (privacy mode)

`ETCHMEM_CLAIMS_ANONYMIZATION=true` (default `false`) anonymizes personal
data at the point where signals are folded into claims/etches — the
retrieval surface:

- **Person/company subject names** become consistent numbered pseudonyms
  (`[PERSON_1]`, `[COMPANY_2]`), assigned once per canonical entity and
  persisted — so corroboration, conflict detection and versioning keep
  working across sources.
- **Addresses, bank cards, IBANs, emails, phone numbers** inside values and
  narratives become generic tokens (`[ADDRESS]`, `[BANK_CARD]`, `[IBAN]`,
  `[EMAIL]`, `[PHONE]`) — enforced by both LLM instructions and a
  deterministic regex safety net (`app/anonymize.py`).
- **Recall returns etches only**: raw signals keep their original text for
  provenance/audit but are never surfaced through retrieval while
  anonymization is on.

### Examples

```bash
# Deposit raw signals (source + scope on every signal)
curl -X POST localhost:8000/remember -H 'content-type: application/json' -d '{
  "data": "Acme Corp signed the enterprise contract today",
  "source": "agent-33", "scope": "sales", "extract_mode": "immediate"
}'

# Run a consolidation tick now (or let the worker do it on its cadence)
curl -X POST localhost:8000/sleep

# Recall the current belief — with confidence, narrative, provenance
curl -X POST localhost:8000/recall -H 'content-type: application/json' \
  -d '{"query": "what is Acme'\''s contract status?", "top_k": 5}'

# Time-travel: what did we believe as of a given moment?
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
- `ETCHMEM_SOURCE_TRUST_JSON` / `ETCHMEM_TRUST_GAP` — trust-based conflict
  resolution: rank your sources, let policy settle disagreements.
- `ETCHMEM_CLAIMS_ANONYMIZATION` — anonymize personal data in claims/etches
  (see above). Default `false`.
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
  anonymize.py  claims anonymization: pseudonym labels, prompt blocks, regex scrub
  worker.py     Pipeline (batch/extract/fold) + in-process WorkerLoop
  service.py    wires everything; remember/recall/sleep/export/stats/history
  mcp_server.py MCP tools (FastMCP, streamable HTTP; mounted at /mcp)
  main.py       FastAPI app + routes + MCP mount + worker lifespan
tests/
  test_integration.py    offline end-to-end (fake embedder + stub agents)
  test_anonymization.py  pseudonym consistency, regex scrub, signal exclusion
  test_mcp.py            tool registration, /mcp mount, tool ↔ service parity
```

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q       # fully offline; no API key needed
```

Covered: entity-boundary (no contamination), corroboration raising confidence,
recency supersession + versioning, genuine conflict → contested (LLM invoked),
time-travel recall, claims anonymization, and the MCP interface.

## Production deployments

etchmem is open source — clone it, ship it, never talk to us. If you want it
integrated into your agent stack faster (signal capture design, consolidation
policy tuning, scoped knowledge across teams, recall wiring), we do
fixed-scope implementations: [etchmem.io](https://etchmem.io).
