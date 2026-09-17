# etchmem-server

**The Knowledge Consolidation Engine for AI agents.**

Your agents' daily activity is full of hard-won facts — deals signed, decisions
made, preferences learned. Saved in transcripts, yes; but as dead text, not as
knowledge the next task can use as a pattern.
etchmem turns that stream of raw signals into a clean, typed, versioned
knowledge base: **one consolidated belief per fact**, with confidence,
provenance, conflict status, and full version history — queryable at any
point in time.

**Extend a skill with dynamic data — without rewriting the skill.** Today,
teaching an agent something new usually means editing its `SKILL.md` by hand:
rewrite the text, redeploy, repeat. etchmem makes skill knowledge a live layer
instead. Scope memory to a skill, and the agent recalls matured, consolidated
beliefs at runtime — the skill's *behavior* grows as new signals arrive, while
its prompt text stays untouched. The skill file stays the stable contract; the
knowledge behind it is a dynamic extension, not another rewrite.

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
  Optionally carries `occurred_at`: when the content actually *happened*, as
  distinct from when we received it. Required for historical loads — see
  [Ingesting a historical archive](#ingesting-a-historical-archive).
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

## Claim extensions (declarative domain vocabulary)

Out of the box the extractor is open-vocabulary: it invents sensible
snake_case properties as it goes. Extensions let you *steer* it with your
domain's vocabulary — declaratively, without touching code or the DuckDB
schema. Drop YAML files into `ext/` (override with `ETCHMEM_EXT_DIR`), one
domain per file; `ext/sales.yaml` ships as an example:

```yaml
domain: sales
properties:
  - name: sales_intent
    description: How strongly the subject signals intent to purchase.
    values: [none, low, medium, high]     # optional enum (case-insensitive)
    entity_types: [company, person]       # optional subject filter
  - name: decision_maker
    description: Named person who owns the buying decision.
    entity_types: [company]
```

### Declaring entities

`properties:` says what to extract. `entities:` says which **subjects** matter
and how strictly to identify them — the difference between a corpus that
consolidates and one that quietly merges two things into one:

```yaml
domain: catalogue-support
entities:
  - type: product
    description: A module identified by its article number.
    match: exact                       # never fuzzy-merge two products
    identifier_pattern: '[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?'
    sim_threshold: 0.97                # optional per-type override
  - type: ticket
    ignore: true                       # record ids are not knowledge
```

| Key | Effect |
|-----|--------|
| `identifier_pattern` | When it matches inside the surface name, the normalized match becomes the entity's **canonical key** and resolution is string equality. Fuzzy matching is skipped entirely. |
| `match: exact` | Never resolve this type by embedding similarity, pattern or not. |
| `sim_threshold` | Per-type override of `ETCHMEM_ENTITY_SIM_THRESHOLD`, for types that still resolve fuzzily. |
| `ignore: true` | Drop claims about this subject type. The extractor is also told not to produce them. |

### One subject per claim

A claim has exactly one subject, and the extractor is told so: "A and B both
fail" arrives as two claims, not one claim named `A und B`. That instruction is
where the language understanding belongs — the extractor reads the whole
sentence, with its grammar and its verb.

When a claim arrives compound anyway, the engine does **not** try to take it
apart. Whether `MSM-0808 und MSM-0810` is a list and `NT-2405 für MSM-0808` is a
relation is a reading of the original sentence, and by the time a claim exists
that sentence is gone. Inspecting the mangled name with a word list would decide
a language question with strictly less information than the model already had,
in one language at a time, and would need extending forever.

So instead:

1. **Detect structurally.** "Does `entity_name` carry more than one identifier?"
   is a count. It behaves identically in German, French, Italian and Romansh,
   with nothing to configure.
2. **Ask the model again.** That one signal is re-read with a correction naming
   exactly what went wrong, and the original sentence attached. Cheap, and only
   for the rare malformed case — the same escalate-the-hard-ones principle as
   the routing gate. `ETCHMEM_SUBJECT_RETRY_ENABLED=false` turns it off.
3. **Drop what is still compound.** One retry, then the claim is dropped and
   counted. Unattached beats attached to the wrong subject.

`subject_retries` on every `/sleep` shows how often the extractor is producing
malformed subjects; a rising number is a prompt problem, not a data problem.

`distributive: true` marks a property that can legitimately hold for several
subjects at once:

```yaml
properties:
  - name: known_fault
    distributive: true          # "A and B both leak" is two facts
  - name: lifecycle_status
    # absent: "A or B is discontinued" says ONE of them
```

It is passed to the model as guidance during the correction, never applied
afterwards as a rule. The engine has no distribution step: several subjects
means several claims *from the model*, or nothing.

Declared properties are injected into the extractor's system prompt (so it
actively looks for them) and **enforced on the way back**: a declared enum
rejects drifting values (no `sales_intent: "very high"`), and `entity_types`
rejects claims about the wrong kind of subject. Undeclared properties still
pass — the core stays open-vocabulary. Each declared property becomes a plain
`(entity, property, value)` claim and therefore a plain etch
(`entity.sales_intent`), flowing through the same corroboration, gate and
versioning as everything else. Extensions are additive vocabulary only; they
never override the core triple. Edit or add files, then restart the server to
pick up changes.

## Ingesting a historical archive

Loading years of existing records — support tickets, case notes, order history —
has one failure mode that is silent and fatal, and one configuration answer.

**Declare `occurred_at` on every signal.** A signal's `created_at` is always
ingest time. Without `occurred_at`, a backfill run today gives *every* claim
today's timestamp, the gate's recency policy has nothing to order by, and a 2009
answer competes with a 2022 answer as an equal. The archive consolidates into
nonsense that looks fine.

```bash
curl -X POST localhost:8000/remember -H 'content-type: application/json' -d '{
  "data": "Modul MSM-0808: NT-2405 is the correct supply up to revision C.",
  "source": "ticket-resolved", "scope": "innoxel",
  "occurred_at": "2014-03-17"
}'
# → {"id": "...", "stored": true, "occurred_at": 1395014400.0,
#    "occurred_at_declared": true}
```

The response echoes the parsed value, so a bulk load can assert the date was
understood rather than discovering months later that it was not.

Precedence, when both a declared `occurred_at` and an in-text date exist:

| `ETCHMEM_OCCURRED_AT_PRECEDENCE` | Winner |
|---|---|
| `extracted` (default) | The date the extractor read from the text. More specific — a 2014 record may state something became true in 2009. |
| `declared` | The caller's `occurred_at`, always. For bulk loads where the record date is authoritative and a misread date would corrupt the timeline silently. |

**Guard the dedup.** Stage 1 collapses near-identical signals, which is an asset
here (500 records of the same fault become one belief with corroboration 500) and
a hazard across decades: archived records are formulaic, so two of them years
apart can sit well inside the distance threshold while describing different
things. `ETCHMEM_DEDUP_MAX_TIME_GAP_SECONDS` adds the missing condition — two
signals further apart in event time than this are never the same signal:

```
ETCHMEM_DEDUP_MAX_TIME_GAP_SECONDS=604800     # 7 days; 0 = off (default)
ETCHMEM_SIGNAL_DEDUP_DISTANCE=0.04            # tighter than the 0.08 default
```

**Check the entity declaration before loading anything.** Entity resolution is
the failure that hurts most and shows least: two article numbers merging into
one subject cross-contaminates two histories, and every answer afterwards looks
plausible. `app.validate_entities` runs the declaration against real names and
reports what would happen — exit code 1 gates a load in CI:

```bash
python -m app.validate_entities --names article-numbers.txt --type product
```

```
AMBIGUOUS — two identifiers, no safe key (1)
  - NT-2405 for MSM-0808   candidates: nt-2405, msm-0808
NO MATCH — pattern found nothing (1)
  - das alte Modul
COLLISIONS — one key, several names (1)
  msm-0808: MSM-0808, MSM 0808, INNOXEL Modul MSM-0808
NEAR MISSES — keys one edit apart (1)
  msm-0808  vs  msm-0809
```

Collisions are what you want (spelling variants of one article). Near misses are
the pairs a similarity threshold would have been at risk of merging. Feed it real
entity names from the archive, not just clean article numbers.

**Watch the resolution counters.** Every `/sleep` reports how subjects were
identified:

```json
"entities": {"entities_by_key": 812, "entities_by_fuzzy": 0,
             "entities_created": 44, "entities_ambiguous": 3,
             "entities_unmatched": 17, "entities_ignored": 900}
```

`by_fuzzy` above zero on a type you declared with an `identifier_pattern` means
the pattern is not matching real names and identity has quietly gone back to
guessing. `ambiguous` counts names carrying two identifiers — there is no safe
key for those, so the fact stays unattached rather than attaching to the wrong
subject; it is a prompt problem, not a data problem.

**Validate the timeline before the full run.** Load a few hundred records
spanning the whole period, then check that `claims.event_time` actually spans it
rather than clustering on load day. If it clusters, nothing downstream is
trustworthy and no amount of later tuning fixes it.

## Relations: when the value is another entity

Declare a property's value as a reference and it stops being a string:

```yaml
properties:
  - name: fix_procedure
    relation: part          # the value IS a part, resolved like any subject
  - name: replaced_by
    relation: part
```

Three things follow. The value goes through the same identifier rules as a
subject, so `DK-1120` and `DK 1120` become one node instead of two values of one
belief fighting at the gate. The edge is followable backwards —
`etches_referencing(entity_id)` answers "which products are fixed by DK-1120?"
as one indexed lookup rather than a text search. And `associate` can walk it.

`relations_resolved` on every `/sleep` counts the edges built.

## Three ways to ask

| You know | Operation | Returns |
|---|---|---|
| the subject | `know(ref)` | every belief about it, exhaustive |
| only words | `recall(query)` | beliefs ranked by wording |
| only words, and want what they connect to | `associate(query)` | the above **plus** every fact of each matched subject **plus** the nodes they are linked to |

`associate` is for the question "what do we have on this?" when you cannot name
the thing. It seeds from beliefs whose narratives match, expands each matched
subject to its full set of facts, then follows declared relations one hop in
both directions.

```python
a = mem.associate("bathroom leaks")
for n in a.nodes:
    print(n["hops"], n["entity"]["name"], n["reached_via"])
```

The point is the hop. A fault matches by wording; the seal that fixes it shares
no word with the query and is reachable only because the memory is connected.
`min_score` (default 0.15) drops seeds that do not really match — `recall`
returns `top_k` whatever the similarity, and a zero-scoring seed would drag its
whole neighbourhood into the answer.

## Recall vs. entity facts

Two different questions, two different operations, and using the wrong one is a
quiet source of wrong answers.

`POST /recall` is **semantic search**: it ranks beliefs by embedding distance to
a query and returns `top_k`. Right for "what do we know about leaking seals",
where the caller cannot name the subject.

`GET /entity/{ref}/etches` is **exhaustive**: every belief held about one
subject, ordered by property. Right for "what do we know about MSM-0808". Recall
would answer that too, but a fact whose narrative embeds poorly against the
article number would simply be absent — and the caller cannot tell the
difference between *not known* and *not ranked*. When an answer depends on
completeness, rank is the wrong instrument.

`ref` is an entity id (`product_msm_0808`) or a surface name (`MSM-0808`,
`MSM 0808`, `INNOXEL Modul MSM-0808`), resolved exactly as ingestion resolved it,
so callers never need the internal id format. `?as_of=` gives the beliefs as
they stood then; the response carries a `contested` count so a caller can apply
the settled/contested rule without walking the list.

```python
facts = mem.know("MSM-0808")
for e in facts.settled(min_confidence=0.6):     # safe to state as fact
    print(e.property, "=", e.current_value)
print(facts.contested, "open questions")        # never in a customer reply
```

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
| `POST /associate` | What the memory holds for a phrase: matching beliefs, their subjects' facts, and connected nodes. |
| `GET /entity/{ref}/etches` | **Every** belief about one subject — exhaustive, not ranked. |
| `GET /etch/{id}/history` | Version timeline of one belief. |
| `GET /etch/{id}/dossier` | Full provenance: etch + versions + claims + source signals. |
| `GET /stats` | Queue depths + counts (signals/claims/entities/etches/contested). |
| `GET /health` | Liveness + active config. |
| `GET /ui` | **Etchmem Beliefs Explorer** — built-in report UI. |

REST, any language, any agent framework. Interactive docs at
`http://localhost:8000/docs`.

## Beliefs Explorer (built-in UI)

A single-page report interface at `http://localhost:8000/ui` — no build step,
no extra service, served by the same process. Two report types:

- **Beliefs report** — search string + as-of date (calendar; default today).
  Today shows the live view including fresh signals; a past date reconstructs
  what the system believed then via time-travel recall.
- **Belief dossier** — click any belief for its full history: version
  timeline, underlying claims (corroboration, polarity, sources), and the raw
  source signals. When claims anonymization is on, raw signals are withheld.

Export via the browser's *Print / Save as PDF* button (print stylesheet
included). Disable the UI with `ETCHMEM_UI_ENABLED=false`.

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
- `ETCHMEM_EXT_DIR` — folder of claim-extension YAML files (see above).
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
  ext.py        claim extensions: YAML domain vocabulary → prompt + enum enforcement
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
