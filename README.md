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

Backed by **PostgreSQL + pgvector** (or embedded **DuckDB** with zero setup)
and a **Pydantic AI** model cascade. One process, one `docker compose up`, a
REST API **and an MCP interface** (same operations as tools at `/v1/mcp`),
optionally behind an API key.

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
2. **Stage 2 — what is it about, then what does it say.**
   - *Subject resolution (top model by default, `ETCHMEM_RESOLVE_MODEL`).* A
     model decides what the signal is about, reading it in its conversation
     first (see [Conversations](#conversations-context-focus-associations)).
     Candidates are options, not content: the conversation's focus, what is
     linked to it by a declared relation, and the subjects that similar earlier
     signals and beliefs were about. Related knowledge it recalls is recorded
     as *associations*, never as subjects.
   - *Claim extraction.* Typed claims about exactly those subjects. The model
     sees the signal, its subjects and — for a reply that refers back — the
     resolve step's *meaning* (references spelled out); never the raw
     conversation. A question gives no claims.
   - *Verification (mini model).* A second reading drops claims the signal
     does not state, and maps a restated fact onto its existing label.

   Measured on real team chat: run resolution and extraction on your top model
   (`ETCHMEM_RESOLVE_MODEL`, `ETCHMEM_CLAIM_MODEL`); small models invent
   subjects like "the link" and mislabel agreement and requests. Verification
   (`ETCHMEM_VERIFY_MODEL`) is a yes/no check and runs well on a small model.
3. **Routing gate (no LLM).** For each `(entity, property)` the gate inspects
   the claims and decides: **AGREE** (one value), **POLICY** (recency /
   source-trust / multi-value cardinality / retraction resolves it), or
   **CONTESTED**. See [How beliefs change over time](#how-beliefs-change-over-time).
4. **Stage 3 — conflict resolution (top model).** Only CONTESTED folds reach
   `etch_agent`, which resolves the disagreement and writes the narrative.
   When a conflict can't be settled, the belief is marked *contested* — your
   agents know what they know and how well they know it.

Each model is a Pydantic-AI model string, swapped via one env var
(`ETCHMEM_CLAIM_MODEL`, `ETCHMEM_ETCH_MODEL`, `ETCHMEM_RESOLVE_MODEL`,
`ETCHMEM_VERIFY_MODEL`). No vendor lock-in. Every judgement about language —
what a record is about, whether it states a fact, whether two wordings are the
same fact — is a model's; the code only counts and compares identities.

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

### Label + detail: a short name with the author's full wording

A fact's `value` has to stay short: it is the claim's identity, so dedup,
corroboration, the gate and confidence all work on it. But many facts are
useless without the author's full wording ("avoid the core-product framing —
this campaign is about CAST Radar, the new product").

**By default every claim may carry it.** The extractor fills `detail` whenever
the short value loses what was said (a message, a complaint, a problem and its
cause, a decision with its reasons), and leaves it empty when the value says
everything (a status, a tier, a date). The `value` of such a claim becomes a
2-5 word label. Nothing has to be declared. `ETCHMEM_DETAIL_DEFAULT=false`
restores the opt-in behaviour: only properties declared `detail: true` keep a
wording.

Declaring a property with `detail: true` still helps: its hints steer the label
and the wording.

**Long sources stay short.** A wording is a condensed version in the author's
own words: the extractor keeps their exact terms, names, numbers, dates and
conditions, and drops greetings, signatures, quoted history, repetition and
other facts (which get claims of their own). It never paraphrases into its own
vocabulary. It is asked for at most `ETCHMEM_DETAIL_MAX_CHARS` / 6 words (400
characters, about 66 words), and a longer wording is cut at a sentence end when
it is stored, so a model that copies a whole document cannot flood recall. The
full text stays in the signal, one dossier call away.

```yaml
properties:
  - name: known_fault            # customer-care tickets
    detail: true
    label_hint:  "short fault name, e.g. 'overnight battery drain'"
    detail_hint: "the symptom as the customer described it plus the confirmed cause"
  - name: resolution
    detail: true
    label_hint:  "the fix in a few words, e.g. 'firmware 2.3.1 update'"
    detail_hint: "the exact steps that worked, with conditions and caveats"
```

The extractor then returns a 2-5 word **label** as `value` and the full wording
as `detail`. The label decides identity: the same label from a second ticket
is corroboration (confidence rises), not a new claim. The detail is not part of
the claim hash. When a label is asserted again, the detail with the **latest
event time** wins, and older wordings stay in the version history. An older
signal arriving late never overwrites a newer wording.

`current_value` keeps the label(s). The wording is in `details`
(`[{"value": label, "detail": wording}]`) on recall, `know`, `export`,
`/history` and the dossier, one entry per label for multi-value properties,
and in the etch's narrative so recall matches on it. With
`ETCHMEM_DETAIL_DEFAULT=false`, a `detail` the model emits for an undeclared
property is dropped.

### Quotes: who said it, where, when, in their words

Each belief returned by `recall` and `know` carries a `quote`: the newest signal
behind its current value, so an agent can cite the fact, judge how fresh it is,
check the wording against the original and open the source record, without a
dossier call.

```json
"quote": {
  "signal_id": "sig_…",
  "excerpt": "…the October invoice was charged twice and asks for a credit note before the end of the month…",
  "source": "zendesk",
  "source_record_id": "zendesk:ticket:123",
  "actor": {"agent": "mossy", "user": "maria@acme.com"},
  "occurred_at": 1762509600.0
}
```

The `excerpt` is the original text around the fact, at most `ETCHMEM_QUOTE_CHARS`
characters (300; 0 turns quotes off). `quotes=ref` on `recall` or `know` gives
shorter replies: the excerpt is then kept only for beliefs without a wording (a
status, a tier), since elsewhere it mostly repeats the wording. Useful when an
agent reads many beliefs at once.

With `as_of`, the quote is the newest signal known at that time. With claims
anonymization on, `excerpt` and `actor` are always left out: raw signals keep
personal data. The dossier still returns every signal in full.

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

## How beliefs change over time

Facts get withdrawn, states move on, and people of different standing
contradict each other. Three deterministic rules in the gate handle that, each
with a recommended default and an env var to tune or switch it off.

**Retraction: deprioritised, not deleted.** A value whose newest negation
("we no longer need CSV export") is newer than its newest assertion is
*retracted*. It is kept, with its weight multiplied by
`ETCHMEM_RETRACTION_WEIGHT` (default `0.1`, one order of magnitude lower):

- `current_value` lists only the values still in force;
- `values` lists every value ever asserted, heaviest first:
  `{"value", "weight", "active", "last_asserted_at", "retracted_at"}`, so the
  memory knows both that a value is no longer valid and that it once was;
- when every value of a belief is retracted, the belief keeps its last value
  with confidence × 0.1, its narrative says so, and recall ranks it × 0.1:
  still findable, no longer competing with what is true;
- asserting the value again after the retraction reactivates it.

```json
"current_value": "parquet export",
"values": [
  {"value": "parquet export", "weight": 0.91, "active": true,
   "last_asserted_at": 1788566400.0, "retracted_at": null},
  {"value": "csv export", "weight": 0.09, "active": false,
   "last_asserted_at": 1788480000.0, "retracted_at": 1790035200.0}]
```

**Agreement half-life: history is not disagreement.** A belief that moved
`proposed → agreed → dropped` has three values in its history. Counting them
equally made the latest one look like a 1-in-3 minority. Now each claim weighs
`0.5 ** (age_days / ETCHMEM_AGREEMENT_HALF_LIFE_DAYS)` (default `14`, one
sprint), with age measured back from the newest claim in the fold, not from
"now", so a belief nobody touches does not erode. `0` restores equal weights.

**Comparable trust: who may overrule whom.** A newer value used to win on
recency alone, so a casual remark on Thursday overrode Monday's decision by
someone with more authority. Now the newer value wins only if its source is
at least as trusted, or *comparably* trusted, as the strongest older value's.
Trust is compared as odds, `t / (1 − t)`, the scale on which people judge
probabilities roughly linearly; the odds ratio may not exceed
`ETCHMEM_TRUST_SAME_ODDS_RATIO` (default `3`, the conventional line below which
evidence is "barely worth mentioning", Jeffreys 1961; Kass & Raftery 1995).
Beyond it the belief is **contested** and goes to conflict resolution, with
policy `trust_dispute`. A very large value restores pure recency.

| newer source | older source | odds ratio | outcome |
|---|---|---|---|
| 0.70 | 0.85 | 2.4 | newer wins (recency) |
| 0.60 | 0.85 | 3.8 | contested |
| 0.85 | 0.95 | 3.4 | contested |
| 0.95 | 0.50 | — | newer wins (more trusted) |

### What the extractor sees

Each signal reaches the extractor with a header,
`[sent 2026-10-16 (Fri) 14:05 UTC · source slack]`, built from `occurred_at`
(else ingest time) and `source`. Relative dates in the text ("by Friday",
"yesterday") are resolved against it into absolute ISO dates. It also gets the
subjects the resolve step found (`[subjects: ...]`) — and nothing else from
memory: lists of known names or labels pull vague records onto known subjects,
and make small models restate old facts instead of the new one. Existing labels
(`ETCHMEM_EXTRACT_KNOWN_LABELS`, default 10 per subject and property, retracted
ones included) go to the verify step instead, which puts a restated fact on its
existing label — so the same fact keeps one label (corroboration) and "we
dropped X" negates exactly the X that was asserted (retraction).

Existing beliefs pick up these rules the next time their `(entity, property)`
is folded.

## Conversations: context, focus, associations

A message like "this ticket is done now" means something precise only in the
conversation it was said in. Pass where it was said with `remember`:
`context` (the conversation — a Slack thread, a ticket's comments, a meeting)
and `context_parent` (the one around it — the channel). Plain ids the client
chooses; etchmem never interprets them, and the same words in another context
are another signal.

- **Focus.** Each conversation keeps the subjects it has been about, most
  recent first. A signal is read with that focus and its conversation's last
  messages; around a thread, the channel's last messages are shown with their
  age, and the model weighs whether they still frame it. The channel's focus
  itself carries over into a new thread within `ETCHMEM_CONTEXT_WINDOW_SECONDS`
  (7200). The focus reaches one hop through declared relations, so a
  conversation about a feature knows the ticket that tracks it.
- **Associations.** Related knowledge recalled while reading — an earlier
  iteration, a ticket that already covers the work — is recorded on the
  signal with a reason, never as a claim. Whether to flag it is the client's
  call.
- **Same area is not the same item.** A signal is about an existing item when
  it names it, or refers to it while the conversation is about it. New work in
  that item's area is a new subject, with the item as an association.
- **Meaning.** A reply that refers back ("let us implement that", "yes, but
  ...", "no, new task") is restated with its references spelled out, kept on
  the signal as `metadata.meaning`, and extracted from. Agreement turns the
  agreed work into requirements; a correction retracts exactly the facts that
  were recorded on the wrong subject, and nothing else.
- **Order.** Signals of one conversation are read in the order they were said,
  each seeing the ones before it; conversations run in parallel. A signal
  without a context is read as before, against the scope's knowledge alone.

`GET /signal/{id}` shows a signal's context, the focus when it was said, its
subjects (with the words that refer to each and `in_focus`), its associations
and its claims. `GET /context/{id}` shows a conversation's focus and signals.

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
| the kind of belief (scope, property, subject type) | `etches(scope, property, entity_type)` | every matching belief across subjects, exhaustive, paged |
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
one tick on demand. Run one etchmem process per database (or per
`ETCHMEM_DATABASE_SCHEMA`): the worker assumes it is the only one folding.

## Storage: PostgreSQL + pgvector, or DuckDB

Set `ETCHMEM_DATABASE_URL` and etchmem keeps everything in PostgreSQL:

```bash
ETCHMEM_DATABASE_URL=postgresql://etchmem:secret@db:5432/etchmem
ETCHMEM_DATABASE_SCHEMA=acme      # optional: one schema per instance / tenant
```

On start it creates the `vector` extension, its tables and HNSW indexes on the
embedding columns (the role needs `CREATE` on the database, or create the
extension beforehand). Similarity search orders by pgvector's cosine distance,
which the index serves. Several instances can share one database, each in its
own schema.

Leave `ETCHMEM_DATABASE_URL` empty and etchmem uses two embedded DuckDB files
in `ETCHMEM_DATA_DIR` (`left.duckdb`: signals and claims; `right.duckdb`:
entities, beliefs, versions) — no setup, fine for a laptop or a test. Both
engines run the same SQL (`app/db.py` renders it per engine) and the same test
suite. There is no automatic copy from DuckDB to Postgres: a new Postgres
instance starts empty.

## Access: API key

```bash
ETCHMEM_API_KEY=etm_...            # comma-separate several keys to rotate
```

With a key set, every REST and MCP call needs `Authorization: Bearer <key>` or
`X-API-Key: <key>`; anything else gets `401`. `/health` stays open for probes,
and the `/ui` page loads and asks for the key once per browser tab. Without a
key etchmem is open — keep it on localhost, or behind a gateway that is the
only thing allowed to reach it.

## Scopes: topics, not permissions

Every signal has a `scope`: the knowledge area it belongs to, e.g. the skill that
produced or needs it (`customer-service-triage`, `spare-parts`). Reads can be narrowed
to one scope or several, so an agent running a skill gets focused knowledge:

```bash
POST /v1/recall  {"query": "Acme renewal", "scope": ["spare-parts", "warranty"]}
GET  /v1/etches?scope=spare-parts&scope=warranty       # or ?scope=spare-parts,warranty
GET  /v1/entity/Acme/etches?scope=spare-parts
```

Omit `scope` for every scope. Scopes say nothing about who may read what: that is
the job of writers (below).

Subjects are resolved **within** a scope: "Acme" in two scopes is two entities, and
`know` merges them when it reads several scopes. Databases created before 0.4 may hold
an entity that collected facts from more than one scope; those keep working.

## Writers: who may read what

Every signal has one writer, its `source`: the agent that deposited it, an import job,
or a source system. A belief records every writer behind it. Reads take `writers` and
return only beliefs **all** of whose writers are in the list:

```bash
POST /v1/recall  {"query": "Acme", "writers": ["mossy", "zendesk"]}
GET  /v1/entity/Acme/etches?writers=mossy&writers=zendesk
GET  /v1/etch/company_acme::tier/dossier?writers=mossy        # 404 if not readable
```

- A belief **all** of whose writers are on the list is returned in full.
- A belief that other writers also contributed to is returned only if a listed writer
  itself states its current value, and then **reduced** to that: the value, the listed
  writers as `source`, and a quote from their own signals, without the wording, evidence,
  other values or names the other writers added. Two agents shown the same ticket each
  see it; a fact only another writer's notes support stays hidden.
- `dossier` and `history` need every writer on the list.

Without `writers` there is no filter. etchmem does not control access: it applies the list it
is given. A gateway such as Myco decides that list from the caller's permissions
(`memory:<writer>:read`), and stamps the calling agent as the writer of what it deposits.

## Provenance on every signal

A signal's identity is its writer plus its text: the same words from the same writer
again are a duplicate, while the same words from two writers (two agents shown the same
ticket) are two signals, one in each writer's memory. Ids from before 0.4 had no writer
in them, so a database upgraded from 0.3 may store a re-deposited text once more; an
erasure recorded under an old id still refuses it.

`remember` also takes:

- `source_record_id` — the record it came from, e.g. `zendesk:ticket:123`, so a
  deletion in the source system can be followed here.
- `actor` — who deposited it, e.g. `{"agent": "support-bot", "user": "anna@acme.com"}`.

## Erasure (GDPR Art. 17)

```bash
POST /v1/forget  {"entity": "Maria Muster", "reason": "DSR-2026-114"}
POST /v1/forget  {"source_record_ids": ["zendesk:ticket:123"]}
POST /v1/forget  {"signal_ids": ["c554…"]}
```

1. **Signals** — the ones named, the ones from those source records, and for a
   subject every signal that cites it as evidence *or names it in its text*
   (including signals not extracted yet) are deleted. The name is matched in the text
   even when memory never made it a subject (a person who is only a value in someone
   else's belief, "primary contact = Maria Muster"): `signals_matched_by_name` counts
   those, and when nothing at all matched the report says so in `note`. A signal is erased whole:
   other facts it carried lose that evidence too.
2. **Claims** lose the erased signals as evidence; a claim left with none, and
   every claim about the subject or pointing at it, is deleted.
3. **Beliefs** — the subject's own, and any left without claims, are deleted with
   all their versions. A belief that lost only part of its evidence keeps its
   timeline, but every stored version is blanked to `[erased]` (when it changed
   stays, what it said goes), its head is blanked at once, and it is
   re-consolidated from what remains (`refold: "done"`, or `"pending"` until the
   next worker tick if no model is reachable).
4. **Tombstone** — the report is kept (`GET /v1/erasures`) with counts and the
   erased signals' ids (content hashes), never the text or the subject's name. A
   signal with the same content is refused afterwards (`status: "erased"`), and
   the tombstones let you re-apply erasures after restoring a backup.

A name stands for every entity it resolves to, in every scope. Erasure is
REST-only on purpose; no MCP tool can delete memory.

## API

All paths live under `/v1` (the same paths without the prefix still answer, for
existing clients).

| Endpoint | Purpose |
|----------|---------|
| `POST /remember` | Deposit a raw signal (`source`, `scope`, `extract_mode`, `occurred_at`, `context`, `context_parent`, `source_record_id`, `actor`). |
| `POST /recall` | Semantic recall over beliefs; `scope` takes a list; `as_of` for time-travel. |
| `POST /sleep` | Run one worker tick now (batch → extract → fold). |
| `POST /export` | Dump all etches to JSON files. |
| `POST /associate` | What the memory holds for a phrase: matching beliefs, their subjects' facts, and connected nodes. |
| `GET /entity/{ref}/etches` | **Every** belief about one subject — exhaustive, not ranked. |
| `GET /etches` | **Every** belief matching filters — `scope`, `property`, `entity_type` (repeat or comma-separate), `status`, `min_confidence`; paged with `limit`/`offset`, `total` counts all matches. Writes nothing to disk. |
| `GET /etch/{id}/history` | Version timeline of one belief. |
| `GET /etch/{id}/dossier` | Full provenance: etch + versions + claims + source signals. |
| `GET /signal/{id}` | What one signal became: status, context, focus, subjects, associations, claims. |
| `GET /context/{id}` | One conversation: its focus and its signals in order. |
| `POST /forget` | Erase a subject, source records or signals; returns the erasure report. |
| `GET /erasures` | Erasure tombstones: id, time, reason, counts. No personal data. |
| `GET /stats` | Queue depths + counts (signals/claims/entities/etches/contested). |
| `GET /health` | Liveness + active config (database engine, whether a key is required). Never needs a key. |
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
`know`, `etches`, `associate`, `sleep`, `export`, `stats`, `etch_history`, `dossier`, `signal`, `context` — over streamable HTTP at
`http://localhost:8000/v1/mcp` (also `/mcp`; stateless, JSON responses). One
process serves both protocols, so Claude Code / Claude Desktop / any MCP client
can use etchmem as a memory backend directly:

```json
{
  "mcpServers": {
    "etchmem": {
      "type": "http",
      "url": "http://localhost:8000/v1/mcp",
      "headers": { "Authorization": "Bearer etm_..." }
    }
  }
}
```

Drop `headers` when no `ETCHMEM_API_KEY` is set. There is no `forget` tool.

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
cp .env.example .env          # add your OPENAI_API_KEY (and an ETCHMEM_API_KEY)
docker compose up --build
```

The compose file runs PostgreSQL with pgvector next to etchmem; its data
persists in the `etchmem-postgres` volume, exports in `etchmem-data`. To run
on DuckDB instead, set `ETCHMEM_DATABASE_URL=` (empty) and drop the `postgres`
service.

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export ETCHMEM_DATA_DIR=./data       # DuckDB, or set ETCHMEM_DATABASE_URL for Postgres
uvicorn app.main:app --reload
```

## Configuration

Via environment / `.env` (see `.env.example`). Highlights:

- `ETCHMEM_DATABASE_URL` — PostgreSQL + pgvector; empty = DuckDB files in
  `ETCHMEM_DATA_DIR`. `ETCHMEM_DATABASE_SCHEMA` — this instance's schema.
- `ETCHMEM_API_KEY` — require a key on every REST and MCP call (comma-separated
  for rotation). Empty = open.

- `ETCHMEM_CLAIM_MODEL` / `ETCHMEM_ETCH_MODEL` — cascade model strings.
- `ETCHMEM_RELAX_CHAT_METADATA` (false) — for an OpenAI-compatible gateway whose
  chat replies add a `metadata` object the SDK rejects (values must be strings).
  One switch wraps every `openai:` model; a real string map is kept. Point
  `OPENAI_BASE_URL` at the gateway. Other providers are unchanged.
- `EMBEDDING_PROVIDER` — `openai` (default), `local` (fastembed/ONNX; needs
  `requirements-local.txt`), or `fake` (tests).
- `LOCAL_EMBEDDING_MODEL` — default
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Weights cache
  under `$ETCHMEM_DATA_DIR/fastembed`.
- `ETCHMEM_SIGNAL_DEDUP_DISTANCE`, `ETCHMEM_ENTITY_SIM_THRESHOLD` — dedup and
  entity-merge thresholds.
- `ETCHMEM_MULTI_VALUE_PROPERTIES` — properties that union instead of conflict.
- `ETCHMEM_EXT_DIR` — folder of claim-extension YAML files (see above).
- `ETCHMEM_SOURCE_TRUST_JSON` / `ETCHMEM_TRUST_GAP` — trust-based conflict
  resolution: rank your sources, let policy settle disagreements.
- `ETCHMEM_RETRACTION_WEIGHT` (0.1), `ETCHMEM_AGREEMENT_HALF_LIFE_DAYS` (14),
  `ETCHMEM_TRUST_SAME_ODDS_RATIO` (3) — how beliefs change over time (see above).
- `ETCHMEM_RESOLVE_MODEL` / `ETCHMEM_VERIFY_MODEL` — models for subject
  resolution and claim verification (empty = `ETCHMEM_CLAIM_MODEL`). Give the
  resolve step your top model: small models invent subjects like "the link"
  instead of resolving them (measured).
- `ETCHMEM_SUBJECT_CANDIDATES` (10) — similar subjects offered per signal, on
  top of the conversation focus and what is linked to it.
  `ETCHMEM_EXTRACT_KNOWN_LABELS` (10) — labels per subject and property shown
  to the verify step. `ETCHMEM_VERIFY_CLAIMS` (true).
- `ETCHMEM_CONTEXT_WINDOW_SECONDS` (7200) — how long a channel's focus carries
  over into a new thread.
- `ETCHMEM_EXTRACT_LLM_BATCH_SIZE` (6) / `ETCHMEM_EXTRACT_CONCURRENCY` (4) —
  signals per model call and parallel calls. Use 1 with small models: several
  signals in one prompt get their subjects and facts mixed.
- `ETCHMEM_CLAIMS_ANONYMIZATION` — anonymize personal data in claims/etches
  (see above). Default `false`.
- `ETCHMEM_DETAIL_DEFAULT` (true) — every claim may keep the author's full
  wording; `false` keeps it only for properties declared `detail: true`.
- `ETCHMEM_DETAIL_MAX_CHARS` (400) — longest wording kept per claim; the
  extractor condenses to it and a longer one is cut when stored. 0 = no limit.
- `ETCHMEM_QUOTE_CHARS` (300) — length of a quote's excerpt; 0 turns quotes off.
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
  db.py         one SQL surface over PostgreSQL + pgvector or DuckDB
  stores.py     LeftStore (signals, claims) + RightStore (entities, etches, versions, erasures)
  dedup.py      Stage 1 — embedding dedup
  ext.py        claim extensions: YAML domain vocabulary → prompt + enum enforcement
  agents.py     Stage 2 claim_agent + Stage 3 conflict resolver (Pydantic AI)
  entities.py   entity resolution (find-or-create, alias, fuzzy)
  gate.py       deterministic routing gate + resolution policy + confidence
  anonymize.py  claims anonymization: pseudonym labels, prompt blocks, regex scrub
  worker.py     Pipeline (batch/extract/fold) + in-process WorkerLoop
  service.py    wires everything; remember/recall/sleep/export/stats/history/forget
  mcp_server.py MCP tools (FastMCP, streamable HTTP; mounted at /v1/mcp and /mcp)
  main.py       FastAPI app + /v1 routes + API-key guard + MCP mount + worker lifespan
tests/
  test_integration.py    offline end-to-end (fake embedder + stub agents)
  test_anonymization.py  pseudonym consistency, regex scrub, signal exclusion
  test_mcp.py            tool registration, /mcp mount, tool ↔ service parity
  test_gateway_api.py    API key, /v1, signal metadata, scopes, forget
```

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q       # fully offline on DuckDB; no API key needed

# the same suite on PostgreSQL + pgvector (one schema per test, dropped after)
docker run -d --rm -p 55432:5432 -e POSTGRES_USER=etchmem -e POSTGRES_PASSWORD=etchmem \
    pgvector/pgvector:pg17
ETCHMEM_TEST_DATABASE_URL=postgresql://etchmem:etchmem@localhost:55432/etchmem pytest -q
```

Covered: entity-boundary (no contamination), corroboration raising confidence,
recency supersession + versioning, genuine conflict → contested (LLM invoked),
time-travel recall, claims anonymization, the MCP interface, and retraction,
agreement half-life, comparable trust and extraction context
(`tests/test_reconsolidation.py`).

## Production deployments

etchmem is open source — clone it, ship it, never talk to us. If you want it
integrated into your agent stack faster (signal capture design, consolidation
policy tuning, scoped knowledge across teams, recall wiring), we do
fixed-scope implementations: [etchmem.io](https://etchmem.io).

## Confidence: who said it, and how many independent witnesses

A settled etch's confidence is built from evidence, deterministically and without
an LLM (`app/gate.py`):

```python
# per source: its declared trust; repeats from the SAME source count decay^(k-1)
effective  = (1 - decay**n) / (1 - decay)            # n signals from this source
belief_s   = 1 - (1 - trust_s) ** effective
strength   = 1 - prod(1 - belief_s)                  # independent sources, noisy-OR
confidence = clamp(agreement * strength * penalty, 0.05, 0.99)
```

`agreement` and `penalty` are unchanged (recency 0.85, source-trust win 0.9,
contested fallback 0.5). `decay` is `ETCHMEM_EVIDENCE_REPEAT_DECAY` (default 0.5).

| evidence | confidence |
|---|---|
| 1 signal, source trust 0.9 | 0.90 |
| 1 signal, source nobody declared (default 0.5) | 0.50 |
| 2 independent sources at 0.5 | 0.75 |
| 3 signals from one 0.5 source | 0.70 |

One deposit that yields several labels is one witness, not several (evidence is
counted by signal id). A source whose signals were purged by the TTL still counts
once, so confidence does not collapse when old signals expire.

### Declaring source trust

Trust is the domain owner's call, declared next to the vocabulary it governs, and
never supplied by the depositing caller or the model:

```yaml
# ext/care.yaml
sources:
  kb.article:         {trust: 0.9, description: published knowledge-base article}
  ticket.agent_closed: 0.8                   # short form
  chat.bot:           {trust: 0.4}
```

The keys are the `source` strings sent with `remember`. A source nobody declared gets
`ETCHMEM_DEFAULT_SOURCE_TRUST` (0.5, the same value a single source scored before).
`ETCHMEM_SOURCE_TRUST_JSON` still works as an override on top of the YAML. `/stats`
lists `sources_without_trust`: deposit sources seen but not declared, so a new channel
does not quietly sit at the default. Renaming a source in the depositor resets its trust.

### Evidence in the API

`recall`, `know`, `etches`, `export`, the dossier and `/history` return an `evidence`
block (and the per-value `values` list) next to `confidence`, with the trust that was declared when the belief was folded:

```json
{"confidence": 0.9,
 "evidence": {"signals": 1, "max_trust": 0.9,
              "sources": [{"source": "slack.reject", "trust": 0.9, "signals": 1}]}}
```

A consumer can use the number, or apply its own rule on the evidence ("only inject
lessons backed by a source above 0.8").

### TODO

- Existing etches keep their old confidence and have no `evidence` until their
  (entity, property) pair is next folded; add a recompute endpoint if that matters.
- The extractor's self-reported claim `confidence` is stored but not used: an LLM
  rating its own certainty is poorly calibrated. Revisit with a calibration set.
- etchmem does not authenticate sources: a caller that can reach `remember` can
  use any `source` string, including a trusted one.
