"""
Stores — two layers of memory, on PostgreSQL + pgvector or embedded DuckDB.

  LeftStore
      signals : raw deposits (status lifecycle: new→batched→extracted)
      claims  : structured assertions extracted from signals (corroboration-aware)

  RightStore
      entities      : canonical entity registry (+ aliases, embedding)
      etches        : current belief per (entity, property)  [HEAD]
      etch_versions : immutable snapshots of every belief change
      erasures      : tombstones of erasure requests (no personal data)

The engine comes from `app.db`: Postgres when ETCHMEM_DATABASE_URL is set (one
database, optionally one schema per instance), else two DuckDB files in
ETCHMEM_DATA_DIR. SQL is written once with `?` placeholders and the tokens
`app.db` renders. Semantic search orders by cosine distance, which pgvector's
HNSW index serves. A per-store lock keeps one connection safe to share across
the API process and the worker task.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.db import Db, connect

# Signal status lifecycle (the in-process "queue" contract).
S_NEW = "new"
S_BATCHED = "batched"
S_EXTRACTED = "extracted"

# Claim status.
C_NEW = "new"
C_CONSOLIDATED = "consolidated"

# What an erased belief version reads afterwards.
ERASED = "[erased]"


# ── Row dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    id: str
    content: str
    source: str
    scope: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    # When the fact actually happened, as declared by the caller. 0.0 = not
    # declared, in which case `event_at` falls back to ingest time. Kept apart
    # from created_at so the audit trail still shows when we received it.
    occurred_at: float = 0.0
    expires_at: float = 0.0
    status: str = S_NEW
    extract_mode: str = "deferred"          # "immediate" | "deferred"
    canonical_id: str | None = None         # representative signal for its dup cluster
    embedding: list[float] | None = None
    # Where it was said, as the client addresses it: the conversation (a Slack
    # thread, a ticket's comments, a meeting) and the one around it (the channel).
    # Plain ids from the client; etchmem never interprets them.
    context_id: str | None = None
    context_parent: str | None = None
    # The record this signal came from in its source system (e.g.
    # "zendesk:ticket:123"), so a deletion there can be followed here.
    source_record_id: str | None = None
    # Who deposited it: the agent and, when known, the person it acted for.
    actor: dict[str, Any] = field(default_factory=dict)

    @property
    def event_at(self) -> float:
        """When this signal's content happened: declared time, else ingest."""
        return self.occurred_at or self.created_at


@dataclass
class Claim:
    id: str
    entity_id: str
    entity_name: str
    property: str
    value: str
    value_norm: str
    polarity: str = "asserted"              # "asserted" | "negated"
    event_time: float = 0.0
    ingest_time: float = 0.0
    sources: list[str] = field(default_factory=list)
    evidence_signal_ids: list[str] = field(default_factory=list)
    corroboration_count: int = 1
    confidence: float = 0.5
    value_entity_id: str | None = None      # set when `property` is a relation
    # Full wording behind the label in `value`, for properties declared
    # `detail: true`. Not part of the claim's identity.
    detail: str | None = None
    scope: str | None = None
    status: str = C_NEW
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class Entity:
    id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    scope: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    embedding: list[float] | None = None


@dataclass
class Etch:
    id: str                                  # entity_id::property
    entity_id: str
    entity_name: str
    property: str
    current_value: str
    status: str                              # "settled" | "contested"
    confidence: float
    narrative: str
    version: int
    scope: str | None = None
    source: str | None = None
    claim_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    embedding: list[float] | None = None
    # Set when `property` is a declared relation: the entity this belief points
    # at. Turns the etch from an edge into a literal into a real graph edge.
    value_entity_id: str | None = None
    # [{"value": label, "detail": full wording}], for `detail: true` properties.
    # `current_value` stays the label(s); this carries the wording.
    details: list[dict] = field(default_factory=list)
    # What the confidence rests on: {"signals", "sources": [{source, trust,
    # signals}], "max_trust"}. Computed at fold time with the trust declared then.
    evidence: dict = field(default_factory=dict)
    # One entry per value ever asserted for this (entity, property), heaviest
    # first: [{"value", "weight", "active", "last_asserted_at", "retracted_at"}].
    # A retracted value is not deleted: it stays here with a low weight, so the
    # memory knows both that it is no longer valid and that it once was.
    values: list[dict] = field(default_factory=list)


@dataclass
class Hit:
    id: str
    content: str
    similarity: float
    row: dict[str, Any]


def _j(x: Any) -> str:
    return json.dumps(x)


def _u(s: str | None, default: Any) -> Any:
    return json.loads(s) if s else default


def _vec(v: Any) -> list[float] | None:
    """An embedding as read back: a list from DuckDB, pgvector text from Postgres."""
    if v is None:
        return None
    if isinstance(v, str):
        return [float(x) for x in json.loads(v)]
    return [float(x) for x in v]


def _open(db: Db | str, dim: int | None) -> Db:
    if isinstance(db, Db):
        return db
    if dim is None:
        raise TypeError("a DuckDB path needs the embedding dimension")
    return connect(database_url=None, path=db, dim=dim)


def _in(column: str, values: Sequence[Any]) -> tuple[str, list[Any]]:
    """`column IN (?, ...)`, or a clause that matches nothing for an empty list."""
    if not values:
        return "1 = 0", []
    return f"{column} IN ({', '.join('?' * len(values))})", list(values)


def scope_clause(column: str, scope: str | Sequence[str] | None) -> tuple[str | None, list[Any]]:
    """Filter by one scope or any of several. None = every scope; [] = none."""
    if scope is None:
        return None, []
    if isinstance(scope, str):
        return (f"{column} = ?", [scope]) if scope else (None, [])
    return _in(column, list(scope))


_SIGNAL_COLS = """id, content, source, scope, metadata, embedding, created_at,
                 occurred_at, expires_at, status, extract_mode, canonical_id,
                 context_id, context_parent, source_record_id, actor"""
_N_SIGNAL_COLS = len(_SIGNAL_COLS.split(","))
_CLAIM_COLS = """id, entity_id, entity_name, property, value, value_norm,
                polarity, event_time, ingest_time, sources,
                evidence_signal_ids, corroboration_count, confidence, scope,
                status, created_at, updated_at, value_entity_id, detail"""

_ETCH_COLS = """id, entity_id, entity_name, property, current_value, status,
                confidence, narrative, version, scope, source, claim_ids,
                source_ids, created_at, updated_at, embedding, value_entity_id,
                details, evidence, value_weights"""
_ENTITY_COLS = "id, name, type, aliases, scope, created_at, updated_at, embedding"


# ── Left store ───────────────────────────────────────────────────────────────

class LeftStore:
    def __init__(self, db: Db | str, dim: int | None = None) -> None:
        # A path (with its dim) opens a DuckDB file, as before the engine choice.
        self._db = _open(db, dim)
        self._lock = self._db.lock
        self._init_schema()

    def _x(self, sql: str, params: Sequence[Any] = ()):
        return self._db.execute(sql, params)

    def _init_schema(self) -> None:
        with self._lock:
            self._x("""
                CREATE TABLE IF NOT EXISTS signals (
                    id            VARCHAR PRIMARY KEY,
                    content       VARCHAR,
                    source        VARCHAR,
                    scope         VARCHAR,
                    metadata      VARCHAR,
                    embedding     {VECTOR},
                    created_at    {DOUBLE},
                    occurred_at   {DOUBLE},
                    expires_at    {DOUBLE},
                    status        VARCHAR,
                    extract_mode  VARCHAR,
                    canonical_id  VARCHAR,
                    context_id    VARCHAR,
                    context_parent VARCHAR,
                    source_record_id VARCHAR,
                    actor         VARCHAR
                )""")
            self._x("""
                CREATE TABLE IF NOT EXISTS claims (
                    id                  VARCHAR PRIMARY KEY,
                    entity_id           VARCHAR,
                    entity_name         VARCHAR,
                    property            VARCHAR,
                    value               VARCHAR,
                    value_norm          VARCHAR,
                    polarity            VARCHAR,
                    event_time          {DOUBLE},
                    ingest_time         {DOUBLE},
                    sources             VARCHAR,
                    evidence_signal_ids VARCHAR,
                    corroboration_count INTEGER,
                    confidence          {DOUBLE},
                    value_entity_id     VARCHAR,
                    detail              VARCHAR,
                    scope               VARCHAR,
                    status              VARCHAR,
                    created_at          {DOUBLE},
                    updated_at          {DOUBLE}
                )""")
            self._migrate()
            for sql in (
                "CREATE INDEX IF NOT EXISTS signals_status_idx ON signals (status)",
                "CREATE INDEX IF NOT EXISTS signals_canonical_idx ON signals (canonical_id)",
                "CREATE INDEX IF NOT EXISTS signals_record_idx ON signals (source_record_id)",
                "CREATE INDEX IF NOT EXISTS claims_pair_idx ON claims (entity_id, property)",
                "CREATE INDEX IF NOT EXISTS claims_status_idx ON claims (status)",
            ):
                self._x(sql)
            self._db.check_vector_dim("signals", "embedding")
            self._db.vector_index("signals", "embedding")

    def _migrate(self) -> None:
        """Additive migrations for databases created by an earlier version.

        Called with the lock held. Adding occurred_at to an existing signals
        table backfills it from created_at, which preserves the old behaviour
        exactly: undeclared event time means ingest time.
        """
        cols = self._db.columns("signals")
        if "occurred_at" not in cols:
            self._x("ALTER TABLE signals ADD COLUMN occurred_at {DOUBLE}")
            self._x("UPDATE signals SET occurred_at = created_at WHERE occurred_at IS NULL")
        for col in ("context_id", "context_parent", "source_record_id", "actor"):
            if col not in cols:
                self._x(f"ALTER TABLE signals ADD COLUMN {col} VARCHAR")
        ccols = self._db.columns("claims")
        for col in ("value_entity_id", "detail"):
            if col not in ccols:
                self._x(f"ALTER TABLE claims ADD COLUMN {col} VARCHAR")

    # ── signals ──────────────────────────────────────────────────────────

    def add_signal(self, sig: Signal) -> bool:
        with self._lock:
            if self._x("SELECT 1 FROM signals WHERE id = ?", [sig.id]).fetchone():
                return False
            self._x(
                """
                INSERT INTO signals (id, content, source, scope, metadata, embedding,
                    created_at, occurred_at, expires_at, status, extract_mode,
                    canonical_id, context_id, context_parent, source_record_id,
                    actor)
                VALUES (?, ?, ?, ?, ?, {VEC}, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [sig.id, sig.content, sig.source, sig.scope, _j(sig.metadata),
                 self._db.vec(sig.embedding), sig.created_at, sig.occurred_at,
                 sig.expires_at, sig.status, sig.extract_mode, sig.canonical_id,
                 sig.context_id, sig.context_parent, sig.source_record_id,
                 _j(sig.actor)],
            )
            return True

    def signals_by_status(self, status: str) -> list[Signal]:
        with self._lock:
            rows = self._x(
                f"""SELECT {_SIGNAL_COLS}
                   FROM signals WHERE status = ? ORDER BY created_at""",
                [status],
            ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def signals_with_canonical(self, canonical_id: str) -> list[Signal]:
        with self._lock:
            rows = self._x(
                f"""SELECT {_SIGNAL_COLS}
                   FROM signals WHERE canonical_id = ?""",
                [canonical_id],
            ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def set_signal_status(self, ids: list[str], status: str) -> None:
        if not ids:
            return
        with self._lock:
            self._db.executemany(
                "UPDATE signals SET status = ? WHERE id = ?",
                [[status, i] for i in ids],
            )

    def set_canonical(self, signal_id: str, canonical_id: str, status: str) -> None:
        with self._lock:
            self._x(
                "UPDATE signals SET canonical_id = ?, status = ? WHERE id = ?",
                [canonical_id, status, signal_id],
            )

    def search_signals(self, qvec, top_k, scope=None, source=None) -> list[Hit]:
        clauses, params = ["embedding IS NOT NULL"], []
        sc, sp = scope_clause("scope", scope)
        if sc:
            clauses.append(sc); params.extend(sp)
        if source:
            clauses.append("source = ?"); params.append(source)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._x(
                f"""SELECT id, content, source, scope, created_at,
                           {{DIST:embedding}} AS dist
                    FROM signals WHERE {where} ORDER BY dist LIMIT ?""",
                [self._db.vec(qvec), *params, top_k],
            ).fetchall()
        return [Hit(r[0], r[1], 1.0 - float(r[5]),
                    {"source": r[2], "scope": r[3], "created_at": r[4]}) for r in rows]

    def expire(self, now: float | None = None) -> int:
        now = now or time.time()
        with self._lock:
            before = self._x("SELECT count(*) FROM signals").fetchone()[0]
            self._x("DELETE FROM signals WHERE expires_at > 0 AND expires_at < ?", [now])
            after = self._x("SELECT count(*) FROM signals").fetchone()[0]
        return int(before - after)

    # ── claims ───────────────────────────────────────────────────────────

    def upsert_claim(self, claim: Claim) -> None:
        """Insert a claim, or merge corroboration into an existing one."""
        with self._lock:
            existing = self._x(
                """SELECT sources, evidence_signal_ids, corroboration_count, event_time,
                          detail
                   FROM claims WHERE id = ?""", [claim.id]).fetchone()
            if existing:
                sources = sorted(set(_u(existing[0], []) + claim.sources))
                evidence = sorted(set(_u(existing[1], []) + claim.evidence_signal_ids))
                count = len(evidence)
                event_time = max(existing[3] or 0.0, claim.event_time)
                # Same label, possibly new wording: the newest wording wins. An
                # older signal arriving late never overwrites a newer detail.
                detail = existing[4]
                if claim.detail and (not detail or claim.event_time >= (existing[3] or 0.0)):
                    detail = claim.detail
                self._x(
                    """UPDATE claims SET sources = ?, evidence_signal_ids = ?,
                       corroboration_count = ?, event_time = ?, detail = ?, status = ?,
                       updated_at = ? WHERE id = ?""",
                    [_j(sources), _j(evidence), count, event_time, detail, C_NEW,
                     time.time(), claim.id])
            else:
                self._x(
                    """INSERT INTO claims (id, entity_id, entity_name, property, value,
                        value_norm, polarity, event_time, ingest_time, sources,
                        evidence_signal_ids, corroboration_count, confidence, scope,
                        status, created_at, updated_at, value_entity_id, detail)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [claim.id, claim.entity_id, claim.entity_name, claim.property,
                     claim.value, claim.value_norm, claim.polarity, claim.event_time,
                     claim.ingest_time, _j(claim.sources), _j(claim.evidence_signal_ids),
                     claim.corroboration_count, claim.confidence, claim.scope,
                     claim.status, claim.created_at, claim.updated_at,
                     claim.value_entity_id, claim.detail])

    def pairs_with_new_claims(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._x(
                """SELECT DISTINCT entity_id, property FROM claims WHERE status = ?""",
                [C_NEW]).fetchall()
        return [(r[0], r[1]) for r in rows]

    def claims_for_pair(self, entity_id: str, prop: str) -> list[Claim]:
        with self._lock:
            rows = self._x(
                f"""SELECT {_CLAIM_COLS}
                   FROM claims WHERE entity_id = ? AND property = ?""",
                [entity_id, prop]).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def claims_by_ids(self, ids: list[str]) -> list[Claim]:
        if not ids:
            return []
        clause, params = _in("id", ids)
        with self._lock:
            rows = self._x(f"SELECT {_CLAIM_COLS} FROM claims WHERE {clause}", params).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def signals_by_ids(self, ids: list[str]) -> list[Signal]:
        if not ids:
            return []
        clause, params = _in("id", ids)
        with self._lock:
            rows = self._x(
                f"""SELECT {_SIGNAL_COLS}
                    FROM signals WHERE {clause}
                    ORDER BY created_at""", params).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def signals_in_context(self, *, context_id: str | None = None,
                           context_parent: str | None = None, before: float,
                           since: float = 0.0, limit: int = 5) -> list[Signal]:
        """The latest signals said in one conversation (or around it), before a time.
        Event time orders them: when it was said, not when it arrived."""
        col, val = ("context_id", context_id) if context_id else ("context_parent", context_parent)
        if not val:
            return []
        # Postgres has no float infinity literal in a bound parameter's place
        # here; "no upper bound" is simply a very large timestamp.
        before = min(before, 1e18)
        with self._lock:
            rows = self._x(
                f"""SELECT {_SIGNAL_COLS} FROM signals
                    WHERE {col} = ?
                      AND COALESCE(NULLIF(occurred_at, 0), created_at) < ?
                      AND COALESCE(NULLIF(occurred_at, 0), created_at) >= ?
                    ORDER BY COALESCE(NULLIF(occurred_at, 0), created_at) DESC LIMIT ?""",
                [val, before, since, limit]).fetchall()
        return list(reversed([self._row_to_signal(r) for r in rows]))

    def similar_signals(self, qvec, *, scope: str | None, top_k: int = 10,
                        exclude: str | None = None) -> list[tuple[Signal, float]]:
        """Already-extracted signals most similar to this one (by embedding)."""
        clauses, params = ["embedding IS NOT NULL", "status = 'extracted'"], []
        if scope:
            clauses.append("scope = ?"); params.append(scope)
        if exclude:
            clauses.append("id <> ?"); params.append(exclude)
        with self._lock:
            rows = self._x(
                f"""SELECT {_SIGNAL_COLS}, {{DIST:embedding}} AS dist
                    FROM signals WHERE {' AND '.join(clauses)}
                    ORDER BY dist LIMIT ?""", [self._db.vec(qvec), *params, top_k]).fetchall()
        return [(self._row_to_signal(r[:_N_SIGNAL_COLS]), 1.0 - float(r[_N_SIGNAL_COLS]))
                for r in rows]

    def get_signal(self, signal_id: str) -> Signal | None:
        found = self.signals_by_ids([signal_id])
        return found[0] if found else None

    def claims_for_signal(self, signal_id: str) -> list[Claim]:
        """Every claim this signal is evidence for (including as a duplicate)."""
        with self._lock:
            rows = self._x(
                f"""SELECT {_CLAIM_COLS} FROM claims
                    WHERE evidence_signal_ids LIKE '%' || ? || '%'
                    ORDER BY created_at""", [signal_id]).fetchall()
        return [c for c in (self._row_to_claim(r) for r in rows)
                if signal_id in c.evidence_signal_ids]

    def update_signal_metadata(self, signal_id: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._x("UPDATE signals SET metadata = ? WHERE id = ?",
                    [_j(metadata), signal_id])

    def set_claims_status(self, ids: list[str], status: str) -> None:
        if not ids:
            return
        with self._lock:
            self._db.executemany(
                "UPDATE claims SET status = ? WHERE id = ?", [[status, i] for i in ids])

    # ── erasure ──────────────────────────────────────────────────────────

    def signal_ids_for_records(self, record_ids: list[str]) -> list[str]:
        clause, params = _in("source_record_id", record_ids)
        with self._lock:
            rows = self._x(f"SELECT id FROM signals WHERE {clause}", params).fetchall()
        return [r[0] for r in rows]

    def signal_ids_mentioning(self, names: list[str]) -> list[str]:
        """Signals whose raw text contains any of these names (case-insensitive)."""
        names = [n for n in dict.fromkeys(n.strip() for n in names) if len(n) >= 3]
        if not names:
            return []
        # strpos, not LIKE: a name with "_" or "%" must match literally.
        clause = " OR ".join("strpos(lower(content), ?) > 0" for _ in names)
        with self._lock:
            rows = self._x(f"SELECT id FROM signals WHERE {clause}",
                           [n.lower() for n in names]).fetchall()
        return [r[0] for r in rows]

    def claims_about(self, entity_id: str) -> list[Claim]:
        """Claims whose subject or whose value is this entity."""
        with self._lock:
            rows = self._x(
                f"""SELECT {_CLAIM_COLS} FROM claims
                    WHERE entity_id = ? OR value_entity_id = ?""",
                [entity_id, entity_id]).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def claims_citing(self, signal_ids: list[str]) -> list[Claim]:
        """Claims that cite any of these signals as evidence."""
        out: dict[str, Claim] = {}
        for sid in dict.fromkeys(signal_ids):
            for c in self.claims_for_signal(sid):
                out[c.id] = c
        return list(out.values())

    def delete_claims(self, ids: list[str]) -> int:
        if not ids:
            return 0
        clause, params = _in("id", ids)
        with self._lock:
            self._x(f"DELETE FROM claims WHERE {clause}", params)
        return len(ids)

    def replace_claim_evidence(self, claim_id: str, evidence: list[str],
                               sources: list[str]) -> None:
        """What a claim rests on after some of its signals were erased; it will
        be folded again."""
        with self._lock:
            self._x(
                """UPDATE claims SET evidence_signal_ids = ?, sources = ?,
                   corroboration_count = ?, status = ?, updated_at = ?
                   WHERE id = ?""",
                [_j(sorted(evidence)), _j(sorted(sources)), len(evidence),
                 C_NEW, time.time(), claim_id])

    def delete_signals(self, ids: list[str]) -> int:
        if not ids:
            return 0
        clause, params = _in("id", ids)
        with self._lock:
            self._x(f"DELETE FROM signals WHERE {clause}", params)
        return len(ids)

    # ── counts ───────────────────────────────────────────────────────────

    def count_signals(self, status: str | None = None) -> int:
        with self._lock:
            if status:
                return int(self._x(
                    "SELECT count(*) FROM signals WHERE status = ?", [status]).fetchone()[0])
            return int(self._x("SELECT count(*) FROM signals").fetchone()[0])

    def count_claims(self) -> int:
        with self._lock:
            return int(self._x("SELECT count(*) FROM claims").fetchone()[0])

    def scopes(self) -> list[str]:
        with self._lock:
            rows = self._x(
                "SELECT DISTINCT scope FROM signals WHERE scope IS NOT NULL").fetchall()
        return sorted(r[0] for r in rows)

    def sources(self) -> list[str]:
        with self._lock:
            rows = self._x(
                "SELECT DISTINCT source FROM signals WHERE source IS NOT NULL").fetchall()
        return sorted(r[0] for r in rows)

    # ── row mappers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_signal(r) -> Signal:
        return Signal(
            id=r[0], content=r[1], source=r[2], scope=r[3], metadata=_u(r[4], {}),
            embedding=_vec(r[5]), created_at=r[6],
            occurred_at=r[7] or 0.0, expires_at=r[8], status=r[9],
            extract_mode=r[10], canonical_id=r[11], context_id=r[12],
            context_parent=r[13], source_record_id=r[14], actor=_u(r[15], {}))

    @staticmethod
    def _row_to_claim(r) -> Claim:
        return Claim(
            id=r[0], entity_id=r[1], entity_name=r[2], property=r[3], value=r[4],
            value_norm=r[5], polarity=r[6], event_time=r[7], ingest_time=r[8],
            sources=_u(r[9], []), evidence_signal_ids=_u(r[10], []),
            corroboration_count=r[11], confidence=r[12], scope=r[13], status=r[14],
            created_at=r[15], updated_at=r[16], value_entity_id=r[17], detail=r[18])


# ── Right store ──────────────────────────────────────────────────────────────

class RightStore:
    def __init__(self, db: Db | str, dim: int | None = None) -> None:
        # A path (with its dim) opens a DuckDB file, as before the engine choice.
        self._db = _open(db, dim)
        self._lock = self._db.lock
        self._init_schema()

    def _x(self, sql: str, params: Sequence[Any] = ()):
        return self._db.execute(sql, params)

    def _init_schema(self) -> None:
        with self._lock:
            self._x("""
                CREATE TABLE IF NOT EXISTS entities (
                    id          VARCHAR PRIMARY KEY,
                    name        VARCHAR,
                    type        VARCHAR,
                    aliases     VARCHAR,
                    scope       VARCHAR,
                    created_at  {DOUBLE},
                    updated_at  {DOUBLE},
                    embedding   {VECTOR}
                )""")
            self._x("""
                CREATE TABLE IF NOT EXISTS etches (
                    id           VARCHAR PRIMARY KEY,
                    entity_id    VARCHAR,
                    entity_name  VARCHAR,
                    property     VARCHAR,
                    current_value VARCHAR,
                    status       VARCHAR,
                    confidence   {DOUBLE},
                    narrative    VARCHAR,
                    version      INTEGER,
                    scope        VARCHAR,
                    source       VARCHAR,
                    claim_ids    VARCHAR,
                    source_ids   VARCHAR,
                    created_at   {DOUBLE},
                    updated_at   {DOUBLE},
                    embedding    {VECTOR},
                    value_entity_id VARCHAR,
                    details      VARCHAR,
                    evidence     VARCHAR,
                    value_weights VARCHAR
                )""")
            self._x("""
                CREATE TABLE IF NOT EXISTS etch_versions (
                    etch_id      VARCHAR,
                    event_at     {DOUBLE},
                    version      INTEGER,
                    current_value VARCHAR,
                    status       VARCHAR,
                    confidence   {DOUBLE},
                    narrative    VARCHAR,
                    triggered_by VARCHAR,
                    created_at   {DOUBLE},
                    details      VARCHAR,
                    evidence     VARCHAR,
                    value_weights VARCHAR,
                    PRIMARY KEY (etch_id, version)
                )""")
            self._x("""
                CREATE TABLE IF NOT EXISTS context_subjects (
                    context_id    VARCHAR,
                    entity_id     VARCHAR,
                    mentions      INTEGER,
                    last_seen_at  {DOUBLE},
                    PRIMARY KEY (context_id, entity_id)
                )""")
            self._x("""
                CREATE TABLE IF NOT EXISTS anon_tokens (
                    entity_id   VARCHAR PRIMARY KEY,
                    token       VARCHAR,
                    created_at  {DOUBLE}
                )""")
            self._x("""
                CREATE TABLE IF NOT EXISTS erasures (
                    id          VARCHAR PRIMARY KEY,
                    created_at  {DOUBLE},
                    reason      VARCHAR,
                    signal_ids  VARCHAR,
                    report      VARCHAR
                )""")
            self._migrate()
            for sql in (
                "CREATE INDEX IF NOT EXISTS etches_entity_idx ON etches (entity_id)",
                "CREATE INDEX IF NOT EXISTS etches_value_entity_idx ON etches (value_entity_id)",
                "CREATE INDEX IF NOT EXISTS etches_scope_idx ON etches (scope)",
            ):
                self._x(sql)
            self._db.check_vector_dim("entities", "embedding")
            self._db.check_vector_dim("etches", "embedding")
            self._db.vector_index("entities", "embedding")
            self._db.vector_index("etches", "embedding")

    # ── entities ─────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Additive migration for right DBs created before relations existed."""
        cols = self._db.columns("etches")
        for col in ("value_entity_id", "details", "evidence", "value_weights"):
            if col not in cols:
                self._x(f"ALTER TABLE etches ADD COLUMN {col} VARCHAR")
        vcols = self._db.columns("etch_versions")
        for col in ("details", "evidence", "value_weights"):
            if col not in vcols:
                self._x(f"ALTER TABLE etch_versions ADD COLUMN {col} VARCHAR")
        if "event_at" not in vcols:
            # Older rows only know when we learned it. Seeding event_at from
            # created_at makes event-basis time travel degrade to ingest-basis
            # for them rather than returning nothing.
            self._x("ALTER TABLE etch_versions ADD COLUMN event_at {DOUBLE}")
            self._x("UPDATE etch_versions SET event_at = created_at WHERE event_at IS NULL")

    # ── conversation focus ──────────────────────────────────────────────────

    def touch_context(self, context_id: str, entity_ids: list[str], at: float) -> None:
        """These subjects came up in this conversation at this (event) time."""
        if not context_id or not entity_ids:
            return
        with self._lock:
            for eid in dict.fromkeys(entity_ids):
                self._x(
                    """INSERT INTO context_subjects VALUES (?, ?, 1, ?)
                       ON CONFLICT (context_id, entity_id) DO UPDATE SET
                         mentions = context_subjects.mentions + 1,
                         last_seen_at = greatest(context_subjects.last_seen_at, excluded.last_seen_at)""",
                    [context_id, eid, at])

    def context_focus(self, context_id: str, *, since: float = 0.0, before: float | None = None,
                      limit: int = 5) -> list[tuple[Entity, int, float]]:
        """What a conversation has been about: subjects, mentions, last seen —
        most recent first."""
        if not context_id:
            return []
        upper = "AND c.last_seen_at < ?" if before is not None else ""
        params = [context_id, since] + ([before] if before is not None else []) + [limit]
        with self._lock:
            rows = self._x(
                f"""SELECT e.id, e.name, e.type, e.aliases, e.scope, e.created_at, e.updated_at,
                           e.embedding, c.mentions, c.last_seen_at
                    FROM context_subjects c JOIN entities e ON e.id = c.entity_id
                    WHERE c.context_id = ? AND c.last_seen_at >= ? {upper}
                    ORDER BY c.last_seen_at DESC, c.mentions DESC LIMIT ?""", params).fetchall()
        return [(self._row_to_entity(r[:8]), int(r[8]), float(r[9])) for r in rows]

    def etches_referencing(self, entity_id: str) -> list[Etch]:
        """Every belief whose VALUE points at this entity — the reverse edge.

        "Which products list DK-1120 as their fix?" is one indexed lookup here,
        and was a text search before the value became a resolved reference.
        """
        with self._lock:
            rows = self._x(
                f"""SELECT {_ETCH_COLS} FROM etches WHERE value_entity_id = ?
                    ORDER BY entity_name, property""", [entity_id]).fetchall()
        return [self._row_to_etch(r) for r in rows]

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._lock:
            r = self._x(f"SELECT {_ENTITY_COLS} FROM entities WHERE id = ?",
                        [entity_id]).fetchone()
        return self._row_to_entity(r) if r else None

    def find_entity_by_alias(self, alias_norm: str,
                             scope: str | Sequence[str] | None = None) -> Entity | None:
        """Match where the normalized alias appears in the stored alias list,
        optionally only among entities of these scopes."""
        found = self.find_entities_by_alias(alias_norm, scope)
        return found[0] if found else None

    def find_entities_by_alias(self, alias_norm: str,
                               scope: str | Sequence[str] | None = None) -> list[Entity]:
        """Every entity carrying this alias — one per scope it was seen in."""
        clauses, params = ["aliases LIKE '%' || ? || '%'"], [json.dumps(alias_norm)]
        sc, sp = scope_clause("scope", scope)
        if sc:
            clauses.append(sc); params.extend(sp)
        with self._lock:
            rows = self._x(
                f"""SELECT {_ENTITY_COLS} FROM entities WHERE {' AND '.join(clauses)}
                    ORDER BY created_at, id""", params).fetchall()
        return [self._row_to_entity(r) for r in rows if alias_norm in set(_u(r[3], []))]

    def recent_context(self, scope: str | None, *, entities: int = 50,
                       label_properties: set[str] | None = None,
                       labels_per_belief: int = 10, beliefs: int = 30,
                       ) -> tuple[list[str], list[str]]:
        """What the extractor should reuse: recently active subject names, and
        the labels already held for label-like properties, in one scope.

        Returns (entity_names, label_lines) where each label line reads
        "subject · property: label, label". Labels include retracted ones, so
        "we dropped X" can name exactly the X that was asserted.
        """
        where, params = ("WHERE e.scope = ?", [scope]) if scope else ("", [])
        names: list[str] = []
        lines: list[str] = []
        with self._lock:
            if entities > 0:
                names = [r[0] for r in self._x(
                    f"""SELECT n.name, max(e.updated_at) AS m FROM etches e
                        JOIN entities n ON n.id = e.entity_id {where}
                        GROUP BY n.name ORDER BY m DESC LIMIT ?""",
                    [*params, entities]).fetchall()]
            props = sorted(label_properties or ())
            if props and labels_per_belief > 0:
                clause = (f"{where} AND" if where else "WHERE") + \
                    f" e.property IN ({', '.join('?' * len(props))})"
                rows = self._x(
                    f"""SELECT e.entity_name, e.property, e.current_value, e.value_weights
                        FROM etches e {clause} ORDER BY e.updated_at DESC LIMIT ?""",
                    [*params, *props, beliefs]).fetchall()
                for ent, prop, current, weights in rows:
                    labels = [v["value"] for v in _u(weights, [])] or \
                        [x.strip() for x in (current or "").split(",") if x.strip()]
                    if labels:
                        lines.append(f"{ent} · {prop}: "
                                     + ", ".join(labels[:labels_per_belief]))
        return names, lines

    def search_entities(self, qvec, top_k=5,
                        scope: str | Sequence[str] | None = None) -> list[tuple[Entity, float]]:
        clauses, params = ["embedding IS NOT NULL"], []
        sc, sp = scope_clause("scope", scope)
        if sc:
            clauses.append(sc); params.extend(sp)
        with self._lock:
            rows = self._x(
                f"""SELECT {_ENTITY_COLS}, {{DIST:embedding}} AS dist
                    FROM entities WHERE {' AND '.join(clauses)} ORDER BY dist LIMIT ?""",
                [self._db.vec(qvec), *params, top_k]).fetchall()
        return [(self._row_to_entity(r[:8]), 1.0 - float(r[8])) for r in rows]

    def upsert_entity(self, e: Entity) -> None:
        with self._lock:
            self._x("DELETE FROM entities WHERE id = ?", [e.id])
            self._x(
                """INSERT INTO entities (id, name, type, aliases, scope, created_at,
                    updated_at, embedding)
                    VALUES (?,?,?,?,?,?,?,{VEC})""",
                [e.id, e.name, e.type, _j(e.aliases), e.scope, e.created_at,
                 e.updated_at, self._db.vec(e.embedding)])

    def delete_entity(self, entity_id: str) -> None:
        """An erased subject: the registry row, its pseudonym and where it came up."""
        with self._lock:
            self._x("DELETE FROM entities WHERE id = ?", [entity_id])
            self._x("DELETE FROM anon_tokens WHERE entity_id = ?", [entity_id])
            self._x("DELETE FROM context_subjects WHERE entity_id = ?", [entity_id])

    def count_entities(self) -> int:
        with self._lock:
            return int(self._x("SELECT count(*) FROM entities").fetchone()[0])

    # ── anonymization tokens ─────────────────────────────────────────────

    def get_or_assign_anon_token(self, entity_id: str, label: str) -> str:
        """Return the pseudonym for an entity, assigning the next numbered
        token of its label ([PERSON_3], [COMPANY_1], ...) on first use.
        Persisted, so the same entity always maps to the same token."""
        with self._lock:
            r = self._x(
                "SELECT token FROM anon_tokens WHERE entity_id = ?", [entity_id]).fetchone()
            if r:
                return r[0]
            n = self._x(
                "SELECT count(*) FROM anon_tokens WHERE starts_with(token, ?)",
                [f"[{label}_"]).fetchone()[0]
            token = f"[{label}_{int(n) + 1}]"
            self._x(
                "INSERT INTO anon_tokens (entity_id, token, created_at) VALUES (?,?,?)",
                [entity_id, token, time.time()])
            return token

    # ── etches ───────────────────────────────────────────────────────────

    def get_etch(self, etch_id: str) -> Etch | None:
        with self._lock:
            r = self._x(
                f"""SELECT {_ETCH_COLS}
                   FROM etches WHERE id = ?""", [etch_id]).fetchone()
        return self._row_to_etch(r) if r else None

    def upsert_etch(self, e: Etch) -> None:
        with self._lock:
            self._x("DELETE FROM etches WHERE id = ?", [e.id])
            self._x(
                """INSERT INTO etches (id, entity_id, entity_name, property,
                    current_value, status, confidence, narrative, version, scope,
                    source, claim_ids, source_ids, created_at, updated_at, embedding,
                    value_entity_id, details, evidence, value_weights)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,{VEC},?,?,?,?)""",
                [e.id, e.entity_id, e.entity_name, e.property, e.current_value, e.status,
                 e.confidence, e.narrative, e.version, e.scope, e.source,
                 _j(e.claim_ids), _j(e.source_ids), e.created_at, e.updated_at,
                 self._db.vec(e.embedding), e.value_entity_id, _j(e.details),
                 _j(e.evidence), _j(e.values)])

    def delete_etch(self, etch_id: str) -> int:
        """A belief and every version of it. Returns the versions removed."""
        with self._lock:
            n = int(self._x("SELECT count(*) FROM etch_versions WHERE etch_id = ?",
                            [etch_id]).fetchone()[0])
            self._x("DELETE FROM etch_versions WHERE etch_id = ?", [etch_id])
            self._x("DELETE FROM etches WHERE id = ?", [etch_id])
        return n

    def redact_versions(self, etch_id: str, note: str) -> int:
        """Blank what every stored version of a belief said, keeping when it
        changed. Returns the number of versions redacted."""
        with self._lock:
            n = int(self._x("SELECT count(*) FROM etch_versions WHERE etch_id = ?",
                            [etch_id]).fetchone()[0])
            self._x(
                """UPDATE etch_versions SET current_value = ?, narrative = ?,
                   details = ?, evidence = ?, value_weights = ?, triggered_by = ?
                   WHERE etch_id = ?""",
                [ERASED, note, _j([]), _j({}), _j([]), _j([]), etch_id])
        return n

    def add_version(self, etch_id, version, value, status, confidence, narrative,
                    triggered_by, created_at, event_at: float | None = None,
                    details: list[dict] | None = None,
                    evidence: dict | None = None,
                    values: list[dict] | None = None) -> None:
        """Snapshot one belief change.

        `created_at` is when we learned it; `event_at` is when the newest fact
        behind it became true. Keeping both is what lets a caller ask either
        "what did we believe in August" or "what was true in August" — which
        are different questions whenever a source reports late.
        """
        with self._lock:
            self._x("DELETE FROM etch_versions WHERE etch_id = ? AND version = ?",
                    [etch_id, version])
            self._x(
                """INSERT INTO etch_versions (etch_id, version, current_value, status,
                    confidence, narrative, triggered_by, created_at, event_at, details,
                    evidence, value_weights)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [etch_id, version, value, status, confidence, narrative,
                 _j(triggered_by), created_at, event_at or created_at,
                 _j(details or []), _j(evidence or {}), _j(values or [])])

    def versions(self, etch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._x(
                """SELECT version, current_value, status, confidence, narrative,
                          triggered_by, created_at, event_at, details, evidence,
                          value_weights
                   FROM etch_versions WHERE etch_id = ? ORDER BY version""",
                [etch_id]).fetchall()
        return [self._row_to_version(r) for r in rows]

    def version_as_of(self, etch_id: str, as_of: float,
                      basis: str = "ingest") -> dict[str, Any] | None:
        """The belief as it stood at `as_of`.

        basis="ingest" (default) — what we BELIEVED then. Answers "what would
            this system have told a customer on that date", which is the
            question an audit or a dispute actually asks.
        basis="event" — what was TRUE then, by the dates on the underlying
            facts. Differs whenever a source reported late: a job change on the
            2nd that reached us on the 15th is invisible to ingest-basis on the
            10th, and visible to event-basis.
        """
        column = "event_at" if basis == "event" else "created_at"
        with self._lock:
            r = self._x(
                f"""SELECT version, current_value, status, confidence, narrative,
                           triggered_by, created_at, event_at, details, evidence,
                           value_weights
                    FROM etch_versions WHERE etch_id = ? AND {column} <= ?
                    ORDER BY {column} DESC, version DESC LIMIT 1""",
                [etch_id, as_of]).fetchone()
        return self._row_to_version(r) if r else None

    def search_etches(self, qvec, top_k, scope=None) -> list[Hit]:
        clauses, params = ["embedding IS NOT NULL"], []
        sc, sp = scope_clause("scope", scope)
        if sc:
            clauses.append(sc); params.extend(sp)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._x(
                f"""SELECT id, entity_id, entity_name, property, current_value, status,
                           confidence, narrative, version, scope, source, created_at,
                           updated_at, details, evidence, value_weights,
                           {{DIST:embedding}} AS dist
                    FROM etches WHERE {where} ORDER BY dist LIMIT ?""",
                [self._db.vec(qvec), *params, top_k]).fetchall()
        out = []
        for r in rows:
            out.append(Hit(r[0], r[7], 1.0 - float(r[16]), {
                "entity_id": r[1], "entity_name": r[2], "property": r[3],
                "current_value": r[4], "status": r[5], "confidence": r[6],
                "version": r[8], "scope": r[9], "source": r[10],
                "created_at": r[11], "updated_at": r[12],
                "details": _u(r[13], []), "evidence": _u(r[14], {}),
                "values": _u(r[15], [])}))
        return out

    def etches_for_entity(self, entity_id: str,
                          scope: str | Sequence[str] | None = None) -> list[Etch]:
        """Every belief held about one subject.

        Exhaustive and ordered, unlike `search_etches`, which ranks by
        embedding distance and returns top_k. "What do we know about X" needs
        completeness: a fact whose narrative embeds poorly against the query
        must not silently drop out of the answer.
        """
        clauses, params = ["entity_id = ?"], [entity_id]
        sc, sp = scope_clause("scope", scope)
        if sc:
            clauses.append(sc); params.extend(sp)
        with self._lock:
            rows = self._x(
                f"""SELECT {_ETCH_COLS} FROM etches WHERE {' AND '.join(clauses)}
                    ORDER BY property""", params).fetchall()
        return [self._row_to_etch(r) for r in rows]

    def list_etches(self, *, scope: str | Sequence[str] | None = None,
                    properties: list[str] | None = None,
                    entity_types: list[str] | None = None,
                    status: str | None = None,
                    min_confidence: float | None = None,
                    limit: int = 500, offset: int = 0,
                    ) -> tuple[list[tuple[Etch, str | None]], int]:
        """Beliefs matching plain filters, newest first, with the subject's type.

        Exhaustive within the filters, unlike `search_etches`: "every
        incorrect_assumption in scope X" is a WHERE clause, not a similarity
        ranking, so nothing drops out because its narrative embeds poorly.
        Returns one page and the total match count, for paging.
        """
        clauses, params = [], []
        sc, sp = scope_clause("e.scope", scope)
        if sc:
            clauses.append(sc); params.extend(sp)
        if properties:
            clauses.append(f"e.property IN ({', '.join('?' * len(properties))})")
            params.extend(properties)
        if entity_types:
            clauses.append(f"n.type IN ({', '.join('?' * len(entity_types))})")
            params.extend(entity_types)
        if status:
            clauses.append("e.status = ?"); params.append(status)
        if min_confidence is not None:
            clauses.append("e.confidence >= ?"); params.append(min_confidence)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cols = ", ".join(f"e.{c.strip()}" for c in _ETCH_COLS.split(","))
        with self._lock:
            total = int(self._x(
                f"""SELECT count(*) FROM etches e
                    LEFT JOIN entities n ON n.id = e.entity_id {where}""",
                params).fetchone()[0])
            rows = self._x(
                f"""SELECT {cols}, n.type FROM etches e
                    LEFT JOIN entities n ON n.id = e.entity_id {where}
                    ORDER BY e.updated_at DESC, e.id LIMIT ? OFFSET ?""",
                [*params, limit, offset]).fetchall()
        return [(self._row_to_etch(r), r[-1]) for r in rows], total

    def all_etches(self) -> list[Etch]:
        with self._lock:
            rows = self._x(
                f"""SELECT {_ETCH_COLS}
                    FROM etches ORDER BY updated_at DESC""").fetchall()
        return [self._row_to_etch(r) for r in rows]

    def count_etches(self) -> int:
        with self._lock:
            return int(self._x("SELECT count(*) FROM etches").fetchone()[0])

    def count_contested(self) -> int:
        with self._lock:
            return int(self._x(
                "SELECT count(*) FROM etches WHERE status = 'contested'").fetchone()[0])

    # ── erasure tombstones ───────────────────────────────────────────────

    def add_erasure(self, erasure_id: str, reason: str | None, signal_ids: list[str],
                    report: dict[str, Any]) -> None:
        """Record an erasure. Holds signal ids (content hashes) and counts, never
        the erased text or the subject's name."""
        with self._lock:
            self._x(
                "INSERT INTO erasures (id, created_at, reason, signal_ids, report) "
                "VALUES (?, ?, ?, ?, ?)",
                [erasure_id, time.time(), reason, _j(sorted(set(signal_ids))), _j(report)])

    def is_erased(self, signal_id: str) -> bool:
        """Whether this exact signal was erased before (so it is not re-ingested)."""
        with self._lock:
            row = self._x(
                "SELECT 1 FROM erasures WHERE signal_ids LIKE '%' || ? || '%' LIMIT 1",
                [json.dumps(signal_id)]).fetchone()
        return row is not None

    def erasures(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._x(
                "SELECT id, created_at, reason, report FROM erasures ORDER BY created_at"
            ).fetchall()
        return [{"id": r[0], "created_at": r[1], "reason": r[2], "report": _u(r[3], {})}
                for r in rows]

    # ── row mappers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_version(r) -> dict[str, Any]:
        return {"version": r[0], "current_value": r[1], "status": r[2],
                "confidence": r[3], "narrative": r[4], "triggered_by": _u(r[5], []),
                "created_at": r[6], "event_at": r[7], "details": _u(r[8], []),
                "evidence": _u(r[9], {}), "values": _u(r[10], [])}

    @staticmethod
    def _row_to_entity(r) -> Entity:
        return Entity(id=r[0], name=r[1], type=r[2], aliases=_u(r[3], []), scope=r[4],
                      created_at=r[5], updated_at=r[6], embedding=_vec(r[7]))

    @staticmethod
    def _row_to_etch(r) -> Etch:
        return Etch(id=r[0], entity_id=r[1], entity_name=r[2], property=r[3],
                    current_value=r[4], status=r[5], confidence=r[6], narrative=r[7],
                    version=r[8], scope=r[9], source=r[10], claim_ids=_u(r[11], []),
                    source_ids=_u(r[12], []), created_at=r[13], updated_at=r[14],
                    embedding=_vec(r[15]), value_entity_id=r[16], details=_u(r[17], []),
                    evidence=_u(r[18], {}), values=_u(r[19], []))


# ── Container ────────────────────────────────────────────────────────────────

class Stores:
    """Both layers. Postgres when `database_url` is given, else DuckDB files."""

    def __init__(self, data_dir: str, dim: int, *, database_url: str | None = None,
                 schema: str | None = None) -> None:
        if not database_url:
            os.makedirs(data_dir, exist_ok=True)
        self.left = LeftStore(connect(database_url=database_url, schema=schema, dim=dim,
                                      path=os.path.join(data_dir, "left.duckdb")))
        self.right = RightStore(connect(database_url=database_url, schema=schema, dim=dim,
                                        path=os.path.join(data_dir, "right.duckdb")))
        self.dim = dim
        self.engine = self.left._db.engine
