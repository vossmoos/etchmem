"""
DuckDB stores — two databases, layered memory.

  LeftStore  → left.duckdb
      signals : raw deposits (status lifecycle: new→batched→extracted)
      claims  : structured assertions extracted from signals (corroboration-aware)

  RightStore → right.duckdb
      entities      : canonical entity registry (+ aliases, embedding)
      etches        : current belief per (entity, property)  [HEAD]
      etch_versions : immutable snapshots of every belief change

Semantic search uses DuckDB's core `array_cosine_similarity` over fixed-size
FLOAT[dim] embedding columns. A per-connection lock keeps the single-writer
DuckDB connections safe to share across the API process + worker task.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb

# Signal status lifecycle (the in-process "queue" contract).
S_NEW = "new"
S_BATCHED = "batched"
S_EXTRACTED = "extracted"

# Claim status.
C_NEW = "new"
C_CONSOLIDATED = "consolidated"


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


_CLAIM_COLS = """id, entity_id, entity_name, property, value, value_norm,
                polarity, event_time, ingest_time, sources,
                evidence_signal_ids, corroboration_count, confidence, scope,
                status, created_at, updated_at, value_entity_id"""

_ETCH_COLS = """id, entity_id, entity_name, property, current_value, status,
                confidence, narrative, version, scope, source, claim_ids,
                source_ids, created_at, updated_at, embedding, value_entity_id"""


# ── Left store ───────────────────────────────────────────────────────────────

class LeftStore:
    def __init__(self, path: str, dim: int) -> None:
        self._dim = dim
        self._lock = threading.Lock()
        self._con = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS signals (
                    id            VARCHAR PRIMARY KEY,
                    content       VARCHAR,
                    source        VARCHAR,
                    scope         VARCHAR,
                    metadata      VARCHAR,
                    embedding     FLOAT[{self._dim}],
                    created_at    DOUBLE,
                    occurred_at   DOUBLE,
                    expires_at    DOUBLE,
                    status        VARCHAR,
                    extract_mode  VARCHAR,
                    canonical_id  VARCHAR
                );
                CREATE TABLE IF NOT EXISTS claims (
                    id                  VARCHAR PRIMARY KEY,
                    entity_id           VARCHAR,
                    entity_name         VARCHAR,
                    property            VARCHAR,
                    value               VARCHAR,
                    value_norm          VARCHAR,
                    polarity            VARCHAR,
                    event_time          DOUBLE,
                    ingest_time         DOUBLE,
                    sources             VARCHAR,
                    evidence_signal_ids VARCHAR,
                    corroboration_count INTEGER,
                    confidence          DOUBLE,
                    value_entity_id     VARCHAR,
                    scope               VARCHAR,
                    status              VARCHAR,
                    created_at          DOUBLE,
                    updated_at          DOUBLE
                );
                """
            )
            self._migrate()

    def _migrate(self) -> None:
        """Additive migrations for databases created by an earlier version.

        Called with the lock held. Adding occurred_at to an existing signals
        table backfills it from created_at, which preserves the old behaviour
        exactly: undeclared event time means ingest time.
        """
        cols = {r[1] for r in self._con.execute("PRAGMA table_info('signals')").fetchall()}
        if "occurred_at" not in cols:
            self._con.execute("ALTER TABLE signals ADD COLUMN occurred_at DOUBLE")
            self._con.execute("UPDATE signals SET occurred_at = created_at "
                              "WHERE occurred_at IS NULL")
        ccols = {r[1] for r in self._con.execute("PRAGMA table_info('claims')").fetchall()}
        if "value_entity_id" not in ccols:
            self._con.execute("ALTER TABLE claims ADD COLUMN value_entity_id VARCHAR")

    # ── signals ──────────────────────────────────────────────────────────

    def add_signal(self, sig: Signal) -> bool:
        with self._lock:
            if self._con.execute("SELECT 1 FROM signals WHERE id = ?", [sig.id]).fetchone():
                return False
            self._con.execute(
                f"""
                INSERT INTO signals (id, content, source, scope, metadata, embedding,
                    created_at, occurred_at, expires_at, status, extract_mode,
                    canonical_id)
                VALUES (?, ?, ?, ?, ?, ?::FLOAT[{self._dim}], ?, ?, ?, ?, ?, ?)
                """,
                [sig.id, sig.content, sig.source, sig.scope, _j(sig.metadata),
                 sig.embedding, sig.created_at, sig.occurred_at, sig.expires_at,
                 sig.status, sig.extract_mode, sig.canonical_id],
            )
            return True

    def signals_by_status(self, status: str) -> list[Signal]:
        with self._lock:
            rows = self._con.execute(
                """SELECT id, content, source, scope, metadata, embedding, created_at,
                          occurred_at, expires_at, status, extract_mode, canonical_id
                   FROM signals WHERE status = ? ORDER BY created_at""",
                [status],
            ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def signals_with_canonical(self, canonical_id: str) -> list[Signal]:
        with self._lock:
            rows = self._con.execute(
                """SELECT id, content, source, scope, metadata, embedding, created_at,
                          occurred_at, expires_at, status, extract_mode, canonical_id
                   FROM signals WHERE canonical_id = ?""",
                [canonical_id],
            ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def set_signal_status(self, ids: list[str], status: str) -> None:
        if not ids:
            return
        with self._lock:
            self._con.executemany(
                "UPDATE signals SET status = ? WHERE id = ?",
                [[status, i] for i in ids],
            )

    def set_canonical(self, signal_id: str, canonical_id: str, status: str) -> None:
        with self._lock:
            self._con.execute(
                "UPDATE signals SET canonical_id = ?, status = ? WHERE id = ?",
                [canonical_id, status, signal_id],
            )

    def search_signals(self, qvec, top_k, scope=None, source=None) -> list[Hit]:
        clauses, params = ["embedding IS NOT NULL"], []
        if scope:
            clauses.append("scope = ?"); params.append(scope)
        if source:
            clauses.append("source = ?"); params.append(source)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._con.execute(
                f"""SELECT id, content, source, scope, created_at,
                           array_cosine_similarity(embedding, ?::FLOAT[{self._dim}]) AS sim
                    FROM signals WHERE {where} ORDER BY sim DESC LIMIT ?""",
                [qvec, *params, top_k],
            ).fetchall()
        return [Hit(r[0], r[1], float(r[5]),
                    {"source": r[2], "scope": r[3], "created_at": r[4]}) for r in rows]

    def expire(self, now: float | None = None) -> int:
        now = now or time.time()
        with self._lock:
            before = self._con.execute("SELECT count(*) FROM signals").fetchone()[0]
            self._con.execute(
                "DELETE FROM signals WHERE expires_at > 0 AND expires_at < ?", [now])
            after = self._con.execute("SELECT count(*) FROM signals").fetchone()[0]
        return int(before - after)

    # ── claims ───────────────────────────────────────────────────────────

    def upsert_claim(self, claim: Claim) -> None:
        """Insert a claim, or merge corroboration into an existing one."""
        with self._lock:
            existing = self._con.execute(
                """SELECT sources, evidence_signal_ids, corroboration_count, event_time
                   FROM claims WHERE id = ?""", [claim.id]).fetchone()
            if existing:
                sources = sorted(set(_u(existing[0], []) + claim.sources))
                evidence = sorted(set(_u(existing[1], []) + claim.evidence_signal_ids))
                count = len(evidence)
                event_time = max(existing[3] or 0.0, claim.event_time)
                self._con.execute(
                    """UPDATE claims SET sources = ?, evidence_signal_ids = ?,
                       corroboration_count = ?, event_time = ?, status = ?,
                       updated_at = ? WHERE id = ?""",
                    [_j(sources), _j(evidence), count, event_time, C_NEW,
                     time.time(), claim.id])
            else:
                self._con.execute(
                    """INSERT INTO claims (id, entity_id, entity_name, property, value,
                        value_norm, polarity, event_time, ingest_time, sources,
                        evidence_signal_ids, corroboration_count, confidence, scope,
                        status, created_at, updated_at, value_entity_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [claim.id, claim.entity_id, claim.entity_name, claim.property,
                     claim.value, claim.value_norm, claim.polarity, claim.event_time,
                     claim.ingest_time, _j(claim.sources), _j(claim.evidence_signal_ids),
                     claim.corroboration_count, claim.confidence, claim.scope,
                     claim.status, claim.created_at, claim.updated_at,
                     claim.value_entity_id])

    def pairs_with_new_claims(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._con.execute(
                """SELECT DISTINCT entity_id, property FROM claims WHERE status = ?""",
                [C_NEW]).fetchall()
        return [(r[0], r[1]) for r in rows]

    def claims_for_pair(self, entity_id: str, prop: str) -> list[Claim]:
        with self._lock:
            rows = self._con.execute(
                f"""SELECT {_CLAIM_COLS}
                   FROM claims WHERE entity_id = ? AND property = ?""",
                [entity_id, prop]).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def claims_by_ids(self, ids: list[str]) -> list[Claim]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._con.execute(
                f"""SELECT {_CLAIM_COLS}
                    FROM claims WHERE id IN ({placeholders})""", ids).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def signals_by_ids(self, ids: list[str]) -> list[Signal]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._con.execute(
                f"""SELECT id, content, source, scope, metadata, embedding, created_at,
                           occurred_at, expires_at, status, extract_mode, canonical_id
                    FROM signals WHERE id IN ({placeholders})
                    ORDER BY created_at""", ids).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def set_claims_status(self, ids: list[str], status: str) -> None:
        if not ids:
            return
        with self._lock:
            self._con.executemany(
                "UPDATE claims SET status = ? WHERE id = ?", [[status, i] for i in ids])

    # ── counts ───────────────────────────────────────────────────────────

    def count_signals(self, status: str | None = None) -> int:
        with self._lock:
            if status:
                return int(self._con.execute(
                    "SELECT count(*) FROM signals WHERE status = ?", [status]).fetchone()[0])
            return int(self._con.execute("SELECT count(*) FROM signals").fetchone()[0])

    def count_claims(self) -> int:
        with self._lock:
            return int(self._con.execute("SELECT count(*) FROM claims").fetchone()[0])

    def scopes(self) -> list[str]:
        with self._lock:
            rows = self._con.execute(
                "SELECT DISTINCT scope FROM signals WHERE scope IS NOT NULL").fetchall()
        return sorted(r[0] for r in rows)

    # ── row mappers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_signal(r) -> Signal:
        return Signal(
            id=r[0], content=r[1], source=r[2], scope=r[3], metadata=_u(r[4], {}),
            embedding=list(r[5]) if r[5] is not None else None, created_at=r[6],
            occurred_at=r[7] or 0.0, expires_at=r[8], status=r[9],
            extract_mode=r[10], canonical_id=r[11])

    @staticmethod
    def _row_to_claim(r) -> Claim:
        return Claim(
            id=r[0], entity_id=r[1], entity_name=r[2], property=r[3], value=r[4],
            value_norm=r[5], polarity=r[6], event_time=r[7], ingest_time=r[8],
            sources=_u(r[9], []), evidence_signal_ids=_u(r[10], []),
            corroboration_count=r[11], confidence=r[12], scope=r[13], status=r[14],
            created_at=r[15], updated_at=r[16], value_entity_id=r[17])


# ── Right store ──────────────────────────────────────────────────────────────

class RightStore:
    def __init__(self, path: str, dim: int) -> None:
        self._dim = dim
        self._lock = threading.Lock()
        self._con = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS entities (
                    id          VARCHAR PRIMARY KEY,
                    name        VARCHAR,
                    type        VARCHAR,
                    aliases     VARCHAR,
                    scope       VARCHAR,
                    created_at  DOUBLE,
                    updated_at  DOUBLE,
                    embedding   FLOAT[{self._dim}]
                );
                CREATE TABLE IF NOT EXISTS etches (
                    id           VARCHAR PRIMARY KEY,
                    entity_id    VARCHAR,
                    entity_name  VARCHAR,
                    property     VARCHAR,
                    current_value VARCHAR,
                    status       VARCHAR,
                    confidence   DOUBLE,
                    narrative    VARCHAR,
                    version      INTEGER,
                    scope        VARCHAR,
                    source       VARCHAR,
                    claim_ids    VARCHAR,
                    source_ids   VARCHAR,
                    created_at   DOUBLE,
                    updated_at   DOUBLE,
                    embedding    FLOAT[{self._dim}],
                    value_entity_id VARCHAR
                );
                CREATE TABLE IF NOT EXISTS etch_versions (
                    etch_id      VARCHAR,
                    event_at     DOUBLE,
                    version      INTEGER,
                    current_value VARCHAR,
                    status       VARCHAR,
                    confidence   DOUBLE,
                    narrative    VARCHAR,
                    triggered_by VARCHAR,
                    created_at   DOUBLE,
                    PRIMARY KEY (etch_id, version)
                );
                CREATE TABLE IF NOT EXISTS anon_tokens (
                    entity_id   VARCHAR PRIMARY KEY,
                    token       VARCHAR,
                    created_at  DOUBLE
                );
                """
            )
            self._migrate()


    # ── entities ─────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Additive migration for right DBs created before relations existed."""
        cols = {r[1] for r in self._con.execute("PRAGMA table_info('etches')").fetchall()}
        if "value_entity_id" not in cols:
            self._con.execute("ALTER TABLE etches ADD COLUMN value_entity_id VARCHAR")
        vcols = {r[1] for r in
                 self._con.execute("PRAGMA table_info('etch_versions')").fetchall()}
        if "event_at" not in vcols:
            # Older rows only know when we learned it. Seeding event_at from
            # created_at makes event-basis time travel degrade to ingest-basis
            # for them rather than returning nothing.
            self._con.execute("ALTER TABLE etch_versions ADD COLUMN event_at DOUBLE")
            self._con.execute("UPDATE etch_versions SET event_at = created_at "
                              "WHERE event_at IS NULL")

    def etches_referencing(self, entity_id: str) -> list[Etch]:
        """Every belief whose VALUE points at this entity — the reverse edge.

        "Which products list DK-1120 as their fix?" is one indexed lookup here,
        and was a text search before the value became a resolved reference.
        """
        with self._lock:
            rows = self._con.execute(
                f"""SELECT {_ETCH_COLS} FROM etches WHERE value_entity_id = ?
                    ORDER BY entity_name, property""", [entity_id]).fetchall()
        return [self._row_to_etch(r) for r in rows]

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._lock:
            r = self._con.execute(
                """SELECT id, name, type, aliases, scope, created_at, updated_at, embedding
                   FROM entities WHERE id = ?""", [entity_id]).fetchone()
        return self._row_to_entity(r) if r else None

    def find_entity_by_alias(self, alias_norm: str) -> Entity | None:
        """Match where the normalized alias appears in the stored alias list."""
        with self._lock:
            rows = self._con.execute(
                """SELECT id, name, type, aliases, scope, created_at, updated_at, embedding
                   FROM entities""").fetchall()
        for r in rows:
            if alias_norm in set(_u(r[3], [])):
                return self._row_to_entity(r)
        return None

    def search_entities(self, qvec, top_k=5) -> list[tuple[Entity, float]]:
        with self._lock:
            rows = self._con.execute(
                f"""SELECT id, name, type, aliases, scope, created_at, updated_at, embedding,
                          array_cosine_similarity(embedding, ?::FLOAT[{self._dim}]) AS sim
                    FROM entities WHERE embedding IS NOT NULL ORDER BY sim DESC LIMIT ?""",
                [qvec, top_k]).fetchall()
        return [(self._row_to_entity(r[:8]), float(r[8])) for r in rows]

    def upsert_entity(self, e: Entity) -> None:
        with self._lock:
            self._con.execute("DELETE FROM entities WHERE id = ?", [e.id])
            self._con.execute(
                f"""INSERT INTO entities (id, name, type, aliases, scope, created_at,
                    updated_at, embedding)
                    VALUES (?,?,?,?,?,?,?,?::FLOAT[{self._dim}])""",
                [e.id, e.name, e.type, _j(e.aliases), e.scope, e.created_at,
                 e.updated_at, e.embedding])

    def count_entities(self) -> int:
        with self._lock:
            return int(self._con.execute("SELECT count(*) FROM entities").fetchone()[0])

    # ── anonymization tokens ─────────────────────────────────────────────

    def get_or_assign_anon_token(self, entity_id: str, label: str) -> str:
        """Return the pseudonym for an entity, assigning the next numbered
        token of its label ([PERSON_3], [COMPANY_1], ...) on first use.
        Persisted, so the same entity always maps to the same token."""
        with self._lock:
            r = self._con.execute(
                "SELECT token FROM anon_tokens WHERE entity_id = ?", [entity_id]).fetchone()
            if r:
                return r[0]
            n = self._con.execute(
                "SELECT count(*) FROM anon_tokens WHERE starts_with(token, ?)",
                [f"[{label}_"]).fetchone()[0]
            token = f"[{label}_{int(n) + 1}]"
            self._con.execute(
                "INSERT INTO anon_tokens (entity_id, token, created_at) VALUES (?,?,?)",
                [entity_id, token, time.time()])
            return token

    # ── etches ───────────────────────────────────────────────────────────

    def get_etch(self, etch_id: str) -> Etch | None:
        with self._lock:
            r = self._con.execute(
                f"""SELECT {_ETCH_COLS}
                   FROM etches WHERE id = ?""", [etch_id]).fetchone()
        return self._row_to_etch(r) if r else None

    def upsert_etch(self, e: Etch) -> None:
        with self._lock:
            self._con.execute("DELETE FROM etches WHERE id = ?", [e.id])
            self._con.execute(
                f"""INSERT INTO etches (id, entity_id, entity_name, property,
                    current_value, status, confidence, narrative, version, scope,
                    source, claim_ids, source_ids, created_at, updated_at, embedding,
                    value_entity_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?::FLOAT[{self._dim}],?)""",
                [e.id, e.entity_id, e.entity_name, e.property, e.current_value, e.status,
                 e.confidence, e.narrative, e.version, e.scope, e.source,
                 _j(e.claim_ids), _j(e.source_ids), e.created_at, e.updated_at,
                 e.embedding, e.value_entity_id])

    def add_version(self, etch_id, version, value, status, confidence, narrative,
                    triggered_by, created_at, event_at: float | None = None) -> None:
        """Snapshot one belief change.

        `created_at` is when we learned it; `event_at` is when the newest fact
        behind it became true. Keeping both is what lets a caller ask either
        "what did we believe in August" or "what was true in August" — which
        are different questions whenever a source reports late.
        """
        with self._lock:
            self._con.execute("DELETE FROM etch_versions WHERE etch_id = ? AND version = ?",
                              [etch_id, version])
            self._con.execute(
                """INSERT INTO etch_versions (etch_id, version, current_value, status,
                    confidence, narrative, triggered_by, created_at, event_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [etch_id, version, value, status, confidence, narrative,
                 _j(triggered_by), created_at, event_at or created_at])

    def versions(self, etch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                """SELECT version, current_value, status, confidence, narrative,
                          triggered_by, created_at, event_at
                   FROM etch_versions WHERE etch_id = ? ORDER BY version""",
                [etch_id]).fetchall()
        return [{"version": r[0], "current_value": r[1], "status": r[2],
                 "confidence": r[3], "narrative": r[4], "triggered_by": _u(r[5], []),
                 "created_at": r[6], "event_at": r[7]} for r in rows]

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
            r = self._con.execute(
                f"""SELECT version, current_value, status, confidence, narrative,
                           triggered_by, created_at, event_at
                    FROM etch_versions WHERE etch_id = ? AND {column} <= ?
                    ORDER BY {column} DESC, version DESC LIMIT 1""",
                [etch_id, as_of]).fetchone()
        if not r:
            return None
        return {"version": r[0], "current_value": r[1], "status": r[2],
                "confidence": r[3], "narrative": r[4], "triggered_by": _u(r[5], []),
                "created_at": r[6], "event_at": r[7]}

    def search_etches(self, qvec, top_k, scope=None) -> list[Hit]:
        clauses, params = ["embedding IS NOT NULL"], []
        if scope:
            clauses.append("scope = ?"); params.append(scope)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._con.execute(
                f"""SELECT id, entity_id, entity_name, property, current_value, status,
                           confidence, narrative, version, scope, source, created_at,
                           updated_at,
                           array_cosine_similarity(embedding, ?::FLOAT[{self._dim}]) AS sim
                    FROM etches WHERE {where} ORDER BY sim DESC LIMIT ?""",
                [qvec, *params, top_k]).fetchall()
        out = []
        for r in rows:
            out.append(Hit(r[0], r[7], float(r[13]), {
                "entity_id": r[1], "entity_name": r[2], "property": r[3],
                "current_value": r[4], "status": r[5], "confidence": r[6],
                "version": r[8], "scope": r[9], "source": r[10],
                "created_at": r[11], "updated_at": r[12]}))
        return out

    def etches_for_entity(self, entity_id: str) -> list[Etch]:
        """Every belief held about one subject.

        Exhaustive and ordered, unlike `search_etches`, which ranks by
        embedding distance and returns top_k. "What do we know about X" needs
        completeness: a fact whose narrative embeds poorly against the query
        must not silently drop out of the answer.
        """
        with self._lock:
            rows = self._con.execute(
                f"""SELECT {_ETCH_COLS} FROM etches WHERE entity_id = ?
                    ORDER BY property""", [entity_id]).fetchall()
        return [self._row_to_etch(r) for r in rows]

    def all_etches(self) -> list[Etch]:
        with self._lock:
            rows = self._con.execute(
                f"""SELECT {_ETCH_COLS}
                    FROM etches ORDER BY updated_at DESC""").fetchall()
        return [self._row_to_etch(r) for r in rows]

    def count_etches(self) -> int:
        with self._lock:
            return int(self._con.execute("SELECT count(*) FROM etches").fetchone()[0])

    def count_contested(self) -> int:
        with self._lock:
            return int(self._con.execute(
                "SELECT count(*) FROM etches WHERE status = 'contested'").fetchone()[0])

    # ── row mappers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_entity(r) -> Entity:
        return Entity(id=r[0], name=r[1], type=r[2], aliases=_u(r[3], []), scope=r[4],
                      created_at=r[5], updated_at=r[6],
                      embedding=list(r[7]) if r[7] is not None else None)

    @staticmethod
    def _row_to_etch(r) -> Etch:
        return Etch(id=r[0], entity_id=r[1], entity_name=r[2], property=r[3],
                    current_value=r[4], status=r[5], confidence=r[6], narrative=r[7],
                    version=r[8], scope=r[9], source=r[10], claim_ids=_u(r[11], []),
                    source_ids=_u(r[12], []), created_at=r[13], updated_at=r[14],
                    embedding=list(r[15]) if r[15] is not None else None,
                    value_entity_id=r[16])


# ── Container ────────────────────────────────────────────────────────────────

class Stores:
    def __init__(self, data_dir: str, dim: int) -> None:
        os.makedirs(data_dir, exist_ok=True)
        self.left = LeftStore(os.path.join(data_dir, "left.duckdb"), dim)
        self.right = RightStore(os.path.join(data_dir, "right.duckdb"), dim)
        self.dim = dim
