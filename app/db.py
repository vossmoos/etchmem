"""
One SQL surface over two engines: PostgreSQL + pgvector, or embedded DuckDB.

The stores write their SQL once, in DuckDB-flavoured text with `?` placeholders
and a few tokens, and this module renders it for the engine in use:

  {DOUBLE}      DOUBLE                      DOUBLE PRECISION
  {VECTOR}      FLOAT[dim]                  vector(dim)
  {VEC}         ?::FLOAT[dim]               %s::vector      (a parameter)
  {DIST:col}    array_cosine_distance(...)  (col <=> %s::vector)

A {VEC} or {DIST:col} parameter is passed through `Db.vec()`, which turns an
embedding into what the driver expects (a list for DuckDB, pgvector's text
form for Postgres). Ordering by {DIST:col} ascending is the form pgvector's
HNSW index can serve.

Postgres is selected by ETCHMEM_DATABASE_URL; without it, etchmem keeps its
zero-setup DuckDB files in ETCHMEM_DATA_DIR. ETCHMEM_DATABASE_SCHEMA puts the
tables in their own schema, so several etchmem instances can share a database.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Iterable, Sequence

_DIST = re.compile(r"\{DIST:([A-Za-z_][\w.]*)\}")
_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Db:
    """A locked connection plus the SQL dialect of its engine."""

    engine: str = ""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.lock = threading.RLock()

    # ── dialect ─────────────────────────────────────────────────────────
    def render(self, sql: str) -> str:
        raise NotImplementedError

    def vec(self, v: Sequence[float] | None) -> Any:
        raise NotImplementedError

    def vec_out(self, v: Any) -> list[float] | None:
        raise NotImplementedError

    # ── execution (caller holds `lock`) ─────────────────────────────────
    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        raise NotImplementedError

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        raise NotImplementedError

    def columns(self, table: str) -> set[str]:
        raise NotImplementedError

    def vector_index(self, table: str, column: str) -> None:
        """An ANN index on an embedding column, where the engine has one."""

    def vector_dim(self, table: str, column: str) -> int | None:
        """The dimension an existing embedding column was created with."""
        raise NotImplementedError

    def check_vector_dim(self, table: str, column: str) -> None:
        """Refuse to run on a table built for another embedding size: every insert would
        fail. Happens when EMBEDDING_PROVIDER or the embedding model changed."""
        found = self.vector_dim(table, column)
        if found is not None and found != self.dim:
            raise RuntimeError(
                f"{table}.{column} holds {found}-dimension embeddings, but the configured "
                f"embedding model produces {self.dim}. Use the embedding provider and model the "
                "data was built with, or a new database (or ETCHMEM_DATABASE_SCHEMA / data dir)."
            )

    def close(self) -> None:
        raise NotImplementedError


class DuckDb(Db):
    engine = "duckdb"

    def __init__(self, path: str, dim: int) -> None:
        super().__init__(dim)
        import duckdb

        self._con = duckdb.connect(path)

    def render(self, sql: str) -> str:
        sql = sql.replace("{DOUBLE}", "DOUBLE").replace("{VECTOR}", f"FLOAT[{self.dim}]")
        sql = sql.replace("{VEC}", f"?::FLOAT[{self.dim}]")
        return _DIST.sub(lambda m: f"array_cosine_distance({m.group(1)}, ?::FLOAT[{self.dim}])", sql)

    def vec(self, v):
        return list(v) if v is not None else None

    def vec_out(self, v):
        return list(v) if v is not None else None

    def execute(self, sql, params=()):
        return self._con.execute(self.render(sql), list(params))

    def executemany(self, sql, rows):
        rows = [list(r) for r in rows]
        if rows:
            self._con.executemany(self.render(sql), rows)

    def columns(self, table):
        rows = self._con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table]).fetchall()
        return {r[0] for r in rows}

    def vector_dim(self, table, column):
        row = self._con.execute(
            "SELECT data_type FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
            [table, column]).fetchone()
        match = re.search(r"\[(\d+)\]", row[0]) if row else None
        return int(match.group(1)) if match else None

    def close(self):
        self._con.close()


class PostgresDb(Db):
    engine = "postgres"

    def __init__(self, url: str, dim: int, schema: str | None = None) -> None:
        super().__init__(dim)
        import psycopg

        if schema and not _SCHEMA_NAME.match(schema):
            raise ValueError(f"invalid ETCHMEM_DATABASE_SCHEMA: {schema!r}")
        self.schema = schema or None
        self._con = psycopg.connect(url, autocommit=True)
        self._con.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if self.schema:
            self._con.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            self._con.execute(f'SET search_path TO "{self.schema}", public')

    def render(self, sql: str) -> str:
        # psycopg uses %s placeholders, so literal percent signs must be doubled
        # before `?` becomes `%s`.
        sql = sql.replace("%", "%%").replace("?", "%s")
        sql = sql.replace("{DOUBLE}", "DOUBLE PRECISION").replace("{VECTOR}", f"vector({self.dim})")
        sql = sql.replace("{VEC}", "%s::vector")
        return _DIST.sub(lambda m: f"({m.group(1)} <=> %s::vector)", sql)

    def vec(self, v):
        return json.dumps([float(x) for x in v]) if v is not None else None

    def vec_out(self, v):
        if v is None:
            return None
        if isinstance(v, str):
            return [float(x) for x in json.loads(v)]
        return [float(x) for x in v]

    def execute(self, sql, params=()):
        return self._con.execute(self.render(sql), list(params))

    def executemany(self, sql, rows):
        rows = [list(r) for r in rows]
        if rows:
            with self._con.cursor() as cur:
                cur.executemany(self.render(sql), rows)

    def columns(self, table):
        rows = self._con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = current_schema()", [table]).fetchall()
        return {r[0] for r in rows}

    def vector_index(self, table, column):
        # pgvector's HNSW index holds up to 2000 dimensions.
        if self.dim <= 2000:
            self._con.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_{column}_hnsw ON {table} "
                f"USING hnsw ({column} vector_cosine_ops)")

    def vector_dim(self, table, column):
        row = self._con.execute(
            "SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relname = %s AND a.attname = %s AND n.nspname = current_schema()",
            [table, column]).fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None

    def close(self):
        self._con.close()


def connect(*, database_url: str | None, path: str, dim: int,
            schema: str | None = None) -> Db:
    """Postgres when a database URL is configured, else a DuckDB file at `path`."""
    if database_url:
        return PostgresDb(database_url, dim, schema)
    return DuckDb(path, dim)
