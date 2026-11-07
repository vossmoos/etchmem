"""
Test configuration.

By default every test runs on embedded DuckDB in a temporary folder. Set
ETCHMEM_TEST_DATABASE_URL to run the same suite against PostgreSQL + pgvector:
each test then gets its own schema per data folder (a test that opens a
second, fresh store in another folder gets a second schema, just as it would
get a second pair of DuckDB files), all dropped afterwards.

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=etchmem \\
        -e POSTGRES_USER=etchmem pgvector/pgvector:pg17
    ETCHMEM_TEST_DATABASE_URL=postgresql://etchmem:etchmem@localhost:55432/etchmem pytest
"""
from __future__ import annotations

import hashlib
import os
import uuid

import pytest

from app import config as _config

_PG_URL = os.environ.get("ETCHMEM_TEST_DATABASE_URL", "")


@pytest.fixture(autouse=True)
def _database_engine(monkeypatch):
    from app import stores as _stores

    settings = _config.settings
    saved = (settings.database_url, settings.database_schema, settings.api_key)
    settings.api_key = ""
    schemas: set[str] = set()
    if _PG_URL:
        run = uuid.uuid4().hex[:8]
        settings.database_url, settings.database_schema = _PG_URL, f"t_{run}"
        original = _stores.Stores.__init__

        def per_folder(self, data_dir, dim, *, database_url=None, schema=None):
            if database_url:
                schema = f"t_{run}_{hashlib.sha1(str(data_dir).encode()).hexdigest()[:8]}"
                schemas.add(schema)
            original(self, data_dir, dim, database_url=database_url, schema=schema)

        monkeypatch.setattr(_stores.Stores, "__init__", per_folder)
    else:
        settings.database_url, settings.database_schema = "", ""
    yield
    settings.database_url, settings.database_schema, settings.api_key = saved
    if schemas:
        import psycopg

        with psycopg.connect(_PG_URL, autocommit=True) as con:
            for schema in schemas:
                con.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
