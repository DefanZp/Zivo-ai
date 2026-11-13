"""Unit tests for the asynchronous thread activity helpers.

Copy this file to src/test/test_thread_activity.py, then run from the
Zivo AI repository root:

    py -m pytest src/test/test_thread_activity.py -v

All PostgreSQL connections are mocked. These tests do not require a running
PostgreSQL server or modify any real data.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# Make src/thread_activity.py importable when this test is run from the
# repository root after being copied to src/test/.
SRC_DIR = Path(__file__).resolve().parents[1]
if not (SRC_DIR / "thread_activity.py").is_file():
    # Also support running this downloaded file directly from /mnt/data after
    # the project's src directory is added to PYTHONPATH.
    SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import thread_activity  # noqa: E402


DB_URI = "postgresql://unit-test:unit-test@localhost:5432/zivo_test"


def run_async(coroutine):
    """Run one async helper in a fresh event loop for this unit test."""
    return asyncio.run(coroutine)


@pytest.fixture
def mocked_connection(monkeypatch):
    """Replace Psycopg's connection factory with an async mock."""
    connection = MagicMock()
    connection.execute = AsyncMock()
    connection.__aenter__ = AsyncMock(return_value=connection)
    connection.__aexit__ = AsyncMock(return_value=False)

    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(
        thread_activity.psycopg,
        "AsyncConnection",
        SimpleNamespace(connect=connect),
    )

    return connection, connect


def test_ensure_thread_activity_table_creates_table_and_index(mocked_connection):
    connection, connect = mocked_connection

    run_async(thread_activity.ensure_thread_activity_table(DB_URI))

    connect.assert_awaited_once_with(DB_URI)
    assert connection.execute.await_count == 2

    table_query = connection.execute.await_args_list[0].args[0]
    index_query = connection.execute.await_args_list[1].args[0]

    assert "CREATE TABLE IF NOT EXISTS ai_thread_activity" in table_query
    assert "thread_id TEXT PRIMARY KEY" in table_query
    assert "user_id BIGINT NOT NULL" in table_query
    assert "CREATE INDEX IF NOT EXISTS idx_ai_thread_activity_last_activity" in index_query
    assert "ON ai_thread_activity (last_activity_at)" in index_query

    connection.__aenter__.assert_awaited_once_with()
    connection.__aexit__.assert_awaited_once_with(None, None, None)


def test_record_thread_activity_inserts_expected_values(mocked_connection):
    connection, connect = mocked_connection

    run_async(
        thread_activity.record_thread_activity(
            db_uri=DB_URI,
            effective_thread_id="user:17:thread:abc",
            user_id=17,
            client_thread_id="abc",
        )
    )

    connect.assert_awaited_once_with(DB_URI)
    connection.execute.assert_awaited_once()

    query, parameters = connection.execute.await_args.args

    assert "INSERT INTO ai_thread_activity" in query
    assert "ON CONFLICT (thread_id)" in query
    assert "last_activity_at = NOW()" in query
    assert parameters == (
        "user:17:thread:abc",
        17,
        "abc",
    )

    connection.__aenter__.assert_awaited_once_with()
    connection.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize(
    "helper",
    [
        thread_activity.ensure_thread_activity_table,
        thread_activity.record_thread_activity,
    ],
)
def test_database_connection_error_is_propagated(helper, monkeypatch):
    connect = AsyncMock(side_effect=RuntimeError("database unavailable"))
    monkeypatch.setattr(
        thread_activity.psycopg,
        "AsyncConnection",
        SimpleNamespace(connect=connect),
    )

    if helper is thread_activity.ensure_thread_activity_table:
        operation = helper(DB_URI)
    else:
        operation = helper(
            db_uri=DB_URI,
            effective_thread_id="user:17:thread:abc",
            user_id=17,
            client_thread_id="abc",
        )

    with pytest.raises(RuntimeError, match="database unavailable"):
        run_async(operation)

    connect.assert_awaited_once_with(DB_URI)


def test_record_thread_activity_propagates_query_error(mocked_connection):
    connection, connect = mocked_connection
    connection.execute.side_effect = RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        run_async(
            thread_activity.record_thread_activity(
                db_uri=DB_URI,
                effective_thread_id="user:17:thread:abc",
                user_id=17,
                client_thread_id="abc",
            )
        )

    connect.assert_awaited_once_with(DB_URI)
    connection.execute.assert_awaited_once()
    connection.__aexit__.assert_awaited_once()


def test_ensure_thread_activity_table_stops_after_table_query_fails(mocked_connection):
    connection, connect = mocked_connection
    connection.execute.side_effect = RuntimeError("create table failed")

    with pytest.raises(RuntimeError, match="create table failed"):
        run_async(thread_activity.ensure_thread_activity_table(DB_URI))

    connect.assert_awaited_once_with(DB_URI)
    connection.execute.assert_awaited_once()
    connection.__aexit__.assert_awaited_once()
