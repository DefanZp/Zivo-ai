"""Unit tests for the FastAPI POST /chat/stream endpoint.

Copy this file to src/test/test_api_chat_stream.py, then run:

    py -m pytest src/test/test_api_chat_stream.py -v

The tests do not start the real FastAPI lifespan, PostgreSQL, MCP server,
Gemini, or Laravel. Agent and database dependencies are mocked.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, status


SRC_DIR = Path(__file__).resolve().parents[1]
if not (SRC_DIR / "api.py").is_file():
    SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault(
    "LANGGRAPH_DB_URI",
    "postgresql://unit-test:unit-test@localhost:5432/zivo_test",
)
os.environ.setdefault("BACKEND_URL", "https://backend.example.test")

# Avoid importing the real service/model setup in endpoint unit tests.
fake_service_module = ModuleType("service")
fake_service_module.ZivoAgentService = MagicMock(name="ZivoAgentService")

with patch.dict(sys.modules, {"service": fake_service_module}):
    api = importlib.import_module("api")


THREAD_ID = "thread-stream-test-123"
USER_MESSAGE = "Find a black hoodie"


def run_async(coroutine):
    return asyncio.run(coroutine)


def post_chat_stream(payload=None, *, headers=None):
    if payload is None:
        payload = {
            "thread_id": THREAD_ID,
            "user_message": USER_MESSAGE,
        }

    async def make_request():
        transport = httpx.ASGITransport(
            app=api.app,
            raise_app_exceptions=False,
        )

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post(
                "/chat/stream",
                json=payload,
                headers=headers,
            )

    return run_async(make_request())


async def make_chunks(*chunks):
    for chunk in chunks:
        yield chunk


@pytest.fixture
def agent_service(monkeypatch):
    service = SimpleNamespace(
        chat_stream=MagicMock(
            return_value=make_chunks("Hello", "", " from Zivo.")
        )
    )

    monkeypatch.setattr(
        api.app.state,
        "agent_service",
        service,
        raising=False,
    )
    return service


def test_chat_stream_returns_token_events_and_done(agent_service):
    response = post_chat_stream()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'event: token\ndata: {"content": "Hello"}\n\n'
        'event: token\ndata: {"content": " from Zivo."}\n\n'
        'event: done\ndata: {}\n\n'
    )

    agent_service.chat_stream.assert_called_once_with(
        f"guest:thread:{THREAD_ID}",
        USER_MESSAGE,
        None,
    )


def test_chat_stream_skips_empty_chunks(agent_service):
    agent_service.chat_stream.return_value = make_chunks("", "Hello", "")

    response = post_chat_stream()

    assert response.status_code == 200
    assert response.text.count("event: token") == 1
    assert 'data: {"content": "Hello"}' in response.text
    assert "event: done\ndata: {}" in response.text


def test_chat_stream_returns_done_when_agent_has_no_chunks(agent_service):
    agent_service.chat_stream.return_value = make_chunks()

    response = post_chat_stream()

    assert response.status_code == 200
    assert response.text == "event: done\ndata: {}\n\n"


def test_chat_stream_emits_error_event_when_agent_fails(agent_service):
    async def failing_chunks():
        yield "Partial response"
        raise RuntimeError("private provider failure details")

    agent_service.chat_stream.return_value = failing_chunks()

    response = post_chat_stream()

    assert response.status_code == 200
    assert 'data: {"content": "Partial response"}' in response.text
    assert "event: error" in response.text
    assert "Unable to stream AI response." in response.text
    assert "private provider failure details" not in response.text
    assert "event: done" not in response.text


def test_chat_stream_logs_agent_exception(agent_service, caplog):
    async def failing_chunks():
        raise RuntimeError("provider unavailable")
        yield "unreachable"

    agent_service.chat_stream.return_value = failing_chunks()

    with caplog.at_level("ERROR", logger="uvicorn.error"):
        response = post_chat_stream()

    assert response.status_code == 200
    assert "[STREAM] AI response stream failed" in caplog.text
    assert "provider unavailable" in caplog.text


def test_chat_stream_rejects_malformed_authorization_header(agent_service):
    response = post_chat_stream(
        headers={"Authorization": "invalid-header"}
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid Authorization header."
    agent_service.chat_stream.assert_not_called()


def test_chat_stream_uses_user_scoped_thread_and_records_activity(
    agent_service,
    monkeypatch,
):
    verify_token = AsyncMock(return_value=23)
    record_activity = AsyncMock()
    monkeypatch.setattr(api, "verify_user_token", verify_token)
    monkeypatch.setattr(api, "record_thread_activity", record_activity)

    response = post_chat_stream(
        headers={"Authorization": "Bearer test-user-token"}
    )

    assert response.status_code == 200
    verify_token.assert_awaited_once_with("test-user-token")
    record_activity.assert_awaited_once_with(
        db_uri=api.db_uri,
        effective_thread_id=f"user:23:thread:{THREAD_ID}",
        user_id=23,
        client_thread_id=THREAD_ID,
    )
    agent_service.chat_stream.assert_called_once_with(
        f"user:23:thread:{THREAD_ID}",
        USER_MESSAGE,
        "test-user-token",
    )


def test_chat_stream_propagates_authentication_http_error(
    agent_service,
    monkeypatch,
):
    verify_token = AsyncMock(
        side_effect=HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    )
    monkeypatch.setattr(api, "verify_user_token", verify_token)

    response = post_chat_stream(
        headers={"Authorization": "Bearer expired-token"}
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid or expired token."
    agent_service.chat_stream.assert_not_called()


def test_chat_stream_rejects_invalid_request_body(agent_service):
    response = post_chat_stream({
        "thread_id": "",
        "user_message": USER_MESSAGE,
    })

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    agent_service.chat_stream.assert_not_called()
