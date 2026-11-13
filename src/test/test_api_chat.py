"""Unit tests for the FastAPI POST /chat endpoint.

Copy this file to src/test/test_api_chat.py, then run from the AI project root:

    py -m pytest src/test/test_api_chat.py -v

The tests do not start the FastAPI lifespan, PostgreSQL, MCP server, Gemini,
or Laravel. External service calls are mocked.
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


SRC_DIR = Path(__file__).resolve().parents[1]
if not (SRC_DIR / "api.py").is_file():
    # Also support running this downloaded test file after adding src to PYTHONPATH.
    SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# api.py reads these variables when imported. Use test-only values if they are
# not already configured in the environment.
os.environ.setdefault(
    "LANGGRAPH_DB_URI",
    "postgresql://unit-test:unit-test@localhost:5432/zivo_test",
)
os.environ.setdefault("BACKEND_URL", "https://backend.example.test")

# Importing the real service would initialize the agent/model module. The chat
# endpoint tests only need the service interface, so provide a lightweight stub.
fake_service_module = ModuleType("service")
fake_service_module.ZivoAgentService = MagicMock(name="ZivoAgentService")

with patch.dict(sys.modules, {"service": fake_service_module}):
    api = importlib.import_module("api")


THREAD_ID = "thread-test-123"
USER_MESSAGE = "Show my cart"


def run_async(coroutine):
    """Run an async HTTP request in a fresh event loop."""
    return asyncio.run(coroutine)


def post_chat(payload=None, *, headers=None):
    """Send a request to the ASGI app without entering its real lifespan."""
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
                "/chat",
                json=payload,
                headers=headers,
            )

    return run_async(make_request())


@pytest.fixture
def agent_service(monkeypatch):
    """Attach a mock chat service to the FastAPI application state."""
    service = SimpleNamespace(
        chat=AsyncMock(
            return_value={
                "messages": [SimpleNamespace(content="Hello from Zivo.")]
            }
        )
    )

    monkeypatch.setattr(
        api.app.state,
        "agent_service",
        service,
        raising=False,
    )
    return service


def test_chat_returns_agent_text_response(agent_service):
    response = post_chat()

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": THREAD_ID,
        "response": "Hello from Zivo.",
    }

    agent_service.chat.assert_awaited_once_with(
        f"guest:thread:{THREAD_ID}",
        USER_MESSAGE,
        None,
    )


def test_chat_normalizes_text_blocks(agent_service):
    agent_service.chat.return_value = {
        "messages": [
            SimpleNamespace(
                content=[
                    {"type": "text", "text": "First part"},
                    {"type": "image", "url": "ignored"},
                    {"type": "text", "text": "Second part"},
                ]
            )
        ]
    }

    response = post_chat()

    assert response.status_code == 200
    assert response.json()["response"] == "First part\nSecond part"


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        "not-a-dictionary",
        [],
        {},
        {"other": []},
        {"messages": None},
        {"messages": "not-a-list"},
        {"messages": []},
        {"messages": [SimpleNamespace()]},
        {"messages": [SimpleNamespace(content=None)]},
        {"messages": [SimpleNamespace(content=123)]},
        {"messages": [SimpleNamespace(content="   ")]},
        {"messages": [SimpleNamespace(content=[])]},
    ],
)
def test_chat_rejects_invalid_or_empty_agent_result(agent_service, invalid_result):
    agent_service.chat.return_value = invalid_result

    response = post_chat()

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to generate AI response."



def test_chat_hides_agent_exception_details(agent_service):
    agent_service.chat.side_effect = RuntimeError(
        "private provider detail and API key context"
    )

    response = post_chat()

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Unable to process your message. Please try again."
    )
    assert "private provider detail" not in response.text


def test_chat_logs_exception_for_server_diagnostics(agent_service, caplog):
    agent_service.chat.side_effect = RuntimeError("provider unavailable")

    with caplog.at_level("ERROR", logger="uvicorn.error"):
        response = post_chat()

    assert response.status_code == 500
    assert "[CHAT] AI agent execution failed" in caplog.text
    assert "provider unavailable" in caplog.text


def test_chat_rejects_malformed_authorization_header(agent_service):
    response = post_chat(headers={"Authorization": "invalid-header"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid Authorization header."
    agent_service.chat.assert_not_awaited()


def test_chat_uses_user_scoped_thread_and_records_activity(
    agent_service,
    monkeypatch,
):
    verify_token = AsyncMock(return_value=17)
    record_activity = AsyncMock()
    monkeypatch.setattr(api, "verify_user_token", verify_token)
    monkeypatch.setattr(api, "record_thread_activity", record_activity)

    response = post_chat(headers={"Authorization": "Bearer test-user-token"})

    assert response.status_code == 200
    verify_token.assert_awaited_once_with("test-user-token")
    record_activity.assert_awaited_once_with(
        db_uri=api.db_uri,
        effective_thread_id=f"user:17:thread:{THREAD_ID}",
        user_id=17,
        client_thread_id=THREAD_ID,
    )
    agent_service.chat.assert_awaited_once_with(
        f"user:17:thread:{THREAD_ID}",
        USER_MESSAGE,
        "test-user-token",
    )


def test_chat_rejects_invalid_request_body(agent_service):
    response = post_chat(
        {
            "thread_id": "",
            "user_message": USER_MESSAGE,
        }
    )

    assert response.status_code == 422
    agent_service.chat.assert_not_awaited()
