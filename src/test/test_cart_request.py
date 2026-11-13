"""Unit tests for cart_request() in src/mcp_server.py.

Run from the Zivo AI repository root:
    py -m pytest src/test/test_cart_request.py -v

The MCP server's Chroma and Ollama objects are stubbed during import so these
unit tests do not initialize a real vector store or contact Ollama, Laravel,
or PostgreSQL.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


SRC_DIR = Path(__file__).resolve().parents[1]
if not (SRC_DIR / "mcp_server.py").is_file():
    SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# mcp_server.py creates Chroma and Ollama objects at import time. Replace only
# those constructors so importing the module has no external side effects.
fake_chroma_module = ModuleType("langchain_chroma")
fake_chroma_module.Chroma = MagicMock(name="Chroma")

fake_ollama_module = ModuleType("langchain_ollama")
fake_ollama_module.OllamaEmbeddings = MagicMock(name="OllamaEmbeddings")

with patch.dict(
    sys.modules,
    {
        "langchain_chroma": fake_chroma_module,
        "langchain_ollama": fake_ollama_module,
    },
):
    mcp_server = importlib.import_module("mcp_server")


MISSING = object()


def make_response(
    status_code: int,
    *,
    payload=MISSING,
    content: bytes = b"response-body",
):
    """Create a requests.Response-like mock."""
    response = MagicMock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.content = content

    if payload is MISSING:
        response.json.side_effect = ValueError("Invalid JSON")
    else:
        response.json.return_value = payload

    return response


def decode_result(result: str) -> dict:
    return json.loads(result)


@pytest.mark.parametrize("backend_url", [None, "", "   "])
def test_cart_request_rejects_missing_backend_url(monkeypatch, backend_url):
    request_mock = MagicMock()
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", backend_url)

    result = decode_result(
        mcp_server.cart_request("GET", "/api/cart", "valid-token")
    )

    assert result["success"] is False
    assert result["code"] == "SERVICE_NOT_CONFIGURED"
    request_mock.assert_not_called()


@pytest.mark.parametrize("token", [None, "", "   ", 123])
def test_cart_request_rejects_invalid_token(monkeypatch, token):
    request_mock = MagicMock()
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    result = decode_result(
        mcp_server.cart_request("GET", "/api/cart", token)
    )

    assert result["success"] is False
    assert result["code"] == "AUTH_REQUIRED"
    request_mock.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (401, "AUTH_REQUIRED"),
        (403, "FORBIDDEN"),
        (404, "NOT_FOUND"),
        (422, "VALIDATION_ERROR"),
        (500, "SERVICE_ERROR"),
        (503, "SERVICE_ERROR"),
    ],
)
def test_cart_request_maps_http_errors(
    monkeypatch,
    status_code,
    expected_code,
):
    response = make_response(status_code)
    request_mock = MagicMock(return_value=response)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    result = decode_result(
        mcp_server.cart_request("GET", "/api/cart", "valid-token")
    )

    assert result["success"] is False
    assert result["code"] == expected_code
    assert result["status_code"] == status_code
    response.json.assert_not_called()


def test_cart_request_returns_backend_json_on_success(monkeypatch):
    payload = {
        "success": True,
        "data": {"items": [{"product_id": 42, "quantity": 2}]},
    }
    response = make_response(200, payload=payload)
    request_mock = MagicMock(return_value=response)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test/")

    result = decode_result(
        mcp_server.cart_request("GET", "/api/cart", "valid-token")
    )

    assert result == payload
    request_mock.assert_called_once()
    response.json.assert_called_once_with()


def test_cart_request_treats_http_204_as_success_without_json(monkeypatch):
    response = make_response(204, content=b"")
    request_mock = MagicMock(return_value=response)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    result = decode_result(
        mcp_server.cart_request(
            "DELETE",
            "/api/cart/items/42",
            "valid-token",
        )
    )

    assert result["success"] is True
    assert result["status_code"] == 204
    response.json.assert_not_called()


def test_cart_request_maps_invalid_success_json(monkeypatch):
    response = make_response(200, content=b"not-json")
    request_mock = MagicMock(return_value=response)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    result = decode_result(
        mcp_server.cart_request("GET", "/api/cart", "valid-token")
    )

    assert result["success"] is False
    assert result["code"] == "INVALID_RESPONSE"


@pytest.mark.parametrize(
    ("method", "expected_code"),
    [
        ("GET", "SERVICE_TIMEOUT"),
        ("POST", "RESULT_UNKNOWN"),
        ("DELETE", "RESULT_UNKNOWN"),
    ],
)
def test_cart_request_handles_timeout_by_method(
    monkeypatch,
    method,
    expected_code,
):
    request_mock = MagicMock(side_effect=mcp_server.requests.Timeout)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    result = decode_result(
        mcp_server.cart_request(method, "/api/cart", "valid-token")
    )

    assert result["success"] is False
    assert result["code"] == expected_code
    request_mock.assert_called_once()


def test_cart_request_handles_network_error(monkeypatch):
    request_mock = MagicMock(
        side_effect=mcp_server.requests.ConnectionError("connection refused")
    )
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    result = decode_result(
        mcp_server.cart_request("GET", "/api/cart", "valid-token")
    )

    assert result["success"] is False
    assert result["code"] == "SERVICE_UNAVAILABLE"


def test_cart_request_sends_expected_request(monkeypatch):
    response = make_response(200, payload={"success": True})
    request_mock = MagicMock(return_value=response)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test/")

    mcp_server.cart_request(
        "POST",
        "/api/cart/items",
        "valid-token",
        json={"product_id": 42, "quantity": 2},
    )

    request_mock.assert_called_once_with(
        method="POST",
        url="https://api.example.test/api/cart/items",
        headers={
            "Authorization": "Bearer valid-token",
            "Accept": "application/json",
        },
        timeout=15,
        json={"product_id": 42, "quantity": 2},
    )


def test_cart_request_normalizes_lowercase_http_method(monkeypatch):
    response = make_response(200, payload={"success": True})
    request_mock = MagicMock(return_value=response)
    monkeypatch.setattr(mcp_server.requests, "request", request_mock)
    monkeypatch.setattr(mcp_server, "backend_url", "https://api.example.test")

    mcp_server.cart_request("get", "/api/cart", "valid-token")

    request_mock.assert_called_once()
    assert request_mock.call_args.kwargs["method"] == "GET"
