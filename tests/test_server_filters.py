import json
from pathlib import Path

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from src.claude_router.config.schema import Config, ProviderConfig
from src.claude_router.server import create_app


class _FakeLoader:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self.config_path = Path("config/router.yaml")

    def get_config(self) -> Config:
        return self._cfg


def _req_with_tools(names: list[str]) -> dict:
    return {
        "model": "gpt-any",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": n, "input_schema": {}} for n in names],
    }


@pytest.mark.asyncio
async def test_server_filters_tools_for_openai_adapter(monkeypatch: pytest.MonkeyPatch):
    # Config with both providers; override will send to OpenAI
    cfg = Config(
        providers={
            "openai": ProviderConfig(
                base_url="https://api.openai.test", adapter="openai"
            ),
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com", adapter="anthropic-passthrough"
            ),
        },
        overrides=[
            {
                "when": {"request": {"model_regex": ".*"}},
                "provider": "openai",
                "model": "gpt-any",
            }
        ],
    )

    captured = {}

    async def fake_handle_request(self, request_data, decision, headers, request_id):
        nonlocal captured
        captured = request_data
        return JSONResponse({"ok": True})

    # Patch the adapter method to capture filtered request
    from src.claude_router.adapters.unified_langchain_adapter import (
        UnifiedLangChainAdapter,
    )

    monkeypatch.setattr(
        UnifiedLangChainAdapter, "handle_request", fake_handle_request, raising=True
    )

    app = create_app(_FakeLoader(cfg))
    client = TestClient(app)

    resp = client.post("/v1/messages", json=_req_with_tools(["web_search", "helper"]))
    assert resp.status_code == 200

    # Ensure restricted tool was stripped by the server before reaching adapter
    tool_names = [t["name"].lower() for t in captured.get("tools", [])]
    assert "web_search" not in tool_names
    assert "helper" in tool_names


def test_server_filters_tools_for_passthrough_adapter(monkeypatch: pytest.MonkeyPatch):
    # Default routing goes to anthropic-passthrough without overrides
    cfg = Config(
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com", adapter="anthropic-passthrough"
            )
        }
    )

    seen = {}

    async def fake_passthrough(self, method, path, headers, body, query_params):
        nonlocal seen
        payload = json.loads(body.decode()) if body else {}
        seen = payload
        return Response(content=b"{}", media_type="application/json")

    from src.claude_router.adapters import PassthroughAdapter

    monkeypatch.setattr(
        PassthroughAdapter, "handle_request", fake_passthrough, raising=True
    )

    app = create_app(_FakeLoader(cfg))
    client = TestClient(app)

    resp = client.post("/v1/messages", json=_req_with_tools(["web_search", "helper"]))
    assert resp.status_code == 200

    # Server should have stripped restricted tool before re-encoding body
    tool_names = [t["name"].lower() for t in seen.get("tools", [])]
    assert "web_search" not in tool_names
    assert "helper" in tool_names

