import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from src.claude_router.adapters.unified_langchain_adapter import (
    UnifiedLangChainAdapter,
)
from src.claude_router.config.schema import Config, ProviderConfig
from src.claude_router.router import ModelRouter, RouterDecision


@pytest.mark.asyncio
async def test_non_anthropic_forces_unauthenticated(monkeypatch: pytest.MonkeyPatch):
    # Config with provider-level default policies
    cfg = Config(
        providers={
            "openai": ProviderConfig(
                base_url="https://api.openai.test",
                adapter="openai",
            ),
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                adapter="anthropic-passthrough",
            ),
        }
    )

    router = ModelRouter(cfg)
    adapter = UnifiedLangChainAdapter(cfg, router)

    # Track the anthropic_request that reaches adapt_request
    captured: dict[str, Any] = {}

    async def fake_adapt_request(**kwargs):
        nonlocal captured
        captured = kwargs.get("anthropic_request", {})
        # Return a minimal adapted_request stub
        return {
            "model": kwargs.get("model", "model"),
            "messages": [],
            "params": {},
            "tools": [],
            "stream": False,
        }

    async def fake_make_request(*args, **kwargs):
        # Return a minimal AIMessage, avoiding any network
        return AIMessage(content="ok")

    async def fake_adapt_response(*args, **kwargs):
        # Non-streaming JSON payload
        return {"id": "x", "type": "message", "role": "assistant", "content": []}

    monkeypatch.setattr(adapter.request_adapter, "adapt_request", fake_adapt_request)
    monkeypatch.setattr(adapter.request_adapter, "make_request", fake_make_request)
    monkeypatch.setattr(adapter.response_adapter, "adapt_response", fake_adapt_response)

    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "web_search", "input_schema": {}},
            {"name": "helper", "input_schema": {}},
        ],
    }
    decision = RouterDecision(
        model="gpt-test", provider="openai", adapter="openai", reason="test"
    )

    resp = await adapter.handle_request(
        request,
        decision,
        headers={"X-User-Authenticated": "true"},
        request_id="r1",
    )
    assert resp.status_code == 200

    # Ensure restricted tool is stripped before reaching adapt_request
    tool_names = [t["name"].lower() for t in captured.get("tools", [])]
    assert "web_search" not in tool_names
    assert "helper" in tool_names
