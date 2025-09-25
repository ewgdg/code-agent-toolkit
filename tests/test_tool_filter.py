from src.claude_router.adapters.tool_filter import filter_tools_in_request
from src.claude_router.config.schema import ToolPolicyConfig


def _req_with_tools(names: list[str]):
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": n, "input_schema": {}} for n in names],
    }


def test_filter_strips_restricted_tools_case_insensitive():
    policy = ToolPolicyConfig(restricted_tool_names=["web_search", "web_fetch"])

    req = _req_with_tools(["web_search", "calculator", "WEB_FETCH"])

    out = filter_tools_in_request(req, policy)

    out_tools = [t["name"].lower() for t in out.get("tools", [])]
    assert set(out_tools) == {"calculator"}


def test_filter_returns_same_request_when_no_restricted_tools():
    policy = ToolPolicyConfig(restricted_tool_names=[])

    req = _req_with_tools(["tool_a", "tool_b"])
    out = filter_tools_in_request(req, policy)

    assert out["tools"] == req["tools"]
