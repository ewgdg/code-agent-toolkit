from __future__ import annotations

from typing import Any

import structlog

from ..config.schema import ToolPolicyConfig

logger = structlog.get_logger(__name__)


def filter_tools_in_request(
    request_data: dict[str, Any],
    policy: ToolPolicyConfig,
) -> dict[str, Any]:
    """Filter restricted tools from an Anthropic Messages request in place."""
    tools = request_data.get("tools") or []
    if not tools:
        return request_data

    # Build case-insensitive set of restricted names
    restricted = {t.lower() for t in (policy.restricted_tool_names or [])}
    if not restricted:
        return request_data

    # Separate tools
    blocked: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for tool in tools:
        name = str(tool.get("name", "")).lower()
        if name in restricted:
            blocked.append(tool)
        else:
            allowed.append(tool)

    if not blocked:
        # Nothing to filter
        return request_data

    blocked_names = [str(t.get("name", "")) for t in blocked]
    logger.info("Blocking restricted tools", blocked=blocked_names)

    request_data["tools"] = allowed
    return request_data
