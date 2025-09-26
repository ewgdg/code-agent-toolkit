"""Helpers for filtering or sanitizing prompt content in Anthropic-style requests."""

from __future__ import annotations

from typing import Any

from ..config.schema import ClauseFilterRule, SystemPromptFilterConfig


def filter_system_prompt_in_request(
    request_data: dict[str, Any],
    filters: SystemPromptFilterConfig,
) -> dict[str, Any]:
    """Apply configured clause filters to the top-level system prompt in place."""

    system_content = request_data.get("system")
    if system_content is None:
        return request_data

    clause_filters = filters.clause_filters
    if not clause_filters:
        return request_data

    def apply_to_text(text: str, rules: tuple[ClauseFilterRule, ...]) -> str:
        result = text
        for rule in rules:
            if pattern := rule.compiled_pattern:
                result = pattern.sub("", result)
            else:
                result = result.replace(rule.pattern, "")
        return result

    if isinstance(system_content, str):
        filtered = apply_to_text(system_content, clause_filters)
        if filtered and filtered.strip():
            request_data["system"] = filtered
        else:
            request_data.pop("system", None)
        return request_data

    if isinstance(system_content, list):
        filtered_blocks: list[Any] = []
        for block in system_content:
            if isinstance(block, dict):
                block_copy = dict(block)
                text_value = block_copy.get("text")
                if isinstance(text_value, str):
                    filtered_text = apply_to_text(text_value, clause_filters)
                    if filtered_text and filtered_text.strip():
                        block_copy["text"] = filtered_text
                        filtered_blocks.append(block_copy)
                    else:
                        block_copy.pop("text", None)
                        if block_copy:
                            filtered_blocks.append(block_copy)
                else:
                    filtered_blocks.append(block_copy)
            elif isinstance(block, str):
                filtered_text = apply_to_text(block, clause_filters)
                if filtered_text and filtered_text.strip():
                    filtered_blocks.append(filtered_text)
            else:
                filtered_blocks.append(block)

        if filtered_blocks:
            request_data["system"] = filtered_blocks
        else:
            request_data.pop("system", None)
        return request_data

    return request_data
