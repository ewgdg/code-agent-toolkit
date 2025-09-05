import asyncio

from langchain_core.messages import AIMessage

from src.claude_router.adapters.langchain_openai_response_adapter import (
    LangChainOpenAIResponseAdapter,
)


def test_non_stream_response_does_not_invent_reasoning_id():
    adapter = LangChainOpenAIResponseAdapter()

    # Simulate an AIMessage with reasoning content but without an rs_* id surfaced
    message = AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [
                    {"type": "output_text", "text": "Internal chain-of-thought."}
                ],
            },
            {"type": "text", "text": "Visible answer."},
        ]
    )

    result = adapter._non_stream_response(message, headers=None, use_responses_api=True)

    # adapter generates its own message id; we don't assert its exact value
    assert result["type"] == "message"

    # ensure thinking block is present but WITHOUT id (since we didn't surface rs_*)
    think_blocks = [b for b in result["content"] if isinstance(b, dict) and b.get("type") == "thinking"]
    assert think_blocks, "expected a thinking block in content"
    assert all("id" not in b for b in think_blocks)
