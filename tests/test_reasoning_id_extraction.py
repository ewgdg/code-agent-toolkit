from src.claude_router.adapters.langchain_openai_response_adapter import (
    LangChainOpenAIResponseAdapter,
)
from langchain_core.messages import AIMessage


def test_reasoning_id_from_content_item_propagates_to_thinking_block():
    adapter = LangChainOpenAIResponseAdapter()

    # Simulate LangChain responses/v1 content with reasoning that includes an id
    rs_id = "rs_abc123"
    message = AIMessage(
        content=[
            {
                "type": "reasoning",
                "id": rs_id,
                "summary": [{"type": "output_text", "text": "hidden chain"}],
            },
            {"type": "text", "text": "final answer"},
        ]
    )

    resp = adapter._non_stream_response(message, headers=None, use_responses_api=True)
    thinking = [b for b in resp["content"] if isinstance(b, dict) and b.get("type") == "thinking"]
    assert thinking, "expected thinking block"
    assert all(b.get("extracted_openai_rs_id") == rs_id for b in thinking)
