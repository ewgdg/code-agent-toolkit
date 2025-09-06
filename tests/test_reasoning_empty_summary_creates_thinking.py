from langchain_core.messages import AIMessage

from src.claude_router.adapters.langchain_openai_response_adapter import (
    LangChainOpenAIResponseAdapter,
)


def test_reasoning_with_id_and_empty_summary_emits_thinking_block():
    adapter = LangChainOpenAIResponseAdapter()
    rs_id = "rs_empty_1"
    message = AIMessage(
        content=[
            {"type": "reasoning", "id": rs_id, "summary": []},
            {"type": "text", "text": "answer"},
        ]
    )

    resp = adapter._non_stream_response(message, headers=None, use_responses_api=True)
    thinking = [
        b
        for b in resp["content"]
        if isinstance(b, dict) and b.get("type") == "thinking"
    ]
    assert thinking, (
        "should create a thinking block even with empty summary when id present"
    )
    assert thinking[0].get("extracted_openai_rs_id") == rs_id
    assert thinking[0].get("thinking") == ""
