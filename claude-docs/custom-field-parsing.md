# Custom Field Parsing for OpenAI Responses

This document explains how the Claude Router now supports custom field parsing from OpenAI API responses, including fields like `reasoning_content` that are not part of the standard OpenAI response format.

## Overview

The custom field parsing system consists of two main components:

1. **`ChatOpenAIWithCustomFields`** - A custom ChatOpenAI subclass that extracts custom fields from OpenAI responses
2. **LangChain Response Adapter** - Enhanced to process custom fields from `additional_kwargs` into Anthropic-compatible format

## How It Works

### 1. Custom Field Extraction

The `ChatOpenAIWithCustomFields` class automatically extracts configured custom fields from OpenAI responses and makes them available in the LangChain message's `additional_kwargs`. This works for both streaming and non-streaming responses.

**Dynamic Extraction:**
```python
# No configuration needed! The system automatically:
# 1. Looks in both streaming and non-streaming paths:
#    - ["choices", 0, "message"]  # Non-streaming responses
#    - ["choices", 0, "delta"]    # Streaming responses
# 
# 2. Extracts ANY field that's not in the standard exclusion list:
STANDARD_OPENAI_FIELDS = {
    "content", "role", "name", "refusal",           # Standard message fields
    "tool_calls", "tool_call_id", "function_call", # Tool calling fields  
    "finish_reason", "index", "logprobs",          # Metadata fields
    "delta", "usage",                              # Streaming fields
}
#
# 3. Result: reasoning_content, thinking_content, custom_metadata, etc. 
#    are all extracted automatically without any configuration!
```

### 2. Response Adapter Processing

The LangChain response adapter then processes these custom fields from `additional_kwargs` and converts them to Anthropic-compatible blocks:

**Configuration:**
```python
CUSTOM_FIELD_MAPPING = {
    "reasoning_content": {"block_type": "thinking", "field_name": "thinking"},
    "thinking_content": {"block_type": "thinking", "field_name": "thinking"},
    "reasoning": {"block_type": "thinking", "field_name": "thinking"},
    "thinking": {"block_type": "thinking", "field_name": "thinking"},
}
```

## Usage Examples

### Example OpenAI Response with Custom Fields

```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "The answer is 42.",
            "reasoning_content": "I need to think about this step by step..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}
```

### Resulting Anthropic-Compatible Response

The custom field is automatically converted to a thinking block:

```json
{
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "model": "gpt-4",
    "content": [
        {
            "type": "thinking",
            "thinking": "I need to think about this step by step..."
        },
        {
            "type": "text",
            "text": "The answer is 42."
        }
    ],
    "stop_reason": "end_turn",
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}
```

## Streaming Support

The system fully supports streaming responses. Custom fields are extracted from streaming chunks and converted to appropriate delta events:

### Streaming Input (OpenAI format)
```json
{
  "choices": [{
    "delta": {
      "content": null,
      "reasoning_content": "Let me think..."
    }
  }]
}
```

### Streaming Output (Anthropic format)
```
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}

event: content_block_delta  
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think..."}}
```

## Configuration

### Adding New Custom Fields

To add support for a new custom field:

1. **Add to extraction configuration** in `custom_chatopenai.py`:
```python
CUSTOM_FIELD_EXTRACTION = {
    # existing fields...
    "my_custom_field": {
        "source_paths": [
            ["choices", 0, "message", "my_custom_field"],  # Non-streaming
            ["choices", 0, "delta", "my_custom_field"],    # Streaming
        ]
    },
}
```

2. **Add to response mapping** in `langchain_openai_response_adapter.py`:
```python
CUSTOM_FIELD_MAPPING = {
    # existing fields...
    "my_custom_field": {"block_type": "custom", "field_name": "custom_data"},
}
```

### Router Integration

The system is automatically integrated into the LangChain request adapter at `langchain_openai_request_adapter.py:129`:

```python
langchain_model = ChatOpenAIWithCustomFields(
    model=model,
    api_key=SecretStr(api_key) if api_key is not None else None,
    base_url=provider_config.base_url,
    # ... other parameters
)
```

## Key Features

- **Automatic Detection**: Custom fields are automatically detected and extracted
- **Streaming Support**: Works with both streaming and non-streaming responses  
- **Configurable**: Easy to add new custom fields by updating configuration
- **Type Safety**: Maintains type safety with proper error handling
- **Backward Compatible**: Standard OpenAI responses work unchanged

## Implementation Details

### Custom ChatOpenAI Class

The `ChatOpenAIWithCustomFields` class properly overrides key methods with `@override` decorators and uses shared utility methods to avoid code duplication:

**Override Methods:**
- `@override _create_chat_result()` - Extracts custom fields from non-streaming responses  
- `@override _convert_chunk_to_generation_chunk()` - Extracts custom fields from Chat Completions API streaming chunks

**Shared Utility Methods:**
- `_extract_custom_fields_from_dict()` - Generic extraction from any dict-like structure
- `_add_custom_fields_to_message()` - Adds custom fields to ChatResult messages
- `_add_custom_fields_to_chunk()` - Adds custom fields to streaming message chunks

All override methods use `super()` to call parent implementations first, then enhance with custom field functionality.

### Response Adapter Enhancement

The response adapter now includes:

- `_custom_field_block()` - Creates custom field blocks based on configuration
- `_send_custom_field_delta()` - Sends custom field streaming deltas
- Enhanced `_content_blocks_from_message()` and `_stream_response()` methods

This system enables seamless integration of custom OpenAI response fields while maintaining full compatibility with existing functionality.