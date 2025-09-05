"""
Unified LangChain adapter that handles both request processing and response generation.
"""

import inspect
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import structlog
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from ..config import Config
from ..router import ModelRouter, RouterDecision
from .base import UnifiedRequestAdapter
from .langchain_openai_request_adapter import LangChainOpenAIRequestAdapter
from .langchain_openai_response_adapter import LangChainOpenAIResponseAdapter

logger = structlog.get_logger(__name__)


class UnifiedLangChainAdapter(UnifiedRequestAdapter):
    """Unified adapter that combines LangChain request and response adapters."""

    def __init__(self, config: Config, router: ModelRouter):
        self.config = config
        self.router = router
        self.request_adapter = LangChainOpenAIRequestAdapter(config, router)
        self.response_adapter = LangChainOpenAIResponseAdapter(config)

    async def handle_request(
        self,
        request_data: dict[str, Any],
        decision: RouterDecision,
        headers: dict[str, str],
        request_id: str,
    ) -> Response:
        """Handle complete LangChain OpenAI request."""
        # Get provider config
        provider_config = self.config.providers.get(decision.provider)
        if not provider_config:
            raise HTTPException(
                status_code=500,
                detail=f"Provider '{decision.provider}' not configured",
            )

        # Determine API type based on adapter type
        # True for "openai" adapter (official OpenAI API with Responses API support)
        # False for "openai-compatible" adapter (third-party APIs)
        use_responses_api = provider_config.adapter == "openai"

        adapted_request = await self.request_adapter.adapt_request(
            anthropic_request=request_data,
            provider_config=provider_config,
            model=decision.model,
            use_responses_api=use_responses_api,
            model_config=decision.model_config,
            support_reasoning=decision.support_reasoning,
        )

        # Make API call
        lc_response = await self.request_adapter.make_request(
            adapted_request, headers, provider_config, use_responses_api
        )
        # Handle the response using the LangChain response adapter
        response = await self.response_adapter.adapt_response(
            lc_response, use_responses_api=use_responses_api
        )

        # Check if it's streaming (AsyncIterator) or non-streaming
        if isinstance(response, AsyncIterator):
            # Streaming response
            async def stream_generator() -> AsyncGenerator[str]:
                try:
                    async for line in response:
                        yield line
                except Exception:
                    if inspect.isasyncgen(lc_response):
                        await lc_response.aclose()
                    raise

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={"x-request-id": request_id},
            )
        else:
            # Non-streaming response
            return Response(
                content=response,
                media_type="application/json",
                headers={"x-request-id": request_id},
            )

    async def close(self) -> None:
        """Clean up resources."""
        await self.request_adapter.close()
