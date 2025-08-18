from collections.abc import AsyncIterator

import httpx
import structlog

from ..config import Config

logger = structlog.get_logger(__name__)


class PassthroughAdapter:
    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.timeouts_ms.connect / 1000,
                read=self.config.timeouts_ms.read / 1000,
            )
        )

    async def forward_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        query_params: dict[str, str],
    ) -> httpx.Response:
        """Forward request to original Anthropic endpoint with headers intact."""

        # Build target URL
        url = f"{self.config.router.original_base_url}{path}"

        # Forward all headers for true passthrough transparency
        forwarded_headers = dict(headers)

        # Sanitize sensitive headers for logging only
        safe_headers = self._sanitize_headers_for_logging(forwarded_headers)
        logger.info(
            "Forwarding request to Anthropic",
            method=method,
            url=url,
            headers=list(safe_headers.keys()),
        )

        # Make the request
        response = await self.client.request(
            method=method,
            url=url,
            headers=forwarded_headers,
            content=body,
            params=query_params,
        )

        return response

    def _sanitize_headers_for_logging(self, headers: dict[str, str]) -> dict[str, str]:
        """Sanitize headers by redacting sensitive values for logging."""

        sensitive_headers = {"authorization", "x-api-key", "x-openai-api-key"}

        sanitized = {}
        for name, value in headers.items():
            if name.lower() in sensitive_headers:
                # Redact but show prefix/suffix for debugging
                if len(value) > 10:
                    sanitized[name] = f"{value[:4]}...{value[-4:]}"
                else:
                    sanitized[name] = "[REDACTED]"
            else:
                sanitized[name] = value

        return sanitized

    async def stream_response(self, response: httpx.Response) -> AsyncIterator[bytes]:
        """Stream response from Anthropic API."""

        async for chunk in response.aiter_bytes():
            yield chunk

    async def get_response_headers(self, response: httpx.Response) -> dict[str, str]:
        """Get all response headers for true passthrough transparency."""
        return dict(response.headers)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
