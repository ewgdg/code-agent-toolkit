from collections.abc import AsyncIterator

import httpx
import structlog
from fastapi.responses import StreamingResponse

from ..config import Config

logger = structlog.get_logger(__name__)


class PassthroughAdapter:
    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                30.0,
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
    ) -> StreamingResponse:
        """Forward request to original Anthropic endpoint with streaming."""

        # Build target URL
        url = f"{self.config.router.original_base_url}{path}"

        # Forward all headers for true passthrough transparency
        forwarded_headers = dict(headers)

        # Strip hop-by-hop headers that should not be forwarded between proxies
        forwarded_headers = self._strip_hop_by_hop_headers(forwarded_headers)

        # Sanitize sensitive headers for logging only
        safe_headers = self._sanitize_headers_for_logging(forwarded_headers)
        logger.info(
            "Forwarding request to Anthropic",
            method=method,
            url=url,
            headers=list(safe_headers.keys()),
        )

        # Build request for streaming
        request = self.client.build_request(
            method=method,
            url=url,
            headers=forwarded_headers,
            content=body,
            params=query_params,
        )

        # Send with streaming enabled - never buffer upstream
        response = await self.client.send(request, stream=True)

        return StreamingResponse(
            self._stream_generator(response),
            media_type=response.headers.get("content-type", "text/plain"),
            headers=self._filter_headers(response.headers),
            status_code=response.status_code,
        )

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

    def _strip_hop_by_hop_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Strip hop-by-hop headers that should not be forwarded between proxies.

        Based on RFC 7230 Section 6.1, these headers are meant for single-hop
        connections and can cause issues when proxying requests.
        """
        # Standard hop-by-hop headers defined in RFC 7230
        hop_by_hop_headers = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "host",  # Host header should point to the target server
        }

        # Additional headers that are commonly problematic in proxy scenarios
        proxy_headers = {
            "proxy-connection",  # Non-standard but used by some clients
            "content-length",  # Let httpx handle this automatically
            "content-encoding",  # Let httpx handle compression
        }

        # Combine all headers to strip
        headers_to_strip = hop_by_hop_headers | proxy_headers

        # Filter out hop-by-hop headers (case-insensitive)
        filtered_headers = {}
        for name, value in headers.items():
            if name.lower() not in headers_to_strip:
                filtered_headers[name] = value

        return filtered_headers

    async def stream_response(self, response: httpx.Response) -> AsyncIterator[bytes]:
        """Stream response from Anthropic API."""

        async for chunk in response.aiter_bytes():
            yield chunk

    async def get_response_headers(self, response: httpx.Response) -> dict[str, str]:
        """Get filtered response headers, stripping hop-by-hop headers."""
        raw_headers = dict(response.headers)
        return self._strip_hop_by_hop_headers(raw_headers)

    async def _stream_generator(self, response: httpx.Response) -> AsyncIterator[bytes]:
        """Generate streaming response chunks."""
        try:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            if not response.is_closed:
                await response.aclose()

    def _filter_headers(self, headers: httpx.Headers) -> dict[str, str]:
        """Filter response headers for streaming response."""
        # Convert to dict and strip hop-by-hop headers
        raw_headers = dict(headers)
        return self._strip_hop_by_hop_headers(raw_headers)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
