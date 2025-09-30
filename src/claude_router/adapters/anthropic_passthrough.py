import json
from collections.abc import AsyncIterator
from typing import Any

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

    async def handle_request(
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

        # Clean request body to handle thinking blocks without signatures
        cleaned_body = self._clean_request_body(body)

        # Sanitize sensitive headers for logging only
        safe_headers = self._sanitize_headers_for_logging(forwarded_headers)
        logger.debug(
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
            content=cleaned_body,
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

    def _clean_request_body(self, body: bytes) -> bytes:
        """
        Clean request body by removing thinking blocks without valid signatures.

        Anthropic API requires thinking blocks to have signature fields for verification.
        Thinking blocks without signatures will cause 400 errors, so we strip them.
        """
        try:
            if not body:
                return body

            request_data = json.loads(body.decode())

            # Process messages if present
            if "messages" in request_data and isinstance(
                request_data["messages"], list
            ):
                original_messages = request_data["messages"]
                total_messages = len(original_messages)
                cleaned_messages: list[dict[str, Any]] = []

                for original_index, message in enumerate(original_messages):
                    content = message.get("content")
                    cleaned_content = self._clean_message_content(content)

                    if self._is_content_empty(cleaned_content):
                        if not self._should_keep_empty_message(
                            message, original_index, total_messages
                        ):
                            logger.debug(
                                "Dropping message with empty content after cleanup",
                                role=message.get("role"),
                                index=original_index,
                            )
                            continue

                    cleaned_message = dict(message)
                    cleaned_message["content"] = cleaned_content
                    cleaned_messages.append(cleaned_message)

                request_data["messages"] = cleaned_messages

            return json.dumps(request_data).encode()

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # If we can't parse the body, return it as-is
            logger.debug(
                "Could not parse request body for thinking block cleanup",
                error=str(e),
            )
            return body
        except Exception as e:
            # For any other errors, log and return original body
            logger.warning(
                "Error during request body cleaning",
                error=str(e),
            )
            return body

    def _clean_message_content(self, content: Any) -> Any:
        """Remove invalid thinking blocks from message content."""
        if isinstance(content, list):
            cleaned_content = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and not block.get("signature")
                ):
                    continue
                cleaned_content.append(block)
            return cleaned_content
        return content

    def _is_content_empty(self, content: Any) -> bool:
        """Determine if a message content payload is effectively empty."""
        if content is None:
            return True
        if isinstance(content, str):
            return not content.strip()
        if isinstance(content, list):
            if not content:
                return True
            return all(self._is_content_block_empty(block) for block in content)
        return False

    def _is_content_block_empty(self, block: Any) -> bool:
        """Check whether a content block lacks user-visible data."""
        if not isinstance(block, dict):
            return False

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            return not isinstance(text, str) or not text.strip()
        if block_type == "thinking":
            # Unsigned thinking blocks are filtered earlier; treat signed ones as meaningful
            return not block.get("signature")
        return False

    def _should_keep_empty_message(
        self, message: dict[str, Any], index: int, total_messages: int
    ) -> bool:
        """Allow empty content only for the optional final assistant message."""
        is_last = index == total_messages - 1
        return is_last and message.get("role") == "assistant"

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
