import json
from pathlib import Path
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .adapters import (
    AnthropicOpenAIRequestAdapter,
    OpenAIAnthropicResponseAdapter,
    PassthroughAdapter,
)
from .config import ConfigLoader
from .router import ModelRouter

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ProxyRouter:
    def __init__(self, config_path: Path | None = None):
        self.config_loader = ConfigLoader(config_path, enable_hot_reload=True)
        self.config = self.config_loader.get_config()

        # Initialize components
        self.router = ModelRouter(self.config)
        self.request_adapter = AnthropicOpenAIRequestAdapter(self.config, self.router)
        self.response_adapter = OpenAIAnthropicResponseAdapter(self.config)
        self.passthrough_adapter = PassthroughAdapter(self.config)

        # Setup FastAPI app
        self.app = FastAPI(
            title="Claude Code Model Router",
            description="Proxy router for Claude Code CLI traffic with OpenAI translation",
            version="1.0.0",
        )

        # Add routes
        self._setup_routes()

        # Configure logging level from config
        log_level = self.config.logging.level.upper()
        log_level_num = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}.get(
            log_level, 20
        )

        # Reconfigure structlog with the config level
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level_num),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _setup_routes(self):
        """Setup FastAPI routes."""

        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "version": "1.0.0"}

        # Main proxy endpoint - catch all routes
        @self.app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        )
        async def proxy_request(request: Request, path: str):
            return await self._handle_request(request, path)

    async def _handle_request(self, request: Request, path: str) -> Response:
        """Handle incoming request and route appropriately."""

        # Get request details
        method = request.method
        headers = dict(request.headers)
        query_params = dict(request.query_params)
        body = await request.body()

        # Generate or preserve request ID
        request_id = headers.get("x-request-id", f"req_{id(request)}")
        headers["x-request-id"] = request_id

        logger.info(
            "Incoming request",
            method=method,
            path=path,
            request_id=request_id,
            user_agent=headers.get("user-agent", ""),
        )

        try:
            # Parse request body for routing decisions (if applicable)
            request_data = {}
            if body and method in ["POST", "PUT", "PATCH"]:
                try:
                    request_data = json.loads(body.decode())
                except json.JSONDecodeError:
                    # Not JSON, proceed with passthrough
                    pass

            # Make routing decision
            decision = self.router.decide_route(headers, request_data)

            logger.info(
                "Routing decision",
                request_id=request_id,
                target=decision.target,
                model=decision.model,
                reason=decision.reason,
            )

            # Route request
            if decision.target == "openai":
                return await self._handle_openai_request(
                    request_data, decision.model, headers, request_id
                )
            else:
                return await self._handle_passthrough_request(
                    method, path, headers, body, query_params, request_id
                )

        except Exception as e:
            logger.error(
                "Request handling error",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _handle_openai_request(
        self,
        request_data: dict[str, Any],
        target_model: str,
        headers: dict[str, str],
        request_id: str,
    ) -> Response:
        """Handle request routed to OpenAI."""

        try:
            # Translate request
            openai_request = await self.request_adapter.adapt_request(
                request_data, target_model
            )

            # Make OpenAI request
            response = await self.request_adapter.make_request(openai_request, headers)

            # Handle streaming vs non-streaming
            if openai_request.get("stream", False):
                return await self._handle_openai_streaming(response, request_id)
            else:
                return await self._handle_openai_non_streaming(response, request_id)

        except Exception as e:
            logger.error(
                "OpenAI request error",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Map common errors to appropriate HTTP status codes
            if "api key" in str(e).lower():
                raise HTTPException(status_code=401, detail="Invalid API key")
            elif "timeout" in str(e).lower():
                raise HTTPException(status_code=504, detail="Gateway timeout")
            else:
                raise HTTPException(status_code=502, detail="Bad gateway")

    async def _handle_openai_streaming(
        self, response: httpx.Response, request_id: str
    ) -> StreamingResponse:
        """Handle streaming OpenAI response."""

        if response.status_code != 200:
            logger.error(
                "OpenAI API error",
                request_id=request_id,
                status_code=response.status_code,
                response=response.text,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {response.text}",
            )

        async def stream_generator():
            try:
                async for anthropic_line in self.response_adapter.adapt_stream(
                    response.aiter_lines()
                ):
                    yield anthropic_line
            except Exception as e:
                logger.error("Streaming error", request_id=request_id, error=str(e))
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"x-request-id": request_id},
        )

    async def _handle_openai_non_streaming(
        self, response: httpx.Response, request_id: str
    ) -> Response:
        """Handle non-streaming OpenAI response."""

        if response.status_code != 200:
            logger.error(
                "OpenAI API error",
                request_id=request_id,
                status_code=response.status_code,
                response=response.text,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {response.text}",
            )

        # Translate response
        openai_data = response.json()
        anthropic_response = await self.response_adapter.adapt_response(openai_data)

        return Response(
            content=json.dumps(anthropic_response),
            media_type="application/json",
            headers={"x-request-id": request_id},
        )

    async def _handle_passthrough_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        query_params: dict[str, str],
        request_id: str,
    ) -> Response:
        """Handle request passed through to Anthropic."""

        try:
            # Forward request
            response = await self.passthrough_adapter.forward_request(
                method, f"/{path}", headers, body, query_params
            )

            # Get filtered response headers
            response_headers = await self.passthrough_adapter.get_response_headers(
                response
            )
            response_headers["x-request-id"] = request_id

            # Handle streaming vs non-streaming
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                return await self._handle_passthrough_streaming(
                    response, response_headers
                )
            else:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

        except Exception as e:
            logger.error(
                "Passthrough request error",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )

            if "timeout" in str(e).lower():
                raise HTTPException(status_code=504, detail="Gateway timeout")
            else:
                raise HTTPException(status_code=502, detail="Bad gateway")

    async def _handle_passthrough_streaming(
        self, response: httpx.Response, response_headers: dict[str, str]
    ) -> StreamingResponse:
        """Handle streaming passthrough response."""

        async def stream_generator():
            async for chunk in self.passthrough_adapter.stream_response(response):
                yield chunk

        return StreamingResponse(
            stream_generator(),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    async def startup(self):
        """Startup tasks."""
        logger.info(
            "Starting Claude Code Model Router",
            config_path=str(self.config_loader.config_path),
            listen=self.config.router.listen,
        )

    async def shutdown(self):
        """Cleanup tasks."""
        logger.info("Shutting down Claude Code Model Router")

        # Close adapters
        await self.request_adapter.close()
        await self.passthrough_adapter.close()

        # Stop config hot reload
        self.config_loader.stop_hot_reload()


def create_app(config_path: Path) -> FastAPI:
    """Create FastAPI application."""
    proxy = ProxyRouter(config_path)

    # Add lifecycle events
    @proxy.app.on_event("startup")
    async def startup():
        await proxy.startup()

    @proxy.app.on_event("shutdown")
    async def shutdown():
        await proxy.shutdown()

    return proxy.app


def main():
    """Main entry point."""
    import argparse
    from dotenv import load_dotenv

    # Load environment variables from .env file (in project root)
    # import os
    # project_root = Path(__file__).parent.parent.parent
    # dotenv_path = project_root / ".env"
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Claude Code Model Router")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("router.yaml"),
        help="Path to configuration file (default: router.yaml)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8787, help="Port to bind to (default: 8787)"
    )

    args = parser.parse_args()

    # Create app and get config
    app = create_app(args.config)

    # Load config to get listen address
    from .config import ConfigLoader

    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Parse listen address from config
    listen_parts = config.router.listen.split(":")
    host = listen_parts[0] if listen_parts[0] != "0.0.0.0" else args.host
    port = int(listen_parts[1]) if len(listen_parts) > 1 else args.port

    # Run server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
