import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .adapters import (
    PassthroughAdapter,
    UnifiedLangChainAdapter,
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
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.config = config_loader.get_config()

        # Initialize components
        self.router = ModelRouter(self.config)

        # Initialize adapters
        self.passthrough_adapter = PassthroughAdapter(self.config)

        # Initialize unified LangChain adapter
        self.unified_langchain_adapter = UnifiedLangChainAdapter(
            self.config, self.router
        )

        # Setup FastAPI app
        self.app = FastAPI(
            title="Claude Code Model Router",
            description="Proxy router for Claude Code CLI traffic with OpenAI",
            version="1.0.0",
            lifespan=self._lifespan,
        )

        # Add CORS middleware - allow all origins
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
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

    def _handle_adapter_error(
        self, e: Exception, request_id: str, context: str
    ) -> None:
        """Common error handling logic for adapters."""
        logger.exception(
            f"{context} error",
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

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        # Health check endpoint
        @self.app.get("/health")
        async def health_check() -> dict[str, str]:
            return {"status": "healthy", "version": "1.0.0"}

        # Main proxy endpoint - catch all routes
        @self.app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        )
        async def proxy_request(request: Request, path: str) -> Response:
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

        logger.debug(
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
                model=decision.model,
                provider=decision.provider,
                adapter=decision.adapter,
                reason=decision.reason,
                model_config=decision.model_config,
            )

            # Route request based on adapter type
            try:
                if decision.adapter == "anthropic-passthrough":
                    return await self.passthrough_adapter.handle_request(
                        method, f"/{path}", headers, body, query_params
                    )
                elif decision.adapter in ("openai", "openai-compatible"):
                    return await self.unified_langchain_adapter.handle_request(
                        request_data, decision, headers, request_id
                    )
                else:
                    logger.error(f"Unknown adapter type: {decision.adapter}")
                    raise HTTPException(status_code=500, detail="Unknown adapter type")
            except HTTPException:
                raise  # Re-raise HTTP exceptions as-is
            except Exception as e:
                self._handle_adapter_error(e, request_id, f"{decision.adapter} request")

        except Exception as e:
            logger.exception(
                "Request handling error",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(status_code=500, detail="Internal server error")
        return Response(status_code=500)

    async def startup(self) -> None:
        """Startup tasks."""
        logger.info(
            "Starting Claude Code Model Router",
            config_path=str(self.config_loader.config_path),
            listen=self.config.router.listen,
        )

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None]:
        """FastAPI lifespan context that runs startup tasks."""
        await self.startup()
        yield
        await self.passthrough_adapter.close()


def create_app(config_loader: ConfigLoader) -> FastAPI:
    """Create FastAPI application with given config loader."""
    proxy = ProxyRouter(config_loader)
    return proxy.app


def main() -> None:
    """Main entry point."""
    import argparse
    import signal
    from threading import Event

    from dotenv import load_dotenv

    # Load environment variables from .env file (in project root)
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Claude Code Model Router")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/router.yaml"),
        help="Path to configuration file (default: config/router.yaml)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8787, help="Port to bind to (default: 8787)"
    )

    args = parser.parse_args()

    # Global state for server control
    server_restart_event = Event()
    should_exit = Event()

    def restart_server():
        """Callback to trigger server restart on config change."""
        logger.info("Config changed, restarting server")
        server_restart_event.set()

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        should_exit.set()
        server_restart_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while not should_exit.is_set():
        try:
            # Create config loader with restart callback
            config_loader = ConfigLoader(
                args.config, enable_hot_reload=True, reload_callback=restart_server
            )
            config = config_loader.get_config()

            # Parse listen address from config
            listen_parts = config.router.listen.split(":")
            host = listen_parts[0] if listen_parts[0] != "0.0.0.0" else args.host
            port = int(listen_parts[1]) if len(listen_parts) > 1 else args.port

            # Create fresh app instance
            app = create_app(config_loader)

            logger.info(f"Starting server on {host}:{port}")

            # Run uvicorn server in thread with proper control
            import asyncio
            import threading

            # Create uvicorn server with clean shutdown control
            server_config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
                access_log=False,  # Reduce noise
            )
            server = uvicorn.Server(server_config)

            def run_server():
                """Run server in thread with asyncio event loop."""
                try:
                    asyncio.run(server.serve())
                except Exception as e:
                    logger.error(f"Server error: {e}")

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()

            # Wait for restart signal or shutdown
            server_restart_event.wait()
            server_restart_event.clear()

            if not should_exit.is_set():
                logger.info("Stopping server for restart")

                # Signal server to shutdown gracefully
                server.should_exit = True
                server_thread.join(timeout=5)

                # Clean up config loader
                config_loader.stop_hot_reload()

                logger.info("Server stopped, restarting...")
            else:
                logger.info("Shutting down server")
                server.should_exit = True
                server_thread.join(timeout=5)
                config_loader.stop_hot_reload()
                break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            should_exit.set()
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            should_exit.set()
            break

    logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
