"""Base adapter interfaces for unified request handling."""

from abc import ABC, abstractmethod
from typing import Any

from fastapi import Response

from claude_router.router import RouterDecision


class UnifiedRequestAdapter(ABC):
    """Base class for unified request adapters."""

    @abstractmethod
    async def handle_request(
        self,
        request_data: dict[str, Any],
        decision: RouterDecision,
        headers: dict[str, str],
        request_id: str,
    ) -> Response:
        """Handle a request and return a Response."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass
