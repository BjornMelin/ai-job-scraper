"""Local vLLM Service for health monitoring and model management.

This module provides health monitoring for vLLM services with correct endpoints
and OpenAI-compatible API access. Focused on service availability checking
and model listing without complex retry logic.
"""

from __future__ import annotations

import logging

from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LocalVLLMService:
    """vLLM service health monitoring."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize vLLM health monitoring service.

        Args:
            base_url: Base URL for vLLM service (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> bool:
        """Check if vLLM server is running.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                # CRITICAL: vLLM uses /health endpoint (not /api/version)
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available vLLM models.

        Returns:
            List of available model dictionaries
        """
        try:
            async with httpx.AsyncClient() as client:
                # Use OpenAI-compatible endpoint
                response = await client.get(f"{self.base_url}/v1/models", timeout=10.0)
                if response.status_code == 200:
                    return response.json().get("data", [])
                return []
        except Exception:
            return []

    async def is_model_available(self, model_name: str) -> bool:
        """Check if specific model is available.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available, False otherwise
        """
        models = await self.list_models()
        return any(model_name in model.get("id", "") for model in models)


# Global service instance
local_service = LocalVLLMService()
