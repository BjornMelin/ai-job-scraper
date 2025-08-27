"""Cloud AI Fallback Service using LiteLLM for unified provider access.

This module provides cloud-based AI services as a fallback when local resources
are unavailable or when tasks require advanced capabilities beyond local models.
Uses LiteLLM for unified access to multiple providers with automatic fallbacks.
"""

from __future__ import annotations

import logging
import time

from pathlib import Path
from typing import Any

import instructor
import yaml

from litellm import Router, acompletion, token_counter
from pydantic import BaseModel, Field

# Import streamlit with fallback for non-Streamlit environments
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    class _DummyStreamlit:
        @staticmethod
        def cache_data(**_kwargs):
            def decorator(wrapped_func):
                return wrapped_func

            return decorator

        @staticmethod
        def cache_resource(**_kwargs):
            def decorator(wrapped_func):
                return wrapped_func

            return decorator

    st = _DummyStreamlit()

from src.config import Settings

logger = logging.getLogger(__name__)


# Module-level cached functions for Streamlit compatibility
@st.cache_resource  # Cache router as resource (persistent connection)
def _create_litellm_router(config: dict[str, Any]) -> Router:
    """Create LiteLLM router with configuration (cached as resource).

    Uses Streamlit resource caching to ensure single router instance
    across all cloud AI operations.
    """
    return Router(
        model_list=config["model_list"],
        **config.get("litellm_settings", {}),
    )


@st.cache_resource  # Cache instructor client as resource
def _create_instructor_client() -> instructor.Instructor:
    """Create instructor client from LiteLLM (cached as resource).

    Uses Streamlit resource caching for singleton instructor client.
    """
    from litellm import completion

    return instructor.from_litellm(
        completion,
        mode=instructor.Mode.JSON,
    )


class CloudServiceHealth(BaseModel):
    """Health status model for cloud AI services."""

    status: str = Field(description="Service status: healthy, unhealthy, degraded")
    available_models: list[str] = Field(description="List of available model names")
    primary_provider: str = Field(description="Primary cloud provider in use")
    fallback_providers: list[str] = Field(description="Available fallback providers")
    response_time_ms: float = Field(description="Average response time in milliseconds")
    success_rate: float = Field(description="Success rate as percentage (0-100)")
    cost_per_1k_tokens: float = Field(description="Average cost per 1000 tokens")
    last_check: float = Field(description="Timestamp of last health check")


class CloudAIService:
    """Cloud AI service providing advanced capabilities via multiple providers.

    Features:
    - Unified access to multiple cloud providers via LiteLLM
    - Automatic fallback between providers
    - Cost optimization and budget tracking
    - Structured output support via Instructor
    - Request/response monitoring and metrics
    """

    def __init__(
        self, config_path: str | None = None, settings: Settings | None = None
    ) -> None:
        """Initialize the Cloud AI Service.

        Args:
            config_path: Path to LiteLLM configuration file
            settings: Application settings, defaults to Settings() if None
        """
        self.settings = settings or Settings()

        # Load LiteLLM configuration
        if config_path is None:
            config_path = "config/litellm.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize LiteLLM router with loaded configuration
        self.router = _create_litellm_router(self.config)

        # Create instructor client from LiteLLM
        self.instructor_client = _create_instructor_client()

        # Service metrics
        self._request_count = 0
        self._success_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._response_times: list[float] = []
        self._last_health_check = 0.0
        self._health_check_interval = 60.0  # seconds
        self._is_healthy = True

        logger.info(
            "Cloud AI Service initialized with %d models",
            len(self.config["model_list"]),
        )

    def _load_config(self) -> dict[str, Any]:
        """Load LiteLLM configuration from YAML file.

        Returns:
            Configuration dictionary for LiteLLM

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            msg = f"LiteLLM configuration file not found: {self.config_path}"
            raise FileNotFoundError(msg)

        try:
            with self.config_path.open() as f:
                config = yaml.safe_load(f)

            # Validate required configuration sections
            required_sections = ["model_list", "litellm_settings"]
            for section in required_sections:
                if section not in config:
                    msg = f"Missing required configuration section: {section}"
                    raise ValueError(msg)

        except yaml.YAMLError as e:
            msg = f"Invalid YAML configuration: {e}"
            raise yaml.YAMLError(msg) from e
        else:
            return config

    def get_available_models(self) -> list[str]:
        """Get list of available model names from configuration.

        Returns:
            List of model names that can be used for completions
        """
        return [model["model_name"] for model in self.config["model_list"]]

    def get_fallback_chain(self, model_name: str) -> list[str]:
        """Get fallback chain for a specific model.

        Args:
            model_name: Name of the primary model

        Returns:
            List of model names in fallback order
        """
        fallbacks = self.config.get("litellm_settings", {}).get("fallbacks", {})
        return fallbacks.get(model_name, [model_name])

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens in message list using LiteLLM.

        Args:
            messages: Chat messages to count tokens for

        Returns:
            Number of tokens in the messages
        """
        try:
            return token_counter(messages=messages)
        except Exception as e:
            logger.warning("Token counting failed: %s", e)
            return 0

    def select_optimal_model(
        self,
        messages: list[dict[str, str]],
        *,
        complexity_score: float = 0.5,
        budget_limit: float | None = None,
    ) -> str:
        """Select optimal model based on context and requirements.

        Args:
            messages: Chat messages for context analysis
            complexity_score: Task complexity (0.0-1.0, higher = more complex)
            budget_limit: Maximum cost per request in USD

        Returns:
            Selected model name
        """
        token_count = self.count_tokens(messages)
        available_models = self.get_available_models()

        # Filter out local models for cloud service
        cloud_models = [m for m in available_models if not m.startswith("local-")]

        if not cloud_models:
            logger.warning("No cloud models available, using first available model")
            return available_models[0] if available_models else "gpt-4o-mini"

        # Simple model selection logic based on complexity and context size
        if complexity_score > 0.7 or token_count > 16000:
            # High complexity or large context - use most capable model
            preferred_models = ["gpt-4o", "claude-3-5-sonnet", "gpt-4"]
            for model in preferred_models:
                if model in cloud_models:
                    logger.info("Selected high-capability model: %s", model)
                    return model

        # Default to efficient model for standard tasks
        preferred_efficient = ["gpt-4o-mini", "claude-3-haiku", "gpt-3.5-turbo"]
        for model in preferred_efficient:
            if model in cloud_models:
                logger.info("Selected efficient model: %s", model)
                return model

        # Fallback to first available cloud model
        selected = cloud_models[0]
        logger.info("Using fallback cloud model: %s", selected)
        return selected

    async def generate_completion(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> str:
        """Generate text completion using cloud AI.

        Args:
            prompt: Input prompt for text generation
            model: Specific model to use (auto-select if None)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Additional parameters for the completion

        Returns:
            Generated text completion

        Raises:
            Exception: For completion errors
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Auto-select model if not specified
            if model is None:
                messages = [{"role": "user", "content": prompt}]
                model = self.select_optimal_model(messages)

            # Generate completion using LiteLLM router
            response = await acompletion(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                router=self.router,
                **kwargs,
            )

            # Track metrics
            self._success_count += 1
            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)

            # Track tokens and cost if available
            if hasattr(response, "usage") and response.usage:
                self._total_tokens += response.usage.total_tokens

                # Estimate cost (this would be more accurate with actual provider costs)
                estimated_cost = response.usage.total_tokens * 0.00002  # Rough estimate
                self._total_cost += estimated_cost

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("Cloud AI completion failed: %s", e)
            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)
            raise

    async def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        complexity_score: float = 0.5,
        **kwargs: Any,
    ) -> str:
        """Generate chat completion using cloud AI.

        Args:
            messages: Chat messages in OpenAI format
            model: Specific model to use (auto-select if None)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            complexity_score: Task complexity for model selection
            **kwargs: Additional parameters for the completion

        Returns:
            Generated chat response content

        Raises:
            Exception: For completion errors
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Auto-select model if not specified
            if model is None:
                model = self.select_optimal_model(
                    messages, complexity_score=complexity_score
                )

            # Generate completion using LiteLLM router
            response = await acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                router=self.router,
                **kwargs,
            )

            # Track metrics
            self._success_count += 1
            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)

            # Track tokens and cost if available
            if hasattr(response, "usage") and response.usage:
                self._total_tokens += response.usage.total_tokens
                estimated_cost = response.usage.total_tokens * 0.00002
                self._total_cost += estimated_cost

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("Cloud AI chat completion failed: %s", e)
            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)
            raise

    def generate_structured_output(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        *,
        model: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        complexity_score: float = 0.5,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output using Instructor with cloud AI.

        Args:
            messages: Chat messages in OpenAI format
            response_model: Pydantic model for structured output
            model: Specific model to use (auto-select if None)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            complexity_score: Task complexity for model selection
            **kwargs: Additional parameters for the completion

        Returns:
            Validated instance of response_model

        Raises:
            Exception: For completion or validation errors
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Auto-select model if not specified
            if model is None:
                model = self.select_optimal_model(
                    messages, complexity_score=complexity_score
                )

            # Generate structured output using Instructor
            result = self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Track metrics
            self._success_count += 1
            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)

            return result

        except Exception as e:
            logger.error("Cloud AI structured output failed: %s", e)
            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)
            raise

    async def is_healthy(self) -> bool:
        """Check if cloud AI services are healthy and responsive.

        Returns:
            True if services are healthy, False otherwise
        """
        current_time = time.time()

        # Use cached result if recent check
        if (current_time - self._last_health_check) < self._health_check_interval:
            return self._is_healthy

        try:
            # Test with a simple completion request using a fast model
            test_messages = [{"role": "user", "content": "Health check"}]
            model = self.select_optimal_model(test_messages, complexity_score=0.1)

            response = await acompletion(
                model=model,
                messages=test_messages,
                max_tokens=1,
                temperature=0.0,
                router=self.router,
            )

            self._is_healthy = True
            self._last_health_check = current_time
            logger.debug("Cloud AI health check passed")
            return True

        except Exception as e:
            logger.warning("Cloud AI health check failed: %s", e)
            self._is_healthy = False
            self._last_health_check = current_time
            return False

    async def get_health_status(self) -> CloudServiceHealth:
        """Get detailed health status of cloud AI services.

        Returns:
            CloudServiceHealth object with comprehensive service metrics
        """
        is_healthy = await self.is_healthy()
        current_time = time.time()

        # Calculate metrics
        success_rate = (
            (self._success_count / self._request_count * 100)
            if self._request_count > 0
            else 100.0
        )

        avg_response_time = (
            sum(self._response_times[-50:]) / len(self._response_times[-50:])
            if self._response_times
            else 0.0
        )

        cost_per_1k_tokens = (
            (self._total_cost / self._total_tokens * 1000)
            if self._total_tokens > 0
            else 0.02  # Default estimate
        )

        available_models = self.get_available_models()
        cloud_models = [m for m in available_models if not m.startswith("local-")]

        # Determine status
        if not is_healthy:
            status = "unhealthy"
        elif success_rate < 95:
            status = "degraded"
        else:
            status = "healthy"

        return CloudServiceHealth(
            status=status,
            available_models=cloud_models,
            primary_provider="openai",  # This could be dynamic based on config
            fallback_providers=["anthropic", "google"],
            response_time_ms=avg_response_time,
            success_rate=success_rate,
            cost_per_1k_tokens=cost_per_1k_tokens,
            last_check=self._last_health_check,
        )

    @st.cache_data(ttl=60, show_spinner=False)  # Cache usage stats for 1 minute
    def _calculate_usage_stats_cached(
        self, stats_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate usage statistics (cached version).

        Caches computed stats to avoid recalculating on every access.
        """
        avg_response_time = (
            sum(stats_snapshot["response_times"])
            / len(stats_snapshot["response_times"])
            if stats_snapshot["response_times"]
            else 0.0
        )

        return {
            "total_requests": stats_snapshot["request_count"],
            "successful_requests": stats_snapshot["success_count"],
            "success_rate": stats_snapshot["success_count"]
            / max(stats_snapshot["request_count"], 1)
            * 100,
            "total_tokens": stats_snapshot["total_tokens"],
            "total_cost_usd": stats_snapshot["total_cost"],
            "average_response_time_ms": avg_response_time,
            "cost_per_1k_tokens": stats_snapshot["total_cost"]
            / max(stats_snapshot["total_tokens"], 1)
            * 1000,
            "cache_info": {
                "cached": True,
                "cache_ttl_seconds": 60,
                "streamlit_caching_enabled": STREAMLIT_AVAILABLE,
            },
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for the cloud AI service.

        Returns:
            Dictionary containing usage metrics
        """
        # Create snapshot for caching
        stats_snapshot = {
            "request_count": self._request_count,
            "success_count": self._success_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "response_times": list(self._response_times),  # Copy for thread safety
        }

        return self._calculate_usage_stats_cached(stats_snapshot)

    def reset_metrics(self) -> None:
        """Reset usage metrics (useful for periodic reporting)."""
        self._request_count = 0
        self._success_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._response_times = []
        logger.info("Cloud AI service metrics reset")

    @staticmethod
    def clear_all_caches() -> None:
        """Clear all Streamlit caches used by the cloud AI service.

        Useful for forcing fresh health checks and stats recalculation.
        """
        if STREAMLIT_AVAILABLE:
            # Clear all caches
            st.cache_data.clear()
            st.cache_resource.clear()
            logger.info("✅ All CloudAIService caches cleared")
        else:
            logger.info("ℹ️ Streamlit not available - no caches to clear")

    @staticmethod
    def get_cache_stats() -> dict[str, Any]:
        """Get cache utilization statistics for the cloud AI service.

        Returns information about cache performance and memory usage.
        """
        return {
            "streamlit_available": STREAMLIT_AVAILABLE,
            "caching_enabled": STREAMLIT_AVAILABLE,
            "cached_functions": [
                "_create_litellm_router",
                "_create_instructor_client",
                "_calculate_usage_stats_cached",
            ],
            "cache_ttls": {
                "router": "permanent",  # Resource cache
                "instructor_client": "permanent",  # Resource cache
                "usage_stats": 60,  # 1 minute
            },
            "performance_benefits": {
                "reduced_router_creation": "Router resource caching",
                "reduced_client_creation": "Instructor client resource caching",
                "reduced_stats_computation": "1min stats caching",
            },
        }


# Module-level singleton for easy access with Streamlit resource caching
_cloud_ai_service: CloudAIService | None = None


@st.cache_resource(ttl=3600)  # Cache service instance for 1 hour
def _create_cloud_ai_service(
    config_hash: str, config_path: str | None = None, settings: Settings | None = None
) -> CloudAIService:
    """Create CloudAIService instance (cached as resource).

    Uses Streamlit resource caching to ensure single service instance
    across the application lifecycle.
    """
    return CloudAIService(config_path, settings)


def get_cloud_ai_service(
    config_path: str | None = None, settings: Settings | None = None
) -> CloudAIService:
    """Get singleton instance of CloudAIService.

    Args:
        config_path: Path to LiteLLM configuration file
        settings: Application settings, defaults to Settings() if None

    Returns:
        CloudAIService singleton instance
    """
    global _cloud_ai_service
    if _cloud_ai_service is None:
        if STREAMLIT_AVAILABLE:
            # Use cached version with config hash for cache invalidation
            import hashlib
            import json

            # Create hash based on config path and settings for cache invalidation
            config_data = {
                "config_path": config_path or "config/litellm.yaml",
                "settings": settings.model_dump() if settings else {},
            }
            config_str = json.dumps(config_data, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

            _cloud_ai_service = _create_cloud_ai_service(
                config_hash, config_path, settings
            )
        else:
            _cloud_ai_service = CloudAIService(config_path, settings)
    return _cloud_ai_service


def reset_cloud_ai_service() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _cloud_ai_service
    _cloud_ai_service = None
