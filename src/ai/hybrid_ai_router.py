"""Hybrid AI Router for intelligent local/cloud AI service routing.

This module provides the core routing intelligence that decides whether to use
local vLLM or cloud AI services based on task complexity, service availability,
cost preferences, and performance requirements.
"""

from __future__ import annotations

import asyncio
import logging
import time

from typing import Any

from pydantic import BaseModel, Field

from src.ai.cloud_ai_service import get_cloud_ai_service
from src.ai.local_vllm_service import get_local_vllm_service
from src.ai.task_complexity_analyzer import (
    ComplexityAnalysis,
    get_complexity_analyzer,
)
from src.config import Settings

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    """Result of AI service routing decision."""

    service_type: str = Field(description="Selected service: 'local' or 'cloud'")
    service_name: str = Field(description="Name of the selected service")
    complexity_analysis: ComplexityAnalysis = Field(
        description="Task complexity analysis"
    )
    routing_reason: str = Field(description="Explanation of routing decision")
    fallback_available: bool = Field(
        description="Whether fallback service is available"
    )
    estimated_cost: float = Field(description="Estimated cost in USD (0 for local)")
    estimated_response_time: float = Field(
        description="Estimated response time in seconds"
    )
    confidence: float = Field(
        description="Confidence in routing decision (0.0-1.0)", ge=0.0, le=1.0
    )


class RoutingMetrics(BaseModel):
    """Metrics for hybrid AI routing performance."""

    total_requests: int = Field(description="Total requests routed")
    local_requests: int = Field(description="Requests routed to local service")
    cloud_requests: int = Field(description="Requests routed to cloud service")
    local_success_rate: float = Field(description="Success rate for local requests (%)")
    cloud_success_rate: float = Field(description="Success rate for cloud requests (%)")
    average_local_response_time: float = Field(
        description="Average local response time (ms)"
    )
    average_cloud_response_time: float = Field(
        description="Average cloud response time (ms)"
    )
    total_cost: float = Field(description="Total cost incurred (USD)")
    cost_savings: float = Field(
        description="Estimated cost savings from local routing (USD)"
    )


class HybridAIRouter:
    """Intelligent router for hybrid local/cloud AI services.

    Features:
    - Task complexity analysis for optimal routing decisions
    - Automatic fallback between services based on availability
    - Cost-aware routing with configurable preferences
    - Performance monitoring and metrics collection
    - Health checking and service status management
    - Structured output support via Instructor
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the Hybrid AI Router.

        Args:
            settings: Application settings, defaults to Settings() if None
        """
        self.settings = settings or Settings()

        # Initialize AI services
        self.local_service = get_local_vllm_service(settings)
        self.cloud_service = get_cloud_ai_service(settings=settings)
        self.complexity_analyzer = get_complexity_analyzer()

        # Routing configuration
        self.cost_preference = "balanced"  # "cost_first", "balanced", "quality_first"
        self.fallback_enabled = True
        self.max_retry_attempts = 3
        self.health_check_interval = 60.0  # seconds

        # Performance tracking
        self._routing_metrics = {
            "total_requests": 0,
            "local_requests": 0,
            "cloud_requests": 0,
            "local_successes": 0,
            "cloud_successes": 0,
            "local_response_times": [],
            "cloud_response_times": [],
            "total_cost": 0.0,
            "fallback_used": 0,
        }

        # Service health status
        self._local_healthy = False
        self._cloud_healthy = False
        self._last_health_check = 0.0

        logger.info(
            "Hybrid AI Router initialized with cost preference: %s",
            self.cost_preference,
        )

    def set_cost_preference(self, preference: str) -> None:
        """Set cost preference for routing decisions.

        Args:
            preference: Cost preference - "cost_first", "balanced", or "quality_first"
        """
        valid_preferences = ["cost_first", "balanced", "quality_first"]
        if preference not in valid_preferences:
            raise ValueError(
                f"Invalid cost preference. Must be one of: {valid_preferences}"
            )

        self.cost_preference = preference
        logger.info("Cost preference updated to: %s", preference)

    async def check_service_health(
        self, force_check: bool = False
    ) -> tuple[bool, bool]:
        """Check health of local and cloud services.

        Args:
            force_check: Force health check even if recent check is cached

        Returns:
            Tuple of (local_healthy, cloud_healthy)
        """
        current_time = time.time()

        # Use cached results if recent check (unless forced)
        if (
            not force_check
            and (current_time - self._last_health_check) < self.health_check_interval
        ):
            return self._local_healthy, self._cloud_healthy

        # Check both services concurrently
        local_check, cloud_check = await asyncio.gather(
            self.local_service.is_healthy(),
            self.cloud_service.is_healthy(),
            return_exceptions=True,
        )

        # Handle exceptions in health checks
        self._local_healthy = local_check if isinstance(local_check, bool) else False
        self._cloud_healthy = cloud_check if isinstance(cloud_check, bool) else False
        self._last_health_check = current_time

        logger.debug(
            "Service health check - Local: %s, Cloud: %s",
            self._local_healthy,
            self._cloud_healthy,
        )

        return self._local_healthy, self._cloud_healthy

    async def analyze_and_route(
        self,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RoutingDecision:
        """Analyze task complexity and make routing decision.

        Args:
            messages: Chat messages (if using chat format)
            prompt: Direct prompt text (if using completion format)
            response_model: Expected response model for structured output
            **kwargs: Additional parameters affecting routing

        Returns:
            RoutingDecision with service selection and reasoning
        """
        # Analyze task complexity
        complexity_analysis = self.complexity_analyzer.analyze_task_complexity(
            messages=messages, prompt=prompt, response_model=response_model, **kwargs
        )

        # Check service availability
        local_healthy, cloud_healthy = await self.check_service_health()

        # Determine which service to use
        use_cloud, routing_reason = self.complexity_analyzer.should_use_cloud(
            complexity_analysis=complexity_analysis,
            local_service_available=local_healthy,
            cloud_service_available=cloud_healthy,
            cost_preference=self.cost_preference,
        )

        # Estimate costs and response times
        if use_cloud:
            estimated_cost = (
                complexity_analysis.token_count * 0.00002
            )  # Rough cloud cost estimate
            estimated_response_time = 2.0  # Cloud typically slower due to network
            service_type = "cloud"
            service_name = "Cloud AI (LiteLLM)"
            fallback_available = local_healthy
        else:
            estimated_cost = 0.0  # Local is free
            estimated_response_time = 0.8  # Local is typically faster
            service_type = "local"
            service_name = "Local vLLM"
            fallback_available = cloud_healthy

        # Calculate confidence based on service availability and complexity confidence
        base_confidence = complexity_analysis.confidence
        if not fallback_available:
            base_confidence -= 0.2  # Reduce confidence if no fallback available
        confidence = max(min(base_confidence, 1.0), 0.0)

        return RoutingDecision(
            service_type=service_type,
            service_name=service_name,
            complexity_analysis=complexity_analysis,
            routing_reason=routing_reason,
            fallback_available=fallback_available,
            estimated_cost=estimated_cost,
            estimated_response_time=estimated_response_time,
            confidence=confidence,
        )

    async def generate_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        auto_fallback: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate text completion using optimal AI service.

        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            auto_fallback: Whether to automatically fallback on failure
            **kwargs: Additional parameters for the completion

        Returns:
            Generated text completion

        Raises:
            Exception: If both services fail and no fallback available
        """
        start_time = time.time()
        self._routing_metrics["total_requests"] += 1

        # Analyze and route the task
        routing_decision = await self.analyze_and_route(prompt=prompt, **kwargs)

        logger.info(
            "Routing decision: %s (%s)",
            routing_decision.service_type,
            routing_decision.routing_reason,
        )

        # Attempt completion with selected service
        try:
            if routing_decision.service_type == "local":
                self._routing_metrics["local_requests"] += 1
                result = await self.local_service.generate_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                self._routing_metrics["local_successes"] += 1
                response_time = (time.time() - start_time) * 1000
                self._routing_metrics["local_response_times"].append(response_time)

            else:  # cloud
                self._routing_metrics["cloud_requests"] += 1
                result = await self.cloud_service.generate_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                self._routing_metrics["cloud_successes"] += 1
                self._routing_metrics["total_cost"] += routing_decision.estimated_cost
                response_time = (time.time() - start_time) * 1000
                self._routing_metrics["cloud_response_times"].append(response_time)

            return result

        except Exception as e:
            logger.warning(
                "Primary service (%s) failed: %s", routing_decision.service_type, e
            )

            # Attempt fallback if enabled and available
            if auto_fallback and routing_decision.fallback_available:
                try:
                    logger.info(
                        "Attempting fallback to %s service",
                        "cloud"
                        if routing_decision.service_type == "local"
                        else "local",
                    )

                    if routing_decision.service_type == "local":
                        # Fallback to cloud
                        self._routing_metrics["cloud_requests"] += 1
                        result = await self.cloud_service.generate_completion(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                        self._routing_metrics["cloud_successes"] += 1
                        self._routing_metrics["total_cost"] += (
                            routing_decision.estimated_cost
                        )
                    else:
                        # Fallback to local
                        self._routing_metrics["local_requests"] += 1
                        result = await self.local_service.generate_completion(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                        self._routing_metrics["local_successes"] += 1

                    self._routing_metrics["fallback_used"] += 1
                    logger.info("Fallback successful")
                    return result

                except Exception as fallback_error:
                    logger.error("Fallback also failed: %s", fallback_error)
                    raise

            raise

    async def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        auto_fallback: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate chat completion using optimal AI service.

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            auto_fallback: Whether to automatically fallback on failure
            **kwargs: Additional parameters for the completion

        Returns:
            Generated chat response content

        Raises:
            Exception: If both services fail and no fallback available
        """
        start_time = time.time()
        self._routing_metrics["total_requests"] += 1

        # Analyze and route the task
        routing_decision = await self.analyze_and_route(messages=messages, **kwargs)

        logger.info(
            "Routing decision: %s (%s)",
            routing_decision.service_type,
            routing_decision.routing_reason,
        )

        # Extract complexity score for cloud service
        complexity_score = routing_decision.complexity_analysis.complexity_score

        # Attempt completion with selected service
        try:
            if routing_decision.service_type == "local":
                self._routing_metrics["local_requests"] += 1
                result = await self.local_service.generate_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                self._routing_metrics["local_successes"] += 1
                response_time = (time.time() - start_time) * 1000
                self._routing_metrics["local_response_times"].append(response_time)

            else:  # cloud
                self._routing_metrics["cloud_requests"] += 1
                result = await self.cloud_service.generate_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    complexity_score=complexity_score,
                    **kwargs,
                )
                self._routing_metrics["cloud_successes"] += 1
                self._routing_metrics["total_cost"] += routing_decision.estimated_cost
                response_time = (time.time() - start_time) * 1000
                self._routing_metrics["cloud_response_times"].append(response_time)

            return result

        except Exception as e:
            logger.warning(
                "Primary service (%s) failed: %s", routing_decision.service_type, e
            )

            # Attempt fallback if enabled and available
            if auto_fallback and routing_decision.fallback_available:
                try:
                    logger.info(
                        "Attempting fallback to %s service",
                        "cloud"
                        if routing_decision.service_type == "local"
                        else "local",
                    )

                    if routing_decision.service_type == "local":
                        # Fallback to cloud
                        self._routing_metrics["cloud_requests"] += 1
                        result = await self.cloud_service.generate_chat_completion(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            complexity_score=complexity_score,
                            **kwargs,
                        )
                        self._routing_metrics["cloud_successes"] += 1
                        self._routing_metrics["total_cost"] += (
                            routing_decision.estimated_cost
                        )
                    else:
                        # Fallback to local
                        self._routing_metrics["local_requests"] += 1
                        result = await self.local_service.generate_chat_completion(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                        self._routing_metrics["local_successes"] += 1

                    self._routing_metrics["fallback_used"] += 1
                    logger.info("Fallback successful")
                    return result

                except Exception as fallback_error:
                    logger.error("Fallback also failed: %s", fallback_error)
                    raise

            raise

    async def generate_structured_output(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        auto_fallback: bool = True,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output using optimal AI service.

        Args:
            messages: Chat messages in OpenAI format
            response_model: Pydantic model for structured output
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            auto_fallback: Whether to automatically fallback on failure
            **kwargs: Additional parameters for the completion

        Returns:
            Validated instance of response_model

        Raises:
            Exception: If both services fail and no fallback available
        """
        start_time = time.time()
        self._routing_metrics["total_requests"] += 1

        # Analyze and route the task
        routing_decision = await self.analyze_and_route(
            messages=messages, response_model=response_model, **kwargs
        )

        logger.info(
            "Routing decision for structured output: %s (%s)",
            routing_decision.service_type,
            routing_decision.routing_reason,
        )

        # Extract complexity score for cloud service
        complexity_score = routing_decision.complexity_analysis.complexity_score

        # Attempt completion with selected service
        try:
            if routing_decision.service_type == "local":
                self._routing_metrics["local_requests"] += 1
                result = self.local_service.generate_structured_output(
                    messages=messages,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                self._routing_metrics["local_successes"] += 1
                response_time = (time.time() - start_time) * 1000
                self._routing_metrics["local_response_times"].append(response_time)

            else:  # cloud
                self._routing_metrics["cloud_requests"] += 1
                result = self.cloud_service.generate_structured_output(
                    messages=messages,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    complexity_score=complexity_score,
                    **kwargs,
                )
                self._routing_metrics["cloud_successes"] += 1
                self._routing_metrics["total_cost"] += routing_decision.estimated_cost
                response_time = (time.time() - start_time) * 1000
                self._routing_metrics["cloud_response_times"].append(response_time)

            return result

        except Exception as e:
            logger.warning(
                "Primary service (%s) failed: %s", routing_decision.service_type, e
            )

            # Attempt fallback if enabled and available
            if auto_fallback and routing_decision.fallback_available:
                try:
                    logger.info(
                        "Attempting fallback to %s service",
                        "cloud"
                        if routing_decision.service_type == "local"
                        else "local",
                    )

                    if routing_decision.service_type == "local":
                        # Fallback to cloud
                        self._routing_metrics["cloud_requests"] += 1
                        result = self.cloud_service.generate_structured_output(
                            messages=messages,
                            response_model=response_model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            complexity_score=complexity_score,
                            **kwargs,
                        )
                        self._routing_metrics["cloud_successes"] += 1
                        self._routing_metrics["total_cost"] += (
                            routing_decision.estimated_cost
                        )
                    else:
                        # Fallback to local
                        self._routing_metrics["local_requests"] += 1
                        result = self.local_service.generate_structured_output(
                            messages=messages,
                            response_model=response_model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                        self._routing_metrics["local_successes"] += 1

                    self._routing_metrics["fallback_used"] += 1
                    logger.info("Fallback successful")
                    return result

                except Exception as fallback_error:
                    logger.error("Fallback also failed: %s", fallback_error)
                    raise

            raise

    def get_routing_metrics(self) -> RoutingMetrics:
        """Get comprehensive routing metrics and performance statistics.

        Returns:
            RoutingMetrics with detailed performance data
        """
        total_requests = self._routing_metrics["total_requests"]
        local_requests = self._routing_metrics["local_requests"]
        cloud_requests = self._routing_metrics["cloud_requests"]

        # Calculate success rates
        local_success_rate = (
            self._routing_metrics["local_successes"] / max(local_requests, 1)
        ) * 100
        cloud_success_rate = (
            self._routing_metrics["cloud_successes"] / max(cloud_requests, 1)
        ) * 100

        # Calculate average response times
        local_times = self._routing_metrics["local_response_times"]
        cloud_times = self._routing_metrics["cloud_response_times"]

        avg_local_response_time = sum(local_times) / max(len(local_times), 1)
        avg_cloud_response_time = sum(cloud_times) / max(len(cloud_times), 1)

        # Estimate cost savings from local routing
        cost_savings = local_requests * 0.00002 * 1000  # Rough estimate of cost avoided

        return RoutingMetrics(
            total_requests=total_requests,
            local_requests=local_requests,
            cloud_requests=cloud_requests,
            local_success_rate=local_success_rate,
            cloud_success_rate=cloud_success_rate,
            average_local_response_time=avg_local_response_time,
            average_cloud_response_time=avg_cloud_response_time,
            total_cost=self._routing_metrics["total_cost"],
            cost_savings=cost_savings,
        )

    def reset_metrics(self) -> None:
        """Reset routing metrics (useful for periodic reporting)."""
        self._routing_metrics = {
            "total_requests": 0,
            "local_requests": 0,
            "cloud_requests": 0,
            "local_successes": 0,
            "cloud_successes": 0,
            "local_response_times": [],
            "cloud_response_times": [],
            "total_cost": 0.0,
            "fallback_used": 0,
        }
        logger.info("Hybrid AI Router metrics reset")

    async def shutdown(self) -> None:
        """Gracefully shutdown the hybrid AI router and its services."""
        logger.info("Shutting down Hybrid AI Router")

        # Shutdown services
        await asyncio.gather(
            self.local_service.shutdown(),
            # Cloud service doesn't need explicit shutdown
            return_exceptions=True,
        )

        logger.info("Hybrid AI Router shutdown complete")


# Module-level singleton for easy access
_hybrid_ai_router: HybridAIRouter | None = None


def get_hybrid_ai_router(settings: Settings | None = None) -> HybridAIRouter:
    """Get singleton instance of HybridAIRouter.

    Args:
        settings: Application settings, defaults to Settings() if None

    Returns:
        HybridAIRouter singleton instance
    """
    global _hybrid_ai_router
    if _hybrid_ai_router is None:
        _hybrid_ai_router = HybridAIRouter(settings)
    return _hybrid_ai_router


def reset_hybrid_ai_router() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _hybrid_ai_router
    _hybrid_ai_router = None
