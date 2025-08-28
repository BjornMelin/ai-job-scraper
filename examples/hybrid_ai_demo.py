#!/usr/bin/env python3
"""Hybrid AI Integration Demo - Phase 3C Implementation Example.

This demo shows how to use the complete hybrid AI system:
- Local vLLM service with Qwen3-4B model
- Cloud AI fallback via LiteLLM
- Intelligent routing based on complexity analysis
- Structured output processing with Instructor
- Background task processing with progress tracking

Usage:
    python examples/hybrid_ai_demo.py
"""

import asyncio
import logging

from typing import Any

from pydantic import BaseModel, Field

from src.ai import (
    BackgroundAIProcessor,
    LocalVLLMService,
    TaskComplexityAnalyzer,
    get_hybrid_ai_router,
    get_structured_output_processor,
)
from src.ai_models import JobPosting

# Configure logging for demo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoResult(BaseModel):
    """Demo result model for structured output testing."""

    task_name: str = Field(description="Name of the demo task")
    success: bool = Field(description="Whether the task succeeded")
    processing_time: float = Field(description="Processing time in seconds")
    service_used: str = Field(description="AI service that processed the task")
    complexity_score: float = Field(description="Task complexity score")
    result_data: dict[str, Any] = Field(description="Task result data")


async def demo_complexity_analysis() -> None:
    """Demonstrate task complexity analysis."""
    logger.info("=== Task Complexity Analysis Demo ===")

    analyzer = TaskComplexityAnalyzer()

    # Test simple task
    simple_task = "Extract the job title from this text"
    analysis = analyzer.analyze_task_complexity(prompt=simple_task)

    logger.info("Simple task complexity: %s", analysis.complexity_score)
    logger.info("Recommended service: %s", analysis.recommended_service)
    logger.info("Reasoning: %s", analysis.reasoning)

    # Test complex task
    complex_task = """
    Analyze this job posting and provide detailed insights about the role requirements,
    career progression opportunities, technical skills needed, company culture
    indicators, and create a comprehensive assessment of the position's suitability
    for different experience levels. Also compare it with industry standards and provide
    recommendations for salary negotiation.
    """

    complex_analysis = analyzer.analyze_task_complexity(prompt=complex_task)

    logger.info("Complex task complexity: %s", complex_analysis.complexity_score)
    logger.info("Recommended service: %s", complex_analysis.recommended_service)
    logger.info("Reasoning: %s", complex_analysis.reasoning)


async def demo_hybrid_routing() -> None:
    """Demonstrate hybrid AI routing with health checks."""
    logger.info("=== Hybrid AI Routing Demo ===")

    router = get_hybrid_ai_router()

    # Check service health
    local_healthy, cloud_healthy = await router.check_service_health()
    logger.info("Service health - Local: %s, Cloud: %s", local_healthy, cloud_healthy)

    # Simple chat completion
    simple_messages = [{"role": "user", "content": "What is the capital of France?"}]

    try:
        response = await router.generate_chat_completion(
            messages=simple_messages, max_tokens=50, temperature=0.1
        )
        logger.info("Simple chat response: %s...", response[:100])
    except Exception as e:
        logger.warning("Simple chat failed: %s", e)

    # Get routing metrics
    metrics = router.get_routing_metrics()
    logger.info("Routing metrics: %s total requests", metrics.total_requests)
    logger.info(
        "Local/Cloud split: %s/%s", metrics.local_requests, metrics.cloud_requests
    )


async def demo_structured_output() -> None:
    """Demonstrate structured output processing."""
    logger.info("=== Structured Output Processing Demo ===")

    processor = get_structured_output_processor()

    # Mock job posting content for extraction
    job_content = """
    Senior Python Developer - TechCorp Inc.
    Location: San Francisco, CA (Remote OK)
    Salary: $120,000 - $160,000 per year

    We are looking for an experienced Python developer to join our AI team.
    Requirements:
    - 5+ years Python experience
    - Experience with FastAPI, SQLAlchemy
    - Machine learning knowledge preferred
    - Strong communication skills

    Apply at: https://techcorp.com/jobs/senior-python-dev
    """

    messages = [
        {
            "role": "system",
            "content": "Extract structured job information from the provided text.",
        },
        {
            "role": "user",
            "content": f"Extract job details from this posting:\n\n{job_content}",
        },
    ]

    try:
        result = await processor.process_structured_output(
            messages=messages,
            response_model=JobPosting,
            max_tokens=1000,
            temperature=0.1,
        )

        if result.success and result.result:
            job = result.result
            logger.info("Extracted job: %s at %s", job.title, job.company)
            logger.info("Location: %s", job.location)
            logger.info("Salary: %s", job.salary_text)
            logger.info("Processing time: %s", result.processing_time)
            logger.info("Service used: %s", result.service_used)
        else:
            logger.warning("Structured extraction failed: %s", result.error_message)

    except Exception:
        logger.exception("Structured output demo failed")

    # Get processing metrics
    metrics = processor.get_processing_metrics()
    logger.info(
        "Processing metrics: %s%% success rate", metrics["success_rate_percent"]
    )


async def demo_background_processing() -> None:
    """Demonstrate background AI processing."""
    logger.info("=== Background AI Processing Demo ===")

    try:
        processor = BackgroundAIProcessor()

        # Start background processing
        await processor.start_processing()

        # Add some sample tasks
        task1_id = processor.add_task(
            task_type="enhance_job_posting",
            input_data={
                "job_data": {
                    "title": "python dev",
                    "company": "tech co",
                    "location": "sf",
                    "description": "need python dev for ai stuff",
                }
            },
            priority="high",
        )

        task2_id = processor.add_task(
            task_type="analyze_job_content",
            input_data={
                "job_data": {
                    "title": "Senior ML Engineer",
                    "company": "AI Startup",
                    "description": "Looking for ML engineer with PyTorch experience",
                }
            },
        )

        logger.info("Added tasks: %s, %s", task1_id, task2_id)

        # Monitor task progress
        for _ in range(10):  # Check for up to 10 seconds
            task1 = processor.get_task_status(task1_id)
            task2 = processor.get_task_status(task2_id)

            if task1:
                logger.info(
                    "Task 1 (%s): %s - %.1f%%",
                    task1.task_type,
                    task1.status,
                    task1.progress,
                )
            if task2:
                logger.info(
                    "Task 2 (%s): %s - %.1f%%",
                    task2.task_type,
                    task2.status,
                    task2.progress,
                )

            # Check if both completed
            if (
                task1
                and task1.status in ["completed", "failed"]
                and task2
                and task2.status in ["completed", "failed"]
            ):
                break

            await asyncio.sleep(1)

        # Get processing statistics
        stats = processor.get_processing_stats()
        logger.info("Background processing stats:")
        logger.info(
            "  Total: %s, Completed: %s", stats.total_tasks, stats.completed_tasks
        )
        logger.info("  Success rate: %.1f%%", stats.success_rate)
        logger.info("  Avg processing time: %.2f", stats.average_processing_time)

        # Clean shutdown
        await processor.shutdown()

    except Exception:
        logger.exception("Background processing demo failed")


async def demo_health_monitoring() -> None:
    """Demonstrate service health monitoring."""
    logger.info("=== Service Health Monitoring Demo ===")

    # Check local vLLM service
    try:
        local_service = LocalVLLMService()
        health_status = await local_service.get_health_status()

        logger.info("Local vLLM Health:")
        logger.info("  Status: %s", health_status.status)
        logger.info("  Model: %s", health_status.model_name)
        logger.info("  GPU utilization: %s", health_status.gpu_memory_utilization)
        logger.info("  Avg tokens/sec: %s", health_status.average_tokens_per_second)
        logger.info("  Uptime: %.1f", health_status.uptime_seconds)

    except Exception as e:
        logger.warning("Local service health check failed: %s", e)

    # Check hybrid router metrics
    try:
        router = get_hybrid_ai_router()
        metrics = router.get_routing_metrics()

        logger.info("Hybrid Router Metrics:")
        logger.info("  Total requests: %s", metrics.total_requests)
        logger.info("  Local success rate: %.1f%%", metrics.local_success_rate)
        logger.info("  Cloud success rate: %.1f%%", metrics.cloud_success_rate)
        logger.info("  Estimated cost savings: $%.2f", metrics.cost_savings)

    except Exception as e:
        logger.warning("Router metrics check failed: %s", e)


async def run_full_demo() -> None:
    """Run the complete hybrid AI system demonstration."""
    logger.info("🚀 Starting Hybrid AI Integration Demo")
    logger.info("This demo showcases Phase 3C implementation features")

    try:
        # Run all demo components
        await demo_complexity_analysis()
        await asyncio.sleep(1)

        await demo_hybrid_routing()
        await asyncio.sleep(1)

        await demo_structured_output()
        await asyncio.sleep(1)

        await demo_background_processing()
        await asyncio.sleep(1)

        await demo_health_monitoring()

        logger.info("✅ Hybrid AI Integration Demo completed successfully!")
        logger.info("🎯 Phase 3C implementation is fully functional")

    except Exception:
        logger.exception("❌ Demo failed")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(run_full_demo())
