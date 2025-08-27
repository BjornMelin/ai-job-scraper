"""Background AI processing system for async job enhancement.

This module provides background task processing for AI-enhanced job data,
allowing for asynchronous AI processing that doesn't block the main application
while providing progress tracking and comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time

from collections.abc import Callable
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.ai.hybrid_ai_router import HybridAIRouter, get_hybrid_ai_router
from src.ai.structured_output_processor import (
    StructuredOutputProcessor,
    get_structured_output_processor,
)
from src.ai_models import JobPosting
from src.config import Settings

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of background AI processing tasks."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority levels for background AI tasks."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AITask(BaseModel):
    """Background AI processing task."""

    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(description="Type of AI processing task")
    status: TaskStatus = Field(
        description="Current task status", default=TaskStatus.PENDING
    )
    priority: TaskPriority = Field(
        description="Task priority level", default=TaskPriority.NORMAL
    )
    created_at: float = Field(description="Task creation timestamp")
    started_at: float | None = Field(description="Task start timestamp", default=None)
    completed_at: float | None = Field(
        description="Task completion timestamp", default=None
    )
    progress: float = Field(description="Task progress percentage (0-100)", default=0.0)
    input_data: dict[str, Any] = Field(description="Input data for processing")
    result: dict[str, Any] | None = Field(description="Processing result", default=None)
    error_message: str | None = Field(
        description="Error message if failed", default=None
    )
    retry_count: int = Field(description="Number of retry attempts", default=0)
    max_retries: int = Field(description="Maximum retry attempts", default=2)


class ProcessingStats(BaseModel):
    """Statistics for background AI processing."""

    total_tasks: int = Field(description="Total tasks processed")
    completed_tasks: int = Field(description="Successfully completed tasks")
    failed_tasks: int = Field(description="Failed tasks")
    pending_tasks: int = Field(description="Tasks currently pending")
    processing_tasks: int = Field(description="Tasks currently processing")
    average_processing_time: float = Field(
        description="Average processing time in seconds"
    )
    success_rate: float = Field(description="Success rate percentage")
    total_cost: float = Field(description="Total AI processing cost (USD)")


class BackgroundAIProcessor:
    """Background AI processing system for job enhancement.

    Features:
    - Asynchronous AI processing with progress tracking
    - Priority-based task queuing
    - Retry logic for failed tasks
    - Cost tracking and optimization
    - Integration with hybrid AI routing
    - Real-time progress monitoring
    """

    def __init__(
        self,
        router: HybridAIRouter | None = None,
        processor: StructuredOutputProcessor | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the Background AI Processor.

        Args:
            router: Hybrid AI router instance, defaults to singleton if None
            processor: Structured output processor instance, defaults to singleton if None
            settings: Application settings, defaults to Settings() if None
        """
        self.settings = settings or Settings()
        self.router = router or get_hybrid_ai_router(settings)
        self.processor = processor or get_structured_output_processor(settings=settings)

        # Task management
        self._tasks: dict[str, AITask] = {}
        self._task_queues: dict[TaskPriority, list[str]] = {
            priority: [] for priority in TaskPriority
        }
        self._processing_tasks: set[str] = set()

        # Processing configuration
        self.max_concurrent_tasks = 3
        self.processing_enabled = True

        # Task handlers
        self._task_handlers: dict[str, Callable[[AITask], Any]] = {}
        self._register_default_handlers()

        # Statistics
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "processing_times": [],
            "total_cost": 0.0,
        }

        # Background processing
        self._processing_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        logger.info("Background AI Processor initialized")

    def _register_default_handlers(self) -> None:
        """Register default task handlers."""
        self._task_handlers["enhance_job_posting"] = self._handle_job_enhancement
        self._task_handlers["extract_job_data"] = self._handle_job_extraction
        self._task_handlers["analyze_job_content"] = self._handle_job_analysis

    async def start_processing(self) -> None:
        """Start background task processing."""
        if self._processing_task is not None and not self._processing_task.done():
            logger.warning("Background processing already running")
            return

        self.processing_enabled = True
        self._shutdown_event.clear()

        self._processing_task = asyncio.create_task(self._process_tasks_loop())
        logger.info("Background AI processing started")

    async def stop_processing(self) -> None:
        """Stop background task processing."""
        self.processing_enabled = False
        self._shutdown_event.set()

        if self._processing_task is not None:
            try:
                await asyncio.wait_for(self._processing_task, timeout=10.0)
            except TimeoutError:
                logger.warning(
                    "Background processing did not stop gracefully, cancelling"
                )
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass

        logger.info("Background AI processing stopped")

    async def _process_tasks_loop(self) -> None:
        """Main processing loop for background tasks."""
        while not self._shutdown_event.is_set() and self.processing_enabled:
            try:
                # Process tasks with priority order
                await self._process_pending_tasks()

                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error("Error in background processing loop: %s", e)
                await asyncio.sleep(1.0)  # Wait longer on error

    async def _process_pending_tasks(self) -> None:
        """Process pending tasks from priority queues."""
        if len(self._processing_tasks) >= self.max_concurrent_tasks:
            return  # Already at capacity

        # Process tasks in priority order
        for priority in [
            TaskPriority.URGENT,
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
            TaskPriority.LOW,
        ]:
            queue = self._task_queues[priority]

            while queue and len(self._processing_tasks) < self.max_concurrent_tasks:
                task_id = queue.pop(0)

                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    if task.status == TaskStatus.PENDING:
                        # Start processing this task
                        asyncio.create_task(self._process_task(task_id))

    async def _process_task(self, task_id: str) -> None:
        """Process a single AI task.

        Args:
            task_id: ID of the task to process
        """
        if task_id not in self._tasks:
            logger.error("Task %s not found", task_id)
            return

        task = self._tasks[task_id]
        self._processing_tasks.add(task_id)

        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            task.progress = 10.0

            logger.info("Processing AI task %s (type: %s)", task_id, task.task_type)

            # Get task handler
            handler = self._task_handlers.get(task.task_type)
            if handler is None:
                raise ValueError(f"No handler found for task type: {task.task_type}")

            # Process the task
            result = await handler(task)

            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.progress = 100.0

            # Update statistics
            self._stats["completed_tasks"] += 1
            if task.started_at:
                processing_time = task.completed_at - task.started_at
                self._stats["processing_times"].append(processing_time)

            logger.info("Task %s completed successfully", task_id)

        except Exception as e:
            logger.error("Task %s failed: %s", task_id, e)

            # Handle retry logic
            task.retry_count += 1
            task.error_message = str(e)

            if task.retry_count <= task.max_retries:
                # Retry the task
                task.status = TaskStatus.PENDING
                task.progress = 0.0
                # Re-add to appropriate queue
                self._task_queues[task.priority].append(task_id)
                logger.info(
                    "Retrying task %s (attempt %d/%d)",
                    task_id,
                    task.retry_count,
                    task.max_retries,
                )
            else:
                # Task permanently failed
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self._stats["failed_tasks"] += 1
                logger.error(
                    "Task %s permanently failed after %d attempts",
                    task_id,
                    task.retry_count,
                )

        finally:
            self._processing_tasks.discard(task_id)

    async def _handle_job_enhancement(self, task: AITask) -> dict[str, Any]:
        """Handle job posting enhancement task.

        Args:
            task: AI task to process

        Returns:
            Enhanced job data
        """
        input_data = task.input_data
        raw_job_data = input_data.get("job_data", {})

        task.progress = 20.0

        # Create enhancement prompt
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that enhances job posting data. 
                Your task is to improve and normalize job information while maintaining accuracy.""",
            },
            {
                "role": "user",
                "content": f"""
                Please enhance this job posting data by:
                1. Cleaning and normalizing the job title
                2. Improving the company description
                3. Standardizing the location format
                4. Extracting and formatting salary information
                5. Enhancing the job description for clarity
                
                Raw job data:
                Title: {raw_job_data.get("title", "N/A")}
                Company: {raw_job_data.get("company", "N/A")}
                Location: {raw_job_data.get("location", "N/A")}
                Description: {raw_job_data.get("description", "N/A")[:500]}...
                """,
            },
        ]

        task.progress = 40.0

        # Process with structured output
        result = await self.processor.process_with_schema_validation(
            messages=messages,
            response_model=JobPosting,
            max_tokens=2000,
            temperature=0.1,
        )

        task.progress = 80.0

        # Convert to dictionary
        enhanced_data = result.model_dump()

        # Add metadata
        enhanced_data["enhancement_metadata"] = {
            "enhanced_at": time.time(),
            "enhancement_version": "1.0",
            "original_data": raw_job_data,
        }

        return enhanced_data

    async def _handle_job_extraction(self, task: AITask) -> dict[str, Any]:
        """Handle job data extraction task.

        Args:
            task: AI task to process

        Returns:
            Extracted job data
        """
        input_data = task.input_data
        raw_content = input_data.get("content", "")

        task.progress = 20.0

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that extracts structured job information from web content.",
            },
            {
                "role": "user",
                "content": f"Extract job posting information from this content:\n\n{raw_content[:2000]}...",
            },
        ]

        task.progress = 50.0

        result = await self.processor.process_with_schema_validation(
            messages=messages,
            response_model=JobPosting,
            max_tokens=1500,
            temperature=0.1,
        )

        task.progress = 90.0

        return result.model_dump()

    async def _handle_job_analysis(self, task: AITask) -> dict[str, Any]:
        """Handle job analysis task.

        Args:
            task: AI task to process

        Returns:
            Job analysis results
        """
        input_data = task.input_data
        job_data = input_data.get("job_data", {})

        task.progress = 30.0

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that analyzes job postings for key insights.",
            },
            {
                "role": "user",
                "content": f"""
                Analyze this job posting and provide insights:
                - Required skills and qualifications
                - Experience level required
                - Remote work availability
                - Growth opportunities
                - Company culture indicators
                
                Job data: {job_data}
                """,
            },
        ]

        task.progress = 60.0

        result = await self.router.generate_chat_completion(
            messages=messages, max_tokens=1000, temperature=0.2
        )

        return {"analysis": result, "analyzed_at": time.time()}

    def add_task(
        self,
        task_type: str,
        input_data: dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 2,
    ) -> str:
        """Add a new AI processing task.

        Args:
            task_type: Type of processing task
            input_data: Input data for the task
            priority: Task priority level
            max_retries: Maximum retry attempts

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid4())

        task = AITask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            created_at=time.time(),
            input_data=input_data,
            max_retries=max_retries,
        )

        self._tasks[task_id] = task
        self._task_queues[priority].append(task_id)
        self._stats["total_tasks"] += 1

        logger.info(
            "Added AI task %s (type: %s, priority: %s)", task_id, task_type, priority
        )

        return task_id

    def get_task_status(self, task_id: str) -> AITask | None:
        """Get status of a specific task.

        Args:
            task_id: Task ID to check

        Returns:
            Task object if found, None otherwise
        """
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancelled, False if not found or cannot be cancelled
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]

        if task.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]:
            return False  # Cannot cancel completed tasks

        if task.status == TaskStatus.PROCESSING:
            # Task is currently processing, mark for cancellation
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
        else:
            # Remove from queue
            for queue in self._task_queues.values():
                if task_id in queue:
                    queue.remove(task_id)
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()

        logger.info("Task %s cancelled", task_id)
        return True

    def get_processing_stats(self) -> ProcessingStats:
        """Get comprehensive processing statistics.

        Returns:
            ProcessingStats with detailed metrics
        """
        # Count tasks by status
        pending_count = sum(
            1 for task in self._tasks.values() if task.status == TaskStatus.PENDING
        )
        processing_count = len(self._processing_tasks)

        # Calculate average processing time
        avg_time = (
            sum(self._stats["processing_times"]) / len(self._stats["processing_times"])
            if self._stats["processing_times"]
            else 0.0
        )

        # Calculate success rate
        total_completed = self._stats["completed_tasks"] + self._stats["failed_tasks"]
        success_rate = (self._stats["completed_tasks"] / max(total_completed, 1)) * 100

        return ProcessingStats(
            total_tasks=self._stats["total_tasks"],
            completed_tasks=self._stats["completed_tasks"],
            failed_tasks=self._stats["failed_tasks"],
            pending_tasks=pending_count,
            processing_tasks=processing_count,
            average_processing_time=avg_time,
            success_rate=success_rate,
            total_cost=self._stats["total_cost"],
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the background processor."""
        logger.info("Shutting down Background AI Processor")

        await self.stop_processing()

        # Wait for any remaining processing tasks to complete
        if self._processing_tasks:
            logger.info("Waiting for %d tasks to complete", len(self._processing_tasks))

            # Wait up to 30 seconds for tasks to complete
            timeout = 30.0
            start_time = time.time()

            while self._processing_tasks and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)

            if self._processing_tasks:
                logger.warning("Some tasks did not complete before shutdown")

        logger.info("Background AI Processor shutdown complete")


# Module-level singleton for easy access
_background_processor: BackgroundAIProcessor | None = None


def get_background_ai_processor(
    router: HybridAIRouter | None = None,
    processor: StructuredOutputProcessor | None = None,
    settings: Settings | None = None,
) -> BackgroundAIProcessor:
    """Get singleton instance of BackgroundAIProcessor.

    Args:
        router: Hybrid AI router instance, defaults to singleton if None
        processor: Structured output processor instance, defaults to singleton if None
        settings: Application settings, defaults to Settings() if None

    Returns:
        BackgroundAIProcessor singleton instance
    """
    global _background_processor
    if _background_processor is None:
        _background_processor = BackgroundAIProcessor(router, processor, settings)
    return _background_processor


def reset_background_ai_processor() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _background_processor
    _background_processor = None
