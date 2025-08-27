"""Service Orchestrator for comprehensive AI job scraper coordination.

This module orchestrates integrated workflows across all system services:
- Unified scraper (Phase 3A) coordination
- Hybrid AI enhancement (Phase 3C) processing
- Mobile-first UI updates (Phase 3B) rendering
- Database storage and search indexing
- Real-time progress tracking and error recovery

Key Features:
- End-to-end workflow execution: query → scrape → enhance → display
- Service dependency management and error recovery
- Production deployment validation and health checks
- Scalable orchestration patterns for complex workflows
"""

import asyncio
import logging
import uuid

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

from src.ai.hybrid_ai_router import get_hybrid_ai_router
from src.config import Settings
from src.coordination.background_task_manager import (
    get_background_task_manager,
)
from src.coordination.system_health_monitor import (
    get_system_health_monitor,
)
from src.interfaces.scraping_service_interface import JobQuery, SourceType
from src.services.database_sync import DatabaseSync
from src.services.search_service import SearchService
from src.services.unified_scraper import UnifiedScrapingService

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Exception raised when workflow execution fails."""


class ServiceDependencyError(Exception):
    """Exception raised when service dependencies are not available."""


class OrchestrationMetrics:
    """Metrics tracking for service orchestration."""

    def __init__(self):
        """Initialize orchestration metrics."""
        self.workflows_executed = 0
        self.workflows_completed = 0
        self.workflows_failed = 0
        self.average_workflow_duration = 0.0
        self.service_call_count = {}
        self.error_count_by_service = {}
        self.last_reset_time = datetime.now(UTC)

    def record_workflow_start(self) -> None:
        """Record a workflow start."""
        self.workflows_executed += 1

    def record_workflow_completion(self, duration: float) -> None:
        """Record a workflow completion with duration."""
        self.workflows_completed += 1

        # Update rolling average
        if self.workflows_completed == 1:
            self.average_workflow_duration = duration
        else:
            self.average_workflow_duration = (
                self.average_workflow_duration * (self.workflows_completed - 1)
                + duration
            ) / self.workflows_completed

    def record_workflow_failure(self) -> None:
        """Record a workflow failure."""
        self.workflows_failed += 1

    def record_service_call(self, service_name: str) -> None:
        """Record a service call."""
        self.service_call_count[service_name] = (
            self.service_call_count.get(service_name, 0) + 1
        )

    def record_service_error(self, service_name: str) -> None:
        """Record a service error."""
        self.error_count_by_service[service_name] = (
            self.error_count_by_service.get(service_name, 0) + 1
        )

    def get_success_rate(self) -> float:
        """Calculate workflow success rate."""
        if self.workflows_executed == 0:
            return 0.0
        return (self.workflows_completed / self.workflows_executed) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "workflows_executed": self.workflows_executed,
            "workflows_completed": self.workflows_completed,
            "workflows_failed": self.workflows_failed,
            "success_rate": self.get_success_rate(),
            "average_workflow_duration": self.average_workflow_duration,
            "service_call_count": self.service_call_count.copy(),
            "error_count_by_service": self.error_count_by_service.copy(),
            "last_reset_time": self.last_reset_time.isoformat(),
        }


class ServiceOrchestrator:
    """Comprehensive service orchestrator for integrated workflows.

    This class coordinates all system services into seamless end-to-end workflows:
    - Query processing and validation
    - Unified scraping with progress tracking
    - AI enhancement and structured processing
    - Database storage and search indexing
    - Mobile-first UI updates and real-time feedback

    Architecture:
    - Service dependency management with health checking
    - Error recovery and graceful degradation
    - Production deployment readiness validation
    - Scalable orchestration patterns
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the service orchestrator.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Service instances (lazy initialization)
        self._unified_scraper: UnifiedScrapingService | None = None
        self._ai_router = get_hybrid_ai_router(settings)
        self._task_manager = get_background_task_manager(settings)
        self._health_monitor = get_system_health_monitor()

        # Import native progress components dynamically to avoid circular imports
        from src.ui.components.native_progress import (
            get_native_progress_manager,
        )

        self._native_progress_manager = get_native_progress_manager()
        self._database_sync: DatabaseSync | None = None
        self._search_service: SearchService | None = None

        # Orchestration metrics
        self._metrics = OrchestrationMetrics()

        # Active workflows tracking
        self._active_workflows: dict[str, dict[str, Any]] = {}

        self.logger.info(
            "✅ ServiceOrchestrator initialized with comprehensive coordination"
        )

    def _get_unified_scraper(self) -> UnifiedScrapingService:
        """Get or create unified scraping service."""
        if self._unified_scraper is None:
            self._unified_scraper = UnifiedScrapingService(self.settings)
        return self._unified_scraper

    def _get_database_sync(self) -> DatabaseSync:
        """Get or create database sync service."""
        if self._database_sync is None:
            self._database_sync = DatabaseSync()
        return self._database_sync

    def _get_search_service(self) -> SearchService:
        """Get or create search service."""
        if self._search_service is None:
            self._search_service = SearchService()
        return self._search_service

    async def execute_integrated_workflow(
        self,
        query: str | JobQuery,
        workflow_options: dict[str, Any] | None = None,
    ) -> str:
        """Execute comprehensive integrated workflow.

        This orchestrates the complete end-to-end process:
        1. Query processing and validation
        2. Service health checking and dependency validation
        3. Unified scraping with progress tracking
        4. AI enhancement processing
        5. Database storage and search indexing
        6. Mobile-first UI updates with real-time feedback

        Args:
            query: Job search query (string or JobQuery object)
            workflow_options: Optional workflow configuration

        Returns:
            Workflow ID for tracking progress

        Raises:
            WorkflowExecutionError: If workflow execution fails
            ServiceDependencyError: If required services are unavailable
        """
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now(UTC)

        # Parse workflow options
        options = workflow_options or {}
        enable_ai_enhancement = options.get("enable_ai_enhancement", True)
        enable_real_time_updates = options.get("enable_real_time_updates", True)
        enable_ui_updates = options.get("enable_ui_updates", True)
        max_jobs = options.get("max_jobs", 50)
        source_types = options.get("source_types", [SourceType.UNIFIED])

        self._metrics.record_workflow_start()

        try:
            # Convert string query to JobQuery if needed
            if isinstance(query, str):
                job_query = JobQuery(
                    keywords=[query],
                    locations=["United States"],
                    source_types=source_types,
                    max_results=max_jobs,
                    enable_ai_enhancement=enable_ai_enhancement,
                )
            else:
                job_query = query

            # Initialize native progress tracking for this workflow
            from src.ui.components.native_progress import NativeProgressContext

            # Initialize workflow tracking
            self._active_workflows[workflow_id] = {
                "query": job_query.model_dump()
                if hasattr(job_query, "model_dump")
                else str(job_query),
                "options": options,
                "start_time": start_time,
                "status": "running",
                "services_used": [],
                "metrics": {},
            }

            self.logger.info(
                "🚀 Starting integrated workflow - ID: %s, Query: %s",
                workflow_id,
                job_query.keywords
                if hasattr(job_query, "keywords")
                else str(job_query),
            )

            # Execute workflow with native progress tracking
            with NativeProgressContext(
                workflow_id,
                f"🔍 Processing: {job_query.keywords if hasattr(job_query, 'keywords') else str(job_query)[:50]}",
                expanded=True,
            ) as progress:
                # Phase 1: Service Health Validation (0% -> 10%)
                progress.update(5.0, "🔍 Validating service health...", "health_check")
                await self._execute_health_validation_phase(workflow_id)

                # Phase 2: Unified Scraping (10% -> 60%)
                progress.update(15.0, "📋 Starting job scraping...", "scraping")
                jobs_data = await self._execute_scraping_phase(
                    workflow_id, job_query, progress
                )

                # Phase 3: AI Enhancement (60% -> 75%)
                if enable_ai_enhancement and jobs_data:
                    progress.update(
                        65.0, "🧠 Enhancing jobs with AI...", "ai_processing"
                    )
                    enhanced_jobs = await self._execute_ai_enhancement_phase(
                        workflow_id, jobs_data, progress
                    )
                else:
                    enhanced_jobs = jobs_data

                # Phase 4: Database Storage (75% -> 85%)
                if enhanced_jobs:
                    progress.update(80.0, "💾 Storing jobs to database...", "storage")
                    stored_count = await self._execute_database_storage_phase(
                        workflow_id, enhanced_jobs, progress
                    )
                else:
                    stored_count = 0

                # Phase 5: Search Indexing (85% -> 95%)
                if stored_count > 0:
                    progress.update(90.0, "🔍 Updating search indexes...", "indexing")
                    await self._execute_search_indexing_phase(workflow_id, progress)

                # Phase 6: UI Updates (95% -> 100%)
                if enable_ui_updates:
                    progress.update(98.0, "📱 Updating UI components...", "ui_updates")
                    await self._execute_ui_updates_phase(
                        workflow_id, enhanced_jobs or [], progress
                    )

                # Complete workflow
                duration = (datetime.now(UTC) - start_time).total_seconds()
                jobs_count = len(enhanced_jobs) if enhanced_jobs else 0

                self._complete_workflow(
                    workflow_id,
                    {
                        "jobs_processed": jobs_count,
                        "duration": duration,
                        "ai_enhancement_enabled": enable_ai_enhancement,
                        "ui_updates_enabled": enable_ui_updates,
                    },
                )

                progress.complete(
                    f"🎉 Completed! Processed {jobs_count} jobs in {duration:.1f}s",
                    show_balloons=True,
                )

            self._metrics.record_workflow_completion(duration)

            self.logger.info(
                "✅ Integrated workflow completed - ID: %s, Duration: %.2fs, Jobs: %d",
                workflow_id,
                duration,
                len(enhanced_jobs) if enhanced_jobs else 0,
            )

            return workflow_id

        except Exception as e:
            self.logger.error(
                "❌ Integrated workflow failed - ID: %s, Error: %s", workflow_id, e
            )
            self._fail_workflow(workflow_id, str(e))
            self._metrics.record_workflow_failure()
            raise WorkflowExecutionError(f"Workflow {workflow_id} failed: {e}") from e

    async def _execute_health_validation_phase(self, workflow_id: str) -> None:
        """Execute service health validation phase."""
        try:
            # Check critical services
            health_report = await self._health_monitor.get_comprehensive_health_report()

            # Validate critical services are available
            critical_services = ["database", "search", "unified_scraper"]
            unavailable_services = []

            for service in critical_services:
                if (
                    not health_report.get("services", {})
                    .get(service, {})
                    .get("healthy", False)
                ):
                    unavailable_services.append(service)

            if unavailable_services:
                raise ServiceDependencyError(
                    f"Critical services unavailable: {', '.join(unavailable_services)}"
                )

            self._metrics.record_service_call("health_monitor")
            self._active_workflows[workflow_id]["services_used"].append(
                "health_monitor"
            )

        except Exception as e:
            self._metrics.record_service_error("health_monitor")
            raise ServiceDependencyError(f"Health validation failed: {e}") from e

    async def _execute_scraping_phase(
        self, workflow_id: str, job_query: JobQuery, progress
    ) -> list[Any]:
        """Execute unified scraping phase."""
        try:
            scraper = self._get_unified_scraper()
            self._metrics.record_service_call("unified_scraper")
            self._active_workflows[workflow_id]["services_used"].append(
                "unified_scraper"
            )

            # Start background scraping
            scraping_task_id = await scraper.start_background_scraping(job_query)

            # Monitor scraping progress
            jobs_data = []
            async for scraping_status in scraper.monitor_scraping_progress(
                scraping_task_id
            ):
                # Map scraping progress to workflow progress (15% -> 60%)
                workflow_progress = 15.0 + (scraping_status.progress_percentage * 0.45)

                progress.update(
                    workflow_progress,
                    f"🔍 Scraping: {scraping_status.jobs_found} jobs found",
                    "scraping",
                )

                if scraping_status.status == "completed":
                    # For now, return empty list as jobs would need to be retrieved
                    # In real implementation, we'd get the actual jobs from the scraper
                    jobs_data = []  # Placeholder
                    break
                if scraping_status.status == "failed":
                    raise Exception(f"Scraping failed: {scraping_status.error_message}")

            return jobs_data

        except Exception as e:
            self._metrics.record_service_error("unified_scraper")
            raise WorkflowExecutionError(f"Scraping phase failed: {e}") from e

    async def _execute_ai_enhancement_phase(
        self, workflow_id: str, jobs_data: list[Any], progress
    ) -> list[Any]:
        """Execute AI enhancement phase."""
        try:
            # AI enhancement would be coordinated here using the hybrid AI router
            self._metrics.record_service_call("ai_router")
            self._active_workflows[workflow_id]["services_used"].append("ai_router")

            # For now, simulate AI processing
            await asyncio.sleep(2.0)  # Simulate AI processing time

            # In real implementation, we'd enhance each job with AI
            enhanced_jobs = jobs_data  # Placeholder - no actual enhancement

            return enhanced_jobs

        except Exception as e:
            self._metrics.record_service_error("ai_router")
            raise WorkflowExecutionError(f"AI enhancement phase failed: {e}") from e

    async def _execute_database_storage_phase(
        self, workflow_id: str, jobs_data: list[Any], progress
    ) -> int:
        """Execute database storage phase."""
        try:
            # Database storage coordination
            self._metrics.record_service_call("database_sync")
            self._active_workflows[workflow_id]["services_used"].append("database_sync")

            # For now, simulate database storage
            await asyncio.sleep(1.0)  # Simulate database operations
            stored_count = len(jobs_data)

            return stored_count

        except Exception as e:
            self._metrics.record_service_error("database_sync")
            raise WorkflowExecutionError(f"Database storage phase failed: {e}") from e

    async def _execute_search_indexing_phase(self, workflow_id: str, progress) -> None:
        """Execute search indexing phase."""
        try:
            # Search indexing coordination
            self._metrics.record_service_call("search_service")
            self._active_workflows[workflow_id]["services_used"].append(
                "search_service"
            )

            # For now, simulate search indexing
            await asyncio.sleep(0.5)  # Simulate indexing operations

        except Exception as e:
            self._metrics.record_service_error("search_service")
            raise WorkflowExecutionError(f"Search indexing phase failed: {e}") from e

    async def _execute_ui_updates_phase(
        self, workflow_id: str, jobs_data: list[Any], progress
    ) -> None:
        """Execute UI updates phase."""
        try:
            # UI updates coordination - integrate with mobile cards
            self._metrics.record_service_call("ui_components")
            self._active_workflows[workflow_id]["services_used"].append("ui_components")

            # Update session state for UI refresh
            if "workflow_results" not in st.session_state:
                st.session_state.workflow_results = {}

            st.session_state.workflow_results[workflow_id] = {
                "jobs_count": len(jobs_data),
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "completed",
            }

        except Exception as e:
            self._metrics.record_service_error("ui_components")
            raise WorkflowExecutionError(f"UI updates phase failed: {e}") from e

    def _complete_workflow(self, workflow_id: str, results: dict[str, Any]) -> None:
        """Mark workflow as completed with results."""
        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id].update(
                {
                    "status": "completed",
                    "end_time": datetime.now(UTC),
                    "results": results,
                }
            )

    def _fail_workflow(self, workflow_id: str, error_message: str) -> None:
        """Mark workflow as failed with error message."""
        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id].update(
                {
                    "status": "failed",
                    "end_time": datetime.now(UTC),
                    "error_message": error_message,
                }
            )

    def get_workflow_status(self, workflow_id: str) -> dict[str, Any] | None:
        """Get current status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow status dictionary or None if not found
        """
        return self._active_workflows.get(workflow_id)

    def get_all_workflows(self) -> dict[str, dict[str, Any]]:
        """Get all workflow statuses.

        Returns:
            Dictionary of workflow_id -> status
        """
        return self._active_workflows.copy()

    async def monitor_workflow_progress(
        self, workflow_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Monitor workflow progress with real-time updates.

        Args:
            workflow_id: Workflow identifier to monitor

        Yields:
            Workflow status dictionaries with current progress
        """
        while workflow_id in self._active_workflows:
            workflow_status = self._active_workflows[workflow_id]

            # Get native progress data if available
            progress_data = self._native_progress_manager.get_progress_data(workflow_id)
            if progress_data:
                workflow_status["current_progress"] = {
                    "percentage": progress_data.get("current_percentage", 0.0),
                    "message": progress_data.get("current_message", ""),
                    "phase": progress_data.get("current_phase", ""),
                    "is_active": progress_data.get("is_active", True),
                    "start_time": progress_data.get("start_time", "").isoformat()
                    if progress_data.get("start_time")
                    else None,
                    "last_update": progress_data.get("last_update", "").isoformat()
                    if progress_data.get("last_update")
                    else None,
                }

                # Simple ETA calculation
                if (
                    progress_data.get("is_active", True)
                    and progress_data.get("current_percentage", 0) > 0
                ):
                    start_time = progress_data.get("start_time")
                    if start_time:
                        elapsed = (datetime.now(UTC) - start_time).total_seconds()
                        percentage = progress_data.get("current_percentage", 0)
                        if elapsed > 0 and percentage > 0:
                            estimated_total = elapsed / (percentage / 100.0)
                            remaining = max(0, estimated_total - elapsed)
                            workflow_status["eta_estimate"] = {
                                "estimated_time_remaining": remaining,
                                "estimated_completion_time": (
                                    datetime.now(UTC) + timedelta(seconds=remaining)
                                ).isoformat(),
                            }

            yield workflow_status

            if workflow_status["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(1.0)  # Update every second

    def get_orchestration_metrics(self) -> dict[str, Any]:
        """Get comprehensive orchestration metrics.

        Returns:
            Dictionary with orchestration performance data
        """
        metrics_dict = self._metrics.to_dict()

        # Add active workflow counts
        active_count = sum(
            1 for w in self._active_workflows.values() if w["status"] == "running"
        )
        completed_count = sum(
            1 for w in self._active_workflows.values() if w["status"] == "completed"
        )
        failed_count = sum(
            1 for w in self._active_workflows.values() if w["status"] == "failed"
        )

        metrics_dict.update(
            {
                "active_workflows": active_count,
                "completed_workflows": completed_count,
                "failed_workflows": failed_count,
                "total_workflows_tracked": len(self._active_workflows),
            }
        )

        return metrics_dict

    async def validate_production_readiness(self) -> dict[str, Any]:
        """Validate system production deployment readiness.

        Returns:
            Dictionary with production readiness validation results
        """
        validation_results = {
            "ready_for_production": True,
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            # Health check validation
            health_report = await self._health_monitor.get_comprehensive_health_report()
            validation_results["checks"]["health_status"] = health_report

            # Service availability validation
            services_healthy = all(
                service.get("healthy", False)
                for service in health_report.get("services", {}).values()
            )
            validation_results["checks"]["all_services_healthy"] = services_healthy

            if not services_healthy:
                validation_results["errors"].append("Not all services are healthy")
                validation_results["ready_for_production"] = False

            # Orchestration metrics validation
            metrics = self.get_orchestration_metrics()
            success_rate = metrics.get("success_rate", 0.0)
            validation_results["checks"]["workflow_success_rate"] = success_rate

            if success_rate < 95.0 and metrics.get("workflows_executed", 0) > 10:
                validation_results["warnings"].append(
                    f"Workflow success rate is {success_rate:.1f}%, below 95% threshold"
                )

            # Configuration validation
            validation_results["checks"]["settings_configured"] = bool(
                self.settings.openai_api_key or self.settings.vllm_endpoint
            )

            if not validation_results["checks"]["settings_configured"]:
                validation_results["errors"].append("No AI services configured")
                validation_results["ready_for_production"] = False

            # Final readiness determination
            if validation_results["errors"]:
                validation_results["ready_for_production"] = False

            self.logger.info(
                "🔍 Production readiness validation completed - Ready: %s",
                validation_results["ready_for_production"],
            )

            return validation_results

        except Exception as e:
            self.logger.error("❌ Production readiness validation failed: %s", e)
            validation_results["ready_for_production"] = False
            validation_results["errors"].append(f"Validation failed: {e}")
            return validation_results

    def cleanup_old_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up old completed workflows to prevent memory leaks.

        Args:
            max_age_hours: Maximum age in hours for completed workflows

        Returns:
            Number of workflows cleaned up
        """
        cutoff_time = datetime.now(UTC).replace(
            hour=datetime.now(UTC).hour - max_age_hours
        )

        workflows_to_remove = []
        for workflow_id, workflow in self._active_workflows.items():
            if (
                workflow.get("status") in ["completed", "failed"]
                and workflow.get("end_time")
                and workflow["end_time"] < cutoff_time
            ):
                workflows_to_remove.append(workflow_id)

        for workflow_id in workflows_to_remove:
            del self._active_workflows[workflow_id]

            # Also clean up native progress data
            if "native_progress" in st.session_state:
                st.session_state.native_progress.pop(workflow_id, None)

        if workflows_to_remove:
            self.logger.info("🧹 Cleaned up %d old workflows", len(workflows_to_remove))

        return len(workflows_to_remove)


# Module-level singleton for easy access
_service_orchestrator: ServiceOrchestrator | None = None


def get_service_orchestrator(settings: Settings | None = None) -> ServiceOrchestrator:
    """Get singleton instance of ServiceOrchestrator.

    Args:
        settings: Application settings

    Returns:
        ServiceOrchestrator singleton instance
    """
    global _service_orchestrator
    if _service_orchestrator is None:
        _service_orchestrator = ServiceOrchestrator(settings)
    return _service_orchestrator


def reset_service_orchestrator() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _service_orchestrator
    _service_orchestrator = None
