"""System Health Monitor for comprehensive service monitoring and availability detection.

This module monitors all services in the AI job scraper system:
- Scraper service health and performance
- AI services (local vLLM + cloud) availability
- UI components and responsive card rendering
- Database connectivity and performance
- Production deployment validation

Key Features:
- Real-time health checks across all services
- Performance metrics collection and analysis
- Service availability detection with smart fallback
- Production readiness validation
- Proactive issue detection and alerting
"""

import asyncio
import logging
import time

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from src.ai.hybrid_ai_router import get_hybrid_ai_router
from src.config import Settings
from src.database import SessionLocal
from src.services.search_service import SearchService
from src.services.unified_scraper import UnifiedScrapingService

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealthStatus:
    """Health status for an individual service."""

    service_name: str
    healthy: bool
    response_time_ms: float
    status_message: str
    last_check_time: datetime
    error_message: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "service_name": self.service_name,
            "healthy": self.healthy,
            "response_time_ms": self.response_time_ms,
            "status_message": self.status_message,
            "last_check_time": self.last_check_time.isoformat(),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""

    overall_healthy: bool
    report_timestamp: datetime
    services: dict[str, ServiceHealthStatus]
    system_metrics: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_healthy": self.overall_healthy,
            "report_timestamp": self.report_timestamp.isoformat(),
            "services": {
                name: status.to_dict() for name, status in self.services.items()
            },
            "system_metrics": self.system_metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class SystemHealthMonitor:
    """Comprehensive system health monitor for all services.

    This class monitors the health and availability of all system components:
    - Database connectivity and performance
    - Unified scraper service availability
    - AI services (local vLLM + cloud) health
    - Search service functionality
    - UI component rendering capability
    - File system and configuration validation

    Architecture:
    - Periodic health checks with configurable intervals
    - Service-specific health validation logic
    - Performance metrics collection and analysis
    - Proactive issue detection and alerting
    - Production readiness assessment
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the system health monitor.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Health check configuration
        self.health_check_interval = 30.0  # seconds
        self.health_check_timeout = 10.0  # seconds
        self.max_response_time_warning = 2000.0  # ms

        # Health status tracking
        self._last_health_report: SystemHealthReport | None = None
        self._last_health_check = 0.0
        self._health_history: list[SystemHealthReport] = []
        self._max_history_size = 50

        # Service instances (lazy initialization)
        self._unified_scraper: UnifiedScrapingService | None = None
        self._ai_router = get_hybrid_ai_router(settings)
        self._search_service: SearchService | None = None
        self._http_client: httpx.AsyncClient | None = None

        # Performance metrics
        self._performance_metrics = {
            "total_health_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "average_check_duration": 0.0,
            "services_monitored": 0,
        }

        self.logger.info(
            "✅ SystemHealthMonitor initialized with comprehensive monitoring"
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for health checks."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.health_check_timeout),
                follow_redirects=True,
            )
        return self._http_client

    def _get_unified_scraper(self) -> UnifiedScrapingService:
        """Get or create unified scraper for health checks."""
        if self._unified_scraper is None:
            self._unified_scraper = UnifiedScrapingService(self.settings)
        return self._unified_scraper

    def _get_search_service(self) -> SearchService:
        """Get or create search service for health checks."""
        if self._search_service is None:
            self._search_service = SearchService()
        return self._search_service

    async def check_database_health(self) -> ServiceHealthStatus:
        """Check database connectivity and performance."""
        start_time = time.time()

        try:
            # Test database connection and basic operations
            session = SessionLocal()
            try:
                # Simple query to test connectivity
                result = session.execute("SELECT 1").scalar()
                if result != 1:
                    raise Exception("Database query returned unexpected result")

                # Test database file access (for SQLite)
                db_path = Path("jobs.db")
                if db_path.exists():
                    db_size = db_path.stat().st_size
                    metadata = {
                        "database_size_mb": round(db_size / (1024 * 1024), 2),
                        "database_path": str(db_path.absolute()),
                    }
                else:
                    metadata = {"database_path": "not_found"}

                response_time = (time.time() - start_time) * 1000

                return ServiceHealthStatus(
                    service_name="database",
                    healthy=True,
                    response_time_ms=response_time,
                    status_message="Database connection successful",
                    last_check_time=datetime.now(UTC),
                    metadata=metadata,
                )

            finally:
                session.close()

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="database",
                healthy=False,
                response_time_ms=response_time,
                status_message="Database connection failed",
                last_check_time=datetime.now(UTC),
                error_message=str(e),
            )

    async def check_unified_scraper_health(self) -> ServiceHealthStatus:
        """Check unified scraper service health."""
        start_time = time.time()

        try:
            scraper = self._get_unified_scraper()

            # Get scraper metrics and success rates
            metrics = await scraper.get_success_rate_metrics()

            # Check if scraper has reasonable success rates
            overall_success_rate = metrics.get("overall", {}).get("success_rate", 0.0)

            response_time = (time.time() - start_time) * 1000

            # Consider healthy if no recent attempts or good success rate
            total_attempts = metrics.get("overall", {}).get("attempts", 0)
            is_healthy = (total_attempts == 0) or (overall_success_rate >= 50.0)

            status_message = (
                f"Scraper available - Success rate: {overall_success_rate:.1f}%"
                if is_healthy
                else f"Scraper degraded - Success rate: {overall_success_rate:.1f}%"
            )

            return ServiceHealthStatus(
                service_name="unified_scraper",
                healthy=is_healthy,
                response_time_ms=response_time,
                status_message=status_message,
                last_check_time=datetime.now(UTC),
                metadata={
                    "success_rate": overall_success_rate,
                    "total_attempts": total_attempts,
                    "metrics": metrics,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="unified_scraper",
                healthy=False,
                response_time_ms=response_time,
                status_message="Scraper service unavailable",
                last_check_time=datetime.now(UTC),
                error_message=str(e),
            )

    async def check_ai_services_health(self) -> ServiceHealthStatus:
        """Check AI services (local vLLM + cloud) health."""
        start_time = time.time()

        try:
            # Check both local and cloud AI service availability
            local_healthy, cloud_healthy = await self._ai_router.check_service_health(
                force_check=True
            )

            # Get AI router metrics
            routing_metrics = self._ai_router.get_routing_metrics()

            response_time = (time.time() - start_time) * 1000

            # Consider healthy if at least one service is available
            is_healthy = local_healthy or cloud_healthy

            if is_healthy:
                services_available = []
                if local_healthy:
                    services_available.append("local")
                if cloud_healthy:
                    services_available.append("cloud")
                status_message = (
                    f"AI services available: {', '.join(services_available)}"
                )
            else:
                status_message = "No AI services available"

            return ServiceHealthStatus(
                service_name="ai_services",
                healthy=is_healthy,
                response_time_ms=response_time,
                status_message=status_message,
                last_check_time=datetime.now(UTC),
                metadata={
                    "local_healthy": local_healthy,
                    "cloud_healthy": cloud_healthy,
                    "routing_metrics": routing_metrics.model_dump()
                    if hasattr(routing_metrics, "model_dump")
                    else routing_metrics.__dict__,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="ai_services",
                healthy=False,
                response_time_ms=response_time,
                status_message="AI services health check failed",
                last_check_time=datetime.now(UTC),
                error_message=str(e),
            )

    async def check_search_service_health(self) -> ServiceHealthStatus:
        """Check search service functionality."""
        start_time = time.time()

        try:
            search_service = self._get_search_service()

            # Test search functionality with a simple query
            test_results = search_service.search_jobs("test", limit=1)

            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="search",
                healthy=True,
                response_time_ms=response_time,
                status_message=f"Search service available - {len(test_results)} test results",
                last_check_time=datetime.now(UTC),
                metadata={
                    "test_results_count": len(test_results),
                    "search_available": True,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="search",
                healthy=False,
                response_time_ms=response_time,
                status_message="Search service unavailable",
                last_check_time=datetime.now(UTC),
                error_message=str(e),
            )

    async def check_ui_components_health(self) -> ServiceHealthStatus:
        """Check UI components and responsive card rendering capability."""
        start_time = time.time()

        try:
            # Check if key UI files and dependencies are available
            ui_components_available = True
            ui_files_to_check = [
                "src/ui/components/cards/job_card.py",
                "src/ui/styles/styles.py",
                "src/ui/utils/mobile_detection.py",
            ]

            missing_files = []
            for file_path in ui_files_to_check:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
                    ui_components_available = False

            response_time = (time.time() - start_time) * 1000

            if ui_components_available:
                status_message = "UI components available and ready"
            else:
                status_message = f"UI components missing: {', '.join(missing_files)}"

            return ServiceHealthStatus(
                service_name="ui_components",
                healthy=ui_components_available,
                response_time_ms=response_time,
                status_message=status_message,
                last_check_time=datetime.now(UTC),
                metadata={
                    "files_checked": len(ui_files_to_check),
                    "missing_files": missing_files,
                    "mobile_cards_available": ui_components_available,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="ui_components",
                healthy=False,
                response_time_ms=response_time,
                status_message="UI components health check failed",
                last_check_time=datetime.now(UTC),
                error_message=str(e),
            )

    async def check_file_system_health(self) -> ServiceHealthStatus:
        """Check file system access and configuration files."""
        start_time = time.time()

        try:
            # Check critical directories and files
            critical_paths = [
                Path("src"),
                Path("src/coordination"),
                Path("src/services"),
                Path("src/ui"),
                Path("src/ai"),
            ]

            missing_paths = []
            for path in critical_paths:
                if not path.exists():
                    missing_paths.append(str(path))

            # Check configuration files
            config_files = [
                Path("pyproject.toml"),
                Path("src/config.py"),
            ]

            missing_config = []
            for config_file in config_files:
                if not config_file.exists():
                    missing_config.append(str(config_file))

            response_time = (time.time() - start_time) * 1000

            is_healthy = len(missing_paths) == 0 and len(missing_config) == 0

            if is_healthy:
                status_message = "File system access healthy"
            else:
                status_message = f"Missing paths: {missing_paths + missing_config}"

            return ServiceHealthStatus(
                service_name="file_system",
                healthy=is_healthy,
                response_time_ms=response_time,
                status_message=status_message,
                last_check_time=datetime.now(UTC),
                metadata={
                    "critical_paths_checked": len(critical_paths),
                    "config_files_checked": len(config_files),
                    "missing_paths": missing_paths,
                    "missing_config": missing_config,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return ServiceHealthStatus(
                service_name="file_system",
                healthy=False,
                response_time_ms=response_time,
                status_message="File system health check failed",
                last_check_time=datetime.now(UTC),
                error_message=str(e),
            )

    async def get_comprehensive_health_report(
        self, force_check: bool = False
    ) -> dict[str, Any]:
        """Get comprehensive health report for all services.

        Args:
            force_check: Force new health check even if recent data available

        Returns:
            Dictionary with comprehensive health report
        """
        current_time = time.time()

        # Use cached report if recent (within health_check_interval)
        if (
            not force_check
            and self._last_health_report
            and current_time - self._last_health_check < self.health_check_interval
        ):
            return self._last_health_report.to_dict()

        start_time = time.time()
        self._performance_metrics["total_health_checks"] += 1

        try:
            # Perform all health checks concurrently
            health_checks = await asyncio.gather(
                self.check_database_health(),
                self.check_unified_scraper_health(),
                self.check_ai_services_health(),
                self.check_search_service_health(),
                self.check_ui_components_health(),
                self.check_file_system_health(),
                return_exceptions=True,
            )

            services = {}
            warnings = []
            errors = []

            # Process health check results
            service_names = [
                "database",
                "unified_scraper",
                "ai_services",
                "search",
                "ui_components",
                "file_system",
            ]

            for i, result in enumerate(health_checks):
                service_name = service_names[i]

                if isinstance(result, Exception):
                    # Handle health check exceptions
                    services[service_name] = ServiceHealthStatus(
                        service_name=service_name,
                        healthy=False,
                        response_time_ms=0.0,
                        status_message="Health check exception",
                        last_check_time=datetime.now(UTC),
                        error_message=str(result),
                    )
                    errors.append(f"{service_name}: {result}")
                else:
                    services[service_name] = result

                    # Check for warnings
                    if (
                        result.healthy
                        and result.response_time_ms > self.max_response_time_warning
                    ):
                        warnings.append(
                            f"{service_name} response time high: {result.response_time_ms:.1f}ms"
                        )

                    if not result.healthy:
                        errors.append(f"{service_name}: {result.status_message}")

            # Determine overall system health
            overall_healthy = all(service.healthy for service in services.values())

            # Calculate system metrics
            total_response_time = sum(
                service.response_time_ms for service in services.values()
            )
            healthy_services = sum(
                1 for service in services.values() if service.healthy
            )

            system_metrics = {
                "total_services": len(services),
                "healthy_services": healthy_services,
                "unhealthy_services": len(services) - healthy_services,
                "total_response_time_ms": total_response_time,
                "average_response_time_ms": total_response_time / len(services)
                if services
                else 0.0,
                "health_check_duration_ms": (time.time() - start_time) * 1000,
            }

            # Create comprehensive health report
            health_report = SystemHealthReport(
                overall_healthy=overall_healthy,
                report_timestamp=datetime.now(UTC),
                services=services,
                system_metrics=system_metrics,
                warnings=warnings,
                errors=errors,
            )

            # Update cache and history
            self._last_health_report = health_report
            self._last_health_check = current_time

            # Add to history (maintain max size)
            self._health_history.append(health_report)
            if len(self._health_history) > self._max_history_size:
                self._health_history.pop(0)

            # Update performance metrics
            self._performance_metrics["successful_checks"] += 1
            check_duration = time.time() - start_time
            self._update_average_check_duration(check_duration)
            self._performance_metrics["services_monitored"] = len(services)

            self.logger.info(
                "🔍 Health check completed - Overall: %s, Services: %d/%d healthy, Duration: %.2fs",
                "✅" if overall_healthy else "❌",
                healthy_services,
                len(services),
                check_duration,
            )

            return health_report.to_dict()

        except Exception as e:
            self.logger.error("❌ Comprehensive health check failed: %s", e)
            self._performance_metrics["failed_checks"] += 1

            # Return error report
            error_report = {
                "overall_healthy": False,
                "report_timestamp": datetime.now(UTC).isoformat(),
                "services": {},
                "system_metrics": {"health_check_failed": True},
                "warnings": [],
                "errors": [f"Health check failed: {e}"],
            }

            return error_report

    def _update_average_check_duration(self, new_duration: float) -> None:
        """Update rolling average check duration."""
        successful = self._performance_metrics["successful_checks"]
        if successful > 0:
            current_avg = self._performance_metrics["average_check_duration"]
            self._performance_metrics["average_check_duration"] = (
                current_avg * (successful - 1) + new_duration
            ) / successful

    async def monitor_system_health(
        self, check_interval: float = 30.0, max_checks: int | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Monitor system health with periodic checks.

        Args:
            check_interval: Time between health checks in seconds
            max_checks: Maximum number of checks (None for unlimited)

        Yields:
            Health report dictionaries
        """
        check_count = 0

        while True:
            # Perform health check
            health_report = await self.get_comprehensive_health_report(force_check=True)
            yield health_report

            check_count += 1
            if max_checks and check_count >= max_checks:
                break

            # Wait for next check
            await asyncio.sleep(check_interval)

    def get_health_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get historical health reports.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of health report dictionaries
        """
        history = self._health_history.copy()
        if limit:
            history = history[-limit:]

        return [report.to_dict() for report in history]

    def get_service_availability_trend(
        self, service_name: str, hours: int = 24
    ) -> dict[str, Any]:
        """Get availability trend for a specific service.

        Args:
            service_name: Name of service to analyze
            hours: Number of hours to analyze

        Returns:
            Dictionary with availability trend data
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        relevant_reports = [
            report
            for report in self._health_history
            if report.report_timestamp >= cutoff_time
        ]

        if not relevant_reports:
            return {
                "service_name": service_name,
                "availability_percentage": 0.0,
                "total_checks": 0,
                "uptime_checks": 0,
                "average_response_time_ms": 0.0,
            }

        service_checks = []
        for report in relevant_reports:
            if service_name in report.services:
                service_checks.append(report.services[service_name])

        if not service_checks:
            return {
                "service_name": service_name,
                "availability_percentage": 0.0,
                "total_checks": 0,
                "uptime_checks": 0,
                "average_response_time_ms": 0.0,
            }

        uptime_checks = sum(1 for check in service_checks if check.healthy)
        avg_response_time = sum(
            check.response_time_ms for check in service_checks
        ) / len(service_checks)

        return {
            "service_name": service_name,
            "availability_percentage": (uptime_checks / len(service_checks)) * 100.0,
            "total_checks": len(service_checks),
            "uptime_checks": uptime_checks,
            "average_response_time_ms": avg_response_time,
            "analysis_period_hours": hours,
        }

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """Get comprehensive monitoring performance metrics.

        Returns:
            Dictionary with monitoring metrics
        """
        metrics = self._performance_metrics.copy()

        # Calculate success rate
        total_checks = metrics["total_health_checks"]
        if total_checks > 0:
            metrics["success_rate"] = (
                metrics["successful_checks"] / total_checks
            ) * 100.0
        else:
            metrics["success_rate"] = 0.0

        # Add current state
        metrics.update(
            {
                "health_history_size": len(self._health_history),
                "last_check_time": self._last_health_check,
                "check_interval_seconds": self.health_check_interval,
                "monitoring_active": True,
            }
        )

        return metrics

    async def shutdown(self) -> None:
        """Gracefully shutdown the health monitor."""
        if self._http_client:
            await self._http_client.aclose()

        self.logger.info("🔌 SystemHealthMonitor shutdown complete")


# Module-level singleton for easy access
_system_health_monitor: SystemHealthMonitor | None = None


def get_system_health_monitor(settings: Settings | None = None) -> SystemHealthMonitor:
    """Get singleton instance of SystemHealthMonitor.

    Args:
        settings: Application settings

    Returns:
        SystemHealthMonitor singleton instance
    """
    global _system_health_monitor
    if _system_health_monitor is None:
        _system_health_monitor = SystemHealthMonitor(settings)
    return _system_health_monitor


def reset_system_health_monitor() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _system_health_monitor
    _system_health_monitor = None
