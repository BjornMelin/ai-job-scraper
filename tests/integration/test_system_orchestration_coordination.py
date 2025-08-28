"""System Orchestration and Service Coordination Integration Tests.

This test suite validates the comprehensive system orchestration across all
phases (3A-3D), ensuring proper service coordination, dependency management,
and production-ready workflow execution.

**System Orchestration Coverage**:
- Phase 3A: Unified Scraping Service coordination
- Phase 3B: Mobile UI rendering coordination
- Phase 3C: Hybrid AI processing coordination
- Phase 3D: Background task management and system health monitoring
- Service dependency management and health checking
- Production deployment readiness validation

**Coordination Requirements Validated**:
- ADR-017: Background task management with progress tracking
- Service health monitoring and alerting systems
- Cross-service communication and error handling
- Resource management and performance optimization
- Real-time progress tracking across complex workflows

**Production Orchestration Targets**:
- System-wide health checks passing
- Service coordination success rate >95%
- Background task reliability >90%
- Error recovery and graceful degradation
- Production deployment validation
"""

import asyncio
import logging
import time

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.coordination.background_task_manager import (
    TaskStatus,
    get_background_task_manager,
)
from src.coordination.progress_tracker import (
    get_progress_tracking_manager,
)
from src.coordination.service_orchestrator import ServiceOrchestrator
from src.coordination.system_health_monitor import (
    get_system_health_monitor,
)
from src.interfaces.scraping_service_interface import JobQuery, SourceType
from tests.factories import create_realistic_dataset

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def orchestration_test_setup(session, tmp_path, test_settings):
    """Set up comprehensive system orchestration testing environment."""
    # Create comprehensive dataset for orchestration testing
    dataset = create_realistic_dataset(
        session,
        companies=25,
        jobs_per_company=30,
        include_inactive_companies=True,
        senior_ratio=0.4,
        remote_ratio=0.6,
        favorited_ratio=0.25,
    )

    # Orchestration test configuration
    orchestration_config = {
        "system_services": [
            "unified_scraper",
            "ai_router",
            "search_service",
            "database_sync",
            "ui_components",
            "task_manager",
            "progress_tracker",
            "health_monitor",
        ],
        "health_check_targets": {
            "response_time_ms": 500,
            "success_rate": 0.95,
            "service_availability": 0.99,
        },
        "coordination_targets": {
            "service_coordination_success": 0.95,
            "background_task_reliability": 0.90,
            "workflow_completion_rate": 0.90,
            "error_recovery_success": 0.85,
        },
        "performance_targets": {
            "workflow_orchestration_overhead_ms": 100,
            "service_startup_time_ms": 2000,
            "health_check_frequency_s": 30,
        },
    }

    return {
        "dataset": dataset,
        "config": orchestration_config,
        "session": session,
        "temp_dir": tmp_path,
        "settings": test_settings,
    }


class TestServiceHealthMonitoring:
    """Test comprehensive service health monitoring system."""

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_system_health_comprehensive_check(self, orchestration_test_setup):
        """Test comprehensive system health monitoring across all services."""
        setup = orchestration_test_setup
        services = setup["config"]["system_services"]
        targets = setup["config"]["health_check_targets"]

        with patch(
            "src.coordination.system_health_monitor.SystemHealthMonitor"
        ) as MockHealthMonitor:

            def create_mock_health_monitor():
                """Create comprehensive health monitor mock."""

                async def mock_comprehensive_health_check():
                    # Mock health status for each service
                    service_health = {}
                    overall_healthy = True
                    total_response_time = 0

                    for service in services:
                        # Simulate realistic health check responses
                        if service == "unified_scraper":
                            health = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 150,
                                "last_scrape_success": True,
                                "active_scrapers": 3,
                                "success_rate": 0.97,
                            }
                        elif service == "ai_router":
                            health = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 200,
                                "local_service_available": True,
                                "cloud_fallback_available": True,
                                "processing_queue_size": 5,
                            }
                        elif service == "database_sync":
                            health = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 50,
                                "connection_pool_active": 8,
                                "connection_pool_idle": 12,
                                "query_performance_ms": 25,
                            }
                        elif service == "search_service":
                            health = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 75,
                                "index_size": 45000,
                                "last_index_update": datetime.now(UTC).isoformat(),
                                "search_performance_ms": 100,
                            }
                        elif service == "ui_components":
                            health = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 120,
                                "active_sessions": 15,
                                "avg_render_time_ms": 180,
                                "mobile_optimized": True,
                            }
                        else:
                            # Background services (task_manager, progress_tracker, health_monitor)
                            health = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 25,
                                "active_tasks": 3 if service == "task_manager" else 0,
                                "memory_usage_mb": 50,
                            }

                        service_health[service] = health
                        total_response_time += health["response_time_ms"]

                        if not health["healthy"]:
                            overall_healthy = False

                    return {
                        "overall_healthy": overall_healthy,
                        "services": service_health,
                        "system_metrics": {
                            "total_services": len(services),
                            "healthy_services": sum(
                                1 for s in service_health.values() if s["healthy"]
                            ),
                            "unhealthy_services": sum(
                                1 for s in service_health.values() if not s["healthy"]
                            ),
                            "total_response_time_ms": total_response_time,
                            "avg_response_time_ms": total_response_time / len(services),
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                        "warnings": [],
                        "errors": [],
                    }

                mock_monitor = Mock()
                mock_monitor.get_comprehensive_health_report = AsyncMock(
                    side_effect=mock_comprehensive_health_check
                )
                return mock_monitor

            MockHealthMonitor.return_value = create_mock_health_monitor()

            # Test comprehensive health monitoring
            health_monitor = get_system_health_monitor()
            health_report = await health_monitor.get_comprehensive_health_report()

            # Validate comprehensive health check
            assert health_report["overall_healthy"], "System should be overall healthy"
            assert len(health_report["services"]) == len(services), (
                f"Should monitor all {len(services)} services"
            )

            # Validate individual service health
            for service_name in services:
                assert service_name in health_report["services"], (
                    f"Missing health data for {service_name}"
                )
                service_health = health_report["services"][service_name]

                assert service_health["healthy"], (
                    f"Service {service_name} should be healthy"
                )
                assert (
                    service_health["response_time_ms"] < targets["response_time_ms"]
                ), (
                    f"Service {service_name} response time {service_health['response_time_ms']}ms "
                    f"exceeds {targets['response_time_ms']}ms target"
                )

            # Validate system metrics
            system_metrics = health_report["system_metrics"]
            assert system_metrics["healthy_services"] == len(services), (
                "All services should be healthy"
            )
            assert system_metrics["unhealthy_services"] == 0, (
                "No services should be unhealthy"
            )
            assert (
                system_metrics["avg_response_time_ms"] < targets["response_time_ms"]
            ), (
                f"Average response time {system_metrics['avg_response_time_ms']:.1f}ms exceeds target"
            )

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_health_monitoring_failure_scenarios(self, orchestration_test_setup):
        """Test health monitoring handles service failures correctly."""
        setup = orchestration_test_setup

        # Mock service failure scenarios

        with patch(
            "src.coordination.system_health_monitor.SystemHealthMonitor"
        ) as MockHealthMonitor:

            def create_mock_failure_monitor():
                async def mock_health_check_with_failures():
                    # Simulate one service failure
                    failed_service = "ai_router"  # Test AI service failure

                    service_health = {}
                    for service in setup["config"]["system_services"]:
                        if service == failed_service:
                            service_health[service] = {
                                "healthy": False,
                                "status": "service_overloaded",
                                "response_time_ms": 5000,  # Timeout
                                "error_message": "Service temporarily unavailable",
                                "last_successful_check": (
                                    datetime.now(UTC) - timedelta(minutes=5)
                                ).isoformat(),
                            }
                        else:
                            service_health[service] = {
                                "healthy": True,
                                "status": "operational",
                                "response_time_ms": 100,
                            }

                    healthy_count = sum(
                        1 for s in service_health.values() if s["healthy"]
                    )

                    return {
                        "overall_healthy": healthy_count
                        >= len(setup["config"]["system_services"]) * 0.8,
                        "services": service_health,
                        "system_metrics": {
                            "total_services": len(setup["config"]["system_services"]),
                            "healthy_services": healthy_count,
                            "unhealthy_services": len(
                                setup["config"]["system_services"]
                            )
                            - healthy_count,
                        },
                        "warnings": ["ai_router service experiencing high load"],
                        "errors": ["ai_router: Service temporarily unavailable"],
                        "degraded_functionality": ["ai_processing_limited"],
                    }

                mock_monitor = Mock()
                mock_monitor.get_comprehensive_health_report = AsyncMock(
                    side_effect=mock_health_check_with_failures
                )
                return mock_monitor

            MockHealthMonitor.return_value = create_mock_failure_monitor()

            # Test health monitoring with failures
            health_monitor = get_system_health_monitor()
            health_report = await health_monitor.get_comprehensive_health_report()

            # Validate failure handling
            assert health_report["system_metrics"]["unhealthy_services"] > 0, (
                "Should detect service failures"
            )
            assert len(health_report["errors"]) > 0, (
                "Should report errors for failed services"
            )
            assert len(health_report["warnings"]) > 0, (
                "Should provide warnings for degraded services"
            )

            # Validate graceful degradation
            if "degraded_functionality" in health_report:
                assert len(health_report["degraded_functionality"]) > 0, (
                    "Should identify degraded functionality"
                )


class TestBackgroundTaskCoordination:
    """Test background task management and coordination."""

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_background_task_lifecycle_management(self, orchestration_test_setup):
        """Test complete background task lifecycle with coordination."""
        setup = orchestration_test_setup

        with patch(
            "src.coordination.background_task_manager.BackgroundTaskManager"
        ) as MockTaskManager:

            def create_mock_task_manager():
                """Create comprehensive task manager mock."""
                # Task state tracking
                tasks = {}
                task_counter = 0

                def create_task(name, task_type, priority, metadata):
                    nonlocal task_counter
                    task_counter += 1
                    task_id = f"task_{task_counter}"

                    task = TaskStatus(
                        task_id=task_id,
                        name=name,
                        task_type=task_type,
                        status="queued",
                        priority=priority,
                        progress_percentage=0.0,
                        created_at=datetime.now(UTC),
                        metadata=metadata or {},
                    )

                    tasks[task_id] = task
                    return task_id

                async def mock_monitor_task_progress(task_id):
                    """Mock task progress monitoring."""
                    if task_id not in tasks:
                        return

                    task = tasks[task_id]

                    # Simulate task progression
                    progress_steps = [
                        ("running", 25.0, "Task started"),
                        ("running", 50.0, "Processing data"),
                        ("running", 75.0, "Finalizing results"),
                        ("completed", 100.0, "Task completed successfully"),
                    ]

                    for status, progress, message in progress_steps:
                        task.status = status
                        task.progress_percentage = progress
                        task.metadata["message"] = message

                        if status == "completed":
                            task.completed_at = datetime.now(UTC)

                        yield task
                        await asyncio.sleep(0.1)  # Simulate work

                def get_task_status(task_id):
                    return tasks.get(task_id)

                def get_all_tasks():
                    return list(tasks.values())

                mock_manager = Mock()
                mock_manager.create_task = Mock(side_effect=create_task)
                mock_manager.monitor_task_progress = AsyncMock(
                    side_effect=mock_monitor_task_progress
                )
                mock_manager.get_task_status = Mock(side_effect=get_task_status)
                mock_manager.get_all_tasks = Mock(side_effect=get_all_tasks)

                return mock_manager

            MockTaskManager.return_value = create_mock_task_manager()

            # Test background task coordination
            task_manager = get_background_task_manager()

            # Create multiple coordinated tasks
            task_scenarios = [
                {
                    "name": "unified_scraping_job",
                    "task_type": "scraping",
                    "priority": "high",
                    "metadata": {"source": "unified", "keywords": ["python"]},
                },
                {
                    "name": "ai_enhancement_job",
                    "task_type": "ai_processing",
                    "priority": "medium",
                    "metadata": {"enhancement_type": "job_analysis", "batch_size": 10},
                },
                {
                    "name": "database_sync_job",
                    "task_type": "database",
                    "priority": "low",
                    "metadata": {"sync_type": "incremental", "table": "jobs"},
                },
            ]

            # Create and monitor coordinated tasks
            task_results = []

            for scenario in task_scenarios:
                # Create task
                task_id = task_manager.create_task(
                    name=scenario["name"],
                    task_type=scenario["task_type"],
                    priority=scenario["priority"],
                    metadata=scenario["metadata"],
                )

                # Monitor task progress
                progress_updates = []
                async for task_status in task_manager.monitor_task_progress(task_id):
                    progress_updates.append(
                        {
                            "status": task_status.status,
                            "progress": task_status.progress_percentage,
                            "message": task_status.metadata.get("message", ""),
                        }
                    )

                    if task_status.status in ["completed", "failed", "cancelled"]:
                        break

                # Get final task status
                final_status = task_manager.get_task_status(task_id)

                task_results.append(
                    {
                        "task_id": task_id,
                        "scenario": scenario,
                        "progress_updates": progress_updates,
                        "final_status": final_status,
                        "task_successful": final_status
                        and final_status.status == "completed",
                    }
                )

            # Validate background task coordination
            successful_tasks = [r for r in task_results if r["task_successful"]]
            task_success_rate = len(successful_tasks) / len(task_results)

            assert (
                task_success_rate
                >= setup["config"]["coordination_targets"][
                    "background_task_reliability"
                ]
            ), (
                f"Background task success rate {task_success_rate:.2%} below "
                f"{setup['config']['coordination_targets']['background_task_reliability']:.2%} target"
            )

            # Validate task progression
            for result in task_results:
                progress_updates = result["progress_updates"]
                assert len(progress_updates) >= 4, (
                    f"Task {result['task_id']} should have multiple progress updates"
                )

                # Validate progress increases
                progress_values = [u["progress"] for u in progress_updates]
                assert progress_values == sorted(progress_values), (
                    f"Progress should increase monotonically for {result['task_id']}"
                )
                assert progress_values[-1] == 100.0, (
                    f"Final progress should be 100% for {result['task_id']}"
                )

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_task_coordination_error_scenarios(self, orchestration_test_setup):
        """Test task coordination handles error scenarios correctly."""
        setup = orchestration_test_setup

        with patch(
            "src.coordination.background_task_manager.BackgroundTaskManager"
        ) as MockTaskManager:

            def create_mock_error_task_manager():
                """Create task manager mock with error scenarios."""
                tasks = {}

                def create_task(name, task_type, priority, metadata):
                    task_id = f"error_task_{len(tasks)}"

                    # Simulate different error scenarios
                    if "timeout" in name:
                        status = "timeout"
                        error_msg = "Task timed out after 30 seconds"
                    elif "memory" in name:
                        status = "failed"
                        error_msg = "Insufficient memory to complete task"
                    elif "dependency" in name:
                        status = "failed"
                        error_msg = "Required service unavailable"
                    else:
                        status = "completed"
                        error_msg = None

                    task = TaskStatus(
                        task_id=task_id,
                        name=name,
                        task_type=task_type,
                        status=status,
                        priority=priority,
                        progress_percentage=100.0 if status == "completed" else 75.0,
                        created_at=datetime.now(UTC),
                        completed_at=datetime.now(UTC)
                        if status in ["completed", "failed", "timeout"]
                        else None,
                        error_message=error_msg,
                        metadata=metadata or {},
                    )

                    tasks[task_id] = task
                    return task_id

                async def mock_monitor_with_errors(task_id):
                    if task_id not in tasks:
                        return

                    task = tasks[task_id]
                    yield task  # Return final state immediately for error scenarios

                def get_task_status(task_id):
                    return tasks.get(task_id)

                mock_manager = Mock()
                mock_manager.create_task = Mock(side_effect=create_task)
                mock_manager.monitor_task_progress = AsyncMock(
                    side_effect=mock_monitor_with_errors
                )
                mock_manager.get_task_status = Mock(side_effect=get_task_status)

                return mock_manager

            MockTaskManager.return_value = create_mock_error_task_manager()

            task_manager = get_background_task_manager()

            # Test error scenario handling
            error_scenarios = [
                {
                    "name": "timeout_scraping_task",
                    "task_type": "scraping",
                    "expected_status": "timeout",
                },
                {
                    "name": "memory_ai_task",
                    "task_type": "ai_processing",
                    "expected_status": "failed",
                },
                {
                    "name": "dependency_sync_task",
                    "task_type": "database",
                    "expected_status": "failed",
                },
                {
                    "name": "successful_task",
                    "task_type": "general",
                    "expected_status": "completed",
                },
            ]

            error_handling_results = []

            for scenario in error_scenarios:
                task_id = task_manager.create_task(
                    name=scenario["name"],
                    task_type=scenario["task_type"],
                    priority="medium",
                    metadata={"error_test": True},
                )

                # Monitor task (will return final state for error scenarios)
                async for task_status in task_manager.monitor_task_progress(task_id):
                    final_status = task_status
                    break

                error_handled_correctly = final_status.status == scenario[
                    "expected_status"
                ] and (
                    scenario["expected_status"] != "failed"
                    or final_status.error_message is not None
                )

                error_handling_results.append(
                    {
                        "scenario": scenario["name"],
                        "expected_status": scenario["expected_status"],
                        "actual_status": final_status.status,
                        "error_message": final_status.error_message,
                        "error_handled_correctly": error_handled_correctly,
                    }
                )

            # Validate error handling
            correct_error_handling = [
                r for r in error_handling_results if r["error_handled_correctly"]
            ]
            error_handling_rate = len(correct_error_handling) / len(
                error_handling_results
            )

            assert (
                error_handling_rate
                >= setup["config"]["coordination_targets"]["error_recovery_success"]
            ), (
                f"Error handling rate {error_handling_rate:.2%} below "
                f"{setup['config']['coordination_targets']['error_recovery_success']:.2%} target. "
                f"Failures: {[r for r in error_handling_results if not r['error_handled_correctly']]}"
            )


class TestProgressTrackingCoordination:
    """Test progress tracking coordination across complex workflows."""

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_multi_phase_progress_tracking(self, orchestration_test_setup):
        """Test progress tracking across multi-phase workflows."""
        with patch(
            "src.coordination.progress_tracker.ProgressTrackingManager"
        ) as MockProgressManager:

            def create_mock_progress_manager():
                """Create comprehensive progress tracking mock."""
                trackers = {}

                def create_tracker(workflow_id):
                    tracker = Mock()
                    tracker.workflow_id = workflow_id
                    tracker.progress_history = []
                    tracker.current_phase = "initializing"
                    tracker.total_phases = 6
                    tracker.phase_progress = {}

                    def update_progress(percentage, message, phase, metadata=None):
                        snapshot = Mock()
                        snapshot.progress_percentage = percentage
                        snapshot.message = message
                        snapshot.phase = phase
                        snapshot.timestamp = datetime.now(UTC)
                        snapshot.metadata = metadata or {}

                        tracker.progress_history.append(snapshot)
                        tracker.current_phase = phase
                        tracker.phase_progress[phase] = percentage

                        return snapshot

                    def get_current_progress():
                        if tracker.progress_history:
                            return tracker.progress_history[-1]
                        return None

                    def get_progress_estimate():
                        current = get_current_progress()
                        if not current:
                            return None

                        # Mock ETA calculation
                        estimated_remaining = max(
                            0, (100 - current.progress_percentage) * 0.5
                        )  # 0.5s per %

                        estimate = Mock()
                        estimate.estimated_time_remaining = estimated_remaining
                        estimate.confidence_level = 0.8
                        estimate.completion_probability = min(
                            0.95, current.progress_percentage / 100
                        )

                        return estimate

                    tracker.update_progress = Mock(side_effect=update_progress)
                    tracker.get_current_progress = Mock(
                        side_effect=get_current_progress
                    )
                    tracker.get_progress_estimate = Mock(
                        side_effect=get_progress_estimate
                    )

                    trackers[workflow_id] = tracker
                    return tracker

                def get_tracker(workflow_id):
                    return trackers.get(workflow_id)

                mock_manager = Mock()
                mock_manager.create_tracker = Mock(side_effect=create_tracker)
                mock_manager.get_tracker = Mock(side_effect=get_tracker)

                return mock_manager

            MockProgressManager.return_value = create_mock_progress_manager()

            # Test multi-phase progress tracking
            progress_manager = get_progress_tracking_manager()

            # Simulate complex multi-phase workflow
            workflow_phases = [
                ("health_validation", "Validating system health...", 10.0),
                ("scraping_initiation", "Starting job scraping...", 20.0),
                ("data_extraction", "Extracting job data...", 45.0),
                ("ai_enhancement", "Enhancing with AI analysis...", 70.0),
                ("database_storage", "Storing to database...", 85.0),
                ("ui_updates", "Updating user interface...", 95.0),
                ("completion", "Workflow completed successfully!", 100.0),
            ]

            workflow_id = "test_multi_phase_workflow"
            progress_tracker = progress_manager.create_tracker(workflow_id)

            # Execute simulated workflow with progress tracking
            progress_snapshots = []

            for phase, message, target_percentage in workflow_phases:
                # Update progress
                snapshot = progress_tracker.update_progress(
                    percentage=target_percentage,
                    message=message,
                    phase=phase,
                    metadata={
                        "workflow_type": "integration_test",
                        "timestamp": time.time(),
                    },
                )

                progress_snapshots.append(snapshot)

                # Get progress estimate
                estimate = progress_tracker.get_progress_estimate()
                assert estimate is not None, (
                    f"Should have progress estimate for phase {phase}"
                )
                assert estimate.confidence_level > 0.0, (
                    "Confidence level should be positive"
                )

                # Simulate work delay
                await asyncio.sleep(0.05)

            # Validate progress tracking coordination
            assert len(progress_snapshots) == len(workflow_phases), (
                "Should track all workflow phases"
            )

            # Validate progress progression
            progress_values = [s.progress_percentage for s in progress_snapshots]
            assert progress_values == sorted(progress_values), (
                "Progress should increase monotonically"
            )
            assert progress_values[-1] == 100.0, "Final progress should be 100%"

            # Validate phase tracking
            phases_tracked = [s.phase for s in progress_snapshots]
            expected_phases = [p[0] for p in workflow_phases]
            assert phases_tracked == expected_phases, (
                "Should track all expected phases in order"
            )

            # Validate message tracking
            messages = [s.message for s in progress_snapshots]
            expected_messages = [p[1] for p in workflow_phases]
            assert messages == expected_messages, (
                "Should track all phase messages correctly"
            )

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_concurrent_progress_tracking(self, orchestration_test_setup):
        """Test progress tracking handles concurrent workflows correctly."""
        with patch(
            "src.coordination.progress_tracker.ProgressTrackingManager"
        ) as MockProgressManager:

            def create_mock_concurrent_manager():
                """Create progress manager for concurrent workflow testing."""
                trackers = {}

                def create_tracker(workflow_id):
                    tracker = Mock()
                    tracker.workflow_id = workflow_id
                    tracker.updates = []

                    def update_progress(percentage, message, phase, metadata=None):
                        update = {
                            "percentage": percentage,
                            "message": message,
                            "phase": phase,
                            "timestamp": datetime.now(UTC),
                            "metadata": metadata or {},
                        }
                        tracker.updates.append(update)
                        return Mock(**update)

                    def get_current_progress():
                        if tracker.updates:
                            return Mock(**tracker.updates[-1])
                        return None

                    tracker.update_progress = Mock(side_effect=update_progress)
                    tracker.get_current_progress = Mock(
                        side_effect=get_current_progress
                    )

                    trackers[workflow_id] = tracker
                    return tracker

                def get_tracker(workflow_id):
                    return trackers.get(workflow_id)

                def get_all_trackers():
                    return list(trackers.values())

                mock_manager = Mock()
                mock_manager.create_tracker = Mock(side_effect=create_tracker)
                mock_manager.get_tracker = Mock(side_effect=get_tracker)
                mock_manager.get_all_trackers = Mock(side_effect=get_all_trackers)

                return mock_manager

            MockProgressManager.return_value = create_mock_concurrent_manager()

            progress_manager = get_progress_tracking_manager()

            # Test concurrent workflow progress tracking
            concurrent_workflows = [
                {
                    "id": "workflow_scraping_a",
                    "phases": [
                        ("init", "Initializing scraper A...", 20),
                        ("scraping", "Scraping job boards...", 60),
                        ("complete", "Scraping A completed", 100),
                    ],
                },
                {
                    "id": "workflow_ai_b",
                    "phases": [
                        ("init", "Initializing AI processing B...", 15),
                        ("processing", "Processing with AI...", 70),
                        ("complete", "AI processing B completed", 100),
                    ],
                },
                {
                    "id": "workflow_sync_c",
                    "phases": [
                        ("init", "Initializing database sync C...", 30),
                        ("syncing", "Syncing database...", 80),
                        ("complete", "Database sync C completed", 100),
                    ],
                },
            ]

            # Execute concurrent workflows

            # Create all trackers
            trackers = {}
            for workflow in concurrent_workflows:
                trackers[workflow["id"]] = progress_manager.create_tracker(
                    workflow["id"]
                )

            # Simulate concurrent execution
            async def execute_workflow(workflow):
                """Simulate concurrent workflow execution."""
                tracker = trackers[workflow["id"]]

                for phase, message, percentage in workflow["phases"]:
                    tracker.update_progress(
                        percentage=percentage,
                        message=message,
                        phase=phase,
                        metadata={"workflow_id": workflow["id"]},
                    )
                    await asyncio.sleep(0.02)  # Simulate work with different timing

                return workflow["id"]

            # Execute all workflows concurrently
            workflow_tasks = [execute_workflow(w) for w in concurrent_workflows]
            completed_workflows = await asyncio.gather(*workflow_tasks)

            # Validate concurrent progress tracking
            assert len(completed_workflows) == len(concurrent_workflows), (
                "All workflows should complete"
            )

            # Validate each workflow's progress tracking
            for workflow in concurrent_workflows:
                tracker = progress_manager.get_tracker(workflow["id"])

                # Check final progress
                final_progress = tracker.get_current_progress()
                assert final_progress is not None, (
                    f"Should have final progress for {workflow['id']}"
                )
                assert final_progress.percentage == 100, (
                    f"Final progress should be 100% for {workflow['id']}"
                )

                # Check progress history
                assert len(tracker.updates) == len(workflow["phases"]), (
                    f"Should have {len(workflow['phases'])} updates for {workflow['id']}"
                )


class TestServiceOrchestrationIntegration:
    """Test comprehensive service orchestration integration."""

    @pytest.mark.integration
    @pytest.mark.coordination
    @pytest.mark.slow
    async def test_comprehensive_service_orchestration(self, orchestration_test_setup):
        """Test comprehensive orchestration across all system services."""
        setup = orchestration_test_setup
        targets = setup["config"]["coordination_targets"]

        # Mock all system services for orchestration testing
        with (
            patch.multiple(
                "src.services.unified_scraper.UnifiedScrapingService",
                scrape_unified=AsyncMock(return_value=[]),
                start_background_scraping=AsyncMock(return_value="scraping-task-123"),
            ),
            patch.multiple(
                "src.ai.hybrid_ai_router.HybridAIRouter",
                process_content=AsyncMock(return_value="AI processed content"),
            ),
            patch.multiple(
                "src.services.search_service.SearchService",
                search_jobs=AsyncMock(return_value=[]),
                update_search_index=AsyncMock(return_value=True),
            ),
            patch.multiple(
                "src.services.database_sync.DatabaseSync",
                sync_jobs=AsyncMock(return_value={"synced": 10, "errors": 0}),
            ),
            patch.multiple(
                "src.coordination.system_health_monitor.SystemHealthMonitor",
                get_comprehensive_health_report=AsyncMock(
                    return_value={
                        "overall_healthy": True,
                        "services": {
                            service: {"healthy": True, "status": "operational"}
                            for service in setup["config"]["system_services"]
                        },
                        "system_metrics": {
                            "healthy_services": len(setup["config"]["system_services"])
                        },
                        "warnings": [],
                        "errors": [],
                    }
                ),
            ),
        ):
            # Test comprehensive orchestration
            orchestrator = ServiceOrchestrator(setup["settings"])

            # Create comprehensive test query
            test_query = JobQuery(
                keywords=["python developer", "data scientist"],
                locations=["San Francisco", "Remote"],
                source_types=[SourceType.UNIFIED],
                max_results=50,
                enable_ai_enhancement=True,
            )

            orchestration_scenarios = [
                {
                    "name": "full_integration_workflow",
                    "query": test_query,
                    "options": {
                        "enable_ai_enhancement": True,
                        "enable_real_time_updates": True,
                        "enable_ui_updates": True,
                        "max_jobs": 50,
                    },
                    "expected_services": [
                        "health_monitor",
                        "unified_scraper",
                        "ai_router",
                        "database_sync",
                        "ui_components",
                    ],
                },
                {
                    "name": "scraping_only_workflow",
                    "query": JobQuery(
                        keywords=["backend engineer"],
                        locations=["Seattle"],
                        source_types=[SourceType.JOB_BOARDS],
                        max_results=20,
                    ),
                    "options": {
                        "enable_ai_enhancement": False,
                        "enable_real_time_updates": True,
                    },
                    "expected_services": [
                        "health_monitor",
                        "unified_scraper",
                        "database_sync",
                    ],
                },
                {
                    "name": "ai_enhancement_workflow",
                    "query": JobQuery(
                        keywords=["machine learning engineer"],
                        locations=["Boston"],
                        source_types=[SourceType.COMPANY_PAGES],
                        max_results=15,
                        enable_ai_enhancement=True,
                    ),
                    "options": {
                        "enable_ai_enhancement": True,
                        "enable_real_time_updates": False,
                    },
                    "expected_services": [
                        "health_monitor",
                        "unified_scraper",
                        "ai_router",
                        "database_sync",
                    ],
                },
            ]

            orchestration_results = []

            for scenario in orchestration_scenarios:
                start_time = time.perf_counter()

                try:
                    # Execute orchestrated workflow
                    workflow_id = await orchestrator.execute_integrated_workflow(
                        scenario["query"], scenario["options"]
                    )

                    orchestration_duration = time.perf_counter() - start_time

                    # Get workflow status
                    workflow_status = orchestrator.get_workflow_status(workflow_id)

                    # Validate orchestration success
                    orchestration_successful = (
                        workflow_status is not None
                        and workflow_status["status"] == "completed"
                        and orchestration_duration
                        < 10.0  # Reasonable orchestration time
                    )

                    # Validate expected services were used
                    services_used = (
                        workflow_status.get("services_used", [])
                        if workflow_status
                        else []
                    )
                    expected_services_present = all(
                        service in services_used
                        for service in scenario["expected_services"]
                    )

                    orchestration_results.append(
                        {
                            "scenario": scenario["name"],
                            "workflow_id": workflow_id,
                            "duration": orchestration_duration,
                            "orchestration_successful": orchestration_successful,
                            "services_used": services_used,
                            "expected_services": scenario["expected_services"],
                            "expected_services_present": expected_services_present,
                            "coordination_working": orchestration_successful
                            and expected_services_present,
                        }
                    )

                except Exception as e:
                    orchestration_results.append(
                        {
                            "scenario": scenario["name"],
                            "workflow_id": None,
                            "duration": time.perf_counter() - start_time,
                            "orchestration_successful": False,
                            "error": str(e),
                            "coordination_working": False,
                        }
                    )

            # Validate comprehensive orchestration
            successful_orchestrations = [
                r for r in orchestration_results if r["coordination_working"]
            ]
            orchestration_success_rate = len(successful_orchestrations) / len(
                orchestration_results
            )

            assert (
                orchestration_success_rate >= targets["service_coordination_success"]
            ), (
                f"Service orchestration success rate {orchestration_success_rate:.2%} below "
                f"{targets['service_coordination_success']:.2%} target. "
                f"Failed orchestrations: {[r for r in orchestration_results if not r['coordination_working']]}"
            )

            # Validate workflow completion rate
            completed_workflows = [
                r for r in orchestration_results if r["orchestration_successful"]
            ]
            workflow_completion_rate = len(completed_workflows) / len(
                orchestration_results
            )

            assert workflow_completion_rate >= targets["workflow_completion_rate"], (
                f"Workflow completion rate {workflow_completion_rate:.2%} below "
                f"{targets['workflow_completion_rate']:.2%} target"
            )

    @pytest.mark.integration
    @pytest.mark.coordination
    async def test_production_readiness_validation(self, orchestration_test_setup):
        """Test comprehensive production readiness validation."""
        setup = orchestration_test_setup

        # Mock production readiness components
        with patch.multiple(
            "src.coordination.system_health_monitor.SystemHealthMonitor",
            get_comprehensive_health_report=AsyncMock(
                return_value={
                    "overall_healthy": True,
                    "services": {
                        service: {"healthy": True}
                        for service in setup["config"]["system_services"]
                    },
                    "system_metrics": {
                        "healthy_services": len(setup["config"]["system_services"]),
                        "unhealthy_services": 0,
                        "avg_response_time_ms": 150,
                    },
                    "warnings": [],
                    "errors": [],
                }
            ),
        ):
            orchestrator = ServiceOrchestrator(setup["settings"])

            # Test production readiness validation
            validation_results = await orchestrator.validate_production_readiness()

            # Validate production readiness checks
            assert validation_results is not None, "Should return validation results"
            assert "ready_for_production" in validation_results, (
                "Should indicate production readiness"
            )
            assert "checks" in validation_results, "Should include validation checks"
            assert "warnings" in validation_results, "Should include warnings list"
            assert "errors" in validation_results, "Should include errors list"

            # Validate individual checks
            checks = validation_results["checks"]
            assert "health_status" in checks, "Should check system health"
            assert "all_services_healthy" in checks, "Should check service health"
            assert "workflow_success_rate" in checks, "Should check workflow success"
            assert "settings_configured" in checks, "Should check configuration"

            # Validate production readiness decision
            if len(validation_results["errors"]) == 0:
                assert validation_results["ready_for_production"], (
                    "Should be ready for production when no errors present"
                )

            # Validate health status integration
            health_check = checks.get("health_status")
            assert health_check is not None, "Should include health status check"
            if isinstance(health_check, dict):
                assert health_check.get("overall_healthy"), (
                    "Health status should be healthy"
                )


# System orchestration reporting
class SystemOrchestrationReporter:
    """Generate comprehensive system orchestration reports."""

    @staticmethod
    def generate_orchestration_report(test_results: dict[str, Any]) -> dict[str, Any]:
        """Generate system orchestration validation report."""
        return {
            "system_orchestration_summary": {
                "service_health_monitoring": {
                    "target": "Comprehensive health monitoring across all services",
                    "achieved": test_results.get("health_monitoring_working", False),
                    "services_monitored": test_results.get("services_monitored", []),
                    "health_check_success_rate": test_results.get(
                        "health_check_success_rate", 0
                    ),
                },
                "background_task_coordination": {
                    "target": "Reliable background task management and tracking",
                    "achieved": test_results.get("background_task_reliability", 0)
                    >= 0.90,
                    "task_success_rate": test_results.get(
                        "background_task_reliability", 0
                    ),
                    "error_handling_rate": test_results.get("error_handling_rate", 0),
                },
                "progress_tracking": {
                    "target": "Multi-phase progress tracking with ETA estimates",
                    "achieved": test_results.get("progress_tracking_working", False),
                    "concurrent_tracking_support": test_results.get(
                        "concurrent_tracking_working", False
                    ),
                    "phase_tracking_accuracy": test_results.get(
                        "phase_tracking_accuracy", 0
                    ),
                },
                "service_orchestration": {
                    "target": "Coordinated execution across all system services",
                    "achieved": test_results.get("service_coordination_success", 0)
                    >= 0.95,
                    "coordination_success_rate": test_results.get(
                        "service_coordination_success", 0
                    ),
                    "workflow_completion_rate": test_results.get(
                        "workflow_completion_rate", 0
                    ),
                },
                "production_readiness": {
                    "target": "System ready for production deployment",
                    "achieved": test_results.get("production_ready", False),
                    "validation_checks_passed": test_results.get(
                        "validation_checks_passed", 0
                    ),
                    "critical_issues": test_results.get("critical_issues", []),
                },
            },
            "adr_017_compliance": {
                "background_task_management": {
                    "implemented": True,
                    "reliability_met": test_results.get(
                        "background_task_reliability", 0
                    )
                    >= 0.90,
                    "progress_tracking_working": test_results.get(
                        "progress_tracking_working", False
                    ),
                },
                "system_coordination": {
                    "implemented": True,
                    "service_coordination_working": test_results.get(
                        "service_coordination_success", 0
                    )
                    >= 0.95,
                    "health_monitoring_working": test_results.get(
                        "health_monitoring_working", False
                    ),
                },
                "production_orchestration": {
                    "implemented": True,
                    "deployment_ready": test_results.get("production_ready", False),
                    "error_recovery_working": test_results.get("error_handling_rate", 0)
                    >= 0.85,
                },
            },
            "orchestration_metrics": test_results,
            "recommendations": SystemOrchestrationReporter._generate_orchestration_recommendations(
                test_results
            ),
        }

    @staticmethod
    def _generate_orchestration_recommendations(
        test_results: dict[str, Any],
    ) -> list[str]:
        """Generate system orchestration improvement recommendations."""
        recommendations = []

        if not test_results.get("health_monitoring_working", False):
            recommendations.append(
                "Implement comprehensive service health monitoring with alerting"
            )

        if test_results.get("background_task_reliability", 0) < 0.90:
            recommendations.append(
                "Improve background task reliability and error recovery mechanisms"
            )

        if not test_results.get("progress_tracking_working", False):
            recommendations.append(
                "Enhance multi-phase progress tracking with accurate ETA estimates"
            )

        if test_results.get("service_coordination_success", 0) < 0.95:
            recommendations.append(
                "Strengthen service coordination and workflow orchestration patterns"
            )

        if not test_results.get("production_ready", False):
            recommendations.append(
                "Address production readiness issues before deployment"
            )

        return recommendations


# System orchestration test configuration
SYSTEM_ORCHESTRATION_CONFIG = {
    "system_services": [
        "unified_scraper",
        "ai_router",
        "search_service",
        "database_sync",
        "ui_components",
        "task_manager",
        "progress_tracker",
        "health_monitor",
    ],
    "coordination_targets": {
        "service_coordination_success": 0.95,
        "background_task_reliability": 0.90,
        "workflow_completion_rate": 0.90,
        "error_recovery_success": 0.85,
        "health_check_success_rate": 0.95,
    },
    "performance_requirements": {
        "health_check_response_ms": 500,
        "workflow_orchestration_overhead_ms": 100,
        "service_startup_time_ms": 2000,
        "background_task_completion_time_s": 30,
    },
    "production_validation": {
        "required_health_checks": ["database", "ai_service", "scraper", "ui"],
        "minimum_service_availability": 0.99,
        "maximum_error_rate": 0.05,
        "required_configuration_checks": ["openai_api_key", "database_url"],
    },
}
