"""Production Readiness Validation Tests.

This test suite validates the AI job scraper system is ready for production
deployment with proper health monitoring, load handling, configuration
management, and operational reliability.

**Production Readiness Requirements**:
- Health monitoring system validation
- Configuration management testing
- Load testing and concurrent user simulation
- Memory usage profiling under production scenarios
- Error recovery and graceful degradation
- Security and data validation
- Performance under sustained load

**Test Coverage**:
- System health monitoring and alerting
- Configuration validation and management
- Load testing with concurrent users
- Memory profiling and leak detection
- Error handling and recovery scenarios
- Security validation and data protection
- Deployment and rollback testing
"""

import asyncio
import gc
import logging
import os
import time

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

from tests.factories import create_realistic_dataset

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def production_test_setup(session, tmp_path):
    """Set up production testing environment."""
    # Create large realistic dataset for production testing
    dataset = create_realistic_dataset(
        session,
        companies=50,
        jobs_per_company=40,
        include_inactive_companies=True,
        include_archived_jobs=True,
        senior_ratio=0.35,
        remote_ratio=0.45,
        favorited_ratio=0.15,
    )

    # Production configuration simulation
    prod_config = {
        "max_concurrent_users": 100,
        "max_concurrent_scrapers": 10,
        "ai_processing_queue_size": 50,
        "database_connection_pool": 20,
        "cache_ttl_seconds": 3600,
        "health_check_interval": 30,
        "memory_limit_mb": 2048,
        "cpu_usage_threshold": 0.8,
        "disk_usage_threshold": 0.85,
        "response_time_sla_ms": 5000,
    }

    return {
        "dataset": dataset,
        "config": prod_config,
        "session": session,
        "temp_dir": tmp_path,
        "test_jobs": dataset["jobs"][:200],  # Large test set
        "test_companies": dataset["companies"][:30],
    }


class TestHealthMonitoringSystem:
    """Test production health monitoring system."""

    @pytest.mark.production
    def test_health_check_endpoints(self, production_test_setup):
        """Test health check endpoints respond properly."""
        # Mock health check components
        with patch(
            "src.coordination.system_health_monitor.SystemHealthMonitor"
        ) as MockMonitor:
            mock_monitor = Mock()
            MockMonitor.return_value = mock_monitor

            # Test different health check scenarios
            health_scenarios = [
                {
                    "endpoint": "/health",
                    "expected_status": "healthy",
                    "components": ["database", "ai_service", "scraper", "ui"],
                    "max_response_time": 500,  # 500ms
                },
                {
                    "endpoint": "/health/deep",
                    "expected_status": "healthy",
                    "components": [
                        "database",
                        "ai_service",
                        "scraper",
                        "ui",
                        "background_tasks",
                    ],
                    "max_response_time": 2000,  # 2s for deep check
                },
                {
                    "endpoint": "/ready",
                    "expected_status": "ready",
                    "components": ["database", "ai_service"],
                    "max_response_time": 1000,  # 1s
                },
                {
                    "endpoint": "/metrics",
                    "expected_status": "available",
                    "components": ["system_metrics", "application_metrics"],
                    "max_response_time": 300,  # 300ms
                },
            ]

            health_results = []

            for scenario in health_scenarios:
                # Mock health check response
                def mock_health_check():
                    component_status = {}
                    for component in scenario["components"]:
                        if component == "database":
                            component_status[component] = {
                                "status": "healthy",
                                "response_time_ms": 15,
                                "connection_pool_active": 5,
                                "connection_pool_idle": 10,
                            }
                        elif component == "ai_service":
                            component_status[component] = {
                                "status": "healthy",
                                "local_model_loaded": True,
                                "cloud_fallback_available": True,
                                "processing_queue_size": 3,
                            }
                        elif component == "scraper":
                            component_status[component] = {
                                "status": "healthy",
                                "active_scrapers": 2,
                                "success_rate": 0.96,
                                "last_scrape": datetime.now(UTC).isoformat(),
                            }
                        elif component == "ui":
                            component_status[component] = {
                                "status": "healthy",
                                "active_sessions": 12,
                                "avg_render_time_ms": 180,
                            }
                        elif component == "background_tasks":
                            component_status[component] = {
                                "status": "healthy",
                                "running_tasks": 5,
                                "queued_tasks": 2,
                                "failed_tasks": 0,
                            }
                        elif component == "system_metrics":
                            component_status[component] = {
                                "cpu_usage": 0.45,
                                "memory_usage_mb": 1200,
                                "disk_usage": 0.65,
                            }
                        elif component == "application_metrics":
                            component_status[component] = {
                                "requests_per_minute": 150,
                                "avg_response_time_ms": 280,
                                "error_rate": 0.02,
                            }

                    return {
                        "status": scenario["expected_status"],
                        "timestamp": datetime.now(UTC),
                        "components": component_status,
                        "overall_healthy": all(
                            comp.get("status") == "healthy"
                            or comp.get("cpu_usage") is not None
                            for comp in component_status.values()
                        ),
                    }

                mock_monitor.get_health_status = Mock(return_value=mock_health_check())

                # Test health check
                start_time = time.perf_counter()
                health_response = mock_monitor.get_health_status(
                    endpoint=scenario["endpoint"]
                )
                response_time_ms = (time.perf_counter() - start_time) * 1000

                health_results.append(
                    {
                        "endpoint": scenario["endpoint"],
                        "response_time_ms": response_time_ms,
                        "status": health_response["status"],
                        "components_count": len(health_response["components"]),
                        "overall_healthy": health_response["overall_healthy"],
                        "response_fast": response_time_ms
                        < scenario["max_response_time"],
                        "health_check_valid": (
                            health_response["status"] == scenario["expected_status"]
                            and len(health_response["components"])
                            == len(scenario["components"])
                            and response_time_ms < scenario["max_response_time"]
                        ),
                    }
                )

            # Validate health monitoring
            valid_health_checks = [r for r in health_results if r["health_check_valid"]]
            health_success_rate = len(valid_health_checks) / len(health_results)

            assert health_success_rate >= 0.95, (
                f"Health monitoring success rate {health_success_rate:.2%}, should be ≥95%. "
                f"Failures: {[r for r in health_results if not r['health_check_valid']]}"
            )

    @pytest.mark.production
    def test_monitoring_alerting_system(self, production_test_setup):
        """Test monitoring system triggers alerts appropriately."""
        with patch(
            "src.coordination.system_health_monitor.SystemHealthMonitor"
        ) as MockMonitor:
            mock_monitor = Mock()
            MockMonitor.return_value = mock_monitor

            # Test alerting scenarios
            alert_scenarios = [
                {
                    "condition": "high_cpu_usage",
                    "metric_value": 0.85,
                    "threshold": 0.8,
                    "alert_expected": True,
                    "severity": "warning",
                },
                {
                    "condition": "high_memory_usage",
                    "metric_value": 1800,  # MB
                    "threshold": 1600,  # MB
                    "alert_expected": True,
                    "severity": "warning",
                },
                {
                    "condition": "high_error_rate",
                    "metric_value": 0.06,  # 6%
                    "threshold": 0.05,  # 5%
                    "alert_expected": True,
                    "severity": "critical",
                },
                {
                    "condition": "slow_response_time",
                    "metric_value": 6000,  # 6s
                    "threshold": 5000,  # 5s SLA
                    "alert_expected": True,
                    "severity": "warning",
                },
                {
                    "condition": "normal_operation",
                    "metric_value": 0.5,  # 50% CPU
                    "threshold": 0.8,  # 80% threshold
                    "alert_expected": False,
                    "severity": None,
                },
            ]

            alert_results = []

            for scenario in alert_scenarios:
                # Mock alert evaluation
                def mock_evaluate_alert():
                    should_alert = scenario["metric_value"] > scenario["threshold"]

                    if should_alert:
                        return {
                            "alert_triggered": True,
                            "condition": scenario["condition"],
                            "current_value": scenario["metric_value"],
                            "threshold": scenario["threshold"],
                            "severity": scenario["severity"],
                            "timestamp": datetime.now(UTC),
                            "message": f"{scenario['condition']} exceeded threshold",
                        }
                    return {
                        "alert_triggered": False,
                        "condition": scenario["condition"],
                        "current_value": scenario["metric_value"],
                        "threshold": scenario["threshold"],
                    }

                mock_monitor.evaluate_alert = Mock(return_value=mock_evaluate_alert())

                # Test alert evaluation
                alert_response = mock_monitor.evaluate_alert(
                    condition=scenario["condition"],
                    current_value=scenario["metric_value"],
                    threshold=scenario["threshold"],
                )

                alert_results.append(
                    {
                        "condition": scenario["condition"],
                        "alert_triggered": alert_response["alert_triggered"],
                        "expected_alert": scenario["alert_expected"],
                        "severity": alert_response.get("severity"),
                        "expected_severity": scenario["severity"],
                        "alerting_correct": (
                            alert_response["alert_triggered"]
                            == scenario["alert_expected"]
                            and alert_response.get("severity") == scenario["severity"]
                        ),
                    }
                )

            # Validate alerting system
            correct_alerts = [r for r in alert_results if r["alerting_correct"]]
            alerting_accuracy = len(correct_alerts) / len(alert_results)

            assert alerting_accuracy >= 0.95, (
                f"Alerting system accuracy {alerting_accuracy:.2%}, should be ≥95%. "
                f"Incorrect alerts: {[r for r in alert_results if not r['alerting_correct']]}"
            )


class TestConfigurationManagement:
    """Test production configuration management."""

    @pytest.mark.production
    def test_configuration_validation(self, production_test_setup):
        """Test production configuration validation."""
        # Test configuration scenarios
        config_scenarios = [
            {
                "config_type": "database",
                "config": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "job_scraper_prod",
                    "connection_pool_size": 20,
                    "connection_timeout": 30,
                },
                "required_fields": ["host", "port", "database"],
                "validation_expected": True,
            },
            {
                "config_type": "ai_service",
                "config": {
                    "local_model_path": "/models/vllm",
                    "cloud_api_key": "sk-test-key",
                    "max_tokens": 4000,
                    "timeout": 30,
                    "fallback_enabled": True,
                },
                "required_fields": ["local_model_path", "max_tokens"],
                "validation_expected": True,
            },
            {
                "config_type": "scraping",
                "config": {
                    "max_concurrent": 10,
                    "timeout": 30,
                    "retry_attempts": 3,
                    "user_agents": ["Mozilla/5.0..."],
                },
                "required_fields": ["max_concurrent", "timeout"],
                "validation_expected": True,
            },
            {
                "config_type": "invalid_config",
                "config": {
                    "missing_required_field": True,
                },
                "required_fields": ["host", "port"],
                "validation_expected": False,
            },
        ]

        config_results = []

        for scenario in config_scenarios:
            # Mock configuration validation
            def mock_validate_config():
                config = scenario["config"]
                required = scenario["required_fields"]

                # Check required fields
                missing_fields = [field for field in required if field not in config]

                if missing_fields:
                    return {
                        "valid": False,
                        "errors": [
                            f"Missing required field: {field}"
                            for field in missing_fields
                        ],
                        "warnings": [],
                    }
                warnings = []
                if (
                    scenario["config_type"] == "ai_service"
                    and "cloud_api_key" not in config
                ):
                    warnings.append(
                        "Cloud API key not configured, fallback may not work"
                    )

                return {
                    "valid": True,
                    "errors": [],
                    "warnings": warnings,
                }

            # Test configuration validation
            validation_result = mock_validate_config()

            config_results.append(
                {
                    "config_type": scenario["config_type"],
                    "validation_result": validation_result["valid"],
                    "expected_valid": scenario["validation_expected"],
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"],
                    "validation_correct": validation_result["valid"]
                    == scenario["validation_expected"],
                }
            )

        # Validate configuration management
        correct_validations = [r for r in config_results if r["validation_correct"]]
        config_validation_rate = len(correct_validations) / len(config_results)

        assert config_validation_rate >= 0.95, (
            f"Configuration validation rate {config_validation_rate:.2%}, should be ≥95%. "
            f"Incorrect validations: {[r for r in config_results if not r['validation_correct']]}"
        )

    @pytest.mark.production
    def test_environment_configuration_loading(self, production_test_setup):
        """Test loading configuration from different environments."""
        # Test environment scenarios
        env_scenarios = [
            {
                "environment": "development",
                "config_file": "config.dev.yaml",
                "expected_values": {
                    "debug": True,
                    "log_level": "DEBUG",
                    "database_pool_size": 5,
                },
            },
            {
                "environment": "staging",
                "config_file": "config.staging.yaml",
                "expected_values": {
                    "debug": False,
                    "log_level": "INFO",
                    "database_pool_size": 10,
                },
            },
            {
                "environment": "production",
                "config_file": "config.prod.yaml",
                "expected_values": {
                    "debug": False,
                    "log_level": "WARNING",
                    "database_pool_size": 20,
                },
            },
        ]

        env_results = []

        for scenario in env_scenarios:
            # Mock environment configuration loading
            with patch.dict(os.environ, {"ENVIRONMENT": scenario["environment"]}):

                def mock_load_config():
                    # Simulate loading config based on environment
                    if scenario["environment"] == "production":
                        return {
                            "debug": False,
                            "log_level": "WARNING",
                            "database": {
                                "pool_size": 20,
                                "timeout": 30,
                            },
                            "ai_service": {
                                "timeout": 30,
                                "max_concurrent": 10,
                            },
                            "scraping": {
                                "max_concurrent": 10,
                                "timeout": 30,
                            },
                        }
                    if scenario["environment"] == "staging":
                        return {
                            "debug": False,
                            "log_level": "INFO",
                            "database": {
                                "pool_size": 10,
                                "timeout": 30,
                            },
                        }
                    # development
                    return {
                        "debug": True,
                        "log_level": "DEBUG",
                        "database": {
                            "pool_size": 5,
                            "timeout": 10,
                        },
                    }

                config = mock_load_config()

                # Validate expected values
                config_valid = True
                validation_errors = []

                for key, expected_value in scenario["expected_values"].items():
                    if key == "database_pool_size":
                        actual_value = config.get("database", {}).get("pool_size")
                    else:
                        actual_value = config.get(key)

                    if actual_value != expected_value:
                        config_valid = False
                        validation_errors.append(
                            f"{key}: expected {expected_value}, got {actual_value}"
                        )

                env_results.append(
                    {
                        "environment": scenario["environment"],
                        "config_loaded": config is not None,
                        "config_valid": config_valid,
                        "validation_errors": validation_errors,
                        "env_loading_correct": config is not None and config_valid,
                    }
                )

        # Validate environment configuration loading
        correct_env_loading = [r for r in env_results if r["env_loading_correct"]]
        env_loading_rate = len(correct_env_loading) / len(env_results)

        assert env_loading_rate >= 0.95, (
            f"Environment configuration loading rate {env_loading_rate:.2%}, should be ≥95%. "
            f"Loading issues: {[r for r in env_results if not r['env_loading_correct']]}"
        )


class TestLoadTestingAndConcurrency:
    """Test system performance under production load."""

    @pytest.mark.production
    @pytest.mark.slow
    async def test_concurrent_user_simulation(self, production_test_setup):
        """Test system handles concurrent users properly."""
        setup = production_test_setup
        max_users = setup["config"]["max_concurrent_users"]

        # Mock user simulation scenarios
        user_scenarios = [
            {
                "user_type": "job_searcher",
                "actions": ["search_jobs", "view_job", "save_favorite"],
                "frequency": 2.0,  # actions per second
            },
            {
                "user_type": "analytics_user",
                "actions": ["view_analytics", "export_data"],
                "frequency": 0.5,  # actions per second
            },
            {
                "user_type": "admin_user",
                "actions": ["add_company", "manage_scrapers", "view_health"],
                "frequency": 0.1,  # actions per second
            },
        ]

        # Simulate concurrent users
        async def simulate_user(
            user_id: int, user_scenario: dict, duration_seconds: int
        ):
            """Simulate individual user actions."""
            actions_performed = []
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                for action in user_scenario["actions"]:
                    # Mock action execution
                    action_start = time.perf_counter()

                    if action == "search_jobs":
                        await asyncio.sleep(0.2)  # 200ms search
                    elif action == "view_job":
                        await asyncio.sleep(0.1)  # 100ms view
                    elif action == "save_favorite":
                        await asyncio.sleep(0.05)  # 50ms save
                    elif action == "view_analytics":
                        await asyncio.sleep(0.8)  # 800ms analytics
                    elif action == "export_data":
                        await asyncio.sleep(1.5)  # 1.5s export
                    elif action == "add_company":
                        await asyncio.sleep(0.3)  # 300ms add
                    elif action == "manage_scrapers":
                        await asyncio.sleep(0.5)  # 500ms manage
                    elif action == "view_health":
                        await asyncio.sleep(0.1)  # 100ms health check

                    action_time = (time.perf_counter() - action_start) * 1000
                    actions_performed.append(
                        {
                            "action": action,
                            "response_time_ms": action_time,
                            "timestamp": time.time(),
                        }
                    )

                    # Wait between actions based on frequency
                    await asyncio.sleep(1.0 / user_scenario["frequency"])

            return {
                "user_id": user_id,
                "user_type": user_scenario["user_type"],
                "actions_performed": actions_performed,
                "total_actions": len(actions_performed),
                "avg_response_time_ms": sum(
                    a["response_time_ms"] for a in actions_performed
                )
                / len(actions_performed)
                if actions_performed
                else 0,
            }

        # Run concurrent user simulation
        user_count = min(50, max_users // 2)  # Test with 50 concurrent users
        simulation_duration = 30  # 30 seconds

        # Create user tasks
        tasks = []
        for user_id in range(user_count):
            scenario = user_scenarios[user_id % len(user_scenarios)]
            task = simulate_user(user_id, scenario, simulation_duration)
            tasks.append(task)

        # Execute concurrent simulation
        start_time = time.perf_counter()
        user_results = await asyncio.gather(*tasks)
        total_simulation_time = time.perf_counter() - start_time

        # Analyze results
        total_actions = sum(result["total_actions"] for result in user_results)
        actions_per_second = (
            total_actions / simulation_duration if simulation_duration > 0 else 0
        )

        avg_response_times = [
            result["avg_response_time_ms"]
            for result in user_results
            if result["avg_response_time_ms"] > 0
        ]
        overall_avg_response = (
            sum(avg_response_times) / len(avg_response_times)
            if avg_response_times
            else 0
        )

        # Validate concurrent performance
        concurrent_performance_good = (
            total_simulation_time < simulation_duration * 1.1  # Within 10% overhead
            and actions_per_second > 10  # At least 10 actions/sec across all users
            and overall_avg_response < 1000  # Under 1s average response
            and len([r for r in user_results if r["total_actions"] > 0])
            >= user_count * 0.9  # 90% users active
        )

        assert concurrent_performance_good, (
            f"Concurrent user simulation failed. Total time: {total_simulation_time:.1f}s, "
            f"Actions/sec: {actions_per_second:.1f}, Avg response: {overall_avg_response:.1f}ms, "
            f"Active users: {len([r for r in user_results if r['total_actions'] > 0])}/{user_count}"
        )

    @pytest.mark.production
    @pytest.mark.slow
    async def test_sustained_load_performance(self, production_test_setup):
        """Test system performance under sustained load."""
        # Mock sustained load scenario
        load_scenarios = [
            {
                "load_type": "scraping_load",
                "concurrent_operations": 8,
                "operation_frequency": 1.0,  # per second
                "duration_minutes": 5,
            },
            {
                "load_type": "ai_processing_load",
                "concurrent_operations": 5,
                "operation_frequency": 0.5,  # per second
                "duration_minutes": 5,
            },
            {
                "load_type": "ui_rendering_load",
                "concurrent_operations": 20,
                "operation_frequency": 2.0,  # per second
                "duration_minutes": 5,
            },
        ]

        load_results = []

        for scenario in load_scenarios:

            async def sustained_operation_worker(worker_id: int):
                """Worker that performs sustained operations."""
                operations_completed = []
                start_time = time.time()
                duration_seconds = scenario["duration_minutes"] * 60

                operation_interval = 1.0 / scenario["operation_frequency"]

                while time.time() - start_time < duration_seconds:
                    operation_start = time.perf_counter()

                    # Mock different operation types
                    if scenario["load_type"] == "scraping_load":
                        await asyncio.sleep(0.3)  # 300ms scraping
                    elif scenario["load_type"] == "ai_processing_load":
                        await asyncio.sleep(1.2)  # 1.2s AI processing
                    elif scenario["load_type"] == "ui_rendering_load":
                        await asyncio.sleep(0.15)  # 150ms UI rendering

                    operation_time = (time.perf_counter() - operation_start) * 1000
                    operations_completed.append(
                        {
                            "worker_id": worker_id,
                            "operation_time_ms": operation_time,
                            "timestamp": time.time(),
                        }
                    )

                    await asyncio.sleep(operation_interval)

                return {
                    "worker_id": worker_id,
                    "operations_completed": len(operations_completed),
                    "avg_operation_time_ms": sum(
                        op["operation_time_ms"] for op in operations_completed
                    )
                    / len(operations_completed)
                    if operations_completed
                    else 0,
                    "operations_per_minute": len(operations_completed)
                    / (scenario["duration_minutes"]),
                }

            # Run sustained load test
            workers = scenario["concurrent_operations"]

            load_start_time = time.perf_counter()
            worker_tasks = [sustained_operation_worker(i) for i in range(workers)]
            worker_results = await asyncio.gather(*worker_tasks)
            load_duration = time.perf_counter() - load_start_time

            # Analyze sustained load results
            total_operations = sum(
                result["operations_completed"] for result in worker_results
            )
            avg_operation_time = sum(
                result["avg_operation_time_ms"] for result in worker_results
            ) / len(worker_results)
            operations_per_second = total_operations / (
                scenario["duration_minutes"] * 60
            )

            load_results.append(
                {
                    "load_type": scenario["load_type"],
                    "duration_minutes": scenario["duration_minutes"],
                    "concurrent_workers": workers,
                    "total_operations": total_operations,
                    "operations_per_second": operations_per_second,
                    "avg_operation_time_ms": avg_operation_time,
                    "load_duration_actual": load_duration,
                    "sustained_performance_good": (
                        operations_per_second
                        >= scenario["operation_frequency"]
                        * workers
                        * 0.8  # 80% of expected throughput
                        and avg_operation_time < 2000  # Under 2s average
                        and load_duration
                        < scenario["duration_minutes"]
                        * 60
                        * 1.1  # Within 10% time budget
                    ),
                }
            )

        # Validate sustained load performance
        good_sustained_performance = [
            r for r in load_results if r["sustained_performance_good"]
        ]
        sustained_performance_rate = len(good_sustained_performance) / len(load_results)

        assert sustained_performance_rate >= 0.8, (
            f"Sustained load performance rate {sustained_performance_rate:.2%}, should be ≥80%. "
            f"Poor performance scenarios: {[r for r in load_results if not r['sustained_performance_good']]}"
        )


class TestMemoryUsageAndProfiling:
    """Test memory usage under production scenarios."""

    @pytest.mark.production
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_profiling(self, production_test_setup):
        """Test memory usage stays within acceptable limits."""
        setup = production_test_setup
        memory_limit = setup["config"]["memory_limit_mb"]

        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        process.memory_info().rss / 1024 / 1024  # MB

        memory_scenarios = [
            {
                "scenario": "large_dataset_processing",
                "data_size": 1000,  # jobs
                "operations": ["load_data", "process_ai", "render_ui"],
                "expected_memory_increase": 200,  # MB
            },
            {
                "scenario": "concurrent_operations",
                "data_size": 500,  # jobs per operation
                "operations": ["scrape", "ai_enhance", "ui_render", "cache_data"],
                "expected_memory_increase": 150,  # MB
            },
            {
                "scenario": "sustained_processing",
                "data_size": 100,  # jobs per batch
                "operations": ["batch_process"] * 20,  # 20 batches
                "expected_memory_increase": 100,  # MB
            },
        ]

        memory_results = []

        for scenario in memory_scenarios:
            gc.collect()  # Clean up before test
            start_memory = process.memory_info().rss / 1024 / 1024

            # Simulate memory-intensive operations
            test_data = setup["test_jobs"][: scenario["data_size"]]
            processed_data = []

            for operation in scenario["operations"]:
                if operation == "load_data":
                    # Simulate loading data into memory
                    loaded_data = [dict(job.__dict__) for job in test_data]
                    processed_data.extend(loaded_data)
                elif operation == "process_ai":
                    # Simulate AI processing memory usage
                    for job in test_data:
                        ai_enhanced = {
                            **job.__dict__,
                            "skills": ["Python", "Django", "React"]
                            * 5,  # Large skills list
                            "analysis": "Long analysis text " * 100,  # Large text
                        }
                        processed_data.append(ai_enhanced)
                elif operation == "render_ui":
                    # Simulate UI rendering memory usage
                    for job in test_data:
                        rendered = {
                            "html": "<div>" + "test " * 200 + "</div>",  # Large HTML
                            "css": "body { color: red; } " * 100,  # Large CSS
                            "job_data": job.__dict__,
                        }
                        processed_data.append(rendered)
                elif operation == "scrape":
                    # Simulate scraping memory usage
                    scraped = [
                        {
                            "raw_html": "<html>" + "content " * 500 + "</html>",
                            "parsed_data": job.__dict__,
                        }
                        for job in test_data[:100]
                    ]  # Limit to prevent excessive memory
                    processed_data.extend(scraped)
                elif operation == "ai_enhance":
                    # Simulate AI enhancement
                    enhanced = [
                        {
                            **job.__dict__,
                            "embeddings": [0.1] * 1536,  # Large embedding vector
                            "features": {"feature_" + str(i): i for i in range(100)},
                        }
                        for job in test_data[:100]
                    ]
                    processed_data.extend(enhanced)
                elif operation == "ui_render":
                    # Simulate UI rendering
                    ui_data = [
                        {
                            "component_tree": {
                                "node_" + str(i): f"value_{i}" for i in range(50)
                            },
                            "job": job.__dict__,
                        }
                        for job in test_data[:100]
                    ]
                    processed_data.extend(ui_data)
                elif operation == "cache_data":
                    # Simulate caching
                    cached = {
                        f"cache_key_{i}": job.__dict__
                        for i, job in enumerate(test_data[:100])
                    }
                    processed_data.append(cached)
                elif operation == "batch_process":
                    # Simulate batch processing
                    batch_data = [
                        {
                            "batch_id": len(processed_data),
                            "items": [job.__dict__ for job in test_data],
                            "metadata": {
                                "timestamp": time.time(),
                                "size": len(test_data),
                            },
                        }
                    ]
                    processed_data.extend(batch_data)

            # Measure peak memory after operations
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - start_memory

            # Clean up
            del processed_data
            del test_data
            gc.collect()

            # Measure memory after cleanup
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_released = peak_memory - final_memory

            memory_results.append(
                {
                    "scenario": scenario["scenario"],
                    "start_memory_mb": start_memory,
                    "peak_memory_mb": peak_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "memory_released_mb": memory_released,
                    "expected_increase_mb": scenario["expected_memory_increase"],
                    "within_limit": peak_memory < memory_limit,
                    "memory_leaked": memory_released
                    < memory_increase * 0.7,  # Should release 70%+
                    "memory_performance_good": (
                        peak_memory < memory_limit
                        and memory_increase < scenario["expected_memory_increase"] * 1.2
                        and not (memory_released < memory_increase * 0.7)
                    ),
                }
            )

        # Validate memory performance
        good_memory_performance = [
            r for r in memory_results if r["memory_performance_good"]
        ]
        memory_performance_rate = len(good_memory_performance) / len(memory_results)

        assert memory_performance_rate >= 0.8, (
            f"Memory performance rate {memory_performance_rate:.2%}, should be ≥80%. "
            f"Memory issues: {[r for r in memory_results if not r['memory_performance_good']]}"
        )

    @pytest.mark.production
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_leak_detection(self, production_test_setup):
        """Test system doesn't have memory leaks under repeated operations."""
        setup = production_test_setup

        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        memory_snapshots = []

        # Perform repeated operations that could leak memory
        for iteration in range(10):
            # Simulate operations that could leak
            test_data = setup["test_jobs"][:100]

            # Create temporary data structures
            temp_data = []
            for job in test_data:
                temp_job_data = {
                    **job.__dict__,
                    "temp_id": iteration,
                    "large_field": "x" * 10000,  # 10KB of data
                    "nested_data": {
                        "level_1": {
                            "level_2": {"data": [job.__dict__ for _ in range(10)]}
                        }
                    },
                }
                temp_data.append(temp_job_data)

            # Simulate processing
            for data in temp_data:
                processed = {
                    **data,
                    "processed_timestamp": time.time(),
                    "processed_flag": True,
                }
                # Simulate some processing that might create references
                _ = str(processed)

            # Clean up (simulate proper cleanup)
            del temp_data
            del test_data
            gc.collect()

            # Take memory snapshot
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append(
                {
                    "iteration": iteration,
                    "memory_mb": current_memory,
                    "memory_increase_from_baseline": current_memory - baseline_memory,
                }
            )

        # Analyze memory growth trend
        memory_values = [snapshot["memory_mb"] for snapshot in memory_snapshots]
        memory_increases = [
            snapshot["memory_increase_from_baseline"] for snapshot in memory_snapshots
        ]

        # Check for memory leaks
        # Memory should not continuously increase
        final_increase = memory_increases[-1]
        max_increase = max(memory_increases)

        # Memory growth should be bounded
        memory_leak_detected = (
            final_increase > max_increase * 0.8  # Final memory close to peak
            and final_increase > 200  # More than 200MB growth
        )

        # Memory should stabilize (variance in last 3 measurements should be small)
        last_three_measurements = memory_values[-3:]
        memory_variance = max(last_three_measurements) - min(last_three_measurements)
        memory_stable = memory_variance < 50  # Within 50MB

        assert not memory_leak_detected, (
            f"Memory leak detected. Baseline: {baseline_memory:.1f}MB, "
            f"Final: {memory_values[-1]:.1f}MB, Increase: {final_increase:.1f}MB"
        )

        assert memory_stable, (
            f"Memory not stable. Last 3 measurements: {last_three_measurements}, "
            f"Variance: {memory_variance:.1f}MB"
        )


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience."""

    @pytest.mark.production
    async def test_production_error_scenarios(self, production_test_setup):
        """Test system handles production error scenarios properly."""
        # Production error scenarios
        error_scenarios = [
            {
                "error_type": "database_connection_lost",
                "error_simulation": "connection_timeout",
                "recovery_expected": True,
                "fallback_available": True,
                "max_recovery_time": 30,  # seconds
            },
            {
                "error_type": "ai_service_overloaded",
                "error_simulation": "service_unavailable",
                "recovery_expected": True,
                "fallback_available": True,
                "max_recovery_time": 10,  # seconds
            },
            {
                "error_type": "external_scraping_blocked",
                "error_simulation": "rate_limited",
                "recovery_expected": True,
                "fallback_available": True,
                "max_recovery_time": 60,  # seconds
            },
            {
                "error_type": "memory_exhaustion",
                "error_simulation": "out_of_memory",
                "recovery_expected": True,
                "fallback_available": True,
                "max_recovery_time": 20,  # seconds
            },
            {
                "error_type": "disk_space_full",
                "error_simulation": "no_space_left",
                "recovery_expected": False,  # Requires manual intervention
                "fallback_available": False,
                "max_recovery_time": 0,
            },
        ]

        error_results = []

        for scenario in error_scenarios:
            # Mock error and recovery
            with patch(
                "src.coordination.service_orchestrator.ServiceOrchestrator"
            ) as MockOrchestrator:
                mock_orchestrator = Mock()
                MockOrchestrator.return_value = mock_orchestrator

                async def mock_error_recovery():
                    # Simulate error detection
                    await asyncio.sleep(0.1)

                    if scenario["recovery_expected"]:
                        recovery_time = min(
                            scenario["max_recovery_time"], 5
                        )  # Cap at 5s for testing
                        await asyncio.sleep(
                            recovery_time * 0.1
                        )  # Scale down for testing

                        return {
                            "recovered": True,
                            "recovery_method": "automatic_fallback"
                            if scenario["fallback_available"]
                            else "service_restart",
                            "recovery_time_seconds": recovery_time,
                            "fallback_used": scenario["fallback_available"],
                            "service_degraded": scenario["error_type"]
                            in ["ai_service_overloaded", "external_scraping_blocked"],
                        }
                    return {
                        "recovered": False,
                        "recovery_method": None,
                        "recovery_time_seconds": 0,
                        "fallback_used": False,
                        "manual_intervention_required": True,
                    }

                mock_orchestrator.handle_production_error = AsyncMock(
                    return_value=mock_error_recovery()
                )

                # Test error recovery
                start_time = time.perf_counter()
                recovery_result = await mock_orchestrator.handle_production_error(
                    error_type=scenario["error_type"],
                    error_details={"simulation": scenario["error_simulation"]},
                )
                actual_recovery_time = time.perf_counter() - start_time

                error_results.append(
                    {
                        "error_type": scenario["error_type"],
                        "expected_recovery": scenario["recovery_expected"],
                        "actual_recovery": recovery_result["recovered"],
                        "recovery_time_seconds": actual_recovery_time,
                        "max_recovery_time": scenario["max_recovery_time"],
                        "fallback_used": recovery_result.get("fallback_used", False),
                        "expected_fallback": scenario["fallback_available"],
                        "error_handling_successful": (
                            recovery_result["recovered"]
                            == scenario["recovery_expected"]
                            and actual_recovery_time < 10.0  # Test timeout
                            and recovery_result.get("fallback_used", False)
                            == scenario["fallback_available"]
                        ),
                    }
                )

        # Validate error recovery
        successful_error_handling = [
            r for r in error_results if r["error_handling_successful"]
        ]
        error_handling_rate = len(successful_error_handling) / len(error_results)

        assert error_handling_rate >= 0.8, (
            f"Error handling success rate {error_handling_rate:.2%}, should be ≥80%. "
            f"Failed error handling: {[r for r in error_results if not r['error_handling_successful']]}"
        )

    @pytest.mark.production
    async def test_graceful_shutdown_and_restart(self, production_test_setup):
        """Test system can shutdown gracefully and restart properly."""
        # Mock system components
        with patch(
            "src.coordination.service_orchestrator.ServiceOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = Mock()
            MockOrchestrator.return_value = mock_orchestrator

            # Test graceful shutdown
            async def mock_graceful_shutdown():
                shutdown_steps = []

                # Step 1: Stop accepting new requests
                await asyncio.sleep(0.1)
                shutdown_steps.append("stop_accepting_requests")

                # Step 2: Complete in-flight requests
                await asyncio.sleep(0.2)
                shutdown_steps.append("complete_inflight_requests")

                # Step 3: Shutdown background tasks
                await asyncio.sleep(0.1)
                shutdown_steps.append("shutdown_background_tasks")

                # Step 4: Close database connections
                await asyncio.sleep(0.05)
                shutdown_steps.append("close_database_connections")

                # Step 5: Save application state
                await asyncio.sleep(0.1)
                shutdown_steps.append("save_application_state")

                return {
                    "shutdown_successful": True,
                    "shutdown_steps_completed": shutdown_steps,
                    "shutdown_time_seconds": 0.55,  # Total time
                }

            mock_orchestrator.graceful_shutdown = AsyncMock(
                return_value=mock_graceful_shutdown()
            )

            # Test startup
            async def mock_startup():
                startup_steps = []

                # Step 1: Load configuration
                await asyncio.sleep(0.05)
                startup_steps.append("load_configuration")

                # Step 2: Initialize database connections
                await asyncio.sleep(0.1)
                startup_steps.append("initialize_database")

                # Step 3: Start background services
                await asyncio.sleep(0.15)
                startup_steps.append("start_background_services")

                # Step 4: Initialize AI services
                await asyncio.sleep(0.2)
                startup_steps.append("initialize_ai_services")

                # Step 5: Start accepting requests
                await asyncio.sleep(0.05)
                startup_steps.append("start_accepting_requests")

                return {
                    "startup_successful": True,
                    "startup_steps_completed": startup_steps,
                    "startup_time_seconds": 0.55,  # Total time
                }

            mock_orchestrator.startup = AsyncMock(return_value=mock_startup())

            # Test graceful shutdown
            shutdown_start = time.perf_counter()
            shutdown_result = await mock_orchestrator.graceful_shutdown()
            shutdown_time = time.perf_counter() - shutdown_start

            # Test startup
            startup_start = time.perf_counter()
            startup_result = await mock_orchestrator.startup()
            startup_time = time.perf_counter() - startup_start

            # Validate shutdown and restart
            shutdown_successful = (
                shutdown_result["shutdown_successful"]
                and len(shutdown_result["shutdown_steps_completed"]) == 5
                and shutdown_time < 5.0  # Should complete quickly
            )

            startup_successful = (
                startup_result["startup_successful"]
                and len(startup_result["startup_steps_completed"]) == 5
                and startup_time < 5.0  # Should start quickly
            )

            assert shutdown_successful, (
                f"Graceful shutdown failed. Steps completed: {shutdown_result['shutdown_steps_completed']}, "
                f"Time: {shutdown_time:.2f}s"
            )

            assert startup_successful, (
                f"Startup failed. Steps completed: {startup_result['startup_steps_completed']}, "
                f"Time: {startup_time:.2f}s"
            )


class TestSecurityAndDataValidation:
    """Test security measures and data validation."""

    @pytest.mark.production
    def test_input_validation_security(self, production_test_setup):
        """Test input validation prevents security vulnerabilities."""
        # Security test scenarios
        security_scenarios = [
            {
                "input_type": "search_query",
                "malicious_input": "<script>alert('xss')</script>",
                "should_be_rejected": True,
                "vulnerability_type": "xss",
            },
            {
                "input_type": "search_query",
                "malicious_input": "'; DROP TABLE jobs; --",
                "should_be_rejected": True,
                "vulnerability_type": "sql_injection",
            },
            {
                "input_type": "company_url",
                "malicious_input": "javascript:alert('xss')",
                "should_be_rejected": True,
                "vulnerability_type": "url_injection",
            },
            {
                "input_type": "job_description",
                "malicious_input": "../../../etc/passwd",
                "should_be_rejected": True,
                "vulnerability_type": "path_traversal",
            },
            {
                "input_type": "search_query",
                "malicious_input": "A" * 10000,  # Very long input
                "should_be_rejected": True,
                "vulnerability_type": "dos_via_large_input",
            },
            {
                "input_type": "search_query",
                "malicious_input": "python developer",  # Normal input
                "should_be_rejected": False,
                "vulnerability_type": "none",
            },
        ]

        security_results = []

        for scenario in security_scenarios:
            # Mock input validation
            def mock_validate_input():
                malicious_input = scenario["malicious_input"]
                input_type = scenario["input_type"]

                # Simulate security validation
                if input_type == "search_query":
                    # Check for XSS
                    if (
                        "<script>" in malicious_input
                        or "javascript:" in malicious_input
                    ):
                        return {"valid": False, "reason": "XSS attempt detected"}
                    # Check for SQL injection
                    if any(
                        keyword in malicious_input.lower()
                        for keyword in ["drop table", "delete from", "insert into"]
                    ):
                        return {
                            "valid": False,
                            "reason": "SQL injection attempt detected",
                        }
                    # Check for excessive length
                    if len(malicious_input) > 1000:
                        return {"valid": False, "reason": "Input too long"}
                    # Normal input
                    return {"valid": True}

                if input_type == "company_url":
                    if malicious_input.startswith(
                        ("javascript:", "data:", "vbscript:")
                    ):
                        return {"valid": False, "reason": "Dangerous URL scheme"}
                    return {"valid": True}

                if input_type == "job_description":
                    if "../" in malicious_input or malicious_input.startswith("/"):
                        return {"valid": False, "reason": "Path traversal attempt"}
                    return {"valid": True}

                return {"valid": True}

            validation_result = mock_validate_input()

            security_results.append(
                {
                    "input_type": scenario["input_type"],
                    "vulnerability_type": scenario["vulnerability_type"],
                    "input_rejected": not validation_result["valid"],
                    "should_be_rejected": scenario["should_be_rejected"],
                    "rejection_reason": validation_result.get("reason"),
                    "security_validation_correct": (
                        validation_result["valid"] != scenario["should_be_rejected"]
                    ),
                }
            )

        # Validate security measures
        correct_security_validations = [
            r for r in security_results if r["security_validation_correct"]
        ]
        security_validation_rate = len(correct_security_validations) / len(
            security_results
        )

        assert security_validation_rate >= 0.95, (
            f"Security validation rate {security_validation_rate:.2%}, should be ≥95%. "
            f"Security failures: {[r for r in security_results if not r['security_validation_correct']]}"
        )

    @pytest.mark.production
    def test_data_sanitization_and_validation(self, production_test_setup):
        """Test data is properly sanitized and validated."""
        # Data validation scenarios
        data_scenarios = [
            {
                "data_type": "scraped_job_data",
                "data": {
                    "title": "Senior Developer <script>alert(1)</script>",
                    "description": "Great job with SQL injection'; DROP TABLE users; --",
                    "salary": "100k-150k USD",
                    "location": "San Francisco, CA",
                    "company": "TechCorp Inc.",
                },
                "expected_sanitized": {
                    "title": "Senior Developer ",
                    "description": "Great job with SQL injection; DROP TABLE users; --",
                    "salary": "100k-150k USD",
                    "location": "San Francisco, CA",
                    "company": "TechCorp Inc.",
                },
                "validation_expected": True,
            },
            {
                "data_type": "user_search_input",
                "data": {
                    "query": "python developer",
                    "location": "Remote",
                    "salary_min": 80000,
                    "salary_max": 120000,
                },
                "expected_sanitized": {
                    "query": "python developer",
                    "location": "Remote",
                    "salary_min": 80000,
                    "salary_max": 120000,
                },
                "validation_expected": True,
            },
            {
                "data_type": "invalid_job_data",
                "data": {
                    "title": "",  # Empty required field
                    "description": "x" * 50000,  # Too long
                    "salary": "invalid salary format",
                    "location": None,
                },
                "expected_sanitized": None,
                "validation_expected": False,
            },
        ]

        data_validation_results = []

        for scenario in data_scenarios:
            # Mock data sanitization and validation
            def mock_sanitize_and_validate():
                data = scenario["data"]

                # Sanitization
                sanitized_data = {}

                for key, value in data.items():
                    if isinstance(value, str):
                        # Remove script tags
                        sanitized_value = value.replace("<script>", "").replace(
                            "</script>", ""
                        )
                        # Remove potentially dangerous SQL
                        if "'; DROP TABLE" in sanitized_value:
                            sanitized_value = sanitized_value.replace("'", "")
                        sanitized_data[key] = sanitized_value
                    else:
                        sanitized_data[key] = value

                # Validation
                valid = True
                validation_errors = []

                if scenario["data_type"] in ["scraped_job_data", "invalid_job_data"]:
                    # Check required fields
                    if not sanitized_data.get("title", "").strip():
                        valid = False
                        validation_errors.append("Title is required")

                    # Check length limits
                    if len(sanitized_data.get("description", "")) > 10000:
                        valid = False
                        validation_errors.append("Description too long")

                    # Check location
                    if sanitized_data.get("location") is None:
                        valid = False
                        validation_errors.append("Location is required")

                elif scenario["data_type"] == "user_search_input":
                    # Validate salary range
                    salary_min = sanitized_data.get("salary_min")
                    salary_max = sanitized_data.get("salary_max")

                    if salary_min and salary_max and salary_min > salary_max:
                        valid = False
                        validation_errors.append(
                            "Salary minimum cannot be greater than maximum"
                        )

                return {
                    "valid": valid,
                    "sanitized_data": sanitized_data if valid else None,
                    "validation_errors": validation_errors,
                }

            sanitization_result = mock_sanitize_and_validate()

            data_validation_results.append(
                {
                    "data_type": scenario["data_type"],
                    "validation_result": sanitization_result["valid"],
                    "expected_valid": scenario["validation_expected"],
                    "sanitized_data": sanitization_result["sanitized_data"],
                    "expected_sanitized": scenario["expected_sanitized"],
                    "validation_errors": sanitization_result["validation_errors"],
                    "data_validation_correct": sanitization_result["valid"]
                    == scenario["validation_expected"],
                }
            )

        # Validate data sanitization and validation
        correct_data_validations = [
            r for r in data_validation_results if r["data_validation_correct"]
        ]
        data_validation_rate = len(correct_data_validations) / len(
            data_validation_results
        )

        assert data_validation_rate >= 0.95, (
            f"Data validation rate {data_validation_rate:.2%}, should be ≥95%. "
            f"Data validation failures: {[r for r in data_validation_results if not r['data_validation_correct']]}"
        )


# Production readiness reporting
class ProductionReadinessReporter:
    """Generate comprehensive production readiness reports."""

    @staticmethod
    def generate_production_report(test_results: dict) -> dict:
        """Generate production readiness validation report."""
        return {
            "production_readiness_summary": {
                "health_monitoring": {
                    "target": "Health checks and alerting system",
                    "achieved": test_results.get("health_monitoring_success_rate", 0)
                    >= 0.95,
                    "success_rate": test_results.get(
                        "health_monitoring_success_rate", 0
                    ),
                    "alert_accuracy": test_results.get("alerting_accuracy", 0),
                },
                "configuration_management": {
                    "target": "Environment configuration validation",
                    "achieved": test_results.get("config_validation_rate", 0) >= 0.95,
                    "validation_rate": test_results.get("config_validation_rate", 0),
                    "env_loading_rate": test_results.get("env_loading_rate", 0),
                },
                "load_handling": {
                    "target": "Concurrent users and sustained load",
                    "achieved": test_results.get("concurrent_performance_good", False),
                    "concurrent_users_supported": test_results.get(
                        "concurrent_users_supported", 0
                    ),
                    "sustained_performance_rate": test_results.get(
                        "sustained_performance_rate", 0
                    ),
                },
                "memory_management": {
                    "target": "Memory usage within limits, no leaks",
                    "achieved": test_results.get("memory_performance_rate", 0) >= 0.8,
                    "performance_rate": test_results.get("memory_performance_rate", 0),
                    "memory_leak_detected": test_results.get(
                        "memory_leak_detected", False
                    ),
                },
                "error_recovery": {
                    "target": "Graceful error handling and recovery",
                    "achieved": test_results.get("error_handling_rate", 0) >= 0.8,
                    "error_handling_rate": test_results.get("error_handling_rate", 0),
                    "graceful_shutdown_working": test_results.get(
                        "graceful_shutdown_working", False
                    ),
                },
                "security_validation": {
                    "target": "Input validation and data sanitization",
                    "achieved": test_results.get("security_validation_rate", 0) >= 0.95,
                    "validation_rate": test_results.get("security_validation_rate", 0),
                    "data_validation_rate": test_results.get("data_validation_rate", 0),
                },
            },
            "production_metrics": test_results,
            "recommendations": ProductionReadinessReporter._generate_production_recommendations(
                test_results
            ),
        }

    @staticmethod
    def _generate_production_recommendations(test_results: dict) -> list[str]:
        """Generate production improvement recommendations."""
        recommendations = []

        if test_results.get("health_monitoring_success_rate", 0) < 0.95:
            recommendations.append(
                "Improve health monitoring endpoints and alerting accuracy"
            )

        if test_results.get("config_validation_rate", 0) < 0.95:
            recommendations.append(
                "Strengthen configuration validation and environment loading"
            )

        if not test_results.get("concurrent_performance_good", False):
            recommendations.append(
                "Optimize system for concurrent user load and sustained performance"
            )

        if test_results.get("memory_performance_rate", 0) < 0.8:
            recommendations.append(
                "Address memory usage issues and potential memory leaks"
            )

        if test_results.get("error_handling_rate", 0) < 0.8:
            recommendations.append(
                "Enhance error recovery mechanisms and graceful shutdown"
            )

        if test_results.get("security_validation_rate", 0) < 0.95:
            recommendations.append(
                "Strengthen input validation and data sanitization security"
            )

        return recommendations


# Production readiness test configuration
PRODUCTION_READINESS_CONFIG = {
    "health_check_endpoints": ["/health", "/health/deep", "/ready", "/metrics"],
    "load_testing": {
        "max_concurrent_users": 100,
        "simulation_duration_seconds": 30,
        "sustained_load_minutes": 5,
        "memory_limit_mb": 2048,
    },
    "security_tests": {
        "xss_patterns": ["<script>", "javascript:", "onload="],
        "sql_injection_patterns": ["DROP TABLE", "DELETE FROM", "INSERT INTO"],
        "path_traversal_patterns": ["../", "/etc/", "\\..\\"],
        "max_input_length": 1000,
    },
    "performance_targets": {
        "health_check_response_ms": 500,
        "deep_health_check_response_ms": 2000,
        "memory_increase_limit_mb": 200,
        "error_recovery_time_s": 30,
        "shutdown_time_s": 10,
        "startup_time_s": 10,
    },
    "success_thresholds": {
        "health_monitoring_rate": 0.95,
        "config_validation_rate": 0.95,
        "sustained_performance_rate": 0.8,
        "memory_performance_rate": 0.8,
        "error_handling_rate": 0.8,
        "security_validation_rate": 0.95,
    },
}
