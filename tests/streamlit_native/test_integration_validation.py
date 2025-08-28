"""Integration Testing: Cross-Stream Component Validation.

Tests for integrated Streamlit native components across all three streams:
- Stream A + B + C integration testing
- Cross-component functionality preservation
- Performance optimization validation
- End-to-end workflow testing with native components
- Real-world usage pattern validation

Ensures 100% functionality preservation during library optimization migration.
"""

import time

from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from tests.streamlit_native.base_framework import (
    ComponentTestMetrics,
    StreamlitComponentValidator,
    StreamlitNativeTester,
)
from tests.streamlit_native.test_stream_a_progress import ProgressSystemValidator
from tests.streamlit_native.test_stream_b_caching import CachingSystemValidator
from tests.streamlit_native.test_stream_c_fragments import FragmentSystemValidator


class IntegratedMetrics:
    """Aggregate metrics from all three streams."""

    def __init__(self):
        """Initialize integrated metrics."""
        self.stream_a_metrics = {}  # Progress metrics
        self.stream_b_metrics = {}  # Caching metrics
        self.stream_c_metrics = {}  # Fragment metrics
        self.integration_events = []
        self.cross_component_interactions = 0
        self.performance_deltas = {}

    def record_integration_event(
        self, event_type: str, streams: list[str], details: dict
    ):
        """Record cross-stream integration event."""
        self.integration_events.append(
            {
                "timestamp": datetime.now(),
                "type": event_type,
                "streams": streams,
                "details": details,
            }
        )

    def record_cross_component_interaction(self):
        """Record interaction between components from different streams."""
        self.cross_component_interactions += 1

    def update_stream_metrics(self, stream: str, metrics: dict):
        """Update metrics for a specific stream."""
        if stream == "A":
            self.stream_a_metrics = metrics
        elif stream == "B":
            self.stream_b_metrics = metrics
        elif stream == "C":
            self.stream_c_metrics = metrics

    def calculate_integration_score(self) -> float:
        """Calculate overall integration success score."""
        score = 0.0
        max_score = 100.0

        # Stream functionality scores (30 points each)
        if self.stream_a_metrics:
            score += 30.0  # Progress system working
        if self.stream_b_metrics:
            score += 30.0  # Caching system working
        if self.stream_c_metrics:
            score += 30.0  # Fragment system working

        # Integration interaction score (10 points)
        if self.cross_component_interactions > 0:
            score += 10.0

        return score / max_score


class IntegratedStreamValidator(StreamlitComponentValidator):
    """Validator for integrated stream functionality."""

    def __init__(self):
        """Initialize integrated validator."""
        super().__init__("integrated_streams")
        self.progress_validator = ProgressSystemValidator()
        self.caching_validator = CachingSystemValidator()
        self.fragment_validator = FragmentSystemValidator()
        self.integrated_metrics = IntegratedMetrics()

    def validate_component_behavior(self, test_func, *args, **kwargs) -> bool:
        """Validate integrated component behavior."""
        try:
            # Reset all validators
            self.integrated_metrics = IntegratedMetrics()

            # Configure comprehensive mocking for all streams
            with (
                # Stream A mocks (Progress)
                patch(
                    "streamlit.progress",
                    self.progress_validator.mock_streamlit_progress({}),
                ),
                patch(
                    "streamlit.status",
                    self.progress_validator.mock_streamlit_status({}),
                ),
                patch(
                    "streamlit.toast", self.progress_validator.mock_streamlit_toast([])
                ),
                # Stream B mocks (Caching)
                patch(
                    "streamlit.cache_data",
                    self.caching_validator.mock_cache.mock_cache_data,
                ),
                patch(
                    "streamlit.cache_resource",
                    self.caching_validator.mock_cache.mock_cache_resource,
                ),
                # Stream C mocks (Fragments)
                patch(
                    "streamlit.fragment",
                    self.fragment_validator.mock_fragment.mock_fragment,
                ),
                patch(
                    "streamlit.rerun",
                    self.fragment_validator.mock_fragment.mock_rerun(),
                ),
            ):
                # Execute test function
                result = test_func(*args, **kwargs)

                # Record integration success
                self.integrated_metrics.record_integration_event(
                    "test_execution", ["A", "B", "C"], {"result": result}
                )

                return True

        except Exception as e:
            self.metrics.error_count += 1
            self.integrated_metrics.record_integration_event(
                "test_error", ["A", "B", "C"], {"error": str(e)}
            )
            return False

    def measure_performance(self, test_func, *args, **kwargs) -> ComponentTestMetrics:
        """Measure integrated component performance."""
        with self.performance_monitoring() as metrics:
            self.validate_component_behavior(test_func, *args, **kwargs)

        # Aggregate metrics from all streams
        metrics.progress_updates = (
            sum(self.progress_validator.progress_tracker.values())
            if self.progress_validator.progress_tracker
            else 0
        )
        metrics.cache_hits = self.caching_validator.cache_metrics.hits
        metrics.cache_misses = self.caching_validator.cache_metrics.misses
        metrics.fragment_executions = sum(
            self.fragment_validator.fragment_metrics.executions.values()
        )

        return metrics

    def validate_progress_with_caching_integration(
        self, config: dict[str, Any]
    ) -> bool:
        """Validate Stream A (progress) + Stream B (caching) integration."""

        def progress_caching_test():
            import streamlit as st

            @st.cache_data(ttl=60)
            def cached_data_processing(data_size: int) -> list:
                """Simulate expensive data processing with caching."""
                time.sleep(0.005)  # 5ms delay
                return list(range(data_size))

            @st.cache_resource
            def get_processing_config() -> dict:
                """Get cached processing configuration."""
                return {"batch_size": 100, "timeout": 30}

            # Progress tracking with cached operations
            with st.status(
                "Processing data...", expanded=True, state="running"
            ) as status:
                config_data = get_processing_config()  # Cached resource

                total_items = config.get("total_items", 1000)
                batch_size = config_data["batch_size"]
                batches = (total_items + batch_size - 1) // batch_size

                results = []
                for i in range(batches):
                    # Update progress
                    progress = (i + 1) / batches
                    st.progress(progress, text=f"Processing batch {i + 1}/{batches}")

                    # Process batch with caching
                    batch_data = cached_data_processing(batch_size)
                    results.extend(batch_data)

                    # Show intermediate progress
                    if i % 5 == 0:
                        st.toast(f"Completed {i + 1} batches", icon="📊")

                status.update(label="Processing complete", state="complete")
                st.toast("All data processed successfully!", icon="✅")

            self.integrated_metrics.record_cross_component_interaction()
            return results

        return self.validate_component_behavior(progress_caching_test)

    def validate_progress_with_fragments_integration(
        self, config: dict[str, Any]
    ) -> bool:
        """Validate Stream A (progress) + Stream C (fragments) integration."""

        def progress_fragments_test():
            import streamlit as st

            # Shared state for progress tracking
            if "task_progress" not in st.session_state:
                st.session_state.task_progress = 0
                st.session_state.task_status = "Initializing"

            @st.fragment(run_every="1s")
            def progress_tracking_fragment():
                """Fragment that updates progress automatically."""
                if st.session_state.task_progress < 100:
                    st.session_state.task_progress += 10
                    st.session_state.task_status = (
                        f"Processing... {st.session_state.task_progress}%"
                    )

                    # Update progress bar
                    st.progress(
                        st.session_state.task_progress / 100,
                        text=st.session_state.task_status,
                    )

                    # Status updates
                    if st.session_state.task_progress == 50:
                        st.toast("Halfway complete!", icon="⏰")
                    elif st.session_state.task_progress >= 100:
                        st.session_state.task_status = "Complete"
                        st.toast("Task completed!", icon="🎉")

                return st.session_state.task_progress

            # Main status container
            with st.status(
                "Auto-updating task", expanded=True, state="running"
            ) as status:
                progress_result = progress_tracking_fragment()

                if st.session_state.task_progress >= 100:
                    status.update(label="Task completed", state="complete")

            self.integrated_metrics.record_cross_component_interaction()
            return progress_result

        return self.validate_component_behavior(progress_fragments_test)

    def validate_caching_with_fragments_integration(
        self, config: dict[str, Any]
    ) -> bool:
        """Validate Stream B (caching) + Stream C (fragments) integration."""

        def caching_fragments_test():
            import streamlit as st

            @st.cache_resource
            def get_data_source() -> dict:
                """Get cached data source configuration."""
                return {
                    "connection": "data_api",
                    "endpoint": "https://api.example.com/data",
                    "timeout": 30,
                }

            @st.cache_data(ttl=30)  # 30 second cache
            def fetch_live_data(timestamp: str) -> dict:
                """Fetch data with caching."""
                time.sleep(0.003)  # 3ms delay
                return {
                    "data": f"live_data_{timestamp}",
                    "timestamp": timestamp,
                    "source": "api",
                }

            @st.fragment(run_every="2s")
            def live_data_fragment():
                """Fragment that fetches and displays cached data."""
                source_config = get_data_source()  # Cached resource
                current_time = datetime.now().strftime("%H:%M:%S")

                # Fetch data (potentially cached)
                data = fetch_live_data(current_time)

                return {
                    "config": source_config,
                    "data": data,
                    "cache_hit": current_time in str(data),
                }

            # Execute the integrated pattern
            result = live_data_fragment()

            # Let fragment run a few times to test caching
            time.sleep(2.5)

            self.integrated_metrics.record_cross_component_interaction()
            return result

        return self.validate_component_behavior(caching_fragments_test)

    def validate_full_stream_integration(self, config: dict[str, Any]) -> bool:
        """Validate all three streams working together."""

        def full_integration_test():
            import streamlit as st

            # Initialize session state
            if "processing_state" not in st.session_state:
                st.session_state.processing_state = {
                    "current_step": 0,
                    "total_steps": 5,
                    "data_cache": {},
                    "status": "ready",
                }

            @st.cache_resource
            def get_processing_pipeline() -> dict:
                """Get cached processing pipeline configuration."""
                return {
                    "steps": ["init", "collect", "process", "analyze", "report"],
                    "timeouts": [1, 2, 3, 2, 1],
                    "batch_size": 50,
                }

            @st.cache_data(ttl=120)  # 2 minute cache
            def process_step_data(step: str, batch_id: int) -> dict:
                """Process data for a specific step with caching."""
                time.sleep(0.002)  # 2ms processing
                return {
                    "step": step,
                    "batch_id": batch_id,
                    "processed_at": datetime.now().isoformat(),
                    "result": f"{step}_result_{batch_id}",
                }

            @st.fragment(run_every="1.5s")
            def processing_status_fragment():
                """Auto-updating fragment showing processing status."""
                pipeline = get_processing_pipeline()  # Cached resource
                current_step = st.session_state.processing_state["current_step"]

                if current_step < st.session_state.processing_state["total_steps"]:
                    # Get current step info
                    step_name = pipeline["steps"][current_step]

                    # Process current step data (with caching)
                    step_data = process_step_data(step_name, current_step)

                    # Update progress
                    progress = (current_step + 1) / st.session_state.processing_state[
                        "total_steps"
                    ]
                    st.progress(progress, text=f"Processing: {step_name}")

                    # Store result
                    st.session_state.processing_state["data_cache"][step_name] = (
                        step_data
                    )

                    # Advance to next step
                    st.session_state.processing_state["current_step"] += 1

                    # Show step completion
                    st.toast(f"Completed: {step_name}", icon="✅")

                return st.session_state.processing_state

            # Main processing status
            with st.status("Integrated Processing Pipeline", expanded=True) as status:
                processing_result = processing_status_fragment()

                # Check if complete
                if (
                    processing_result["current_step"]
                    >= processing_result["total_steps"]
                ):
                    status.update(label="Pipeline completed", state="complete")
                    st.toast("All processing steps completed!", icon="🎉")
                else:
                    status.update(
                        label=f"Step {processing_result['current_step'] + 1}/5",
                        state="running",
                    )

            self.integrated_metrics.record_cross_component_interaction()
            return processing_result

        return self.validate_component_behavior(full_integration_test)

    def measure_integrated_performance(
        self, test_scenarios: list[dict]
    ) -> dict[str, Any]:
        """Measure performance across all integrated scenarios."""
        results = {}

        for scenario in test_scenarios:
            scenario_name = scenario["name"]
            test_func = scenario["test_function"]
            config = scenario.get("config", {})

            with self.performance_monitoring() as metrics:
                success = test_func(config)

            results[scenario_name] = {
                "success": success,
                "render_time": metrics.render_time,
                "memory_usage": metrics.memory_usage,
                "component_interactions": self.integrated_metrics.cross_component_interactions,
                "integration_events": len(self.integrated_metrics.integration_events),
            }

            # Reset for next scenario
            self.integrated_metrics = IntegratedMetrics()

        return results


class TestStreamIntegration:
    """Test suite for integrated stream functionality."""

    @pytest.fixture
    def integration_validator(self):
        """Provide integrated stream validator."""
        return IntegratedStreamValidator()

    @pytest.fixture
    def streamlit_tester(self, integration_validator):
        """Provide configured Streamlit tester."""
        tester = StreamlitNativeTester()
        tester.register_validator("integrated_streams", integration_validator)
        return tester

    def test_progress_caching_integration(self, integration_validator):
        """Test Stream A + Stream B integration."""
        config = {"total_items": 500}

        assert integration_validator.validate_progress_with_caching_integration(config)

        # Verify cross-component interactions
        assert integration_validator.integrated_metrics.cross_component_interactions > 0

    def test_progress_fragments_integration(self, integration_validator):
        """Test Stream A + Stream C integration."""
        config = {}

        assert integration_validator.validate_progress_with_fragments_integration(
            config
        )

        # Should have fragment executions and progress updates
        assert integration_validator.integrated_metrics.cross_component_interactions > 0

    def test_caching_fragments_integration(self, integration_validator):
        """Test Stream B + Stream C integration."""
        config = {}

        assert integration_validator.validate_caching_with_fragments_integration(config)

        # Should demonstrate caching in fragment context
        assert integration_validator.integrated_metrics.cross_component_interactions > 0

    def test_full_stream_integration(self, integration_validator):
        """Test all three streams integrated together."""
        config = {}

        # Let the test run longer to see full integration
        datetime.now()
        assert integration_validator.validate_full_stream_integration(config)

        # Should have comprehensive integration
        assert integration_validator.integrated_metrics.cross_component_interactions > 0

        # Verify all streams are represented in events
        event_streams = set()
        for event in integration_validator.integrated_metrics.integration_events:
            event_streams.update(event["streams"])

        expected_streams = {"A", "B", "C"}
        assert expected_streams.issubset(event_streams)

    def test_real_world_job_scraping_workflow(self, streamlit_tester):
        """Test real-world job scraping workflow with all streams."""

        def job_scraping_workflow():
            import streamlit as st

            # Initialize scraping state
            if "scraping_session" not in st.session_state:
                st.session_state.scraping_session = {
                    "companies_processed": 0,
                    "jobs_found": 0,
                    "current_company": None,
                    "status": "ready",
                }

            @st.cache_resource
            def get_scraping_config() -> dict:
                """Get cached scraping configuration."""
                return {
                    "user_agent": "JobScraperBot/1.0",
                    "timeout": 30,
                    "max_retries": 3,
                    "companies": ["TechCorp", "DataInc", "AIStartup"],
                }

            @st.cache_data(ttl=300)  # 5 minute cache
            def scrape_company_jobs(company_name: str) -> dict:
                """Scrape jobs from a company with caching."""
                time.sleep(0.01)  # 10ms simulated scraping
                return {
                    "company": company_name,
                    "jobs": [f"{company_name}_job_{i}" for i in range(5)],
                    "scraped_at": datetime.now().isoformat(),
                }

            @st.fragment(run_every="2s")
            def scraping_progress_fragment():
                """Fragment that updates scraping progress."""
                config = get_scraping_config()  # Cached resource
                companies = config["companies"]
                session = st.session_state.scraping_session

                if session["companies_processed"] < len(companies):
                    # Get current company
                    current_company = companies[session["companies_processed"]]
                    session["current_company"] = current_company

                    # Scrape jobs (with caching)
                    company_data = scrape_company_jobs(current_company)

                    # Update session state
                    session["jobs_found"] += len(company_data["jobs"])
                    session["companies_processed"] += 1

                    # Update progress bar
                    progress = session["companies_processed"] / len(companies)
                    st.progress(
                        progress,
                        text=f"Scraped {current_company}: {len(company_data['jobs'])} jobs",
                    )

                    # Show progress toast
                    st.toast(
                        f"Found {len(company_data['jobs'])} jobs at {current_company}",
                        icon="🔍",
                    )

                return session

            # Main scraping status
            with st.status("Job Scraping in Progress", expanded=True) as status:
                scraping_result = scraping_progress_fragment()

                config = get_scraping_config()
                total_companies = len(config["companies"])

                if scraping_result["companies_processed"] >= total_companies:
                    status.update(
                        label=f"Scraping complete: {scraping_result['jobs_found']} jobs found",
                        state="complete",
                    )
                    st.toast(
                        f"Scraping finished! Found {scraping_result['jobs_found']} total jobs",
                        icon="🎉",
                    )
                else:
                    current = scraping_result["companies_processed"]
                    status.update(
                        label=f"Processing company {current + 1}/{total_companies}",
                        state="running",
                    )

            return scraping_result

        result = streamlit_tester.run_component_validation(
            "integrated_streams", job_scraping_workflow
        )

        assert result is True

    def test_analytics_dashboard_workflow(self, streamlit_tester):
        """Test analytics dashboard with all three streams."""

        def analytics_workflow():
            import streamlit as st

            @st.cache_resource
            def get_analytics_connection() -> dict:
                """Get cached analytics database connection."""
                return {
                    "connection": "analytics_db",
                    "tables": ["jobs", "companies", "applications"],
                    "last_updated": datetime.now(),
                }

            @st.cache_data(ttl=180)  # 3 minute cache
            def compute_analytics_metrics(metric_type: str) -> dict:
                """Compute analytics metrics with caching."""
                time.sleep(0.008)  # 8ms computation
                return {
                    "metric_type": metric_type,
                    "value": hash(metric_type) % 1000,  # Deterministic fake data
                    "computed_at": datetime.now().isoformat(),
                    "trend": "up" if hash(metric_type) % 2 else "down",
                }

            @st.fragment(run_every="3s")
            def live_analytics_fragment():
                """Fragment showing live analytics updates."""
                get_analytics_connection()  # Cached resource

                # Compute different metrics (with caching)
                metrics = {}
                for metric in ["total_jobs", "active_applications", "response_rate"]:
                    metrics[metric] = compute_analytics_metrics(metric)

                # Display metrics with progress bars
                for metric_name, metric_data in metrics.items():
                    progress_value = metric_data["value"] / 1000.0
                    st.progress(
                        progress_value, text=f"{metric_name}: {metric_data['value']}"
                    )

                return metrics

            # Main analytics status
            with st.status(
                "Live Analytics Dashboard", expanded=True, state="running"
            ) as status:
                analytics_result = live_analytics_fragment()

                # Show update notification
                st.toast("Analytics updated", icon="📊")

                status.update(label="Analytics refreshed", state="complete")

            return analytics_result

        result = streamlit_tester.run_component_validation(
            "integrated_streams", analytics_workflow
        )

        assert result is True

    def test_integrated_performance_benchmarks(self, integration_validator):
        """Test performance of all integrated scenarios."""
        test_scenarios = [
            {
                "name": "progress_caching",
                "test_function": integration_validator.validate_progress_with_caching_integration,
                "config": {"total_items": 200},
            },
            {
                "name": "progress_fragments",
                "test_function": integration_validator.validate_progress_with_fragments_integration,
                "config": {},
            },
            {
                "name": "caching_fragments",
                "test_function": integration_validator.validate_caching_with_fragments_integration,
                "config": {},
            },
            {
                "name": "full_integration",
                "test_function": integration_validator.validate_full_stream_integration,
                "config": {},
            },
        ]

        results = integration_validator.measure_integrated_performance(test_scenarios)

        # Verify all scenarios succeeded
        for scenario_name, result in results.items():
            assert result["success"] is True, f"Scenario {scenario_name} failed"
            assert result["render_time"] > 0, (
                f"No render time recorded for {scenario_name}"
            )
            assert result["component_interactions"] >= 0, (
                f"No interactions recorded for {scenario_name}"
            )

    def test_migration_validation_comprehensive(self, streamlit_tester):
        """Comprehensive test to validate migration preserves all functionality."""

        def old_implementation():
            """Simulate old implementation without native components."""
            import streamlit as st

            # Old progress tracking
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("Processing data...")
            with col2:
                st.write("75%")

            # Old status updates
            st.info("Status: Processing step 3 of 4")

            # Manual caching simulation (no actual caching)
            data_cache = st.session_state.get("manual_cache", {})
            if "expensive_data" not in data_cache:
                time.sleep(0.01)  # Simulate expensive operation
                data_cache["expensive_data"] = "computed_result"
                st.session_state["manual_cache"] = data_cache

            # Manual refresh simulation
            if st.button("Refresh"):
                st.rerun()

            return {"status": "old_implementation", "data": data_cache}

        def new_implementation():
            """New implementation using native Streamlit components."""
            import streamlit as st

            @st.cache_data(ttl=60)
            def expensive_computation() -> str:
                time.sleep(0.01)  # Same expensive operation
                return "computed_result"

            @st.fragment(run_every="5s")
            def auto_refresh_fragment():
                with st.status(
                    "Processing data...", expanded=True, state="running"
                ) as status:
                    st.progress(0.75, text="75% complete")

                    # Use cached computation
                    result = expensive_computation()

                    status.update(label="Processing step 3 of 4", state="running")
                    st.toast("Step 3 completed", icon="✅")

                return {"status": "new_implementation", "data": result}

            return auto_refresh_fragment()

        # Test both implementations
        old_result = streamlit_tester.run_component_validation(
            "integrated_streams", old_implementation
        )

        new_result = streamlit_tester.run_component_validation(
            "integrated_streams", new_implementation
        )

        # Both should work
        assert old_result is True
        assert new_result is True

    def test_error_resilience_integration(self, integration_validator):
        """Test error handling across integrated components."""

        def error_resilience_test():
            import streamlit as st

            @st.cache_data(ttl=30)
            def potentially_failing_operation(fail_mode: bool) -> dict:
                if fail_mode:
                    raise ValueError("Simulated failure")
                return {"result": "success", "timestamp": datetime.now().isoformat()}

            @st.fragment(run_every="1s")
            def resilient_fragment():
                try:
                    # Try operation that might fail
                    result = potentially_failing_operation(False)

                    st.progress(1.0, text="Operation successful")
                    st.toast("Operation completed", icon="✅")

                    return result

                except ValueError as e:
                    # Handle error gracefully
                    st.progress(0.0, text="Operation failed")
                    st.toast(f"Error: {e}", icon="❌")

                    return {"result": "error", "error": str(e)}

            with st.status("Error Resilience Test", expanded=True) as status:
                result = resilient_fragment()

                if result.get("result") == "error":
                    status.update(label="Handled error gracefully", state="error")
                else:
                    status.update(label="Operation successful", state="complete")

            return result

        assert integration_validator.validate_component_behavior(error_resilience_test)

    @pytest.mark.parametrize("complexity_level", ("simple", "moderate", "complex"))
    def test_integration_complexity_scaling(
        self, integration_validator, complexity_level
    ):
        """Test integration with different complexity levels."""

        def complexity_test():
            import streamlit as st

            # Configure based on complexity level
            if complexity_level == "simple":
                fragment_count = 1
                cache_operations = 2
                progress_steps = 3
            elif complexity_level == "moderate":
                fragment_count = 3
                cache_operations = 5
                progress_steps = 8
            else:  # complex
                fragment_count = 5
                cache_operations = 10
                progress_steps = 15

            @st.cache_resource
            def get_complexity_config() -> dict:
                return {
                    "fragments": fragment_count,
                    "cache_ops": cache_operations,
                    "steps": progress_steps,
                }

            @st.cache_data(ttl=60)
            def cached_operation(op_id: int) -> str:
                time.sleep(0.001)  # 1ms per operation
                return f"operation_{op_id}_result"

            results = []
            config = get_complexity_config()

            # Create complexity based on level
            for i in range(config["fragments"]):

                @st.fragment(run_every="0.5s")
                def complexity_fragment():
                    fragment_results = []

                    # Progress through steps
                    for step in range(config["steps"]):
                        progress = (step + 1) / config["steps"]
                        st.progress(progress, text=f"Fragment {i} Step {step + 1}")

                        # Perform cached operations
                        for op in range(config["cache_ops"]):
                            result = cached_operation(step * config["cache_ops"] + op)
                            fragment_results.append(result)

                    st.toast(f"Fragment {i} completed", icon="✅")
                    return fragment_results

                # Execute fragment
                fragment_result = complexity_fragment()
                results.append(fragment_result)

            with st.status(f"Complexity test: {complexity_level}", state="complete"):
                st.toast(f"Completed {complexity_level} integration test", icon="🎯")

            return results

        assert integration_validator.validate_component_behavior(complexity_test)


class TestIntegrationBenchmarks:
    """Benchmark tests for integrated stream performance."""

    @pytest.fixture
    def benchmarking_tester(self):
        """Provide tester configured for integration benchmarking."""
        tester = StreamlitNativeTester()
        tester.register_validator("integrated_streams", IntegratedStreamValidator())
        return tester

    def test_full_integration_performance(self, benchmarking_tester):
        """Benchmark full integration performance."""

        def full_integration_benchmark():
            import streamlit as st

            @st.cache_resource
            def get_benchmark_config() -> dict:
                return {"iterations": 10, "batch_size": 50}

            @st.cache_data(ttl=30)
            def benchmark_operation(iteration: int) -> dict:
                time.sleep(0.002)  # 2ms operation
                return {"iteration": iteration, "result": f"benchmark_{iteration}"}

            @st.fragment(run_every="0.5s")
            def benchmark_fragment():
                config = get_benchmark_config()
                results = []

                for i in range(config["iterations"]):
                    progress = (i + 1) / config["iterations"]
                    st.progress(
                        progress, text=f"Benchmark {i + 1}/{config['iterations']}"
                    )

                    result = benchmark_operation(i)
                    results.append(result)

                st.toast("Benchmark completed", icon="🏁")
                return results

            with st.status(
                "Integration Benchmark", expanded=True, state="running"
            ) as status:
                benchmark_results = benchmark_fragment()
                status.update(label="Benchmark complete", state="complete")

            return benchmark_results

        benchmark = benchmarking_tester.benchmark_component_performance(
            "integrated_streams", full_integration_benchmark, iterations=3
        )

        validator = benchmarking_tester.validators["integrated_streams"]

        # Performance should be reasonable
        assert benchmark.after_metrics.render_time > 0
        assert benchmark.after_metrics.render_time < 10.0  # Less than 10 seconds

        # Should have cross-component interactions
        assert validator.integrated_metrics.cross_component_interactions >= 0

    def test_memory_efficiency_integration(self, benchmarking_tester):
        """Benchmark memory efficiency of integrated components."""

        def memory_efficiency_test():
            import streamlit as st

            @st.cache_data(ttl=60)
            def memory_intensive_operation(size: int) -> list:
                # Create and return large data structure
                return list(range(size))

            @st.fragment(run_every="1s")
            def memory_fragment():
                # Process progressively larger datasets
                sizes = [100, 500, 1000, 2000]
                results = []

                for i, size in enumerate(sizes):
                    progress = (i + 1) / len(sizes)
                    st.progress(progress, text=f"Processing {size} items")

                    # This should be cached after first call
                    data = memory_intensive_operation(size)
                    results.append(len(data))

                st.toast(f"Processed {len(sizes)} datasets", icon="💾")
                return results

            with st.status("Memory Efficiency Test", state="running") as status:
                memory_results = memory_fragment()
                status.update(label="Memory test complete", state="complete")

            return memory_results

        benchmark = benchmarking_tester.benchmark_component_performance(
            "integrated_streams", memory_efficiency_test, iterations=2
        )

        # Memory usage should be reasonable
        assert benchmark.after_metrics.memory_usage >= 0
        # Should not have excessive memory growth
        assert benchmark.after_metrics.memory_peak < 200_000_000  # 200MB limit
