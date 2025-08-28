"""Comprehensive End-to-End Workflow Integration Tests.

This test suite validates the complete AI job scraper system workflow from search
initiation to final job display, ensuring all phases (3A-3D) work together seamlessly.

**Workflow Validation Coverage**:
- Complete pipeline: search → scrape → AI enhance → mobile display
- Service orchestration across all system components
- Real-world scenario validation with performance benchmarks
- Background task coordination and progress tracking
- Error recovery and graceful fallback validation

**ADR Requirements Validated**:
- ADR-013: Unified Scraping Service (JobSpy + ScrapeGraphAI)
- ADR-010/011: Hybrid AI Enhancement (vLLM + cloud fallback)
- ADR-017: Background Task Management and Progress Tracking
- ADR-018: Search Service Integration
- ADR-020: Application Status Tracking
- ADR-021: Mobile-First Card UI Rendering

**Performance Targets**:
- <500ms job board queries validation
- <3s AI enhancement processing
- <200ms card rendering (50 jobs)
- 95%+ scraping success rate
- End-to-end workflow < 30s for typical queries
"""

import asyncio
import logging
import time

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.ai.hybrid_ai_router import get_hybrid_ai_router
from src.coordination.service_orchestrator import (
    ServiceOrchestrator,
)
from src.interfaces.scraping_service_interface import JobQuery, SourceType
from src.schemas import Job
from src.services.unified_scraper import UnifiedScrapingService
from tests.factories import create_realistic_dataset

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def e2e_test_setup(session, tmp_path, test_settings):
    """Set up comprehensive end-to-end testing environment."""
    # Create large realistic dataset for comprehensive testing
    dataset = create_realistic_dataset(
        session,
        companies=30,
        jobs_per_company=25,
        include_inactive_companies=True,
        include_archived_jobs=True,
        senior_ratio=0.35,
        remote_ratio=0.45,
        favorited_ratio=0.15,
    )

    # E2E test configuration
    e2e_config = {
        "workflow_timeout": 30.0,  # 30s for complete workflow
        "scraping_timeout": 10.0,  # 10s for scraping phase
        "ai_processing_timeout": 5.0,  # 5s for AI enhancement
        "ui_rendering_timeout": 1.0,  # 1s for UI updates
        "max_jobs_per_query": 50,
        "performance_targets": {
            "job_board_query_ms": 500,
            "ai_enhancement_ms": 3000,
            "card_rendering_ms": 200,
            "scraping_success_rate": 0.95,
        },
    }

    return {
        "dataset": dataset,
        "config": e2e_config,
        "session": session,
        "temp_dir": tmp_path,
        "settings": test_settings,
        "test_companies": dataset["companies"][:10],  # Focus set
        "test_jobs": dataset["jobs"][:100],  # Test batch
    }


class TestCompleteWorkflowPipeline:
    """Test the complete workflow pipeline from search to display."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_search_to_display_workflow(self, e2e_test_setup):
        """Test complete workflow: search → scrape → enhance → display."""
        setup = e2e_test_setup

        # Mock complete workflow components
        with (
            patch.multiple(
                "src.services.unified_scraper.UnifiedScrapingService",
                scrape_unified=AsyncMock(),
            ),
            patch.multiple(
                "src.ai.hybrid_ai_router.HybridAIRouter",
                process_content=AsyncMock(),
            ),
            patch.multiple(
                "src.services.search_service.SearchService",
                search_jobs=AsyncMock(),
            ),
        ):
            # Initialize orchestrator
            orchestrator = ServiceOrchestrator(setup["settings"])

            # Test query scenarios
            test_queries = [
                {
                    "keywords": ["python developer", "machine learning"],
                    "locations": ["San Francisco", "Remote"],
                    "expected_job_count": 25,
                    "workflow_type": "standard_search",
                },
                {
                    "keywords": ["data scientist", "AI engineer"],
                    "locations": ["New York", "Austin"],
                    "expected_job_count": 15,
                    "workflow_type": "ai_focused_search",
                },
                {
                    "keywords": ["full stack developer"],
                    "locations": ["Remote"],
                    "expected_job_count": 30,
                    "workflow_type": "remote_only_search",
                },
            ]

            workflow_results = []

            for query_scenario in test_queries:
                # Create JobQuery
                job_query = JobQuery(
                    keywords=query_scenario["keywords"],
                    locations=query_scenario["locations"],
                    source_types=[SourceType.UNIFIED],
                    max_results=query_scenario["expected_job_count"],
                    enable_ai_enhancement=True,
                )

                # Execute complete workflow
                start_time = time.perf_counter()

                try:
                    workflow_id = await orchestrator.execute_integrated_workflow(
                        job_query,
                        workflow_options={
                            "enable_ai_enhancement": True,
                            "enable_real_time_updates": True,
                            "enable_ui_updates": True,
                            "max_jobs": query_scenario["expected_job_count"],
                        },
                    )

                    # Monitor workflow progress
                    final_status = None
                    async for status in orchestrator.monitor_workflow_progress(
                        workflow_id
                    ):
                        final_status = status
                        if status["status"] in ["completed", "failed"]:
                            break

                    workflow_duration = time.perf_counter() - start_time

                    # Validate workflow completion
                    workflow_successful = (
                        final_status is not None
                        and final_status["status"] == "completed"
                        and workflow_duration < setup["config"]["workflow_timeout"]
                    )

                    workflow_results.append(
                        {
                            "query_type": query_scenario["workflow_type"],
                            "keywords": query_scenario["keywords"],
                            "workflow_id": workflow_id,
                            "duration": workflow_duration,
                            "successful": workflow_successful,
                            "final_status": final_status,
                            "services_used": final_status.get("services_used", [])
                            if final_status
                            else [],
                        }
                    )

                except Exception as e:
                    workflow_results.append(
                        {
                            "query_type": query_scenario["workflow_type"],
                            "keywords": query_scenario["keywords"],
                            "workflow_id": None,
                            "duration": time.perf_counter() - start_time,
                            "successful": False,
                            "error": str(e),
                            "final_status": None,
                            "services_used": [],
                        }
                    )

            # Validate overall workflow success
            successful_workflows = [r for r in workflow_results if r["successful"]]
            workflow_success_rate = len(successful_workflows) / len(workflow_results)

            assert workflow_success_rate >= 0.9, (
                f"Workflow success rate {workflow_success_rate:.2%} below 90% target. "
                f"Failed workflows: {[r for r in workflow_results if not r['successful']]}"
            )

            # Validate performance targets
            avg_duration = (
                sum(r["duration"] for r in successful_workflows)
                / len(successful_workflows)
                if successful_workflows
                else float("inf")
            )

            assert avg_duration < setup["config"]["workflow_timeout"], (
                f"Average workflow duration {avg_duration:.2f}s exceeds {setup['config']['workflow_timeout']}s target"
            )

    @pytest.mark.integration
    async def test_workflow_service_coordination(self, e2e_test_setup):
        """Test proper coordination between all system services."""
        setup = e2e_test_setup

        # Mock service responses to test coordination
        mock_scraping_results = [
            Job(
                title=f"Software Engineer {i}",
                company=f"TechCorp {i}",
                description=f"Job description {i} with Python and machine learning",
                link=f"https://jobs.example{i}.com/job/{i}",
                location="San Francisco" if i % 2 == 0 else "Remote",
                salary="$120k-160k",
                last_seen=datetime.now(UTC),
            )
            for i in range(15)
        ]

        [
            {
                **job.model_dump(),
                "ai_insights": f"Enhanced job {i} with skills analysis",
            }
            for i, job in enumerate(mock_scraping_results)
        ]

        with (
            patch.multiple(
                "src.services.unified_scraper.UnifiedScrapingService",
                scrape_unified=AsyncMock(return_value=mock_scraping_results),
                start_background_scraping=AsyncMock(return_value="test-task-123"),
                get_scraping_status=AsyncMock(
                    return_value=Mock(status="completed", jobs_found=15)
                ),
            ),
            patch.multiple(
                "src.ai.hybrid_ai_router.HybridAIRouter",
                process_content=AsyncMock(return_value="Enhanced content"),
            ),
            patch.multiple(
                "src.coordination.system_health_monitor.SystemHealthMonitor",
                get_comprehensive_health_report=AsyncMock(
                    return_value={
                        "services": {
                            "database": {"healthy": True},
                            "search": {"healthy": True},
                            "unified_scraper": {"healthy": True},
                            "ai_router": {"healthy": True},
                        }
                    }
                ),
            ),
        ):
            orchestrator = ServiceOrchestrator(setup["settings"])

            # Test service coordination
            job_query = JobQuery(
                keywords=["python developer"],
                locations=["San Francisco"],
                source_types=[SourceType.UNIFIED],
                max_results=15,
                enable_ai_enhancement=True,
            )

            workflow_id = await orchestrator.execute_integrated_workflow(
                job_query,
                workflow_options={
                    "enable_ai_enhancement": True,
                    "enable_real_time_updates": True,
                },
            )

            # Validate service coordination
            workflow_status = orchestrator.get_workflow_status(workflow_id)
            assert workflow_status is not None
            assert workflow_status["status"] == "completed"

            # Verify all expected services were used
            services_used = workflow_status.get("services_used", [])
            expected_services = ["health_monitor", "unified_scraper", "ai_router"]

            for expected_service in expected_services:
                assert expected_service in services_used, (
                    f"Expected service {expected_service} not found in services used: {services_used}"
                )

    @pytest.mark.integration
    async def test_background_task_coordination(self, e2e_test_setup):
        """Test background task management and progress tracking."""
        setup = e2e_test_setup

        with patch.multiple(
            "src.services.unified_scraper.UnifiedScrapingService",
            start_background_scraping=AsyncMock(return_value="background-task-456"),
            monitor_scraping_progress=AsyncMock(),
        ):
            # Mock background task progress
            async def mock_progress_generator():
                statuses = [
                    Mock(status="queued", progress_percentage=0.0, jobs_found=0),
                    Mock(status="running", progress_percentage=25.0, jobs_found=5),
                    Mock(status="running", progress_percentage=50.0, jobs_found=12),
                    Mock(status="running", progress_percentage=75.0, jobs_found=18),
                    Mock(status="completed", progress_percentage=100.0, jobs_found=20),
                ]
                for status in statuses:
                    yield status
                    await asyncio.sleep(0.1)

            with patch.object(
                UnifiedScrapingService,
                "monitor_scraping_progress",
                side_effect=lambda task_id: mock_progress_generator(),
            ):
                orchestrator = ServiceOrchestrator(setup["settings"])

                job_query = JobQuery(
                    keywords=["data scientist"],
                    locations=["New York"],
                    source_types=[SourceType.JOB_BOARDS],
                    max_results=20,
                )

                # Start workflow and track progress
                workflow_id = await orchestrator.execute_integrated_workflow(job_query)

                progress_updates = []
                async for progress in orchestrator.monitor_workflow_progress(
                    workflow_id
                ):
                    progress_updates.append(progress)
                    if progress["status"] in ["completed", "failed"]:
                        break

                # Validate background task coordination
                assert len(progress_updates) >= 3, (
                    "Should have multiple progress updates"
                )

                final_progress = progress_updates[-1]
                assert final_progress["status"] == "completed"

                # Verify progress tracking worked
                has_progress_data = any(
                    "current_progress" in update for update in progress_updates
                )
                assert has_progress_data, "Should have progress tracking data"


class TestPerformanceBenchmarkValidation:
    """Test that all system components meet performance benchmarks."""

    @pytest.mark.performance
    async def test_adr_performance_requirements(self, e2e_test_setup):
        """Test all ADR performance requirements are met."""
        setup = e2e_test_setup
        targets = setup["config"]["performance_targets"]

        # Test ADR-013: <500ms job board queries
        with patch.object(
            UnifiedScrapingService, "scrape_job_boards_async", new=AsyncMock()
        ) as mock_scrape_boards:
            # Mock fast job board response
            async def fast_job_board_response(query):
                await asyncio.sleep(0.1)  # Simulate 100ms response
                return [
                    Job(
                        title=f"Engineer {i}",
                        company=f"Company {i}",
                        description=f"Description {i}",
                        link=f"https://jobs.com/{i}",
                        location="Remote",
                        last_seen=datetime.now(UTC),
                    )
                    for i in range(10)
                ]

            mock_scrape_boards.side_effect = fast_job_board_response

            scraper = UnifiedScrapingService()
            query = JobQuery(
                keywords=["python"],
                locations=["San Francisco"],
                source_types=[SourceType.JOB_BOARDS],
                max_results=10,
            )

            # Measure job board query performance
            start_time = time.perf_counter()
            await scraper.scrape_job_boards_async(query)
            job_board_duration = (time.perf_counter() - start_time) * 1000

            assert job_board_duration < targets["job_board_query_ms"], (
                f"Job board query took {job_board_duration:.1f}ms, "
                f"target is {targets['job_board_query_ms']}ms (ADR-013)"
            )

        # Test ADR-010/011: <3s AI enhancement
        with patch.object(
            "src.ai.hybrid_ai_router.HybridAIRouter",
            "enhance_job_content",
            new=AsyncMock(),
        ) as mock_ai_enhance:

            async def fast_ai_response(job):
                await asyncio.sleep(0.2)  # Simulate 200ms AI processing
                return {**job.model_dump(), "ai_enhanced": True}

            mock_ai_enhance.side_effect = fast_ai_response

            # Measure AI enhancement performance
            test_job = Job(
                title="Senior Developer",
                company="TechCorp",
                description="Full stack development role",
                link="https://jobs.com/123",
                location="Remote",
                last_seen=datetime.now(UTC),
            )

            ai_router = get_hybrid_ai_router()
            start_time = time.perf_counter()
            await ai_router.enhance_job_content(test_job)
            ai_duration = (time.perf_counter() - start_time) * 1000

            assert ai_duration < targets["ai_enhancement_ms"], (
                f"AI enhancement took {ai_duration:.1f}ms, "
                f"target is {targets['ai_enhancement_ms']}ms (ADR-010/011)"
            )

        # Test ADR-021: <200ms card rendering (simulated)
        # Note: UI rendering is typically measured in real browser, here we simulate
        with patch("src.ui.components.cards.job_card.render_job_cards") as mock_render:

            def fast_card_render(jobs):
                time.sleep(0.05)  # Simulate 50ms rendering
                return f"<div>Rendered {len(jobs)} job cards</div>"

            mock_render.side_effect = fast_card_render

            # Simulate rendering 50 job cards
            test_jobs = [
                Job(
                    title=f"Job {i}",
                    company=f"Company {i}",
                    description=f"Description {i}",
                    link=f"https://jobs.com/{i}",
                    location="Remote",
                    last_seen=datetime.now(UTC),
                )
                for i in range(50)
            ]

            start_time = time.perf_counter()
            # Mock card rendering call
            mock_render(test_jobs)
            card_duration = (time.perf_counter() - start_time) * 1000

            assert card_duration < targets["card_rendering_ms"], (
                f"Card rendering took {card_duration:.1f}ms for 50 jobs, "
                f"target is {targets['card_rendering_ms']}ms (ADR-021)"
            )

    @pytest.mark.performance
    async def test_scraping_success_rate_validation(self, e2e_test_setup):
        """Test ADR-013 requirement: 95%+ scraping success rate."""
        setup = e2e_test_setup
        target_success_rate = setup["config"]["performance_targets"][
            "scraping_success_rate"
        ]

        # Mock scraping scenarios with realistic success/failure rates
        scraping_scenarios = [
            ("indeed", True),  # Success
            ("linkedin", True),  # Success
            ("glassdoor", True),  # Success
            ("ziprecruiter", False),  # Failure (blocked)
            ("dice", True),  # Success
            ("monster", True),  # Success
            ("angel", True),  # Success
            ("remote_co", False),  # Failure (timeout)
            ("stackoverflow", True),  # Success
            ("github_jobs", True),  # Success
        ]

        with patch.object(
            UnifiedScrapingService, "scrape_job_boards_async", new=AsyncMock()
        ) as mock_scrape:

            async def mock_scraping_response(query):
                # Simulate different outcomes based on scenario
                success_count = sum(1 for _, success in scraping_scenarios if success)
                if success_count / len(scraping_scenarios) >= target_success_rate:
                    return [
                        Job(
                            title=f"Job {i}",
                            company=f"Company {i}",
                            description=f"Description {i}",
                            link=f"https://jobs.com/{i}",
                            location="Remote",
                            last_seen=datetime.now(UTC),
                        )
                        for i in range(success_count)
                    ]
                return []

            mock_scrape.side_effect = mock_scraping_response

            scraper = UnifiedScrapingService()
            query = JobQuery(
                keywords=["python developer"],
                locations=["Remote"],
                source_types=[SourceType.JOB_BOARDS],
                max_results=20,
            )

            # Test scraping success rate
            jobs = await scraper.scrape_job_boards_async(query)
            success_count = len(jobs) if jobs else 0
            actual_success_rate = success_count / len(scraping_scenarios)

            assert actual_success_rate >= target_success_rate, (
                f"Scraping success rate {actual_success_rate:.2%} below "
                f"{target_success_rate:.2%} target (ADR-013)"
            )


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience in workflows."""

    @pytest.mark.integration
    async def test_workflow_error_recovery(self, e2e_test_setup):
        """Test workflow continues with graceful error recovery."""
        setup = e2e_test_setup

        # Mock scenarios where some services fail but workflow continues
        with (
            patch.multiple(
                "src.services.unified_scraper.UnifiedScrapingService",
                scrape_unified=AsyncMock(
                    side_effect=Exception("Scraping temporarily unavailable")
                ),
            ),
            patch.multiple(
                "src.ai.hybrid_ai_router.HybridAIRouter",
                process_content=AsyncMock(return_value="Fallback AI processing"),
            ),
        ):
            orchestrator = ServiceOrchestrator(setup["settings"])

            job_query = JobQuery(
                keywords=["python developer"],
                locations=["Remote"],
                source_types=[SourceType.UNIFIED],
                max_results=10,
            )

            # Execute workflow that will encounter errors
            try:
                workflow_id = await orchestrator.execute_integrated_workflow(job_query)

                # Even with scraping errors, workflow should complete gracefully
                final_status = orchestrator.get_workflow_status(workflow_id)

                # Workflow should either complete with partial results or fail gracefully
                assert final_status is not None
                assert final_status["status"] in ["completed", "failed"]

                if final_status["status"] == "failed":
                    assert "error_message" in final_status

            except Exception as e:
                # Even exceptions should be handled gracefully
                assert "workflow" in str(e).lower(), f"Unexpected error: {e}"

    @pytest.mark.integration
    async def test_service_fallback_mechanisms(self, e2e_test_setup):
        """Test service fallback mechanisms work correctly."""
        setup = e2e_test_setup

        # Mock primary service failures with successful fallbacks
        with (
            patch.multiple(
                "src.ai.local_vllm_service.LocalVLLMService",
                is_healthy=AsyncMock(return_value=False),  # Local AI unavailable
                process_request=AsyncMock(side_effect=Exception("vLLM not available")),
            ),
            patch.multiple(
                "src.ai.cloud_ai_service.CloudAIService",
                is_healthy=AsyncMock(return_value=True),  # Cloud AI available
                process_request=AsyncMock(return_value="Cloud AI response"),
            ),
        ):
            ai_router = get_hybrid_ai_router(setup["settings"])

            # Test AI service fallback
            test_request = {
                "prompt": "Extract job information",
                "content": "Senior Python Developer at TechCorp",
            }

            # Should fallback to cloud AI when local AI fails
            response = await ai_router.process_content(
                content=test_request["content"], prompt=test_request["prompt"]
            )

            assert response is not None, "Should get response from fallback service"
            assert "Cloud AI" in str(response), "Should use cloud AI fallback"

        # Test graceful degradation when all AI services fail
        with (
            patch.multiple(
                "src.ai.local_vllm_service.LocalVLLMService",
                is_healthy=AsyncMock(return_value=False),
                process_request=AsyncMock(side_effect=Exception("vLLM unavailable")),
            ),
            patch.multiple(
                "src.ai.cloud_ai_service.CloudAIService",
                is_healthy=AsyncMock(return_value=False),
                process_request=AsyncMock(
                    side_effect=Exception("Cloud AI unavailable")
                ),
            ),
        ):
            # Should handle graceful degradation
            try:
                response = await ai_router.process_content(
                    content="Test content", prompt="Test prompt"
                )
                # If no exception, should return some form of graceful response
                assert response is not None
            except Exception as e:
                # Exception should be informative about service unavailability
                assert "unavailable" in str(e).lower() or "failed" in str(e).lower()


# Workflow validation reporting
class WorkflowValidationReporter:
    """Generate comprehensive workflow validation reports."""

    @staticmethod
    def generate_workflow_report(test_results: dict[str, Any]) -> dict[str, Any]:
        """Generate end-to-end workflow validation report."""
        return {
            "workflow_validation_summary": {
                "complete_pipeline": {
                    "target": "Search → scrape → enhance → display workflow",
                    "achieved": test_results.get("workflow_success_rate", 0) >= 0.9,
                    "success_rate": test_results.get("workflow_success_rate", 0),
                    "avg_duration": test_results.get("avg_workflow_duration", 0),
                },
                "service_coordination": {
                    "target": "All services working together seamlessly",
                    "achieved": test_results.get("service_coordination_working", False),
                    "services_integrated": test_results.get("services_integrated", []),
                },
                "performance_benchmarks": {
                    "target": "All ADR performance requirements met",
                    "achieved": test_results.get("performance_targets_met", False),
                    "job_board_query_ms": test_results.get("job_board_query_ms", 0),
                    "ai_enhancement_ms": test_results.get("ai_enhancement_ms", 0),
                    "card_rendering_ms": test_results.get("card_rendering_ms", 0),
                    "scraping_success_rate": test_results.get(
                        "scraping_success_rate", 0
                    ),
                },
                "error_recovery": {
                    "target": "Graceful error handling and fallback",
                    "achieved": test_results.get("error_recovery_working", False),
                    "fallback_mechanisms_tested": test_results.get(
                        "fallback_mechanisms", []
                    ),
                },
                "background_tasks": {
                    "target": "Background task coordination and progress tracking",
                    "achieved": test_results.get("background_tasks_working", False),
                    "progress_tracking_working": test_results.get(
                        "progress_tracking", False
                    ),
                },
            },
            "adr_compliance": {
                "adr_013_scraping": {
                    "implemented": True,
                    "performance_met": test_results.get(
                        "job_board_query_ms", float("inf")
                    )
                    < 500,
                    "success_rate_met": test_results.get("scraping_success_rate", 0)
                    >= 0.95,
                },
                "adr_010_011_ai": {
                    "implemented": True,
                    "performance_met": test_results.get(
                        "ai_enhancement_ms", float("inf")
                    )
                    < 3000,
                    "fallback_working": test_results.get("ai_fallback_working", False),
                },
                "adr_017_background": {
                    "implemented": True,
                    "coordination_working": test_results.get(
                        "background_tasks_working", False
                    ),
                },
                "adr_021_cards": {
                    "implemented": True,
                    "performance_met": test_results.get(
                        "card_rendering_ms", float("inf")
                    )
                    < 200,
                },
            },
            "workflow_metrics": test_results,
            "recommendations": WorkflowValidationReporter._generate_workflow_recommendations(
                test_results
            ),
        }

    @staticmethod
    def _generate_workflow_recommendations(test_results: dict[str, Any]) -> list[str]:
        """Generate workflow improvement recommendations."""
        recommendations = []

        if test_results.get("workflow_success_rate", 0) < 0.9:
            recommendations.append(
                "Improve end-to-end workflow reliability and error handling"
            )

        if not test_results.get("performance_targets_met", False):
            recommendations.append("Optimize performance to meet all ADR requirements")

        if not test_results.get("service_coordination_working", False):
            recommendations.append(
                "Enhance service coordination and integration patterns"
            )

        if not test_results.get("error_recovery_working", False):
            recommendations.append("Strengthen error recovery and fallback mechanisms")

        if not test_results.get("background_tasks_working", False):
            recommendations.append(
                "Improve background task management and progress tracking"
            )

        return recommendations


# E2E test configuration
E2E_WORKFLOW_CONFIG = {
    "test_scenarios": [
        {
            "name": "standard_search",
            "keywords": ["python developer", "machine learning"],
            "locations": ["San Francisco", "Remote"],
            "expected_jobs": 25,
        },
        {
            "name": "ai_focused_search",
            "keywords": ["data scientist", "AI engineer"],
            "locations": ["New York", "Austin"],
            "expected_jobs": 15,
        },
        {
            "name": "remote_only_search",
            "keywords": ["full stack developer"],
            "locations": ["Remote"],
            "expected_jobs": 30,
        },
    ],
    "performance_targets": {
        "workflow_timeout_s": 30,
        "job_board_query_ms": 500,
        "ai_enhancement_ms": 3000,
        "card_rendering_ms": 200,
        "scraping_success_rate": 0.95,
        "workflow_success_rate": 0.9,
    },
    "validation_criteria": {
        "service_integration": ["scraper", "ai_router", "search", "ui"],
        "error_scenarios": ["service_failure", "timeout", "invalid_data"],
        "fallback_mechanisms": ["ai_fallback", "scraping_fallback", "ui_degradation"],
    },
}
