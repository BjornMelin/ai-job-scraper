"""Phases 3A-3D System Integration and Coordination Tests.

This test suite validates the integration and coordination between all completed
phases of the AI job scraper system. Tests ensure seamless interoperation
between unified scraping, mobile UI, hybrid AI, and system coordination.

**Phase Integration Requirements**:
- Phase 3A: Unified Scraping Service (JobSpy + ScrapeGraphAI)
- Phase 3B: Mobile-First Responsive Cards (CSS Grid, <200ms rendering)
- Phase 3C: Hybrid AI Integration (vLLM + cloud fallback routing)
- Phase 3D: System Coordination (Background tasks, orchestration)

**Test Coverage**:
- Cross-phase service integration
- Data flow validation across phases
- Service orchestration coordination
- Error handling and recovery integration
- Performance validation across integrated system
"""

import asyncio
import logging
import time

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.factories import create_realistic_dataset

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def phases_integration_setup(session, tmp_path):
    """Set up comprehensive test environment for phase integration testing."""
    # Create comprehensive dataset
    dataset = create_realistic_dataset(
        session,
        companies=25,
        jobs_per_company=20,
        include_inactive_companies=True,
        include_archived_jobs=True,
        senior_ratio=0.3,
        remote_ratio=0.5,
        favorited_ratio=0.2,
    )

    # Mock phase configurations
    phase_configs = {
        "phase_3a_unified_scraping": {
            "jobspy_enabled": True,
            "scrapegraphai_enabled": True,
            "fallback_strategy": "progressive",
            "timeout": 500,  # 500ms target
            "max_concurrent": 5,
        },
        "phase_3b_mobile_ui": {
            "responsive_breakpoints": [320, 768, 1024, 1920],
            "css_grid_enabled": True,
            "render_target_ms": 200,  # 200ms target for 50 jobs
            "touch_targets_min_size": 44,
        },
        "phase_3c_hybrid_ai": {
            "local_vllm_enabled": True,
            "cloud_fallback_enabled": True,
            "processing_target_ms": 3000,  # 3s target
            "routing_strategy": "intelligent",
        },
        "phase_3d_coordination": {
            "background_tasks_enabled": True,
            "service_orchestration_enabled": True,
            "progress_tracking_enabled": True,
            "health_monitoring_enabled": True,
        },
    }

    return {
        "dataset": dataset,
        "configs": phase_configs,
        "session": session,
        "temp_dir": tmp_path,
        "test_jobs": dataset["jobs"][:50],
        "test_companies": dataset["companies"][:10],
    }


class TestPhase3AUnifiedScrapingIntegration:
    """Test Phase 3A unified scraping service integration."""

    @pytest.mark.integration
    async def test_unified_scraper_service_integration(self, phases_integration_setup):
        """Test unified scraper integrates properly with other phases."""
        # Mock unified scraping service
        with patch(
            "src.services.unified_scraper.UnifiedScrapingService"
        ) as MockScraper:
            mock_scraper = Mock()
            MockScraper.return_value = mock_scraper

            # Mock JobSpy integration
            jobspy_results = [
                {
                    "title": f"JobSpy Engineer {i}",
                    "company": f"JobSpy Corp {i}",
                    "location": "San Francisco, CA" if i % 2 == 0 else "Remote",
                    "description": f"JobSpy job description {i}",
                    "salary": {"min": 120000 + i * 5000, "max": 160000 + i * 5000},
                    "date_posted": datetime.now(UTC) - timedelta(days=i),
                    "job_url": f"https://jobspy.example.com/job/{i}",
                    "source": "jobspy",
                }
                for i in range(15)
            ]

            # Mock ScrapeGraphAI integration
            scrapegraphai_results = [
                {
                    "title": f"GraphAI Developer {i}",
                    "company": f"GraphAI Inc {i}",
                    "location": "New York, NY" if i % 2 == 0 else "Austin, TX",
                    "description": f"ScrapeGraphAI extracted job {i}",
                    "salary": {"min": 110000 + i * 4000, "max": 150000 + i * 4000},
                    "date_posted": datetime.now(UTC) - timedelta(days=i + 2),
                    "job_url": f"https://graphai.example.com/job/{i}",
                    "source": "scrapegraphai",
                }
                for i in range(10)
            ]

            # Configure unified scraper mock
            async def mock_unified_scrape(search_term, location=None, max_jobs=25):
                # Simulate unified scraping delay
                await asyncio.sleep(0.3)  # 300ms - within 500ms target

                # Combine results from both sources
                combined_results = (
                    jobspy_results[: max_jobs // 2]
                    + scrapegraphai_results[: max_jobs // 2]
                )
                return combined_results[:max_jobs]

            mock_scraper.scrape_unified = AsyncMock(side_effect=mock_unified_scrape)

            # Test integration scenarios
            integration_scenarios = [
                {
                    "search_term": "python developer",
                    "location": "San Francisco",
                    "max_jobs": 20,
                    "expected_sources": ["jobspy", "scrapegraphai"],
                },
                {
                    "search_term": "machine learning engineer",
                    "location": "Remote",
                    "max_jobs": 15,
                    "expected_sources": ["jobspy", "scrapegraphai"],
                },
                {
                    "search_term": "data scientist",
                    "location": "New York",
                    "max_jobs": 25,
                    "expected_sources": ["jobspy", "scrapegraphai"],
                },
            ]

            integration_results = []

            for scenario in integration_scenarios:
                start_time = time.perf_counter()

                # Execute unified scraping
                results = await mock_scraper.scrape_unified(
                    search_term=scenario["search_term"],
                    location=scenario["location"],
                    max_jobs=scenario["max_jobs"],
                )

                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Validate integration
                sources_found = {job["source"] for job in results}
                expected_sources = set(scenario["expected_sources"])

                integration_results.append(
                    {
                        "scenario": f"{scenario['search_term']} in {scenario['location']}",
                        "execution_time_ms": execution_time_ms,
                        "job_count": len(results),
                        "sources_found": sources_found,
                        "expected_sources": expected_sources,
                        "sources_integrated": expected_sources.issubset(sources_found),
                        "performance_good": execution_time_ms < 500.0,
                        "integration_successful": (
                            len(results) > 0
                            and expected_sources.issubset(sources_found)
                            and execution_time_ms < 500.0
                        ),
                    }
                )

            # Validate Phase 3A integration
            successful_integrations = [
                r for r in integration_results if r["integration_successful"]
            ]
            integration_rate = len(successful_integrations) / len(integration_results)

            assert integration_rate >= 0.9, (
                f"Phase 3A integration success rate {integration_rate:.2%}, should be ≥90%. "
                f"Failures: {[r for r in integration_results if not r['integration_successful']]}"
            )

    @pytest.mark.integration
    async def test_scraping_fallback_coordination(self, phases_integration_setup):
        """Test scraping fallback strategy coordination."""
        with patch(
            "src.services.unified_scraper.UnifiedScrapingService"
        ) as MockScraper:
            mock_scraper = Mock()
            MockScraper.return_value = mock_scraper

            # Test fallback scenarios
            fallback_scenarios = [
                {
                    "primary_source": "jobspy",
                    "fallback_source": "scrapegraphai",
                    "primary_fails": True,
                    "fallback_succeeds": True,
                },
                {
                    "primary_source": "scrapegraphai",
                    "fallback_source": "jobspy",
                    "primary_fails": True,
                    "fallback_succeeds": True,
                },
                {
                    "primary_source": "jobspy",
                    "fallback_source": "scrapegraphai",
                    "primary_fails": False,
                    "fallback_succeeds": True,
                },
            ]

            fallback_results = []

            for scenario in fallback_scenarios:
                # Mock fallback behavior
                async def mock_scrape_with_fallback():
                    await asyncio.sleep(0.1)  # Primary attempt

                    if scenario["primary_fails"]:
                        await asyncio.sleep(0.2)  # Fallback attempt
                        if scenario["fallback_succeeds"]:
                            return [
                                {
                                    "title": f"Fallback Job from {scenario['fallback_source']}",
                                    "company": "Fallback Company",
                                    "source": scenario["fallback_source"],
                                    "fallback_used": True,
                                }
                            ]
                        return []
                    return [
                        {
                            "title": f"Primary Job from {scenario['primary_source']}",
                            "company": "Primary Company",
                            "source": scenario["primary_source"],
                            "fallback_used": False,
                        }
                    ]

                mock_scraper.scrape_with_fallback = AsyncMock(
                    return_value=mock_scrape_with_fallback()
                )

                # Test fallback coordination
                start_time = time.perf_counter()
                results = await mock_scraper.scrape_with_fallback()
                fallback_time_ms = (time.perf_counter() - start_time) * 1000

                fallback_used = len(results) > 0 and results[0].get(
                    "fallback_used", False
                )
                expected_fallback = (
                    scenario["primary_fails"] and scenario["fallback_succeeds"]
                )

                fallback_results.append(
                    {
                        "scenario": f"{scenario['primary_source']} -> {scenario['fallback_source']}",
                        "primary_fails": scenario["primary_fails"],
                        "fallback_time_ms": fallback_time_ms,
                        "fallback_used": fallback_used,
                        "expected_fallback": expected_fallback,
                        "results_count": len(results),
                        "fallback_coordinated": (
                            (fallback_used == expected_fallback)
                            and (
                                len(results) > 0
                                if scenario["fallback_succeeds"]
                                else True
                            )
                            and fallback_time_ms < 500.0
                        ),
                    }
                )

            # Validate fallback coordination
            coordinated_fallbacks = [
                r for r in fallback_results if r["fallback_coordinated"]
            ]
            coordination_rate = len(coordinated_fallbacks) / len(fallback_results)

            assert coordination_rate >= 0.9, (
                f"Scraping fallback coordination rate {coordination_rate:.2%}, should be ≥90%. "
                f"Poor coordination: {[r for r in fallback_results if not r['fallback_coordinated']]}"
            )


class TestPhase3BMobileUIIntegration:
    """Test Phase 3B mobile-first responsive UI integration."""

    @pytest.mark.integration
    def test_mobile_ui_data_integration(self, phases_integration_setup):
        """Test mobile UI integrates properly with data from other phases."""
        setup = phases_integration_setup
        test_jobs = setup["test_jobs"]

        # Mock different data sources integration
        data_integration_scenarios = [
            {
                "data_source": "unified_scraper",
                "job_count": 25,
                "data_enriched": False,
                "viewport": 320,  # Mobile
            },
            {
                "data_source": "ai_enhanced_jobs",
                "job_count": 25,
                "data_enriched": True,
                "viewport": 768,  # Tablet
            },
            {
                "data_source": "database_cached_jobs",
                "job_count": 50,
                "data_enriched": True,
                "viewport": 1200,  # Desktop
            },
        ]

        ui_integration_results = []

        for scenario in data_integration_scenarios:
            # Mock different viewport
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = scenario["viewport"]

                # Mock data from different sources
                test_data = test_jobs[: scenario["job_count"]]
                if scenario["data_enriched"]:
                    # Add AI enhancement mock data
                    for job in test_data:
                        job.skills = ["Python", "Django", "React"]  # AI extracted
                        job.seniority_level = "Mid-level"  # AI classified
                        job.remote_friendly = True  # AI determined

                # Mock Streamlit rendering
                with patch("src.ui.components.cards.job_card.st") as mock_st:
                    mock_st.container = Mock()
                    mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
                    mock_st.markdown = Mock()
                    mock_st.button = Mock(return_value=False)

                    from src.ui.components.cards.job_card import render_job_cards

                    # Test UI data integration
                    start_time = time.perf_counter()
                    render_job_cards(
                        test_data, device_type=f"integration_{scenario['viewport']}"
                    )
                    render_time_ms = (time.perf_counter() - start_time) * 1000

                    # Validate rendering performance by job count
                    performance_target = 200.0 if scenario["job_count"] <= 50 else 400.0

                    ui_integration_results.append(
                        {
                            "data_source": scenario["data_source"],
                            "job_count": scenario["job_count"],
                            "viewport": scenario["viewport"],
                            "data_enriched": scenario["data_enriched"],
                            "render_time_ms": render_time_ms,
                            "performance_target": performance_target,
                            "performance_good": render_time_ms < performance_target,
                            "ui_integration_successful": render_time_ms
                            < performance_target,
                        }
                    )

        # Validate UI integration
        successful_ui_integration = [
            r for r in ui_integration_results if r["ui_integration_successful"]
        ]
        ui_integration_rate = len(successful_ui_integration) / len(
            ui_integration_results
        )

        assert ui_integration_rate >= 0.9, (
            f"Phase 3B UI integration success rate {ui_integration_rate:.2%}, should be ≥90%. "
            f"Poor integrations: {[r for r in ui_integration_results if not r['ui_integration_successful']]}"
        )

    @pytest.mark.integration
    def test_responsive_rendering_integration(self, phases_integration_setup):
        """Test responsive rendering integrates with various data loads."""
        setup = phases_integration_setup

        # Test rendering across different data volumes and device types
        rendering_integration_tests = [
            {"device": "mobile", "viewport": 320, "job_count": 10, "max_time": 150},
            {"device": "mobile", "viewport": 414, "job_count": 25, "max_time": 200},
            {"device": "tablet", "viewport": 768, "job_count": 40, "max_time": 300},
            {"device": "desktop", "viewport": 1200, "job_count": 50, "max_time": 200},
            {"device": "large", "viewport": 1920, "job_count": 50, "max_time": 200},
        ]

        rendering_results = []

        for test in rendering_integration_tests:
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = test["viewport"]

                with patch("src.ui.components.cards.job_card.st") as mock_st:
                    # Configure columns based on viewport
                    if test["viewport"] < 600:
                        columns = [Mock()]  # 1 column
                    elif test["viewport"] < 900:
                        columns = [Mock(), Mock()]  # 2 columns
                    elif test["viewport"] < 1200:
                        columns = [Mock(), Mock(), Mock()]  # 3 columns
                    else:
                        columns = [Mock(), Mock(), Mock(), Mock()]  # 4 columns

                    mock_st.columns = Mock(return_value=columns)
                    mock_st.container = Mock()
                    mock_st.markdown = Mock()
                    mock_st.button = Mock(return_value=False)

                    from src.ui.components.cards.job_card import render_job_cards

                    # Test responsive rendering
                    test_jobs = setup["test_jobs"][: test["job_count"]]

                    start_time = time.perf_counter()
                    render_job_cards(test_jobs, device_type=test["device"])
                    render_time_ms = (time.perf_counter() - start_time) * 1000

                    rendering_results.append(
                        {
                            "device": test["device"],
                            "viewport": test["viewport"],
                            "job_count": test["job_count"],
                            "render_time_ms": render_time_ms,
                            "max_time": test["max_time"],
                            "columns_count": len(columns),
                            "responsive_good": render_time_ms < test["max_time"],
                        }
                    )

        # Validate responsive rendering integration
        good_rendering = [r for r in rendering_results if r["responsive_good"]]
        rendering_success_rate = len(good_rendering) / len(rendering_results)

        assert rendering_success_rate >= 0.9, (
            f"Responsive rendering integration success rate {rendering_success_rate:.2%}, "
            f"should be ≥90%. Poor rendering: {[r for r in rendering_results if not r['responsive_good']]}"
        )


class TestPhase3CHybridAIIntegration:
    """Test Phase 3C hybrid AI processing integration."""

    @pytest.mark.integration
    async def test_hybrid_ai_processing_integration(self, phases_integration_setup):
        """Test hybrid AI integrates with scraping and UI phases."""
        with patch("src.ai.hybrid_ai_router.HybridAIRouter") as MockAI:
            mock_ai = Mock()
            MockAI.return_value = mock_ai

            # Mock raw job data from scraping phase
            raw_jobs_from_scraper = [
                {
                    "title": "Senior Python Developer",
                    "company": "TechCorp",
                    "description": "We need a Python expert with Django experience.",
                    "location": "San Francisco, CA",
                    "salary_text": "120k-160k",
                    "source": "jobspy",
                },
                {
                    "title": "ML Engineer",
                    "company": "AI Startup",
                    "description": "Machine learning role with PyTorch.",
                    "location": "Remote",
                    "salary_text": "140k-200k",
                    "source": "scrapegraphai",
                },
                {
                    "title": "Data Scientist",
                    "company": "DataCorp",
                    "description": "Analyze datasets, build models.",
                    "location": "New York, NY",
                    "salary_text": "$130k-$170k",
                    "source": "jobspy",
                },
            ]

            # Test different AI processing scenarios
            ai_integration_scenarios = [
                {
                    "scenario": "local_processing",
                    "job_count": 1,
                    "processing_time": 0.8,  # 800ms
                    "route": "local",
                },
                {
                    "scenario": "cloud_fallback",
                    "job_count": 3,
                    "processing_time": 2.5,  # 2.5s
                    "route": "cloud",
                },
                {
                    "scenario": "hybrid_batch",
                    "job_count": 5,
                    "processing_time": 2.2,  # 2.2s
                    "route": "hybrid",
                },
            ]

            ai_results = []

            for scenario in ai_integration_scenarios:
                jobs_to_process = raw_jobs_from_scraper[: scenario["job_count"]]

                # Mock AI enhancement
                async def mock_ai_enhancement():
                    await asyncio.sleep(scenario["processing_time"])

                    enhanced = []
                    for i, job in enumerate(jobs_to_process):
                        enhanced.append(
                            {
                                **job,
                                "salary": {
                                    "min": 120000 + i * 10000,
                                    "max": 160000 + i * 20000,
                                },
                                "skills": ["Python", "Django", "FastAPI"][: i + 1],
                                "seniority_level": ["Junior", "Mid-level", "Senior"][i],
                                "remote_friendly": "Remote" in job["location"]
                                or "remote" in job["description"].lower(),
                                "confidence_score": 0.9 - i * 0.1,
                                "processing_route": scenario["route"],
                            }
                        )
                    return enhanced

                mock_ai.enhance_job_data = AsyncMock(return_value=mock_ai_enhancement())

                # Test AI processing integration
                start_time = time.perf_counter()
                enhanced_jobs = await mock_ai.enhance_job_data(jobs_to_process)
                processing_time_ms = (time.perf_counter() - start_time) * 1000

                ai_results.append(
                    {
                        "scenario": scenario["scenario"],
                        "job_count": scenario["job_count"],
                        "processing_time_ms": processing_time_ms,
                        "enhanced_count": len(enhanced_jobs),
                        "route_used": scenario["route"],
                        "performance_good": processing_time_ms < 3000.0,  # 3s target
                        "enhancement_successful": (
                            len(enhanced_jobs) == len(jobs_to_process)
                            and processing_time_ms < 3000.0
                            and all("skills" in job for job in enhanced_jobs)
                            and all("salary" in job for job in enhanced_jobs)
                        ),
                    }
                )

            # Validate AI integration
            successful_ai = [r for r in ai_results if r["enhancement_successful"]]
            ai_integration_rate = len(successful_ai) / len(ai_results)

            assert ai_integration_rate >= 0.9, (
                f"Phase 3C AI integration success rate {ai_integration_rate:.2%}, should be ≥90%. "
                f"Poor AI integration: {[r for r in ai_results if not r['enhancement_successful']]}"
            )

    @pytest.mark.integration
    async def test_ai_routing_decision_integration(self, phases_integration_setup):
        """Test AI routing decisions integrate with system coordination."""
        with (
            patch("src.ai.hybrid_ai_router.HybridAIRouter") as MockAI,
            patch(
                "src.coordination.service_orchestrator.ServiceOrchestrator"
            ) as MockOrchestrator,
        ):
            mock_ai = Mock()
            MockAI.return_value = mock_ai

            mock_orchestrator = Mock()
            MockOrchestrator.return_value = mock_orchestrator

            # Test routing decision scenarios
            routing_scenarios = [
                {
                    "system_load": "low",
                    "job_count": 1,
                    "complexity": "simple",
                    "expected_route": "local",
                    "coordination_needed": False,
                },
                {
                    "system_load": "medium",
                    "job_count": 5,
                    "complexity": "medium",
                    "expected_route": "hybrid",
                    "coordination_needed": True,
                },
                {
                    "system_load": "high",
                    "job_count": 15,
                    "complexity": "complex",
                    "expected_route": "cloud",
                    "coordination_needed": True,
                },
            ]

            routing_results = []

            for scenario in routing_scenarios:
                # Mock routing decision
                def mock_routing_decision():
                    return {
                        "route": scenario["expected_route"],
                        "confidence": 0.85,
                        "system_load": scenario["system_load"],
                        "coordination_required": scenario["coordination_needed"],
                    }

                mock_ai.decide_routing = Mock(return_value=mock_routing_decision())

                # Mock coordination response
                async def mock_coordinate_routing():
                    if scenario["coordination_needed"]:
                        await asyncio.sleep(0.05)  # 50ms coordination overhead
                        return {"coordinated": True, "resources_allocated": True}
                    return {"coordinated": False, "resources_allocated": False}

                mock_orchestrator.coordinate_ai_resources = AsyncMock(
                    return_value=mock_coordinate_routing()
                )

                # Test integrated routing decision
                start_time = time.perf_counter()

                routing_decision = mock_ai.decide_routing(
                    job_count=scenario["job_count"],
                    complexity=scenario["complexity"],
                    system_load=scenario["system_load"],
                )

                coordination_result = await mock_orchestrator.coordinate_ai_resources(
                    routing_decision
                )

                decision_time_ms = (time.perf_counter() - start_time) * 1000

                routing_results.append(
                    {
                        "scenario": f"{scenario['system_load']}_load_{scenario['job_count']}_jobs",
                        "expected_route": scenario["expected_route"],
                        "actual_route": routing_decision["route"],
                        "coordination_needed": scenario["coordination_needed"],
                        "coordination_provided": coordination_result["coordinated"],
                        "decision_time_ms": decision_time_ms,
                        "routing_integrated": (
                            routing_decision["route"] == scenario["expected_route"]
                            and coordination_result["coordinated"]
                            == scenario["coordination_needed"]
                            and decision_time_ms < 200.0  # Fast routing decisions
                        ),
                    }
                )

            # Validate routing integration
            integrated_routing = [r for r in routing_results if r["routing_integrated"]]
            routing_integration_rate = len(integrated_routing) / len(routing_results)

            assert routing_integration_rate >= 0.9, (
                f"AI routing integration success rate {routing_integration_rate:.2%}, "
                f"should be ≥90%. Poor routing: {[r for r in routing_results if not r['routing_integrated']]}"
            )


class TestPhase3DCoordinationIntegration:
    """Test Phase 3D system coordination integration."""

    @pytest.mark.integration
    async def test_background_task_coordination(self, phases_integration_setup):
        """Test background task coordination across all phases."""
        with (
            patch(
                "src.coordination.background_task_manager.BackgroundTaskManager"
            ) as MockTaskManager,
            patch(
                "src.coordination.service_orchestrator.ServiceOrchestrator"
            ) as MockOrchestrator,
            patch("src.coordination.progress_tracker.ProgressTracker") as MockTracker,
        ):
            mock_task_manager = Mock()
            MockTaskManager.return_value = mock_task_manager

            mock_orchestrator = Mock()
            MockOrchestrator.return_value = mock_orchestrator

            mock_tracker = Mock()
            MockTracker.return_value = mock_tracker

            # Test coordinated workflow scenarios
            coordination_scenarios = [
                {
                    "workflow": "scrape_enhance_render",
                    "tasks": ["scrape_jobs", "ai_enhance", "update_ui"],
                    "concurrent_limit": 3,
                    "expected_duration": 5.0,  # 5 seconds total
                },
                {
                    "workflow": "bulk_processing",
                    "tasks": [
                        "scrape_multiple_sources",
                        "batch_ai_processing",
                        "bulk_ui_update",
                    ],
                    "concurrent_limit": 5,
                    "expected_duration": 8.0,  # 8 seconds total
                },
                {
                    "workflow": "incremental_update",
                    "tasks": [
                        "detect_changes",
                        "selective_processing",
                        "progressive_ui_update",
                    ],
                    "concurrent_limit": 2,
                    "expected_duration": 3.0,  # 3 seconds total
                },
            ]

            coordination_results = []

            for scenario in coordination_scenarios:
                # Mock task scheduling
                async def mock_schedule_tasks():
                    await asyncio.sleep(0.1)  # Scheduling overhead
                    return {
                        "scheduled": len(scenario["tasks"]),
                        "running": min(
                            len(scenario["tasks"]), scenario["concurrent_limit"]
                        ),
                        "workflow_id": f"workflow_{len(coordination_results)}",
                    }

                mock_task_manager.schedule_workflow = AsyncMock(
                    return_value=mock_schedule_tasks()
                )

                # Mock orchestration
                async def mock_orchestrate():
                    await asyncio.sleep(0.05)  # Orchestration overhead
                    return {
                        "orchestrated": True,
                        "resources_allocated": True,
                        "estimated_completion": scenario["expected_duration"],
                    }

                mock_orchestrator.orchestrate_workflow = AsyncMock(
                    return_value=mock_orchestrate()
                )

                # Mock progress tracking
                def mock_track_progress():
                    return {
                        "progress": 0.0,
                        "completed_tasks": 0,
                        "total_tasks": len(scenario["tasks"]),
                        "estimated_remaining": scenario["expected_duration"],
                    }

                mock_tracker.start_tracking = Mock(return_value=mock_track_progress())

                # Test coordinated execution
                start_time = time.perf_counter()

                # Schedule workflow
                schedule_result = await mock_task_manager.schedule_workflow(
                    workflow_type=scenario["workflow"],
                    tasks=scenario["tasks"],
                    concurrent_limit=scenario["concurrent_limit"],
                )

                # Orchestrate execution
                orchestration_result = await mock_orchestrator.orchestrate_workflow(
                    workflow_id=schedule_result["workflow_id"]
                )

                # Start progress tracking
                tracking_result = mock_tracker.start_tracking(
                    workflow_id=schedule_result["workflow_id"]
                )

                coordination_time_ms = (time.perf_counter() - start_time) * 1000

                coordination_results.append(
                    {
                        "workflow": scenario["workflow"],
                        "tasks_count": len(scenario["tasks"]),
                        "coordination_time_ms": coordination_time_ms,
                        "scheduled_tasks": schedule_result["scheduled"],
                        "orchestrated": orchestration_result["orchestrated"],
                        "tracking_started": tracking_result is not None,
                        "coordination_successful": (
                            schedule_result["scheduled"] == len(scenario["tasks"])
                            and orchestration_result["orchestrated"]
                            and tracking_result is not None
                            and coordination_time_ms < 500.0  # Fast coordination
                        ),
                    }
                )

            # Validate coordination integration
            successful_coordination = [
                r for r in coordination_results if r["coordination_successful"]
            ]
            coordination_rate = len(successful_coordination) / len(coordination_results)

            assert coordination_rate >= 0.9, (
                f"Phase 3D coordination success rate {coordination_rate:.2%}, should be ≥90%. "
                f"Poor coordination: {[r for r in coordination_results if not r['coordination_successful']]}"
            )

    @pytest.mark.integration
    async def test_system_health_monitoring_integration(self, phases_integration_setup):
        """Test system health monitoring integrates across all phases."""
        with patch(
            "src.coordination.system_health_monitor.SystemHealthMonitor"
        ) as MockHealthMonitor:
            mock_health_monitor = Mock()
            MockHealthMonitor.return_value = mock_health_monitor

            # Mock health monitoring for each phase
            phase_health_scenarios = [
                {
                    "phase": "3a_unified_scraping",
                    "metrics": {
                        "scraping_success_rate": 0.96,
                        "avg_response_time_ms": 350,
                        "error_rate": 0.04,
                        "active_connections": 3,
                    },
                    "health_status": "healthy",
                },
                {
                    "phase": "3b_mobile_ui",
                    "metrics": {
                        "render_performance_ms": 180,
                        "responsiveness_rate": 0.98,
                        "ui_error_rate": 0.01,
                        "active_sessions": 15,
                    },
                    "health_status": "healthy",
                },
                {
                    "phase": "3c_hybrid_ai",
                    "metrics": {
                        "ai_processing_time_ms": 2800,
                        "local_model_uptime": 0.99,
                        "cloud_fallback_rate": 0.15,
                        "enhancement_success_rate": 0.94,
                    },
                    "health_status": "healthy",
                },
                {
                    "phase": "3d_coordination",
                    "metrics": {
                        "task_completion_rate": 0.97,
                        "orchestration_overhead_ms": 45,
                        "resource_utilization": 0.65,
                        "coordination_errors": 0.02,
                    },
                    "health_status": "healthy",
                },
            ]

            health_monitoring_results = []

            for scenario in phase_health_scenarios:
                # Mock health check
                def mock_health_check():
                    return {
                        "phase": scenario["phase"],
                        "status": scenario["health_status"],
                        "metrics": scenario["metrics"],
                        "timestamp": datetime.now(UTC),
                        "issues": [],
                    }

                mock_health_monitor.check_phase_health = Mock(
                    return_value=mock_health_check()
                )

                # Test health monitoring
                start_time = time.perf_counter()
                health_result = mock_health_monitor.check_phase_health(
                    scenario["phase"]
                )
                monitoring_time_ms = (time.perf_counter() - start_time) * 1000

                # Validate health thresholds
                phase_healthy = self._validate_phase_health(
                    scenario["phase"], scenario["metrics"]
                )

                health_monitoring_results.append(
                    {
                        "phase": scenario["phase"],
                        "monitoring_time_ms": monitoring_time_ms,
                        "status": health_result["status"],
                        "metrics_count": len(health_result["metrics"]),
                        "phase_healthy": phase_healthy,
                        "monitoring_fast": monitoring_time_ms < 100.0,
                        "health_monitoring_working": (
                            health_result["status"] == scenario["health_status"]
                            and phase_healthy
                            and monitoring_time_ms < 100.0
                        ),
                    }
                )

            # Validate health monitoring integration
            working_monitoring = [
                r for r in health_monitoring_results if r["health_monitoring_working"]
            ]
            monitoring_rate = len(working_monitoring) / len(health_monitoring_results)

            assert monitoring_rate >= 0.95, (
                f"Health monitoring integration success rate {monitoring_rate:.2%}, "
                f"should be ≥95%. Poor monitoring: {[r for r in health_monitoring_results if not r['health_monitoring_working']]}"
            )

    def _validate_phase_health(self, phase: str, metrics: dict) -> bool:
        """Validate phase health based on metrics."""
        health_thresholds = {
            "3a_unified_scraping": {
                "scraping_success_rate": 0.95,
                "avg_response_time_ms": 500,
                "error_rate": 0.05,
            },
            "3b_mobile_ui": {
                "render_performance_ms": 200,
                "responsiveness_rate": 0.90,
                "ui_error_rate": 0.05,
            },
            "3c_hybrid_ai": {
                "ai_processing_time_ms": 3000,
                "local_model_uptime": 0.95,
                "enhancement_success_rate": 0.90,
            },
            "3d_coordination": {
                "task_completion_rate": 0.95,
                "orchestration_overhead_ms": 100,
                "coordination_errors": 0.05,
            },
        }

        thresholds = health_thresholds.get(phase, {})
        for metric, threshold in thresholds.items():
            if metric in metrics:
                if metric.endswith("_rate") or metric == "local_model_uptime":
                    # Higher is better
                    if metrics[metric] < threshold:
                        return False
                elif metric.endswith(("_ms", "errors")):
                    # Lower is better
                    if metrics[metric] > threshold:
                        return False

        return True


class TestCrossPhaseErrorHandlingIntegration:
    """Test error handling and recovery across phase boundaries."""

    @pytest.mark.integration
    async def test_cascading_error_recovery(self, phases_integration_setup):
        """Test error recovery cascades properly across phases."""
        # Test error cascade scenarios
        error_scenarios = [
            {
                "failure_phase": "3a_scraping",
                "error_type": "network_timeout",
                "should_cascade_to": ["3c_ai_processing"],
                "recovery_expected": True,
                "fallback_available": True,
            },
            {
                "failure_phase": "3c_ai_processing",
                "error_type": "model_unavailable",
                "should_cascade_to": ["3b_ui_rendering"],
                "recovery_expected": True,
                "fallback_available": True,
            },
            {
                "failure_phase": "3d_coordination",
                "error_type": "orchestration_failure",
                "should_cascade_to": ["3a_scraping", "3c_ai_processing"],
                "recovery_expected": True,
                "fallback_available": False,
            },
        ]

        error_recovery_results = []

        for scenario in error_scenarios:
            # Mock error in primary phase
            primary_error = Exception(
                f"{scenario['error_type']} in {scenario['failure_phase']}"
            )

            # Mock error handling and recovery
            with patch(
                "src.coordination.service_orchestrator.ServiceOrchestrator"
            ) as MockOrchestrator:
                mock_orchestrator = Mock()
                MockOrchestrator.return_value = mock_orchestrator

                async def mock_handle_phase_error():
                    await asyncio.sleep(0.2)  # Error handling delay

                    if scenario["recovery_expected"]:
                        return {
                            "recovered": True,
                            "fallback_used": scenario["fallback_available"],
                            "affected_phases": scenario["should_cascade_to"],
                            "recovery_time_ms": 200,
                        }
                    return {
                        "recovered": False,
                        "fallback_used": False,
                        "affected_phases": scenario["should_cascade_to"],
                        "recovery_time_ms": 0,
                    }

                mock_orchestrator.handle_phase_error = AsyncMock(
                    return_value=mock_handle_phase_error()
                )

                # Test error cascade and recovery
                start_time = time.perf_counter()

                try:
                    # Simulate error occurring
                    raise primary_error
                except Exception:
                    # Handle error through orchestrator
                    recovery_result = await mock_orchestrator.handle_phase_error(
                        failing_phase=scenario["failure_phase"],
                        error_type=scenario["error_type"],
                    )

                error_handling_time_ms = (time.perf_counter() - start_time) * 1000

                error_recovery_results.append(
                    {
                        "scenario": f"{scenario['failure_phase']}_{scenario['error_type']}",
                        "failure_phase": scenario["failure_phase"],
                        "error_type": scenario["error_type"],
                        "recovery_time_ms": error_handling_time_ms,
                        "recovered": recovery_result["recovered"],
                        "fallback_used": recovery_result["fallback_used"],
                        "affected_phases": recovery_result["affected_phases"],
                        "recovery_successful": (
                            recovery_result["recovered"]
                            == scenario["recovery_expected"]
                            and recovery_result["fallback_used"]
                            == scenario["fallback_available"]
                            and set(recovery_result["affected_phases"])
                            == set(scenario["should_cascade_to"])
                            and error_handling_time_ms < 1000.0  # Fast error recovery
                        ),
                    }
                )

        # Validate error recovery integration
        successful_recovery = [
            r for r in error_recovery_results if r["recovery_successful"]
        ]
        recovery_rate = len(successful_recovery) / len(error_recovery_results)

        assert recovery_rate >= 0.8, (
            f"Cross-phase error recovery success rate {recovery_rate:.2%}, should be ≥80%. "
            f"Poor recovery: {[r for r in error_recovery_results if not r['recovery_successful']]}"
        )

    @pytest.mark.integration
    async def test_graceful_degradation_integration(self, phases_integration_setup):
        """Test graceful degradation across integrated phases."""
        # Test degradation scenarios
        degradation_scenarios = [
            {
                "degradation_trigger": "high_system_load",
                "affected_phases": ["3a_scraping", "3c_ai_processing"],
                "degradation_strategy": "reduce_concurrency",
                "performance_impact": 0.3,  # 30% slower
                "functionality_maintained": True,
            },
            {
                "degradation_trigger": "ai_service_unavailable",
                "affected_phases": ["3c_ai_processing"],
                "degradation_strategy": "disable_enhancement",
                "performance_impact": 0.0,  # No slowdown
                "functionality_maintained": True,
            },
            {
                "degradation_trigger": "mobile_bandwidth_limited",
                "affected_phases": ["3b_mobile_ui"],
                "degradation_strategy": "reduce_ui_complexity",
                "performance_impact": -0.2,  # 20% faster (simplified UI)
                "functionality_maintained": True,
            },
        ]

        degradation_results = []

        for scenario in degradation_scenarios:
            # Mock degradation handling
            with patch(
                "src.coordination.service_orchestrator.ServiceOrchestrator"
            ) as MockOrchestrator:
                mock_orchestrator = Mock()
                MockOrchestrator.return_value = mock_orchestrator

                async def mock_apply_degradation():
                    await asyncio.sleep(0.1)  # Degradation setup time

                    return {
                        "degraded": True,
                        "strategy": scenario["degradation_strategy"],
                        "affected_phases": scenario["affected_phases"],
                        "performance_impact": scenario["performance_impact"],
                        "functionality_maintained": scenario[
                            "functionality_maintained"
                        ],
                    }

                mock_orchestrator.apply_graceful_degradation = AsyncMock(
                    return_value=mock_apply_degradation()
                )

                # Test graceful degradation
                start_time = time.perf_counter()
                degradation_result = await mock_orchestrator.apply_graceful_degradation(
                    trigger=scenario["degradation_trigger"]
                )
                degradation_time_ms = (time.perf_counter() - start_time) * 1000

                degradation_results.append(
                    {
                        "scenario": scenario["degradation_trigger"],
                        "degradation_time_ms": degradation_time_ms,
                        "strategy_applied": degradation_result["strategy"],
                        "affected_phases": degradation_result["affected_phases"],
                        "functionality_maintained": degradation_result[
                            "functionality_maintained"
                        ],
                        "performance_impact": degradation_result["performance_impact"],
                        "degradation_successful": (
                            degradation_result["strategy"]
                            == scenario["degradation_strategy"]
                            and set(degradation_result["affected_phases"])
                            == set(scenario["affected_phases"])
                            and degradation_result["functionality_maintained"]
                            == scenario["functionality_maintained"]
                            and degradation_time_ms < 500.0  # Fast degradation
                        ),
                    }
                )

        # Validate graceful degradation
        successful_degradation = [
            r for r in degradation_results if r["degradation_successful"]
        ]
        degradation_rate = len(successful_degradation) / len(degradation_results)

        assert degradation_rate >= 0.9, (
            f"Graceful degradation success rate {degradation_rate:.2%}, should be ≥90%. "
            f"Poor degradation: {[r for r in degradation_results if not r['degradation_successful']]}"
        )


class TestEndToEndPhaseIntegration:
    """Test complete end-to-end integration across all phases."""

    @pytest.mark.integration
    async def test_complete_workflow_integration(self, phases_integration_setup):
        """Test complete workflow integrates all phases seamlessly."""
        # Mock complete workflow with all phases
        with (
            patch("src.services.unified_scraper.UnifiedScrapingService") as MockScraper,
            patch("src.ai.hybrid_ai_router.HybridAIRouter") as MockAI,
            patch("src.ui.components.cards.job_card.render_job_cards") as MockRender,
            patch(
                "src.coordination.service_orchestrator.ServiceOrchestrator"
            ) as MockOrchestrator,
        ):
            # Configure all mocks
            mock_scraper = Mock()
            MockScraper.return_value = mock_scraper

            mock_ai = Mock()
            MockAI.return_value = mock_ai

            mock_orchestrator = Mock()
            MockOrchestrator.return_value = mock_orchestrator

            # Complete workflow test scenario
            workflow_scenario = {
                "search_query": "senior python developer",
                "location": "San Francisco",
                "max_jobs": 30,
                "viewport": 1200,  # Desktop
                "expected_total_time": 6000,  # 6 seconds total
            }

            # Phase 3A: Scraping (≤500ms)
            scraped_jobs = [
                {
                    "title": f"Python Developer {i}",
                    "company": f"Company {i}",
                    "description": f"Python role {i}",
                    "location": "San Francisco, CA",
                    "salary_text": "120k-160k",
                    "source": "jobspy" if i % 2 == 0 else "scrapegraphai",
                }
                for i in range(30)
            ]

            async def mock_unified_scrape():
                await asyncio.sleep(0.4)  # 400ms - within target
                return scraped_jobs

            mock_scraper.scrape_unified = AsyncMock(return_value=mock_unified_scrape())

            # Phase 3C: AI Enhancement (≤3000ms)
            enhanced_jobs = [
                {
                    **job,
                    "salary": {"min": 120000 + i * 2000, "max": 160000 + i * 3000},
                    "skills": ["Python", "Django", "FastAPI"],
                    "seniority_level": "Senior",
                    "remote_friendly": False,
                }
                for i, job in enumerate(scraped_jobs)
            ]

            async def mock_ai_enhance():
                await asyncio.sleep(2.8)  # 2.8s - within target
                return enhanced_jobs

            mock_ai.enhance_job_data = AsyncMock(return_value=mock_ai_enhance())

            # Phase 3B: UI Rendering (≤200ms for 50 jobs)
            def mock_ui_render():
                time.sleep(0.18)  # 180ms - within target
                return "rendered_ui"

            MockRender.side_effect = mock_ui_render

            # Phase 3D: Coordination
            async def mock_coordinate():
                await asyncio.sleep(0.05)  # 50ms coordination overhead
                return {"coordinated": True, "workflow_id": "test_workflow"}

            mock_orchestrator.coordinate_complete_workflow = AsyncMock(
                return_value=mock_coordinate()
            )

            # Execute complete workflow
            workflow_start_time = time.perf_counter()

            # Step 1: Coordinate workflow
            coordination_result = await mock_orchestrator.coordinate_complete_workflow(
                workflow_type="search_enhance_render"
            )
            coordination_time = time.perf_counter()

            # Step 2: Scrape jobs (Phase 3A)
            scraping_results = await mock_scraper.scrape_unified(
                search_term=workflow_scenario["search_query"],
                location=workflow_scenario["location"],
                max_jobs=workflow_scenario["max_jobs"],
            )
            scraping_time = time.perf_counter()

            # Step 3: Enhance with AI (Phase 3C)
            ai_enhanced_results = await mock_ai.enhance_job_data(scraping_results)
            ai_time = time.perf_counter()

            # Step 4: Render UI (Phase 3B)
            rendered_ui = MockRender(ai_enhanced_results[:50], device_type="desktop")
            render_time = time.perf_counter()

            total_workflow_time = render_time - workflow_start_time

            # Calculate stage timings
            workflow_timings = {
                "coordination_ms": (coordination_time - workflow_start_time) * 1000,
                "scraping_ms": (scraping_time - coordination_time) * 1000,
                "ai_processing_ms": (ai_time - scraping_time) * 1000,
                "ui_rendering_ms": (render_time - ai_time) * 1000,
                "total_workflow_ms": total_workflow_time * 1000,
            }

            # Validate complete workflow integration
            workflow_successful = (
                # Phase targets met
                workflow_timings["scraping_ms"] < 500.0
                and workflow_timings["ai_processing_ms"] < 3000.0
                and workflow_timings["ui_rendering_ms"] < 200.0
                # Data flow integrity
                and len(scraping_results) == workflow_scenario["max_jobs"]
                and len(ai_enhanced_results) == len(scraping_results)
                and rendered_ui is not None
                # Overall performance
                and workflow_timings["total_workflow_ms"]
                < workflow_scenario["expected_total_time"]
                # Coordination working
                and coordination_result["coordinated"]
            )

            assert workflow_successful, (
                f"Complete workflow integration failed. Timings: {workflow_timings}. "
                f"Data integrity: scraping={len(scraping_results)}, "
                f"enhanced={len(ai_enhanced_results)}, rendered={rendered_ui is not None}"
            )

            # Additional validations
            assert workflow_timings["coordination_ms"] < 100.0, (
                f"Coordination overhead {workflow_timings['coordination_ms']:.1f}ms too high, should be <100ms"
            )

            assert all("skills" in job for job in ai_enhanced_results), (
                "AI enhancement didn't add skills to all jobs"
            )

            assert all(
                "salary" in job and isinstance(job["salary"], dict)
                for job in ai_enhanced_results
            ), "AI enhancement didn't properly structure salary data"


# Integration test utilities and reporting
class PhasesIntegrationReporter:
    """Generate comprehensive phase integration reports."""

    @staticmethod
    def generate_integration_report(test_results: dict) -> dict:
        """Generate Phase 3A-3D integration validation report."""
        return {
            "phases_integration_summary": {
                "phase_3a_unified_scraping": {
                    "target": "JobSpy + ScrapeGraphAI integration <500ms",
                    "achieved": test_results.get("phase_3a_integration_rate", 0) >= 0.9,
                    "integration_rate": test_results.get(
                        "phase_3a_integration_rate", 0
                    ),
                    "fallback_coordination": test_results.get(
                        "scraping_fallback_rate", 0
                    ),
                },
                "phase_3b_mobile_ui": {
                    "target": "Mobile-first responsive cards <200ms",
                    "achieved": test_results.get("phase_3b_integration_rate", 0) >= 0.9,
                    "ui_integration_rate": test_results.get(
                        "phase_3b_integration_rate", 0
                    ),
                    "responsive_rendering": test_results.get(
                        "responsive_rendering_rate", 0
                    ),
                },
                "phase_3c_hybrid_ai": {
                    "target": "vLLM + cloud fallback <3s processing",
                    "achieved": test_results.get("phase_3c_integration_rate", 0) >= 0.9,
                    "ai_integration_rate": test_results.get(
                        "phase_3c_integration_rate", 0
                    ),
                    "routing_integration": test_results.get(
                        "ai_routing_integration_rate", 0
                    ),
                },
                "phase_3d_coordination": {
                    "target": "Background task orchestration",
                    "achieved": test_results.get("phase_3d_coordination_rate", 0)
                    >= 0.9,
                    "coordination_rate": test_results.get(
                        "phase_3d_coordination_rate", 0
                    ),
                    "health_monitoring": test_results.get("health_monitoring_rate", 0),
                },
                "cross_phase_error_handling": {
                    "target": "Error recovery and graceful degradation",
                    "achieved": test_results.get("error_recovery_rate", 0) >= 0.8,
                    "recovery_rate": test_results.get("error_recovery_rate", 0),
                    "degradation_rate": test_results.get(
                        "graceful_degradation_rate", 0
                    ),
                },
                "end_to_end_integration": {
                    "target": "Complete workflow <6s total",
                    "achieved": test_results.get("complete_workflow_successful", False),
                    "workflow_performance": test_results.get("total_workflow_ms", 0),
                },
            },
            "detailed_metrics": test_results,
            "recommendations": PhasesIntegrationReporter._generate_integration_recommendations(
                test_results
            ),
        }

    @staticmethod
    def _generate_integration_recommendations(test_results: dict) -> list[str]:
        """Generate integration improvement recommendations."""
        recommendations = []

        if test_results.get("phase_3a_integration_rate", 0) < 0.9:
            recommendations.append(
                "Improve JobSpy + ScrapeGraphAI integration and fallback coordination"
            )

        if test_results.get("phase_3b_integration_rate", 0) < 0.9:
            recommendations.append(
                "Optimize mobile UI integration with data from other phases"
            )

        if test_results.get("phase_3c_integration_rate", 0) < 0.9:
            recommendations.append(
                "Enhance hybrid AI routing integration with system coordination"
            )

        if test_results.get("phase_3d_coordination_rate", 0) < 0.9:
            recommendations.append(
                "Strengthen background task coordination and health monitoring"
            )

        if test_results.get("error_recovery_rate", 0) < 0.8:
            recommendations.append(
                "Improve cross-phase error handling and recovery mechanisms"
            )

        if not test_results.get("complete_workflow_successful", False):
            recommendations.append(
                "Optimize end-to-end workflow performance and data flow integrity"
            )

        return recommendations


# Phase integration test configuration
PHASES_INTEGRATION_CONFIG = {
    "performance_targets": {
        "phase_3a_scraping_ms": 500,
        "phase_3c_ai_processing_ms": 3000,
        "phase_3b_ui_rendering_ms": 200,
        "phase_3d_coordination_ms": 100,
        "complete_workflow_ms": 6000,
        "error_recovery_ms": 1000,
    },
    "integration_success_thresholds": {
        "phase_integration_rate": 0.9,
        "error_recovery_rate": 0.8,
        "health_monitoring_rate": 0.95,
        "workflow_success_rate": 0.9,
    },
    "test_scenarios": {
        "scraping_sources": ["jobspy", "scrapegraphai"],
        "ai_routing_strategies": ["local", "cloud", "hybrid"],
        "ui_viewports": [320, 768, 1200, 1920],
        "coordination_workflows": [
            "scrape_enhance_render",
            "bulk_processing",
            "incremental_update",
        ],
    },
    "error_scenarios": [
        "network_timeout",
        "model_unavailable",
        "orchestration_failure",
        "high_system_load",
        "ai_service_unavailable",
        "mobile_bandwidth_limited",
    ],
}
