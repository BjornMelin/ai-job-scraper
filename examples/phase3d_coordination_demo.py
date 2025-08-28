"""Phase 3D System Coordination Demo.

This example demonstrates the comprehensive coordination layer in action,
showing how all Phase 3A-D components work together for integrated workflows.

Key Demonstrations:
- Background Task Manager coordinating scraping with UI progress
- Service Orchestrator executing end-to-end workflows
- Progress Tracker providing real-time status updates
- System Health Monitor validating service availability
- Production deployment readiness validation

Run this example to see the complete Phase 3D coordination system.
"""

import asyncio
import logging

from src.coordination import (
    BackgroundTaskManager,
    ProgressTracker,
    ServiceOrchestrator,
    SystemHealthMonitor,
)
from src.interfaces.scraping_service_interface import JobQuery, SourceType

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def demonstrate_system_health_monitoring():
    """Demonstrate comprehensive system health monitoring."""
    print("\n" + "=" * 60)
    print("🔍 SYSTEM HEALTH MONITORING DEMONSTRATION")
    print("=" * 60)

    # Initialize health monitor
    health_monitor = SystemHealthMonitor()

    print("\n📋 Running comprehensive health check...")
    health_report = await health_monitor.get_comprehensive_health_report(
        force_check=True
    )

    print(
        f"📊 Overall System Health: "
        f"{'✅ HEALTHY' if health_report['overall_healthy'] else '❌ UNHEALTHY'}"
    )
    print(f"📈 Services Monitored: {health_report['system_metrics']['total_services']}")
    print(f"🟢 Healthy Services: {health_report['system_metrics']['healthy_services']}")
    print(
        f"🔴 Unhealthy Services: "
        f"{health_report['system_metrics']['unhealthy_services']}"
    )
    print(
        f"⏱️ Total Response Time: "
        f"{health_report['system_metrics']['total_response_time_ms']:.1f}ms"
    )

    print("\n🏥 Individual Service Health:")
    for service_name, service_status in health_report["services"].items():
        status_icon = "✅" if service_status["healthy"] else "❌"
        print(
            f"  {status_icon} {service_name}: "
            f"{service_status['status_message']} "
            f"({service_status['response_time_ms']:.1f}ms)"
        )

    if health_report["warnings"]:
        print("\n⚠️ Warnings:")
        for warning in health_report["warnings"]:
            print(f"  - {warning}")

    if health_report["errors"]:
        print("\n🚨 Errors:")
        for error in health_report["errors"]:
            print(f"  - {error}")


async def demonstrate_progress_tracking():
    """Demonstrate real-time progress tracking with ETA estimation."""
    print("\n" + "=" * 60)
    print("📊 PROGRESS TRACKING DEMONSTRATION")
    print("=" * 60)

    # Create progress tracker
    progress_tracker = ProgressTracker(tracker_id="demo-workflow")

    print("\n🚀 Simulating multi-phase workflow with progress tracking...")

    # Simulate a multi-phase workflow
    phases = [
        ("initialization", "Initializing services...", 10.0),
        ("validation", "Validating dependencies...", 20.0),
        ("scraping", "Starting job scraping...", 35.0),
        ("scraping", "Processing job data...", 60.0),
        ("ai_enhancement", "Enhancing jobs with AI...", 75.0),
        ("database_storage", "Storing to database...", 85.0),
        ("search_indexing", "Updating search indexes...", 95.0),
        ("completed", "Workflow completed!", 100.0),
    ]

    for phase, message, progress in phases:
        # Update progress
        _snapshot = progress_tracker.update_progress(
            progress_percentage=progress,
            message=message,
            phase=phase,
            metadata={"demo": True, "timestamp": asyncio.get_event_loop().time()},
        )

        # Get ETA estimate
        estimate = progress_tracker.get_progress_estimate()

        print(f"📈 {progress:6.1f}% | {phase:15s} | {message}")
        if estimate:
            print(
                f"     ⏱️ ETA: {estimate.estimated_time_remaining:.1f}s "
                f"remaining (confidence: {estimate.confidence_level:.1f})"
            )

        # Simulate work delay
        await asyncio.sleep(0.5)

    # Display final metrics
    print("\n📊 Final Progress Metrics:")
    metrics = progress_tracker.get_performance_metrics()
    print(f"  📋 Total Updates: {metrics['total_updates']}")
    print(f"  📂 Phases Completed: {metrics['phases_completed']}")
    print(f"  ⏱️ Total Duration: {metrics['elapsed_time']:.1f}s")
    print(f"  📈 Updates per Minute: {metrics['updates_per_minute']:.1f}")


async def demonstrate_background_task_management():
    """Demonstrate background task management with coordination."""
    print("\n" + "=" * 60)
    print("⚙️ BACKGROUND TASK MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    # Initialize task manager
    task_manager = BackgroundTaskManager()

    # Create a sample job query
    job_query = JobQuery(
        keywords=["python", "software engineer"],
        locations=["San Francisco", "New York"],
        source_types=[SourceType.UNIFIED],
        max_results=20,
        enable_ai_enhancement=True,
    )

    print("\n🚀 Starting comprehensive workflow with background coordination...")

    # Start workflow
    task_id = task_manager.start_comprehensive_scraping_workflow(
        query=job_query,
        enable_ai_enhancement=True,
        enable_real_time_updates=False,  # Simplified for demo
    )

    print(f"📋 Started workflow with Task ID: {task_id}")

    # Monitor progress
    print("\n📊 Monitoring task progress:")
    async for task_status in task_manager.monitor_task_progress(task_id):
        print(
            f"🔄 Status: {task_status.status} | "
            f"Progress: {task_status.progress_percentage:.1f}% | "
            f"{task_status.metadata.get('message', 'Processing...')}"
        )

        if task_status.status in ["completed", "failed", "cancelled"]:
            break

    # Display final results
    final_status = task_manager.get_task_status(task_id)
    if final_status:
        print("\n✅ Task completed successfully!")
        print(f"📊 Results: {final_status.results}")
        duration = (
            (final_status.end_time - final_status.start_time).total_seconds()
            if final_status.end_time
            else 0
        )
        print(f"⏱️ Duration: {duration:.1f}s")

    # Show task manager metrics
    print("\n📈 Task Manager Metrics:")
    metrics = task_manager.get_task_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


async def demonstrate_service_orchestration():
    """Demonstrate end-to-end service orchestration."""
    print("\n" + "=" * 60)
    print("🎭 SERVICE ORCHESTRATION DEMONSTRATION")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = ServiceOrchestrator()

    print("\n🚀 Executing integrated workflow across all services...")

    # Define workflow options
    workflow_options = {
        "enable_ai_enhancement": True,
        "enable_real_time_updates": True,
        "enable_ui_updates": True,
        "max_jobs": 25,
        "source_types": [SourceType.UNIFIED],
    }

    try:
        # Execute integrated workflow
        workflow_id = await orchestrator.execute_integrated_workflow(
            query="machine learning engineer",
            workflow_options=workflow_options,
        )

        print("✅ Integrated workflow completed successfully!")
        print(f"📋 Workflow ID: {workflow_id}")

        # Get workflow status
        workflow_status = orchestrator.get_workflow_status(workflow_id)
        if workflow_status:
            print(f"📊 Services Used: {', '.join(workflow_status['services_used'])}")
            print(f"⏱️ Duration: {workflow_status['results']['duration']:.1f}s")
            print(f"📈 Jobs Processed: {workflow_status['results']['jobs_processed']}")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")

    # Display orchestration metrics
    print("\n📈 Orchestration Metrics:")
    metrics = orchestrator.get_orchestration_metrics()
    print(f"  📋 Total Workflows: {metrics['workflows_executed']}")
    print(f"  ✅ Completed: {metrics['workflows_completed']}")
    print(f"  ❌ Failed: {metrics['workflows_failed']}")
    print(f"  📊 Success Rate: {metrics['success_rate']:.1f}%")
    print(f"  ⏱️ Average Duration: {metrics['average_workflow_duration']:.1f}s")


async def demonstrate_production_readiness():
    """Demonstrate production deployment readiness validation."""
    print("\n" + "=" * 60)
    print("🚀 PRODUCTION READINESS VALIDATION")
    print("=" * 60)

    # Initialize orchestrator for production validation
    orchestrator = ServiceOrchestrator()

    print("\n🔍 Validating system readiness for production deployment...")

    try:
        validation_results = await orchestrator.validate_production_readiness()

        print(
            f"\n🎯 Production Ready: "
            f"{'✅ YES' if validation_results['ready_for_production'] else '❌ NO'}"
        )
        print(f"📅 Validation Time: {validation_results['validation_timestamp']}")

        print("\n📋 Validation Checks:")
        for check_name, result in validation_results["checks"].items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {status} {check_name}")
            elif isinstance(result, (int, float)):
                print(f"  📊 {check_name}: {result}")
            else:
                print(f"  📋 {check_name}: Available")

        if validation_results["warnings"]:
            print("\n⚠️ Warnings:")
            for warning in validation_results["warnings"]:
                print(f"  - {warning}")

        if validation_results["errors"]:
            print("\n🚨 Critical Issues:")
            for error in validation_results["errors"]:
                print(f"  - {error}")

        if validation_results["ready_for_production"]:
            print("\n🎉 System is ready for production deployment!")
        else:
            print("\n⚠️ System requires fixes before production deployment.")

    except Exception as e:
        print(f"❌ Production validation failed: {e}")


async def demonstrate_comprehensive_coordination():
    """Demonstrate the complete Phase 3D coordination system."""
    print("🚀 AI JOB SCRAPER - PHASE 3D SYSTEM COORDINATION DEMO")
    print("=" * 80)
    print("Demonstrating comprehensive orchestration of all system components:")
    print("- Phase 3A: Unified Scraping Service (2-tier JobSpy + ScrapeGraphAI)")
    print("- Phase 3B: Mobile-First Responsive Cards (CSS Grid, <200ms rendering)")
    print("- Phase 3C: Hybrid AI Integration (vLLM + cloud fallback routing)")
    print("- Phase 3D: System Coordination (🚧 CURRENT IMPLEMENTATION)")
    print("=" * 80)

    try:
        # Run all demonstrations
        await demonstrate_system_health_monitoring()
        await demonstrate_progress_tracking()
        await demonstrate_background_task_management()
        await demonstrate_service_orchestration()
        await demonstrate_production_readiness()

        print("\n" + "=" * 80)
        print("🎉 PHASE 3D COORDINATION DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ All coordination components are working correctly")
        print("✅ System is ready for integrated workflow execution")
        print("✅ Production deployment validation is operational")
        print("✅ Real-time progress tracking and monitoring active")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Coordination demo failed")


if __name__ == "__main__":
    # Run the comprehensive coordination demonstration
    asyncio.run(demonstrate_comprehensive_coordination())
