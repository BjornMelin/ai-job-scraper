"""System Coordination Module for AI Job Scraper.

This module provides comprehensive orchestration and coordination for all system components:
- Background Task Manager: Coordinate scraping with UI progress updates
- Service Orchestrator: Execute integrated workflows across all services
- Progress Tracker: Real-time status updates for background operations
- System Health Monitor: Monitor all services and their availability

Phase 3D implementation bringing together all components into production-ready workflows.
"""

from src.coordination.background_task_manager import BackgroundTaskManager
from src.coordination.progress_tracker import ProgressTracker
from src.coordination.service_orchestrator import ServiceOrchestrator
from src.coordination.system_health_monitor import SystemHealthMonitor

__all__ = [
    "BackgroundTaskManager",
    "ProgressTracker",
    "ServiceOrchestrator",
    "SystemHealthMonitor",
]
