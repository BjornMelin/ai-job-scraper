"""Progress Tracking System for real-time status updates and ETA estimation.

This module provides comprehensive progress tracking for background operations
with real-time status updates, percentage calculation, and ETA estimation.
Integrates with responsive mobile cards for live updates.

Key Features:
- Real-time progress percentage calculation
- ETA estimation based on historical performance
- Integration with mobile-first responsive UI updates
- Memory-efficient progress history management
- Production-ready performance monitoring
"""

import asyncio
import logging
import time
import uuid

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a specific time."""

    timestamp: datetime
    progress_percentage: float
    message: str
    phase: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "progress_percentage": self.progress_percentage,
            "message": self.message,
            "phase": self.phase,
            "metadata": self.metadata,
        }


@dataclass
class ProgressEstimate:
    """Progress estimation with ETA calculation."""

    current_progress: float
    estimated_total_duration: float
    estimated_time_remaining: float
    estimated_completion_time: datetime
    confidence_level: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_progress": self.current_progress,
            "estimated_total_duration": self.estimated_total_duration,
            "estimated_time_remaining": self.estimated_time_remaining,
            "estimated_completion_time": self.estimated_completion_time.isoformat(),
            "confidence_level": self.confidence_level,
        }


class ProgressTracker:
    """Real-time progress tracking with ETA estimation and UI coordination.

    This class provides comprehensive progress tracking for background operations:
    - Real-time progress updates with percentage calculation
    - ETA estimation based on historical performance data
    - Integration with responsive mobile cards
    - Memory-efficient progress history management
    - Performance analytics and metrics collection

    Architecture:
    - Tracks progress snapshots with timestamps
    - Calculates ETAs using linear regression on recent progress
    - Provides real-time updates to UI components
    - Manages memory with configurable history limits
    """

    def __init__(
        self,
        tracker_id: str | None = None,
        max_history_size: int = 100,
        eta_calculation_window: int = 10,
    ) -> None:
        """Initialize the progress tracker.

        Args:
            tracker_id: Unique identifier for this tracker
            max_history_size: Maximum number of progress snapshots to keep
            eta_calculation_window: Number of recent snapshots to use for ETA calculation
        """
        self.tracker_id = tracker_id or str(uuid.uuid4())
        self.max_history_size = max_history_size
        self.eta_calculation_window = min(eta_calculation_window, max_history_size)
        self.logger = logging.getLogger(f"{__name__}.{self.tracker_id}")

        # Progress tracking state
        self._progress_history: deque[ProgressSnapshot] = deque(maxlen=max_history_size)
        self._start_time = datetime.now(UTC)
        self._current_phase = "initializing"
        self._is_active = True
        self._completion_time: datetime | None = None

        # Performance metrics
        self._metrics = {
            "total_updates": 0,
            "phases_completed": 0,
            "average_phase_duration": 0.0,
            "accuracy_score": 0.0,
            "updates_per_minute": 0.0,
        }

        # ETA calculation state
        self._last_eta_calculation = 0.0
        self._cached_estimate: ProgressEstimate | None = None

        self.logger.info(
            "✅ ProgressTracker initialized - ID: %s, History: %d, ETA Window: %d",
            self.tracker_id,
            max_history_size,
            eta_calculation_window,
        )

    def update_progress(
        self,
        progress_percentage: float,
        message: str,
        phase: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProgressSnapshot:
        """Update progress with new information.

        Args:
            progress_percentage: Current progress (0.0-100.0)
            message: Human-readable progress message
            phase: Current phase of operation
            metadata: Additional metadata for this progress update

        Returns:
            ProgressSnapshot of the recorded progress
        """
        if not self._is_active:
            self.logger.warning("Progress tracker is inactive, ignoring update")
            return self._progress_history[-1] if self._progress_history else None

        # Validate progress percentage
        progress_percentage = max(0.0, min(100.0, progress_percentage))

        # Update current phase if provided
        if phase and phase != self._current_phase:
            self.logger.info(
                "📋 Phase transition: %s -> %s", self._current_phase, phase
            )
            self._current_phase = phase
            self._metrics["phases_completed"] += 1

        # Create progress snapshot
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(UTC),
            progress_percentage=progress_percentage,
            message=message,
            phase=self._current_phase,
            metadata=metadata or {},
        )

        # Add to history
        self._progress_history.append(snapshot)
        self._metrics["total_updates"] += 1

        # Update session state for UI coordination
        self._update_ui_state(snapshot)

        # Clear cached estimate to force recalculation
        self._cached_estimate = None

        # Mark as completed if 100%
        if progress_percentage >= 100.0:
            self._complete_tracking()

        self.logger.debug(
            "📊 Progress updated: %.1f%% - %s (Phase: %s)",
            progress_percentage,
            message,
            self._current_phase,
        )

        return snapshot

    def _update_ui_state(self, snapshot: ProgressSnapshot) -> None:
        """Update Streamlit session state for UI coordination."""
        try:
            # Update progress tracker state
            if "progress_trackers" not in st.session_state:
                st.session_state.progress_trackers = {}

            st.session_state.progress_trackers[self.tracker_id] = {
                "current_snapshot": snapshot.to_dict(),
                "start_time": self._start_time.isoformat(),
                "is_active": self._is_active,
                "current_phase": self._current_phase,
                "metrics": self._metrics.copy(),
            }

            # Update for mobile card integration
            if "mobile_progress_updates" not in st.session_state:
                st.session_state.mobile_progress_updates = {}

            st.session_state.mobile_progress_updates[self.tracker_id] = {
                "progress": snapshot.progress_percentage,
                "message": snapshot.message,
                "phase": snapshot.phase,
                "timestamp": snapshot.timestamp.isoformat(),
                "eta": self.get_progress_estimate().to_dict()
                if self._progress_history
                else None,
            }

        except Exception as e:
            self.logger.warning("Failed to update UI state: %s", e)

    def get_progress_estimate(self) -> ProgressEstimate | None:
        """Calculate progress estimate with ETA prediction.

        Returns:
            ProgressEstimate with ETA calculation, or None if insufficient data
        """
        if not self._progress_history or not self._is_active:
            return None

        current_time = time.time()

        # Use cached estimate if recent (within 5 seconds)
        if self._cached_estimate and current_time - self._last_eta_calculation < 5.0:
            return self._cached_estimate

        # Get recent snapshots for ETA calculation
        recent_snapshots = list(self._progress_history)[-self.eta_calculation_window :]
        if len(recent_snapshots) < 2:
            return None

        current_snapshot = recent_snapshots[-1]
        current_progress = current_snapshot.progress_percentage

        # Calculate progress rate using linear regression on recent data
        progress_rate = self._calculate_progress_rate(recent_snapshots)

        if progress_rate <= 0:
            # If no progress or going backwards, use simple time-based estimate
            elapsed_time = (
                current_snapshot.timestamp - self._start_time
            ).total_seconds()
            if current_progress > 0:
                estimated_total_duration = (elapsed_time / current_progress) * 100.0
                estimated_time_remaining = estimated_total_duration - elapsed_time
                confidence_level = 0.3  # Low confidence for simple estimate
            else:
                estimated_total_duration = 300.0  # Default 5 minutes
                estimated_time_remaining = estimated_total_duration
                confidence_level = 0.1  # Very low confidence
        else:
            # Use progress rate for estimation
            remaining_progress = 100.0 - current_progress
            estimated_time_remaining = remaining_progress / progress_rate

            elapsed_time = (
                current_snapshot.timestamp - self._start_time
            ).total_seconds()
            estimated_total_duration = elapsed_time + estimated_time_remaining

            # Calculate confidence based on data consistency
            confidence_level = min(
                0.9, len(recent_snapshots) / self.eta_calculation_window
            )

        # Create estimate
        estimated_completion_time = datetime.now(UTC) + timedelta(
            seconds=estimated_time_remaining
        )

        estimate = ProgressEstimate(
            current_progress=current_progress,
            estimated_total_duration=estimated_total_duration,
            estimated_time_remaining=max(0.0, estimated_time_remaining),
            estimated_completion_time=estimated_completion_time,
            confidence_level=confidence_level,
        )

        # Cache the estimate
        self._cached_estimate = estimate
        self._last_eta_calculation = current_time

        return estimate

    def _calculate_progress_rate(self, snapshots: list[ProgressSnapshot]) -> float:
        """Calculate progress rate (percentage per second) using linear regression.

        Args:
            snapshots: Recent progress snapshots for calculation

        Returns:
            Progress rate in percentage per second
        """
        if len(snapshots) < 2:
            return 0.0

        # Calculate simple linear regression
        n = len(snapshots)
        base_time = snapshots[0].timestamp

        sum_x = sum((s.timestamp - base_time).total_seconds() for s in snapshots)
        sum_y = sum(s.progress_percentage for s in snapshots)
        sum_xy = sum(
            (s.timestamp - base_time).total_seconds() * s.progress_percentage
            for s in snapshots
        )
        sum_x2 = sum((s.timestamp - base_time).total_seconds() ** 2 for s in snapshots)

        # Calculate slope (progress rate)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return max(0.0, slope)  # Ensure non-negative rate

    def get_current_progress(self) -> ProgressSnapshot | None:
        """Get the current progress snapshot.

        Returns:
            Most recent ProgressSnapshot or None if no progress recorded
        """
        return self._progress_history[-1] if self._progress_history else None

    def get_progress_history(self, limit: int | None = None) -> list[ProgressSnapshot]:
        """Get progress history.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of ProgressSnapshot objects
        """
        snapshots = list(self._progress_history)
        if limit:
            snapshots = snapshots[-limit:]
        return snapshots

    def get_phase_summary(self) -> dict[str, Any]:
        """Get summary of phases and their performance.

        Returns:
            Dictionary with phase performance data
        """
        phase_data = {}

        for snapshot in self._progress_history:
            phase = snapshot.phase
            if phase not in phase_data:
                phase_data[phase] = {
                    "first_seen": snapshot.timestamp,
                    "last_seen": snapshot.timestamp,
                    "progress_range": [
                        snapshot.progress_percentage,
                        snapshot.progress_percentage,
                    ],
                    "message_count": 0,
                }

            phase_info = phase_data[phase]
            phase_info["last_seen"] = snapshot.timestamp
            phase_info["progress_range"][0] = min(
                phase_info["progress_range"][0], snapshot.progress_percentage
            )
            phase_info["progress_range"][1] = max(
                phase_info["progress_range"][1], snapshot.progress_percentage
            )
            phase_info["message_count"] += 1

        # Calculate durations
        for phase, info in phase_data.items():
            info["duration"] = (info["last_seen"] - info["first_seen"]).total_seconds()
            info["progress_span"] = (
                info["progress_range"][1] - info["progress_range"][0]
            )

        return phase_data

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics.

        Returns:
            Dictionary with performance data and analytics
        """
        if not self._progress_history:
            return self._metrics.copy()

        current_time = datetime.now(UTC)
        elapsed_time = (current_time - self._start_time).total_seconds()

        # Calculate updates per minute
        if elapsed_time > 0:
            self._metrics["updates_per_minute"] = (
                self._metrics["total_updates"] / elapsed_time
            ) * 60.0

        # Calculate average phase duration
        phase_summary = self.get_phase_summary()
        if phase_summary:
            total_phase_time = sum(info["duration"] for info in phase_summary.values())
            self._metrics["average_phase_duration"] = total_phase_time / len(
                phase_summary
            )

        # Add current state metrics
        metrics = self._metrics.copy()
        metrics.update(
            {
                "tracker_id": self.tracker_id,
                "elapsed_time": elapsed_time,
                "is_active": self._is_active,
                "current_phase": self._current_phase,
                "history_size": len(self._progress_history),
                "completion_time": self._completion_time.isoformat()
                if self._completion_time
                else None,
            }
        )

        return metrics

    def _complete_tracking(self) -> None:
        """Mark tracking as completed."""
        if self._is_active:
            self._is_active = False
            self._completion_time = datetime.now(UTC)

            # Final UI update
            if self._progress_history:
                self._update_ui_state(self._progress_history[-1])

            self.logger.info(
                "🎉 Progress tracking completed - Duration: %.2fs, Updates: %d",
                (self._completion_time - self._start_time).total_seconds(),
                self._metrics["total_updates"],
            )

    def reset(self) -> None:
        """Reset progress tracker to initial state."""
        self._progress_history.clear()
        self._start_time = datetime.now(UTC)
        self._current_phase = "initializing"
        self._is_active = True
        self._completion_time = None
        self._cached_estimate = None

        # Reset metrics
        self._metrics = {
            "total_updates": 0,
            "phases_completed": 0,
            "average_phase_duration": 0.0,
            "accuracy_score": 0.0,
            "updates_per_minute": 0.0,
        }

        self.logger.info("🔄 Progress tracker reset")

    async def stream_progress_updates(self, update_interval: float = 1.0):
        """Stream progress updates for real-time monitoring.

        Args:
            update_interval: Time between updates in seconds

        Yields:
            ProgressSnapshot objects with current progress
        """
        while self._is_active and self._progress_history:
            current_snapshot = self._progress_history[-1]
            yield current_snapshot

            await asyncio.sleep(update_interval)

        # Yield final snapshot if completed
        if self._progress_history:
            yield self._progress_history[-1]


class ProgressTrackingManager:
    """Manager for multiple progress trackers with coordination."""

    def __init__(self):
        """Initialize the progress tracking manager."""
        self._trackers: dict[str, ProgressTracker] = {}
        self.logger = logging.getLogger(f"{__name__}.ProgressTrackingManager")

    def create_tracker(
        self, tracker_id: str | None = None, **kwargs
    ) -> ProgressTracker:
        """Create a new progress tracker.

        Args:
            tracker_id: Optional tracker ID, generates UUID if None
            **kwargs: Additional arguments for ProgressTracker

        Returns:
            New ProgressTracker instance
        """
        if tracker_id is None:
            tracker_id = str(uuid.uuid4())

        if tracker_id in self._trackers:
            self.logger.warning(
                "Tracker %s already exists, returning existing", tracker_id
            )
            return self._trackers[tracker_id]

        tracker = ProgressTracker(tracker_id=tracker_id, **kwargs)
        self._trackers[tracker_id] = tracker

        self.logger.info("✅ Created progress tracker: %s", tracker_id)
        return tracker

    def get_tracker(self, tracker_id: str) -> ProgressTracker | None:
        """Get existing progress tracker by ID.

        Args:
            tracker_id: Tracker identifier

        Returns:
            ProgressTracker instance or None if not found
        """
        return self._trackers.get(tracker_id)

    def get_all_trackers(self) -> dict[str, ProgressTracker]:
        """Get all managed progress trackers.

        Returns:
            Dictionary of tracker_id -> ProgressTracker
        """
        return self._trackers.copy()

    def remove_tracker(self, tracker_id: str) -> bool:
        """Remove a progress tracker.

        Args:
            tracker_id: Tracker identifier to remove

        Returns:
            True if tracker was removed, False if not found
        """
        if tracker_id in self._trackers:
            del self._trackers[tracker_id]
            self.logger.info("🗑️ Removed progress tracker: %s", tracker_id)
            return True
        return False

    def cleanup_completed_trackers(self) -> int:
        """Remove all completed/inactive trackers.

        Returns:
            Number of trackers cleaned up
        """
        to_remove = [
            tracker_id
            for tracker_id, tracker in self._trackers.items()
            if not tracker._is_active
        ]

        for tracker_id in to_remove:
            del self._trackers[tracker_id]

        if to_remove:
            self.logger.info("🧹 Cleaned up %d completed trackers", len(to_remove))

        return len(to_remove)


# Module-level singleton for easy access
_progress_tracking_manager: ProgressTrackingManager | None = None


def get_progress_tracking_manager() -> ProgressTrackingManager:
    """Get singleton instance of ProgressTrackingManager.

    Returns:
        ProgressTrackingManager singleton instance
    """
    global _progress_tracking_manager
    if _progress_tracking_manager is None:
        _progress_tracking_manager = ProgressTrackingManager()
    return _progress_tracking_manager


def reset_progress_tracking_manager() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _progress_tracking_manager
    _progress_tracking_manager = None
