"""Fragment Performance Monitoring Dashboard.

This module provides real-time monitoring of fragment performance metrics,
including update frequencies, render times, and optimization suggestions.
Implements performance tracking for the st.fragment() architecture optimization.

Key features:
- Real-time fragment performance metrics display
- Update frequency analysis and optimization suggestions
- Fragment health monitoring with error tracking
- Memory usage optimization insights
- Auto-refresh performance dashboard

This supports the 30% performance improvement target through intelligent
monitoring and optimization recommendations.
"""

import logging

from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

from src.ui.utils.fragment_orchestrator import get_fragment_orchestrator

logger = logging.getLogger(__name__)


@st.fragment(run_every="5s")
def render_fragment_performance_monitor() -> None:
    """Fragment-based performance monitoring dashboard.

    This fragment displays real-time performance metrics for all active
    fragments in the application, providing insights for optimization.
    """
    try:
        # Get fragment orchestrator
        orchestrator = get_fragment_orchestrator()

        # Get performance summary
        perf_summary = orchestrator.get_orchestration_summary()

        if "error" in perf_summary:
            st.error(f"⚠️ Performance monitoring error: {perf_summary['error']}")
            return

        # Performance header
        st.markdown("### ⚡ Fragment Performance Monitor")

        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_fragments = perf_summary.get("registered_fragments", 0)
            st.metric("Active Fragments", total_fragments)

        with col2:
            coord_stats = perf_summary.get("coordination_stats", {})
            total_updates = coord_stats.get("total_updates", 0)
            st.metric("Total Updates", f"{total_updates:,}")

        with col3:
            perf_data = perf_summary.get("performance_summary", {})
            avg_time = perf_data.get("average_update_time", 0)
            st.metric("Avg Update Time", f"{avg_time:.3f}s")

        with col4:
            total_errors = coord_stats.get("total_errors", 0)
            error_rate = (total_errors / max(total_updates, 1)) * 100
            st.metric("Error Rate", f"{error_rate:.1f}%")

        # Performance insights
        _render_performance_insights(perf_summary)

        # Optimization suggestions
        _render_optimization_suggestions(perf_summary)

        # Fragment health status
        _render_fragment_health(perf_summary)

    except Exception as e:
        logger.exception("Error in fragment performance monitor")
        st.error(f"⚠️ Performance monitoring temporarily unavailable: {str(e)[:100]}...")


def _render_performance_insights(perf_summary: dict[str, Any]) -> None:
    """Render performance insights section.

    Args:
        perf_summary: Performance summary data from orchestrator.
    """
    st.markdown("#### 📊 Performance Insights")

    perf_data = perf_summary.get("performance_summary", {})

    # Performance highlights
    col1, col2 = st.columns(2)

    with col1:
        # Slowest fragment
        slowest = perf_data.get("slowest_fragment")
        if slowest:
            st.warning(
                f"🐌 Slowest: {slowest['id'][:20]}... ({slowest['avg_time']:.3f}s)"
            )
        else:
            st.info("🚀 All fragments performing well!")

    with col2:
        # Most active fragment
        most_active = perf_data.get("most_active_fragment")
        if most_active:
            st.info(
                f"🔥 Most Active: {most_active['id'][:20]}... "
                f"({most_active['update_count']} updates)"
            )
        else:
            st.info("📊 No activity data available")


def _render_optimization_suggestions(perf_summary: dict[str, Any]) -> None:
    """Render optimization suggestions section.

    Args:
        perf_summary: Performance summary data from orchestrator.
    """
    st.markdown("#### 🎯 Optimization Suggestions")

    suggestions = perf_summary.get("optimization_suggestions", {})

    if not suggestions:
        st.success("✨ All fragments optimally configured!")
        return

    # Group suggestions by recommended interval
    interval_groups = {}
    for fragment_id, suggested_interval in suggestions.items():
        if suggested_interval not in interval_groups:
            interval_groups[suggested_interval] = []
        interval_groups[suggested_interval].append(fragment_id)

    # Display suggestions grouped by interval
    for interval, fragments in interval_groups.items():
        with st.expander(
            f"🔧 Suggested interval: {interval} ({len(fragments)} fragments)"
        ):
            for fragment_id in fragments:
                st.caption(f"• {fragment_id}")


def _render_fragment_health(perf_summary: dict[str, Any]) -> None:
    """Render fragment health status section.

    Args:
        perf_summary: Performance summary data from orchestrator.
    """
    st.markdown("#### 🏥 Fragment Health")

    error_summary = perf_summary.get("error_summary", {})

    total_errors = error_summary.get("total_fragments_with_errors", 0)

    if total_errors == 0:
        st.success("💚 All fragments healthy!")
        return

    # Error overview
    st.error(f"⚠️ {total_errors} fragment(s) with errors")

    # Recent errors
    recent_errors = error_summary.get("recent_errors", [])
    if recent_errors:
        st.markdown("**Recent Errors:**")
        for error in recent_errors[-3:]:  # Show last 3 errors
            fragment_id = error["fragment_id"][:30]
            error_msg = (
                error["error"][:50] + "..."
                if len(error["error"]) > 50
                else error["error"]
            )
            timestamp = error["timestamp"][:19]  # Remove microseconds
            st.caption(f"• {fragment_id}: {error_msg} ({timestamp})")

    # Fragments with multiple failures
    multiple_failures = error_summary.get("fragments_with_multiple_failures", [])
    if multiple_failures:
        st.warning(f"🚨 {len(multiple_failures)} fragment(s) with multiple failures")
        with st.expander("View failing fragments"):
            for fragment_id in multiple_failures:
                st.caption(f"• {fragment_id}")


@st.fragment(run_every="10s")
def render_fragment_coordination_status() -> None:
    """Fragment for displaying coordination and communication status.

    Shows real-time fragment communication and coordination metrics.
    """
    try:
        # Get coordination data
        if "fragment_orchestrator" not in st.session_state:
            st.info("🔄 Fragment orchestrator not initialized")
            return

        orchestrator_data = st.session_state.get("fragment_orchestrator", {})

        st.markdown("#### 🔗 Fragment Coordination")

        # Coordination stats
        coord_stats = orchestrator_data.get("coordination_stats", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            total_messages = coord_stats.get("total_messages", 0)
            st.metric("Messages", f"{total_messages:,}")

        with col2:
            total_updates = coord_stats.get("total_updates", 0)
            st.metric("Updates", f"{total_updates:,}")

        with col3:
            total_errors = coord_stats.get("total_errors", 0)
            st.metric("Errors", total_errors)

        # Communication status
        comm_data = st.session_state.get("fragment_communication", {})
        if comm_data:
            recent_messages = len(
                [
                    msg
                    for msg in comm_data.values()
                    if msg.get("timestamp", "")
                    > (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
                ]
            )
            if recent_messages > 0:
                st.success(f"📡 {recent_messages} recent messages")
            else:
                st.info("📡 Communication idle")
        else:
            st.info("📡 No communication data")

    except Exception as e:
        logger.exception("Error in fragment coordination status")
        st.error(f"⚠️ Coordination status error: {str(e)[:50]}...")


def render_fragment_performance_dashboard() -> None:
    """Render the complete fragment performance dashboard.

    This function combines all performance monitoring fragments into a
    comprehensive dashboard for fragment performance optimization.
    """
    st.markdown("---")

    # Performance monitoring fragment
    render_fragment_performance_monitor()

    st.markdown("---")

    # Coordination status fragment
    render_fragment_coordination_status()

    # Performance tips
    with st.expander("💡 Performance Optimization Tips"):
        st.markdown("""
        **Fragment Performance Best Practices:**

        1. **Optimal Intervals**: Use 1-2s for active data, 5-10s for slower updates
        2. **Early Returns**: Exit fragments early when no data changes
        3. **Static Rendering**: Use static components for completed/inactive states
        4. **Error Boundaries**: Implement proper error handling in fragments
        5. **Memory Management**: Clear unnecessary session state regularly
        6. **Selective Updates**: Use `st.rerun(scope="fragment")` for isolated updates

        **Current Optimizations Applied:**
        - Job cards: 2s intervals with intelligent refresh control
        - Progress dashboard: 1s intervals when active, early exit when idle
        - Notifications: 3s intervals with selective rendering
        - Static cards for completed tasks to reduce CPU usage
        """)
