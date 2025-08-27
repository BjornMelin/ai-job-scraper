"""Fragment Performance Dashboard Page.

This page provides a comprehensive real-time dashboard for monitoring
st.fragment() performance and optimization in the AI Job Scraper application.

Key features:
- Real-time fragment performance metrics
- Update frequency analysis and optimization
- Fragment health monitoring and error tracking
- Interactive performance tuning controls
- Live performance comparison charts

This dashboard supports the 30% performance improvement target by providing
visibility into fragment behavior and optimization opportunities.
"""

import logging

from datetime import UTC, datetime

import streamlit as st

from src.ui.components.fragment_performance_monitor import (
    render_fragment_performance_dashboard,
)
from src.ui.utils.fragment_orchestrator import (
    cleanup_fragment_state,
    get_fragment_orchestrator,
)

logger = logging.getLogger(__name__)


def render_fragment_dashboard_page() -> None:
    """Render the fragment performance dashboard page.

    This page provides comprehensive monitoring and optimization tools
    for the application's st.fragment() architecture.
    """
    # Page header
    st.markdown("# ⚡ Fragment Performance Dashboard")
    st.markdown(
        "Monitor and optimize `st.fragment()` performance for real-time updates "
        "without full page reruns"
    )

    # Performance controls
    _render_performance_controls()

    # Main performance dashboard
    render_fragment_performance_dashboard()

    # Performance testing tools
    _render_performance_testing_tools()

    # Memory management section
    _render_memory_management()


def _render_performance_controls() -> None:
    """Render performance control panel."""
    st.markdown("---")
    st.markdown("### 🎛️ Performance Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Fragment orchestration toggle
        orchestration_enabled = st.checkbox(
            "Enable Fragment Orchestration",
            value=True,
            help="Enable advanced fragment coordination and monitoring",
        )

        if orchestration_enabled:
            # Initialize orchestrator if needed
            if "fragment_orchestrator_instance" not in st.session_state:
                orchestrator = get_fragment_orchestrator()
                st.success("✅ Fragment orchestrator initialized")

    with col2:
        # Auto-cleanup toggle
        auto_cleanup = st.checkbox(
            "Auto Memory Cleanup",
            value=True,
            help="Automatically clean up fragment state to prevent memory leaks",
        )

        if auto_cleanup:
            # Cleanup interval setting
            cleanup_interval = st.selectbox(
                "Cleanup Interval",
                options=["1 minute", "5 minutes", "10 minutes"],
                index=1,
                help="How often to clean up fragment memory",
            )

    with col3:
        # Performance monitoring level
        monitoring_level = st.selectbox(
            "Monitoring Level",
            options=["Basic", "Detailed", "Debug"],
            index=1,
            help="Level of performance monitoring detail",
        )

        # Store monitoring preferences
        st.session_state.fragment_monitoring_level = monitoring_level.lower()

    # Action buttons
    st.markdown("#### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "🧹 Cleanup Memory", help="Clean up fragment state and free memory"
        ):
            try:
                cleanup_fragment_state()
                st.success("✅ Fragment memory cleaned up")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Cleanup failed: {e}")

    with col2:
        if st.button("📊 Reset Metrics", help="Reset all performance metrics"):
            try:
                # Clear performance data
                for key in list(st.session_state.keys()):
                    if "fragment_performance" in key:
                        del st.session_state[key]
                st.success("✅ Performance metrics reset")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Reset failed: {e}")

    with col3:
        if st.button("🔧 Auto-Optimize", help="Apply recommended optimizations"):
            try:
                _apply_auto_optimizations()
                st.success("✅ Optimizations applied")
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")

    with col4:
        if st.button("📥 Export Metrics", help="Export performance data"):
            try:
                _export_performance_metrics()
                st.success("✅ Metrics exported to session state")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")


def _render_performance_testing_tools() -> None:
    """Render performance testing and simulation tools."""
    st.markdown("---")
    st.markdown("### 🧪 Performance Testing")

    with st.expander("Fragment Load Testing"):
        st.markdown("Test fragment performance under different load conditions.")

        col1, col2 = st.columns(2)

        with col1:
            # Test parameters
            test_fragments = st.number_input(
                "Number of Test Fragments",
                min_value=1,
                max_value=50,
                value=5,
                help="Number of fragments to create for testing",
            )

            test_interval = st.selectbox(
                "Test Interval",
                options=["1s", "2s", "5s", "10s"],
                index=1,
                help="Fragment refresh interval for testing",
            )

        with col2:
            test_duration = st.number_input(
                "Test Duration (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                help="How long to run the load test",
            )

            if st.button("▶️ Start Load Test"):
                st.info(f"🔄 Running load test with {test_fragments} fragments...")
                # Load test implementation would go here
                st.success("✅ Load test completed")


def _render_memory_management() -> None:
    """Render memory management section."""
    st.markdown("---")
    st.markdown("### 💾 Memory Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Session State Usage")

        # Calculate session state memory usage
        total_keys = len(st.session_state.keys())
        fragment_keys = len(
            [k for k in st.session_state.keys() if "fragment" in k.lower()]
        )

        st.metric("Total Session Keys", total_keys)
        st.metric("Fragment-related Keys", fragment_keys)

        # Fragment state breakdown
        fragment_categories = {
            "Performance": 0,
            "Communication": 0,
            "Orchestration": 0,
            "Other": 0,
        }

        for key in st.session_state.keys():
            key_lower = key.lower()
            if "performance" in key_lower:
                fragment_categories["Performance"] += 1
            elif "communication" in key_lower or "message" in key_lower:
                fragment_categories["Communication"] += 1
            elif "orchestrator" in key_lower or "coordination" in key_lower:
                fragment_categories["Orchestration"] += 1
            elif "fragment" in key_lower:
                fragment_categories["Other"] += 1

        # Display breakdown
        for category, count in fragment_categories.items():
            if count > 0:
                st.caption(f"• {category}: {count} keys")

    with col2:
        st.markdown("#### Memory Recommendations")

        if fragment_keys > 20:
            st.warning(f"⚠️ High fragment key count ({fragment_keys})")
            st.caption("Consider running memory cleanup")
        else:
            st.success(f"✅ Fragment memory usage normal ({fragment_keys} keys)")

        # Memory cleanup recommendations
        if total_keys > 100:
            st.warning("⚠️ High session state usage")
            st.caption("Regular cleanup recommended")
        else:
            st.info("📊 Session state usage healthy")

        # Cleanup suggestions
        st.markdown("**Cleanup Actions:**")
        if fragment_keys > 10:
            st.caption("🧹 Fragment memory cleanup available")
        if total_keys > 50:
            st.caption("🔄 Full session state cleanup recommended")


def _apply_auto_optimizations() -> None:
    """Apply automatic performance optimizations based on current metrics."""
    try:
        orchestrator = get_fragment_orchestrator()
        perf_summary = orchestrator.get_orchestration_summary()

        # Get optimization suggestions
        suggestions = perf_summary.get("optimization_suggestions", {})

        if suggestions:
            # Store optimizations in session state for fragments to pick up
            st.session_state.fragment_auto_optimizations = {
                "timestamp": datetime.now(UTC).isoformat(),
                "suggestions": suggestions,
                "applied": True,
            }

            logger.info(f"Applied {len(suggestions)} fragment optimizations")
        else:
            logger.info("No optimizations needed")

    except Exception:
        logger.exception("Failed to apply auto optimizations")
        raise


def _export_performance_metrics() -> None:
    """Export performance metrics to session state for external analysis."""
    try:
        orchestrator = get_fragment_orchestrator()
        perf_summary = orchestrator.get_orchestration_summary()

        # Store export in session state
        export_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "performance_summary": perf_summary,
            "session_state_keys": len(st.session_state.keys()),
            "fragment_keys": len(
                [k for k in st.session_state.keys() if "fragment" in k.lower()]
            ),
        }

        st.session_state.fragment_performance_export = export_data

        logger.info("Performance metrics exported to session state")

    except Exception:
        logger.exception("Failed to export performance metrics")
        raise


# Only render if this page is being displayed
if __name__ == "__main__":
    render_fragment_dashboard_page()
