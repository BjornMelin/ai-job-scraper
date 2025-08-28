"""Native Progress System Demo - Library-First Implementation.

This demo showcases the 612→25 line native progress system replacement,
demonstrating 96% code reduction with enhanced functionality.

Run with: streamlit run examples/native_progress_demo.py
"""

import asyncio
import time

from typing import Any

import streamlit as st

# Import our native progress components
from src.ui.components.native_progress import (
    NativeProgressContext,
    get_native_progress_manager,
    render_real_time_progress,
    show_progress_toast,
)


def demo_basic_progress() -> None:
    """Demonstrate basic progress tracking with native components."""
    st.markdown("## 📊 Basic Progress Demo")

    if st.button("Start Basic Progress", key="basic"):
        manager = get_native_progress_manager()

        # Show progress with native components
        with st.status("Processing basic task...", expanded=True) as status:
            progress_bar = st.progress(0.0, text="Starting...")

            for i in range(101):
                percentage = float(i)
                message = f"Processing step {i}/100"

                # Update native progress
                manager.update_progress("basic_demo", percentage, message, "processing")

                # Update UI
                progress_bar.progress(percentage / 100.0, text=message)

                # Update status
                if percentage >= 100.0:
                    status.update(label="✅ Completed", state="complete")
                else:
                    status.update(label=f"🔄 {message}", state="running")

                time.sleep(0.05)  # Simulate work

            # Show completion toast
            show_progress_toast("✅ Basic progress completed!", icon="✅")
            st.balloons()


def demo_context_manager() -> None:
    """Demonstrate progress tracking with context manager."""
    st.markdown("## 🎯 Context Manager Demo")

    if st.button("Start Context Progress", key="context"):
        # Use our native progress context
        with NativeProgressContext(
            "context_demo", "🔍 Context Manager Demo", expanded=True
        ) as progress:
            # Simulate multi-phase operation
            progress.update(10.0, "🔍 Initializing system...", "init")
            time.sleep(1)

            progress.update(30.0, "📋 Loading data...", "loading")
            time.sleep(1.5)

            progress.update(60.0, "🧠 Processing with AI...", "ai_processing")
            time.sleep(2)

            progress.update(85.0, "💾 Saving results...", "saving")
            time.sleep(1)

            progress.update(95.0, "📱 Updating UI...", "ui_update")
            time.sleep(0.5)

            progress.complete("🎉 All done! Processed 150 items.", show_balloons=True)


async def demo_async_progress() -> None:
    """Demonstrate async progress tracking."""
    manager = get_native_progress_manager()

    async def simulate_async_work(progress_id: str, total_steps: int) -> list[Any]:
        results = []
        for i in range(total_steps):
            # Simulate async work
            await asyncio.sleep(0.1)

            percentage = (i + 1) / total_steps * 100
            message = f"Async step {i + 1}/{total_steps}"

            manager.update_progress(
                progress_id, percentage, message, "async_processing"
            )

            results.append(f"result_{i + 1}")

        manager.complete_progress(
            progress_id, f"✅ Async completed! {len(results)} items processed"
        )
        return results

    # Start async work
    return await simulate_async_work("async_demo", 20)


def demo_real_time_fragment() -> None:
    """Demonstrate real-time progress with fragments."""
    st.markdown("## ⚡ Real-Time Fragment Demo")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Fragment Progress", key="fragment"):
            manager = get_native_progress_manager()

            # Initialize progress
            manager.update_progress(
                "fragment_demo", 0.0, "Starting fragment demo...", "init"
            )

            # Simulate background work
            import threading

            def background_work():
                for i in range(51):
                    percentage = i * 2.0  # 0 to 100
                    message = f"Background step {i}/50"

                    manager.update_progress(
                        "fragment_demo", percentage, message, "background_work"
                    )

                    time.sleep(0.2)

                manager.complete_progress(
                    "fragment_demo", "🎉 Fragment demo completed!"
                )

            # Start background thread
            thread = threading.Thread(target=background_work)
            thread.daemon = True
            thread.start()

    with col2:
        # Real-time progress display
        render_real_time_progress("fragment_demo")


def demo_error_handling() -> None:
    """Demonstrate error handling with native progress."""
    st.markdown("## ⚠️ Error Handling Demo")

    if st.button("Simulate Error", key="error"):
        try:
            with NativeProgressContext(
                "error_demo", "🔥 Error Simulation", expanded=True
            ) as progress:
                progress.update(20.0, "🔍 Starting risky operation...", "risky")
                time.sleep(1)

                progress.update(50.0, "⚠️ Entering danger zone...", "danger")
                time.sleep(1)

                # Simulate error
                raise ValueError("Simulated error for demonstration!")

        except Exception as e:
            st.error(f"❌ Error caught and handled: {e}")
            show_progress_toast("❌ Operation failed - see error above", icon="❌")


def demo_multiple_progress() -> None:
    """Demonstrate multiple concurrent progress trackers."""
    st.markdown("## 🔀 Multiple Progress Demo")

    if st.button("Start Multiple Progress", key="multiple"):
        manager = get_native_progress_manager()

        # Start three concurrent operations
        import threading

        def worker(worker_id: str, steps: int, delay: float):
            for i in range(steps + 1):
                percentage = (i / steps) * 100
                message = f"Worker {worker_id} - step {i}/{steps}"

                manager.update_progress(
                    f"worker_{worker_id}", percentage, message, f"worker_{worker_id}"
                )

                time.sleep(delay)

            manager.complete_progress(
                f"worker_{worker_id}", f"✅ Worker {worker_id} completed!"
            )

        # Start workers
        for worker_id, (steps, delay) in [
            ("A", (10, 0.3)),
            ("B", (15, 0.2)),
            ("C", (8, 0.4)),
        ]:
            thread = threading.Thread(target=worker, args=(worker_id, steps, delay))
            thread.daemon = True
            thread.start()

        show_progress_toast("🚀 Started 3 concurrent workers!", icon="🚀")

    # Show progress for all workers
    if "native_progress" in st.session_state:
        for worker_id in ["worker_A", "worker_B", "worker_C"]:
            render_real_time_progress(worker_id)


def main() -> None:
    """Main demo application."""
    st.set_page_config(
        page_title="Native Progress Demo",
        page_icon="🚀",
        layout="wide",
    )

    st.title("🚀 Native Progress System Demo")
    st.markdown("""
    **Library-First Implementation**: 612 lines → 25 lines (96% reduction)

    This demo showcases the native Streamlit progress components replacing
    our custom ProgressTracker with maximum library leverage:
    - `st.status()` - Expandable containers with state
    - `st.progress()` - Native progress bars with text
    - `st.toast()` - Non-intrusive notifications
    - `@st.fragment()` - Real-time auto-updating components
    - `st.session_state` - Persistence across reruns
    """)

    # Sidebar controls
    st.sidebar.title("🎛️ Demo Controls")

    # Cleanup button
    if st.sidebar.button("🧹 Clear All Progress"):
        if "native_progress" in st.session_state:
            st.session_state.native_progress.clear()
        show_progress_toast("🧹 All progress data cleared!", icon="🧹")
        st.rerun()

    # Auto-cleanup
    manager = get_native_progress_manager()
    cleaned = manager.cleanup_completed(max_age_minutes=2)
    if cleaned > 0:
        st.sidebar.success(f"🧹 Auto-cleaned {cleaned} old trackers")

    # Demo sections
    demo_basic_progress()
    st.markdown("---")
    demo_context_manager()
    st.markdown("---")
    demo_real_time_fragment()
    st.markdown("---")
    demo_error_handling()
    st.markdown("---")
    demo_multiple_progress()

    # Performance comparison
    st.markdown("---")
    st.markdown("## 📈 Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Custom ProgressTracker",
            "612 lines",
            delta="100% complexity",
            delta_color="inverse",
        )

    with col2:
        st.metric(
            "Native Progress System",
            "25 lines",
            delta="96% reduction",
            delta_color="normal",
        )

    # Features comparison
    st.markdown("### ✨ Feature Comparison")

    comparison_data = {
        "Feature": [
            "Real-time Updates",
            "Toast Notifications",
            "ETA Calculation",
            "Error Handling",
            "Mobile Responsive",
            "Fragment Auto-refresh",
            "Session Persistence",
            "Memory Management",
            "Code Maintainability",
            "Library Leverage",
        ],
        "Custom System": ["✅", "❌", "✅", "✅", "⚠️", "❌", "✅", "⚠️", "❌", "❌"],
        "Native System": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅"],
    }

    st.table(comparison_data)

    st.success(
        "🎉 Native progress system provides enhanced functionality with 96% less code!"
    )


if __name__ == "__main__":
    main()
