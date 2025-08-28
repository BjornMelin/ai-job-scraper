"""Native Streamlit Progress Components Demo - Zero Custom Code.

This demo showcases using native Streamlit components (st.progress, st.status, st.toast)
directly without any custom wrapper classes. This demonstrates the library-first approach
where we leverage built-in Streamlit functionality instead of writing custom code.

Run with: streamlit run examples/native_progress_demo.py
"""

import threading
import time

from datetime import UTC, datetime

import streamlit as st


def demo_basic_progress() -> None:
    """Demonstrate basic progress tracking with native st.progress()."""
    st.markdown("## 📊 Basic Progress with st.progress()")

    if st.button("Start Basic Progress", key="basic"):
        # Native progress bar
        progress_bar = st.progress(0.0, text="Starting...")

        # Simulate work with progress updates
        for i in range(101):
            percentage = i
            message = f"Processing step {i + 1}/101"
            progress_bar.progress(percentage / 100.0, text=f"{message} ({percentage}%)")
            time.sleep(0.02)  # Simulate work

        # Show completion toast
        st.toast("✅ Basic progress completed!", icon="✅")
        st.balloons()


def demo_status_container() -> None:
    """Demonstrate progress tracking with native st.status()."""
    st.markdown("## 🎯 Progress with st.status()")

    if st.button("Start Status Progress", key="status"):
        # Native status container
        with st.status("Processing data...", expanded=True) as status:
            # Step 1
            status.update(label="🔍 Analyzing data...", state="running")
            progress_bar = st.progress(0.0, text="Analyzing...")
            for i in range(34):
                progress_bar.progress(i / 100.0, text=f"Analyzing... {i}%")
                time.sleep(0.03)

            # Step 2
            status.update(label="🔄 Processing results...", state="running")
            for i in range(34, 67):
                progress_bar.progress(i / 100.0, text=f"Processing... {i}%")
                time.sleep(0.03)

            # Step 3
            status.update(label="💾 Saving results...", state="running")
            for i in range(67, 101):
                progress_bar.progress(i / 100.0, text=f"Saving... {i}%")
                time.sleep(0.03)

            # Complete
            progress_bar.progress(1.0, text="Complete!")
            status.update(label="✅ All steps completed!", state="complete")

        st.toast("🎉 Status progress completed!", icon="🎉")


def demo_spinner() -> None:
    """Demonstrate progress with native st.spinner()."""
    st.markdown("## 🌀 Progress with st.spinner()")

    if st.button("Start Spinner Demo", key="spinner"):
        with st.spinner("Loading data..."):
            time.sleep(2)
            st.success("✅ Data loaded!")

        with st.spinner("Processing results..."):
            time.sleep(1.5)
            st.success("✅ Processing complete!")

        with st.spinner("Saving to database..."):
            time.sleep(1)
            st.success("✅ Saved successfully!")

        st.toast("🎯 Spinner demo completed!", icon="🎯")


def demo_session_state_progress() -> None:
    """Demonstrate progress tracking with session state."""
    st.markdown("## 💾 Progress with Session State")

    if st.button("Start Session Progress", key="session"):
        # Initialize progress in session state
        if "demo_progress" not in st.session_state:
            st.session_state.demo_progress = {
                "percentage": 0,
                "message": "Starting...",
                "is_active": True,
                "start_time": datetime.now(UTC),
            }

        # Update progress
        for i in range(101):
            st.session_state.demo_progress.update(
                {
                    "percentage": i,
                    "message": f"Processing item {i}/100",
                    "is_active": i < 100,
                }
            )

            # Display current state
            progress_data = st.session_state.demo_progress
            st.progress(
                progress_data["percentage"] / 100.0,
                text=f"{progress_data['message']} ({progress_data['percentage']}%)",
            )

            if progress_data["is_active"]:
                elapsed = (
                    datetime.now(UTC) - progress_data["start_time"]
                ).total_seconds()
                if elapsed > 0 and i > 0:
                    eta = (elapsed / i) * (100 - i)
                    st.caption(f"⏱️ ETA: {eta:.1f}s remaining")

            time.sleep(0.03)

        st.toast("💾 Session state progress completed!", icon="💾")
        st.session_state.demo_progress["is_active"] = False


def demo_concurrent_progress() -> None:
    """Demonstrate multiple concurrent progress trackers."""
    st.markdown("## 🔀 Concurrent Progress")

    if st.button("Start Concurrent Demo", key="concurrent"):
        # Initialize session state for multiple workers
        if "workers" not in st.session_state:
            st.session_state.workers = {}

        # Create worker function
        def worker(worker_id: str, steps: int, delay: float) -> None:
            """Worker function that updates progress."""
            st.session_state.workers[worker_id] = {
                "percentage": 0,
                "message": f"Worker {worker_id} starting...",
                "is_active": True,
            }

            for i in range(steps + 1):
                if worker_id in st.session_state.workers:
                    st.session_state.workers[worker_id].update(
                        {
                            "percentage": (i / steps) * 100,
                            "message": f"Worker {worker_id} step {i}/{steps}",
                            "is_active": i < steps,
                        }
                    )
                time.sleep(delay)

        # Start three workers with different speeds
        workers = [("A", 50, 0.05), ("B", 30, 0.08), ("C", 40, 0.06)]

        for worker_id, steps, delay in workers:
            thread = threading.Thread(target=worker, args=(worker_id, steps, delay))
            thread.daemon = True
            thread.start()

        st.toast("🚀 Started 3 concurrent workers!", icon="🚀")

    # Display worker progress if any exist
    if "workers" in st.session_state and st.session_state.workers:
        st.markdown("### Worker Progress")

        for worker_id, worker_data in st.session_state.workers.items():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{worker_data['message']}**")
                    st.progress(
                        worker_data["percentage"] / 100.0,
                        text=f"{worker_data['percentage']:.1f}%",
                    )

                with col2:
                    if not worker_data["is_active"]:
                        st.success("✅ Done")
                    else:
                        st.info("🔄 Active")


def main() -> None:
    """Main demo application."""
    st.set_page_config(
        page_title="Native Progress Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Native Streamlit Progress Components")
    st.markdown("**Zero custom code - Pure native Streamlit components**")

    # Sidebar info
    st.sidebar.markdown("## 🎯 Library-First Approach")
    st.sidebar.markdown("""
    This demo uses only native Streamlit components:
    - `st.progress()` for progress bars
    - `st.status()` for status containers  
    - `st.toast()` for notifications
    - `st.spinner()` for loading states
    - `st.session_state` for progress tracking
    - `st.fragment()` for real-time updates
    """)

    # Cleanup controls
    st.sidebar.markdown("## 🧹 Controls")
    if st.sidebar.button("Clear All Progress"):
        # Clear session state
        for key in ["demo_progress", "workers"]:
            if key in st.session_state:
                del st.session_state[key]
        st.toast("🧹 All progress data cleared!", icon="🧹")
        st.rerun()

    # Demo sections
    demo_basic_progress()
    st.markdown("---")

    demo_status_container()
    st.markdown("---")

    demo_spinner()
    st.markdown("---")

    demo_session_state_progress()
    st.markdown("---")

    demo_concurrent_progress()

    # Footer
    st.markdown("---")
    st.markdown("### ✨ Key Benefits")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**📦 Library-First**\nNo custom progress classes")

    with col2:
        st.info("**🚀 Native Performance**\nDirect Streamlit components")

    with col3:
        st.info("**🔧 Zero Maintenance**\nNo custom code to maintain")


if __name__ == "__main__":
    main()
