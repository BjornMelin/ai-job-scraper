"""Streamlit testing utilities and helpers.

This module provides utilities for testing Streamlit components, session state,
fragments, and UI interactions in a controlled test environment.
"""

import threading

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest


class MockStreamlitSession:
    """Mock Streamlit session for testing."""

    def __init__(self):
        """Initialize mock session with empty state."""
        self.state = {}
        self.widgets = {}
        self.fragment_state = {}
        self.rerun_count = 0
        self.callbacks = []

    def __getitem__(self, key: str) -> Any:
        """Get item from session state."""
        return self.state.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in session state."""
        self.state[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in session state."""
        return key in self.state

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from session state with default."""
        return self.state.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return item from session state."""
        return self.state.pop(key, default)

    def clear(self) -> None:
        """Clear all session state."""
        self.state.clear()
        self.widgets.clear()
        self.fragment_state.clear()

    def keys(self):
        """Get all keys in session state."""
        return self.state.keys()

    def values(self):
        """Get all values in session state."""
        return self.state.values()

    def items(self):
        """Get all items in session state."""
        return self.state.items()

    def update(self, other: dict) -> None:
        """Update session state with another dictionary."""
        self.state.update(other)

    def trigger_rerun(self) -> None:
        """Trigger a mock rerun."""
        self.rerun_count += 1
        for callback in self.callbacks:
            callback()

    def add_rerun_callback(self, callback: Callable) -> None:
        """Add callback to be called on rerun."""
        self.callbacks.append(callback)


class MockStreamlitContext:
    """Mock Streamlit script run context."""

    def __init__(self, session: MockStreamlitSession = None):
        """Initialize mock context."""
        self.session = session or MockStreamlitSession()
        self.session_state = self.session


@contextmanager
def mock_streamlit_context(
    initial_state: dict[str, Any] | None = None,
    session: MockStreamlitSession = None,
) -> Generator[MockStreamlitContext, None, None]:
    """Create a mock Streamlit context for testing.

    Args:
        initial_state: Initial session state values
        session: Pre-configured mock session (optional)

    Yields:
        Mock Streamlit context with session state
    """
    if session is None:
        session = MockStreamlitSession()

    if initial_state:
        session.update(initial_state)

    context = MockStreamlitContext(session)

    # Patch Streamlit's session state access
    with (
        patch("streamlit.session_state", session),
        patch(
            "streamlit.runtime.scriptrunner.get_script_run_ctx",
            return_value=context,
        ),
    ):
        yield context


@contextmanager
def isolated_streamlit_session(
    initial_state: dict[str, Any] | None = None,
) -> Generator[MockStreamlitSession, None, None]:
    """Create an isolated Streamlit session for testing.

    Args:
        initial_state: Initial session state values

    Yields:
        Isolated mock session
    """
    with mock_streamlit_context(initial_state) as context:
        yield context.session


def create_mock_streamlit_widgets():
    """Create mock Streamlit widget functions for testing."""
    widgets = {
        "button": Mock(return_value=False),
        "checkbox": Mock(return_value=False),
        "radio": Mock(return_value=None),
        "selectbox": Mock(return_value=None),
        "multiselect": Mock(return_value=[]),
        "text_input": Mock(return_value=""),
        "number_input": Mock(return_value=0),
        "slider": Mock(return_value=0),
        "select_slider": Mock(return_value=None),
        "text_area": Mock(return_value=""),
        "date_input": Mock(return_value=None),
        "time_input": Mock(return_value=None),
        "file_uploader": Mock(return_value=None),
        "color_picker": Mock(return_value="#000000"),
        "columns": Mock(return_value=[Mock(), Mock()]),
        "container": Mock(),
        "expander": Mock(),
        "form": Mock(),
        "empty": Mock(),
    }

    # Configure widgets to store values in session state
    def make_stateful_widget(widget_name: str, default_value: Any):
        def widget_func(*args, key: str | None = None, **kwargs):
            if key and "session_state" in globals():
                if key not in session_state:  # type: ignore
                    session_state[key] = default_value  # type: ignore
                return session_state[key]  # type: ignore
            return default_value

        return widget_func

    # Make some widgets stateful
    widgets["button"] = make_stateful_widget("button", False)
    widgets["checkbox"] = make_stateful_widget("checkbox", False)
    widgets["text_input"] = make_stateful_widget("text_input", "")

    return widgets


@contextmanager
def mock_streamlit_app():
    """Mock complete Streamlit app environment."""
    widgets = create_mock_streamlit_widgets()

    with patch.multiple(
        "streamlit",
        **widgets,
        title=Mock(),
        header=Mock(),
        subheader=Mock(),
        markdown=Mock(),
        write=Mock(),
        success=Mock(),
        info=Mock(),
        warning=Mock(),
        error=Mock(),
        exception=Mock(),
        json=Mock(),
        dataframe=Mock(),
        table=Mock(),
        metric=Mock(),
        plotly_chart=Mock(),
        pyplot=Mock(),
        map=Mock(),
        image=Mock(),
        audio=Mock(),
        video=Mock(),
        progress=Mock(),
        spinner=Mock(),
        balloons=Mock(),
        snow=Mock(),
        rerun=Mock(),
        stop=Mock(),
    ):
        yield


class StreamlitComponentTester:
    """Helper class for testing Streamlit components."""

    def __init__(self, component_func: Callable):
        """Initialize component tester.

        Args:
            component_func: Streamlit component function to test
        """
        self.component_func = component_func
        self.session = MockStreamlitSession()

    def run_component(self, *args, **kwargs) -> Any:
        """Run component with mock Streamlit environment.

        Args:
            *args: Component arguments
            **kwargs: Component keyword arguments

        Returns:
            Component return value
        """
        with mock_streamlit_context(session=self.session):
            with mock_streamlit_app():
                return self.component_func(*args, **kwargs)

    def set_session_state(self, **kwargs) -> None:
        """Set session state values."""
        self.session.update(kwargs)

    def get_session_state(self, key: str | None = None) -> Any:
        """Get session state value or entire state."""
        if key:
            return self.session.get(key)
        return dict(self.session.state)

    def clear_session_state(self) -> None:
        """Clear session state."""
        self.session.clear()

    def trigger_rerun(self) -> None:
        """Trigger component rerun."""
        self.session.trigger_rerun()


def test_streamlit_component(component_func: Callable):
    """Decorator to create Streamlit component test environment.

    Args:
        component_func: Component function to test

    Returns:
        Decorated test function with component tester
    """

    def decorator(test_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tester = StreamlitComponentTester(component_func)
            return test_func(tester, *args, **kwargs)

        return wrapper

    return decorator


class MockStreamlitFragment:
    """Mock Streamlit fragment for testing."""

    def __init__(self, func: Callable, run_every: float | None = None):
        """Initialize mock fragment.

        Args:
            func: Fragment function
            run_every: Auto-refresh interval in seconds
        """
        self.func = func
        self.run_every = run_every
        self.run_count = 0
        self.is_running = False
        self.timer = None

    def __call__(self, *args, **kwargs):
        """Call fragment function."""
        self.run_count += 1
        return self.func(*args, **kwargs)

    def start_auto_refresh(self) -> None:
        """Start auto-refresh timer."""
        if self.run_every and not self.is_running:
            self.is_running = True
            self._schedule_next_run()

    def stop_auto_refresh(self) -> None:
        """Stop auto-refresh timer."""
        self.is_running = False
        if self.timer:
            self.timer.cancel()

    def _schedule_next_run(self) -> None:
        """Schedule next auto-refresh run."""
        if self.is_running and self.run_every:
            self.timer = threading.Timer(self.run_every, self._auto_refresh)
            self.timer.start()

    def _auto_refresh(self) -> None:
        """Execute auto-refresh."""
        if self.is_running:
            self.run_count += 1
            self._schedule_next_run()


def mock_streamlit_fragment(run_every: float | None = None):
    """Mock st.fragment decorator for testing.

    Args:
        run_every: Auto-refresh interval in seconds

    Returns:
        Mock fragment decorator
    """

    def decorator(func: Callable) -> MockStreamlitFragment:
        return MockStreamlitFragment(func, run_every)

    return decorator


@contextmanager
def mock_database_connection(mock_data: dict[str, list] | None = None):
    """Mock database connection for Streamlit tests.

    Args:
        mock_data: Mock data to return from queries

    Yields:
        Mock database session
    """
    mock_session = Mock()

    if mock_data:
        # Configure mock to return specific data
        def mock_exec(query):
            # Simple query matching - in real tests, you'd want more sophisticated matching
            query_str = str(query).lower()
            if "job" in query_str:
                return Mock(fetchall=Mock(return_value=mock_data.get("jobs", [])))
            if "company" in query_str:
                return Mock(fetchall=Mock(return_value=mock_data.get("companies", [])))
            return Mock(fetchall=Mock(return_value=[]))

        mock_session.exec.side_effect = mock_exec

    yield mock_session


def assert_streamlit_component_output(
    component_func: Callable,
    expected_outputs: dict[str, Any],
    session_state: dict[str, Any] | None = None,
    *args,
    **kwargs,
):
    """Assert that Streamlit component produces expected outputs.

    Args:
        component_func: Component function to test
        expected_outputs: Expected Streamlit calls and their arguments
        session_state: Initial session state
        *args: Component arguments
        **kwargs: Component keyword arguments
    """
    tester = StreamlitComponentTester(component_func)

    if session_state:
        tester.set_session_state(**session_state)

    # Capture Streamlit calls
    with mock_streamlit_app() as mock_st:
        tester.run_component(*args, **kwargs)

        # Verify expected outputs
        for call_name, expected_args in expected_outputs.items():
            mock_func = getattr(mock_st, call_name, None)
            if mock_func:
                if expected_args is None:
                    # Just check that function was called
                    mock_func.assert_called()
                else:
                    # Check specific arguments
                    mock_func.assert_called_with(*expected_args)


def simulate_streamlit_interaction(
    component_func: Callable,
    interactions: list[dict[str, Any]],
    initial_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Simulate user interactions with Streamlit component.

    Args:
        component_func: Component function to test
        interactions: List of interaction dictionaries
        initial_state: Initial session state

    Returns:
        List of session state snapshots after each interaction
    """
    tester = StreamlitComponentTester(component_func)

    if initial_state:
        tester.set_session_state(**initial_state)

    snapshots = []

    for interaction in interactions:
        # Set widget values based on interaction
        for key, value in interaction.items():
            tester.set_session_state(**{key: value})

        # Run component
        tester.run_component()

        # Capture state snapshot
        snapshots.append(tester.get_session_state())

        # Trigger rerun if needed
        tester.trigger_rerun()

    return snapshots


# Pytest fixtures


@pytest.fixture
def streamlit_session():
    """Provide isolated Streamlit session for tests."""
    return MockStreamlitSession()


@pytest.fixture
def streamlit_context(streamlit_session):
    """Provide Streamlit context with session."""
    return MockStreamlitContext(streamlit_session)


@pytest.fixture
def mock_st_components():
    """Provide mock Streamlit components."""
    return create_mock_streamlit_widgets()


@pytest.fixture
def component_tester():
    """Factory fixture for creating component testers."""

    def _create_tester(component_func: Callable) -> StreamlitComponentTester:
        return StreamlitComponentTester(component_func)

    return _create_tester
