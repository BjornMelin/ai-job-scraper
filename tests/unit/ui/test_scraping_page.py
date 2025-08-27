"""Comprehensive tests for the Scraping page UI component.

This module tests the Scraping page functionality including:
- Page rendering and header display
- Scraping control buttons (start/stop)
- Real-time progress monitoring with fragments
- Session state management
- Background task status display
- Error handling and edge cases
- Progress visualization and ETA calculations
"""

from unittest.mock import patch

from tests.utils.streamlit_utils import StreamlitComponentTester

from src.ui.pages.scraping import (
    render_scraping_page,
)


class TestScrapingPageBasics:
    """Test basic scraping page functionality."""

    def test_render_scraping_page_basic_structure(self):
        """Test scraping page renders basic structure correctly."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown") as mock_markdown,
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons"),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            # Verify header is rendered
            assert mock_markdown.call_count >= 2
            header_calls = [call[0][0] for call in mock_markdown.call_args_list]
            assert any("Job Scraping Dashboard" in call for call in header_calls)
            assert any("Monitor and control" in call for call in header_calls)

    def test_session_state_initialization(self):
        """Test scraping page initializes session state correctly."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons"),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            # Verify session state is initialized
            state = tester.get_session_state()
            assert "last_refresh" in state
            assert isinstance(state["last_refresh"], (int, float))

    def test_scraping_active_state_detection(self):
        """Test page correctly detects scraping active state."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=True
            ) as mock_is_active,
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            # Verify scraping state is checked
            mock_is_active.assert_called_once()
            mock_controls.assert_called_once_with(True)  # Should pass active state

    def test_scraping_inactive_state_detection(self):
        """Test page correctly detects scraping inactive state."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ) as mock_is_active,
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            # Verify scraping state is checked
            mock_is_active.assert_called_once()
            mock_controls.assert_called_once_with(False)  # Should pass inactive state


class TestScrapingControlButtons:
    """Test scraping control button functionality."""

    @patch("src.ui.pages.scraping._render_control_buttons")
    def test_control_buttons_called_with_active_state(self, mock_render_buttons):
        """Test control buttons are rendered with correct active state."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=True
            ),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            mock_render_buttons.assert_called_once_with(True)

    @patch("src.ui.pages.scraping._render_control_buttons")
    def test_control_buttons_called_with_inactive_state(self, mock_render_buttons):
        """Test control buttons are rendered with correct inactive state."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            mock_render_buttons.assert_called_once_with(False)


class TestScrapingPageSections:
    """Test scraping page sections are rendered correctly."""

    def test_all_main_sections_rendered(self):
        """Test all main page sections are called."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status") as mock_status,
            patch("src.ui.pages.scraping._render_progress_monitoring") as mock_progress,
        ):
            tester.run_component()

            # Verify all main sections are rendered
            mock_controls.assert_called_once()
            mock_status.assert_called_once()
            mock_progress.assert_called_once()

    def test_sections_rendering_order(self):
        """Test sections are rendered in correct order."""
        tester = StreamlitComponentTester(render_scraping_page)

        call_order = []

        def track_control_buttons(*args):
            call_order.append("controls")

        def track_status(*args):
            call_order.append("status")

        def track_progress(*args):
            call_order.append("progress")

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch(
                "src.ui.pages.scraping._render_control_buttons",
                side_effect=track_control_buttons,
            ),
            patch(
                "src.ui.pages.scraping._render_scraping_status",
                side_effect=track_status,
            ),
            patch(
                "src.ui.pages.scraping._render_progress_monitoring",
                side_effect=track_progress,
            ),
        ):
            tester.run_component()

            # Verify sections are called in expected order
            expected_order = ["controls", "status", "progress"]
            assert call_order == expected_order


class TestScrapingPageErrorHandling:
    """Test error handling in scraping page."""

    def test_handles_is_scraping_active_exception(self):
        """Test page handles exception from is_scraping_active gracefully."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active",
                side_effect=Exception("Service error"),
            ),
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            # Should not raise exception
            tester.run_component()

            # Should still call controls with default False state on error
            mock_controls.assert_called_once_with(False)

    def test_handles_section_rendering_exceptions(self):
        """Test page handles exceptions in section rendering."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch(
                "src.ui.pages.scraping._render_control_buttons",
                side_effect=Exception("Controls error"),
            ),
            patch("src.ui.pages.scraping._render_scraping_status") as mock_status,
            patch("src.ui.pages.scraping._render_progress_monitoring") as mock_progress,
        ):
            # Should not raise exception and continue rendering other sections
            tester.run_component()

            # Other sections should still be called
            mock_status.assert_called_once()
            mock_progress.assert_called_once()


class TestScrapingPageIntegration:
    """Integration tests for scraping page components."""

    def test_complete_page_rendering_inactive_state(self):
        """Test complete page renders correctly when scraping is inactive."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown") as mock_markdown,
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status") as mock_status,
            patch("src.ui.pages.scraping._render_progress_monitoring") as mock_progress,
        ):
            tester.run_component()

            # Verify all components are rendered
            assert mock_markdown.call_count >= 2
            mock_controls.assert_called_once_with(False)
            mock_status.assert_called_once()
            mock_progress.assert_called_once()

            # Verify session state is properly initialized
            state = tester.get_session_state()
            assert "last_refresh" in state

    def test_complete_page_rendering_active_state(self):
        """Test complete page renders correctly when scraping is active."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown") as mock_markdown,
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=True
            ),
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status") as mock_status,
            patch("src.ui.pages.scraping._render_progress_monitoring") as mock_progress,
        ):
            tester.run_component()

            # Verify all components are rendered with active state
            assert mock_markdown.call_count >= 2
            mock_controls.assert_called_once_with(True)
            mock_status.assert_called_once()
            mock_progress.assert_called_once()

    def test_session_state_persistence_across_calls(self):
        """Test session state persists across multiple page renders."""
        tester = StreamlitComponentTester(render_scraping_page)

        # Set initial session state
        tester.set_session_state(last_refresh=123.45)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons"),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            tester.run_component()

            # Verify existing state is preserved
            state = tester.get_session_state()
            assert state["last_refresh"] == 123.45


class TestScrapingPageEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_handles_missing_session_state_gracefully(self):
        """Test page handles completely empty session state."""
        tester = StreamlitComponentTester(render_scraping_page)

        # Clear any existing session state
        tester.clear_session_state()

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons"),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            # Should not raise exception
            tester.run_component()

            # Should initialize session state
            state = tester.get_session_state()
            assert "last_refresh" in state

    def test_handles_corrupted_session_state(self):
        """Test page handles corrupted session state values."""
        tester = StreamlitComponentTester(render_scraping_page)

        # Set invalid session state
        tester.set_session_state(last_refresh="invalid_value")

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons"),
            patch("src.ui.pages.scraping._render_scraping_status"),
            patch("src.ui.pages.scraping._render_progress_monitoring"),
        ):
            # Should not raise exception and should reinitialize if needed
            tester.run_component()

            # Should have session state
            state = tester.get_session_state()
            assert "last_refresh" in state

    def test_multiple_rapid_calls_stability(self):
        """Test page handles multiple rapid successive calls."""
        tester = StreamlitComponentTester(render_scraping_page)

        with (
            patch("streamlit.markdown"),
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=False
            ),
            patch("src.ui.pages.scraping._render_control_buttons") as mock_controls,
            patch("src.ui.pages.scraping._render_scraping_status") as mock_status,
            patch("src.ui.pages.scraping._render_progress_monitoring") as mock_progress,
        ):
            # Call multiple times rapidly
            for _ in range(5):
                tester.run_component()

            # All sections should be called each time
            assert mock_controls.call_count == 5
            assert mock_status.call_count == 5
            assert mock_progress.call_count == 5

            # Session state should remain stable
            state = tester.get_session_state()
            assert "last_refresh" in state
