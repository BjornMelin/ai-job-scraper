"""Tests for view mode UI helper functions.

These tests validate the view mode selection and application functions including
selectbox controls, grid column selection, job rendering in different modes,
and various edge cases for UI state management.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.schemas import Job
from src.ui.helpers.view_mode import apply_view_mode, select_view_mode


class TestSelectViewMode:
    """Test cases for select_view_mode function."""

    def test_select_view_mode_returns_grid_with_default_columns(self, mock_streamlit):
        """Test select_view_mode returns grid mode with default 3 columns."""
        # Mock columns to return two column objects
        col1, col2 = MagicMock(), MagicMock()
        mock_streamlit["columns"].side_effect = [
            [col1, col2],
            [MagicMock(), MagicMock()],
        ]

        # Mock selectbox to return Grid mode and default columns
        mock_streamlit["selectbox"].side_effect = ["Grid", 3]

        view_mode, grid_columns = select_view_mode("test_tab")

        # Verify return values
        assert view_mode == "Grid"
        assert grid_columns == 3

        # Verify selectbox calls
        selectbox_calls = mock_streamlit["selectbox"].call_args_list
        assert len(selectbox_calls) == 2

        # First call for view mode
        assert selectbox_calls[0][0][0] == "View"  # First positional arg
        assert selectbox_calls[0][1]["key"] == "view_mode_test_tab"

        # Second call for grid columns
        assert selectbox_calls[1][0][0] == "Columns"  # First positional arg
        assert selectbox_calls[1][1]["key"] == "grid_columns_test_tab"
        assert selectbox_calls[1][1]["index"] == 1  # Default to 3 columns (index 1)

    def test_select_view_mode_returns_list_with_none_columns(self, mock_streamlit):
        """Test select_view_mode returns list mode with None columns."""
        # Mock columns to return two column objects
        col1, col2 = MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2]

        # Mock selectbox to return List mode
        mock_streamlit["selectbox"].return_value = "List"

        view_mode, grid_columns = select_view_mode("list_tab")

        # Verify return values
        assert view_mode == "List"
        assert grid_columns is None

        # Verify only one selectbox call for view mode (no columns selector for List)
        assert mock_streamlit["selectbox"].call_count == 1
        selectbox_call = mock_streamlit["selectbox"].call_args
        assert selectbox_call[0][0] == "View"  # First positional arg
        assert selectbox_call[1]["key"] == "view_mode_list_tab"

    def test_select_view_mode_with_different_tab_keys(self, mock_streamlit):
        """Test select_view_mode generates unique keys for different tabs."""
        # Mock columns
        col1, col2 = MagicMock(), MagicMock()
        mock_streamlit["columns"].side_effect = [
            [col1, col2],
            [MagicMock(), MagicMock()],  # First call
            [col1, col2],
            [MagicMock(), MagicMock()],  # Second call
        ]

        # Mock selectbox to return Grid mode
        mock_streamlit["selectbox"].side_effect = ["Grid", 2, "Grid", 4]

        # Test different tab keys
        view_mode1, _ = select_view_mode("jobs_tab")
        view_mode2, _ = select_view_mode("favorites_tab")

        # Verify selectbox calls have different keys
        selectbox_calls = mock_streamlit["selectbox"].call_args_list
        assert len(selectbox_calls) == 4

        # First tab keys
        assert "view_mode_jobs_tab" in selectbox_calls[0][1]["key"]
        assert "grid_columns_jobs_tab" in selectbox_calls[1][1]["key"]

        # Second tab keys
        assert "view_mode_favorites_tab" in selectbox_calls[2][1]["key"]
        assert "grid_columns_favorites_tab" in selectbox_calls[3][1]["key"]

    @pytest.mark.parametrize("grid_columns", [2, 3, 4])
    def test_select_view_mode_with_different_grid_columns(
        self, mock_streamlit, grid_columns
    ):
        """Test select_view_mode with different grid column options."""
        # Mock columns
        col1, col2 = MagicMock(), MagicMock()
        mock_streamlit["columns"].side_effect = [
            [col1, col2],
            [MagicMock(), MagicMock()],
        ]

        # Mock selectbox to return Grid mode and specified columns
        mock_streamlit["selectbox"].side_effect = ["Grid", grid_columns]

        view_mode, returned_columns = select_view_mode("test_tab")

        assert view_mode == "Grid"
        assert returned_columns == grid_columns

        # Verify columns selectbox options
        columns_call = mock_streamlit["selectbox"].call_args_list[1]
        assert columns_call[0][0] == "Columns"  # First positional arg
        assert columns_call[1]["help"] == "Number of columns in grid view"

    def test_select_view_mode_column_layout(self, mock_streamlit):
        """Test select_view_mode creates correct column layout."""
        # Mock columns to return expected column objects
        first_cols = [MagicMock(), MagicMock()]  # For initial layout
        second_cols = [MagicMock(), MagicMock()]  # For grid columns selector
        mock_streamlit["columns"].side_effect = [first_cols, second_cols]

        # Mock selectbox to return Grid mode
        mock_streamlit["selectbox"].side_effect = ["Grid", 3]

        select_view_mode("layout_test")

        # Verify columns are called with correct proportions
        columns_calls = mock_streamlit["columns"].call_args_list
        assert len(columns_calls) == 2

        # First call should be for main layout [2, 1]
        assert columns_calls[0][0] == ([2, 1],)

        # Second call should be for grid columns layout [3, 1]
        assert columns_calls[1][0] == ([3, 1],)

    def test_select_view_mode_help_text(self, mock_streamlit):
        """Test select_view_mode includes appropriate help text."""
        # Mock columns and selectbox
        mock_streamlit["columns"].side_effect = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock()],
        ]
        mock_streamlit["selectbox"].side_effect = ["Grid", 3]

        select_view_mode("help_test")

        # Check view mode selectbox help
        view_mode_call = mock_streamlit["selectbox"].call_args_list[0]
        assert view_mode_call[1]["help"] == "Choose how to display jobs"

        # Check grid columns selectbox help
        grid_columns_call = mock_streamlit["selectbox"].call_args_list[1]
        assert grid_columns_call[1]["help"] == "Number of columns in grid view"

    def test_select_view_mode_with_empty_tab_key(self, mock_streamlit):
        """Test select_view_mode with empty tab key."""
        # Mock components
        mock_streamlit["columns"].return_value = [MagicMock(), MagicMock()]
        mock_streamlit["selectbox"].return_value = "List"

        view_mode, grid_columns = select_view_mode("")

        # Should still work with empty key
        assert view_mode == "List"
        assert grid_columns is None

        # Verify key generation
        selectbox_call = mock_streamlit["selectbox"].call_args
        assert selectbox_call[1]["key"] == "view_mode_"

    def test_select_view_mode_with_special_characters_in_tab_key(self, mock_streamlit):
        """Test select_view_mode with special characters in tab key."""
        # Mock components
        mock_streamlit["columns"].side_effect = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock()],
        ]
        mock_streamlit["selectbox"].side_effect = ["Grid", 2]

        tab_key = "special-tab_key!@#"
        view_mode, grid_columns = select_view_mode(tab_key)

        assert view_mode == "Grid"
        assert grid_columns == 2

        # Verify keys are properly formed despite special characters
        selectbox_calls = mock_streamlit["selectbox"].call_args_list
        assert f"view_mode_{tab_key}" == selectbox_calls[0][1]["key"]
        assert f"grid_columns_{tab_key}" == selectbox_calls[1][1]["key"]


class TestApplyViewMode:
    """Test cases for apply_view_mode function."""

    def test_apply_view_mode_renders_grid_with_columns(self, sample_jobs_dto):
        """Test apply_view_mode renders jobs in grid mode with specified columns."""
        jobs = sample_jobs_dto[:3]  # Use first 3 jobs

        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            apply_view_mode(jobs, "Grid", grid_columns=3)

            # Verify grid renderer is called with correct parameters
            mock_grid.assert_called_once_with(jobs, num_columns=3)

    def test_apply_view_mode_renders_list_mode(self, sample_jobs_dto):
        """Test apply_view_mode renders jobs in list mode."""
        jobs = sample_jobs_dto[:2]  # Use first 2 jobs

        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            apply_view_mode(jobs, "List", grid_columns=None)

            # Verify list renderer is called with jobs
            mock_list.assert_called_once_with(jobs)

    def test_apply_view_mode_falls_back_to_list_for_invalid_grid(self, sample_jobs_dto):
        """Test apply_view_mode falls back to list mode for invalid grid config."""
        jobs = sample_jobs_dto

        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            # Test with None columns (should fallback to list)
            apply_view_mode(jobs, "Grid", grid_columns=None)
            mock_list.assert_called_once_with(jobs)

            mock_list.reset_mock()

            # Test with 0 columns (should fallback to list)
            apply_view_mode(jobs, "Grid", grid_columns=0)
            mock_list.assert_called_once_with(jobs)

    def test_apply_view_mode_with_empty_jobs_list(self):
        """Test apply_view_mode handles empty jobs list."""
        empty_jobs = []

        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            apply_view_mode(empty_jobs, "Grid", grid_columns=2)
            mock_grid.assert_called_once_with([], num_columns=2)

        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            apply_view_mode(empty_jobs, "List")
            mock_list.assert_called_once_with([])

    @pytest.mark.parametrize(
        "view_mode", ["LIST", "list", "List", "GRID", "grid", "Grid"]
    )
    def test_apply_view_mode_case_sensitivity(self, sample_jobs_dto, view_mode):
        """Test apply_view_mode handles different case variations."""
        jobs = sample_jobs_dto[:1]

        if view_mode.lower() == "grid":
            with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
                apply_view_mode(jobs, view_mode, grid_columns=2)
                if view_mode == "Grid":  # Only exact match should trigger grid
                    mock_grid.assert_called_once_with(jobs, num_columns=2)
                else:  # Other cases should fallback to list
                    mock_grid.assert_not_called()
        else:
            with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
                apply_view_mode(jobs, view_mode)
                # All non-"Grid" modes should use list
                mock_list.assert_called_once_with(jobs)

    def test_apply_view_mode_with_single_job(self):
        """Test apply_view_mode with single job."""
        job = Job(
            id=1,
            company_id=1,
            company="Single Co",
            title="Single Job",
            description="Single description",
            location="Remote",
            application_status="New",
        )
        jobs = [job]

        # Test grid mode
        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            apply_view_mode(jobs, "Grid", grid_columns=1)
            mock_grid.assert_called_once_with(jobs, num_columns=1)

        # Test list mode
        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            apply_view_mode(jobs, "List")
            mock_list.assert_called_once_with(jobs)

    def test_apply_view_mode_with_large_jobs_list(self, sample_jobs_dto):
        """Test apply_view_mode with large jobs list."""
        # Create extended jobs list
        extended_jobs = sample_jobs_dto * 10  # 40 jobs total

        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            apply_view_mode(extended_jobs, "Grid", grid_columns=4)
            mock_grid.assert_called_once_with(extended_jobs, num_columns=4)

    def test_apply_view_mode_with_different_grid_columns(self, sample_jobs_dto):
        """Test apply_view_mode with different grid column values."""
        jobs = sample_jobs_dto

        for columns in [1, 2, 3, 4, 5, 10]:
            with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
                apply_view_mode(jobs, "Grid", grid_columns=columns)
                mock_grid.assert_called_once_with(jobs, num_columns=columns)

    def test_apply_view_mode_unknown_view_mode(self, sample_jobs_dto):
        """Test apply_view_mode with unknown view mode defaults to list."""
        jobs = sample_jobs_dto

        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            # Test various unknown modes
            for unknown_mode in ["Unknown", "Table", "Card", "", None]:
                mock_list.reset_mock()
                apply_view_mode(jobs, unknown_mode)
                mock_list.assert_called_once_with(jobs)

    def test_apply_view_mode_import_paths(self):
        """Test that apply_view_mode imports from correct modules."""
        jobs = []

        # Test that the import paths are correct by checking the patch targets
        with (
            patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid,
            patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list,
        ):
            apply_view_mode(jobs, "Grid", grid_columns=2)
            apply_view_mode(jobs, "List")

            # Verify both functions were imported and called
            mock_grid.assert_called_once()
            mock_list.assert_called_once()


class TestViewModeEdgeCases:
    """Test edge cases and error handling for view mode functions."""

    def test_select_view_mode_with_none_tab_key(self, mock_streamlit):
        """Test select_view_mode handles None tab key gracefully."""
        mock_streamlit["columns"].return_value = [MagicMock(), MagicMock()]
        mock_streamlit["selectbox"].return_value = "List"

        # Should handle None gracefully by converting to string
        view_mode, grid_columns = select_view_mode(None)

        assert view_mode == "List"
        assert grid_columns is None

        # Verify key generation with None
        selectbox_call = mock_streamlit["selectbox"].call_args
        assert selectbox_call[1]["key"] == "view_mode_None"

    def test_apply_view_mode_with_none_jobs(self):
        """Test apply_view_mode handles None jobs gracefully."""
        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            # Should not crash with None jobs
            apply_view_mode(None, "List")
            mock_list.assert_called_once_with(None)

    def test_apply_view_mode_with_negative_grid_columns(self, sample_jobs_dto):
        """Test apply_view_mode handles negative grid columns."""
        jobs = sample_jobs_dto

        with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
            # Negative columns should fallback to list
            apply_view_mode(jobs, "Grid", grid_columns=-1)
            mock_list.assert_called_once_with(jobs)

    def test_view_mode_functions_with_mock_jobs(self):
        """Test view mode functions work with various job data types."""
        # Test with mock objects that behave like jobs
        mock_jobs = [Mock(), Mock(), Mock()]

        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            apply_view_mode(mock_jobs, "Grid", grid_columns=2)
            mock_grid.assert_called_once_with(mock_jobs, num_columns=2)

    def test_select_view_mode_columns_layout_edge_cases(self, mock_streamlit):
        """Test select_view_mode column layout with edge cases."""
        # Test when columns return fewer than expected objects
        mock_streamlit["columns"].side_effect = [
            [MagicMock()],  # Only one column instead of two
            [MagicMock(), MagicMock()],  # Normal second call
        ]
        mock_streamlit["selectbox"].side_effect = ["Grid", 3]

        # Should handle gracefully
        view_mode, grid_columns = select_view_mode("edge_test")
        assert view_mode == "Grid"
        assert grid_columns == 3

    def test_apply_view_mode_extreme_column_values(self, sample_jobs_dto):
        """Test apply_view_mode with extreme column values."""
        jobs = sample_jobs_dto

        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            # Test very large column count
            apply_view_mode(jobs, "Grid", grid_columns=1000)
            mock_grid.assert_called_once_with(jobs, num_columns=1000)


class TestViewModeIntegration:
    """Integration tests for view mode functions working together."""

    def test_full_view_mode_workflow(self, mock_streamlit, sample_jobs_dto):
        """Test complete workflow of selecting and applying view mode."""
        # Mock select_view_mode components
        mock_streamlit["columns"].side_effect = [
            [MagicMock(), MagicMock()],  # Main layout
            [MagicMock(), MagicMock()],  # Grid columns layout
        ]
        mock_streamlit["selectbox"].side_effect = ["Grid", 3]

        # Select view mode
        view_mode, grid_columns = select_view_mode("workflow_test")

        # Apply view mode with selected settings
        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            apply_view_mode(sample_jobs_dto, view_mode, grid_columns)

            # Verify complete workflow
            assert view_mode == "Grid"
            assert grid_columns == 3
            mock_grid.assert_called_once_with(sample_jobs_dto, num_columns=3)

    def test_view_mode_state_persistence_simulation(self, mock_streamlit):
        """Test simulating view mode state persistence across selections."""
        jobs = [Mock() for _ in range(5)]  # Mock jobs

        # Simulate multiple view mode selections with different tabs
        tab_configs = [
            ("jobs_tab", "Grid", 2),
            ("favorites_tab", "List", None),
            ("archived_tab", "Grid", 4),
        ]

        for tab_key, expected_mode, expected_columns in tab_configs:
            # Mock the selection for this tab
            if expected_mode == "Grid":
                mock_streamlit["columns"].side_effect = [
                    [MagicMock(), MagicMock()],
                    [MagicMock(), MagicMock()],
                ]
                mock_streamlit["selectbox"].side_effect = [
                    expected_mode,
                    expected_columns,
                ]
            else:
                mock_streamlit["columns"].return_value = [MagicMock(), MagicMock()]
                mock_streamlit["selectbox"].return_value = expected_mode

            # Select and apply view mode
            view_mode, grid_columns = select_view_mode(tab_key)

            # Verify each tab gets correct configuration
            assert view_mode == expected_mode
            assert grid_columns == expected_columns

            # Apply the view mode
            if expected_mode == "Grid":
                with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
                    apply_view_mode(jobs, view_mode, grid_columns)
                    mock_grid.assert_called_once_with(
                        jobs, num_columns=expected_columns
                    )
            else:
                with patch("src.ui.helpers.view_mode.render_jobs_list") as mock_list:
                    apply_view_mode(jobs, view_mode)
                    mock_list.assert_called_once_with(jobs)

            # Reset mocks for next iteration
            mock_streamlit["columns"].reset_mock()
            mock_streamlit["selectbox"].reset_mock()

    def test_view_mode_with_dynamic_job_updates(self, sample_jobs_dto):
        """Test view mode handling with dynamically changing job lists."""
        initial_jobs = sample_jobs_dto[:2]
        updated_jobs = sample_jobs_dto  # Full list

        with patch("src.ui.helpers.view_mode.render_jobs_grid") as mock_grid:
            # Initial render with fewer jobs
            apply_view_mode(initial_jobs, "Grid", grid_columns=2)
            mock_grid.assert_called_with(initial_jobs, num_columns=2)

            mock_grid.reset_mock()

            # Updated render with more jobs (same view mode)
            apply_view_mode(updated_jobs, "Grid", grid_columns=2)
            mock_grid.assert_called_with(updated_jobs, num_columns=2)

    def test_view_mode_error_recovery(self, mock_streamlit):
        """Test view mode functions recover gracefully from errors."""
        # Mock selectbox to raise an exception first time, then work normally
        mock_streamlit["columns"].return_value = [MagicMock(), MagicMock()]
        mock_streamlit["selectbox"].side_effect = [Exception("Mock error"), "List"]

        # First call should raise exception
        with pytest.raises(Exception):
            select_view_mode("error_test")

        # Reset side effect for normal operation
        mock_streamlit["selectbox"].side_effect = None
        mock_streamlit["selectbox"].return_value = "List"

        # Second call should work normally
        view_mode, grid_columns = select_view_mode("recovery_test")
        assert view_mode == "List"
        assert grid_columns is None
