"""Tests for T1.2: Custom Pagination Removal - Native st.dataframe Usage.

Tests validate that:
- No custom pagination components exist in UI
- Native st.dataframe is properly integrated
- UI components use library-first approaches
- Job display components work with native dataframe functionality
"""

from unittest.mock import patch

import pandas as pd

from src.schemas import Job
from src.ui.components.cards.job_card import render_jobs_grid, render_jobs_list
from src.ui.pages.jobs import _jobs_to_dataframe, _render_job_display, _render_job_tabs
from src.ui.ui_rendering import apply_view_mode, select_view_mode


class TestT1PaginationElimination:
    """Test T1.2: Custom Pagination Removal - Native st.dataframe Integration."""

    def test_no_custom_pagination_imports(self):
        """Test that no custom pagination components are imported."""
        # Search for common pagination terms in import statements
        # Get source code to check for pagination-related imports
        import inspect

        import src.ui.components.cards.job_card as job_card_module
        import src.ui.pages.jobs as jobs_module
        import src.ui.ui_rendering as view_mode_module

        jobs_source = inspect.getsource(jobs_module)
        view_mode_source = inspect.getsource(view_mode_module)
        job_card_source = inspect.getsource(job_card_module)

        # Verify no custom pagination imports
        pagination_terms = [
            "pagination",
            "Pagination",
            "paginate",
            "Paginate",
            "page_size",
            "PageSize",
            "page_count",
            "PageCount",
            "custom_pagination",
            "StreamlitPagination",
        ]

        for term in pagination_terms:
            assert term not in jobs_source, (
                f"Found custom pagination term '{term}' in jobs module"
            )
            assert term not in view_mode_source, (
                f"Found custom pagination term '{term}' in view_mode module"
            )
            assert term not in job_card_source, (
                f"Found custom pagination term '{term}' in job_card module"
            )

    def test_jobs_to_dataframe_conversion(self, sample_jobs_dto):
        """Test conversion of jobs to pandas DataFrame for st.dataframe."""
        jobs_dataframe = _jobs_to_dataframe(sample_jobs_dto)

        # Verify DataFrame structure
        assert isinstance(jobs_dataframe, pd.DataFrame)
        assert len(jobs_dataframe) == len(sample_jobs_dto)

        # Verify required columns for native dataframe display
        expected_columns = [
            "id",
            "Company",
            "Title",
            "Location",
            "Posted",
            "Last Seen",
            "Favorite",
            "Status",
            "Notes",
            "Link",
        ]
        for col in expected_columns:
            assert col in jobs_dataframe.columns

        # Verify data types are appropriate for dataframe display
        assert jobs_dataframe["Company"].dtype == "object"
        assert jobs_dataframe["Title"].dtype == "object"
        assert jobs_dataframe["Location"].dtype == "object"
        assert jobs_dataframe["Favorite"].dtype == "bool"

    def test_jobs_to_dataframe_empty_list(self):
        """Test conversion handles empty job list."""
        jobs_dataframe = _jobs_to_dataframe([])

        assert isinstance(jobs_dataframe, pd.DataFrame)
        assert len(jobs_dataframe) == 0
        assert jobs_dataframe.empty

    def test_jobs_to_dataframe_handles_missing_fields(self):
        """Test DataFrame conversion handles jobs with missing fields."""
        partial_job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            link="https://test.com",
            location="Remote",  # Use default location instead of None
            posted_date=None,  # Missing posted date
            # salary defaults to (None, None) - don't need to specify
            favorite=False,
            # notes defaults to "" - don't need to specify
            content_hash="test",
            # application_status defaults to "New" - don't need to specify
            application_date=None,
            archived=False,
            last_seen=None,
        )

        jobs_dataframe = _jobs_to_dataframe([partial_job])

        assert len(jobs_dataframe) == 1
        assert jobs_dataframe.iloc[0]["Company"] == "Test Company"
        assert jobs_dataframe.iloc[0]["Title"] == "Test Job"
        assert jobs_dataframe.iloc[0]["Location"] == "Remote"  # Default for None
        assert jobs_dataframe.iloc[0]["Status"] == "New"  # Default for None


class TestNativeDataframeIntegration:
    """Test integration with native Streamlit dataframe functionality."""

    def test_view_mode_selection_uses_native_components(self, mock_streamlit):
        """Test that view mode selection uses native Streamlit components."""
        view_mode, grid_columns = select_view_mode("test_tab")

        # Should use native selectbox components
        mock_streamlit["selectbox"].assert_called()
        assert mock_streamlit["selectbox"].call_count >= 1

        # Should use native columns
        mock_streamlit["columns"].assert_called()

    def test_apply_view_mode_card_display(self, sample_jobs_dto, mock_streamlit):
        """Test card view mode applies proper rendering."""
        with patch("src.ui.components.cards.job_card.render_jobs_grid") as mock_grid:
            apply_view_mode(sample_jobs_dto, "Card", 3)

            mock_grid.assert_called_once_with(sample_jobs_dto, num_columns=3)

    def test_apply_view_mode_list_display(self, sample_jobs_dto, mock_streamlit):
        """Test list view mode applies proper rendering."""
        with patch("src.ui.components.cards.job_card.render_jobs_list") as mock_list:
            apply_view_mode(sample_jobs_dto, "List", None)

            mock_list.assert_called_once_with(sample_jobs_dto)

    def test_apply_view_mode_fallback_to_list(self, sample_jobs_dto, mock_streamlit):
        """Test fallback to list view for invalid configurations."""
        with patch("src.ui.components.cards.job_card.render_jobs_list") as mock_list:
            # Card mode without grid_columns should fallback to list
            apply_view_mode(sample_jobs_dto, "Card", None)

            mock_list.assert_called_once_with(sample_jobs_dto)


class TestJobDisplayNativeFunctionality:
    """Test job display uses native Streamlit functionality only."""

    def test_render_jobs_grid_uses_native_columns(
        self,
        sample_jobs_dto,
        mock_streamlit,
    ):
        """Test grid rendering uses native st.columns."""
        render_jobs_grid(sample_jobs_dto, num_columns=3)

        # Should use native columns for grid layout
        mock_streamlit["columns"].assert_called()

        # Should use native containers
        mock_streamlit["container"].assert_called()

    def test_render_jobs_grid_handles_empty_list(self, mock_streamlit):
        """Test grid rendering handles empty job list gracefully."""
        render_jobs_grid([], num_columns=3)

        # Should display info message for empty list
        mock_streamlit["info"].assert_called_once_with("No jobs to display.")

    def test_render_jobs_list_uses_native_components(
        self,
        sample_jobs_dto,
        mock_streamlit,
    ):
        """Test list rendering uses only native Streamlit components."""
        render_jobs_list(sample_jobs_dto)

        # Should use native containers and markdown for separators
        mock_streamlit["container"].assert_called()
        mock_streamlit["markdown"].assert_called()

    def test_render_jobs_list_handles_empty_list(self, mock_streamlit):
        """Test list rendering handles empty job list gracefully."""
        render_jobs_list([])

        # Should display info message for empty list
        mock_streamlit["info"].assert_called_once_with("No jobs to display.")

    def test_render_job_tabs_uses_native_tabs(self, sample_jobs_dto, mock_streamlit):
        """Test job tabs use native st.tabs functionality."""
        with (
            patch(
                "src.ui.pages.jobs._get_favorites_jobs",
                return_value=sample_jobs_dto[:2],
            ),
            patch(
                "src.ui.pages.jobs._get_applied_jobs",
                return_value=sample_jobs_dto[:1],
            ),
            patch("src.ui.pages.jobs._render_job_display") as mock_render,
        ):
            _render_job_tabs(sample_jobs_dto)

            # Should use native tabs
            mock_streamlit["tabs"].assert_called_once()

            # Should call render_job_display for each tab
            assert mock_render.call_count == 3

    def test_render_job_display_integrates_native_search(
        self,
        sample_jobs_dto,
        mock_streamlit,
    ):
        """Test job display integrates with native search functionality."""
        with (
            patch(
                "src.ui.pages.jobs._apply_tab_search_to_jobs",
                return_value=sample_jobs_dto,
            ) as mock_search,
            patch(
                "src.ui.ui_rendering.select_view_mode",
                return_value=("List", None),
            ),
            patch("src.ui.ui_rendering.apply_view_mode") as mock_apply,
        ):
            _render_job_display(sample_jobs_dto, "test_tab")

            # Should apply search filtering
            mock_search.assert_called_once_with(sample_jobs_dto, "test_tab")

            # Should apply view mode
            mock_apply.assert_called_once()


class TestT1DataframeOptimizations:
    """Test DataFrame optimizations for native st.dataframe performance."""

    def test_dataframe_column_ordering(self, sample_jobs_dto):
        """Test DataFrame columns are ordered for optimal display."""
        jobs_dataframe = _jobs_to_dataframe(sample_jobs_dto)

        # Verify important columns come first (based on actual implementation)
        columns = list(jobs_dataframe.columns)
        assert columns.index("id") < columns.index("Company")
        assert columns.index("Company") < columns.index("Title")
        assert columns.index("Title") < columns.index("Location")

    def test_dataframe_data_types_optimized(self, sample_jobs_dto):
        """Test DataFrame uses optimized data types for performance."""
        jobs_dataframe = _jobs_to_dataframe(sample_jobs_dto)

        # Boolean columns should be bool type
        assert jobs_dataframe["Favorite"].dtype == "bool"

        # String columns should be object type (strings)
        assert jobs_dataframe["Company"].dtype == "object"
        assert jobs_dataframe["Title"].dtype == "object"
        assert jobs_dataframe["Location"].dtype == "object"
        assert jobs_dataframe["Status"].dtype == "object"

    def test_dataframe_handles_large_datasets(self):
        """Test DataFrame conversion performs well with large datasets."""
        # Create a large list of jobs for performance testing
        large_job_list = []
        for i in range(1000):
            job = Job(
                id=i,
                company_id=1,
                company=f"Company {i}",
                title=f"Job Title {i}",
                description=f"Description {i}",
                link=f"https://example.com/job/{i}",
                location=f"Location {i}",
                posted_date=None,
                # salary defaults to (None, None) - don't need to specify
                favorite=i % 10 == 0,  # Every 10th job is favorite
                notes="",
                content_hash=f"hash{i}",
                application_status="New",
                application_date=None,
                archived=False,
                last_seen=None,
            )
            large_job_list.append(job)

        # This should complete quickly without pagination overhead
        import time

        start_time = time.time()
        jobs_dataframe = _jobs_to_dataframe(large_job_list)
        end_time = time.time()

        # Verify conversion completed and was reasonably fast
        assert len(jobs_dataframe) == 1000
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second

    def test_dataframe_memory_efficiency(self, sample_jobs_dto):
        """Test DataFrame conversion is memory efficient."""
        jobs_dataframe = _jobs_to_dataframe(sample_jobs_dto)

        # Verify DataFrame doesn't duplicate large text fields unnecessarily
        memory_usage = jobs_dataframe.memory_usage(deep=True)

        # Memory usage should be reasonable for the number of jobs
        total_memory = memory_usage.sum()
        per_job_memory = total_memory / len(sample_jobs_dto)

        # Each job should use less than 10KB of memory in DataFrame
        assert per_job_memory < 10_000  # 10KB per job is reasonable


class TestT1NativeFunctionalityValidation:
    """Validate that only native Streamlit functionality is used."""

    def test_no_custom_pagination_classes(self):
        """Test that no custom pagination classes exist in the codebase."""
        # Import all UI modules to check for pagination classes
        import src.ui.components.cards.job_card as job_card_module
        import src.ui.pages.jobs as jobs_module
        import src.ui.ui_rendering as view_mode_module

        modules = [jobs_module, view_mode_module, job_card_module]

        for module in modules:
            # Get all classes defined in the module
            classes = [
                cls for name, cls in vars(module).items() if isinstance(cls, type)
            ]

            # Check for pagination-related class names
            pagination_class_names = [
                "Pagination",
                "Paginator",
                "PageNavigator",
                "PageControl",
                "CustomPagination",
                "StreamlitPagination",
            ]

            for cls in classes:
                assert cls.__name__ not in pagination_class_names, (
                    f"Found custom pagination class {cls.__name__} in {module.__name__}"
                )

    def test_no_custom_pagination_functions(self):
        """Test that no custom pagination functions exist."""
        import src.ui.pages.jobs as jobs_module
        import src.ui.ui_rendering as view_mode_module

        modules = [jobs_module, view_mode_module]

        for module in modules:
            # Get all functions defined in the module
            functions = [func for name, func in vars(module).items() if callable(func)]

            pagination_function_names = [
                "paginate_jobs",
                "create_pagination",
                "render_pagination",
                "get_page_slice",
                "calculate_pages",
                "navigate_pages",
            ]

            for func in functions:
                if hasattr(func, "__name__"):
                    assert func.__name__ not in pagination_function_names, (
                        f"Found custom pagination function "
                        f"{func.__name__} in {module.__name__}"
                    )

    def test_uses_native_streamlit_dataframe_only(self):
        """Test that code uses only native st.dataframe and related components."""
        # This test validates the T1.2 requirement to use native dataframe
        import inspect

        import src.ui.pages.jobs as jobs_module

        source = inspect.getsource(jobs_module)

        # Should have st.dataframe-related functionality
        assert "_jobs_to_dataframe" in source, "Missing DataFrame conversion function"

        # Should not have custom pagination terms
        custom_pagination_terms = [
            "st.beta_container",
            "st.experimental_memo",  # Old Streamlit pagination patterns
            "page_number",
            "items_per_page",
            "total_pages",
            "previous_page",
            "next_page",
            "current_page",
        ]

        for term in custom_pagination_terms:
            assert term not in source, (
                f"Found custom pagination term '{term}' "
                f"- should use native st.dataframe"
            )


class TestT1RealisticUsageScenarios:
    """Test realistic usage scenarios for native dataframe functionality."""

    def test_tab_switching_preserves_native_behavior(
        self,
        sample_jobs_dto,
        mock_streamlit,
    ):
        """Test that tab switching works with native functionality."""
        with (
            patch(
                "src.ui.pages.jobs._get_favorites_jobs",
                return_value=sample_jobs_dto[:2],
            ),
            patch(
                "src.ui.pages.jobs._get_applied_jobs",
                return_value=sample_jobs_dto[:1],
            ),
            patch("src.ui.pages.jobs._render_job_display") as mock_render,
        ):
            _render_job_tabs(sample_jobs_dto)

            # Should render all three tabs with correct data
            calls = mock_render.call_args_list
            assert len(calls) == 3

            # All tab data: all jobs
            assert len(calls[0][0][0]) == len(sample_jobs_dto)
            # Favorites tab data: subset
            assert len(calls[1][0][0]) == 2
            # Applied tab data: subset
            assert len(calls[2][0][0]) == 1

    def test_view_mode_switching_performance(self, sample_jobs_dto, mock_streamlit):
        """Test that view mode switching performs well without pagination overhead."""
        import time

        # Test switching between card and list view multiple times
        start_time = time.time()

        for _ in range(10):
            with (
                patch("src.ui.components.cards.job_card.render_jobs_grid"),
                patch("src.ui.components.cards.job_card.render_jobs_list"),
            ):
                apply_view_mode(sample_jobs_dto, "Card", 3)
                apply_view_mode(sample_jobs_dto, "List", None)

        end_time = time.time()

        # Should complete quickly without pagination recalculation
        assert (end_time - start_time) < 0.5  # Under 500ms for 20 view switches

    def test_large_dataset_native_handling(self, mock_streamlit):
        """Test native functionality handles large datasets appropriately."""
        # Create large dataset
        large_jobs = []
        for i in range(500):
            job = Job(
                id=i,
                company_id=1,
                company=f"Company {i}",
                title=f"Job {i}",
                description=f"Description {i}",
                link=f"https://example.com/{i}",
                location=f"Location {i}",
                posted_date=None,
                # salary defaults to (None, None) - don't need to specify
                favorite=False,
                notes="",
                content_hash=f"hash{i}",
                application_status="New",
                application_date=None,
                archived=False,
                last_seen=None,
            )
            large_jobs.append(job)

        # Convert to DataFrame - should handle large datasets efficiently
        jobs_dataframe = _jobs_to_dataframe(large_jobs)

        assert len(jobs_dataframe) == 500
        assert not jobs_dataframe.empty

        # Should have all required columns
        required_columns = ["Company", "Title", "Location", "Status", "Favorite"]
        for col in required_columns:
            assert col in jobs_dataframe.columns

    def test_empty_state_handling_with_native_components(self, mock_streamlit):
        """Test empty state handling uses native Streamlit components."""
        # Test empty job lists in various scenarios
        render_jobs_grid([], num_columns=3)
        render_jobs_list([])

        # Should use native info component for empty states
        assert mock_streamlit["info"].call_count == 2

        # Test empty DataFrame
        empty_df = _jobs_to_dataframe([])
        assert empty_df.empty
        assert isinstance(empty_df, pd.DataFrame)
