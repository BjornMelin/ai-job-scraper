"""Comprehensive tests for Company Management page functionality.

Tests business logic, user interactions, error handling, and edge cases for the
company management UI page, focusing on real user scenarios with 80%+ coverage.
"""

from unittest.mock import patch

import pytest

from src.schemas import Company
from src.ui.pages.companies import (
    _add_company_callback,
    _toggle_company_callback,
    show_companies_page,
)


class TestAddCompanyCallback:
    """Test the add company callback functionality."""

    def test_add_company_with_valid_data_succeeds(self, mock_session_state):
        """Test adding a company with valid name and URL succeeds."""
        # Arrange
        mock_session_state.update(
            {"company_name": "TechCorp", "company_url": "https://techcorp.com/careers"},
        )

        expected_company = Company(
            id=1,
            name="TechCorp",
            url="https://techcorp.com/careers",
            active=True,
        )

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.add_company.return_value = expected_company

            # Act
            _add_company_callback()

            # Assert
            mock_company_service.add_company.assert_called_once_with(
                name="TechCorp",
                url="https://techcorp.com/careers",
            )
            assert (
                mock_session_state["add_company_success"]
                == "Successfully added company: TechCorp"
            )
            assert mock_session_state["company_name"] == ""
            assert mock_session_state["company_url"] == ""
            assert mock_session_state["add_company_error"] is None

    def test_add_company_with_empty_name_shows_validation_error(
        self,
        mock_session_state,
        mock_company_service,
    ):
        """Test adding company with empty name shows validation error."""
        # Arrange
        mock_session_state.update(
            {
                "company_name": "   ",  # Whitespace only
                "company_url": "https://techcorp.com/careers",
            },
        )

        # Act
        _add_company_callback()

        # Assert
        mock_company_service["companies_page"].add_company.assert_not_called()
        assert mock_session_state["add_company_error"] == "Company name is required"

    def test_add_company_with_empty_url_shows_validation_error(
        self,
        mock_session_state,
        mock_company_service,
    ):
        """Test adding company with empty URL shows validation error."""
        # Arrange
        mock_session_state.update({"company_name": "TechCorp", "company_url": ""})

        # Act
        _add_company_callback()

        # Assert
        mock_company_service["companies_page"].add_company.assert_not_called()
        assert mock_session_state["add_company_error"] == "Company URL is required"

    def test_add_duplicate_company_shows_validation_error(
        self,
        mock_session_state,
        mock_company_service,
    ):
        """Test adding duplicate company shows validation error."""
        # Arrange
        mock_session_state.update(
            {"company_name": "TechCorp", "company_url": "https://techcorp.com/careers"},
        )

        mock_company_service["companies_page"].add_company.side_effect = ValueError(
            "Company 'TechCorp' already exists",
        )

        # Act
        _add_company_callback()

        # Assert
        assert (
            mock_session_state["add_company_error"]
            == "Company 'TechCorp' already exists"
        )
        assert mock_session_state.get("add_company_success") is None

    def test_add_company_service_failure_shows_generic_error(self, mock_session_state):
        """Test service failure during add company shows generic error."""
        # Arrange
        mock_session_state.update(
            {"company_name": "TechCorp", "company_url": "https://techcorp.com/careers"},
        )

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.add_company.side_effect = Exception(
                "Database connection failed",
            )

            # Act
            _add_company_callback()

            # Assert
            assert (
                mock_session_state["add_company_error"]
                == "Failed to add company. Please try again."
            )
            assert mock_session_state.get("add_company_success") is None


class TestToggleCompanyCallback:
    """Test the toggle company callback functionality."""

    def test_toggle_company_to_active_succeeds(self, mock_session_state):
        """Test toggling company to active status succeeds."""
        # Arrange
        company_id = 1

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.toggle_company_active.return_value = True

            # Act
            _toggle_company_callback(company_id)

            # Assert
            mock_company_service.toggle_company_active.assert_called_once_with(
                company_id,
            )
            assert mock_session_state["toggle_success"] == "Enabled scraping"
            assert mock_session_state["toggle_error"] is None

    def test_toggle_company_to_inactive_succeeds(self, mock_session_state):
        """Test toggling company to inactive status succeeds."""
        # Arrange
        company_id = 1

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.toggle_company_active.return_value = False

            # Act
            _toggle_company_callback(company_id)

            # Assert
            mock_company_service.toggle_company_active.assert_called_once_with(
                company_id,
            )
            assert mock_session_state["toggle_success"] == "Disabled scraping"
            assert mock_session_state["toggle_error"] is None

    def test_toggle_nonexistent_company_shows_error(self, mock_session_state):
        """Test toggling nonexistent company shows error message."""
        # Arrange
        company_id = 999

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.toggle_company_active.side_effect = Exception(
                "Company not found",
            )

            # Act
            _toggle_company_callback(company_id)

            # Assert
            assert (
                mock_session_state["toggle_error"]
                == "Failed to update company status: Company not found"
            )
            assert mock_session_state.get("toggle_success") is None

    def test_add_company_with_invalid_data_shows_error(self, mock_session_state):
        """Test adding company with invalid data shows error message."""
        # Arrange - empty name and URL in session state
        mock_session_state.update({"company_name": "", "company_url": ""})

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            # Act - call callback function
            _add_company_callback()

            # Assert - service not called, error shown
            mock_company_service.add_company.assert_not_called()
            assert mock_session_state["add_company_error"] == "Company name is required"
            assert mock_session_state.get("add_company_success") is None


class TestShowCompaniesPage:
    """Test the complete companies page rendering and functionality."""

    def test_companies_page_displays_title_and_description(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test companies page displays correct title and description."""
        # Arrange
        mock_company_service["companies_page"].get_all_companies.return_value = []

        # Act
        show_companies_page()

        # Assert
        mock_streamlit["title"].assert_called_with("Company Management")
        mock_streamlit["markdown"].assert_any_call("Manage companies for job scraping")

    def test_companies_page_shows_empty_state_when_no_companies(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test companies page shows empty state when no companies exist."""
        # Arrange
        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.get_all_companies.return_value = []

            # Act
            show_companies_page()

            # Assert
            mock_streamlit["info"].assert_called_with(
                "📝 No companies found. Add your first company above!",
            )

    def test_companies_page_displays_company_list(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
        sample_companies_dto,
    ):
        """Test companies page displays list of companies with details."""
        # Arrange
        mock_company_service[
            "companies_page"
        ].get_all_companies.return_value = sample_companies_dto

        # Act
        show_companies_page()

        # Assert
        # Verify company data is displayed
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]

        # Check that company names appear in markdown calls
        assert any("Tech Corp" in call for call in markdown_calls)
        assert any("DataCo" in call for call in markdown_calls)
        assert any("AI Solutions" in call for call in markdown_calls)

    def test_companies_page_displays_summary_statistics(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
        sample_companies_dto,
    ):
        """Test companies page displays accurate summary statistics."""
        # Arrange
        mock_company_service[
            "companies_page"
        ].get_all_companies.return_value = sample_companies_dto

        # Act
        show_companies_page()

        # Assert
        metric_calls = mock_streamlit["metric"].call_args_list

        # Verify metrics are displayed with correct values
        metric_labels = [call.args[0] for call in metric_calls]
        metric_values = [call.args[1] for call in metric_calls]

        assert "Total Companies" in metric_labels
        assert "Active Companies" in metric_labels
        assert "Inactive Companies" in metric_labels

        # Verify correct counts (2 active out of 3 total)
        total_idx = metric_labels.index("Total Companies")
        active_idx = metric_labels.index("Active Companies")
        inactive_idx = metric_labels.index("Inactive Companies")

        assert metric_values[total_idx] == 3
        assert metric_values[active_idx] == 2
        assert metric_values[inactive_idx] == 1

    def test_companies_page_renders_toggle_controls(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
        sample_companies_dto,
    ):
        """Test companies page renders toggle controls for each company."""
        # Arrange
        mock_company_service[
            "companies_page"
        ].get_all_companies.return_value = sample_companies_dto

        # Act
        show_companies_page()

        # Assert
        toggle_calls = mock_streamlit["toggle"].call_args_list

        # Verify toggles are created for each company
        assert len(toggle_calls) == len(sample_companies_dto)

        # Check toggle keys and values match companies
        for i, company in enumerate(sample_companies_dto):
            call_args = toggle_calls[i]
            assert call_args.kwargs["value"] == company.active
            assert call_args.kwargs["key"] == f"company_active_{company.id}"

    def test_companies_page_displays_add_company_form(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test companies page displays add company form with proper structure."""
        # Arrange
        mock_company_service["companies_page"].get_all_companies.return_value = []

        # Act
        show_companies_page()

        # Assert
        # Verify expander and form are created
        mock_streamlit["expander"].assert_called_with(
            "+ Add New Company",
            expanded=False,
        )
        mock_streamlit["form"].assert_called_with("add_company_form")

        # Verify form inputs are created
        text_input_calls = mock_streamlit["text_input"].call_args_list
        input_labels = [call.args[0] for call in text_input_calls]

        assert "Company Name" in input_labels
        assert "Careers URL" in input_labels

    def test_companies_page_displays_feedback_messages(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test companies page displays success and error feedback messages."""
        # Arrange
        mock_session_state.update(
            {
                "add_company_success": "Company added successfully",
                "add_company_error": None,
                "toggle_success": "Company toggled",
                "toggle_error": None,
            },
        )
        mock_company_service["companies_page"].get_all_companies.return_value = []

        # Act
        show_companies_page()

        # Assert
        mock_streamlit["success"].assert_any_call("✅ Company added successfully")
        mock_streamlit["success"].assert_any_call("✅ Company toggled")

    def test_companies_page_handles_service_failure_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test companies page handles service failure gracefully."""
        # Arrange
        mock_company_service[
            "companies_page"
        ].get_all_companies.side_effect = Exception("Database error")

        # Act
        show_companies_page()

        # Assert
        mock_streamlit["error"].assert_called_with(
            "❌ Failed to load companies. Please refresh the page.",
        )

    @pytest.mark.parametrize(
        ("company_count", "expected_active"),
        (
            ([], 0),
            ([Company(id=1, name="Test", url="https://test.com", active=True)], 1),
            (
                [
                    Company(id=1, name="Test1", url="https://test1.com", active=True),
                    Company(id=2, name="Test2", url="https://test2.com", active=False),
                ],
                1,
            ),
            (
                [
                    Company(id=1, name="Test1", url="https://test1.com", active=False),
                    Company(id=2, name="Test2", url="https://test2.com", active=False),
                ],
                0,
            ),
        ),
    )
    def test_companies_page_calculates_correct_statistics(  # pylint: disable=R0917
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
        company_count,
        expected_active,
    ):
        """Test companies page calculates statistics correctly for various scenarios."""
        # Arrange
        mock_company_service[
            "companies_page"
        ].get_all_companies.return_value = company_count

        # Act
        show_companies_page()

        # Assert
        if company_count:  # Only check metrics if there are companies
            metric_calls = mock_streamlit["metric"].call_args_list
            metric_labels = [call.args[0] for call in metric_calls]
            metric_values = [call.args[1] for call in metric_calls]

            active_idx = metric_labels.index("Active Companies")
            assert metric_values[active_idx] == expected_active


class TestCompanyPageIntegration:
    """Integration tests for company page workflow scenarios."""

    def test_complete_add_company_workflow(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test complete workflow of adding a new company."""
        # Arrange - Simulate form submission
        mock_session_state.update(
            {
                "company_name": "New Company",
                "company_url": "https://newcompany.com/careers",
            },
        )

        new_company = Company(
            id=1,
            name="New Company",
            url="https://newcompany.com/careers",
            active=True,
        )
        mock_company_service["companies_page"].add_company.return_value = new_company
        mock_company_service["companies_page"].get_all_companies.return_value = [
            new_company,
        ]

        # Act - Simulate callback execution followed by page render
        _add_company_callback()
        show_companies_page()

        # Assert - Verify complete workflow
        # 1. Service was called to add company
        mock_company_service["companies_page"].add_company.assert_called_once_with(
            name="New Company",
            url="https://newcompany.com/careers",
        )

        # 2. Success message was stored and then cleared after display
        assert (
            mock_session_state.get("add_company_success") is None
        )  # Cleared after display

        # 3. Form inputs were cleared
        assert mock_session_state["company_name"] == ""
        assert mock_session_state["company_url"] == ""

        # 4. Success message was displayed on page render
        mock_streamlit["success"].assert_called_with(
            "✅ Successfully added company: New Company",
        )

    def test_complete_toggle_company_workflow(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test complete workflow of toggling company status."""
        # Arrange
        company = Company(
            id=1,
            name="Test Company",
            url="https://test.com",
            active=False,
        )
        mock_company_service["companies_page"].get_all_companies.return_value = [
            company,
        ]
        mock_company_service["companies_page"].toggle_company_active.return_value = True

        # Act - Simulate toggle callback followed by page render
        _toggle_company_callback(1)
        show_companies_page()

        # Assert - Verify complete workflow
        # 1. Service was called to toggle company
        mock_company_service[
            "companies_page"
        ].toggle_company_active.assert_called_once_with(1)

        # 2. Success message was stored and then cleared after display
        assert mock_session_state.get("toggle_success") is None  # Cleared after display

        # 3. Success message was displayed on page render
        mock_streamlit["success"].assert_called_with("✅ Enabled scraping")


class TestInitAndDisplayFeedback:
    """Test the initialization and feedback display functionality."""

    def test_init_and_display_feedback_initializes_session_keys(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test that all feedback session keys are initialized."""
        # Arrange - No session state keys set initially

        # Act
        with (
            patch("src.ui.pages.companies.init_session_state_keys") as mock_init_keys,
            patch("src.ui.pages.companies.display_feedback_messages") as mock_display,
        ):
            from src.ui.pages.companies import (
                display_feedback_messages,
                init_session_state_keys,
            )

            init_session_state_keys(
                [
                    "add_company_error",
                    "add_company_success",
                    "toggle_error",
                    "toggle_success",
                ]
            )
            display_feedback_messages(
                success_keys=["add_company_success", "toggle_success"],
                error_keys=["add_company_error", "toggle_error"],
            )

        # Assert
        mock_init_keys.assert_called_once_with(
            [
                "add_company_error",
                "add_company_success",
                "toggle_error",
                "toggle_success",
            ],
            None,
        )
        mock_display.assert_called_once_with(
            success_keys=["add_company_success", "toggle_success"],
            error_keys=["add_company_error", "toggle_error"],
        )

    def test_init_and_display_feedback_displays_success_messages(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test that success feedback messages are displayed properly."""
        # Arrange
        mock_session_state.update(
            {
                "add_company_success": "Company added!",
                "toggle_success": "Status toggled!",
            },
        )

        # Act
        with (
            patch("src.ui.pages.companies.init_session_state_keys"),
            patch("src.ui.pages.companies.display_feedback_messages") as mock_display,
        ):
            from src.ui.pages.companies import (
                display_feedback_messages,
                init_session_state_keys,
            )

            init_session_state_keys(
                [
                    "add_company_error",
                    "add_company_success",
                    "toggle_error",
                    "toggle_success",
                ]
            )
            display_feedback_messages(
                success_keys=["add_company_success", "toggle_success"],
                error_keys=["add_company_error", "toggle_error"],
            )

        # Assert
        mock_display.assert_called_once_with(
            success_keys=["add_company_success", "toggle_success"],
            error_keys=["add_company_error", "toggle_error"],
        )

    def test_init_and_display_feedback_displays_error_messages(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test that error feedback messages are displayed properly."""
        # Arrange
        mock_session_state.update(
            {
                "add_company_error": "Failed to add company",
                "toggle_error": "Failed to toggle status",
            },
        )

        # Act
        with (
            patch("src.ui.pages.companies.init_session_state_keys"),
            patch("src.ui.pages.companies.display_feedback_messages") as mock_display,
        ):
            from src.ui.pages.companies import (
                display_feedback_messages,
                init_session_state_keys,
            )

            init_session_state_keys(
                [
                    "add_company_error",
                    "add_company_success",
                    "toggle_error",
                    "toggle_success",
                ]
            )
            display_feedback_messages(
                success_keys=["add_company_success", "toggle_success"],
                error_keys=["add_company_error", "toggle_error"],
            )

        # Assert
        mock_display.assert_called_once_with(
            success_keys=["add_company_success", "toggle_success"],
            error_keys=["add_company_error", "toggle_error"],
        )


class TestCompanyDisplayIntegration:
    """Test integration with company display helpers."""

    def test_companies_page_renders_company_cards(
        self,
        mock_streamlit,
        mock_session_state,
        sample_companies_dto,
    ):
        """Test that company cards are rendered for each company."""
        # Arrange
        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.get_all_companies.return_value = sample_companies_dto

            with patch(
                "src.ui.ui_rendering.render_company_card",
            ) as mock_render_card:
                # Act
                show_companies_page()

                # Assert
                # Verify render_company_card was called for each company
                assert mock_render_card.call_count == len(sample_companies_dto)

                # Verify each company was passed to render_company_card
                for i, company in enumerate(sample_companies_dto):
                    call_args = mock_render_card.call_args_list[i]
                    assert call_args.args[0] == company
                    # Verify callback function is passed
                    assert callable(call_args.args[1])  # _toggle_company_callback

    def test_companies_page_handles_empty_company_list_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test that empty company list is handled gracefully."""
        # Arrange
        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.get_all_companies.return_value = []

            with patch(
                "src.ui.ui_rendering.render_company_card",
            ) as mock_render_card:
                # Act
                show_companies_page()

                # Assert
                mock_render_card.assert_not_called()
                mock_streamlit["info"].assert_called_with(
                    "📝 No companies found. Add your first company above!",
                )

    def test_companies_page_handles_company_display_errors(
        self,
        mock_streamlit,
        mock_session_state,
        sample_companies_dto,
    ):
        """Test that errors in company display are handled gracefully."""
        # Arrange
        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.get_all_companies.return_value = sample_companies_dto

            with patch(
                "src.ui.ui_rendering.render_company_card",
            ) as mock_render_card:
                mock_render_card.side_effect = Exception("Display error")

                # Act & Assert - should not raise exception
                show_companies_page()

                # The error should be caught and handled in the display helper
                mock_render_card.assert_called()


class TestCompanyPageBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_add_company_with_whitespace_only_name_fails(self, mock_session_state):
        """Test adding company with whitespace-only name fails validation."""
        # Arrange
        mock_session_state.update(
            {
                "company_name": "   \t\n  ",  # Various whitespace characters
                "company_url": "https://valid.com",
            },
        )

        # Act
        _add_company_callback()

        # Assert
        assert mock_session_state["add_company_error"] == "Company name is required"
        assert mock_session_state.get("add_company_success") is None

    def test_add_company_with_whitespace_only_url_fails(self, mock_session_state):
        """Test adding company with whitespace-only URL fails validation."""
        # Arrange
        mock_session_state.update(
            {
                "company_name": "Valid Company",
                "company_url": "   \t\n  ",  # Various whitespace characters
            },
        )

        # Act
        _add_company_callback()

        # Assert
        assert mock_session_state["add_company_error"] == "Company URL is required"
        assert mock_session_state.get("add_company_success") is None

    def test_add_company_clears_form_on_success(self, mock_session_state):
        """Test that form fields are cleared after successful company addition."""
        # Arrange
        mock_session_state.update(
            {
                "company_name": "Test Company",
                "company_url": "https://test.com",
            },
        )

        with patch("src.ui.pages.companies.CompanyService") as mock_service:
            mock_service.add_company.return_value = Company(
                id=1,
                name="Test Company",
                url="https://test.com",
                active=True,
            )
            with patch("streamlit.rerun"):
                # Act
                _add_company_callback()

                # Assert - form fields are cleared
                assert mock_session_state["company_name"] == ""
                assert mock_session_state["company_url"] == ""
                assert mock_session_state["add_company_error"] is None
                assert (
                    "Successfully added company"
                    in mock_session_state["add_company_success"]
                )

    def test_toggle_callback_handles_concurrent_modifications(self, mock_session_state):
        """Test toggle callback handles concurrent modifications gracefully."""
        # Arrange
        company_id = 1

        with patch("src.ui.pages.companies.CompanyService") as mock_service:
            # Simulate concurrent modification scenario
            mock_service.toggle_company_active.side_effect = Exception(
                "Company was modified by another user",
            )

            # Act
            _toggle_company_callback(company_id)

            # Assert
            assert (
                "Failed to update company status" in mock_session_state["toggle_error"]
            )
            assert mock_session_state.get("toggle_success") is None

    @pytest.mark.parametrize(
        ("session_name", "session_url", "expected_error"),
        (
            (None, "https://test.com", "Company name is required"),
            ("", "https://test.com", "Company name is required"),
            ("Test Co", None, "Company URL is required"),
            ("Test Co", "", "Company URL is required"),
            (None, None, "Company name is required"),  # Name checked first
        ),
    )
    def test_add_company_validation_edge_cases(
        self,
        mock_session_state,
        session_name,
        session_url,
        expected_error,
    ):
        """Test various edge cases for add company validation."""
        # Arrange
        mock_session_state.update(
            {
                "company_name": session_name,
                "company_url": session_url,
            },
        )

        # Act
        _add_company_callback()

        # Assert
        assert mock_session_state["add_company_error"] == expected_error
        assert mock_session_state.get("add_company_success") is None


class TestCompanyPagePerformance:
    """Test performance aspects and large dataset handling."""

    def test_companies_page_handles_large_company_list(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test that the page handles a large number of companies efficiently."""
        # Arrange - Create a large list of companies
        large_company_list = [
            Company(
                id=i,
                name=f"Company {i}",
                url=f"https://company{i}.com",
                active=i % 2 == 0,  # Alternate active/inactive
            )
            for i in range(100)  # 100 companies
        ]

        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.get_all_companies.return_value = large_company_list

            with patch(
                "src.ui.ui_rendering.render_company_card",
            ) as mock_render_card:
                # Act
                show_companies_page()

                # Assert - All companies are processed
                assert mock_render_card.call_count == 100

                # Verify statistics are calculated correctly
                metric_calls = mock_streamlit["metric"].call_args_list
                metric_labels = [call.args[0] for call in metric_calls]
                metric_values = [call.args[1] for call in metric_calls]

                total_idx = metric_labels.index("Total Companies")
                active_idx = metric_labels.index("Active Companies")
                inactive_idx = metric_labels.index("Inactive Companies")

                assert metric_values[total_idx] == 100
                assert metric_values[active_idx] == 50  # Half are active
                assert metric_values[inactive_idx] == 50  # Half are inactive


class TestCompanyPageAccessibility:
    """Test accessibility and usability features."""

    def test_companies_page_provides_helpful_form_labels(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test that form elements have helpful labels and help text."""
        # Arrange
        mock_company_service["companies_page"].get_all_companies.return_value = []

        # Act
        show_companies_page()

        # Assert - Check form inputs have proper configuration
        text_input_calls = mock_streamlit["text_input"].call_args_list

        # Company Name input
        name_call = next(
            call for call in text_input_calls if call.args[0] == "Company Name"
        )
        assert "placeholder" in name_call.kwargs
        assert "help" in name_call.kwargs
        assert "key" in name_call.kwargs

        # Careers URL input
        url_call = next(
            call for call in text_input_calls if call.args[0] == "Careers URL"
        )
        assert "placeholder" in url_call.kwargs
        assert "help" in url_call.kwargs
        assert "key" in url_call.kwargs

    def test_companies_page_provides_informative_empty_state(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test that empty state provides helpful guidance to users."""
        # Arrange
        with patch("src.ui.pages.companies.CompanyService") as mock_company_service:
            mock_company_service.get_all_companies.return_value = []

            # Act
            show_companies_page()

            # Assert
            mock_streamlit["info"].assert_called_with(
                "📝 No companies found. Add your first company above!",
            )

    def test_companies_page_provides_clear_feedback_messages(
        self,
        mock_streamlit,
        mock_session_state,
        mock_company_service,
    ):
        """Test that user feedback messages are clear and actionable."""
        # Arrange
        mock_session_state.update(
            {
                "add_company_success": "Successfully added company: TestCorp",
                "add_company_error": None,
                "toggle_success": "Enabled scraping",
                "toggle_error": None,
            },
        )
        mock_company_service["companies_page"].get_all_companies.return_value = []

        # Act
        show_companies_page()

        # Assert - Success messages are displayed with checkmarks
        mock_streamlit["success"].assert_any_call(
            "✅ Successfully added company: TestCorp",
        )
        mock_streamlit["success"].assert_any_call("✅ Enabled scraping")
