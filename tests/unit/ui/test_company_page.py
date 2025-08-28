"""Comprehensive tests for the Companies page UI component.

This module tests the Companies page functionality including:
- Company CRUD operations (Create, Read, Update, Delete)
- Form validation and submission
- Bulk operations (select all/none, bulk activate/deactivate/delete)
- Session state management and URL sync
- Real-time scraping progress display with fragments
- Error handling and user feedback
- Company statistics and metrics display
"""

import uuid

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from tests.utils.streamlit_utils import StreamlitComponentTester

from src.schemas import Company
from src.ui.pages.companies import (
    _add_company_callback,
    _bulk_activate_callback,
    _bulk_deactivate_callback,
    _bulk_delete_callback,
    _company_scraping_progress_fragment,
    _company_selection_callback,
    _delete_company_callback,
    _execute_bulk_delete,
    _init_and_display_feedback,
    _render_company_statistics,
    _select_all_callback,
    _select_none_callback,
    _show_bulk_delete_dialog,
    _toggle_company_callback,
    show_companies_page,
)


@pytest.fixture
def sample_companies():
    """Create sample Company objects for testing."""
    return [
        Company(
            id=1,
            name="Tech Corp",
            url="https://techcorp.com/careers",
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        Company(
            id=2,
            name="AI Startup",
            url="https://aistartup.com/jobs",
            active=False,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        Company(
            id=3,
            name="Scale Inc",
            url="https://scale.com/careers",
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
    ]


@pytest.fixture
def company_progress_data():
    """Create sample company progress data for testing."""
    from src.ui.utils.background_helpers import CompanyProgress

    return {
        "Tech Corp": CompanyProgress(
            company_name="Tech Corp",
            status="Completed",
            jobs_found=15,
            error=None,
            start_time=datetime.now(UTC),
        ),
        "AI Startup": CompanyProgress(
            company_name="AI Startup",
            status="In Progress",
            jobs_found=8,
            error=None,
            start_time=datetime.now(UTC),
        ),
        "Scale Inc": CompanyProgress(
            company_name="Scale Inc",
            status="Error",
            jobs_found=0,
            error="Connection timeout",
            start_time=datetime.now(UTC),
        ),
    }


class TestAddCompanyFunctionality:
    """Test add company functionality."""

    def test_add_company_callback_successful_creation(self):
        """Test successful company creation with valid inputs."""
        tester = StreamlitComponentTester(_add_company_callback)

        # Set up form inputs in session state
        tester.set_session_state(
            company_name="New Tech Corp", company_url="https://newtechcorp.com/careers"
        )

        # Mock company creation
        mock_company = Company(
            id=1,
            name="New Tech Corp",
            url="https://newtechcorp.com/careers",
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch(
                "src.services.company_service.CompanyService.add_company",
                return_value=mock_company,
            ) as mock_add,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component()

            # Verify company service was called correctly
            mock_add.assert_called_once_with(
                name="New Tech Corp", url="https://newtechcorp.com/careers"
            )

            # Verify success feedback is set
            state = tester.get_session_state()
            assert "Successfully added company: New Tech Corp" in state.get(
                "add_company_success", ""
            )
            assert state.get("add_company_error") is None

            # Verify form is cleared
            assert state.get("company_name") == ""
            assert state.get("company_url") == ""

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_add_company_callback_missing_name_validation(self):
        """Test validation error when company name is missing."""
        tester = StreamlitComponentTester(_add_company_callback)

        # Set up form inputs with missing name
        tester.set_session_state(
            company_name="", company_url="https://example.com/careers"
        )

        tester.run_component()

        # Verify error is set
        state = tester.get_session_state()
        assert state.get("add_company_error") == "Company name is required"
        assert state.get("add_company_success") is None

    def test_add_company_callback_missing_url_validation(self):
        """Test validation error when company URL is missing."""
        tester = StreamlitComponentTester(_add_company_callback)

        # Set up form inputs with missing URL
        tester.set_session_state(company_name="Valid Company", company_url="")

        tester.run_component()

        # Verify error is set
        state = tester.get_session_state()
        assert state.get("add_company_error") == "Company URL is required"
        assert state.get("add_company_success") is None

    def test_add_company_callback_whitespace_inputs_validation(self):
        """Test validation with whitespace-only inputs."""
        tester = StreamlitComponentTester(_add_company_callback)

        # Set up form inputs with whitespace
        tester.set_session_state(company_name="   ", company_url="   ")

        tester.run_component()

        # Verify name validation error (first validation check)
        state = tester.get_session_state()
        assert state.get("add_company_error") == "Company name is required"

    def test_add_company_callback_service_value_error(self):
        """Test handling of ValueError from company service."""
        tester = StreamlitComponentTester(_add_company_callback)

        tester.set_session_state(
            company_name="Duplicate Company",
            company_url="https://duplicate.com/careers",
        )

        with patch(
            "src.services.company_service.CompanyService.add_company",
            side_effect=ValueError("Company name already exists"),
        ):
            tester.run_component()

            # Verify error handling
            state = tester.get_session_state()
            assert state.get("add_company_error") == "Company name already exists"
            assert state.get("add_company_success") is None

    def test_add_company_callback_service_general_exception(self):
        """Test handling of general exception from company service."""
        tester = StreamlitComponentTester(_add_company_callback)

        tester.set_session_state(
            company_name="Test Company", company_url="https://test.com/careers"
        )

        with patch(
            "src.services.company_service.CompanyService.add_company",
            side_effect=Exception("Database connection failed"),
        ):
            tester.run_component()

            # Verify generic error handling
            state = tester.get_session_state()
            assert "Failed to add company. Please try again." in state.get(
                "add_company_error", ""
            )
            assert state.get("add_company_success") is None


class TestDeleteCompanyFunctionality:
    """Test delete company functionality."""

    def test_delete_company_callback_confirmed_deletion(self):
        """Test successful company deletion when confirmed."""
        tester = StreamlitComponentTester(_delete_company_callback)

        # Set confirmation state
        company_id = 1
        tester.set_session_state(**{f"delete_confirm_{company_id}": True})

        with (
            patch(
                "src.services.company_service.CompanyService.delete_company",
                return_value=True,
            ) as mock_delete,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component(company_id)

            # Verify deletion service was called
            mock_delete.assert_called_once_with(company_id)

            # Verify success feedback
            state = tester.get_session_state()
            assert state.get("delete_success") == "Company deleted successfully"

            # Verify confirmation state is cleared
            assert state.get(f"delete_confirm_{company_id}") is None

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_delete_company_callback_not_confirmed(self):
        """Test company deletion when not confirmed."""
        tester = StreamlitComponentTester(_delete_company_callback)

        company_id = 1
        # Don't set confirmation state

        with patch(
            "src.services.company_service.CompanyService.delete_company"
        ) as mock_delete:
            tester.run_component(company_id)

            # Verify deletion service is not called
            mock_delete.assert_not_called()

    def test_delete_company_callback_company_not_found(self):
        """Test handling when company to delete is not found."""
        tester = StreamlitComponentTester(_delete_company_callback)

        company_id = 999
        tester.set_session_state(**{f"delete_confirm_{company_id}": True})

        with patch(
            "src.services.company_service.CompanyService.delete_company",
            return_value=False,
        ):
            tester.run_component(company_id)

            # Verify error feedback
            state = tester.get_session_state()
            assert state.get("delete_error") == "Company not found"

    def test_delete_company_callback_service_exception(self):
        """Test handling of service exceptions during deletion."""
        tester = StreamlitComponentTester(_delete_company_callback)

        company_id = 1
        tester.set_session_state(**{f"delete_confirm_{company_id}": True})

        with patch(
            "src.services.company_service.CompanyService.delete_company",
            side_effect=Exception("Database error"),
        ):
            tester.run_component(company_id)

            # Verify error handling
            state = tester.get_session_state()
            assert "Failed to delete company: Database error" in state.get(
                "delete_error", ""
            )


class TestToggleCompanyFunctionality:
    """Test toggle company active status functionality."""

    def test_toggle_company_callback_activate(self):
        """Test toggling company to active status."""
        tester = StreamlitComponentTester(_toggle_company_callback)

        company_id = 1

        with patch(
            "src.services.company_service.CompanyService.toggle_company_active",
            return_value=True,
        ) as mock_toggle:
            tester.run_component(company_id)

            # Verify service was called
            mock_toggle.assert_called_once_with(company_id)

            # Verify success feedback for activation
            state = tester.get_session_state()
            assert state.get("toggle_success") == "Enabled scraping"
            assert state.get("toggle_error") is None

    def test_toggle_company_callback_deactivate(self):
        """Test toggling company to inactive status."""
        tester = StreamlitComponentTester(_toggle_company_callback)

        company_id = 1

        with patch(
            "src.services.company_service.CompanyService.toggle_company_active",
            return_value=False,
        ):
            tester.run_component(company_id)

            # Verify success feedback for deactivation
            state = tester.get_session_state()
            assert state.get("toggle_success") == "Disabled scraping"
            assert state.get("toggle_error") is None

    def test_toggle_company_callback_service_exception(self):
        """Test handling of service exceptions during toggle."""
        tester = StreamlitComponentTester(_toggle_company_callback)

        company_id = 1

        with patch(
            "src.services.company_service.CompanyService.toggle_company_active",
            side_effect=Exception("Service unavailable"),
        ):
            tester.run_component(company_id)

            # Verify error handling
            state = tester.get_session_state()
            assert "Failed to update company status: Service unavailable" in state.get(
                "toggle_error", ""
            )
            assert state.get("toggle_success") is None


class TestCompanySelectionFunctionality:
    """Test company selection functionality."""

    def test_company_selection_callback_select_company(self):
        """Test selecting a company updates selection state."""
        tester = StreamlitComponentTester(_company_selection_callback)

        company_id = 1

        # Set checkbox to checked
        tester.set_session_state(**{f"select_company_{company_id}": True})

        with patch(
            "src.ui.utils.url_state.update_url_from_company_selection"
        ) as mock_url_update:
            tester.run_component(company_id)

            # Verify company is added to selection
            state = tester.get_session_state()
            assert company_id in state.get("selected_companies", set())

            # Verify URL sync is called
            mock_url_update.assert_called_once()

    def test_company_selection_callback_deselect_company(self):
        """Test deselecting a company removes from selection state."""
        tester = StreamlitComponentTester(_company_selection_callback)

        company_id = 1

        # Initialize with company already selected
        tester.set_session_state(
            selected_companies={company_id}, **{f"select_company_{company_id}": False}
        )

        with patch(
            "src.ui.utils.url_state.update_url_from_company_selection"
        ) as mock_url_update:
            tester.run_component(company_id)

            # Verify company is removed from selection
            state = tester.get_session_state()
            assert company_id not in state.get("selected_companies", set())

            # Verify URL sync is called
            mock_url_update.assert_called_once()

    def test_company_selection_callback_initializes_selection_set(self):
        """Test selection callback initializes selection set if not present."""
        tester = StreamlitComponentTester(_company_selection_callback)

        company_id = 1

        # Don't initialize selected_companies
        tester.set_session_state(**{f"select_company_{company_id}": True})

        with patch("src.ui.utils.url_state.update_url_from_company_selection"):
            tester.run_component(company_id)

            # Verify selection set is initialized
            state = tester.get_session_state()
            assert isinstance(state.get("selected_companies"), set)
            assert company_id in state["selected_companies"]


class TestBulkSelectionOperations:
    """Test bulk selection operations."""

    def test_select_all_callback_success(self, sample_companies):
        """Test select all functionality."""
        tester = StreamlitComponentTester(_select_all_callback)

        with (
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=sample_companies,
            ),
            patch(
                "src.ui.utils.url_state.update_url_from_company_selection"
            ) as mock_url_update,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component()

            # Verify all companies are selected
            state = tester.get_session_state()
            expected_ids = {1, 2, 3}
            assert state.get("selected_companies") == expected_ids

            # Verify individual checkboxes are set
            for company_id in expected_ids:
                assert state.get(f"select_company_{company_id}") is True

            # Verify URL sync and rerun
            mock_url_update.assert_called_once()
            mock_rerun.assert_called_once()

    def test_select_all_callback_service_exception(self):
        """Test select all handles service exceptions."""
        tester = StreamlitComponentTester(_select_all_callback)

        with patch(
            "src.services.company_service.CompanyService.get_all_companies",
            side_effect=Exception("Service error"),
        ):
            tester.run_component()

            # Verify error is handled gracefully
            state = tester.get_session_state()
            assert "Failed to select all companies: Service error" in state.get(
                "bulk_operation_error", ""
            )

    def test_select_none_callback_success(self):
        """Test select none functionality."""
        tester = StreamlitComponentTester(_select_none_callback)

        # Initialize with some companies selected
        selected_companies = {1, 2, 3}
        initial_state = {"selected_companies": selected_companies}
        for company_id in selected_companies:
            initial_state[f"select_company_{company_id}"] = True

        tester.set_session_state(**initial_state)

        with (
            patch(
                "src.ui.utils.url_state.update_url_from_company_selection"
            ) as mock_url_update,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component()

            # Verify all selections are cleared
            state = tester.get_session_state()
            assert len(state.get("selected_companies", set())) == 0

            # Verify individual checkboxes are cleared
            for company_id in selected_companies:
                assert state.get(f"select_company_{company_id}") is False

            # Verify URL sync and rerun
            mock_url_update.assert_called_once()
            mock_rerun.assert_called_once()

    def test_select_none_callback_service_exception(self):
        """Test select none handles exceptions gracefully."""
        tester = StreamlitComponentTester(_select_none_callback)

        tester.set_session_state(selected_companies={1, 2})

        with patch(
            "src.ui.utils.url_state.update_url_from_company_selection",
            side_effect=Exception("URL sync error"),
        ):
            tester.run_component()

            # Verify error is handled
            state = tester.get_session_state()
            assert "Failed to clear selections: URL sync error" in state.get(
                "bulk_operation_error", ""
            )


class TestBulkOperations:
    """Test bulk operations on selected companies."""

    def test_bulk_activate_callback_success(self):
        """Test successful bulk activation."""
        tester = StreamlitComponentTester(_bulk_activate_callback)

        selected_companies = {1, 2, 3}
        tester.set_session_state(selected_companies=selected_companies)

        with (
            patch(
                "src.services.company_service.CompanyService.bulk_update_status",
                return_value=3,
            ) as mock_bulk_update,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component()

            # Verify service was called with correct parameters
            mock_bulk_update.assert_called_once_with([1, 2, 3], active=True)

            # Verify success feedback
            state = tester.get_session_state()
            assert (
                state.get("bulk_operation_success")
                == "Successfully activated 3 companies"
            )

            # Verify selections are cleared
            assert len(state.get("selected_companies", set())) == 0

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_bulk_activate_callback_no_selection(self):
        """Test bulk activation with no companies selected."""
        tester = StreamlitComponentTester(_bulk_activate_callback)

        # No companies selected
        tester.set_session_state(selected_companies=set())

        with patch(
            "src.services.company_service.CompanyService.bulk_update_status"
        ) as mock_bulk_update:
            tester.run_component()

            # Verify service is not called
            mock_bulk_update.assert_not_called()

            # Verify error feedback
            state = tester.get_session_state()
            assert (
                state.get("bulk_operation_error")
                == "No companies selected for activation"
            )

    def test_bulk_deactivate_callback_success(self):
        """Test successful bulk deactivation."""
        tester = StreamlitComponentTester(_bulk_deactivate_callback)

        selected_companies = {1, 2}
        tester.set_session_state(selected_companies=selected_companies)

        with (
            patch(
                "src.services.company_service.CompanyService.bulk_update_status",
                return_value=2,
            ) as mock_bulk_update,
            patch("streamlit.rerun"),
        ):
            tester.run_component()

            # Verify service was called with correct parameters
            mock_bulk_update.assert_called_once_with([1, 2], active=False)

            # Verify success feedback
            state = tester.get_session_state()
            assert (
                state.get("bulk_operation_success")
                == "Successfully deactivated 2 companies"
            )

    def test_bulk_operations_service_exception(self):
        """Test bulk operations handle service exceptions."""
        tester = StreamlitComponentTester(_bulk_activate_callback)

        tester.set_session_state(selected_companies={1, 2})

        with patch(
            "src.services.company_service.CompanyService.bulk_update_status",
            side_effect=Exception("Database connection failed"),
        ):
            tester.run_component()

            # Verify error handling
            state = tester.get_session_state()
            assert (
                "Failed to activate companies: Database connection failed"
                in state.get("bulk_operation_error", "")
            )
            assert state.get("bulk_operation_success") is None


class TestBulkDeleteFunctionality:
    """Test bulk delete functionality."""

    def test_bulk_delete_callback_with_selection(self):
        """Test bulk delete callback with companies selected."""
        tester = StreamlitComponentTester(_bulk_delete_callback)

        tester.set_session_state(selected_companies={1, 2, 3})

        with patch("streamlit.rerun") as mock_rerun:
            tester.run_component()

            # Verify confirmation flag is set
            state = tester.get_session_state()
            assert state.get("show_bulk_delete_confirm") is True

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_bulk_delete_callback_no_selection(self):
        """Test bulk delete callback with no companies selected."""
        tester = StreamlitComponentTester(_bulk_delete_callback)

        # No selection
        tester.set_session_state(selected_companies=set())

        tester.run_component()

        # Verify error feedback
        state = tester.get_session_state()
        assert state.get("bulk_operation_error") == "No companies selected for deletion"

    def test_execute_bulk_delete_success(self):
        """Test successful bulk delete execution."""
        tester = StreamlitComponentTester(_execute_bulk_delete)

        selected_companies = {1, 2, 3}
        tester.set_session_state(selected_companies=selected_companies)

        with (
            patch(
                "src.services.company_service.CompanyService.bulk_delete_companies",
                return_value=3,
            ) as mock_bulk_delete,
            patch(
                "uuid.uuid4",
                return_value=uuid.UUID("12345678-1234-5678-9abc-123456789abc"),
            ),
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component()

            # Verify service was called
            mock_bulk_delete.assert_called_once_with([1, 2, 3])

            # Verify success feedback
            state = tester.get_session_state()
            assert (
                state.get("bulk_operation_success")
                == "Successfully deleted 3 companies"
            )

            # Verify selections are cleared
            assert len(state.get("selected_companies", set())) == 0

            # Verify idempotency token is recorded
            executed_tokens = state.get("executed_delete_tokens", set())
            assert (
                str(uuid.UUID("12345678-1234-5678-9abc-123456789abc"))
                in executed_tokens
            )

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_execute_bulk_delete_no_selection(self):
        """Test bulk delete with no companies selected."""
        tester = StreamlitComponentTester(_execute_bulk_delete)

        # No selection
        tester.set_session_state(selected_companies=set())

        tester.run_component()

        # Verify error feedback
        state = tester.get_session_state()
        assert state.get("bulk_operation_error") == "No companies selected for deletion"

    def test_execute_bulk_delete_idempotency_check(self):
        """Test bulk delete idempotency prevents duplicate operations."""
        tester = StreamlitComponentTester(_execute_bulk_delete)

        operation_token = str(uuid.uuid4())
        tester.set_session_state(
            selected_companies={1, 2}, executed_delete_tokens={operation_token}
        )

        with (
            patch("uuid.uuid4", return_value=uuid.UUID(operation_token)),
            patch(
                "src.services.company_service.CompanyService.bulk_delete_companies"
            ) as mock_bulk_delete,
        ):
            tester.run_component()

            # Verify service is not called due to idempotency
            mock_bulk_delete.assert_not_called()

    def test_execute_bulk_delete_service_exception(self):
        """Test bulk delete handles service exceptions."""
        tester = StreamlitComponentTester(_execute_bulk_delete)

        tester.set_session_state(selected_companies={1, 2})

        with patch(
            "src.services.company_service.CompanyService.bulk_delete_companies",
            side_effect=Exception("Database error"),
        ):
            tester.run_component()

            # Verify error handling
            state = tester.get_session_state()
            assert "Failed to delete companies: Database error" in state.get(
                "bulk_operation_error", ""
            )
            assert state.get("bulk_operation_success") is None


class TestBulkDeleteDialog:
    """Test bulk delete confirmation dialog."""

    def test_show_bulk_delete_dialog_structure(self):
        """Test bulk delete dialog displays correctly."""
        tester = StreamlitComponentTester(_show_bulk_delete_dialog)

        tester.set_session_state(selected_companies={1, 2, 3})

        with (
            patch("streamlit.warning") as mock_warning,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            mock_button.return_value = False

            tester.run_component()

            # Verify warning message
            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "3 companies" in warning_message

            # Verify columns and buttons are created
            mock_columns.assert_called_once_with([1, 1])
            assert mock_button.call_count == 2  # Confirm and Cancel buttons

    def test_show_bulk_delete_dialog_confirm_action(self):
        """Test bulk delete dialog confirm button triggers deletion."""
        tester = StreamlitComponentTester(_show_bulk_delete_dialog)

        tester.set_session_state(selected_companies={1, 2})

        with (
            patch("streamlit.warning"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
            patch("src.ui.pages.companies._execute_bulk_delete") as mock_execute,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            # Mock confirm button clicked
            mock_button.side_effect = [True, False]

            tester.run_component()

            # Verify bulk delete execution is called
            mock_execute.assert_called_once()


class TestCompanyPageMainFunction:
    """Test main companies page function."""

    def test_show_companies_page_basic_structure(self, sample_companies):
        """Test basic page structure renders correctly."""
        tester = StreamlitComponentTester(show_companies_page)

        with (
            patch("src.ui.utils.url_state.validate_url_params", return_value={}),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=sample_companies,
            ),
            patch("src.ui.ui_rendering.render_company_card_with_selection"),
            patch("src.ui.pages.companies._company_scraping_progress_fragment"),
            patch("streamlit.title") as mock_title,
            patch("streamlit.markdown") as mock_markdown,
        ):
            tester.run_component()

            # Verify page title and content
            mock_title.assert_called_with("Company Management")

            # Verify key sections are rendered
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            assert any("Add New Company" in call for call in markdown_calls)
            assert any("Companies" in call for call in markdown_calls)

    def test_show_companies_page_with_url_validation_errors(self):
        """Test page handles URL validation errors."""
        tester = StreamlitComponentTester(show_companies_page)

        with (
            patch(
                "src.ui.utils.url_state.validate_url_params",
                return_value={"invalid_param": "Invalid company selection"},
            ),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=[],
            ),
            patch("streamlit.warning") as mock_warning,
        ):
            tester.run_component()

            # Verify warning is displayed
            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "URL parameters are invalid" in warning_message

    def test_show_companies_page_no_companies(self):
        """Test page displays message when no companies exist."""
        tester = StreamlitComponentTester(show_companies_page)

        with (
            patch("src.ui.utils.url_state.validate_url_params", return_value={}),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=[],
            ),
            patch("streamlit.info") as mock_info,
        ):
            tester.run_component()

            # Verify info message is displayed
            mock_info.assert_called_once()
            info_message = mock_info.call_args[0][0]
            assert "No companies found" in info_message

    def test_show_companies_page_with_companies_and_selections(self, sample_companies):
        """Test page displays companies with selection functionality."""
        tester = StreamlitComponentTester(show_companies_page)

        # Set up some companies as selected
        tester.set_session_state(selected_companies={1, 2})

        with (
            patch("src.ui.utils.url_state.validate_url_params", return_value={}),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=sample_companies,
            ),
            patch(
                "src.ui.ui_rendering.render_company_card_with_selection"
            ) as mock_render_card,
            patch("src.ui.pages.companies._company_scraping_progress_fragment"),
            patch("streamlit.button") as mock_button,
            patch("streamlit.markdown") as mock_markdown,
        ):
            mock_button.return_value = False

            tester.run_component()

            # Verify companies are rendered
            assert mock_render_card.call_count == 3  # One for each company

            # Verify selection count is displayed
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            assert any("2 of 3 companies selected" in call for call in markdown_calls)

    def test_show_companies_page_service_exception(self):
        """Test page handles service exceptions gracefully."""
        tester = StreamlitComponentTester(show_companies_page)

        with (
            patch("src.ui.utils.url_state.validate_url_params", return_value={}),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                side_effect=Exception("Database connection failed"),
            ),
            patch("streamlit.error") as mock_error,
        ):
            tester.run_component()

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to load companies" in error_message


class TestCompanyStatistics:
    """Test company statistics functionality."""

    def test_render_company_statistics(self, sample_companies):
        """Test company statistics rendering."""
        tester = StreamlitComponentTester(_render_company_statistics)

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.metric") as mock_metric,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            tester.run_component(sample_companies)

            # Verify metrics are displayed
            assert mock_metric.call_count == 3

            # Verify correct statistics
            metric_calls = mock_metric.call_args_list
            total_call = metric_calls[0][0]
            active_call = metric_calls[1][0]
            inactive_call = metric_calls[2][0]

            assert total_call == ("Total Companies", 3)
            assert active_call == (
                "Active Companies",
                2,
            )  # Tech Corp and Scale Inc are active
            assert inactive_call == ("Inactive Companies", 1)  # AI Startup is inactive

    def test_render_company_statistics_empty_list(self):
        """Test statistics with empty company list."""
        tester = StreamlitComponentTester(_render_company_statistics)

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.metric") as mock_metric,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            tester.run_component([])

            # Verify zero statistics
            metric_calls = mock_metric.call_args_list
            total_call = metric_calls[0][0]
            active_call = metric_calls[1][0]
            inactive_call = metric_calls[2][0]

            assert total_call == ("Total Companies", 0)
            assert active_call == ("Active Companies", 0)
            assert inactive_call == ("Inactive Companies", 0)


class TestScrapingProgressFragment:
    """Test scraping progress fragment functionality."""

    def test_company_scraping_progress_fragment_not_active(self):
        """Test fragment returns early when scraping is not active."""
        tester = StreamlitComponentTester(_company_scraping_progress_fragment)

        with patch(
            "src.ui.utils.background_helpers.is_scraping_active", return_value=False
        ):
            result = tester.run_component()

            # Fragment should return early without rendering anything
            assert result is None

    def test_company_scraping_progress_fragment_no_progress_data(self):
        """Test fragment displays initialization message when no progress data."""
        tester = StreamlitComponentTester(_company_scraping_progress_fragment)

        with (
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=True
            ),
            patch(
                "src.ui.utils.background_helpers.get_company_progress", return_value={}
            ),
            patch("streamlit.info") as mock_info,
        ):
            tester.run_component()

            # Verify initialization message
            mock_info.assert_called_once()
            info_message = mock_info.call_args[0][0]
            assert "Initializing scraping" in info_message

    def test_company_scraping_progress_fragment_with_progress(
        self, company_progress_data
    ):
        """Test fragment displays progress for multiple companies."""
        tester = StreamlitComponentTester(_company_scraping_progress_fragment)

        with (
            patch(
                "src.ui.utils.background_helpers.is_scraping_active", return_value=True
            ),
            patch(
                "src.ui.utils.background_helpers.get_company_progress",
                return_value=company_progress_data,
            ),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.metric") as mock_metric,
            patch("streamlit.error") as mock_error,
            patch("streamlit.progress") as mock_progress,
        ):
            # Mock columns for progress display
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            tester.run_component()

            # Verify metrics are displayed for each company
            assert mock_metric.call_count == 3

            # Verify error is displayed for failed company
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Connection timeout" in error_message

            # Verify overall progress bar
            mock_progress.assert_called_once()
            progress_args = mock_progress.call_args[0]
            assert progress_args[0] == 1 / 3  # 1 out of 3 companies completed

    def test_company_scraping_progress_fragment_auto_refresh(self):
        """Test fragment is configured for auto-refresh."""
        # Verify the fragment has the expected attributes for auto-refresh
        assert hasattr(_company_scraping_progress_fragment, "__wrapped__")
        # Fragment should have run_every parameter set to "2s"


class TestFeedbackAndSessionState:
    """Test feedback display and session state management."""

    def test_init_and_display_feedback_initializes_state(self):
        """Test feedback initialization sets up session state properly."""
        tester = StreamlitComponentTester(_init_and_display_feedback)

        with (
            patch(
                "src.ui.utils.database_helpers.init_session_state_keys"
            ) as mock_init_keys,
            patch(
                "src.ui.utils.database_helpers.display_feedback_messages"
            ) as mock_display,
        ):
            tester.run_component()

            # Verify session state keys are initialized
            mock_init_keys.assert_called_once()
            init_args = mock_init_keys.call_args[0]
            expected_keys = [
                "add_company_error",
                "add_company_success",
                "toggle_error",
                "toggle_success",
                "bulk_operation_error",
                "bulk_operation_success",
                "selected_companies",
            ]
            assert all(key in init_args[0] for key in expected_keys)

            # Verify feedback messages are displayed
            mock_display.assert_called_once()

    def test_init_and_display_feedback_ensures_selected_companies_set(self):
        """Test feedback initialization ensures selected_companies is a set."""
        tester = StreamlitComponentTester(_init_and_display_feedback)

        # Start with selected_companies as None
        tester.set_session_state(selected_companies=None)

        with (
            patch("src.ui.utils.database_helpers.init_session_state_keys"),
            patch("src.ui.utils.database_helpers.display_feedback_messages"),
        ):
            tester.run_component()

            # Verify selected_companies is initialized as empty set
            state = tester.get_session_state()
            assert isinstance(state.get("selected_companies"), set)
            assert len(state["selected_companies"]) == 0


class TestSessionStateManagement:
    """Test session state management across page interactions."""

    def test_company_selection_state_persistence(self):
        """Test company selections persist across page interactions."""
        tester = StreamlitComponentTester(show_companies_page)

        # Initialize with some companies selected
        initial_selection = {1, 3}
        tester.set_session_state(selected_companies=initial_selection)

        with (
            patch("src.ui.utils.url_state.validate_url_params", return_value={}),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=[],
            ),
            patch("src.ui.pages.companies._company_scraping_progress_fragment"),
        ):
            tester.run_component()

            # Verify selections are maintained
            state = tester.get_session_state()
            assert state.get("selected_companies") == initial_selection

    def test_feedback_message_persistence(self):
        """Test feedback messages persist in session state."""
        tester = StreamlitComponentTester(_add_company_callback)

        tester.set_session_state(
            company_name="Test Company", company_url="https://test.com/careers"
        )

        mock_company = Company(
            id=1,
            name="Test Company",
            url="https://test.com/careers",
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch(
                "src.services.company_service.CompanyService.add_company",
                return_value=mock_company,
            ),
            patch("streamlit.rerun"),
        ):
            tester.run_component()

            # Verify success message persists
            state = tester.get_session_state()
            assert "Successfully added company: Test Company" in state.get(
                "add_company_success", ""
            )

    def test_bulk_operation_token_persistence(self):
        """Test bulk operation tokens are tracked to prevent duplicates."""
        tester = StreamlitComponentTester(_execute_bulk_delete)

        operation_token = str(uuid.uuid4())
        tester.set_session_state(
            selected_companies={1, 2}, executed_delete_tokens={operation_token}
        )

        with patch("uuid.uuid4", return_value=uuid.UUID(operation_token)):
            tester.run_component()

            # Verify token is still in the set
            state = tester.get_session_state()
            assert operation_token in state.get("executed_delete_tokens", set())


# Integration tests combining multiple components
class TestCompaniesPageIntegration:
    """Integration tests for companies page components working together."""

    def test_full_company_workflow_add_to_delete(self, sample_companies):
        """Test complete workflow from adding to deleting a company."""
        # Test adding company
        add_tester = StreamlitComponentTester(_add_company_callback)
        add_tester.set_session_state(
            company_name="Integration Test Corp",
            company_url="https://integrationtest.com/careers",
        )

        mock_company = Company(
            id=4,
            name="Integration Test Corp",
            url="https://integrationtest.com/careers",
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch(
                "src.services.company_service.CompanyService.add_company",
                return_value=mock_company,
            ),
            patch("streamlit.rerun"),
        ):
            add_tester.run_component()
            add_state = add_tester.get_session_state()
            assert "Successfully added company" in add_state.get(
                "add_company_success", ""
            )

        # Test deleting the same company
        delete_tester = StreamlitComponentTester(_delete_company_callback)
        delete_tester.set_session_state(delete_confirm_4=True)

        with (
            patch(
                "src.services.company_service.CompanyService.delete_company",
                return_value=True,
            ),
            patch("streamlit.rerun"),
        ):
            delete_tester.run_component(4)
            delete_state = delete_tester.get_session_state()
            assert delete_state.get("delete_success") == "Company deleted successfully"

    def test_bulk_operations_workflow(self, sample_companies):
        """Test complete bulk operations workflow."""
        # Test selecting all companies
        select_tester = StreamlitComponentTester(_select_all_callback)

        with (
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=sample_companies,
            ),
            patch("src.ui.utils.url_state.update_url_from_company_selection"),
            patch("streamlit.rerun"),
        ):
            select_tester.run_component()
            select_state = select_tester.get_session_state()
            assert select_state.get("selected_companies") == {1, 2, 3}

        # Test bulk activation using the selection
        activate_tester = StreamlitComponentTester(_bulk_activate_callback)
        activate_tester.set_session_state(selected_companies={1, 2, 3})

        with (
            patch(
                "src.services.company_service.CompanyService.bulk_update_status",
                return_value=3,
            ),
            patch("streamlit.rerun"),
        ):
            activate_tester.run_component()
            activate_state = activate_tester.get_session_state()
            assert "Successfully activated 3 companies" in activate_state.get(
                "bulk_operation_success", ""
            )

            # Verify selections are cleared after bulk operation
            assert len(activate_state.get("selected_companies", set())) == 0

    def test_form_to_display_integration(self, sample_companies):
        """Test integration between form submission and page display."""
        tester = StreamlitComponentTester(show_companies_page)

        # Set up form data as if user just submitted
        tester.set_session_state(
            add_company_success="Successfully added company: New Company",
            selected_companies={1, 2},
        )

        with (
            patch("src.ui.utils.url_state.validate_url_params", return_value={}),
            patch("src.ui.utils.url_state.sync_company_selection_from_url"),
            patch(
                "src.services.company_service.CompanyService.get_all_companies",
                return_value=sample_companies,
            ),
            patch(
                "src.ui.ui_rendering.render_company_card_with_selection"
            ) as mock_render_card,
            patch("src.ui.pages.companies._company_scraping_progress_fragment"),
            patch("streamlit.success"),
        ):  # Feedback should be displayed
            tester.run_component()

            # Verify companies are rendered
            assert mock_render_card.call_count == 3

            # Verify feedback is displayed (mock_success would be called by display_feedback_messages)
            state = tester.get_session_state()
            assert "Successfully added company" in state.get("add_company_success", "")

    def test_error_recovery_workflow(self):
        """Test error handling and recovery across multiple operations."""
        # Test failed company addition
        add_tester = StreamlitComponentTester(_add_company_callback)
        add_tester.set_session_state(company_name="", company_url="")

        add_tester.run_component()
        add_state = add_tester.get_session_state()
        assert add_state.get("add_company_error") == "Company name is required"

        # Test recovery with valid input
        add_tester.set_session_state(
            company_name="Recovery Corp", company_url="https://recovery.com/careers"
        )

        mock_company = Company(
            id=5,
            name="Recovery Corp",
            url="https://recovery.com/careers",
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch(
                "src.services.company_service.CompanyService.add_company",
                return_value=mock_company,
            ),
            patch("streamlit.rerun"),
        ):
            add_tester.run_component()
            recovery_state = add_tester.get_session_state()

            # Verify error is cleared and success is set
            assert recovery_state.get("add_company_error") is None
            assert "Successfully added company: Recovery Corp" in recovery_state.get(
                "add_company_success", ""
            )
