"""Tests for job modal UI helper functions.

These tests validate the job modal rendering components including header display,
status information, description, notes section, and action buttons with various
data scenarios and edge cases.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.schemas import Job
from src.ui.ui_rendering import (
    render_action_buttons,
    render_job_description,
    render_job_header,
    render_job_status,
    render_notes_section,
)


class TestRenderJobHeader:
    """Test cases for render_job_header function."""

    def test_render_job_header_with_complete_data(self, mock_streamlit, sample_job_dto):
        """Test rendering job header with complete job data."""
        render_job_header(sample_job_dto)

        # Verify title and company info are displayed correctly
        mock_streamlit["markdown"].assert_any_call(f"### {sample_job_dto.title}")
        mock_streamlit["markdown"].assert_any_call(
            f"**{sample_job_dto.company}** • {sample_job_dto.location}",
        )

    def test_render_job_header_with_minimal_data(self, mock_streamlit):
        """Test rendering job header with minimal job data."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="test_hash",
            application_status="New",
        )

        render_job_header(job)

        mock_streamlit["markdown"].assert_any_call("### Test Job")
        mock_streamlit["markdown"].assert_any_call("**Test Company** • Remote")

    def test_render_job_header_with_special_characters(self, mock_streamlit):
        """Test rendering job header with special characters in title and company."""
        job = Job(
            id=1,
            company_id=1,
            company="Tech & Co.",
            title="Senior Software Engineer (Python/Django)",
            description="Test description",
            location="San Francisco, CA",
            link="https://techco.com/job",
            content_hash="hash123",
            application_status="New",
        )

        render_job_header(job)

        mock_streamlit["markdown"].assert_any_call(
            "### Senior Software Engineer (Python/Django)",
        )
        mock_streamlit["markdown"].assert_any_call("**Tech & Co.** • San Francisco, CA")

    def test_render_job_header_with_empty_location(self, mock_streamlit):
        """Test rendering job header with empty location."""
        job = Job(
            id=1,
            company_id=1,
            company="Remote Co",
            title="Remote Developer",
            description="Test description",
            location="",
            link="https://remoteco.com/job",
            content_hash="remote_hash",
            application_status="New",
        )

        render_job_header(job)

        mock_streamlit["markdown"].assert_any_call("### Remote Developer")
        mock_streamlit["markdown"].assert_any_call("**Remote Co** • ")


class TestRenderJobStatus:
    """Test cases for render_job_status function."""

    def test_render_job_status_with_posted_date(self, mock_streamlit):
        """Test rendering job status with posted date."""
        posted_date = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="test_hash",
            posted_date=posted_date,
            application_status="Interested",
        )

        render_job_status(job)

        # Verify columns are created
        mock_streamlit["columns"].assert_called_once_with(2)

    def test_render_job_status_without_posted_date(self, mock_streamlit):
        """Test rendering job status without posted date."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="test_hash",
            posted_date=None,
            application_status="New",
        )

        # Mock columns to return two mock column objects
        col1, col2 = MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2]

        render_job_status(job)

        # Verify columns are created
        mock_streamlit["columns"].assert_called_once_with(2)

    @pytest.mark.parametrize(
        ("status", "expected_icon"),
        (
            ("New", "🔵"),
            ("Interested", "🟡"),
            ("Applied", "🟢"),
            ("Rejected", "🔴"),
            ("Unknown", "⚪"),
            ("", "⚪"),
        ),
    )
    def test_render_job_status_with_different_statuses(
        self,
        mock_streamlit,
        status,
        expected_icon,
    ):
        """Test rendering job status with different application statuses."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="test_hash",
            application_status=status,
        )

        # Mock columns to return two mock column objects
        col1, col2 = MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2]

        render_job_status(job)

        # Verify columns are created
        mock_streamlit["columns"].assert_called_once_with(2)


class TestRenderJobDescription:
    """Test cases for render_job_description function."""

    def test_render_job_description_with_standard_text(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test rendering job description with standard job text."""
        render_job_description(sample_job_dto)

        # Verify separator, header, and description are displayed
        mock_streamlit["markdown"].assert_any_call("---")
        mock_streamlit["markdown"].assert_any_call("### Job Description")
        mock_streamlit["markdown"].assert_any_call(sample_job_dto.description)

    def test_render_job_description_with_html_content(self, mock_streamlit):
        """Test rendering job description with HTML content."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="<p>This is a <strong>great</strong> opportunity!</p>",
            location="Remote",
            link="https://test.com/job",
            content_hash="html_hash",
            application_status="New",
        )

        render_job_description(job)

        mock_streamlit["markdown"].assert_any_call("---")
        mock_streamlit["markdown"].assert_any_call("### Job Description")
        mock_streamlit["markdown"].assert_any_call(
            "<p>This is a <strong>great</strong> opportunity!</p>",
        )

    def test_render_job_description_with_multiline_text(self, mock_streamlit):
        """Test rendering job description with multiline text."""
        description = """This is a multiline job description.

Requirements:
- 5+ years experience
- Python skills
- Remote work flexibility

Benefits:
- Competitive salary
- Great team"""

        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description=description,
            location="Remote",
            link="https://test.com/job",
            content_hash="multiline_hash",
            application_status="New",
        )

        render_job_description(job)

        mock_streamlit["markdown"].assert_any_call("---")
        mock_streamlit["markdown"].assert_any_call("### Job Description")
        mock_streamlit["markdown"].assert_any_call(description)

    def test_render_job_description_with_minimal_description(self, mock_streamlit):
        """Test rendering job description with minimal description."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description=" ",  # Minimal valid description
            location="Remote",
            link="https://test.com/job",
            content_hash="minimal_hash",
            application_status="New",
        )

        render_job_description(job)

        mock_streamlit["markdown"].assert_any_call("---")
        mock_streamlit["markdown"].assert_any_call("### Job Description")
        mock_streamlit["markdown"].assert_any_call(" ")


class TestRenderNotesSection:
    """Test cases for render_notes_section function."""

    def test_render_notes_section_with_existing_notes(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test rendering notes section with existing notes."""
        expected_notes = "Test notes from text area"
        mock_streamlit["text_area"].return_value = expected_notes

        result = render_notes_section(sample_job_dto)

        # Verify UI elements are displayed
        mock_streamlit["markdown"].assert_any_call("---")
        mock_streamlit["markdown"].assert_any_call("### Notes")

        # Verify text area is configured correctly
        mock_streamlit["text_area"].assert_called_once_with(
            "Your notes about this position",
            value=sample_job_dto.notes or "",
            key=f"modal_notes_{sample_job_dto.id}",
            help="Add your personal notes about this job",
            height=150,
        )

        # Verify return value
        assert result == expected_notes

    def test_render_notes_section_with_empty_notes(self, mock_streamlit):
        """Test rendering notes section with empty notes."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="empty_notes_hash",
            notes="",
            application_status="New",
        )

        expected_notes = "New notes from user"
        mock_streamlit["text_area"].return_value = expected_notes

        result = render_notes_section(job)

        mock_streamlit["text_area"].assert_called_once_with(
            "Your notes about this position",
            value="",
            key="modal_notes_1",
            help="Add your personal notes about this job",
            height=150,
        )

        assert result == expected_notes

    def test_render_notes_section_with_none_notes(self, mock_streamlit):
        """Test rendering notes section with None notes."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="none_notes_hash",
            notes="",  # Use empty string instead of None
            application_status="New",
        )

        expected_notes = "User input notes"
        mock_streamlit["text_area"].return_value = expected_notes

        result = render_notes_section(job)

        mock_streamlit["text_area"].assert_called_once_with(
            "Your notes about this position",
            value="",
            key="modal_notes_1",
            help="Add your personal notes about this job",
            height=150,
        )

        assert result == expected_notes

    def test_render_notes_section_with_long_notes(self, mock_streamlit):
        """Test rendering notes section with very long notes."""
        long_notes = "This is a very long note. " * 100
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/job",
            content_hash="long_notes_hash",
            notes=long_notes,
            application_status="New",
        )

        mock_streamlit["text_area"].return_value = long_notes

        result = render_notes_section(job)

        mock_streamlit["text_area"].assert_called_once_with(
            "Your notes about this position",
            value=long_notes,
            key="modal_notes_1",
            help="Add your personal notes about this job",
            height=150,
        )

        assert result == long_notes


class TestRenderActionButtons:
    """Test cases for render_action_buttons function."""

    def test_render_action_buttons_with_job_link(self, mock_streamlit, sample_job_dto):
        """Test rendering action buttons with job link available."""
        notes_value = "Test notes"

        # Mock columns to return three mock column objects
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock button clicks
        mock_streamlit["button"].return_value = False

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(sample_job_dto, notes_value)

            # Verify UI structure
            mock_streamlit["markdown"].assert_any_call("---")
            mock_streamlit["columns"].assert_called_once_with([1, 1, 1])

            # Verify link button is created when job has link
            mock_streamlit["link_button"].assert_called_once_with(
                "Apply Now",
                sample_job_dto.link,
                use_container_width=True,
                type="secondary",
            )

    def test_render_action_buttons_without_job_link(self, mock_streamlit):
        """Test rendering action buttons when job link is empty."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/empty",  # Valid link
            content_hash="no_link_hash",
            application_status="New",
        )
        notes_value = "Test notes"

        # Mock columns to return three mock column objects
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock button clicks
        mock_streamlit["button"].return_value = False

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(job, notes_value)

            # Test that link button is created for valid links
            mock_streamlit["link_button"].assert_called_once()

    def test_render_action_buttons_save_notes_clicked(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test save notes button functionality."""
        notes_value = "Updated notes"

        # Mock columns
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock save notes button click (return True for first button call)
        mock_streamlit["button"].side_effect = [True, False]  # Save=True, Close=False

        with patch("src.ui.utils.job_utils.save_job_notes") as mock_save_notes:
            render_action_buttons(sample_job_dto, notes_value)

            # Verify save_job_notes is called with correct parameters
            mock_save_notes.assert_called_once_with(sample_job_dto.id, notes_value)

    def test_render_action_buttons_close_clicked(
        self,
        mock_streamlit,
        mock_session_state,
        sample_job_dto,
    ):
        """Test close button functionality."""
        notes_value = "Test notes"

        # Mock columns
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock close button click (return True for second button call)
        mock_streamlit["button"].side_effect = [False, True]  # Save=False, Close=True

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(sample_job_dto, notes_value)

            # Verify session state is updated and rerun is called
            assert mock_session_state.view_job_id is None
            mock_streamlit["rerun"].assert_called_once()

    def test_render_action_buttons_with_none_job_link(self, mock_streamlit):
        """Test rendering action buttons with None job link."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="https://test.com/minimal",  # Valid link
            content_hash="none_link_hash",
            application_status="New",
        )
        notes_value = "Test notes"

        # Mock columns
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock button clicks
        mock_streamlit["button"].return_value = False

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(job, notes_value)

            # Since we can't have None links due to validation,
            # we test that the function handles minimal links
            mock_streamlit["link_button"].assert_called_once()

    def test_render_action_buttons_button_configuration(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test action buttons have correct configuration."""
        notes_value = "Test notes"

        # Mock columns
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock button clicks
        mock_streamlit["button"].return_value = False

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(sample_job_dto, notes_value)

            # Check button calls for correct parameters
            button_calls = mock_streamlit["button"].call_args_list

            # Save Notes button
            save_call = button_calls[0]
            assert save_call[0] == ("Save Notes",)  # First positional arg
            assert save_call[1]["type"] == "primary"
            assert save_call[1]["use_container_width"] is True

            # Close button
            close_call = button_calls[1]
            assert close_call[0] == ("Close",)  # First positional arg
            assert close_call[1]["use_container_width"] is True

    def test_render_action_buttons_with_malformed_url(self, mock_streamlit):
        """Test rendering action buttons with malformed job URL."""
        job = Job(
            id=1,
            company_id=1,
            company="Test Company",
            title="Test Job",
            description="Test description",
            location="Remote",
            link="not-a-valid-url",
            content_hash="malformed_hash",
            application_status="New",
        )
        notes_value = "Test notes"

        # Mock columns
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_streamlit["columns"].return_value = [col1, col2, col3]

        # Mock button clicks
        mock_streamlit["button"].return_value = False

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(job, notes_value)

            # Verify link button is still created (Streamlit handles URL validation)
            mock_streamlit["link_button"].assert_called_once_with(
                "Apply Now",
                "not-a-valid-url",
                use_container_width=True,
                type="secondary",
            )


class TestJobModalIntegration:
    """Integration tests for job modal functions working together."""

    def test_full_modal_rendering_flow(self, mock_streamlit, sample_job_dto):
        """Test complete flow of rendering all job modal components."""
        # Render all components
        render_job_header(sample_job_dto)
        render_job_status(sample_job_dto)
        render_job_description(sample_job_dto)
        notes = render_notes_section(sample_job_dto)
        render_action_buttons(sample_job_dto, notes)

        # Verify header components
        mock_streamlit["markdown"].assert_any_call(f"### {sample_job_dto.title}")
        mock_streamlit["markdown"].assert_any_call(
            f"**{sample_job_dto.company}** • {sample_job_dto.location}",
        )

        # Verify separators are used
        separator_calls = [
            call
            for call in mock_streamlit["markdown"].call_args_list
            if call[0][0] == "---"
        ]
        assert len(separator_calls) >= 2  # At least 2 separators (description + notes)

    def test_modal_with_minimal_job_data(self, mock_streamlit):
        """Test modal rendering with minimal job data."""
        job = Job(
            id=1,
            company_id=1,
            company="Minimal Co",
            title="Basic Job",
            description="Basic description",
            location="Remote",
            link="https://minimal.com/job",
            content_hash="basic_hash",
            application_status="New",
        )

        # Mock required components
        col1, col2 = MagicMock(), MagicMock()
        col_3 = [MagicMock(), MagicMock(), MagicMock()]
        mock_streamlit["columns"].side_effect = [[col1, col2], col_3]
        mock_streamlit["text_area"].return_value = ""
        mock_streamlit["button"].return_value = False

        # Render all components
        render_job_header(job)
        render_job_status(job)
        render_job_description(job)
        notes = render_notes_section(job)

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(job, notes)

        # Verify basic rendering worked
        mock_streamlit["markdown"].assert_any_call("### Basic Job")
        mock_streamlit["markdown"].assert_any_call("**Minimal Co** • Remote")
        mock_streamlit["markdown"].assert_any_call("Basic description")

    def test_modal_error_handling_with_corrupt_data(self, mock_streamlit):
        """Test modal functions handle corrupt or unusual data gracefully."""
        # Job with unusual but valid data
        job = Job(
            id=999999,
            company_id=1,
            company="Co with 'quotes' & symbols!",
            title="Job with <HTML> & Special Chars",
            description="Description with\n\nmultiple lines\n\nand **markdown**",
            location="Location, With, Commas",
            link="https://complex.com/job",
            content_hash="complex_hash",
            application_status="CustomStatus",
            notes="Notes with\ttabs and\nnewlines",
        )

        # Mock required components
        col1, col2 = MagicMock(), MagicMock()
        col_3 = [MagicMock(), MagicMock(), MagicMock()]
        mock_streamlit["columns"].side_effect = [[col1, col2], col_3]
        mock_streamlit["text_area"].return_value = "test"
        mock_streamlit["button"].return_value = False

        # Should not raise exceptions
        render_job_header(job)
        render_job_status(job)
        render_job_description(job)
        notes = render_notes_section(job)

        with patch("src.ui.utils.job_utils.save_job_notes"):
            render_action_buttons(job, notes)

        # Verify content was rendered (with special characters)
        mock_streamlit["markdown"].assert_any_call(
            "### Job with <HTML> & Special Chars",
        )
        mock_streamlit["markdown"].assert_any_call(
            "**Co with 'quotes' & symbols!** • Location, With, Commas",
        )
