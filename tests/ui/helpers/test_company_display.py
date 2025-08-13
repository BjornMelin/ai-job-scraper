"""Tests for company display UI helpers.

Tests the rendering functions for company information, statistics, and cards.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.schemas import Company
from src.ui.helpers.company_display import (
    render_company_card,
    render_company_info,
    render_company_statistics,
    render_company_toggle,
)


class TestRenderCompanyInfo:
    """Test company information rendering."""

    def test_render_company_info_displays_name_and_url(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test that company name and URL are displayed correctly."""
        # Act
        render_company_info(sample_company_dto)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Check name is displayed with bold formatting
        assert f"**{sample_company_dto.name}**" in markdown_texts

        # Check URL is displayed as a clickable link
        expected_link = f"🔗 [{sample_company_dto.url}]({sample_company_dto.url})"
        assert expected_link in markdown_texts

    def test_render_company_info_handles_special_characters_in_name(
        self,
        mock_streamlit,
    ):
        """Test rendering company with special characters in name."""
        # Arrange
        company = Company(
            id=1,
            name="Tech & Data Co. (AI)",
            url="https://tech-data.com",
            active=True,
        )

        # Act
        render_company_info(company)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Check name with special characters is displayed
        assert "**Tech & Data Co. (AI)**" in markdown_texts

    def test_render_company_info_handles_long_urls(self, mock_streamlit):
        """Test rendering company with very long URL."""
        # Arrange
        long_url = "https://very-long-company-name-with-many-subdomains.example.com/careers/jobs/listings"
        company = Company(
            id=1,
            name="Long URL Company",
            url=long_url,
            active=True,
        )

        # Act
        render_company_info(company)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Check full URL is displayed
        expected_link = f"🔗 [{long_url}]({long_url})"
        assert expected_link in markdown_texts


class TestRenderCompanyStatistics:
    """Test company statistics rendering."""

    def test_render_company_statistics_with_scraping_history(self, mock_streamlit):
        """Test rendering statistics for company with scraping history."""
        # Arrange
        company = Company(
            id=1,
            name="Test Company",
            url="https://test.com",
            active=True,
            last_scraped=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            scrape_count=25,
            success_rate=0.84,
        )

        # Act
        render_company_statistics(company)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Check last scraped date
        assert "📅 Last scraped: 2024-01-15 10:30" in markdown_texts

        # Check scraping statistics
        assert "📊 Scrapes: 25 | Success: 84.0%" in markdown_texts

    def test_render_company_statistics_never_scraped(self, mock_streamlit):
        """Test rendering statistics for company that's never been scraped."""
        # Arrange
        company = Company(
            id=1,
            name="New Company",
            url="https://new.com",
            active=True,
            last_scraped=None,
            scrape_count=0,
            success_rate=1.0,
        )

        # Act
        render_company_statistics(company)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Check never scraped message
        assert "📅 Never scraped" in markdown_texts
        assert "📊 No scraping history" in markdown_texts

    def test_render_company_statistics_zero_scrape_count(self, mock_streamlit):
        """Test rendering statistics for company with zero scrape count."""
        # Arrange
        company = Company(
            id=1,
            name="Zero Scrapes",
            url="https://zero.com",
            active=True,
            last_scraped=datetime.now(UTC),
            scrape_count=0,
            success_rate=1.0,
        )

        # Act
        render_company_statistics(company)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Should show no scraping history despite having last_scraped
        assert "📊 No scraping history" in markdown_texts

    @pytest.mark.parametrize(
        ("success_rate", "expected_percentage"),
        (
            (0.0, "0.0%"),
            (0.5, "50.0%"),
            (0.846, "84.6%"),
            (1.0, "100.0%"),
        ),
    )
    def test_render_company_statistics_success_rate_formatting(
        self,
        mock_streamlit,
        success_rate,
        expected_percentage,
    ):
        """Test success rate is formatted correctly as percentage."""
        # Arrange
        company = Company(
            id=1,
            name="Test Company",
            url="https://test.com",
            active=True,
            last_scraped=datetime.now(UTC),
            scrape_count=10,
            success_rate=success_rate,
        )

        # Act
        render_company_statistics(company)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list
        markdown_texts = [call.args[0] for call in markdown_calls]

        # Check success rate formatting
        expected_text = f"📊 Scrapes: 10 | Success: {expected_percentage}"
        assert expected_text in markdown_texts


class TestRenderCompanyToggle:
    """Test company active toggle rendering."""

    def test_render_company_toggle_active_company(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test rendering toggle for active company."""
        # Arrange
        mock_callback = Mock()

        # Act
        render_company_toggle(sample_company_dto, mock_callback)

        # Assert
        mock_streamlit["toggle"].assert_called_once()
        call_args = mock_streamlit["toggle"].call_args

        assert call_args.args[0] == "Active"
        assert call_args.kwargs["value"] == sample_company_dto.active
        assert call_args.kwargs["key"] == f"company_active_{sample_company_dto.id}"
        assert sample_company_dto.name in call_args.kwargs["help"]
        assert call_args.kwargs["on_change"] == mock_callback
        assert call_args.kwargs["args"] == (sample_company_dto.id,)

    def test_render_company_toggle_inactive_company(self, mock_streamlit):
        """Test rendering toggle for inactive company."""
        # Arrange
        company = Company(
            id=2,
            name="Inactive Company",
            url="https://inactive.com",
            active=False,
        )
        mock_callback = Mock()

        # Act
        render_company_toggle(company, mock_callback)

        # Assert
        mock_streamlit["toggle"].assert_called_once()
        call_args = mock_streamlit["toggle"].call_args

        assert call_args.kwargs["value"] is False
        assert call_args.kwargs["key"] == "company_active_2"
        assert "Inactive Company" in call_args.kwargs["help"]

    def test_render_company_toggle_callback_configuration(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test that toggle callback is configured correctly."""
        # Arrange
        mock_callback = Mock()

        # Act
        render_company_toggle(sample_company_dto, mock_callback)

        # Assert
        call_args = mock_streamlit["toggle"].call_args

        # Verify callback function and arguments are set correctly
        assert call_args.kwargs["on_change"] == mock_callback
        assert call_args.kwargs["args"] == (sample_company_dto.id,)

    def test_render_company_toggle_help_text_includes_company_name(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test that help text includes the company name."""
        # Arrange
        mock_callback = Mock()

        # Act
        render_company_toggle(sample_company_dto, mock_callback)

        # Assert
        call_args = mock_streamlit["toggle"].call_args
        help_text = call_args.kwargs["help"]

        assert f"Toggle scraping for {sample_company_dto.name}" == help_text


class TestRenderCompanyCard:
    """Test complete company card rendering."""

    def test_render_company_card_creates_container_and_columns(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test company card creates bordered container with proper column layout."""
        # Arrange
        mock_callback = Mock()

        # Act
        render_company_card(sample_company_dto, mock_callback)

        # Assert
        # Check container with border is created
        mock_streamlit["container"].assert_called_with(border=True)

        # Check 3-column layout is created with proper ratios
        mock_streamlit["columns"].assert_called_with([3, 2, 1])

    def test_render_company_card_renders_all_components(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test that company card renders all components (info, stats, toggle)."""
        # Arrange
        mock_callback = Mock()

        with (
            patch("src.ui.helpers.company_display.render_company_info") as mock_info,
            patch(
                "src.ui.helpers.company_display.render_company_statistics",
            ) as mock_stats,
            patch(
                "src.ui.helpers.company_display.render_company_toggle",
            ) as mock_toggle,
        ):
            # Act
            render_company_card(sample_company_dto, mock_callback)

            # Assert
            # Check all render functions are called with correct parameters
            mock_info.assert_called_once_with(sample_company_dto)
            mock_stats.assert_called_once_with(sample_company_dto)
            mock_toggle.assert_called_once_with(sample_company_dto, mock_callback)

    def test_render_company_card_with_different_company_data(self, mock_streamlit):
        """Test company card with different company configurations."""
        # Arrange
        companies = [
            Company(id=1, name="Active Co", url="https://active.com", active=True),
            Company(id=2, name="Inactive Co", url="https://inactive.com", active=False),
            Company(
                id=3,
                name="Scraped Co",
                url="https://scraped.com",
                active=True,
                last_scraped=datetime.now(UTC),
                scrape_count=5,
                success_rate=0.8,
            ),
        ]
        mock_callback = Mock()

        with (
            patch("src.ui.helpers.company_display.render_company_info") as mock_info,
            patch(
                "src.ui.helpers.company_display.render_company_statistics",
            ) as mock_stats,
            patch(
                "src.ui.helpers.company_display.render_company_toggle",
            ) as mock_toggle,
        ):
            # Act - Render cards for all companies
            for company in companies:
                render_company_card(company, mock_callback)

            # Assert - Each company was rendered
            assert mock_info.call_count == 3
            assert mock_stats.call_count == 3
            assert mock_toggle.call_count == 3

            # Check each company was passed correctly
            for i, company in enumerate(companies):
                assert mock_info.call_args_list[i][0][0] == company
                assert mock_stats.call_args_list[i][0][0] == company
                assert mock_toggle.call_args_list[i][0][0] == company

    def test_render_company_card_container_context_manager(
        self,
        mock_streamlit,
        sample_company_dto,
    ):
        """Test that company card properly uses container context manager."""
        # Arrange
        mock_callback = Mock()
        mock_container = Mock()
        mock_streamlit["container"].return_value.__enter__ = Mock(
            return_value=mock_container,
        )
        mock_streamlit["container"].return_value.__exit__ = Mock(return_value=None)

        # Act
        render_company_card(sample_company_dto, mock_callback)

        # Assert
        # Check container context manager was used
        mock_streamlit["container"].assert_called_with(border=True)
        mock_streamlit["container"].return_value.__enter__.assert_called()
        mock_streamlit["container"].return_value.__exit__.assert_called()

    def test_render_company_card_column_usage(self, mock_streamlit, sample_company_dto):
        """Test that company card properly uses column context managers."""
        # Arrange
        mock_callback = Mock()

        # Use the default mock columns from fixture which should be configured
        # as context managers
        # Act
        render_company_card(sample_company_dto, mock_callback)

        # Assert
        # Check that columns was called with the correct layout
        mock_streamlit["columns"].assert_called_with([3, 2, 1])

        # Check that container was used as context manager
        mock_streamlit["container"].assert_called_with(border=True)


class TestCompanyDisplayIntegration:
    """Test integration scenarios for company display components."""

    def test_complete_company_display_workflow(
        self,
        mock_streamlit,
        sample_companies_dto,
    ):
        """Test complete workflow of displaying multiple companies."""
        # Arrange
        mock_callback = Mock()

        # Act - Render cards for all companies
        for company in sample_companies_dto:
            render_company_card(company, mock_callback)

        # Assert - Verify proper rendering sequence
        # Check containers were created for each company
        assert mock_streamlit["container"].call_count == len(sample_companies_dto)

        # Check columns were created for each company
        assert mock_streamlit["columns"].call_count == len(sample_companies_dto)

        # Check all columns calls use the same layout
        for call in mock_streamlit["columns"].call_args_list:
            assert call.args[0] == [3, 2, 1]

    def test_company_display_with_mixed_data_states(self, mock_streamlit):
        """Test display with companies in various data states."""
        # Arrange
        companies = [
            # New company - never scraped
            Company(id=1, name="New Co", url="https://new.com", active=True),
            # Scraped company with good success rate
            Company(
                id=2,
                name="Good Co",
                url="https://good.com",
                active=True,
                last_scraped=datetime.now(UTC),
                scrape_count=20,
                success_rate=0.95,
            ),
            # Scraped company with poor success rate
            Company(
                id=3,
                name="Poor Co",
                url="https://poor.com",
                active=False,
                last_scraped=datetime.now(UTC),
                scrape_count=10,
                success_rate=0.3,
            ),
            # Company with zero scrapes but has last_scraped (edge case)
            Company(
                id=4,
                name="Edge Co",
                url="https://edge.com",
                active=True,
                last_scraped=datetime.now(UTC),
                scrape_count=0,
                success_rate=1.0,
            ),
        ]

        mock_callback = Mock()

        with (
            patch("src.ui.helpers.company_display.render_company_info") as mock_info,
            patch(
                "src.ui.helpers.company_display.render_company_statistics",
            ) as mock_stats,
            patch(
                "src.ui.helpers.company_display.render_company_toggle",
            ) as mock_toggle,
        ):
            # Act
            for company in companies:
                render_company_card(company, mock_callback)

            # Assert
            # All companies were processed
            assert mock_info.call_count == 4
            assert mock_stats.call_count == 4
            assert mock_toggle.call_count == 4

            # Each company data was passed correctly
            for i, company in enumerate(companies):
                assert mock_info.call_args_list[i][0][0] == company
                assert mock_stats.call_args_list[i][0][0] == company
                assert mock_toggle.call_args_list[i][0][0] == company

    def test_company_display_error_handling(self, mock_streamlit, sample_company_dto):
        """Test that display components handle errors gracefully."""
        # Arrange
        mock_callback = Mock()

        with (
            patch("src.ui.helpers.company_display.render_company_info") as mock_info,
            patch(
                "src.ui.helpers.company_display.render_company_statistics",
            ) as mock_stats,
            patch(
                "src.ui.helpers.company_display.render_company_toggle",
            ) as mock_toggle,
        ):
            # Configure one component to raise an exception
            mock_stats.side_effect = Exception("Display error")

            # Act & Assert - Should not raise exception
            render_company_card(sample_company_dto, mock_callback)

            # Other components should still be called
            mock_info.assert_called_once()
            mock_toggle.assert_called_once()

    def test_company_display_callback_preservation(
        self,
        mock_streamlit,
        sample_companies_dto,
    ):
        """Test that callbacks are properly preserved across multiple renders."""
        # Arrange
        mock_callback = Mock()

        with patch(
            "src.ui.helpers.company_display.render_company_toggle",
        ) as mock_toggle:
            # Act - Render multiple companies
            for company in sample_companies_dto:
                render_company_card(company, mock_callback)

            # Assert - Same callback is used for all companies
            for call in mock_toggle.call_args_list:
                assert call.args[1] == mock_callback
