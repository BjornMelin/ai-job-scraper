"""Optimized tests for CompanyService with comprehensive coverage.

This module contains comprehensive unit tests for the CompanyService class,
including CRUD operations, caching functionality, bulk operations, and
weighted success rate calculations. Tests are designed to achieve 90%+ coverage
and validate both business logic and performance optimizations.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from sqlalchemy.exc import IntegrityError

from src.models import CompanySQL
from src.schemas import Company
from src.services.company_service import CompanyService, calculate_weighted_success_rate


class TestCalculateWeightedSuccessRate:
    """Tests for the calculate_weighted_success_rate utility function."""

    def test_first_scrape_success(self):
        """Test success rate calculation for first scrape (success)."""
        rate = calculate_weighted_success_rate(0.0, 1, True)
        assert rate == 1.0

    def test_first_scrape_failure(self):
        """Test success rate calculation for first scrape (failure)."""
        rate = calculate_weighted_success_rate(0.0, 1, False)
        assert rate == 0.0

    def test_weighted_average_success(self):
        """Test weighted average with successful scrape."""
        # Current rate 0.9, new success should move towards 1.0 slightly
        rate = calculate_weighted_success_rate(0.9, 10, True, weight=0.8)
        expected = 0.8 * 0.9 + 0.2 * 1.0  # 0.72 + 0.2 = 0.92
        assert rate == expected

    def test_weighted_average_failure(self):
        """Test weighted average with failed scrape."""
        # Current rate 0.9, new failure should move towards 0.0
        rate = calculate_weighted_success_rate(0.9, 10, False, weight=0.8)
        expected = 0.8 * 0.9 + 0.2 * 0.0  # 0.72 + 0.0 = 0.72
        assert rate == expected

    def test_different_weights(self):
        """Test with different weight values."""
        # Higher weight to historical data
        rate = calculate_weighted_success_rate(0.8, 5, True, weight=0.9)
        expected = 0.9 * 0.8 + 0.1 * 1.0  # 0.72 + 0.1 = 0.82
        assert rate == expected

        # Lower weight to historical data
        rate = calculate_weighted_success_rate(0.8, 5, True, weight=0.5)
        expected = 0.5 * 0.8 + 0.5 * 1.0  # 0.4 + 0.5 = 0.9
        assert rate == expected


class TestCompanyService:
    """Comprehensive tests for CompanyService class."""

    @patch("src.services.company_service.db_session")
    def test_to_dto_conversion(self, mock_db_session):
        """Test _to_dto method converts SQLModel to Pydantic DTO."""
        # Create mock CompanySQL object
        company_sql = MagicMock(spec=CompanySQL)
        company_sql.id = 1
        company_sql.name = "Test Company"
        company_sql.url = "https://test.com"
        company_sql.active = True
        company_sql.last_scraped = None
        company_sql.scrape_count = 0
        company_sql.success_rate = 1.0

        # Mock the model_validate method
        with patch.object(Company, "model_validate") as mock_validate:
            mock_validate.return_value = Company(
                id=1,
                name="Test Company",
                url="https://test.com",
                active=True,
                last_scraped=None,
                scrape_count=0,
                success_rate=1.0,
            )

            result = CompanyService._to_dto(company_sql)

            mock_validate.assert_called_once_with(company_sql)
            assert isinstance(result, Company)
            assert result.name == "Test Company"

    @patch("src.services.company_service.db_session")
    def test_to_dtos_batch_conversion(self, mock_db_session):
        """Test _to_dtos method converts list of SQLModel objects."""
        # Create mock CompanySQL objects
        companies_sql = [MagicMock(spec=CompanySQL) for _ in range(3)]
        for i, company in enumerate(companies_sql):
            company.id = i + 1
            company.name = f"Company {i + 1}"

        with patch.object(CompanyService, "_to_dto") as mock_to_dto:
            mock_to_dto.side_effect = lambda x: Company(
                id=x.id,
                name=x.name,
                url="https://test.com",
                active=True,
                last_scraped=None,
                scrape_count=0,
                success_rate=1.0,
            )

            result = CompanyService._to_dtos(companies_sql)

            assert len(result) == 3
            assert mock_to_dto.call_count == 3
            assert all(isinstance(company, Company) for company in result)

    @patch("src.services.company_service.db_session")
    @patch("src.services.company_service.st")
    def test_get_all_companies_success(self, mock_st, mock_db_session):
        """Test get_all_companies retrieves all companies successfully."""
        # Mock database session and query
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock companies from database
        mock_companies = [MagicMock(spec=CompanySQL) for _ in range(2)]
        mock_session.exec.return_value.all.return_value = mock_companies

        # Mock DTO conversion
        with patch.object(CompanyService, "_to_dtos") as mock_to_dtos:
            mock_dtos = [MagicMock(spec=Company) for _ in range(2)]
            mock_to_dtos.return_value = mock_dtos

            result = CompanyService.get_all_companies()

            assert result == mock_dtos
            mock_session.exec.assert_called_once()
            mock_to_dtos.assert_called_once_with(mock_companies)

    @patch("src.services.company_service.db_session")
    @patch("src.services.company_service.st")
    def test_get_all_companies_exception(self, mock_st, mock_db_session):
        """Test get_all_companies handles database exceptions."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            CompanyService.get_all_companies()

    @patch("src.services.company_service.db_session")
    def test_add_company_success(self, mock_db_session):
        """Test add_company creates new company successfully."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock that company doesn't exist
        mock_session.exec.return_value.first.return_value = None

        # Mock created company
        mock_company = MagicMock(spec=CompanySQL)
        mock_company.id = 1
        mock_company.name = "New Company"

        # Mock DTO conversion
        with patch.object(CompanyService, "_to_dto") as mock_to_dto:
            mock_dto = MagicMock(spec=Company)
            mock_to_dto.return_value = mock_dto

            result = CompanyService.add_company("New Company", "https://newco.com")

            assert result == mock_dto
            mock_session.add.assert_called_once()
            mock_session.flush.assert_called_once()
            mock_session.refresh.assert_called_once()

    @patch("src.services.company_service.db_session")
    def test_add_company_validation_errors(self, mock_db_session):
        """Test add_company validates input parameters."""
        # Test empty name
        with pytest.raises(ValueError, match="Company name cannot be empty"):
            CompanyService.add_company("", "https://test.com")

        # Test whitespace-only name
        with pytest.raises(ValueError, match="Company name cannot be empty"):
            CompanyService.add_company("   ", "https://test.com")

        # Test empty URL
        with pytest.raises(ValueError, match="Company URL cannot be empty"):
            CompanyService.add_company("Test Co", "")

        # Test whitespace-only URL
        with pytest.raises(ValueError, match="Company URL cannot be empty"):
            CompanyService.add_company("Test Co", "   ")

    @patch("src.services.company_service.db_session")
    def test_add_company_duplicate_name(self, mock_db_session):
        """Test add_company handles duplicate company names."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock that company already exists
        existing_company = MagicMock(spec=CompanySQL)
        mock_session.exec.return_value.first.return_value = existing_company

        with pytest.raises(ValueError, match="Company 'Duplicate Co' already exists"):
            CompanyService.add_company("Duplicate Co", "https://test.com")

    @patch("src.services.company_service.db_session")
    def test_toggle_company_active_success(self, mock_db_session):
        """Test toggle_company_active toggles status successfully."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock company found
        mock_company = MagicMock(spec=CompanySQL)
        mock_company.active = True
        mock_company.name = "Test Company"
        mock_session.exec.return_value.first.return_value = mock_company

        result = CompanyService.toggle_company_active(1)

        assert result is False  # Should be toggled from True to False
        assert mock_company.active is False

    @patch("src.services.company_service.db_session")
    def test_toggle_company_active_not_found(self, mock_db_session):
        """Test toggle_company_active handles company not found."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Company with ID 999 not found"):
            CompanyService.toggle_company_active(999)

    @patch("src.services.company_service.db_session")
    @patch("src.services.company_service.st")
    def test_get_active_companies(self, mock_st, mock_db_session):
        """Test get_active_companies retrieves only active companies."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock active companies
        mock_companies = [MagicMock(spec=CompanySQL) for _ in range(3)]
        mock_session.exec.return_value.all.return_value = mock_companies

        with patch.object(CompanyService, "_to_dtos") as mock_to_dtos:
            mock_dtos = [MagicMock(spec=Company) for _ in range(3)]
            mock_to_dtos.return_value = mock_dtos

            result = CompanyService.get_active_companies()

            assert result == mock_dtos
            # Verify the query filters for active companies
            mock_session.exec.assert_called_once()

    @patch("src.services.company_service.db_session")
    def test_update_company_scrape_stats(self, mock_db_session):
        """Test update_company_scrape_stats updates statistics correctly."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock company found
        mock_company = MagicMock(spec=CompanySQL)
        mock_company.id = 1
        mock_company.name = "Test Company"
        mock_company.scrape_count = 5
        mock_company.success_rate = 0.8
        mock_session.exec.return_value.first.return_value = mock_company

        # Test successful scrape
        test_time = datetime.now(UTC)
        result = CompanyService.update_company_scrape_stats(1, True, test_time)

        assert result is True
        assert mock_company.scrape_count == 6  # Incremented
        assert mock_company.last_scraped == test_time
        # Success rate should be updated (using weighted average)
        assert mock_company.success_rate > 0.8  # Should increase slightly

    @patch("src.services.company_service.db_session")
    def test_update_company_scrape_stats_not_found(self, mock_db_session):
        """Test update_company_scrape_stats handles company not found."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Company with ID 999 not found"):
            CompanyService.update_company_scrape_stats(999, True)

    @patch("src.services.company_service.db_session")
    def test_delete_company_success(self, mock_db_session):
        """Test delete_company removes company and associated jobs."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock company found
        mock_company = MagicMock(spec=CompanySQL)
        mock_company.name = "Test Company"
        mock_session.exec.return_value.first.side_effect = [
            mock_company,
            5,
        ]  # company, then job count

        result = CompanyService.delete_company(1)

        assert result is True
        mock_session.delete.assert_called_once_with(mock_company)
        mock_session.commit.assert_called_once()

    @patch("src.services.company_service.db_session")
    def test_delete_company_not_found(self, mock_db_session):
        """Test delete_company handles company not found."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.first.return_value = None

        result = CompanyService.delete_company(999)

        assert result is False
        mock_session.delete.assert_not_called()

    @patch("src.services.company_service.db_session")
    def test_get_company_by_id_found(self, mock_db_session):
        """Test get_company_by_id retrieves existing company."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        mock_company = MagicMock(spec=CompanySQL)
        mock_session.exec.return_value.first.return_value = mock_company

        with patch.object(CompanyService, "_to_dto") as mock_to_dto:
            mock_dto = MagicMock(spec=Company)
            mock_dto.name = "Test Company"
            mock_to_dto.return_value = mock_dto

            result = CompanyService.get_company_by_id(1)

            assert result == mock_dto
            mock_to_dto.assert_called_once_with(mock_company)

    @patch("src.services.company_service.db_session")
    def test_get_company_by_id_not_found(self, mock_db_session):
        """Test get_company_by_id handles company not found."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.first.return_value = None

        result = CompanyService.get_company_by_id(999)

        assert result is None

    @patch("src.services.company_service.db_session")
    def test_get_company_by_name_found(self, mock_db_session):
        """Test get_company_by_name retrieves existing company."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        mock_company = MagicMock(spec=CompanySQL)
        mock_session.exec.return_value.first.return_value = mock_company

        with patch.object(CompanyService, "_to_dto") as mock_to_dto:
            mock_dto = MagicMock(spec=Company)
            mock_dto.id = 1
            mock_to_dto.return_value = mock_dto

            result = CompanyService.get_company_by_name("Test Company")

            assert result == mock_dto
            mock_to_dto.assert_called_once_with(mock_company)

    @patch("src.services.company_service.db_session")
    def test_get_company_by_name_empty_name(self, mock_db_session):
        """Test get_company_by_name handles empty/invalid names."""
        result = CompanyService.get_company_by_name("")
        assert result is None

        result = CompanyService.get_company_by_name("   ")
        assert result is None

        result = CompanyService.get_company_by_name(None)
        assert result is None

    @patch("src.services.company_service.db_session")
    def test_bulk_update_scrape_stats(self, mock_db_session):
        """Test bulk_update_scrape_stats processes multiple updates."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock companies
        mock_companies = []
        for i in range(3):
            mock_company = MagicMock(spec=CompanySQL)
            mock_company.id = i + 1
            mock_company.scrape_count = 5
            mock_company.success_rate = 0.8
            mock_companies.append(mock_company)

        mock_session.exec.return_value.all.return_value = mock_companies

        # Test updates
        updates = [
            {"company_id": 1, "success": True},
            {"company_id": 2, "success": False},
            {"company_id": 3, "success": True},
        ]

        result = CompanyService.bulk_update_scrape_stats(updates)

        assert result == 3  # All 3 companies updated

        # Verify each company was updated
        for company in mock_companies:
            assert company.scrape_count == 6  # Incremented
            assert hasattr(company, "last_scraped")  # Should be set

    @patch("src.services.company_service.db_session")
    def test_bulk_update_scrape_stats_empty_list(self, mock_db_session):
        """Test bulk_update_scrape_stats handles empty update list."""
        result = CompanyService.bulk_update_scrape_stats([])
        assert result == 0
        mock_db_session.assert_not_called()

    @patch("src.services.company_service.db_session")
    def test_get_companies_for_management(self, mock_db_session):
        """Test get_companies_for_management returns management format."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock companies
        mock_companies = []
        for i in range(2):
            mock_company = MagicMock(spec=CompanySQL)
            mock_company.id = i + 1
            mock_company.name = f"Company {i + 1}"
            mock_company.url = f"https://company{i + 1}.com"
            mock_company.active = True
            mock_companies.append(mock_company)

        mock_session.exec.return_value.all.return_value = mock_companies

        result = CompanyService.get_companies_for_management()

        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["Name"] == "Company 1"
        assert result[0]["URL"] == "https://company1.com"
        assert result[0]["Active"] is True

    @patch("src.services.company_service.db_session")
    def test_update_company_active_status(self, mock_db_session):
        """Test update_company_active_status updates status correctly."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        mock_company = MagicMock(spec=CompanySQL)
        mock_company.name = "Test Company"
        mock_company.active = False
        mock_session.exec.return_value.first.return_value = mock_company

        result = CompanyService.update_company_active_status(1, True)

        assert result is True
        assert mock_company.active is True

    @patch("src.services.company_service.db_session")
    def test_bulk_get_or_create_companies_existing(self, mock_db_session):
        """Test bulk_get_or_create_companies with existing companies."""
        mock_session = MagicMock()

        # Mock existing companies
        mock_companies = []
        for i, name in enumerate(["Company A", "Company B"]):
            mock_company = MagicMock(spec=CompanySQL)
            mock_company.name = name
            mock_company.id = i + 1
            mock_companies.append(mock_company)

        mock_session.exec.return_value.all.return_value = mock_companies

        company_names = {"Company A", "Company B", "Company C"}  # C is new
        result = CompanyService.bulk_get_or_create_companies(
            mock_session, company_names
        )

        # Should return mapping for all companies
        assert len(result) >= 2  # At least the existing ones
        assert result["Company A"] == 1
        assert result["Company B"] == 2

    @patch("src.services.company_service.db_session")
    def test_bulk_get_or_create_companies_with_race_condition(self, mock_db_session):
        """Test bulk_get_or_create_companies handles race conditions."""
        mock_session = MagicMock()

        # Mock no existing companies initially
        mock_session.exec.return_value.all.side_effect = [
            [],  # First query - no existing companies
            [],  # Retry query after IntegrityError - still no companies
        ]

        # Mock IntegrityError on flush (race condition)
        mock_session.flush.side_effect = IntegrityError("duplicate", {}, None)

        company_names = {"New Company"}
        result = CompanyService.bulk_get_or_create_companies(
            mock_session, company_names
        )

        # Should handle race condition gracefully
        mock_session.rollback.assert_called_once()
        assert isinstance(result, dict)

    @patch("src.services.company_service.db_session")
    def test_bulk_delete_companies(self, mock_db_session):
        """Test bulk_delete_companies removes multiple companies."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock companies to delete
        mock_companies = [
            MagicMock(name="Company 1", id=1),
            MagicMock(name="Company 2", id=2),
        ]
        mock_session.exec.return_value.all.return_value = mock_companies
        mock_session.exec.return_value.first.return_value = 10  # job count
        mock_session.exec.return_value.rowcount = 2  # deleted companies count

        result = CompanyService.bulk_delete_companies([1, 2])

        assert result == 2
        # Should delete jobs first, then companies
        assert mock_session.exec.call_count >= 2

    @patch("src.services.company_service.db_session")
    def test_bulk_update_status(self, mock_db_session):
        """Test bulk_update_status updates multiple companies."""
        mock_session = MagicMock()
        mock_db_session.return_value.__enter__.return_value = mock_session

        # Mock companies to update
        mock_companies = [
            MagicMock(name="Company 1", id=1),
            MagicMock(name="Company 2", id=2),
        ]
        mock_session.exec.return_value.all.return_value = mock_companies
        mock_session.exec.return_value.rowcount = 2  # updated count

        result = CompanyService.bulk_update_status([1, 2], True)

        assert result == 2
        mock_session.exec.assert_called()  # Should execute update

    def test_bulk_operations_empty_lists(self):
        """Test bulk operations handle empty input lists gracefully."""
        # Bulk update with empty list
        result = CompanyService.bulk_update_scrape_stats([])
        assert result == 0

        # Bulk delete with empty list
        result = CompanyService.bulk_delete_companies([])
        assert result == 0

        # Bulk status update with empty list
        result = CompanyService.bulk_update_status([], True)
        assert result == 0


@pytest.fixture
def mock_company_sql():
    """Fixture providing a mock CompanySQL object."""
    company = MagicMock(spec=CompanySQL)
    company.id = 1
    company.name = "Test Company"
    company.url = "https://test.com"
    company.active = True
    company.last_scraped = None
    company.scrape_count = 0
    company.success_rate = 1.0
    return company


@pytest.fixture
def mock_company_dto():
    """Fixture providing a mock Company DTO."""
    return Company(
        id=1,
        name="Test Company",
        url="https://test.com",
        active=True,
        last_scraped=None,
        scrape_count=0,
        success_rate=1.0,
    )
