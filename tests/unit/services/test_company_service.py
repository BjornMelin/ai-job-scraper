"""Comprehensive tests for CompanyService class.

This test suite validates CompanyService methods for real-world usage scenarios,
focusing on business functionality, Pydantic DTO conversion, and bulk operations.
Tests cover CRUD operations, edge cases, error conditions, and weighted
success rate calculations.
"""

import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select
from tests.factories import CompanyFactory, JobFactory

from src.models import CompanySQL
from src.schemas import Company
from src.services.company_service import (
    CompanyService,
    CompanyValidationError,
    calculate_weighted_success_rate,
)

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)


@pytest.fixture
def test_engine():
    """Create a test-specific SQLite engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def sample_companies(test_session):
    """Create sample companies for testing."""
    CompanyFactory._meta.sqlalchemy_session = test_session
    base_date = datetime.now(UTC)

    # Create companies with different characteristics
    return [
        CompanyFactory.create(
            name="TechCorp",
            url="https://techcorp.com/careers",
            active=True,
            scrape_count=10,
            success_rate=0.8,
            last_scraped=base_date - timedelta(hours=2),
        ),
        CompanyFactory.create(
            name="InnovateLabs",
            url="https://innovatelabs.com/jobs",
            active=True,
            scrape_count=5,
            success_rate=1.0,
            last_scraped=base_date - timedelta(days=1),
        ),
        CompanyFactory.create(
            name="DataDriven Inc",
            url="https://datadriven.com/careers",
            active=False,
            scrape_count=3,
            success_rate=0.5,
            last_scraped=base_date - timedelta(days=7),
        ),
        CompanyFactory.create(
            name="StartupCo",
            url="https://startup.co/team",
            active=True,
            scrape_count=0,
            success_rate=1.0,
            last_scraped=None,  # Never scraped
        ),
    ]


@pytest.fixture
def sample_jobs(test_session, sample_companies):
    """Create sample jobs linked to companies for testing job counts."""
    JobFactory._meta.sqlalchemy_session = test_session
    base_date = datetime.now(UTC)

    jobs = []

    # TechCorp jobs (3 total: 2 active, 1 archived)
    jobs.extend(
        [
            JobFactory.create(
                company_id=sample_companies[0].id,
                title="Senior Python Developer",
                archived=False,
                posted_date=base_date - timedelta(days=1),
            ),
            JobFactory.create(
                company_id=sample_companies[0].id,
                title="ML Engineer",
                archived=False,
                posted_date=base_date - timedelta(days=2),
            ),
            JobFactory.create(
                company_id=sample_companies[0].id,
                title="Archived Developer",
                archived=True,
                posted_date=base_date - timedelta(days=30),
            ),
        ]
    )

    # InnovateLabs jobs (1 active)
    jobs.append(
        JobFactory.create(
            company_id=sample_companies[1].id,
            title="Full Stack Developer",
            archived=False,
            posted_date=base_date - timedelta(days=3),
        )
    )

    # DataDriven Inc jobs (1 active) - inactive company
    jobs.append(
        JobFactory.create(
            company_id=sample_companies[2].id,
            title="Data Scientist",
            archived=False,
            posted_date=base_date - timedelta(days=5),
        )
    )

    # StartupCo has no jobs

    return jobs


@pytest.fixture
def mock_db_session(test_session):
    """Mock db_session context manager to use test session."""
    with patch("src.services.company_service.db_session") as mock_session:
        mock_session.return_value.__enter__.return_value = test_session
        mock_session.return_value.__exit__.return_value = None
        yield mock_session


class TestCompanyServiceRetrieval:
    """Test CompanyService methods that retrieve company data."""

    def test_get_all_companies_success(self, mock_db_session, sample_companies):
        """Test retrieving all companies ordered by name."""
        companies = CompanyService.get_all_companies()

        # Should return all companies (4) ordered by name
        assert len(companies) == 4
        assert all(isinstance(company, Company) for company in companies)

        # Verify ordering by name
        company_names = [company.name for company in companies]
        assert company_names == sorted(company_names)
        expected_names = ["DataDriven Inc", "InnovateLabs", "StartupCo", "TechCorp"]
        assert company_names == expected_names

        # Verify DTO conversion
        techcorp = next(c for c in companies if c.name == "TechCorp")
        assert isinstance(techcorp, Company)
        assert not isinstance(techcorp, CompanySQL)
        assert techcorp.scrape_count == 10
        assert techcorp.success_rate == 0.8
        assert techcorp.active is True

    def test_get_active_companies_success(self, mock_db_session, sample_companies):
        """Test retrieving only active companies."""
        companies = CompanyService.get_active_companies()

        # Should return only active companies (3)
        assert len(companies) == 3
        assert all(isinstance(company, Company) for company in companies)
        assert all(company.active for company in companies)

        # Verify ordering by name
        company_names = [company.name for company in companies]
        expected_active = ["InnovateLabs", "StartupCo", "TechCorp"]
        assert company_names == expected_active

        # DataDriven Inc should not be included (inactive)
        inactive_names = [c.name for c in companies if not c.active]
        assert len(inactive_names) == 0

    def test_get_company_by_id_success(self, mock_db_session, sample_companies):
        """Test retrieving company by valid ID."""
        company_id = sample_companies[0].id

        company = CompanyService.get_company_by_id(company_id)

        assert company is not None
        assert isinstance(company, Company)
        assert company.id == company_id
        assert company.name == "TechCorp"
        assert company.url == "https://techcorp.com/careers"
        assert company.active is True

    def test_get_company_by_id_not_found(self, mock_db_session, sample_companies):
        """Test retrieving company by invalid ID."""
        company = CompanyService.get_company_by_id(99999)

        assert company is None

    def test_get_company_by_name_success(self, mock_db_session, sample_companies):
        """Test retrieving company by valid name."""
        company = CompanyService.get_company_by_name("TechCorp")

        assert company is not None
        assert isinstance(company, Company)
        assert company.name == "TechCorp"
        assert company.url == "https://techcorp.com/careers"

    def test_get_company_by_name_case_sensitive(
        self, mock_db_session, sample_companies
    ):
        """Test company name lookup is case sensitive."""
        company = CompanyService.get_company_by_name("techcorp")

        assert company is None  # Case sensitive lookup

    def test_get_company_by_name_empty_string(self, mock_db_session, sample_companies):
        """Test handling of empty or whitespace-only names."""
        assert CompanyService.get_company_by_name("") is None
        assert CompanyService.get_company_by_name("   ") is None
        assert CompanyService.get_company_by_name(None) is None

    def test_get_companies_with_job_counts(
        self, mock_db_session, sample_companies, sample_jobs
    ):
        """Test retrieving companies with job statistics."""
        companies_with_stats = CompanyService.get_companies_with_job_counts()

        # Should return all companies with job counts
        assert len(companies_with_stats) == 4

        # Find TechCorp stats
        techcorp_stats = next(
            stats
            for stats in companies_with_stats
            if stats["company"].name == "TechCorp"
        )

        # TechCorp should have 2 active jobs (archived jobs not counted in active_jobs)
        assert techcorp_stats["total_jobs"] == 2  # Only non-archived jobs counted
        assert techcorp_stats["active_jobs"] == 2

        # Find StartupCo stats (no jobs)
        startup_stats = next(
            stats
            for stats in companies_with_stats
            if stats["company"].name == "StartupCo"
        )
        assert startup_stats["total_jobs"] == 0
        assert startup_stats["active_jobs"] == 0

    def test_get_active_companies_count(self, mock_db_session, sample_companies):
        """Test getting count of active companies."""
        count = CompanyService.get_active_companies_count()

        assert isinstance(count, int)
        assert count == 3  # 3 active companies out of 4 total

    def test_get_companies_for_management(self, mock_db_session, sample_companies):
        """Test retrieving companies formatted for management UI."""
        companies = CompanyService.get_companies_for_management()

        assert len(companies) == 4
        assert all(isinstance(company, dict) for company in companies)

        # Verify required keys
        required_keys = {"id", "Name", "URL", "Active"}
        for company in companies:
            assert required_keys.issubset(company.keys())

        # Verify ordering by name
        names = [company["Name"] for company in companies]
        assert names == sorted(names)


class TestCompanyServiceCreation:
    """Test CompanyService methods for creating new companies."""

    def test_add_company_success(self, mock_db_session, test_session):
        """Test successful company creation."""
        company = CompanyService.add_company("NewTech Corp", "https://newtech.com/jobs")

        # Verify DTO returned
        assert isinstance(company, Company)
        assert company.name == "NewTech Corp"
        assert company.url == "https://newtech.com/jobs"
        assert company.active is True
        assert company.scrape_count == 0
        assert company.success_rate == 1.0
        assert company.id is not None

        # Verify actually saved in database
        saved_company = test_session.exec(
            select(CompanySQL).filter_by(name="NewTech Corp")
        ).first()
        assert saved_company is not None
        assert saved_company.name == "NewTech Corp"

    def test_add_company_duplicate_name_error(self, mock_db_session, sample_companies):
        """Test error when adding company with duplicate name."""
        with pytest.raises(
            CompanyValidationError, match="Company 'TechCorp' already exists"
        ):
            CompanyService.add_company("TechCorp", "https://duplicate.com/careers")

    def test_add_company_validation_errors(self, mock_db_session, test_session):
        """Test validation errors for company creation."""
        # Empty name
        with pytest.raises(
            CompanyValidationError, match="Company name cannot be empty"
        ):
            CompanyService.add_company("", "https://example.com")

        # Whitespace-only name
        with pytest.raises(
            CompanyValidationError, match="Company name cannot be empty"
        ):
            CompanyService.add_company("   ", "https://example.com")

        # Empty URL
        with pytest.raises(CompanyValidationError, match="Company URL cannot be empty"):
            CompanyService.add_company("ValidName", "")

        # Whitespace-only URL
        with pytest.raises(CompanyValidationError, match="Company URL cannot be empty"):
            CompanyService.add_company("ValidName", "   ")

    def test_add_company_strips_whitespace(self, mock_db_session, test_session):
        """Test that company name and URL are stripped of whitespace."""
        company = CompanyService.add_company(
            "  SpaceTech  ",
            "  https://space.com/careers  ",
        )

        assert company.name == "SpaceTech"
        assert company.url == "https://space.com/careers"


class TestCompanyServiceUpdates:
    """Test CompanyService methods that modify company data."""

    def test_toggle_company_active_success(self, mock_db_session, sample_companies):
        """Test successful company active status toggle."""
        company_id = sample_companies[0].id  # TechCorp (active)

        # Toggle from active to inactive
        result = CompanyService.toggle_company_active(company_id)
        assert result is False  # New status (inactive)

        # Toggle back to active
        result = CompanyService.toggle_company_active(company_id)
        assert result is True  # Back to active

    def test_toggle_company_active_not_found(self, mock_db_session, sample_companies):
        """Test toggling active status for non-existent company."""
        with pytest.raises(ValueError, match="Company with ID 99999 not found"):
            CompanyService.toggle_company_active(99999)

    def test_update_company_active_status(self, mock_db_session, sample_companies):
        """Test updating company active status directly."""
        company_id = sample_companies[0].id

        # Set to inactive
        result = CompanyService.update_company_active_status(company_id, False)
        assert result is True

        # Set back to active
        result = CompanyService.update_company_active_status(company_id, True)
        assert result is True

    def test_update_company_active_status_not_found(self, mock_db_session):
        """Test updating status for non-existent company."""
        with pytest.raises(ValueError, match="Company with ID 99999 not found"):
            CompanyService.update_company_active_status(99999, True)

    def test_update_company_scrape_stats_success(
        self, mock_db_session, sample_companies
    ):
        """Test successful scraping statistics update."""
        company_id = sample_companies[0].id  # TechCorp
        original_count = sample_companies[0].scrape_count
        original_rate = sample_companies[0].success_rate

        # Update with successful scrape
        result = CompanyService.update_company_scrape_stats(company_id, success=True)
        assert result is True

        # Verify the update in database
        updated_company = mock_db_session.return_value.__enter__.return_value.exec(
            select(CompanySQL).filter_by(id=company_id)
        ).first()

        assert updated_company.scrape_count == original_count + 1
        assert updated_company.last_scraped is not None
        assert isinstance(updated_company.last_scraped, datetime)
        # Success rate should be recalculated using weighted average
        assert updated_company.success_rate != original_rate

    def test_update_company_scrape_stats_with_custom_date(
        self, mock_db_session, sample_companies
    ):
        """Test scrape stats update with custom timestamp."""
        company_id = sample_companies[0].id
        custom_date = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        result = CompanyService.update_company_scrape_stats(
            company_id,
            success=True,
            last_scraped=custom_date,
        )
        assert result is True

        # Verify custom date was used
        updated_company = mock_db_session.return_value.__enter__.return_value.exec(
            select(CompanySQL).filter_by(id=company_id)
        ).first()
        assert updated_company.last_scraped == custom_date

    def test_update_company_scrape_stats_not_found(
        self, mock_db_session, sample_companies
    ):
        """Test updating scrape stats for non-existent company."""
        with pytest.raises(ValueError, match="Company with ID 99999 not found"):
            CompanyService.update_company_scrape_stats(99999, success=True)

    def test_delete_company_success(
        self, mock_db_session, sample_companies, sample_jobs
    ):
        """Test successful company deletion with cascading job deletion."""
        company_id = sample_companies[0].id  # TechCorp with jobs

        result = CompanyService.delete_company(company_id)
        assert result is True

        # Verify company was deleted
        deleted_company = mock_db_session.return_value.__enter__.return_value.exec(
            select(CompanySQL).filter_by(id=company_id)
        ).first()
        assert deleted_company is None

    def test_delete_company_not_found(self, mock_db_session, sample_companies):
        """Test deleting non-existent company."""
        result = CompanyService.delete_company(99999)
        assert result is False


class TestCompanyServiceBulkOperations:
    """Test CompanyService bulk operations."""

    def test_bulk_update_scrape_stats_success(self, mock_db_session, sample_companies):
        """Test bulk updating scrape statistics."""
        base_date = datetime.now(UTC)

        updates = [
            {
                "company_id": sample_companies[0].id,
                "success": True,
                "last_scraped": base_date,
            },
            {
                "company_id": sample_companies[1].id,
                "success": False,
                "last_scraped": base_date - timedelta(minutes=5),
            },
        ]

        result = CompanyService.bulk_update_scrape_stats(updates)
        assert result == 2  # Two companies updated

    def test_bulk_update_scrape_stats_empty_list(
        self, mock_db_session, sample_companies
    ):
        """Test bulk update with empty updates list."""
        result = CompanyService.bulk_update_scrape_stats([])
        assert result == 0

    def test_bulk_update_scrape_stats_partial_success(
        self, mock_db_session, sample_companies
    ):
        """Test bulk update with some invalid company IDs."""
        updates = [
            {
                "company_id": sample_companies[0].id,
                "success": True,
            },
            {
                "company_id": 99999,  # Non-existent
                "success": False,
            },
        ]

        # Should still process valid updates
        result = CompanyService.bulk_update_scrape_stats(updates)
        assert result == 1  # Returns count of successfully updated companies

    def test_bulk_get_or_create_companies_empty_set(self, test_session):
        """Test bulk get/create with empty company names set."""
        company_map = CompanyService.bulk_get_or_create_companies(test_session, set())
        assert company_map == {}

    def test_bulk_get_or_create_companies_existing_companies(
        self, test_session, sample_companies
    ):
        """Test bulk get/create with all existing companies."""
        company_names = {"TechCorp", "InnovateLabs"}

        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            company_names,
        )

        # Should return mapping for existing companies
        assert len(company_map) == 2
        assert "TechCorp" in company_map
        assert "InnovateLabs" in company_map

        # Verify the IDs match the existing companies
        techcorp_id = next(c.id for c in sample_companies if c.name == "TechCorp")
        innovate_id = next(c.id for c in sample_companies if c.name == "InnovateLabs")

        assert company_map["TechCorp"] == techcorp_id
        assert company_map["InnovateLabs"] == innovate_id

    def test_bulk_get_or_create_companies_new_companies(self, test_session):
        """Test bulk get/create with all new companies."""
        company_names = {"NewCorp1", "NewCorp2", "NewCorp3"}

        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            company_names,
        )

        # Should create all new companies
        assert len(company_map) == 3
        assert all(name in company_map for name in company_names)
        assert all(isinstance(company_id, int) for company_id in company_map.values())

        # Verify companies were actually created in database
        for name in company_names:
            company = test_session.exec(select(CompanySQL).filter_by(name=name)).first()
            assert company is not None
            assert company.name == name
            assert company.url == ""  # Default empty URL
            assert company.active is True  # Default active status

    def test_bulk_get_or_create_companies_mixed(self, test_session, sample_companies):
        """Test bulk get/create with mix of existing and new companies."""
        company_names = {"TechCorp", "NewMixedCorp1", "InnovateLabs", "NewMixedCorp2"}

        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            company_names,
        )

        # Should return all companies
        assert len(company_map) == 4
        assert all(name in company_map for name in company_names)

        # Verify existing companies have correct IDs
        techcorp_id = next(c.id for c in sample_companies if c.name == "TechCorp")
        innovate_id = next(c.id for c in sample_companies if c.name == "InnovateLabs")
        assert company_map["TechCorp"] == techcorp_id
        assert company_map["InnovateLabs"] == innovate_id

        # Verify new companies were created
        for new_name in ["NewMixedCorp1", "NewMixedCorp2"]:
            company = test_session.exec(
                select(CompanySQL).filter_by(name=new_name)
            ).first()
            assert company is not None
            assert company.name == new_name

    def test_bulk_get_or_create_companies_race_condition(self, test_session):
        """Test bulk get/create handles race condition scenario."""
        company_names = {"RaceConditionCorp1", "RaceConditionCorp2"}

        # Pre-create one company to simulate another process creating it
        race_company = CompanySQL(name="RaceConditionCorp1", url="", active=True)
        test_session.add(race_company)
        test_session.commit()
        test_session.refresh(race_company)

        # Now call bulk_get_or_create which should handle the mix
        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            company_names,
        )

        # Should still return all companies
        assert len(company_map) == 2
        assert "RaceConditionCorp1" in company_map
        assert "RaceConditionCorp2" in company_map

        # The existing company should have the right ID
        assert company_map["RaceConditionCorp1"] == race_company.id

    def test_bulk_delete_companies_success(self, mock_db_session, sample_companies):
        """Test bulk deletion of companies."""
        company_ids = [sample_companies[0].id, sample_companies[1].id]

        result = CompanyService.bulk_delete_companies(company_ids)
        assert result == 2  # Two companies deleted

    def test_bulk_delete_companies_empty_list(self, mock_db_session):
        """Test bulk delete with empty list."""
        result = CompanyService.bulk_delete_companies([])
        assert result == 0

    def test_bulk_update_status_success(self, mock_db_session, sample_companies):
        """Test bulk status update."""
        company_ids = [sample_companies[0].id, sample_companies[1].id]

        result = CompanyService.bulk_update_status(company_ids, False)
        assert result == 2  # Two companies updated

    def test_bulk_update_status_empty_list(self, mock_db_session):
        """Test bulk status update with empty list."""
        result = CompanyService.bulk_update_status([], True)
        assert result == 0


class TestWeightedSuccessRateCalculation:
    """Test the calculate_weighted_success_rate helper function."""

    def test_first_scrape_success(self):
        """Test success rate calculation for first scrape."""
        # First scrape, success
        result = calculate_weighted_success_rate(0.0, 1, True)
        assert result == 1.0

        # First scrape, failure
        result = calculate_weighted_success_rate(1.0, 1, False)
        assert result == 0.0

    def test_weighted_average_calculation(self):
        """Test weighted average calculation for subsequent scrapes."""
        # Success case: current_rate * 0.8 + new_success * 0.2
        result = calculate_weighted_success_rate(0.8, 5, True)
        expected = 0.8 * 0.8 + 0.2 * 1.0  # 0.64 + 0.2 = 0.84
        assert result == expected

        # Failure case: current_rate * 0.8 + new_failure * 0.2
        result = calculate_weighted_success_rate(0.8, 5, False)
        expected = 0.8 * 0.8 + 0.2 * 0.0  # 0.64 + 0.0 = 0.64
        assert result == expected

    def test_custom_weight_parameter(self):
        """Test weighted average with custom weight parameter."""
        # Using weight 0.9 instead of default 0.8
        result = calculate_weighted_success_rate(0.5, 3, True, weight=0.9)
        expected = 0.9 * 0.5 + 0.1 * 1.0  # 0.45 + 0.1 = 0.55
        assert result == expected

    def test_extreme_values(self):
        """Test calculation with extreme values."""
        # Perfect success rate with failure
        result = calculate_weighted_success_rate(1.0, 10, False)
        expected = 0.8 * 1.0 + 0.2 * 0.0  # 0.8
        assert result == expected

        # Zero success rate with success
        result = calculate_weighted_success_rate(0.0, 10, True)
        expected = 0.8 * 0.0 + 0.2 * 1.0  # 0.2
        assert abs(result - expected) < 1e-10

    def test_edge_case_values(self):
        """Test edge cases for weight calculation."""
        # Weight of 1.0 (ignore new result completely)
        result = calculate_weighted_success_rate(0.5, 5, True, weight=1.0)
        assert result == 0.5

        # Weight of 0.0 (only consider new result)
        result = calculate_weighted_success_rate(0.5, 5, True, weight=0.0)
        assert result == 1.0


class TestCompanyServiceEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_all_companies_empty_database(self, mock_db_session, test_session):
        """Test retrieving companies from empty database."""
        companies = CompanyService.get_all_companies()

        assert len(companies) == 0
        assert isinstance(companies, list)

    def test_get_active_companies_all_inactive(self, mock_db_session, test_session):
        """Test retrieving active companies when all are inactive."""
        # Add only inactive company
        inactive_company = CompanySQL(
            name="InactiveCorp",
            url="https://inactive.com",
            active=False,
        )
        test_session.add(inactive_company)
        test_session.commit()

        companies = CompanyService.get_active_companies()
        assert len(companies) == 0

    def test_company_dto_conversion_all_fields(self, mock_db_session, test_session):
        """Test that all CompanySQL fields are properly converted to Company DTO."""
        base_date = datetime.now(UTC)

        # Create company with all fields populated
        company_sql = CompanySQL(
            name="FullDataCorp",
            url="https://fulldata.com/careers",
            active=False,
            last_scraped=base_date,
            scrape_count=42,
            success_rate=0.75,
        )
        test_session.add(company_sql)
        test_session.commit()
        test_session.refresh(company_sql)

        # Retrieve as DTO
        company_dto = CompanyService.get_company_by_id(company_sql.id)

        # Verify all fields transferred correctly
        assert company_dto.id == company_sql.id
        assert company_dto.name == "FullDataCorp"
        assert company_dto.url == "https://fulldata.com/careers"
        assert company_dto.active is False
        # Compare timestamps without timezone info since SQLite doesn't preserve tz
        assert company_dto.last_scraped.replace(tzinfo=None) == base_date.replace(
            tzinfo=None
        )
        assert company_dto.scrape_count == 42
        assert company_dto.success_rate == 0.75


class TestCompanyServiceErrorHandling:
    """Test error handling and exception scenarios."""

    def test_database_error_during_get_all(self, sample_companies):
        """Test handling of database errors during get_all_companies."""
        with patch("src.services.company_service.db_session") as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            with pytest.raises(Exception, match="Database connection failed"):
                CompanyService.get_all_companies()

    def test_database_error_during_add(self, sample_companies):
        """Test handling of database errors during add_company."""
        with patch("src.services.company_service.db_session") as mock_session:
            mock_session.side_effect = Exception("Database write failed")

            with pytest.raises(Exception, match="Database write failed"):
                CompanyService.add_company("TestCorp", "https://test.com")

    def test_database_error_during_update(self, sample_companies):
        """Test handling of database errors during status updates."""
        with patch("src.services.company_service.db_session") as mock_session:
            mock_session.side_effect = Exception("Database update failed")

            with pytest.raises(Exception, match="Database update failed"):
                CompanyService.update_company_scrape_stats(1, True)


class TestCompanyServiceIntegration:
    """Integration tests combining multiple CompanyService operations."""

    def test_company_lifecycle_workflow(self, mock_db_session, test_session):
        """Test complete company lifecycle.

        Create -> activate/deactivate -> scrape -> delete.
        """
        # 1. Create new company
        company = CompanyService.add_company(
            "LifecycleCorp",
            "https://lifecycle.com/jobs",
        )
        assert company.name == "LifecycleCorp"
        assert company.active is True
        assert company.scrape_count == 0
        company_id = company.id

        # 2. Deactivate company
        result = CompanyService.toggle_company_active(company_id)
        assert result is False  # Now inactive

        # 3. Verify it's not in active companies list
        active_companies = CompanyService.get_active_companies()
        active_names = [c.name for c in active_companies]
        assert "LifecycleCorp" not in active_names

        # 4. Reactivate company
        result = CompanyService.toggle_company_active(company_id)
        assert result is True  # Now active again

        # 5. Update scrape statistics multiple times
        CompanyService.update_company_scrape_stats(company_id, success=True)
        CompanyService.update_company_scrape_stats(company_id, success=True)
        CompanyService.update_company_scrape_stats(company_id, success=False)

        # 6. Verify final state
        final_company = CompanyService.get_company_by_id(company_id)
        assert final_company.active is True
        assert final_company.scrape_count == 3
        assert 0.0 < final_company.success_rate < 1.0  # Mixed success/failure

        # 7. Delete company
        result = CompanyService.delete_company(company_id)
        assert result is True

    def test_filtering_and_search_workflow(self, mock_db_session, sample_companies):
        """Test realistic filtering and search workflow."""
        # 1. Get all companies
        all_companies = CompanyService.get_all_companies()
        assert len(all_companies) == 4

        # 2. Filter to active only
        active_companies = CompanyService.get_active_companies()
        assert len(active_companies) == 3

        # 3. Search for specific company
        techcorp = CompanyService.get_company_by_name("TechCorp")
        assert techcorp is not None
        assert techcorp.active is True

        # 4. Get company with specific ID
        same_company = CompanyService.get_company_by_id(techcorp.id)
        assert same_company.name == techcorp.name
        assert same_company.id == techcorp.id

    def test_bulk_operations_workflow(self, mock_db_session, sample_companies):
        """Test bulk operations workflow."""
        base_date = datetime.now(UTC)

        # Capture original scrape counts before updates
        original_counts = {
            company.id: company.scrape_count for company in sample_companies[:3]
        }

        # 1. Bulk update scrape stats
        updates = [
            {
                "company_id": sample_companies[0].id,
                "success": True,
                "last_scraped": base_date,
            },
            {
                "company_id": sample_companies[1].id,
                "success": False,
                "last_scraped": base_date,
            },
            {
                "company_id": sample_companies[2].id,
                "success": True,
                "last_scraped": base_date,
            },
        ]

        result = CompanyService.bulk_update_scrape_stats(updates)
        assert result == 3

        # 2. Verify updates were applied
        updated_companies = CompanyService.get_all_companies()

        # Check only the companies that were updated
        updated_company_ids = {update["company_id"] for update in updates}
        for company in updated_companies:
            if company.id in updated_company_ids:
                assert company.last_scraped is not None
                # Scrape count should have increased by 1
                original_count = original_counts[company.id]
                assert company.scrape_count == original_count + 1

    def test_statistics_and_management_workflow(
        self, mock_db_session, sample_companies, sample_jobs
    ):
        """Test statistics and management operations workflow."""
        # 1. Get companies with job counts
        companies_with_stats = CompanyService.get_companies_with_job_counts()
        assert len(companies_with_stats) == 4

        # 2. Get management view
        management_companies = CompanyService.get_companies_for_management()
        assert len(management_companies) == 4

        # 3. Get active companies count
        active_count = CompanyService.get_active_companies_count()
        assert active_count == 3

        # 4. Verify TechCorp has the most jobs
        techcorp_stats = next(
            stats
            for stats in companies_with_stats
            if stats["company"].name == "TechCorp"
        )
        assert techcorp_stats["total_jobs"] == 2  # 2 non-archived jobs
        assert techcorp_stats["active_jobs"] == 2
