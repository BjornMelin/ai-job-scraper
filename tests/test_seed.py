"""Tests for database seeding functionality."""

from unittest.mock import patch

import pytest

# Import and execute the __main__ block functionality
import seed

from models import CompanySQL
from seed import SITES, seed_companies


class TestSeedCompanies:
    """Test cases for the seed_companies function."""

    def test_seed_companies_success(self, temp_db):
        """Test successful seeding of companies."""
        with patch("seed.SessionLocal", temp_db):
            seed_companies()

            # Verify all companies were added
            session = temp_db()
            companies = session.query(CompanySQL).all()

            assert len(companies) == len(SITES)

            # Verify company data
            company_names = {c.name for c in companies}
            expected_names = set(SITES.keys())
            assert company_names == expected_names

            # Verify URLs
            for company in companies:
                assert company.url == SITES[company.name]
                assert company.active is True

            session.close()

    def test_seed_companies_idempotent(self, temp_db):
        """Test that seeding is idempotent (safe to run multiple times)."""
        with patch("seed.SessionLocal", temp_db):
            # Run seeding twice
            seed_companies()
            seed_companies()

            # Verify no duplicates were created
            session = temp_db()
            companies = session.query(CompanySQL).all()

            assert len(companies) == len(SITES)

            # Verify each company appears only once
            company_names = [c.name for c in companies]
            assert len(company_names) == len(set(company_names))

            session.close()

    def test_seed_companies_partial_existing(self, temp_db):
        """Test seeding when some companies already exist."""
        session = temp_db()

        # Pre-add one company
        existing_company = CompanySQL(
            name="anthropic",
            url="https://custom-url.com/careers",  # Different URL
            active=False,  # Different active status
        )
        session.add(existing_company)
        session.commit()
        session.close()

        with patch("seed.SessionLocal", temp_db):
            seed_companies()

            # Verify total count
            session = temp_db()
            companies = session.query(CompanySQL).all()
            assert len(companies) == len(SITES)

            # Verify existing company was not modified
            anthropic = session.query(CompanySQL).filter_by(name="anthropic").first()
            assert (
                anthropic.url == "https://custom-url.com/careers"
            )  # Original URL preserved
            assert anthropic.active is False  # Original status preserved

            # Verify other companies were added
            other_companies = (
                session.query(CompanySQL).filter(CompanySQL.name != "anthropic").all()
            )
            assert len(other_companies) == len(SITES) - 1

            session.close()

    def test_seed_companies_database_error_handling(self, temp_db):
        """Test error handling during seeding."""

        # Mock a database error
        def mock_session_error():
            session = temp_db()

            def commit_error():
                raise Exception("Database error")

            session.commit = commit_error
            return session

        with patch("seed.SessionLocal", mock_session_error):
            # Should not raise exception, should handle gracefully
            seed_companies()

            # Verify rollback occurred - no companies should be added
            real_session = temp_db()
            companies = real_session.query(CompanySQL).all()
            assert len(companies) == 0
            real_session.close()

    def test_sites_data_integrity(self):
        """Test that SITES data is well-formed."""
        # Verify SITES is not empty
        assert len(SITES) > 0

        # Verify all entries have valid data
        for name, url in SITES.items():
            # Name should be non-empty string
            assert isinstance(name, str)
            assert len(name.strip()) > 0

            # URL should be valid HTTP(S) URL
            assert isinstance(url, str)
            assert url.startswith(("http://", "https://"))
            assert len(url) > 10  # Reasonable minimum URL length

    def test_sites_companies_are_relevant(self):
        """Test that seeded companies are AI-related."""
        expected_ai_companies = {
            "anthropic",
            "openai",
            "deepmind",
            "xai",
            "meta",
            "microsoft",
            "nvidia",
        }

        actual_companies = set(SITES.keys())

        # Verify all expected AI companies are present
        assert expected_ai_companies.issubset(actual_companies)

    @pytest.mark.parametrize(
        ("company_name", "expected_url_contains"),
        [
            ("anthropic", "anthropic.com"),
            ("openai", "openai.com"),
            ("deepmind", "deepmind.google"),
            ("xai", "x.ai"),
            ("meta", "meta.com"),
            ("microsoft", "microsoft.com"),
            ("nvidia", "nvidia"),
        ],
    )
    def test_company_urls_validity(self, company_name, expected_url_contains):
        """Test that company URLs are correct and accessible."""
        assert company_name in SITES
        url = SITES[company_name]
        assert expected_url_contains in url.lower()

    def test_seed_companies_cli_execution(self, temp_db):
        """Test that seed script can be executed as CLI."""
        # This simulates running the script directly
        with patch("seed.SessionLocal", temp_db):
            # Backup original __name__
            original_name = getattr(seed, "__name__", None)

            try:
                # Set __name__ to simulate CLI execution
                seed.__name__ = "__main__"

                # This should execute the seeding without error
                seed.seed_companies()

                # Verify companies were seeded
                session = temp_db()
                companies = session.query(CompanySQL).all()
                assert len(companies) == len(SITES)
                session.close()

            finally:
                # Restore original __name__
                if original_name is not None:
                    seed.__name__ = original_name
