"""Tests for constants module."""

import re

import pytest

from src.constants import (
    AI_REGEX,
    RELEVANT_PHRASES,
    SEARCH_KEYWORDS,
    SEARCH_LOCATIONS,
)


class TestRelevantPhrases:
    """Test the relevant phrases list."""

    def test_relevant_phrases_exist(self):
        """Test that relevant phrases are defined and not empty."""
        assert RELEVANT_PHRASES is not None
        assert len(RELEVANT_PHRASES) > 0
        assert isinstance(RELEVANT_PHRASES, list)

    @pytest.mark.parametrize(
        "expected_term",
        (
            "ai",
            "artificial intelligence",
            "machine learning",
            "data scientist",
            "mlops",
            "deep learning",
        ),
    )
    def test_relevant_phrases_contain_expected_terms(self, expected_term):
        """Test that the list contains expected AI/ML terms."""
        assert expected_term in RELEVANT_PHRASES, (
            f"Expected term '{expected_term}' not found"
        )

    @pytest.mark.parametrize("phrase", RELEVANT_PHRASES)
    def test_relevant_phrases_are_strings(self, phrase):
        """Test that all phrases are strings."""
        assert isinstance(phrase, str)
        assert phrase.strip() == phrase  # No leading/trailing whitespace
        assert len(phrase) > 0  # Not empty

    @pytest.mark.parametrize("phrase", RELEVANT_PHRASES)
    def test_relevant_phrases_are_lowercase(self, phrase):
        """Test that all phrases are lowercase for consistency."""
        assert phrase == phrase.lower(), f"Phrase '{phrase}' is not lowercase"


class TestAIRegex:
    """Test the AI regex pattern."""

    def test_ai_regex_exists(self):
        """Test that AI_REGEX is defined as a compiled regex."""
        assert AI_REGEX is not None
        assert isinstance(AI_REGEX, re.Pattern)

    @pytest.mark.parametrize(
        ("test_title", "should_match"),
        (
            ("Senior AI Engineer", True),
            ("Machine Learning Engineer", True),
            ("Data Scientist", True),
            ("MLOps Engineer", True),
            ("Deep Learning Engineer", True),
            ("AI Research Scientist", True),
            ("Software Engineer", False),
            ("Frontend Developer", False),
            ("Backend Engineer", False),
            ("DevOps Engineer", False),
            ("AI engineer position", True),  # case insensitive
            ("Senior ML Engineer", True),
            ("NLP Engineer", True),
            ("Computer Vision Engineer", True),
            ("Full Stack AI Developer", True),  # partial match
            ("Python Developer with AI experience", True),  # AI in description
            ("Agentic AI Engineer", True),
            ("RAG Engineer", True),
            ("CUDA Engineer", True),
            ("Staff ML Engineer", True),
            ("Principal ML Engineer", True),
            ("Generative AI Engineer", True),
        ),
    )
    def test_ai_regex_matches(self, test_title, should_match):
        """Test that the AI regex correctly matches AI-related job titles."""
        match = AI_REGEX.search(test_title)
        if should_match:
            assert match is not None, f"Expected '{test_title}' to match AI regex"
        else:
            assert match is None, f"Expected '{test_title}' not to match AI regex"

    def test_ai_regex_case_insensitive(self):
        """Test that the regex is case insensitive."""
        test_cases = [
            "AI Engineer",
            "ai engineer",
            "Ai Engineer",
            "AI ENGINEER",
            "Machine Learning",
            "MACHINE LEARNING",
            "machine learning",
        ]

        for test_case in test_cases:
            match = AI_REGEX.search(test_case)
            assert match is not None, (
                f"Expected '{test_case}' to match (case insensitive)"
            )

    def test_ai_regex_word_boundaries(self):
        """Test that the regex respects word boundaries."""
        # Should match - complete words
        assert AI_REGEX.search("AI Engineer") is not None
        assert AI_REGEX.search("Machine Learning") is not None

        # Should not match - partial words (if implemented correctly)
        # Note: This depends on how the regex is constructed
        # Most of these should still match because they contain valid AI terms
        assert (
            AI_REGEX.search("The main goal") is None
        )  # "ai" in "main" shouldn't match

    def test_ai_regex_covers_all_phrases(self):
        """Test that the regex can match all phrases in RELEVANT_PHRASES."""
        for phrase in RELEVANT_PHRASES:
            # Test the phrase in context
            test_title = f"Senior {phrase.title()} Engineer"
            match = AI_REGEX.search(test_title)
            assert match is not None, (
                f"Phrase '{phrase}' should be matchable in context"
            )

            # Test the phrase standalone
            match = AI_REGEX.search(phrase)
            assert match is not None, f"Phrase '{phrase}' should match standalone"


class TestSearchConfiguration:
    """Test search keywords and locations."""

    def test_search_keywords_exist(self):
        """Test that search keywords are defined."""
        assert SEARCH_KEYWORDS is not None
        assert isinstance(SEARCH_KEYWORDS, list)
        assert len(SEARCH_KEYWORDS) > 0

    @pytest.mark.parametrize(
        "expected_keyword",
        ("ai", "machine learning", "data science"),
    )
    def test_search_keywords_contain_expected_terms(self, expected_keyword):
        """Test that search keywords contain expected AI/ML terms."""
        assert expected_keyword in SEARCH_KEYWORDS, (
            f"Expected keyword '{expected_keyword}' not found"
        )

    @pytest.mark.parametrize("keyword", SEARCH_KEYWORDS)
    def test_search_keywords_are_strings(self, keyword):
        """Test that all search keywords are strings."""
        assert isinstance(keyword, str)
        assert len(keyword) > 0

    def test_search_locations_exist(self):
        """Test that search locations are defined."""
        assert SEARCH_LOCATIONS is not None
        assert isinstance(SEARCH_LOCATIONS, list)
        assert len(SEARCH_LOCATIONS) > 0

    @pytest.mark.parametrize("expected_location", ("USA", "Remote"))
    def test_search_locations_contain_expected_values(self, expected_location):
        """Test that search locations contain expected values."""
        assert expected_location in SEARCH_LOCATIONS, (
            f"Expected location '{expected_location}' not found"
        )

    @pytest.mark.parametrize("location", SEARCH_LOCATIONS)
    def test_search_locations_are_strings(self, location):
        """Test that all search locations are strings."""
        assert isinstance(location, str)
        assert len(location) > 0


class TestConstantsIntegration:
    """Test integration between different constants."""

    @pytest.mark.parametrize("keyword", SEARCH_KEYWORDS)
    def test_search_keywords_match_relevant_phrases(self, keyword):
        """Test that search keywords are covered by relevant phrases."""
        # Check if the keyword or related terms are in relevant phrases
        keyword_lower = keyword.lower()
        found = any(keyword_lower in phrase for phrase in RELEVANT_PHRASES)
        assert found, (
            f"Search keyword '{keyword}' should be related to relevant phrases"
        )

    @pytest.mark.parametrize("keyword", SEARCH_KEYWORDS)
    def test_regex_matches_search_keywords(self, keyword):
        """Test that the AI regex matches our search keywords."""
        match = AI_REGEX.search(keyword)
        assert match is not None, f"Search keyword '{keyword}' should match AI regex"


class TestApplicationStatuses:
    """Test application status constants."""

    def test_application_statuses_exist(self):
        """Test that APPLICATION_STATUSES constant is defined and not empty."""
        from src.constants import APPLICATION_STATUSES

        assert APPLICATION_STATUSES is not None
        assert len(APPLICATION_STATUSES) > 0
        assert isinstance(APPLICATION_STATUSES, list)

    def test_application_statuses_contain_expected_values(self):
        """Test that APPLICATION_STATUSES contains expected status values."""
        from src.constants import APPLICATION_STATUSES

        # Test the currently defined status values
        expected_statuses = [
            "New",
            "Interested",
            "Applied",
            "Rejected",
        ]

        for expected_status in expected_statuses:
            assert expected_status in APPLICATION_STATUSES, (
                f"Expected status '{expected_status}' not found in APPLICATION_STATUSES"
            )

    def test_application_statuses_are_strings(self):
        """Test that all application statuses are strings."""
        from src.constants import APPLICATION_STATUSES

        for status in APPLICATION_STATUSES:
            assert isinstance(status, str)
            assert status.strip() == status  # No leading/trailing whitespace
            assert len(status) > 0  # Not empty

    def test_application_statuses_proper_case(self):
        """Test that all application statuses use proper case."""
        from src.constants import APPLICATION_STATUSES

        for status in APPLICATION_STATUSES:
            # Should be title case (first letter capitalized for each word)
            assert status.istitle(), f"Status '{status}' should be title case"

    def test_application_statuses_no_duplicates(self):
        """Test that APPLICATION_STATUSES contains no duplicate values."""
        from src.constants import APPLICATION_STATUSES

        # Convert to set to remove duplicates, length should be same
        unique_statuses = set(APPLICATION_STATUSES)
        assert len(unique_statuses) == len(APPLICATION_STATUSES), (
            "APPLICATION_STATUSES contains duplicate values"
        )

    def test_application_statuses_accessible_from_job_card(self):
        """Test that APPLICATION_STATUSES can be imported by job_card.py."""
        # Test import path that job_card.py would use
        try:
            from src.constants import APPLICATION_STATUSES

            assert APPLICATION_STATUSES is not None
        except ImportError as e:
            pytest.fail(f"job_card.py cannot import APPLICATION_STATUSES: {e}")

    def test_application_statuses_workflow_completeness(self):
        """Test that APPLICATION_STATUSES covers basic job application workflow."""
        from src.constants import APPLICATION_STATUSES

        # Should cover the current basic job application workflow
        workflow_statuses = {
            "New",  # Initial discovery
            "Interested",  # Marked as interesting
            "Applied",  # After applying
            "Rejected",  # Application rejected
        }

        for workflow_status in workflow_statuses:
            assert workflow_status in APPLICATION_STATUSES, (
                f"Workflow status '{workflow_status}' missing from APPLICATION_STATUSES"
            )

    def test_application_statuses_integration_with_job_service(self):
        """Test that APPLICATION_STATUSES values work with JobService filtering."""
        from src.constants import APPLICATION_STATUSES

        # All statuses should be valid filter values (no special characters)
        for status in APPLICATION_STATUSES:
            # Should not contain characters that could break SQL queries
            invalid_chars = ["'", '"', ";", "\\", "\n", "\r", "\t"]
            for invalid_char in invalid_chars:
                assert invalid_char not in status, (
                    f"Status '{status}' contains invalid character '{invalid_char}'"
                )

            # Should be reasonable length for database storage and UI display
            assert len(status) <= 50, f"Status '{status}' is too long"
            assert len(status) >= 2, f"Status '{status}' is too short"

    def test_application_statuses_used_in_sample_data(self):
        """Test that APPLICATION_STATUSES values are used in test sample data."""
        from src.constants import APPLICATION_STATUSES

        # Verify that our sample job data uses valid status values
        sample_statuses = ["New", "Interested", "Applied", "Rejected"]

        for sample_status in sample_statuses:
            assert sample_status in APPLICATION_STATUSES, (
                f"Sample status '{sample_status}' should be in APPLICATION_STATUSES"
            )
