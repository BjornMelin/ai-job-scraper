"""Tests for constants module."""

import re

import pytest

from src.constants import AI_REGEX, RELEVANT_PHRASES, SEARCH_KEYWORDS, SEARCH_LOCATIONS


class TestRelevantPhrases:
    """Test the relevant phrases list."""

    def test_relevant_phrases_exist(self):
        """Test that relevant phrases are defined and not empty."""
        assert RELEVANT_PHRASES is not None
        assert len(RELEVANT_PHRASES) > 0
        assert isinstance(RELEVANT_PHRASES, list)

    def test_relevant_phrases_contain_expected_terms(self):
        """Test that the list contains expected AI/ML terms."""
        expected_terms = [
            "ai",
            "artificial intelligence",
            "machine learning",
            "data scientist",
            "mlops",
            "deep learning",
        ]

        for term in expected_terms:
            assert term in RELEVANT_PHRASES, f"Expected term '{term}' not found"

    def test_relevant_phrases_are_strings(self):
        """Test that all phrases are strings."""
        for phrase in RELEVANT_PHRASES:
            assert isinstance(phrase, str)
            assert phrase.strip() == phrase  # No leading/trailing whitespace
            assert len(phrase) > 0  # Not empty

    def test_relevant_phrases_are_lowercase(self):
        """Test that all phrases are lowercase for consistency."""
        for phrase in RELEVANT_PHRASES:
            assert phrase == phrase.lower(), f"Phrase '{phrase}' is not lowercase"


class TestAIRegex:
    """Test the AI regex pattern."""

    def test_ai_regex_exists(self):
        """Test that AI_REGEX is defined as a compiled regex."""
        assert AI_REGEX is not None
        assert isinstance(AI_REGEX, re.Pattern)

    @pytest.mark.parametrize(
        ("test_title", "should_match"),
        [
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
        ],
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

    def test_search_keywords_contain_expected_terms(self):
        """Test that search keywords contain expected AI/ML terms."""
        expected_keywords = ["ai", "machine learning", "data science"]

        for keyword in expected_keywords:
            assert keyword in SEARCH_KEYWORDS, f"Expected keyword '{keyword}' not found"

    def test_search_keywords_are_strings(self):
        """Test that all search keywords are strings."""
        for keyword in SEARCH_KEYWORDS:
            assert isinstance(keyword, str)
            assert len(keyword) > 0

    def test_search_locations_exist(self):
        """Test that search locations are defined."""
        assert SEARCH_LOCATIONS is not None
        assert isinstance(SEARCH_LOCATIONS, list)
        assert len(SEARCH_LOCATIONS) > 0

    def test_search_locations_contain_expected_values(self):
        """Test that search locations contain expected values."""
        expected_locations = ["USA", "Remote"]

        for location in expected_locations:
            assert location in SEARCH_LOCATIONS, (
                f"Expected location '{location}' not found"
            )

    def test_search_locations_are_strings(self):
        """Test that all search locations are strings."""
        for location in SEARCH_LOCATIONS:
            assert isinstance(location, str)
            assert len(location) > 0


class TestConstantsIntegration:
    """Test integration between different constants."""

    def test_search_keywords_match_relevant_phrases(self):
        """Test that search keywords are covered by relevant phrases."""
        for keyword in SEARCH_KEYWORDS:
            # Check if the keyword or related terms are in relevant phrases
            keyword_lower = keyword.lower()
            found = any(keyword_lower in phrase for phrase in RELEVANT_PHRASES)
            assert found, (
                f"Search keyword '{keyword}' should be related to relevant phrases"
            )

    def test_regex_matches_search_keywords(self):
        """Test that the AI regex matches our search keywords."""
        for keyword in SEARCH_KEYWORDS:
            match = AI_REGEX.search(keyword)
            assert match is not None, (
                f"Search keyword '{keyword}' should match AI regex"
            )
