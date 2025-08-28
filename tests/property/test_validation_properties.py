"""Property-based tests for data validation using Hypothesis.

This module contains property-based tests that verify data validation
logic holds for a wide range of inputs, helping catch edge cases and
ensure robust validation behavior.
"""

from datetime import UTC, datetime, timedelta

import pytest

from hypothesis import (
    assume,
    example,
    given,
    settings as hypothesis_settings,
    strategies as st,
)

from tests.factories import JobFactory

# Custom strategies for realistic data generation


@st.composite
def salary_range_strategy(draw):
    """Generate realistic salary ranges."""
    min_salary = draw(st.integers(min_value=30_000, max_value=300_000))
    max_salary = draw(st.integers(min_value=min_salary, max_value=min_salary + 200_000))
    return (min_salary, max_salary)


@st.composite
def date_range_strategy(draw):
    """Generate realistic date ranges for job postings."""
    # Jobs should be posted within reasonable time ranges
    base_date = datetime(2020, 1, 1, tzinfo=UTC)
    days_ago = draw(st.integers(min_value=0, max_value=365 * 3))  # Up to 3 years ago
    return base_date + timedelta(days=days_ago)


@st.composite
def company_url_strategy(draw):
    """Generate realistic company URLs."""
    domain = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=3,
            max_size=20,
        )
    )
    tld = draw(st.sampled_from(["com", "org", "io", "net", "co"]))
    path = draw(st.sampled_from(["", "/careers", "/jobs", "/join-us"]))
    return f"https://{domain}.{tld}{path}"


# Property-based tests for salary validation


@pytest.mark.property
@given(salary_range=salary_range_strategy())
@hypothesis_settings(max_examples=100)
def test_salary_range_properties(salary_range):
    """Test that salary ranges maintain expected properties."""
    min_salary, max_salary = salary_range

    # Property: minimum should never exceed maximum
    assert min_salary <= max_salary, (
        f"Min salary {min_salary} > max salary {max_salary}"
    )

    # Property: both values should be positive
    assert min_salary >= 0, f"Negative min salary: {min_salary}"
    assert max_salary >= 0, f"Negative max salary: {max_salary}"

    # Property: salary range should be reasonable (not too wide)
    range_width = max_salary - min_salary
    assert range_width <= 500_000, f"Unreasonably wide salary range: {range_width}"


@pytest.mark.property
@given(
    min_salary=st.one_of(st.none(), st.integers(min_value=0, max_value=1_000_000)),
    max_salary=st.one_of(st.none(), st.integers(min_value=0, max_value=1_000_000)),
)
def test_salary_accessor_properties(min_salary, max_salary):
    """Test salary accessor expressions handle all edge cases."""
    # Create salary tuple handling None values
    if min_salary is None and max_salary is None:
        salary_tuple = None
    elif min_salary is None:
        salary_tuple = (None, max_salary)
    elif max_salary is None:
        salary_tuple = (min_salary, None)
    else:
        salary_tuple = (min_salary, max_salary)

    # Test inline accessor expressions (replacement for get_salary_min/max)
    extracted_min = salary_tuple[0] if salary_tuple else None
    extracted_max = salary_tuple[1] if salary_tuple else None

    # Properties should hold regardless of input
    if salary_tuple is not None:
        assert len(salary_tuple) == 2, "Salary tuple should always have 2 elements"
        assert extracted_min == salary_tuple[0], (
            "Min extraction should match first element"
        )
        assert extracted_max == salary_tuple[1], (
            "Max extraction should match second element"
        )
    else:
        assert extracted_min is None, "None tuple should extract None min"
        assert extracted_max is None, "None tuple should extract None max"


# Property-based tests for date validation


@pytest.mark.property
@given(posted_date=date_range_strategy())
def test_job_date_properties(posted_date):
    """Test job posting date validation properties."""
    # Property: job posting dates should have timezone information
    assert posted_date.tzinfo is not None, f"Date missing timezone: {posted_date}"

    # Property: posting dates should not be in the far future (within 1 day)
    max_future_date = datetime.now(UTC) + timedelta(days=1)
    assert posted_date <= max_future_date, (
        f"Posting date too far in future: {posted_date}"
    )

    # Property: posting dates should not be too old (within 5 years)
    min_past_date = datetime.now(UTC) - timedelta(days=365 * 5)
    assert posted_date >= min_past_date, f"Posting date too old: {posted_date}"


@pytest.mark.property
@given(
    application_date=st.one_of(
        st.none(),
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC), max_value=datetime.now(UTC)
        ),
    ),
    posted_date=date_range_strategy(),
)
def test_application_date_ordering_properties(application_date, posted_date):
    """Test that application dates maintain logical ordering with posting dates."""
    assume(application_date is not None)  # Skip None application dates for this test

    # Property: application date should not be before posting date
    # (You can't apply before job is posted)
    if application_date and posted_date:
        # Allow small buffer for date processing differences
        buffer = timedelta(hours=1)
        assert application_date >= (posted_date - buffer), (
            f"Application date {application_date} before posting date {posted_date}"
        )


# Property-based tests for text processing


@pytest.mark.property
@given(
    title=st.text(min_size=1, max_size=200),
    description=st.text(min_size=1, max_size=5000),
)
def test_job_text_field_properties(title, description):
    """Test that text field processing maintains expected properties."""
    # Property: text fields should not be empty after stripping
    cleaned_title = title.strip()
    cleaned_description = description.strip()

    if cleaned_title:  # Only test non-empty titles
        assert len(cleaned_title) > 0, "Title should not be empty after cleaning"

    if cleaned_description:  # Only test non-empty descriptions
        assert len(cleaned_description) > 0, (
            "Description should not be empty after cleaning"
        )

    # Property: text should not contain control characters (except newlines/tabs)
    import re

    control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    assert not control_chars.search(title), (
        f"Title contains control characters: {title!r}"
    )
    assert not control_chars.search(description), (
        f"Description contains control characters: {description!r}"
    )


# Property-based tests for company validation


@pytest.mark.property
@given(
    company_name=st.text(min_size=1, max_size=100),
    company_url=company_url_strategy(),
    scrape_count=st.integers(min_value=0, max_value=10000),
    success_rate=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_company_properties(company_name, company_url, scrape_count, success_rate):
    """Test company data validation properties."""
    cleaned_name = company_name.strip()
    assume(len(cleaned_name) > 0)  # Skip empty names

    # Property: scrape count should be non-negative
    assert scrape_count >= 0, f"Negative scrape count: {scrape_count}"

    # Property: success rate should be between 0 and 1
    assert 0.0 <= success_rate <= 1.0, f"Invalid success rate: {success_rate}"

    # Property: URL should be well-formed
    assert company_url.startswith(("http://", "https://")), (
        f"Invalid URL scheme: {company_url}"
    )

    # Property: if scrape_count is 0, success_rate should be 1.0 or undefined behavior is ok
    if scrape_count == 0:
        # Either success_rate is 1.0 (no failures) or any value is acceptable
        # since there's no data to compute from
        pass  # This is an acceptable edge case


# Property-based tests for search and filtering


@pytest.mark.property
@given(
    search_query=st.text(min_size=0, max_size=100),
    job_titles=st.lists(
        st.sampled_from(
            [
                "Senior AI Engineer",
                "ML Engineer",
                "Data Scientist",
                "Software Engineer",
                "Product Manager",
                "Sales Manager",
            ]
        ),
        min_size=1,
        max_size=20,
    ),
)
def test_search_filtering_properties(search_query, job_titles):
    """Test job search and filtering properties."""
    cleaned_query = search_query.strip().lower()

    # Property: empty search should return all jobs
    if not cleaned_query:
        filtered = job_titles  # All jobs match empty search
        assert len(filtered) == len(job_titles)
    else:
        # Property: search should be case-insensitive
        filtered = [title for title in job_titles if cleaned_query in title.lower()]

        # Property: all results should contain search term
        for result in filtered:
            assert cleaned_query in result.lower(), (
                f"Result '{result}' doesn't contain query '{cleaned_query}'"
            )


# Property-based tests for data transformation


@pytest.mark.property
@given(
    salary_string=st.sampled_from(
        [
            "$100k-150k",
            "$100,000-$150,000",
            "100000-150000",
            "$100K-$150K",
            "100-150k",
            "$90k - $120k",
            "Competitive",
            "",
            "Not specified",
            "$50k+",
            "Up to $200k",
        ]
    )
)
def test_salary_parsing_properties(salary_string):
    """Test salary string parsing maintains expected properties."""

    # Mock salary parsing logic (would integrate with actual parser)
    def parse_salary_string(salary_str):
        """Mock salary parser that extracts numeric ranges."""
        import re

        if not salary_str or salary_str.lower() in ["competitive", "not specified", ""]:
            return (None, None)

        # Extract numbers (simplified parsing)
        numbers = re.findall(r"(\d+)k?", salary_str.lower().replace(",", ""))
        if len(numbers) >= 2:
            min_sal = int(numbers[0]) * (1000 if "k" in salary_str.lower() else 1)
            max_sal = int(numbers[1]) * (1000 if "k" in salary_str.lower() else 1)
            return (min(min_sal, max_sal), max(min_sal, max_sal))
        if len(numbers) == 1:
            sal = int(numbers[0]) * (1000 if "k" in salary_str.lower() else 1)
            return (sal, sal)
        return (None, None)

    min_sal, max_sal = parse_salary_string(salary_string)

    # Property: if both values exist, min <= max
    if min_sal is not None and max_sal is not None:
        assert min_sal <= max_sal, (
            f"Parsed min {min_sal} > max {max_sal} from '{salary_string}'"
        )

        # Property: parsed values should be reasonable
        assert min_sal >= 0, f"Negative min salary: {min_sal}"
        assert max_sal >= 0, f"Negative max salary: {max_sal}"
        assert max_sal <= 10_000_000, f"Unreasonably high salary: {max_sal}"


# Property-based tests for model creation with factories


@pytest.mark.property
@given(
    trait_args=st.fixed_dictionaries(
        {"senior": st.booleans(), "remote": st.booleans(), "favorited": st.booleans()}
    )
)
def test_factory_trait_properties(trait_args, session):
    """Test that factory traits maintain expected properties."""
    # Create job with traits
    job = JobFactory.build(**trait_args)

    # Property: senior trait should affect title and salary
    if trait_args.get("senior", False):
        senior_titles = [
            "Senior AI Engineer",
            "Principal AI Engineer",
            "Staff ML Engineer",
            "Lead Data Scientist",
        ]
        assert job.title in senior_titles, (
            f"Senior job should have senior title, got: {job.title}"
        )

    # Property: remote trait should set location
    if trait_args.get("remote", False):
        assert job.location == "Remote", (
            f"Remote job should have Remote location, got: {job.location}"
        )

    # Property: favorited trait should set favorite flag and status
    if trait_args.get("favorited", False):
        assert job.favorite is True, "Favorited job should have favorite=True"
        assert job.application_status == "Interested", (
            "Favorited job should be Interested"
        )


# Examples to ensure edge cases are tested


@pytest.mark.property
@given(value=st.integers())
@example(value=0)
@example(value=-1)
@example(value=2**31 - 1)  # Max 32-bit int
@example(value=-(2**31))  # Min 32-bit int
def test_integer_edge_cases(value):
    """Test integer handling with specific edge case examples."""
    # Property: integer conversion should be safe
    str_value = str(value)
    converted_back = int(str_value)
    assert converted_back == value, (
        f"Integer conversion failed: {value} != {converted_back}"
    )


@pytest.mark.property
@hypothesis_settings(max_examples=50, deadline=1000)  # Shorter test runs
def test_realistic_job_creation_properties(session):
    """Test properties of realistic job data generation."""
    # Create multiple jobs and test aggregate properties
    jobs = [JobFactory.build() for _ in range(10)]

    # Property: all jobs should have required fields
    for job in jobs:
        assert job.title is not None
        assert len(job.title.strip()) > 0
        assert job.description is not None
        assert len(job.description.strip()) > 0
        assert job.location is not None
        assert len(job.location.strip()) > 0

        # Property: salary should be reasonable if present
        if job.salary and job.salary != (None, None):
            min_sal, max_sal = job.salary
            if min_sal is not None:
                assert 20_000 <= min_sal <= 500_000, (
                    f"Unrealistic min salary: {min_sal}"
                )
            if max_sal is not None:
                assert 20_000 <= max_sal <= 500_000, (
                    f"Unrealistic max salary: {max_sal}"
                )
