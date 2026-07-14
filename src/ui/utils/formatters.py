"""Salary formatting shared by persisted job models."""

type SalaryTuple = tuple[int | None, int | None]


def format_salary_range(salary: SalaryTuple | None) -> str:
    """Format an optional minimum and maximum salary for display."""
    if not salary or salary == (None, None):
        return "Not specified"

    minimum, maximum = salary
    if minimum and maximum:
        if minimum == maximum:
            return f"${minimum:,}"
        return f"${minimum:,} - ${maximum:,}"
    if minimum:
        return f"${minimum:,}+"
    if maximum:
        return f"Up to ${maximum:,}"
    return "Not specified"
