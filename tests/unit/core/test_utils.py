"""Tests for canonical datetime normalization."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from src.core_utils import ensure_timezone_aware


@pytest.mark.parametrize("value", [None, "", "not-a-date", object()])
def test_invalid_datetime_values_return_none(value):
    assert ensure_timezone_aware(value) is None


def test_iso_and_naive_datetimes_get_a_timezone():
    parsed = ensure_timezone_aware("2024-01-15T10:30:00Z")
    naive = ensure_timezone_aware(datetime(2024, 1, 15, 10, 30))

    assert parsed == datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
    assert naive == datetime(2024, 1, 15, 10, 30, tzinfo=UTC)


def test_aware_datetime_is_preserved():
    value = datetime(2024, 1, 15, 10, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert ensure_timezone_aware(value) is value
