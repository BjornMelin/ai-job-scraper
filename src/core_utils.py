"""Shared datetime normalization."""

from __future__ import annotations

from datetime import UTC, datetime


def ensure_timezone_aware(value: object) -> datetime | None:
    """Parse an ISO datetime and attach UTC when it has no timezone."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)
