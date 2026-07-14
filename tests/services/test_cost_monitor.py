"""Behavioral tests for canonical operational cost tracking."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select
from src.database_models import CostEntry
from src.services.cost_monitor import CostMonitor


def test_cost_entry_has_typed_metadata_and_utc_timestamp():
    entry = CostEntry.create_validated(
        service="ai",
        operation="job_extraction",
        cost_usd=0.05,
        extra_data={"model": "test-model", "tokens": 1000},
    )
    assert entry.extra_data == {"model": "test-model", "tokens": 1000}
    assert entry.timestamp.tzinfo == UTC

    normalized = CostEntry.create_validated(
        service="ai",
        operation="job_extraction",
        cost_usd=0,
        timestamp=datetime(2026, 7, 14),
    )
    assert normalized.timestamp.tzinfo == UTC


@pytest.mark.parametrize(
    "values",
    [
        {"service": "", "operation": "op", "cost_usd": 1},
        {"service": "ai", "operation": "", "cost_usd": 1},
        {"service": "ai", "operation": "op", "cost_usd": -0.01},
        {"service": "ai", "operation": "op", "cost_usd": float("nan")},
        {"service": "ai", "operation": "op", "cost_usd": float("inf")},
        {"service": "ai", "operation": "op", "cost_usd": float("-inf")},
    ],
)
def test_cost_entry_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        CostEntry.create_validated(**values)


@pytest.mark.parametrize(
    "budget",
    [0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_monitor_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="finite and positive"):
        CostMonitor(monthly_budget=budget)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf")])
def test_tracking_rejects_nonfinite_cost(session: Session, cost):
    with pytest.raises(ValidationError):
        CostMonitor().track_ai_cost("model", 1, cost, "operation")


@pytest.mark.parametrize(
    ("method", "args", "message"),
    [
        ("track_ai_cost", ("model", -1, 0, "operation"), "tokens"),
        ("track_proxy_cost", (-1, 0, "endpoint"), "requests"),
        ("track_scraping_cost", ("company", -1, 0), "jobs_found"),
    ],
)
def test_tracking_rejects_negative_usage_counts(method, args, message):
    monitor = CostMonitor()
    with pytest.raises(ValueError, match=message):
        getattr(monitor, method)(*args)


def test_tracking_methods_persist_on_canonical_engine(session: Session):
    monitor = CostMonitor()
    with patch.object(monitor, "_check_budget_alerts") as check:
        monitor.track_ai_cost("test-model", 1000, 0.02, "job_extraction")
        monitor.track_proxy_cost(25, 0.15, "proxy.example")
        monitor.track_scraping_cost("Remote AI", 5, 0.08)

    session.expire_all()
    entries = session.exec(select(CostEntry).order_by(CostEntry.id)).all()
    assert [(entry.service, entry.cost_usd) for entry in entries] == [
        ("ai", 0.02),
        ("proxy", 0.15),
        ("scraping", 0.08),
    ]
    assert entries[0].extra_data == {"model": "test-model", "tokens": 1000}
    assert entries[1].extra_data == {"requests": 25, "endpoint": "proxy.example"}
    assert entries[2].operation == "search_run"
    assert check.call_count == 3


def test_monthly_summary_aggregates_services(session: Session):
    monitor = CostMonitor(monthly_budget=50)
    with patch.object(monitor, "_check_budget_alerts"):
        monitor.track_ai_cost("model", 100, 10, "one")
        monitor.track_ai_cost("model", 200, 5, "two")
        monitor.track_proxy_cost(10, 2, "proxy")

    summary = monitor.get_monthly_summary()
    assert summary["costs_by_service"] == {"ai": 15, "proxy": 2}
    assert summary["operation_counts"] == {"ai": 2, "proxy": 1}
    assert summary["total_cost"] == 17
    assert summary["remaining"] == 33
    assert summary["utilization_percent"] == 34
    assert summary["budget_status"] == "within_budget"


@pytest.mark.parametrize(
    ("total_cost", "expected"),
    [
        (0, "within_budget"),
        (29.99, "within_budget"),
        (30, "moderate_usage"),
        (40, "approaching_limit"),
        (50, "over_budget"),
        (60, "over_budget"),
    ],
)
def test_budget_status_thresholds(total_cost, expected):
    assert CostMonitor()._get_budget_status(total_cost) == expected


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            {
                "budget_status": "within_budget",
                "total_cost": 20,
                "monthly_budget": 50,
                "utilization_percent": 40,
            },
            [],
        ),
        (
            {
                "budget_status": "approaching_limit",
                "total_cost": 42,
                "monthly_budget": 50,
                "utilization_percent": 84,
            },
            [{"type": "warning", "message": "Approaching budget limit: 84% used"}],
        ),
        (
            {
                "budget_status": "over_budget",
                "total_cost": 55,
                "monthly_budget": 50,
                "utilization_percent": 110,
            },
            [
                {
                    "type": "error",
                    "message": "Monthly budget exceeded: $55.00 / $50.00",
                }
            ],
        ),
    ],
)
def test_cost_alerts_are_presentation_neutral(summary, expected):
    monitor = CostMonitor()
    with patch.object(monitor, "get_monthly_summary", return_value=summary):
        assert monitor.get_cost_alerts() == expected


def test_cost_alerts_return_stable_error_envelope():
    monitor = CostMonitor()
    with patch.object(
        monitor,
        "get_monthly_summary",
        side_effect=RuntimeError("database offline"),
    ):
        alerts = monitor.get_cost_alerts()
    assert alerts == [
        {"type": "error", "message": "Cost monitoring error: database offline"}
    ]


def test_alert_evaluation_failure_does_not_invalidate_persisted_cost(
    session: Session,
):
    monitor = CostMonitor()
    with patch.object(
        monitor,
        "get_monthly_summary",
        side_effect=RuntimeError("database offline"),
    ):
        monitor.track_ai_cost("model", 10, 0.01, "operation")

    session.expire_all()
    entry = session.exec(select(CostEntry)).one()
    assert entry.cost_usd == 0.01
