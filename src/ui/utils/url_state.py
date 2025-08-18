"""URL state management utilities for deep linking support.

This module provides utilities to sync application state with URL query parameters,
enabling shareable job searches, bookmarkable views, and browser history support.
It uses Streamlit's st.query_params for native URL parameter management.
"""

import logging

from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

from src.constants import SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN

logger = logging.getLogger(__name__)


def sync_filters_from_url() -> None:
    """Sync filter state from URL parameters on page load.

    Reads filter parameters from URL and updates session state accordingly.
    This should be called early in page rendering to initialize filters from URL.
    """
    try:
        # Initialize filters dict if not present
        if "filters" not in st.session_state:
            st.session_state.filters = _get_default_filters()

        current_filters = st.session_state.filters.copy()

        # Sync keyword search
        if "keyword" in st.query_params:
            current_filters["keyword"] = st.query_params["keyword"]

        # Sync company filters (comma-separated list)
        if "company" in st.query_params:
            company_param = st.query_params["company"]
            if company_param:
                current_filters["company"] = company_param.split(",")
            else:
                current_filters["company"] = []

        # Sync salary filters
        if "salary_min" in st.query_params:
            try:
                salary_min = int(st.query_params["salary_min"])
                if 0 <= salary_min <= 500000:  # Reasonable bounds
                    current_filters["salary_min"] = salary_min
            except ValueError:
                logger.warning(
                    "Invalid salary_min parameter: %s", st.query_params["salary_min"]
                )

        if "salary_max" in st.query_params:
            try:
                salary_max = int(st.query_params["salary_max"])
                if 0 <= salary_max <= 500000:  # Reasonable bounds
                    current_filters["salary_max"] = salary_max
            except ValueError:
                logger.warning(
                    "Invalid salary_max parameter: %s", st.query_params["salary_max"]
                )

        # Sync date filters
        if "date_from" in st.query_params:
            try:
                date_from = datetime.fromisoformat(
                    st.query_params["date_from"]
                ).replace(tzinfo=UTC)
                # Validate date is reasonable (not too far in past/future)
                if datetime(2020, 1, 1, tzinfo=UTC) <= date_from <= datetime.now(UTC):
                    current_filters["date_from"] = date_from
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid date_from parameter: %s", st.query_params["date_from"]
                )

        if "date_to" in st.query_params:
            try:
                date_to = datetime.fromisoformat(st.query_params["date_to"]).replace(
                    tzinfo=UTC
                )
                # Validate date is reasonable (not too far in past/future)
                if (
                    datetime(2020, 1, 1, tzinfo=UTC)
                    <= date_to
                    <= datetime.now(UTC) + timedelta(days=30)
                ):
                    current_filters["date_to"] = date_to
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid date_to parameter: %s", st.query_params["date_to"]
                )

        # Update session state
        st.session_state.filters = current_filters

        logger.debug("Synced filters from URL: %s", current_filters)

    except Exception:
        logger.exception("Failed to sync filters from URL parameters")


def sync_tab_from_url() -> None:
    """Sync tab selection from URL parameters on page load."""
    try:
        if "tab" in st.query_params:
            tab_value = st.query_params["tab"]
            # Validate tab value is one of the expected values
            if tab_value in ["all", "favorites", "applied"]:
                st.session_state.selected_tab = tab_value
                logger.debug("Synced tab from URL: %s", tab_value)

    except Exception:
        logger.exception("Failed to sync tab from URL parameters")


def sync_company_selection_from_url() -> None:
    """Sync company selection from URL parameters for companies page."""
    try:
        if "selected" in st.query_params:
            selected_param = st.query_params["selected"]
            if selected_param:
                # Parse comma-separated company IDs
                try:
                    selected_ids = [
                        int(id_str.strip())
                        for id_str in selected_param.split(",")
                        if id_str.strip()
                    ]
                    st.session_state.selected_companies = set(selected_ids)
                    logger.debug("Synced company selection from URL: %s", selected_ids)
                except ValueError:
                    logger.warning("Invalid company IDs in URL: %s", selected_param)

    except Exception:
        logger.exception("Failed to sync company selection from URL parameters")


def update_url_from_filters() -> None:
    """Update URL parameters when filters change.

    This function should be called whenever filter state changes to keep
    the URL synchronized with the application state.
    """
    try:
        params = {}

        # Only include non-default filter values to keep URLs clean
        current_filters = st.session_state.get("filters", {})

        # Add keyword search if present
        keyword = current_filters.get("keyword", "")
        if keyword:
            params["keyword"] = keyword

        # Add company filters if present
        companies = current_filters.get("company", [])
        if companies:
            params["company"] = ",".join(companies)

        # Add salary filters if different from defaults
        salary_min = current_filters.get("salary_min", SALARY_DEFAULT_MIN)
        salary_max = current_filters.get("salary_max", SALARY_DEFAULT_MAX)

        if salary_min != SALARY_DEFAULT_MIN:
            params["salary_min"] = str(salary_min)
        if salary_max != SALARY_DEFAULT_MAX:
            params["salary_max"] = str(salary_max)

        # Add date filters if different from defaults
        default_date_from = datetime.now(UTC) - timedelta(days=30)
        default_date_to = datetime.now(UTC)

        date_from = current_filters.get("date_from")
        date_to = current_filters.get("date_to")

        if date_from and abs((date_from - default_date_from).days) > 1:
            params["date_from"] = date_from.isoformat()
        if date_to and abs((date_to - default_date_to).days) > 1:
            params["date_to"] = date_to.isoformat()

        # Add tab selection if present
        selected_tab = st.session_state.get("selected_tab")
        if selected_tab and selected_tab != "all":  # Don't include default 'all' tab
            params["tab"] = selected_tab

        # Update URL parameters
        _update_query_params(params)

        logger.debug("Updated URL with params: %s", params)

    except Exception:
        logger.exception("Failed to update URL from filters")


def update_url_from_company_selection() -> None:
    """Update URL parameters when company selection changes on companies page."""
    try:
        params = {}

        # Add selected company IDs if any
        selected_companies = st.session_state.get("selected_companies", set())
        if selected_companies:
            # Convert set to sorted list for consistent URLs
            selected_list = sorted(selected_companies)
            params["selected"] = ",".join(
                str(company_id) for company_id in selected_list
            )

        # Update URL parameters
        _update_query_params(params)

        logger.debug("Updated URL with company selection: %s", params)

    except Exception:
        logger.exception("Failed to update URL from company selection")


def clear_url_params() -> None:
    """Clear all URL parameters."""
    try:
        st.query_params.clear()
        logger.debug("Cleared all URL parameters")
    except Exception:
        logger.exception("Failed to clear URL parameters")


def _update_query_params(params: dict[str, str]) -> None:
    """Update query parameters without triggering a rerun.

    Args:
        params: Dictionary of parameters to set.
    """
    try:
        # Clear existing params and set new ones
        st.query_params.clear()
        if params:
            st.query_params.from_dict(params)
    except Exception:
        logger.exception("Failed to update query parameters")


def _get_default_filters() -> dict[str, Any]:
    """Get default filter configuration.

    Returns:
        Dictionary with default filter values.
    """
    return {
        "company": [],
        "keyword": "",
        "date_from": datetime.now(UTC) - timedelta(days=30),
        "date_to": datetime.now(UTC),
        "salary_min": SALARY_DEFAULT_MIN,
        "salary_max": SALARY_DEFAULT_MAX,
    }


def get_shareable_url() -> str:
    """Get current page URL with all parameters for sharing.

    Returns:
        Complete URL with current filter and view state.
    """
    try:
        # Get base URL (this would need to be configured based on deployment)
        # For now, we'll return just the query string
        params_dict = st.query_params.to_dict()
        if params_dict:
            param_string = "&".join(f"{k}={v}" for k, v in params_dict.items())
            return f"?{param_string}"
    except Exception:
        return ""
        logger.exception("Failed to generate shareable URL")
        return ""


def validate_url_params() -> dict[str, str]:
    """Validate and sanitize URL parameters.

    Returns:
        Dictionary of validation errors if any.
    """
    errors = {}

    try:
        # Validate salary parameters
        if "salary_min" in st.query_params:
            try:
                salary_min = int(st.query_params["salary_min"])
                if salary_min < 0 or salary_min > 500000:
                    errors["salary_min"] = (
                        "Salary minimum must be between 0 and 500,000"
                    )
            except ValueError:
                errors["salary_min"] = "Salary minimum must be a valid number"

        if "salary_max" in st.query_params:
            try:
                salary_max = int(st.query_params["salary_max"])
                if salary_max < 0 or salary_max > 500000:
                    errors["salary_max"] = (
                        "Salary maximum must be between 0 and 500,000"
                    )
            except ValueError:
                errors["salary_max"] = "Salary maximum must be a valid number"

        # Validate date parameters
        if "date_from" in st.query_params:
            try:
                date_from = datetime.fromisoformat(st.query_params["date_from"])
                if date_from.year < 2020 or date_from > datetime.now(UTC):
                    errors["date_from"] = "Date from must be between 2020 and now"
            except ValueError:
                errors["date_from"] = "Date from must be in ISO format (YYYY-MM-DD)"

        if "date_to" in st.query_params:
            try:
                date_to = datetime.fromisoformat(st.query_params["date_to"])
                if date_to.year < 2020 or date_to > datetime.now(UTC) + timedelta(
                    days=30
                ):
                    errors["date_to"] = "Date to must be between 2020 and next month"
            except ValueError:
                errors["date_to"] = "Date to must be in ISO format (YYYY-MM-DD)"

        # Validate tab parameter
        if "tab" in st.query_params and st.query_params["tab"] not in [
            "all",
            "favorites",
            "applied",
        ]:
            errors["tab"] = "Tab must be one of: all, favorites, applied"

    except Exception:
        logger.exception("Error validating URL parameters")
        errors["general"] = "Error validating URL parameters"

    return errors
