"""Widget-first URL state management for deep linking support.

This module provides utilities to sync WIDGET KEYS with URL query parameters,
eliminating manual session state management in favor of native Streamlit widgets.
It enables shareable job searches, bookmarkable views, and browser history support.
"""

import logging

from datetime import UTC, datetime, timedelta

import streamlit as st

from src.constants import SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN

logger = logging.getLogger(__name__)


def sync_widgets_from_url() -> None:
    """Sync WIDGET KEYS from URL parameters on page load.

    Reads filter parameters from URL and updates widget keys directly.
    This eliminates manual session state management in favor of widget keys.
    """
    try:
        # Sync keyword search to widget key
        if "keyword" in st.query_params:
            st.session_state["keyword_search"] = st.query_params["keyword"]

        # Sync company filters to widget key (comma-separated list)
        if "company" in st.query_params:
            company_param = st.query_params["company"]
            if company_param:
                st.session_state["company_filter"] = company_param.split(",")
            else:
                st.session_state["company_filter"] = []

        # Sync salary range to widget key
        salary_min = SALARY_DEFAULT_MIN
        salary_max = SALARY_DEFAULT_MAX

        if "salary_min" in st.query_params:
            try:
                salary_min = int(st.query_params["salary_min"])
                if not (0 <= salary_min <= 500000):  # Reasonable bounds check
                    salary_min = SALARY_DEFAULT_MIN
            except ValueError:
                logger.warning(
                    "Invalid salary_min parameter: %s", st.query_params["salary_min"]
                )

        if "salary_max" in st.query_params:
            try:
                salary_max = int(st.query_params["salary_max"])
                if not (0 <= salary_max <= 500000):  # Reasonable bounds check
                    salary_max = SALARY_DEFAULT_MAX
            except ValueError:
                logger.warning(
                    "Invalid salary_max parameter: %s", st.query_params["salary_max"]
                )

        # Set salary range widget key as tuple
        if salary_min != SALARY_DEFAULT_MIN or salary_max != SALARY_DEFAULT_MAX:
            st.session_state["salary_range_filter"] = (salary_min, salary_max)

        # Sync date filters to widget keys
        if "date_from" in st.query_params:
            try:
                date_from = datetime.fromisoformat(
                    st.query_params["date_from"]
                ).replace(tzinfo=UTC)
                # Validate date is reasonable
                if datetime(2020, 1, 1, tzinfo=UTC) <= date_from <= datetime.now(UTC):
                    st.session_state["date_from_filter"] = date_from.date()
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid date_from parameter: %s", st.query_params["date_from"]
                )

        if "date_to" in st.query_params:
            try:
                date_to = datetime.fromisoformat(st.query_params["date_to"]).replace(
                    tzinfo=UTC
                )
                # Validate date is reasonable
                if (
                    datetime(2020, 1, 1, tzinfo=UTC)
                    <= date_to
                    <= datetime.now(UTC) + timedelta(days=30)
                ):
                    st.session_state["date_to_filter"] = date_to.date()
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid date_to parameter: %s", st.query_params["date_to"]
                )

        logger.debug("Synced widget keys from URL parameters")

    except Exception:
        logger.exception("Failed to sync widgets from URL parameters")


def sync_tab_from_url() -> None:
    """Sync tab selection from URL parameters on page load.

    Tab selection is one of the few remaining session state keys we preserve
    since it represents cross-page navigation state.
    """
    try:
        if "tab" in st.query_params:
            tab_value = st.query_params["tab"]
            # Validate tab value is one of the expected values
            if tab_value in ["all", "favorites", "applied"]:
                st.session_state.selected_tab = tab_value
                logger.debug("Synced tab from URL: %s", tab_value)

    except Exception:
        logger.exception("Failed to sync tab from URL parameters")


def update_url_from_widgets() -> None:
    """Update URL parameters based on current WIDGET KEY values.

    Synchronizes widget key values to URL query parameters for bookmarking
    and sharing functionality. Omits default values to keep URLs clean.
    """
    try:
        # Get current values from widget keys
        from src.ui.state.session_state import get_current_filters

        filters = get_current_filters()

        # Clear existing query params first
        query_params = {}

        # Add non-default filter values to URL
        if filters.get("keyword"):
            query_params["keyword"] = filters["keyword"]

        if filters.get("company"):
            # Join company list with commas for URL
            query_params["company"] = ",".join(filters["company"])

        # Salary range
        if filters.get("salary_min", SALARY_DEFAULT_MIN) > SALARY_DEFAULT_MIN:
            query_params["salary_min"] = str(filters["salary_min"])

        if filters.get("salary_max", SALARY_DEFAULT_MAX) < SALARY_DEFAULT_MAX:
            query_params["salary_max"] = str(filters["salary_max"])

        # Date range (only if different from defaults)
        default_date_from = datetime.now(UTC) - timedelta(days=30)
        if (
            filters.get("date_from")
            and abs((filters["date_from"] - default_date_from).days) > 1
        ):
            query_params["date_from"] = filters["date_from"].isoformat()

        default_date_to = datetime.now(UTC)
        if (
            filters.get("date_to")
            and abs((filters["date_to"] - default_date_to).days) > 1
        ):
            query_params["date_to"] = filters["date_to"].isoformat()

        # Add tab selection if not default
        current_tab = st.session_state.get("selected_tab", "all")
        if current_tab != "all":
            query_params["tab"] = current_tab

        # Update URL with new parameters
        st.query_params.clear()
        st.query_params.update(query_params)

        logger.debug("Updated URL from widget values: %s", query_params)

    except Exception:
        logger.exception("Failed to update URL from widget values")


def get_shareable_url() -> str | None:
    """Generate shareable URL with current WIDGET-based filter parameters.

    Returns:
        Shareable URL string with query parameters, or None if no filters active.
    """
    try:
        from src.ui.state.session_state import get_current_filters

        filters = get_current_filters()

        # Check if any filters are active (different from defaults)
        has_active_filters = False

        if filters.get("keyword"):
            has_active_filters = True
        if filters.get("company"):
            has_active_filters = True
        if filters.get("salary_min", SALARY_DEFAULT_MIN) > SALARY_DEFAULT_MIN:
            has_active_filters = True
        if filters.get("salary_max", SALARY_DEFAULT_MAX) < SALARY_DEFAULT_MAX:
            has_active_filters = True

        # Check date filters
        default_date_from = datetime.now(UTC) - timedelta(days=30)
        if (
            filters.get("date_from")
            and abs((filters["date_from"] - default_date_from).days) > 1
        ):
            has_active_filters = True

        default_date_to = datetime.now(UTC)
        if (
            filters.get("date_to")
            and abs((filters["date_to"] - default_date_to).days) > 1
        ):
            has_active_filters = True

        if not has_active_filters:
            return None

        # Build URL with current page and filters
        base_url = st.runtime.get_instance().get_url()
        current_params = dict(st.query_params)

        if current_params:
            param_str = "&".join(f"{k}={v}" for k, v in current_params.items())
            return f"{base_url}?{param_str}"

        return None

    except Exception:
        logger.exception("Failed to generate shareable URL")
        return None


def clear_url_params() -> None:
    """Clear all URL query parameters and reset widget keys."""
    st.query_params.clear()

    # Clear widget keys that sync with URL
    url_synced_widgets = [
        "keyword_search",
        "company_filter",
        "salary_range_filter",
        "date_from_filter",
        "date_to_filter",
    ]

    for widget_key in url_synced_widgets:
        if widget_key in st.session_state:
            del st.session_state[widget_key]


def validate_url_params() -> dict[str, str]:
    """Validate URL parameters and return any validation errors.

    Returns:
        Dictionary mapping parameter names to error messages for invalid values.
    """
    errors = {}

    try:
        # Validate salary parameters
        if "salary_min" in st.query_params:
            try:
                salary_min = int(st.query_params["salary_min"])
                if not (0 <= salary_min <= 500000):
                    errors["salary_min"] = (
                        "Salary minimum must be between $0 and $500,000"
                    )
            except ValueError:
                errors["salary_min"] = "Invalid salary minimum value"

        if "salary_max" in st.query_params:
            try:
                salary_max = int(st.query_params["salary_max"])
                if not (0 <= salary_max <= 500000):
                    errors["salary_max"] = (
                        "Salary maximum must be between $0 and $500,000"
                    )
            except ValueError:
                errors["salary_max"] = "Invalid salary maximum value"

        # Validate date parameters
        if "date_from" in st.query_params:
            try:
                date_from = datetime.fromisoformat(st.query_params["date_from"])
                if not (datetime(2020, 1, 1) <= date_from <= datetime.now()):
                    errors["date_from"] = "Date from must be between 2020 and today"
            except (ValueError, TypeError):
                errors["date_from"] = "Invalid date from format"

        if "date_to" in st.query_params:
            try:
                date_to = datetime.fromisoformat(st.query_params["date_to"])
                if not (
                    datetime(2020, 1, 1)
                    <= date_to
                    <= datetime.now() + timedelta(days=30)
                ):
                    errors["date_to"] = (
                        "Date to must be between 2020 and 30 days from now"
                    )
            except (ValueError, TypeError):
                errors["date_to"] = "Invalid date to format"

        # Validate tab parameter
        if "tab" in st.query_params:
            if st.query_params["tab"] not in ["all", "favorites", "applied"]:
                errors["tab"] = "Invalid tab selection"

    except Exception:
        logger.exception("Error validating URL parameters")
        errors["general"] = "Error validating URL parameters"

    return errors
