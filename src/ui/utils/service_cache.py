"""Service caching utilities using Streamlit's native caching.

This module implements @st.cache_resource for service objects,
replacing the anti-pattern of storing services in session_state.
This reduces memory usage and follows Streamlit best practices.
"""

import streamlit as st

from src.services.analytics_service import AnalyticsService
from src.services.company_service import CompanyService
from src.services.job_service import JobService
from src.services.search_service import JobSearchService


@st.cache_resource
def get_analytics_service() -> AnalyticsService:
    """Get cached AnalyticsService instance.

    Returns:
        Singleton AnalyticsService instance cached by Streamlit.
    """
    return AnalyticsService()


@st.cache_resource
def get_company_service() -> CompanyService:
    """Get cached CompanyService instance.

    Returns:
        Singleton CompanyService instance cached by Streamlit.
    """
    return CompanyService()


@st.cache_resource
def get_job_service() -> JobService:
    """Get cached JobService instance.

    Returns:
        Singleton JobService instance cached by Streamlit.
    """
    return JobService()


@st.cache_resource
def get_search_service() -> JobSearchService:
    """Get cached JobSearchService instance.

    Returns:
        Singleton JobSearchService instance cached by Streamlit.
    """
    return JobSearchService()


def clear_service_cache() -> None:
    """Clear all cached service instances.

    Use this when services need to be refreshed, such as after
    configuration changes or database updates.
    """
    st.cache_resource.clear()
