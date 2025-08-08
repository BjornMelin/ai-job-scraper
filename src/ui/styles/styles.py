"""Centralized CSS styles for UI components.

This module provides CSS styles for various UI components, following Streamlit's
best practices for CSS organization and maintainability. All component-specific
styles are centralized here to avoid duplication and improve maintenance.
"""

import streamlit as st

# Job grid component styles
JOB_GRID_CSS = """
<style>
.job-card-grid {
    margin-bottom: 1rem;
}
.job-card-container [data-testid="stVerticalBlock"] {
    height: 100%;
}
.job-card-container .element-container {
    height: 100%;
}
.job-card-container [data-testid="stContainer"] {
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Responsive grid adjustments */
@media (max-width: 768px) {
    .job-card-grid {
        flex-direction: column !important;
    }
}

/* Status badge styles */
.status-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
    text-transform: capitalize;
}
.status-badge.status-new {
    background-color: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
}
.status-badge.status-interested {
    background-color: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
}
.status-badge.status-applied {
    background-color: rgba(34, 197, 94, 0.2);
    color: #4ade80;
}
.status-badge.status-rejected {
    background-color: rgba(239, 68, 68, 0.2);
    color: #f87171;
}
</style>
"""

# Company progress card styles
PROGRESS_CARD_CSS = """
<style>
.progress-card {
    background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.progress-card:hover {
    transform: translateY(-1px);
    border-color: #4a4a4a;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.progress-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #3a3a3a;
}
.progress-item:last-child {
    border-bottom: none;
}
.progress-label {
    font-weight: 500;
    color: #d0d0d0;
}
.progress-value {
    font-weight: 600;
    color: #1f77b4;
}
</style>
"""

# Sidebar styles
SIDEBAR_CSS = """
<style>
.sidebar-title {
    font-size: 1.2em;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #1f77b4;
}
.sidebar-section {
    margin-bottom: 24px;
}
.sidebar-section h4 {
    color: #d0d0d0;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
</style>
"""

# Form and input styles
FORM_CSS = """
<style>
.form-container {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}
.form-title {
    font-size: 1.1em;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 16px;
}
.input-group {
    margin-bottom: 16px;
}
.input-label {
    font-size: 0.9em;
    color: #d0d0d0;
    margin-bottom: 4px;
    display: block;
}
.required::after {
    content: " *";
    color: #f87171;
}
</style>
"""


def apply_job_grid_styles() -> None:
    """Apply CSS styles for job grid components."""
    st.markdown(JOB_GRID_CSS, unsafe_allow_html=True)


def apply_progress_card_styles() -> None:
    """Apply CSS styles for progress card components."""
    st.markdown(PROGRESS_CARD_CSS, unsafe_allow_html=True)


def apply_sidebar_styles() -> None:
    """Apply CSS styles for sidebar components."""
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)


def apply_form_styles() -> None:
    """Apply CSS styles for form components."""
    st.markdown(FORM_CSS, unsafe_allow_html=True)


def apply_all_component_styles() -> None:
    """Apply all component styles in one call.

    This is a convenience function for applying all component-specific
    styles when needed globally in the application.
    """
    apply_job_grid_styles()
    apply_progress_card_styles()
    apply_sidebar_styles()
    apply_form_styles()
