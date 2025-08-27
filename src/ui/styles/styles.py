"""Centralized CSS styles for UI components.

This module provides CSS styles for various UI components, following Streamlit's
best practices for CSS organization and maintainability. All component-specific
styles are centralized here to avoid duplication and improve maintenance.
"""

import streamlit as st

# Mobile-First Responsive Job Grid CSS with CSS Grid Layout
JOB_GRID_CSS = """
<style>
/* Mobile-First Responsive Job Cards using CSS Grid */
.job-cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr));
    gap: 1rem;
    padding: 0.5rem;
    width: 100%;
}

/* Tablet and Desktop Enhancements */
@media (min-width: 640px) {
    .job-cards-container {
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 1.5rem;
        padding: 1rem;
    }
}

@media (min-width: 1024px) {
    .job-cards-container {
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 2rem;
    }
}

/* Individual Job Card Styling */
.job-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #e9ecef;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease-in-out;
    padding: 1rem;
    height: auto;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    cursor: pointer;
}

.job-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border-color: #007bff;
}

/* Job Card Header */
.job-card-header {
    margin-bottom: 0.75rem;
}

.job-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #212529;
    margin: 0 0 0.25rem 0;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.job-card-company {
    color: #6c757d;
    font-size: 0.9rem;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

/* Job Card Body */
.job-card-body {
    flex-grow: 1;
    margin-bottom: 0.75rem;
}

.job-card-location {
    color: #495057;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

.job-card-description {
    color: #6c757d;
    font-size: 0.8rem;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.job-card-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #6c757d;
    margin-bottom: 0.75rem;
}

/* Job Card Footer */
.job-card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #e9ecef;
}

/* Status Badge Styles - Enhanced for Mobile */
.status-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    min-height: 24px;
    line-height: 1;
}

.status-badge.status-new {
    background-color: rgba(59, 130, 246, 0.15);
    color: #1e40af;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.status-badge.status-interested {
    background-color: rgba(251, 191, 36, 0.15);
    color: #d97706;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.status-badge.status-applied {
    background-color: rgba(34, 197, 94, 0.15);
    color: #059669;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-badge.status-rejected {
    background-color: rgba(239, 68, 68, 0.15);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Mobile Touch Targets - Minimum 44px for accessibility */
.job-card-action {
    min-height: 44px;
    min-width: 44px;
    touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
}

/* Favorite Button */
.favorite-btn {
    background: none;
    border: none;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 50%;
    transition: all 0.2s ease;
    min-height: 32px;
    min-width: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.favorite-btn:hover {
    background-color: rgba(255, 193, 7, 0.1);
    transform: scale(1.1);
}

/* Performance Optimizations */
.job-card * {
    backface-visibility: hidden;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* GPU Acceleration for Smooth Animations */
.job-card {
    will-change: transform, box-shadow;
}

/* High Contrast Mode Support */
@media (prefers-contrast: high) {
    .job-card {
        border-width: 2px;
        border-color: #000;
    }

    .status-badge {
        border-width: 2px;
    }
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    .job-card {
        transition: none;
    }

    .job-card:hover {
        transform: none;
    }

    .favorite-btn:hover {
        transform: none;
    }
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    .job-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-color: #4a5568;
        color: #f7fafc;
    }

    .job-card:hover {
        border-color: #63b3ed;
    }

    .job-card-title {
        color: #f7fafc;
    }

    .job-card-company,
    .job-card-location,
    .job-card-description,
    .job-card-meta {
        color: #cbd5e0;
    }

    .job-card-footer {
        border-top-color: #4a5568;
    }
}

/* Loading State */
.job-card-loading {
    opacity: 0.7;
    pointer-events: none;
}

.job-card-loading::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Print Styles */
@media print {
    .job-card {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #000;
        margin-bottom: 1rem;
    }
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
