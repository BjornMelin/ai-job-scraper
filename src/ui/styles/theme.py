"""Optimized theme using CSS variables for maintainability.

This module provides a library-first approach to theming using CSS custom properties
for consistent visual styling across the application.

Key improvements:
- CSS variables for easy maintenance and theming
- Consolidated selectors for better performance
- Reduced code duplication through reusable properties
- Consistent color scheme and component styling
- Dark mode optimized design
"""

import streamlit as st

# Optimized CSS with CSS custom properties
OPTIMIZED_CSS = """
/* CSS custom properties for maintainability */
:root {
    --primary: #1f77b4;
    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
    --bg-primary: #121212;
    --bg-secondary: #1e1e1e;
    --bg-tertiary: #2d2d2d;
    --border: #3a3a3a;
    --border-hover: #4a4a4a;
    --text-primary: #ffffff;
    --text-secondary: #d0d0d0;
    --text-muted: #b0b0b0;
    --shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    --shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.4);

    /* Status badge color variables */
    --status-new-bg: rgba(59, 130, 246, 0.2);
    --status-new-fg: #60a5fa;
    --status-applied-bg: rgba(34, 197, 94, 0.2);
    --status-rejected-bg: rgba(239, 68, 68, 0.2);
    --status-interview-bg: rgba(251, 191, 36, 0.2);
}

/* Base app styles */
body, .stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Button and interactive elements */
.stButton > button {
    background-color: var(--primary);
    color: var(--text-primary);
}

.stDataFrame {
    background-color: var(--bg-secondary);
    color: var(--text-primary);
}

/* Typography */
h1, h2, h3 { color: var(--text-primary); }
a { color: var(--primary); }

/* Card components */
.card {
    background: linear-gradient(
        135deg,
        var(--bg-secondary) 0%,
        var(--bg-tertiary) 100%
    );
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}

.card-title {
    font-size: 1.4em;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
    line-height: 1.3;
}

.card-meta {
    color: var(--text-muted);
    font-size: 0.9em;
    margin-bottom: 12px;
}

.card-desc {
    color: var(--text-secondary);
    font-size: 0.95em;
    line-height: 1.5;
    margin-bottom: 16px;
}

/* Status badges */
.status-new { background: var(--status-new-bg); color: var(--status-new-fg); }
.status-applied { background: var(--status-applied-bg); color: var(--success); }
.status-rejected { background: var(--status-rejected-bg); color: var(--danger); }
.status-interview { background: var(--status-interview-bg); color: var(--warning); }

/* Metric cards */
.metric-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-1px);
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: var(--primary);
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.9em;
    margin-top: 4px;
}
"""


def load_theme() -> None:
    """Load optimized theme with CSS variables.

    Uses library-first CSS custom properties for better maintainability
    and performance compared to the previous implementation.
    """
    st.markdown(f"<style>{OPTIMIZED_CSS}</style>", unsafe_allow_html=True)
