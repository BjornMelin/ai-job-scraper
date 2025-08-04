"""Theme and styling configuration for the AI Job Scraper UI.

This module contains CSS styles and theme constants for the Streamlit application,
providing a centralized location for managing visual appearance and styling.
"""

import streamlit as st

# Main CSS styles for the application
MAIN_CSS = """
body {
  background-color: #121212;
  color: #ffffff;
}

.stApp {
  background-color: #121212;
}

.stButton > button {
  background-color: #1f77b4;
  color: white;
}

.stDataFrame {
  background-color: #1e1e1e;
  color: #ffffff;
}

div.row-widget.stHorizontal > div {
  background-color: #1e1e1e;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
}

h1,
h2,
h3 {
  color: #ffffff;
}

a {
  color: #1f77b4;
}
"""

# Additional CSS for job cards and components
COMPONENT_CSS = """
.card {
    background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    border-color: #4a4a4a;
}

.card-title {
    font-size: 1.4em;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 8px;
    line-height: 1.3;
}

.card-meta {
    color: #b0b0b0;
    font-size: 0.9em;
    margin-bottom: 12px;
    line-height: 1.4;
}

.card-desc {
    color: #d0d0d0;
    font-size: 0.95em;
    line-height: 1.5;
    margin-bottom: 16px;
}

.card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #3a3a3a;
}

.status-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.status-new {
    background-color: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.status-interested {
    background-color: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-applied {
    background-color: rgba(34, 197, 94, 0.2);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-rejected {
    background-color: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.metric-card {
    background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.9em;
    color: #b0b0b0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* CSS Variables for consistent theming */
:root {
    --primary-color: #1f77b4;
    --success-color: #4ade80;
    --warning-color: #fbbf24;
    --danger-color: #f87171;
    --text-muted: #b0b0b0;
    --bg-primary: #121212;
    --bg-secondary: #1e1e1e;
    --border-color: #3a3a3a;
}
"""


def load_theme() -> None:
    """Load the application theme by injecting CSS styles into Streamlit.

    This function combines the main CSS styles with component-specific styles
    and injects them into the Streamlit application for consistent theming.
    """
    combined_css = MAIN_CSS + COMPONENT_CSS
    st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)


def apply_custom_styles() -> None:
    """Apply additional custom styles and configurations.

    This function can be used to apply any additional styling that requires
    dynamic configuration or context-specific adjustments.
    """
    # Additional custom styling can be added here
    # For example, responsive design adjustments or theme variations
    pass
