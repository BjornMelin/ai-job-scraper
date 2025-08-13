"""UI styles package for centralized CSS management.

This package provides centralized CSS styles and theming for the UI components.
It includes both global theme configuration and component-specific styles.
"""

from src.ui.styles.styles import (
    apply_all_component_styles,
    apply_form_styles,
    apply_job_grid_styles,
    apply_progress_card_styles,
    apply_sidebar_styles,
)
from src.ui.styles.theme import load_theme

__all__ = [
    "apply_all_component_styles",
    "apply_form_styles",
    "apply_job_grid_styles",
    "apply_progress_card_styles",
    "apply_sidebar_styles",
    "load_theme",
]
