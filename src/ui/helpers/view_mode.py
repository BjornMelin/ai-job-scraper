"""View mode selection helpers.

Helper functions for managing view mode selection in UI components.
"""

import streamlit as st


def select_view_mode(tab_key: str) -> tuple[str, int | None]:
    """Create view mode selector with grid/list options.

    Args:
        tab_key: Unique key for the tab to ensure unique widget keys.

    Returns:
        tuple: (view_mode, grid_columns) where view_mode is 'Grid' or 'List'
               and grid_columns is the number of columns for grid view or None for list.
    """
    _, menu_col = st.columns([2, 1])

    with menu_col:
        view_mode = st.selectbox(
            "View",
            ["Grid", "List"],
            key=f"view_mode_{tab_key}",
            help="Choose how to display jobs",
        )

    grid_columns = None
    if view_mode == "Grid":
        _, col_selector = st.columns([3, 1])
        with col_selector:
            grid_columns = st.selectbox(
                "Columns",
                [2, 3, 4],
                index=1,  # Default to 3 columns
                key=f"grid_columns_{tab_key}",
                help="Number of columns in grid view",
            )

    return view_mode, grid_columns


def apply_view_mode(
    jobs: list, view_mode: str, grid_columns: int | None = None
) -> None:
    """Apply the selected view mode to render jobs.

    Args:
        jobs: List of jobs to display.
        view_mode: Either 'Grid' or 'List'.
        grid_columns: Number of columns for grid view, ignored for list view.
    """
    from src.ui.components.cards.job_card import render_jobs_grid, render_jobs_list

    if view_mode == "Grid" and grid_columns:
        render_jobs_grid(jobs, num_columns=grid_columns)
    else:
        render_jobs_list(jobs)
