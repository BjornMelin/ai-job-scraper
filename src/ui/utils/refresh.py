"""Utilities for safe, throttled Streamlit reruns.

This module provides small helpers to centralize rerun throttling logic so
pages can trigger real-time updates without duplicating `st.session_state`
bookkeeping.
"""

from __future__ import annotations

import time

from typing import Final

import streamlit as st

DEFAULT_INTERVAL_SECONDS: Final[float] = 2.0


def throttled_rerun(
    session_key: str = "last_refresh",
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    *,
    should_rerun: bool = True,
) -> None:
    """Trigger `st.rerun()` at most once per interval when condition is true.

    Args:
        session_key: Name of the session_state key storing last refresh time.
        interval_seconds: Minimum seconds between reruns.
        should_rerun: Gate condition; when False, does nothing.
    """
    if not should_rerun:
        return

    now = time.time()
    last = float(st.session_state.get(session_key, 0.0) or 0.0)

    if (now - last) >= max(0.0, interval_seconds):
        st.session_state[session_key] = now
        st.rerun()


