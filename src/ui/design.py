"""Shared visual language and small presentation helpers."""

from datetime import UTC, datetime

import humanize
import streamlit as st

from src.models.job_models import ApplicationStage

WORKFLOW_STAGES = tuple(ApplicationStage)


def apply_design() -> None:
    """Apply the restrained editorial visual system."""
    st.markdown(
        """
        <style>
        :root {
          --paper: #f6f5f1;
          --surface: #ffffff;
          --ink: #17201d;
          --muted: #64706b;
          --line: #d9ddd9;
          --accent: #176b5b;
          --accent-strong: #0f5346;
          --focus: #217c69;
        }

        html, body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(246, 245, 241, 0.96); }
        [data-testid="stMainBlockContainer"] {
          max-width: 1180px;
          padding-top: 2.75rem;
          padding-bottom: 5rem;
        }

        h1, h2, h3 {
          color: var(--ink);
          font-weight: 650;
          letter-spacing: -0.035em;
        }
        h1 { max-width: 16ch; }
        p, label { color: var(--ink); }
        html body .stApp button p { color: inherit !important; }
        [data-testid="stCaptionContainer"] { color: var(--muted); }

        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--surface);
          border-color: var(--line);
          border-radius: 0.75rem;
          box-shadow: 0 1px 0 rgba(23, 32, 29, 0.035);
        }

        button, a[data-testid="stBaseButton-secondary"] {
          transition: border-color 140ms ease, background-color 140ms ease,
            color 140ms ease;
        }
        html body .stApp button:focus-visible,
        html body .stApp a:focus-visible,
        html body .stApp input:focus-visible,
        html body .stApp textarea:focus-visible,
        html body .stApp [role="tab"]:focus-visible {
          outline: 3px solid var(--focus);
          outline-offset: 2px;
        }
        a { color: var(--accent-strong); }
        hr { border-color: var(--line); }

        [data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInputStepUp"] {
          display: none;
        }

        .jt-kicker {
          color: var(--accent-strong);
          font-size: 0.74rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          margin: 0 0 0.5rem;
          text-transform: uppercase;
        }
        .jt-deck {
          color: var(--muted);
          font-size: 1.06rem;
          line-height: 1.65;
          margin: -0.4rem 0 2rem;
          max-width: 62ch;
        }
        .jt-empty {
          color: var(--muted);
          padding: 1rem 0 0.35rem;
        }

        @media (max-width: 700px) {
          [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 3.5rem;
          }
          h1 { font-size: 2.15rem; }
        }

        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, description: str) -> None:
    """Render a consistent page introduction."""
    st.markdown(f'<p class="jt-kicker">{kicker}</p>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<p class="jt-deck">{description}</p>', unsafe_allow_html=True)


def empty_state(title: str, description: str) -> None:
    """Render a quiet, actionable empty state."""
    with st.container(border=True):
        st.subheader(title, anchor=False)
        st.markdown(f'<p class="jt-empty">{description}</p>', unsafe_allow_html=True)


def relative_time(value: datetime | None) -> str:
    """Format a timestamp for compact UI copy."""
    if value is None:
        return "Never"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return humanize.naturaltime(value)


def sentence_case(value: str) -> str:
    """Format enum-like values without title-casing every word."""
    return value.replace("_", " ").capitalize()
