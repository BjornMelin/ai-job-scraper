"""Settings management page for the AI Job Scraper application.

This module provides the Streamlit UI for managing application settings including
API keys, LLM provider selection, and scraping limits.
"""

import logging
import os

from typing import Any

import streamlit as st

from src.ui.utils import is_streamlit_context

logger = logging.getLogger(__name__)


def test_openai_connection(api_key: str) -> tuple[bool, str]:
    """Test OpenAI API connection for cloud fallback using direct API key passing.

    Makes actual API calls to validate connectivity and authentication.
    Uses LiteLLM's direct API key parameter to avoid environment variable manipulation.

    Args:
        api_key: The OpenAI API key to test.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if not api_key or not api_key.strip():
        return False, "API key is required"

    success = False
    message = ""

    try:
        # Basic format validation first
        if not api_key.startswith("sk-"):
            message = "Invalid OpenAI API key format (should start with 'sk-')"
        else:
            # Test actual API connectivity using LiteLLM's direct API key passing
            # This avoids global environment variable manipulation
            from litellm import completion

            test_messages = [{"role": "user", "content": "Hello"}]
            completion(
                model="gpt-4o-mini",
                messages=test_messages,
                api_key=api_key,  # Pass API key directly to avoid env manipulation
                max_tokens=1,
            )
            success = True
            message = "✅ Connected successfully. OpenAI API key is valid"

    except Exception as e:
        logger.exception("API connection test failed for OpenAI")

        # Provide more specific error messages based on exception type
        error_msg = str(e).lower()
        if (
            "authentication" in error_msg
            or "unauthorized" in error_msg
            or "401" in error_msg
        ):
            message = "❌ Authentication failed. Please check your API key"
        elif (
            "connection" in error_msg
            or "network" in error_msg
            or "timeout" in error_msg
        ):
            message = (
                "❌ Network connection failed. Please check your internet connection"
            )
        elif "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
            message = "❌ Rate limit exceeded. Please try again later"
        elif "not found" in error_msg or "404" in error_msg:
            message = "❌ API endpoint not found. Service may be unavailable"
        else:
            message = f"❌ Connection failed: {e!s}"

    return success, message


def load_settings() -> dict[str, "Any"]:
    """Load current settings from environment and session state.

    Returns:
        Dictionary containing current settings.
    """
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "ai_token_threshold": st.session_state.get("ai_token_threshold", 8000),
        "max_jobs_per_company": st.session_state.get("max_jobs_per_company", 50),
    }


def save_settings(settings: dict[str, "Any"]) -> None:
    """Save settings to session state.

    Args:
        settings: Dictionary containing settings to save.
    """
    # Save to session state with defaults
    st.session_state["ai_token_threshold"] = settings.get("ai_token_threshold", 8000)
    st.session_state["max_jobs_per_company"] = settings.get("max_jobs_per_company", 50)

    logger.info(
        "Settings updated: Token Threshold=%s, Max Jobs=%s",
        st.session_state["ai_token_threshold"],
        st.session_state["max_jobs_per_company"],
    )


def show_settings_page() -> None:
    """Display the simplified settings management page.

    Provides functionality to:
    - Configure OpenAI API key for cloud fallback
    - Set token threshold for local/cloud routing
    - Set maximum jobs per company limit
    """
    st.title("Settings")
    st.markdown("Configure your AI Job Scraper settings - Phase 1 (Simplified)")

    # Load current settings
    settings = load_settings()

    # AI Configuration Section
    st.markdown("### 🤖 AI Configuration")

    with st.container(border=True):
        st.markdown("**Local + Cloud Hybrid Routing**")
        st.info(
            "🏠 **Local Model**: Qwen3-4B-Instruct (primary) | "
            "☁️ **Cloud Fallback**: OpenAI GPT-4o-mini"
        )

        # Token threshold slider
        token_threshold = st.slider(
            "Token Threshold for Cloud Fallback",
            min_value=4000,
            max_value=16000,
            value=settings["ai_token_threshold"],
            step=500,
            help="Requests above this token count will use cloud fallback (OpenAI)",
        )

        # API Key Configuration
        st.markdown("#### Cloud Fallback Configuration")

        # OpenAI API Key
        openai_col1, openai_col2 = st.columns([3, 1])
        with openai_col1:
            openai_key = st.text_input(
                "OpenAI API Key (for cloud fallback)",
                type="password",
                value=settings["openai_api_key"],
                placeholder="sk-...",
                help="Required for cloud fallback when local model context is exceeded",
            )

        with openai_col2:
            test_openai = st.button(
                "Test Connection",
                key="test_openai",
                disabled=not openai_key,
                help="Test your OpenAI API key",
            )

        if test_openai and openai_key:
            success, message = test_openai_connection(openai_key)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    # Scraping Configuration Section
    st.markdown("### 🔧 Scraping Configuration")

    with st.container(border=True):
        # Max jobs per company slider
        max_jobs = st.slider(
            "Maximum Jobs Per Company",
            min_value=10,
            max_value=200,
            value=settings["max_jobs_per_company"],
            step=10,
            help="Limit jobs to scrape per company to prevent runaway scraping",
        )

        # Show current limit info
        if max_jobs <= 30:
            st.info(
                f"📊 Conservative limit: Will scrape up to {max_jobs} jobs per company",
            )
        elif max_jobs <= 100:
            st.info(f"📊 Moderate limit: Will scrape up to {max_jobs} jobs per company")
        else:
            warning_text = (
                f"📊 High limit: Will scrape up to {max_jobs} jobs per company "
                "(may take longer)"
            )
            st.warning(warning_text)

    # Save Settings Button
    st.markdown("---")

    col1, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            try:
                # Update settings dictionary
                settings.update(
                    {
                        "openai_api_key": openai_key,
                        "ai_token_threshold": token_threshold,
                        "max_jobs_per_company": max_jobs,
                    },
                )

                # Save settings
                save_settings(settings)

                st.success("✅ Settings saved successfully!")
                logger.info("User saved application settings")

                # Show reminder about API keys
                if openai_key:
                    st.info(
                        "💡 **Note:** Set OPENAI_API_KEY as environment variable "
                        "for production security."
                    )

            except Exception:
                st.error("❌ Failed to save settings. Please try again.")
                logger.exception("Failed to save settings")

    # Current Settings Summary
    st.markdown("### 📋 Current Settings Summary")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**AI Configuration**")
            st.markdown(f"🤖 Token Threshold: **{settings['ai_token_threshold']:,}**")
            st.markdown("🏠 Primary: **Local Qwen3-4B**")
            st.markdown("☁️ Fallback: **OpenAI GPT-4o-mini**")

        with col2:
            st.markdown("**Status & Limits**")
            st.markdown(
                f"📊 Max jobs per company: **{settings['max_jobs_per_company']}**"
            )

            # API Key status
            openai_status = "✅ Set" if settings["openai_api_key"] else "❌ Not Set"
            st.markdown(f"🔑 OpenAI Key: {openai_status}")

            # Environment variable status
            env_openai = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
            st.markdown(f"🌍 ENV Variable: {env_openai}")


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    show_settings_page()
