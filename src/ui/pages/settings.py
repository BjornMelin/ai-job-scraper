"""Settings management page for the AI Job Scraper application.

This module provides the Streamlit UI for managing application settings including
API keys, LLM provider selection, and scraping limits.
"""

import logging
import os

from typing import Any

import streamlit as st

from groq import Groq
from openai import OpenAI

logger = logging.getLogger(__name__)


def test_api_connection(provider: str, api_key: str) -> tuple[bool, str]:
    """Test API connection for the specified provider.

    Makes actual API calls to validate connectivity and authentication.
    Uses lightweight endpoints to minimize cost and latency.

    Args:
        provider: The LLM provider ("OpenAI" or "Groq").
        api_key: The API key to test.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if not api_key or not api_key.strip():
        return False, "API key is required"

    try:
        if provider == "OpenAI":
            # Basic format validation first
            if not api_key.startswith("sk-"):
                return False, "Invalid OpenAI API key format (should start with 'sk-')"

            # Test actual API connectivity using lightweight models.list() endpoint
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            model_count = len(models.data) if models.data else 0
            return True, f"✅ Connected successfully. {model_count} models available"

        elif provider == "Groq":
            # Basic format validation first
            if len(api_key) < 20:
                return False, "Groq API key appears to be too short"

            # Test actual API connectivity using minimal chat completion
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            completion_id = completion.id[:8] if completion.id else "unknown"
            return True, f"✅ Connected successfully. Response ID: {completion_id}"

        else:
            return False, f"Unknown provider: {provider}"

    except Exception as e:
        logger.error(f"API connection test failed for {provider}: {e}")

        # Provide more specific error messages based on exception type
        error_msg = str(e).lower()
        if (
            "authentication" in error_msg
            or "unauthorized" in error_msg
            or "401" in error_msg
        ):
            return False, "❌ Authentication failed. Please check your API key"
        elif (
            "connection" in error_msg
            or "network" in error_msg
            or "timeout" in error_msg
        ):
            return (
                False,
                "❌ Network connection failed. Please check your internet connection",
            )
        elif "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
            return False, "❌ Rate limit exceeded. Please try again later"
        elif "not found" in error_msg or "404" in error_msg:
            return False, "❌ API endpoint not found. Service may be unavailable"
        else:
            return False, f"❌ Connection failed: {str(e)}"


def load_settings() -> dict[str, Any]:
    """Load current settings from environment and session state.

    Returns:
        Dictionary containing current settings.
    """
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "llm_provider": st.session_state.get("llm_provider", "OpenAI"),
        "max_jobs_per_company": st.session_state.get("max_jobs_per_company", 50),
    }


def save_settings(settings: dict[str, Any]) -> None:
    """Save settings to session state and environment variables.

    Args:
        settings: Dictionary containing settings to save.
    """
    # Save to session state
    st.session_state["llm_provider"] = settings["llm_provider"]
    st.session_state["max_jobs_per_company"] = settings["max_jobs_per_company"]

    # Note: In a production app, you would save API keys securely
    # For now, we'll just note that they should be set as environment variables
    logger.info(
        f"Settings updated: LLM Provider={settings['llm_provider']}, "
        f"Max Jobs={settings['max_jobs_per_company']}"
    )


def show_settings_page() -> None:
    """Display the settings management page.

    Provides functionality to:
    - Configure API keys for OpenAI and Groq
    - Switch between LLM providers
    - Set maximum jobs per company limit
    - Test API connections
    """
    st.title("Settings")
    st.markdown("Configure your AI Job Scraper settings")

    # Load current settings
    settings = load_settings()

    # API Configuration Section
    st.markdown("### 🔑 API Configuration")

    with st.container(border=True):
        # LLM Provider Selection
        col1, col2 = st.columns([2, 1])

        with col1:
            provider = st.radio(
                "LLM Provider",
                options=["OpenAI", "Groq"],
                index=0 if settings["llm_provider"] == "OpenAI" else 1,
                horizontal=True,
                help="Choose your preferred Large Language Model provider",
            )

        with col2:
            st.markdown("**Current Provider**")
            if provider == "OpenAI":
                st.markdown("🤖 OpenAI GPT")
            else:
                st.markdown("⚡ Groq (Ultra-fast)")

        # API Key Configuration
        st.markdown("#### API Keys")

        # OpenAI API Key
        openai_col1, openai_col2 = st.columns([3, 1])
        with openai_col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=settings["openai_api_key"],
                placeholder="sk-...",
                help="Your OpenAI API key (starts with 'sk-')",
            )

        with openai_col2:
            test_openai = st.button(
                "Test Connection",
                key="test_openai",
                disabled=not openai_key,
                help="Test your OpenAI API key",
            )

        if test_openai and openai_key:
            success, message = test_api_connection("OpenAI", openai_key)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

        # Groq API Key
        groq_col1, groq_col2 = st.columns([3, 1])
        with groq_col1:
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                value=settings["groq_api_key"],
                placeholder="gsk_...",
                help="Your Groq API key",
            )

        with groq_col2:
            test_groq = st.button(
                "Test Connection",
                key="test_groq",
                disabled=not groq_key,
                help="Test your Groq API key",
            )

        if test_groq and groq_key:
            success, message = test_api_connection("Groq", groq_key)
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
                f"📊 Conservative limit: Will scrape up to {max_jobs} jobs per company"
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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            try:
                # Update settings dictionary
                settings.update(
                    {
                        "openai_api_key": openai_key,
                        "groq_api_key": groq_key,
                        "llm_provider": provider,
                        "max_jobs_per_company": max_jobs,
                    }
                )

                # Save settings
                save_settings(settings)

                st.success("✅ Settings saved successfully!")
                logger.info("User saved application settings")

                # Show reminder about API keys
                if openai_key or groq_key:
                    st.info(
                        "💡 **Note:** API keys should be set as environment variables "
                        "(OPENAI_API_KEY, GROQ_API_KEY) for security in production."
                    )

            except Exception as e:
                st.error("❌ Failed to save settings. Please try again.")
                logger.error(f"Failed to save settings: {e}", exc_info=True)

    # Current Settings Summary
    st.markdown("### 📋 Current Settings Summary")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**LLM Provider**")
            if settings["llm_provider"] == "OpenAI":
                st.markdown("🤖 OpenAI")
            else:
                st.markdown("⚡ Groq")

            st.markdown("**API Keys Status**")
            openai_status = "✅ Set" if settings["openai_api_key"] else "❌ Not Set"
            groq_status = "✅ Set" if settings["groq_api_key"] else "❌ Not Set"
            st.markdown(f"OpenAI: {openai_status}")
            st.markdown(f"Groq: {groq_status}")

        with col2:
            st.markdown("**Scraping Limits**")
            st.markdown(f"Max jobs per company: **{settings['max_jobs_per_company']}**")

            st.markdown("**Environment Variables**")
            env_openai = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
            env_groq = "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Not Set"
            st.markdown(f"OPENAI_API_KEY: {env_openai}")
            st.markdown(f"GROQ_API_KEY: {env_groq}")


if __name__ == "__main__":
    # For testing the page standalone
    show_settings_page()
