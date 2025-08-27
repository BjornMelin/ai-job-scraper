"""Comprehensive pytest tests for the Settings page implementation.

Tests cover:
- API key storage and retrieval from session state
- Mock OpenAI/Groq API validation calls
- LLM provider toggle updates
- Job limit slider functionality
- Settings persistence
"""

import logging
import os

from unittest import mock

import streamlit as st

from openai import OpenAI

from src.ui.pages.settings import (
    load_settings,
    save_settings,
    test_openai_connection,
)


class TestApiConnection:
    """Test API connection validation functionality."""

    def setup_method(self):
        """Reset session state before each test."""
        st.session_state.clear()

    def test_empty_api_key_returns_error(self):
        """Test that empty API key returns appropriate error."""
        success, message = test_openai_connection("")
        assert not success
        assert message == "API key is required"

        success, message = test_openai_connection("   ")
        assert not success
        assert message == "API key is required"

    def test_openai_invalid_format_returns_error(self):
        """Test OpenAI API key format validation."""
        success, message = test_openai_connection("invalid-key")
        assert not success
        assert message == "Invalid OpenAI API key format (should start with 'sk-')"

    @mock.patch.object(OpenAI, "models")
    def test_openai_successful_connection(self, mock_models):
        """Test successful OpenAI API connection."""
        # Mock the models response
        mock_list_result = mock.Mock()
        mock_list_result.data = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_models.list.return_value = mock_list_result

        success, message = test_openai_connection("sk-test_openai_api_key_format")

        assert success
        assert "✅ Connected successfully. 3 models available" in message
        mock_models.list.assert_called_once()

    @mock.patch.object(OpenAI, "models")
    def test_openai_successful_connection_empty_models(self, mock_models):
        """Test successful OpenAI connection with empty models list."""
        # Mock empty models response
        mock_list_result = mock.Mock()
        mock_list_result.data = []
        mock_models.list.return_value = mock_list_result

        success, message = test_openai_connection("sk-test_openai_api_key_format")

        assert success
        assert "✅ Connected successfully. 0 models available" in message

    @mock.patch.object(OpenAI, "models")
    def test_openai_authentication_error(self, mock_models):
        """Test OpenAI authentication error handling."""
        mock_models.list.side_effect = Exception("authentication failed")

        success, message = test_openai_connection("sk-test_invalid_openai_key")

        assert not success
        assert "❌ Authentication failed. Please check your API key" in message

    @mock.patch.object(OpenAI, "models")
    def test_openai_network_error(self, mock_models):
        """Test OpenAI network error handling."""
        mock_models.list.side_effect = Exception("connection timeout")

        success, message = test_openai_connection("sk-test_openai_api_key_format")

        assert not success
        assert (
            "❌ Network connection failed. Please check your internet connection"
            in message
        )

    @mock.patch.object(OpenAI, "models")
    def test_openai_rate_limit_error(self, mock_models):
        """Test OpenAI rate limit error handling."""
        mock_models.list.side_effect = Exception("rate limit exceeded")

        success, message = test_openai_connection("sk-test_openai_api_key_format")

        assert not success
        assert "❌ Rate limit exceeded. Please try again later" in message

    @mock.patch.object(OpenAI, "models")
    def test_openai_not_found_error(self, mock_models):
        """Test OpenAI not found error handling."""
        mock_models.list.side_effect = Exception("404 not found")

        success, message = test_openai_connection("sk-test_openai_api_key_format")

        assert not success
        assert "❌ API endpoint not found. Service may be unavailable" in message

    @mock.patch.object(OpenAI, "models")
    def test_openai_generic_error(self, mock_models):
        """Test OpenAI generic error handling."""
        mock_models.list.side_effect = Exception("Some other error")

        success, message = test_openai_connection("sk-test_openai_api_key_format")

        assert not success
        assert "❌ Connection failed: Some other error" in message


class TestSettingsLoadSave:
    """Test settings loading and saving functionality."""

    def _clear_api_env_vars(self):
        """Helper to clear API key environment variables."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    def setup_method(self):
        """Reset session state and environment before each test."""
        st.session_state.clear()
        self._clear_api_env_vars()

    def teardown_method(self):
        """Clean up after each test."""
        self._clear_api_env_vars()

    def test_load_settings_defaults(self):
        """Test loading settings with default values."""
        settings = load_settings()

        assert settings["openai_api_key"] == ""
        assert settings["ai_token_threshold"] == 8000
        assert settings["max_jobs_per_company"] == 50

    def test_load_settings_from_environment(self):
        """Test loading settings from environment variables."""
        os.environ["OPENAI_API_KEY"] = "test_env_openai_key"

        settings = load_settings()

        assert settings["openai_api_key"] == "test_env_openai_key"

    def test_load_settings_from_session_state(self):
        """Test loading settings from session state."""
        st.session_state["ai_token_threshold"] = 10000
        st.session_state["max_jobs_per_company"] = 75

        settings = load_settings()

        assert settings["ai_token_threshold"] == 10000
        assert settings["max_jobs_per_company"] == 75

    def test_save_settings_updates_session_state(self):
        """Test saving settings updates session state."""
        test_settings = {
            "openai_api_key": "test_openai_api_key_format",
            "ai_token_threshold": 12000,
            "max_jobs_per_company": 100,
        }

        save_settings(test_settings)

        assert st.session_state["ai_token_threshold"] == 12000
        assert st.session_state["max_jobs_per_company"] == 100

    def test_save_settings_logs_correctly(self, caplog):
        """Test that save_settings logs the correct information."""
        test_settings = {
            "openai_api_key": "test_openai_api_key_format",
            "ai_token_threshold": 9000,
            "max_jobs_per_company": 30,
        }

        with caplog.at_level(logging.INFO):
            save_settings(test_settings)

        assert "Settings updated: Token Threshold=9000, Max Jobs=30" in caplog.text


class TestSettingsValidation:
    """Test settings validation and edge cases."""

    def setup_method(self):
        """Reset session state before each test."""
        st.session_state.clear()

    def test_max_jobs_boundary_values(self):
        """Test max jobs boundary values."""
        # Test minimum value
        st.session_state["max_jobs_per_company"] = 10
        settings = load_settings()
        assert settings["max_jobs_per_company"] == 10

        # Test maximum value
        st.session_state["max_jobs_per_company"] = 200
        settings = load_settings()
        assert settings["max_jobs_per_company"] == 200

    def test_invalid_token_threshold_handled(self):
        """Test invalid token threshold value."""
        st.session_state["ai_token_threshold"] = "InvalidThreshold"  # noqa: S105
        settings = load_settings()
        # Should still return the stored value, validation happens at UI level
        assert settings["ai_token_threshold"] == "InvalidThreshold"  # noqa: S105

    def test_negative_max_jobs_handled(self):
        """Test negative max jobs value."""
        st.session_state["max_jobs_per_company"] = -10
        settings = load_settings()
        assert settings["max_jobs_per_company"] == -10  # Raw value returned

    def test_zero_max_jobs_handled(self):
        """Test zero max jobs value."""
        st.session_state["max_jobs_per_company"] = 0
        settings = load_settings()
        assert settings["max_jobs_per_company"] == 0

    def test_empty_string_api_keys(self):
        """Test empty string API keys."""
        os.environ["OPENAI_API_KEY"] = ""

        settings = load_settings()

        assert settings["openai_api_key"] == ""

    def test_whitespace_api_keys(self):
        """Test whitespace-only API keys."""
        os.environ["OPENAI_API_KEY"] = "   "

        settings = load_settings()

        assert settings["openai_api_key"] == "   "


class TestSettingsDataTypes:
    """Test settings handle different data types correctly."""

    def setup_method(self):
        """Reset session state before each test."""
        st.session_state.clear()

    def test_max_jobs_as_string(self):
        """Test max jobs as string value."""
        st.session_state["max_jobs_per_company"] = "75"
        settings = load_settings()
        assert settings["max_jobs_per_company"] == "75"  # Raw value returned

    def test_max_jobs_as_float(self):
        """Test max jobs as float value."""
        st.session_state["max_jobs_per_company"] = 75.5
        settings = load_settings()
        assert settings["max_jobs_per_company"] == 75.5

    def test_none_values_in_session_state(self):
        """Test None values in session state."""
        st.session_state["ai_token_threshold"] = None
        st.session_state["max_jobs_per_company"] = None

        settings = load_settings()

        # st.session_state.get() returns None when value is explicitly None
        # The defaults are only used when keys are missing
        assert settings["ai_token_threshold"] is None
        assert settings["max_jobs_per_company"] is None

    def test_missing_keys_in_session_state(self):
        """Test missing keys in session state use defaults."""
        # Don't set any session state values
        settings = load_settings()

        assert settings["ai_token_threshold"] == 8000
        assert settings["max_jobs_per_company"] == 50


class TestSettingsIntegration:
    """Integration tests for settings functionality."""

    def setup_method(self):
        """Reset session state before each test."""
        st.session_state.clear()

    def test_complete_settings_workflow(self):
        """Test complete settings save and load workflow."""
        # Start with defaults
        initial_settings = load_settings()
        assert initial_settings["ai_token_threshold"] == 8000
        assert initial_settings["max_jobs_per_company"] == 50

        # Update settings
        new_settings = {
            "openai_api_key": "test_new_openai_key",
            "ai_token_threshold": 12000,
            "max_jobs_per_company": 25,
        }
        save_settings(new_settings)

        # Load updated settings
        updated_settings = load_settings()
        assert updated_settings["ai_token_threshold"] == 12000
        assert updated_settings["max_jobs_per_company"] == 25

    def test_settings_persistence_across_loads(self):
        """Test settings persist across multiple loads."""
        # Save settings
        test_settings = {
            "ai_token_threshold": 15000,
            "max_jobs_per_company": 150,
        }
        save_settings(test_settings)

        # Load multiple times
        for _ in range(3):
            settings = load_settings()
            assert settings["ai_token_threshold"] == 15000
            assert settings["max_jobs_per_company"] == 150

    def test_environment_override_behavior(self):
        """Test environment variables override session state."""
        # Set session state values
        st.session_state["openai_api_key"] = "test_session_openai_key"

        # Set environment variables (should override)
        os.environ["OPENAI_API_KEY"] = "test_env_openai_priority"

        settings = load_settings()

        # Environment should take precedence
        assert settings["openai_api_key"] == "test_env_openai_priority"

        # Clean up
        del os.environ["OPENAI_API_KEY"]
