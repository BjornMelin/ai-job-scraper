"""Comprehensive tests for Settings page functionality.

Tests all user interactions, API connection testing, validation logic, and
configuration management for the settings page, achieving 80%+ coverage.
"""

import os

from unittest.mock import Mock, patch

import pytest

from groq import Groq
from openai import OpenAI

from src.ui.pages.settings import (
    load_settings,
    save_settings,
    show_settings_page,
    test_api_connection,
)


class TestApiConnectionTesting:
    """Test API connection validation functionality."""

    def test_test_api_connection_openai_success(self):
        """Test successful OpenAI API connection returns success message."""
        # Arrange
        api_key = "sk-test123456789"

        mock_models = Mock()
        mock_models.data = [{"id": "gpt-4"}, {"id": "gpt-3.5-turbo"}]

        with patch.object(OpenAI, "models") as mock_models_attr:
            mock_models_attr.list.return_value = mock_models

            # Act
            success, message = test_api_connection("OpenAI", api_key)

            # Assert
            assert success is True
            assert "✅ Connected successfully" in message
            assert "2 models available" in message

    def test_test_api_connection_openai_invalid_key_format(self):
        """Test OpenAI API connection with invalid key format."""
        # Arrange
        api_key = "invalid-key-format"

        # Act
        success, message = test_api_connection("OpenAI", api_key)

        # Assert
        assert success is False
        assert "Invalid OpenAI API key format" in message
        assert "should start with 'sk-'" in message

    def test_test_api_connection_openai_authentication_error(self):
        """Test OpenAI API connection with authentication error."""
        # Arrange
        api_key = "sk-invalid123456789"

        with patch.object(OpenAI, "models") as mock_models_attr:
            mock_models_attr.list.side_effect = Exception("401 Unauthorized")

            # Act
            success, message = test_api_connection("OpenAI", api_key)

            # Assert
            assert success is False
            assert "❌ Authentication failed" in message
            assert "Please check your API key" in message

    def test_test_api_connection_groq_success(self):
        """Test successful Groq API connection returns success message."""
        # Arrange
        api_key = "gsk_long_valid_groq_key_12345"

        mock_completion = Mock()
        mock_completion.id = "chatcmpl-123456789"

        with patch.object(Groq, "chat") as mock_chat:
            mock_chat.completions.create.return_value = mock_completion

            # Act
            success, message = test_api_connection("Groq", api_key)

            # Assert
            assert success is True
            assert "✅ Connected successfully" in message
            assert "Response ID: chatcmpl-" in message

    def test_test_api_connection_groq_key_too_short(self):
        """Test Groq API connection with key that's too short."""
        # Arrange
        api_key = "short_key"

        # Act
        success, message = test_api_connection("Groq", api_key)

        # Assert
        assert success is False
        assert "Groq API key appears to be too short" in message

    def test_test_api_connection_groq_network_error(self):
        """Test Groq API connection with network error."""
        # Arrange
        api_key = "gsk_long_valid_groq_key_12345"

        with patch.object(Groq, "chat") as mock_chat:
            mock_chat.completions.create.side_effect = Exception("Connection timeout")

            # Act
            success, message = test_api_connection("Groq", api_key)

            # Assert
            assert success is False
            assert "❌ Network connection failed" in message
            assert "Please check your internet connection" in message

    def test_test_api_connection_groq_rate_limit(self):
        """Test Groq API connection with rate limit error."""
        # Arrange
        api_key = "gsk_long_valid_groq_key_12345"

        with patch.object(Groq, "chat") as mock_chat:
            mock_chat.completions.create.side_effect = Exception(
                "429 rate limit exceeded",
            )

            # Act
            success, message = test_api_connection("Groq", api_key)

            # Assert
            assert success is False
            assert "❌ Rate limit exceeded" in message
            assert "Please try again later" in message

    def test_test_api_connection_empty_api_key(self):
        """Test API connection with empty or whitespace-only API key."""
        # Act & Assert for empty key
        success, message = test_api_connection("OpenAI", "")
        assert success is False
        assert "API key is required" in message

        # Act & Assert for whitespace-only key
        success, message = test_api_connection("OpenAI", "   ")
        assert success is False
        assert "API key is required" in message

        # Act & Assert for None key
        success, message = test_api_connection("OpenAI", None)
        assert success is False
        assert "API key is required" in message

    def test_test_api_connection_unknown_provider(self):
        """Test API connection with unknown provider."""
        # Act
        success, message = test_api_connection("UnknownProvider", "test-key")

        # Assert
        assert success is False
        assert "Unknown provider: UnknownProvider" in message

    @pytest.mark.parametrize(
        ("error_message", "expected_message"),
        (
            ("404 not found", "❌ API endpoint not found"),
            ("quota exceeded", "❌ Rate limit exceeded"),
            ("network error", "❌ Network connection failed"),
            ("Some unexpected error", "❌ Connection failed: Some unexpected error"),
        ),
    )
    def test_test_api_connection_error_handling(self, error_message, expected_message):
        """Test various error conditions are handled appropriately."""
        # Arrange
        api_key = "sk-test123456789"

        with patch.object(OpenAI, "models") as mock_models_attr:
            mock_models_attr.list.side_effect = Exception(error_message)

            # Act
            success, message = test_api_connection("OpenAI", api_key)

            # Assert
            assert success is False
            assert expected_message in message


class TestSettingsLoading:
    """Test settings loading and persistence functionality."""

    def test_load_settings_with_environment_variables(self):
        """Test loading settings when environment variables are set."""
        # Arrange
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-env-openai-key",
                    "GROQ_API_KEY": "gsk_env_groq_key",
                },
            ),
            patch("streamlit.session_state", new_callable=dict) as mock_session,
        ):
            mock_session.update(
                {
                    "llm_provider": "Groq",
                    "max_jobs_per_company": 75,
                },
            )

            # Act
            settings = load_settings()

            # Assert
            assert settings["openai_api_key"] == "sk-env-openai-key"
            assert settings["groq_api_key"] == "gsk_env_groq_key"
            assert settings["llm_provider"] == "Groq"
            assert settings["max_jobs_per_company"] == 75

    def test_load_settings_with_defaults(self):
        """Test loading settings with default values when nothing is set."""
        # Arrange
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("streamlit.session_state", new_callable=dict),
        ):
            # Act
            settings = load_settings()

            # Assert
            assert settings["openai_api_key"] == ""
            assert settings["groq_api_key"] == ""
            assert settings["llm_provider"] == "OpenAI"  # Default
            assert settings["max_jobs_per_company"] == 50  # Default

    def test_load_settings_mixed_sources(self):
        """Test loading settings from mixed sources (env vars + session state)."""
        # Arrange
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk-env-key"},
                clear=True,
            ),
            patch("streamlit.session_state", new_callable=dict) as mock_session,
        ):
            mock_session.update(
                {
                    "llm_provider": "OpenAI",
                    "max_jobs_per_company": 100,
                },
            )

            # Act
            settings = load_settings()

            # Assert
            assert settings["openai_api_key"] == "sk-env-key"  # From env
            assert settings["groq_api_key"] == ""  # Not set
            assert settings["llm_provider"] == "OpenAI"  # From session
            assert settings["max_jobs_per_company"] == 100  # From session


class TestSettingsSaving:
    """Test settings saving and session state management."""

    def test_save_settings_updates_session_state(self):
        """Test that save_settings updates session state correctly."""
        # Arrange
        test_settings = {
            "openai_api_key": "sk-test-key",
            "groq_api_key": "gsk_test_key",
            "llm_provider": "Groq",
            "max_jobs_per_company": 80,
        }

        with patch("streamlit.session_state", new_callable=dict) as mock_session:
            # Act
            save_settings(test_settings)

            # Assert
            assert mock_session["llm_provider"] == "Groq"
            assert mock_session["max_jobs_per_company"] == 80

    def test_save_settings_with_defaults(self):
        """Test that save_settings uses defaults for missing values."""
        # Arrange
        incomplete_settings = {
            "openai_api_key": "sk-test-key",
        }

        with patch("streamlit.session_state", new_callable=dict) as mock_session:
            # Act
            save_settings(incomplete_settings)

            # Assert
            assert mock_session["llm_provider"] == "OpenAI"  # Default
            assert mock_session["max_jobs_per_company"] == 50  # Default

    def test_save_settings_logs_information(self):
        """Test that save_settings logs the configuration."""
        # Arrange
        test_settings = {
            "llm_provider": "OpenAI",
            "max_jobs_per_company": 60,
        }

        with (
            patch("streamlit.session_state", new_callable=dict),
            patch("src.ui.pages.settings.logger") as mock_logger,
        ):
            # Act
            save_settings(test_settings)

            # Assert
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0]
            assert "Settings updated" in call_args[0]
            assert "OpenAI" in str(call_args)
            assert "60" in str(call_args)


class TestSettingsPageRendering:
    """Test the complete settings page rendering and interactions."""

    def test_settings_page_displays_title_and_description(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays correct title and description."""
        # Arrange
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert
            mock_streamlit["title"].assert_called_with("Settings")
            mock_streamlit["markdown"].assert_any_call(
                "Configure your AI Job Scraper settings",
            )

    def test_settings_page_displays_llm_provider_selection(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays LLM provider radio selection."""
        # Arrange
        mock_session_state.update({"llm_provider": "Groq"})
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert
            radio_calls = mock_streamlit["radio"].call_args_list
            provider_call = next(
                call for call in radio_calls if "LLM Provider" in call.args
            )

            assert provider_call.args[0] == "LLM Provider"
            assert provider_call.kwargs["options"] == ["OpenAI", "Groq"]
            assert provider_call.kwargs["index"] == 1  # Groq is selected (index 1)
            assert provider_call.kwargs["horizontal"] is True

    def test_settings_page_displays_api_key_inputs(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays API key input fields."""
        # Arrange
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "GROQ_API_KEY": "gsk_test"},
        ):
            # Act
            show_settings_page()

            # Assert
            text_input_calls = mock_streamlit["text_input"].call_args_list

            # OpenAI API Key input
            openai_call = next(
                call for call in text_input_calls if "OpenAI API Key" in call.args
            )
            assert openai_call.kwargs["type"] == "password"
            assert openai_call.kwargs["value"] == "sk-test"
            assert "sk-..." in openai_call.kwargs["placeholder"]

            # Groq API Key input
            groq_call = next(
                call for call in text_input_calls if "Groq API Key" in call.args
            )
            assert groq_call.kwargs["type"] == "password"
            assert groq_call.kwargs["value"] == "gsk_test"
            assert "gsk_..." in groq_call.kwargs["placeholder"]

    def test_settings_page_displays_test_connection_buttons(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays test connection buttons."""
        # Arrange
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert
            button_calls = mock_streamlit["button"].call_args_list

            # Test OpenAI button
            openai_button_call = next(
                call for call in button_calls if call.kwargs.get("key") == "test_openai"
            )
            assert "Test Connection" in openai_button_call.args
            assert openai_button_call.kwargs["disabled"] is True  # No API key

            # Test Groq button
            groq_button_call = next(
                call for call in button_calls if call.kwargs.get("key") == "test_groq"
            )
            assert "Test Connection" in groq_button_call.args
            assert groq_button_call.kwargs["disabled"] is True  # No API key

    def test_settings_page_displays_scraping_configuration(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays scraping configuration section."""
        # Arrange
        mock_session_state.update({"max_jobs_per_company": 75})
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert
            slider_calls = mock_streamlit["slider"].call_args_list

            max_jobs_call = next(
                call for call in slider_calls if "Maximum Jobs Per Company" in call.args
            )
            assert max_jobs_call.kwargs["min_value"] == 10
            assert max_jobs_call.kwargs["max_value"] == 200
            assert max_jobs_call.kwargs["value"] == 75
            assert max_jobs_call.kwargs["step"] == 10

    @pytest.mark.parametrize(
        ("max_jobs", "expected_message_type", "message_keywords"),
        (
            (25, "info", ["Conservative limit", "25 jobs"]),
            (75, "info", ["Moderate limit", "75 jobs"]),
            (150, "warning", ["High limit", "150 jobs", "may take longer"]),
        ),
    )
    def test_settings_page_displays_scraping_limit_feedback(
        self,
        mock_streamlit,
        mock_session_state,
        max_jobs,
        expected_message_type,
        message_keywords,
    ):
        """Test settings page displays appropriate feedback for different job limits."""
        # Arrange
        mock_session_state.update({"max_jobs_per_company": max_jobs})
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert
            message_calls = mock_streamlit[expected_message_type].call_args_list

            # Find the call related to job limits
            limit_message_call = None
            for call in message_calls:
                call_text = call.args[0] if call.args else ""
                if "jobs per company" in call_text:
                    limit_message_call = call
                    break

            assert limit_message_call is not None
            for keyword in message_keywords:
                assert keyword in limit_message_call.args[0]

    def test_settings_page_displays_current_settings_summary(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays current settings summary."""
        # Arrange
        mock_session_state.update({"llm_provider": "Groq", "max_jobs_per_company": 80})
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "GROQ_API_KEY": "gsk_test"},
        ):
            # Act
            show_settings_page()

            # Assert
            markdown_calls = mock_streamlit["markdown"].call_args_list
            markdown_texts = [call.args[0] for call in markdown_calls]

            # Check for summary section
            assert any("Current Settings Summary" in text for text in markdown_texts)
            assert any("⚡ Groq" in text for text in markdown_texts)  # Current provider
            assert any("**80**" in text for text in markdown_texts)  # Max jobs
            assert any("✅ Set" in text for text in markdown_texts)  # API keys status

    def test_settings_page_handles_api_connection_test_success(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page handles successful API connection test."""
        # Arrange
        mock_streamlit["button"].return_value = True  # Simulate button click

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}),
            patch("src.ui.pages.settings.test_api_connection") as mock_test,
        ):
            mock_test.return_value = (True, "✅ Connected successfully")

            # Act
            show_settings_page()

            # Assert
            mock_test.assert_called()
            mock_streamlit["success"].assert_called()

    def test_settings_page_handles_api_connection_test_failure(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page handles failed API connection test."""
        # Arrange
        mock_streamlit["button"].return_value = True  # Simulate button click

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-invalid-key"}),
            patch("src.ui.pages.settings.test_api_connection") as mock_test,
        ):
            mock_test.return_value = (False, "❌ Authentication failed")

            # Act
            show_settings_page()

            # Assert
            mock_test.assert_called()
            mock_streamlit["error"].assert_called()

    def test_settings_page_saves_settings_successfully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page saves settings and displays success message."""
        # Arrange
        # Configure mock inputs to return specific values
        mock_streamlit["radio"].return_value = "OpenAI"
        mock_streamlit["text_input"].side_effect = ["sk-new-key", "gsk_new_key"]
        mock_streamlit["slider"].return_value = 90
        mock_streamlit["button"].return_value = True  # Save button clicked

        with patch("src.ui.pages.settings.save_settings") as mock_save:
            # Act
            show_settings_page()

            # Assert
            mock_save.assert_called_once()
            saved_settings = mock_save.call_args[0][0]
            assert saved_settings["llm_provider"] == "OpenAI"
            assert saved_settings["max_jobs_per_company"] == 90

            # Check success message
            success_calls = mock_streamlit["success"].call_args_list
            success_messages = [call.args[0] for call in success_calls]
            assert any(
                "✅ Settings saved successfully!" in msg for msg in success_messages
            )

    def test_settings_page_handles_save_settings_error(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page handles errors during save operation."""
        # Arrange
        mock_streamlit["radio"].return_value = "OpenAI"
        mock_streamlit["text_input"].side_effect = ["sk-key", "gsk_key"]
        mock_streamlit["slider"].return_value = 60
        mock_streamlit["button"].return_value = True  # Save button clicked

        with patch("src.ui.pages.settings.save_settings") as mock_save:
            mock_save.side_effect = Exception("Database error")

            # Act
            show_settings_page()

            # Assert
            mock_streamlit["error"].assert_called()
            error_calls = mock_streamlit["error"].call_args_list
            error_messages = [call.args[0] for call in error_calls]
            assert any("❌ Failed to save settings" in msg for msg in error_messages)

    def test_settings_page_displays_api_key_security_reminder(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays security reminder for API keys."""
        # Arrange
        mock_streamlit["radio"].return_value = "OpenAI"
        mock_streamlit["text_input"].side_effect = ["sk-key", "gsk_key"]
        mock_streamlit["slider"].return_value = 60
        mock_streamlit["button"].return_value = True  # Save button clicked

        # Act
        show_settings_page()

        # Assert
        info_calls = mock_streamlit["info"].call_args_list
        info_messages = [call.args[0] for call in info_calls]

        # Check for security reminder
        security_reminder = any(
            "API keys should be set as environment variables" in msg
            for msg in info_messages
        )
        assert security_reminder

    def test_settings_page_provider_icons_display_correctly(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page displays correct provider icons."""
        # Arrange
        mock_streamlit["radio"].return_value = "OpenAI"
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert
            markdown_calls = mock_streamlit["markdown"].call_args_list
            markdown_texts = [call.args[0] for call in markdown_calls]

            # Check for provider icons
            assert any("🤖 OpenAI GPT" in text for text in markdown_texts)
            assert any("⚡ Groq (Ultra-fast)" in text for text in markdown_texts)


class TestSettingsPageBoundaryConditions:
    """Test boundary conditions and edge cases for settings page."""

    def test_settings_page_handles_missing_session_state(self, mock_streamlit):
        """Test settings page handles missing session state gracefully."""
        # Arrange
        with (
            patch("streamlit.session_state", new_callable=dict),
            patch.dict(os.environ, {}, clear=True),
        ):
            # Act & Assert - should not raise exception
            show_settings_page()

            # Verify page still renders basic elements
            mock_streamlit["title"].assert_called_with("Settings")

    def test_settings_page_handles_invalid_session_values(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page handles invalid session values gracefully."""
        # Arrange
        mock_session_state.update(
            {
                "llm_provider": "InvalidProvider",  # Invalid provider
                "max_jobs_per_company": "not_a_number",  # Invalid type
            },
        )
        with patch.dict(os.environ, {}, clear=True):
            # Act & Assert - should not raise exception
            show_settings_page()

            # Verify page still renders with defaults
            mock_streamlit["title"].assert_called_with("Settings")

    def test_settings_page_handles_extremely_large_job_limits(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page handles extremely large job limit values."""
        # Arrange
        mock_session_state.update({"max_jobs_per_company": 9999})
        with patch.dict(os.environ, {}, clear=True):
            # Act
            show_settings_page()

            # Assert - Should be clamped to max slider value
            slider_calls = mock_streamlit["slider"].call_args_list
            max_jobs_call = next(
                call for call in slider_calls if "Maximum Jobs Per Company" in call.args
            )
            # Value should be used as-is in slider, but clamped by slider constraints
            assert max_jobs_call.kwargs["max_value"] == 200

    def test_settings_page_environment_variables_override_detection(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test settings page correctly detects environment variable status."""
        # Arrange
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-from-env"},
            clear=True,
        ):
            # Act
            show_settings_page()

            # Assert
            markdown_calls = mock_streamlit["markdown"].call_args_list
            markdown_texts = [call.args[0] for call in markdown_calls]

            # Check environment variable status display
            assert any("OPENAI_API_KEY: ✅ Set" in text for text in markdown_texts)
            assert any("GROQ_API_KEY: ❌ Not Set" in text for text in markdown_texts)


class TestSettingsPageIntegration:
    """Test integration scenarios and complete workflows."""

    def test_complete_settings_configuration_workflow(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test complete workflow of configuring and saving settings."""
        # Arrange - Simulate user interactions
        call_sequence = [
            "Groq",  # Provider selection
            "sk-openai-key-123",  # OpenAI key
            "gsk_groq_key_456",  # Groq key
            100,  # Max jobs slider
            True,  # Save button
        ]
        mock_streamlit["radio"].return_value = call_sequence[0]
        mock_streamlit["text_input"].side_effect = call_sequence[1:3]
        mock_streamlit["slider"].return_value = call_sequence[3]
        mock_streamlit["button"].return_value = call_sequence[4]

        with patch("src.ui.pages.settings.save_settings") as mock_save:
            # Act
            show_settings_page()

            # Assert - Complete workflow executed
            # 1. Form was rendered with inputs
            assert mock_streamlit["radio"].called
            assert mock_streamlit["text_input"].call_count >= 2
            assert mock_streamlit["slider"].called
            assert mock_streamlit["button"].called

            # 2. Settings were saved with correct values
            mock_save.assert_called_once()
            saved_settings = mock_save.call_args[0][0]
            assert saved_settings["llm_provider"] == "Groq"
            assert saved_settings["openai_api_key"] == "sk-openai-key-123"
            assert saved_settings["groq_api_key"] == "gsk_groq_key_456"
            assert saved_settings["max_jobs_per_company"] == 100

            # 3. Success feedback was shown
            mock_streamlit["success"].assert_called()

    def test_api_testing_workflow_with_both_providers(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test API testing workflow for both providers."""
        # Arrange
        mock_streamlit["button"].side_effect = [
            True,
            False,
            False,
            True,
        ]  # Test buttons

        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk-test", "GROQ_API_KEY": "gsk_test"},
            ),
            patch("src.ui.pages.settings.test_api_connection") as mock_test,
        ):
            mock_test.side_effect = [
                (True, "✅ OpenAI connected"),
                (False, "❌ Groq failed"),
            ]

            # Act
            show_settings_page()

            # Assert - Both APIs were tested
            assert mock_test.call_count == 2
            test_calls = mock_test.call_args_list

            # Check OpenAI test
            openai_call = test_calls[0]
            assert openai_call.args[0] == "OpenAI"
            assert openai_call.args[1] == "sk-test"

            # Check Groq test
            groq_call = test_calls[1]
            assert groq_call.args[0] == "Groq"
            assert groq_call.args[1] == "gsk_test"

            # Check feedback was displayed
            mock_streamlit["success"].assert_called()
            mock_streamlit["error"].assert_called()

    def test_settings_persistence_across_page_loads(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test that settings persist correctly across multiple page loads."""
        # Arrange - Set initial session state
        mock_session_state.update(
            {
                "llm_provider": "Groq",
                "max_jobs_per_company": 75,
            },
        )

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-persistent", "GROQ_API_KEY": "gsk_persistent"},
        ):
            # Act - Load page multiple times
            show_settings_page()
            show_settings_page()

            # Assert - Settings are loaded consistently
            radio_calls = mock_streamlit["radio"].call_args_list

            # Check that provider selection uses persisted value
            for call in radio_calls:
                if "LLM Provider" in call.args:
                    assert call.kwargs["index"] == 1  # Groq (index 1)

            # Check that slider uses persisted value
            slider_calls = mock_streamlit["slider"].call_args_list
            for call in slider_calls:
                if "Maximum Jobs Per Company" in call.args:
                    assert call.kwargs["value"] == 75
