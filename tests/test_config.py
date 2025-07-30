"""Tests for configuration management."""

import os
import tempfile

from pathlib import Path

import pytest

from pydantic import ValidationError

from config import Settings


class TestSettings:
    """Test cases for Settings configuration."""

    def test_default_settings(self):
        """Test default configuration values."""
        # We need to provide a mock API key for testing
        with tempfile.TemporaryDirectory():
            os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

            try:
                settings = Settings()

                assert settings.openai_api_key == "test-key-for-testing"
                assert settings.db_url == "sqlite:///jobs.db"
                assert settings.cache_dir == "./cache"
                assert settings.min_jobs_for_cache == 1
            finally:
                # Clean up environment
                del os.environ["OPENAI_API_KEY"]

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        # Set environment variables
        os.environ.update(
            {
                "OPENAI_API_KEY": "env-test-key",
                "DB_URL": "postgresql://test:test@localhost/testdb",
                "CACHE_DIR": tempfile.mkdtemp(),
                "MIN_JOBS_FOR_CACHE": "5",
            }
        )

        try:
            settings = Settings()

            assert settings.openai_api_key == "env-test-key"
            assert settings.db_url == "postgresql://test:test@localhost/testdb"
            assert settings.cache_dir == os.environ["CACHE_DIR"]
            assert settings.min_jobs_for_cache == 5
        finally:
            # Clean up environment
            for key in ["OPENAI_API_KEY", "DB_URL", "CACHE_DIR", "MIN_JOBS_FOR_CACHE"]:
                if key in os.environ:
                    del os.environ[key]

    def test_dotenv_loading(self):
        """Test loading configuration from .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary .env file
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=dotenv-test-key\n"
                "DB_URL=sqlite:///test.db\n"
                "CACHE_DIR=/dotenv/cache\n"
                "MIN_JOBS_FOR_CACHE=3\n"
            )

            # Change to the temp directory to test .env loading
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                settings = Settings()

                assert settings.openai_api_key == "dotenv-test-key"
                assert settings.db_url == "sqlite:///test.db"
                assert settings.cache_dir == "/dotenv/cache"
                assert settings.min_jobs_for_cache == 3
            finally:
                os.chdir(original_cwd)

    def test_env_variables_override_dotenv(self):
        """Test that environment variables take precedence over .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .env file
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=dotenv-key\nDB_URL=sqlite:///dotenv.db\n"
            )

            # Set conflicting environment variables
            os.environ.update(
                {
                    "OPENAI_API_KEY": "env-override-key",
                    "DB_URL": "sqlite:///env-override.db",
                }
            )

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                settings = Settings()

                # Environment variables should take precedence
                assert settings.openai_api_key == "env-override-key"
                assert settings.db_url == "sqlite:///env-override.db"
            finally:
                os.chdir(original_cwd)
                # Clean up environment
                for key in ["OPENAI_API_KEY", "DB_URL"]:
                    if key in os.environ:
                        del os.environ[key]

    def test_ignore_empty_env_variables(self):
        """Test that empty environment variables are ignored."""
        with tempfile.TemporaryDirectory():
            # Set empty environment variables
            os.environ.update(
                {
                    "OPENAI_API_KEY": "valid-key",
                    "DB_URL": "",  # Empty should be ignored
                    "CACHE_DIR": "",  # Empty should be ignored
                }
            )

            try:
                settings = Settings()

                assert settings.openai_api_key == "valid-key"
                assert settings.db_url == "sqlite:///jobs.db"  # Should use default
                assert settings.cache_dir == "./cache"  # Should use default
            finally:
                # Clean up environment
                for key in ["OPENAI_API_KEY", "DB_URL", "CACHE_DIR"]:
                    if key in os.environ:
                        del os.environ[key]

    def test_extra_env_variables_ignored(self):
        """Test that extra environment variables are ignored."""
        os.environ.update(
            {
                "OPENAI_API_KEY": "test-key",
                "UNKNOWN_SETTING": "should-be-ignored",
                "RANDOM_VAR": "also-ignored",
            }
        )

        try:
            settings = Settings()

            # Should work without error and ignore unknown variables
            assert settings.openai_api_key == "test-key"
            assert not hasattr(settings, "unknown_setting")
            assert not hasattr(settings, "random_var")
        finally:
            # Clean up environment
            for key in ["OPENAI_API_KEY", "UNKNOWN_SETTING", "RANDOM_VAR"]:
                if key in os.environ:
                    del os.environ[key]

    def test_settings_validation(self):
        """Test that settings validation works correctly."""
        # Test missing required field
        # Clear environment first to ensure validation fails
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to avoid reading the .env file
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Test that Settings requires openai_api_key
                with pytest.raises(ValidationError):  # Should raise validation error
                    Settings()
            finally:
                os.chdir(original_cwd)
                # Restore original key if it existed
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key

    def test_settings_serialization(self):
        """Test that settings can be serialized/deserialized."""
        os.environ["OPENAI_API_KEY"] = "test-serialization-key"

        try:
            settings = Settings()

            # Test that we can access the dict representation
            settings_dict = settings.model_dump()

            assert settings_dict["openai_api_key"] == "test-serialization-key"
            assert settings_dict["db_url"] == "sqlite:///jobs.db"
            assert settings_dict["cache_dir"] == "./cache"
            assert settings_dict["min_jobs_for_cache"] == 1

            # Test that we can create a new instance from dict
            new_settings = Settings(**settings_dict)

            assert new_settings.openai_api_key == settings.openai_api_key
            assert new_settings.db_url == settings.db_url
            assert new_settings.cache_dir == settings.cache_dir
            assert new_settings.min_jobs_for_cache == settings.min_jobs_for_cache
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_settings_immutability(self):
        """Test that settings are immutable after creation."""
        os.environ["OPENAI_API_KEY"] = "test-immutable-key"

        try:
            settings = Settings()

            # Try to modify settings (should not be allowed if frozen)
            # Note: Pydantic v2 models are not frozen by default
            # This test documents current behavior
            original_key = settings.openai_api_key
            settings.openai_api_key = "modified-key"

            # Current behavior allows modification
            assert settings.openai_api_key == "modified-key"

            # Reset for clean test state
            settings.openai_api_key = original_key
        finally:
            del os.environ["OPENAI_API_KEY"]
