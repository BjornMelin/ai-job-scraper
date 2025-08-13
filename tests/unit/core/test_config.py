"""Tests for configuration management."""

import os
import tempfile

from pathlib import Path
from unittest.mock import patch

from src.config import Settings


class TestSettings:
    """Test cases for Settings configuration."""

    def test_default_settings(self):
        """Test default configuration values with empty environment."""
        # Set all environment variables to empty strings to test defaults
        env_overrides = {
            "OPENAI_API_KEY": "",
            "GROQ_API_KEY": "",
            "USE_GROQ": "",
            "PROXY_POOL": "",
            "USE_PROXIES": "",
            "USE_CHECKPOINTING": "",
            "DB_URL": "",
            "EXTRACTION_MODEL": "",
        }

        with (
            patch.dict(os.environ, env_overrides, clear=True),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            # Change to empty temp directory to avoid loading .env
            original_cwd = str(Path.cwd())
            try:
                os.chdir(temp_dir)
                settings = Settings()
            finally:
                os.chdir(original_cwd)

            assert settings.openai_api_key == ""  # Now defaults to empty string
            assert settings.groq_api_key == ""  # Now defaults to empty string
            assert settings.use_groq is False
            assert settings.proxy_pool == []
            assert settings.use_proxies is False
            assert settings.use_checkpointing is False
            assert settings.db_url == "sqlite:///jobs.db"
            assert settings.extraction_model == "gpt-4o-mini"

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        os.environ.update(
            {
                "OPENAI_API_KEY": "env-openai-key",
                "GROQ_API_KEY": "env-groq-key",
                "USE_GROQ": "True",
                "PROXY_POOL": '["proxy1", "proxy2"]',
                "USE_PROXIES": "True",
                "USE_CHECKPOINTING": "True",
                "DB_URL": "sqlite:///test.db",
                "EXTRACTION_MODEL": "gpt-4",
            },
        )

        try:
            settings = Settings()

            assert settings.openai_api_key == "env-openai-key"
            assert settings.groq_api_key == "env-groq-key"
            assert settings.use_groq is True
            assert settings.proxy_pool == ["http://proxy1", "http://proxy2"]
            assert settings.use_proxies is True
            assert settings.use_checkpointing is True
            assert settings.db_url == "sqlite:///test.db"
            assert settings.extraction_model == "gpt-4"
        finally:
            for key in [
                "OPENAI_API_KEY",
                "GROQ_API_KEY",
                "USE_GROQ",
                "PROXY_POOL",
                "USE_PROXIES",
                "USE_CHECKPOINTING",
                "DB_URL",
                "EXTRACTION_MODEL",
            ]:
                os.environ.pop(key, None)

    def test_dotenv_loading(self):
        """Test loading configuration from .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=dotenv-openai\n"
                "GROQ_API_KEY=dotenv-groq\n"
                "USE_GROQ=True\n"
                'PROXY_POOL=["p1", "p2"]\n'
                "USE_PROXIES=True\n"
                "USE_CHECKPOINTING=True\n"
                "DB_URL=sqlite:///dotenv.db\n"
                "EXTRACTION_MODEL=gpt-3.5\n",
            )

            original_cwd = str(Path.cwd())
            try:
                os.chdir(temp_dir)
                settings = Settings()

                assert settings.openai_api_key == "dotenv-openai"
                assert settings.groq_api_key == "dotenv-groq"
                assert settings.use_groq is True
                assert settings.proxy_pool == ["http://p1", "http://p2"]
                assert settings.use_proxies is True
                assert settings.use_checkpointing is True
                assert settings.db_url == "sqlite:///dotenv.db"
                assert settings.extraction_model == "gpt-3.5"
            finally:
                os.chdir(original_cwd)

    def test_env_variables_override_dotenv(self):
        """Test that environment variables take precedence over .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=dotenv-openai\nGROQ_API_KEY=dotenv-groq\n",
            )

            os.environ["OPENAI_API_KEY"] = "env-openai"
            os.environ["GROQ_API_KEY"] = "env-groq"

            original_cwd = str(Path.cwd())
            try:
                os.chdir(temp_dir)
                settings = Settings()

                assert settings.openai_api_key == "env-openai"
                assert settings.groq_api_key == "env-groq"
            finally:
                os.chdir(original_cwd)
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("GROQ_API_KEY", None)

    def test_ignore_empty_env_variables(self):
        """Test that empty environment variables are ignored."""
        os.environ["OPENAI_API_KEY"] = "valid-openai"
        os.environ["GROQ_API_KEY"] = "valid-groq"
        os.environ["DB_URL"] = ""

        try:
            settings = Settings()

            assert settings.openai_api_key == "valid-openai"
            assert settings.groq_api_key == "valid-groq"
            assert settings.db_url == "sqlite:///jobs.db"  # Default
        finally:
            for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "DB_URL"]:
                os.environ.pop(key, None)

    def test_extra_env_variables_ignored(self):
        """Test that extra environment variables are ignored."""
        os.environ["OPENAI_API_KEY"] = "test-openai"
        os.environ["GROQ_API_KEY"] = "test-groq"
        os.environ["UNKNOWN_VAR"] = "ignored"

        try:
            settings = Settings()
            assert settings.openai_api_key == "test-openai"
            assert not hasattr(settings, "unknown_var")
        finally:
            for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "UNKNOWN_VAR"]:
                os.environ.pop(key, None)

    def test_settings_validation_required_fields(self):
        """Test validation behavior - API keys are now optional with empty defaults."""
        # Set API keys to empty strings
        env_overrides = {
            "OPENAI_API_KEY": "",
            "GROQ_API_KEY": "",
        }

        with (
            patch.dict(os.environ, env_overrides, clear=True),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            # Change to empty temp directory to avoid loading .env
            original_cwd = str(Path.cwd())
            try:
                os.chdir(temp_dir)
                settings = Settings()
                assert settings.openai_api_key == ""  # Empty string default
                assert settings.groq_api_key == ""  # Empty string default
            finally:
                os.chdir(original_cwd)

    def test_settings_serialization(self):
        """Test settings serialization and deserialization."""
        os.environ["OPENAI_API_KEY"] = "serialize-openai"
        os.environ["GROQ_API_KEY"] = "serialize-groq"

        try:
            settings = Settings()
            settings_dict = settings.model_dump()

            assert settings_dict["openai_api_key"] == "serialize-openai"
            assert settings_dict["groq_api_key"] == "serialize-groq"

            new_settings = Settings(**settings_dict)
            assert new_settings.openai_api_key == settings.openai_api_key
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("GROQ_API_KEY", None)
