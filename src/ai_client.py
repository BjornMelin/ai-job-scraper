"""Modern AI Client for AI Job Scraper using LiteLLM and Instructor.

This module provides a unified AI client that leverages:
- LiteLLM for model routing, fallbacks, and provider abstraction
- Instructor for structured output with automatic validation
- Native token counting and context window management
- Simplified configuration via YAML

This replaces the previous custom OpenAI/Groq client management with
a library-first approach that minimizes maintenance and complexity.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, TypeVar

import instructor
import yaml

from litellm import Router, token_counter
from pydantic import BaseModel

from src.config import Settings

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class AIClient:
    """Unified AI client using LiteLLM and Instructor for structured output.

    This client provides:
    - Automatic model routing with fallbacks
    - Structured output with validation
    - Token counting and context window management
    - Retry logic and error handling
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the AI client with LiteLLM configuration.

        Args:
            config_path: Path to LiteLLM YAML configuration file.
                        Defaults to config/litellm.yaml
        """
        self.settings = Settings()

        # Load LiteLLM configuration
        if config_path is None:
            config_path = "config/litellm.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize router with configuration
        self.router = self._create_router()

        # Create instructor client from LiteLLM
        self.instructor_client = instructor.from_litellm(
            completion=self.router.completion,
            mode=instructor.Mode.JSON,
        )

        logger.info(
            "AI Client initialized with %d models", len(self.config["model_list"])
        )

    def _load_config(self) -> dict[str, Any]:
        """Load LiteLLM configuration from YAML file.

        Returns:
            Configuration dictionary for LiteLLM.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        if not self.config_path.exists():
            msg = f"LiteLLM configuration file not found: {self.config_path}"
            raise FileNotFoundError(msg)

        try:
            with self.config_path.open() as f:
                config = yaml.safe_load(f)

            # Validate required configuration sections
            required_sections = ["model_list", "litellm_settings"]
            for section in required_sections:
                if section not in config:
                    msg = f"Missing required configuration section: {section}"
                    raise ValueError(msg)

        except yaml.YAMLError as e:
            msg = f"Invalid YAML configuration: {e}"
            raise yaml.YAMLError(msg) from e
        else:
            return config

    def _create_router(self) -> Router:
        """Create LiteLLM router with loaded configuration.

        Returns:
            Configured LiteLLM Router instance.
        """
        return Router(
            model_list=self.config["model_list"],
            **self.config.get("litellm_settings", {}),
        )

    def get_structured_completion(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        model: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Get structured completion using Instructor for validation.

        Args:
            messages: Chat messages in OpenAI format.
            response_model: Pydantic model for structured response.
            model: Specific model to use (defaults to routing logic).
            **kwargs: Additional parameters for LiteLLM completion.

        Returns:
            Validated instance of response_model.

        Raises:
            Exception: If completion fails after retries.
        """
        # Auto-select model based on context length if not specified
        if model is None:
            model = self._select_model(messages)

        try:
            return self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                **kwargs,
            )
        except Exception:
            logger.exception("Structured completion failed")
            raise

    def get_simple_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Get simple text completion using LiteLLM.

        Args:
            messages: Chat messages in OpenAI format.
            model: Specific model to use (defaults to routing logic).
            **kwargs: Additional parameters for LiteLLM completion.

        Returns:
            Text content from the completion.

        Raises:
            Exception: If completion fails after retries.
        """
        if model is None:
            model = self._select_model(messages)

        try:
            response = self.router.completion(
                model=model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("Simple completion failed")
            raise

    def _select_model(self, messages: list[dict[str, str]]) -> str:
        """Select appropriate model based on context window.

        Args:
            messages: Chat messages for token counting.

        Returns:
            Model name to use for the request.
        """
        try:
            # Count tokens in messages
            token_count = token_counter(messages=messages)

            # Use local model for contexts under 8k tokens
            # Fall back to cloud model for larger contexts
            threshold = 8000  # Based on local model context window

            if token_count < threshold:
                return "local-qwen"
            logger.info(
                "Context size %d exceeds threshold, using cloud model", token_count
            )
            return "gpt-4o-mini"  # noqa: TRY300

        except Exception as e:
            logger.warning("Token counting failed: %s, defaulting to local model", e)
            return "local-qwen"

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens in message list.

        Args:
            messages: Chat messages to count tokens for.

        Returns:
            Number of tokens in the messages.
        """
        try:
            return token_counter(messages=messages)
        except Exception as e:
            logger.warning("Token counting failed: %s", e)
            return 0

    def is_local_available(self) -> bool:
        """Check if local model is available.

        Returns:
            True if local model responds to health check.
        """
        try:
            # Simple health check using a minimal completion
            test_messages = [{"role": "user", "content": "Hello"}]
            self.router.completion(
                model="local-qwen",
                messages=test_messages,
                max_tokens=1,
            )
        except Exception:
            return False
        else:
            return True


class _AIClientSingleton:
    """Singleton holder for AI client instance."""

    def __init__(self) -> None:
        self._instance: AIClient | None = None

    def get_client(self, config_path: str | None = None) -> AIClient:
        """Get singleton AI client instance.

        Args:
            config_path: Path to LiteLLM configuration file.

        Returns:
            Configured AIClient instance.
        """
        if self._instance is None:
            self._instance = AIClient(config_path)
        return self._instance

    def reset(self) -> None:
        """Reset AI client instance (useful for testing)."""
        self._instance = None


# Singleton instance
_singleton = _AIClientSingleton()


def get_ai_client(config_path: str | None = None) -> AIClient:
    """Get singleton AI client instance.

    Args:
        config_path: Path to LiteLLM configuration file.

    Returns:
        Configured AIClient instance.
    """
    return _singleton.get_client(config_path)


def reset_ai_client() -> None:
    """Reset global AI client instance (useful for testing)."""
    _singleton.reset()
