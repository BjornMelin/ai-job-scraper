"""Shared test utilities for UI component testing.

This module provides common test utilities, mocks, and fixtures that are shared
across multiple test modules to reduce code duplication and improve maintainability.
"""

from typing import Any


class MockSessionState:
    """Mock Streamlit session state that supports both dict and attribute access.

    This is a shared utility class for testing Streamlit components that require
    session state functionality. It provides a complete implementation that mimics
    the behavior of streamlit.session_state.
    """

    def __init__(self, initial_data: dict[str, Any] | None = None) -> None:
        """Initialize mock session state with optional initial data.

        Args:
            initial_data: Optional dictionary of initial session state data.
        """
        self._data = initial_data or {}

    def __getitem__(self, key: str) -> Any:
        """Get item from data dict.

        Args:
            key: The key to retrieve from the session state.

        Returns:
            The value associated with the given key.

        Raises:
            KeyError: If the key doesn't exist in the session state.
        """
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in data dict.

        Args:
            key: The key to set in the session state.
            value: The value to associate with the key.
        """
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in data dict.

        Args:
            key: The key to check for existence.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        """Get attribute from data dict.

        Args:
            name: The attribute name to retrieve.

        Returns:
            The value of the attribute, or None if not found.
        """
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in data dict.

        If the name starts with an underscore, sets it as a class attribute.
        Otherwise, sets it in the internal data dictionary.

        Args:
            name: The attribute name to set.
            value: The value to associate with the attribute.
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete an attribute from the session state.

        Args:
            name: The attribute name to delete.

        Raises:
            AttributeError: If the attribute doesn't exist.
        """
        if name.startswith("_"):
            super().__delattr__(name)
        elif name in self._data:
            del self._data[name]
        else:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from data dict with optional default.

        Args:
            key: The key to retrieve.
            default: The default value to return if the key doesn't exist.

        Returns:
            The value associated with the key, or the default value.
        """
        return self._data.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Get value from data dict with optional default, setting it if missing.

        Args:
            key: The key to retrieve/set.
            default: The default value to set and return if the key doesn't exist.

        Returns:
            The existing value or the newly set default value.
        """
        return self._data.setdefault(key, default)

    def update(self, other: dict[str, Any]) -> None:
        """Update internal data dictionary with values from another dict.

        Args:
            other: Dictionary containing values to update.
        """
        self._data.update(other)

    def clear(self) -> None:
        """Clear all data from the session state."""
        self._data.clear()

    def keys(self):
        """Get all keys in the session state."""
        return self._data.keys()

    def values(self):
        """Get all values in the session state."""
        return self._data.values()

    def items(self):
        """Get all key-value pairs in the session state."""
        return self._data.items()
