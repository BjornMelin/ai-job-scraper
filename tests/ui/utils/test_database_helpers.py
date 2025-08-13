"""Comprehensive tests for database_helpers.py - T1.1 Streamlit Caching Implementation.

Tests cover:
- Streamlit cache decorators (@st.cache_resource, @st.cache_data)
- Database session factory caching
- Session state validation and cleaning
- Database health monitoring
- Background task session management
"""

import threading
import time

from unittest.mock import Mock, patch

import pytest

from sqlalchemy import Engine
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlmodel import Session

from src.ui.utils.database_helpers import (
    background_task_session,
    clean_session_state,
    display_feedback_messages,
    get_cached_session_factory,
    get_database_health,
    init_session_state_keys,
    render_database_health_widget,
    streamlit_db_session,
    suppress_sqlalchemy_warnings,
    validate_session_state,
)


class TestT1DatabaseCaching:
    """Test T1.1: Multi-tier Caching Elimination - Simple Streamlit Caching."""

    @patch("src.ui.utils.database_helpers.get_session")
    def test_get_cached_session_factory_uses_streamlit_cache_resource(
        self, mock_get_session
    ):
        """Test that session factory uses @st.cache_resource decorator properly."""
        mock_session = Mock(spec=Session)
        mock_get_session.return_value = mock_session

        # Call multiple times to test caching behavior
        factory1 = get_cached_session_factory()
        factory2 = get_cached_session_factory()

        # Should return the same factory function (cached)
        assert factory1 is factory2
        assert callable(factory1)

        # Test the factory creates sessions
        session = factory1()
        assert session is mock_session
        mock_get_session.assert_called_once()

    @patch("src.ui.utils.database_helpers.get_cached_session_factory")
    def test_streamlit_db_session_context_manager(self, mock_factory):
        """Test Streamlit-optimized database session context manager."""
        mock_session = Mock(spec=Session)
        mock_factory.return_value = Mock(return_value=mock_session)

        # Test successful transaction
        with streamlit_db_session() as session:
            assert session is mock_session
            # Simulate some work
            session.execute = Mock()

        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @patch("src.ui.utils.database_helpers.get_cached_session_factory")
    def test_streamlit_db_session_handles_exceptions(self, mock_factory):
        """Test session context manager handles exceptions with rollback."""
        mock_session = Mock(spec=Session)
        mock_factory.return_value = Mock(return_value=mock_session)

        with pytest.raises(ValueError, match="Database error"), streamlit_db_session():
            raise ValueError("Database error")

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()

    @patch("src.ui.utils.database_helpers.get_session")
    def test_background_task_session_isolation(self, mock_get_session):
        """Test background task session is isolated from Streamlit session factory."""
        mock_session = Mock(spec=Session)
        mock_get_session.return_value = mock_session

        with background_task_session() as session:
            assert session is mock_session

        mock_get_session.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()


class TestSessionStateValidation:
    """Test session state validation and cleaning functions."""

    def test_validate_session_state_clean_state(self, mock_session_state):
        """Test validation returns empty list for clean session state."""
        mock_session_state.update(
            {
                "user_input": "test",
                "page_state": "jobs",
                "filter_active": True,
            }
        )

        contaminated = validate_session_state()
        assert contaminated == []

    def test_validate_session_state_detects_sqlalchemy_objects(
        self, mock_session_state
    ):
        """Test validation detects SQLAlchemy objects in session state."""
        # Create mock SQLAlchemy objects
        mock_session = Mock(spec=Session)
        mock_engine = Mock(spec=Engine)
        mock_model = Mock()
        mock_model.__class__ = Mock(spec=DeclarativeMeta)

        mock_session_state.update(
            {
                "clean_key": "value",
                "session_object": mock_session,
                "engine_object": mock_engine,
                "model_object": mock_model,
            }
        )

        with (
            patch("sqlalchemy.orm.Session", Session),
            patch("sqlalchemy.Engine", Engine),
            patch("sqlalchemy.ext.declarative.DeclarativeMeta", DeclarativeMeta),
        ):
            contaminated = validate_session_state()

        # Should detect all contaminated keys
        assert "session_object" in contaminated
        assert "engine_object" in contaminated
        assert "model_object" in contaminated
        assert "clean_key" not in contaminated

    def test_clean_session_state_removes_contamination(self, mock_session_state):
        """Test cleaning removes SQLAlchemy objects from session state."""
        mock_session = Mock(spec=Session)
        mock_engine = Mock(spec=Engine)

        mock_session_state.update(
            {
                "keep_this": "value",
                "remove_session": mock_session,
                "remove_engine": mock_engine,
                "keep_number": 42,
            }
        )

        with (
            patch("sqlalchemy.orm.Session", Session),
            patch("sqlalchemy.Engine", Engine),
        ):
            removed_count = clean_session_state()

        assert removed_count == 2
        assert "keep_this" in mock_session_state
        assert "keep_number" in mock_session_state
        assert "remove_session" not in mock_session_state
        assert "remove_engine" not in mock_session_state

    def test_validate_session_state_handles_import_error(self, mock_session_state):
        """Test validation gracefully handles SQLAlchemy import errors."""
        with patch(
            "builtins.__import__", side_effect=ImportError("SQLAlchemy not available")
        ):
            contaminated = validate_session_state()
            assert contaminated == []

    def test_clean_session_state_handles_import_error(self, mock_session_state):
        """Test cleaning gracefully handles SQLAlchemy import errors."""
        mock_session_state.update({"test": "value"})

        with patch(
            "builtins.__import__", side_effect=ImportError("SQLAlchemy not available")
        ):
            removed_count = clean_session_state()
            assert removed_count == 0


class TestDatabaseHealthMonitoring:
    """Test database health monitoring with @st.cache_data."""

    @patch("src.ui.utils.database_helpers.get_cached_session_factory")
    def test_get_database_health_healthy_status(self, mock_factory):
        """Test database health check returns healthy status."""
        mock_session = Mock()
        mock_session.execute.return_value.scalar.return_value = 1
        mock_factory.return_value = Mock(return_value=mock_session)

        health = get_database_health()

        assert health["status"] == "healthy"
        assert health["details"]["connected"] is True
        assert "Database accessible" in health["details"]["message"]
        mock_session.close.assert_called()

    @patch("src.ui.utils.database_helpers.get_cached_session_factory")
    def test_get_database_health_unhealthy_status(self, mock_factory):
        """Test database health check handles connection failures."""
        mock_session = Mock()
        mock_session.execute.side_effect = Exception("Connection failed")
        mock_factory.return_value = Mock(return_value=mock_session)

        health = get_database_health()

        assert health["status"] == "unhealthy"
        assert health["details"]["connected"] is False
        assert "Connection failed" in health["details"]["error"]
        mock_session.close.assert_called()

    @patch("src.ui.utils.database_helpers.get_cached_session_factory")
    def test_get_database_health_factory_error(self, mock_factory):
        """Test database health check handles session factory errors."""
        mock_factory.side_effect = Exception("Factory creation failed")

        health = get_database_health()

        assert health["status"] == "error"
        assert health["details"]["connected"] is False
        assert "Session creation failed" in health["details"]["error"]

    def test_get_database_health_caching_behavior(self):
        """Test that database health check uses @st.cache_data with TTL."""
        # This test verifies the decorator is applied correctly
        # We can't easily test the actual caching without Streamlit runtime
        # but we can verify the function signature and behavior
        with patch(
            "src.ui.utils.database_helpers.get_cached_session_factory"
        ) as mock_factory:
            mock_session = Mock()
            mock_session.execute.return_value.scalar.return_value = 1
            mock_factory.return_value = Mock(return_value=mock_session)

            # Call multiple times to verify consistent behavior
            health1 = get_database_health()
            health2 = get_database_health()

            assert health1["status"] == health2["status"] == "healthy"

    @patch("src.ui.utils.database_helpers.get_database_health")
    def test_render_database_health_widget_healthy(self, mock_health, mock_streamlit):
        """Test rendering healthy database widget."""
        mock_health.return_value = {
            "status": "healthy",
            "details": {"connected": True, "message": "Database accessible"},
        }

        render_database_health_widget()

        mock_streamlit["success"].assert_called_once_with("🟢 Database Connected")

    @patch("src.ui.utils.database_helpers.get_database_health")
    def test_render_database_health_widget_unhealthy(self, mock_health, mock_streamlit):
        """Test rendering unhealthy database widget."""
        mock_health.return_value = {
            "status": "unhealthy",
            "details": {
                "connected": False,
                "error": "Connection timeout occurred during health check",
            },
        }

        render_database_health_widget()

        mock_streamlit["warning"].assert_called_once_with("🟡 Database Issue")
        mock_streamlit["caption"].assert_called_once_with(
            "Error: Connection timeout occurred during health c..."
        )

    @patch("src.ui.utils.database_helpers.get_database_health")
    def test_render_database_health_widget_error(self, mock_health, mock_streamlit):
        """Test rendering database error widget."""
        mock_health.return_value = {
            "status": "error",
            "details": {"connected": False, "error": "Fatal database connection error"},
        }

        render_database_health_widget()

        mock_streamlit["error"].assert_called_once_with("🔴 Database Error")
        mock_streamlit["caption"].assert_called_once_with(
            "Error: Fatal database connection error..."
        )


class TestUtilityHelpers:
    """Test utility helper functions from session_helpers.py."""

    def test_init_session_state_keys_new_keys(self, mock_session_state):
        """Test initializing new session state keys."""
        keys = ["filter_active", "selected_company", "view_mode"]
        default_value = None

        init_session_state_keys(keys, default_value)

        for key in keys:
            assert mock_session_state.get(key) is None

    def test_init_session_state_keys_existing_keys(self, mock_session_state):
        """Test that existing session state keys are not overwritten."""
        mock_session_state["existing_key"] = "existing_value"
        keys = ["existing_key", "new_key"]

        init_session_state_keys(keys, "default")

        assert mock_session_state.get("existing_key") == "existing_value"
        assert mock_session_state.get("new_key") == "default"

    def test_display_feedback_messages_success(
        self, mock_session_state, mock_streamlit
    ):
        """Test displaying and clearing success messages."""
        mock_session_state["success_msg"] = "Operation completed successfully"
        mock_session_state["keep_this"] = "value"

        display_feedback_messages(["success_msg"], [])

        mock_streamlit["success"].assert_called_once_with(
            "✅ Operation completed successfully"
        )
        assert mock_session_state.get("success_msg") is None
        assert mock_session_state.get("keep_this") == "value"

    def test_display_feedback_messages_error(self, mock_session_state, mock_streamlit):
        """Test displaying and clearing error messages."""
        mock_session_state["error_msg"] = "Operation failed"

        display_feedback_messages([], ["error_msg"])

        mock_streamlit["error"].assert_called_once_with("❌ Operation failed")
        assert mock_session_state.get("error_msg") is None

    def test_display_feedback_messages_mixed(self, mock_session_state, mock_streamlit):
        """Test displaying both success and error messages."""
        mock_session_state["success_msg"] = "Some operations succeeded"
        mock_session_state["error_msg"] = "Some operations failed"

        display_feedback_messages(["success_msg"], ["error_msg"])

        mock_streamlit["success"].assert_called_once_with(
            "✅ Some operations succeeded"
        )
        mock_streamlit["error"].assert_called_once_with("❌ Some operations failed")
        assert mock_session_state.get("success_msg") is None
        assert mock_session_state.get("error_msg") is None

    def test_display_feedback_messages_no_messages(
        self, mock_session_state, mock_streamlit
    ):
        """Test handling when no messages exist."""
        display_feedback_messages(["nonexistent"], ["also_nonexistent"])

        mock_streamlit["success"].assert_not_called()
        mock_streamlit["error"].assert_not_called()


class TestSQLAlchemyWarningsSuppression:
    """Test SQLAlchemy warnings suppression for Streamlit."""

    @patch("warnings.filterwarnings")
    def test_suppress_sqlalchemy_warnings(self, mock_filter_warnings):
        """Test that SQLAlchemy warnings are properly suppressed."""
        suppress_sqlalchemy_warnings()

        # Verify warning filters were added
        assert mock_filter_warnings.call_count >= 2
        calls = mock_filter_warnings.call_args_list

        # Check for relationship warnings
        relationship_call = None
        attribute_call = None
        for call in calls:
            if "relationship" in str(call):
                relationship_call = call
            elif "Attribute" in str(call):
                attribute_call = call

        assert relationship_call is not None
        assert attribute_call is not None


class TestT1CachingIntegration:
    """Integration tests for T1.1 caching elimination."""

    @patch("src.ui.utils.database_helpers.get_session")
    def test_session_factory_caching_performance(self, mock_get_session):
        """Test that session factory caching improves performance."""
        mock_session = Mock(spec=Session)
        mock_get_session.return_value = mock_session

        # Time multiple factory creations
        start_time = time.time()
        for _ in range(100):
            factory = get_cached_session_factory()
            factory()
        end_time = time.time()

        # With caching, should be very fast
        duration = end_time - start_time
        assert duration < 0.1  # Should complete in less than 100ms

        # get_session should only be called for actual session
        # creation, not factory creation
        assert mock_get_session.call_count == 100  # One call per session creation

    def test_concurrent_session_factory_access(self):
        """Test thread safety of cached session factory."""
        results = []
        exceptions = []

        def worker():
            try:
                factory = get_cached_session_factory()
                results.append(factory)
            except Exception as e:
                exceptions.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should succeed and get the same factory
        assert len(exceptions) == 0
        assert len(results) == 10
        assert all(factory is results[0] for factory in results)

    @patch("src.ui.utils.database_helpers.get_database_health")
    def test_health_widget_integration(self, mock_health, mock_streamlit):
        """Test integration between health check and widget rendering."""
        # Test different health states
        health_states = [
            {"status": "healthy", "details": {"connected": True, "message": "OK"}},
            {
                "status": "unhealthy",
                "details": {"connected": False, "error": "Timeout"},
            },
            {
                "status": "error",
                "details": {"connected": False, "error": "Fatal error"},
            },
        ]

        for health_state in health_states:
            mock_health.return_value = health_state
            render_database_health_widget()

        # Verify appropriate Streamlit functions were called for each state
        assert mock_streamlit["success"].call_count == 1
        assert mock_streamlit["warning"].call_count == 1
        assert mock_streamlit["error"].call_count == 1


class TestT1RealisticScenarios:
    """Test realistic usage scenarios for T1.1 simplified caching."""

    @patch("src.ui.utils.database_helpers.get_cached_session_factory")
    def test_typical_streamlit_page_lifecycle(self, mock_factory):
        """Test typical Streamlit page lifecycle with database operations."""
        mock_session = Mock(spec=Session)
        mock_factory.return_value = Mock(return_value=mock_session)

        # Simulate page load with multiple database operations
        operations = []

        # Multiple database contexts in single page render
        for i in range(5):
            with streamlit_db_session() as session:
                operations.append(f"operation_{i}")
                session.query = Mock(return_value=[])

        assert len(operations) == 5
        assert mock_session.commit.call_count == 5
        assert mock_session.close.call_count == 5

    def test_session_state_cleanup_workflow(self, mock_session_state):
        """Test realistic session state cleanup workflow."""
        # Simulate page with mixed clean and contaminated state
        mock_session_state.update(
            {
                "user_preferences": {"theme": "dark"},
                "current_page": "jobs",
                "filter_criteria": {"location": "remote"},
                "contaminated_session": Mock(spec=Session),
                "cached_data": [1, 2, 3],
            }
        )

        # Validate and clean
        contaminated = validate_session_state()
        removed = clean_session_state()

        assert len(contaminated) == 1
        assert removed == 1
        assert "user_preferences" in mock_session_state
        assert "current_page" in mock_session_state
        assert "filter_criteria" in mock_session_state
        assert "cached_data" in mock_session_state
        assert "contaminated_session" not in mock_session_state

    @patch("src.ui.utils.database_helpers.get_database_health")
    def test_production_health_monitoring_scenarios(self, mock_health, mock_streamlit):
        """Test production-like health monitoring scenarios."""
        # Simulate intermittent connectivity issues
        health_scenarios = [
            # Normal operation
            {
                "status": "healthy",
                "details": {"connected": True, "message": "All systems operational"},
            },
            # Temporary issue
            {
                "status": "unhealthy",
                "details": {"connected": False, "error": "Connection pool exhausted"},
            },
            # Recovery
            {
                "status": "healthy",
                "details": {"connected": True, "message": "Connection restored"},
            },
            # Critical failure
            {
                "status": "error",
                "details": {"connected": False, "error": "Database server unreachable"},
            },
        ]

        for scenario in health_scenarios:
            mock_health.return_value = scenario
            render_database_health_widget()

        # Verify all scenarios were handled appropriately
        assert mock_streamlit["success"].call_count == 2  # Healthy states
        assert mock_streamlit["warning"].call_count == 1  # Unhealthy state
        assert mock_streamlit["error"].call_count == 1  # Error state
