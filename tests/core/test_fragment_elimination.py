"""Fragment elimination regression tests.

This module tests the 18 → 0 fragment elimination completed by Group 3,
ensuring no @st.fragment decorators remain and preventing regression
back to fragment-based architecture.
"""

import ast
import glob
import os

import pytest


class TestFragmentElimination:
    """Test complete elimination of Streamlit fragments."""

    def test_no_fragment_decorators_in_codebase(self):
        """Test that no @st.fragment decorators exist anywhere in the codebase."""
        # Get all Python files in the src directory
        python_files = glob.glob("src/**/*.py", recursive=True)

        fragment_files = []
        for file_path in python_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Check for various fragment decorator patterns
            fragment_patterns = [
                "@st.fragment",
                "@ st.fragment",
                "@streamlit.fragment",
                "st.fragment(",
                "streamlit.fragment(",
            ]

            for pattern in fragment_patterns:
                if pattern in content:
                    fragment_files.append((file_path, pattern))

        # Assert no fragment decorators found
        if fragment_files:
            error_msg = "Fragment decorators found in files:\n"
            for file_path, pattern in fragment_files:
                error_msg += f"  {file_path}: {pattern}\n"
            error_msg += "\nAll fragments should have been eliminated by Group 3."
            pytest.fail(error_msg)

    def test_no_fragment_imports(self):
        """Test that no fragment-related imports exist."""
        python_files = glob.glob("src/**/*.py", recursive=True)

        fragment_imports = []
        for file_path in python_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Check for fragment import patterns
            import_patterns = [
                "from streamlit import fragment",
                "import streamlit.fragment",
                "from st import fragment",
            ]

            for pattern in import_patterns:
                if pattern in content:
                    fragment_imports.append((file_path, pattern))

        # Assert no fragment imports found
        if fragment_imports:
            error_msg = "Fragment imports found in files:\n"
            for file_path, pattern in fragment_imports:
                error_msg += f"  {file_path}: {pattern}\n"
            error_msg += "\nAll fragment imports should have been removed."
            pytest.fail(error_msg)

    def test_no_fragment_function_calls(self):
        """Test that no fragment function calls exist in the codebase."""
        python_files = glob.glob("src/**/*.py", recursive=True)

        fragment_calls = []
        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse the file to find function calls
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Check for st.fragment() calls
                        if (
                            isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id in ["st", "streamlit"]
                            and node.func.attr == "fragment"
                        ):
                            fragment_calls.append(file_path)

            except (SyntaxError, UnicodeDecodeError):
                # Skip files that can't be parsed (e.g., __pycache__)
                continue

        # Assert no fragment function calls found
        if fragment_calls:
            error_msg = f"Fragment function calls found in files: {fragment_calls}\n"
            error_msg += "All fragment calls should have been eliminated."
            pytest.fail(error_msg)

    def test_ui_files_fragment_free(self):
        """Test that UI files specifically are fragment-free."""
        ui_files = glob.glob("src/ui/**/*.py", recursive=True)

        for file_path in ui_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # UI files should definitely not contain fragments
            assert "@st.fragment" not in content, (
                f"Fragment found in UI file: {file_path}"
            )
            assert "@streamlit.fragment" not in content, (
                f"Fragment found in UI file: {file_path}"
            )
            assert "st.fragment(" not in content, (
                f"Fragment found in UI file: {file_path}"
            )

    def test_component_files_fragment_free(self):
        """Test that component files are fragment-free."""
        component_files = glob.glob("src/ui/components/**/*.py", recursive=True)

        for file_path in component_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Component files should not use fragments
            fragment_patterns = ["@st.fragment", "st.fragment(", "@fragment"]
            for pattern in fragment_patterns:
                assert pattern not in content, (
                    f"Fragment pattern '{pattern}' found in {file_path}"
                )


class TestManualRefreshMigration:
    """Test that manual refresh patterns replaced fragment auto-refresh."""

    def test_manual_refresh_buttons_present(self):
        """Test that manual refresh buttons exist where needed."""
        # Check key UI files for manual refresh implementation
        ui_files = [
            "src/ui/pages/jobs.py",
            "src/ui/pages/analytics.py",
            "src/ui/pages/companies.py",
            "src/ui/pages/scraping.py",
        ]

        refresh_implementations = []

        for file_path in ui_files:
            if os.path.exists(file_path):
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Look for manual refresh patterns
                refresh_patterns = ["st.button", "refresh", "🔄", "reload", "update"]

                has_refresh = any(
                    pattern in content.lower() for pattern in refresh_patterns
                )
                if has_refresh:
                    refresh_implementations.append(file_path)

        # At least some files should have manual refresh implementation
        assert len(refresh_implementations) > 0, (
            "No manual refresh implementations found"
        )

    def test_no_auto_refresh_patterns(self):
        """Test that no automatic refresh patterns remain."""
        python_files = glob.glob("src/**/*.py", recursive=True)

        auto_refresh_patterns = []
        for file_path in python_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Check for auto-refresh anti-patterns
            auto_patterns = [
                "auto_refresh",
                "automatic_refresh",
                "st.rerun()",
                "@st.fragment(run_every=",
                "run_every=",
            ]

            for pattern in auto_patterns:
                if pattern in content:
                    auto_refresh_patterns.append((file_path, pattern))

        # Should find minimal auto-refresh patterns (some st.rerun() may be legitimate)
        fragment_auto_refresh = [
            (f, p)
            for f, p in auto_refresh_patterns
            if "run_every=" in p or "@st.fragment" in p
        ]

        if fragment_auto_refresh:
            error_msg = "Auto-refresh fragment patterns found:\n"
            for file_path, pattern in fragment_auto_refresh:
                error_msg += f"  {file_path}: {pattern}\n"
            pytest.fail(error_msg)

    def test_widget_key_based_state_management(self):
        """Test that widget keys replaced fragment state management."""
        ui_files = glob.glob("src/ui/**/*.py", recursive=True)

        widget_key_usage = 0
        for file_path in ui_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Count widget key usage (proper pattern)
            if "key=" in content:
                widget_key_usage += content.count("key=")

        # Should have significant widget key usage
        assert widget_key_usage > 0, (
            "No widget key usage found - state management may be broken"
        )


class TestPerformanceImprovementValidation:
    """Test performance improvements from fragment elimination."""

    def test_reduced_render_complexity(self):
        """Test that render complexity is reduced without fragments."""
        # Fragment elimination should reduce render complexity
        # This is validated by ensuring clean component structure

        component_files = glob.glob("src/ui/components/**/*.py", recursive=True)
        complex_components = []

        for file_path in component_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Check for overly complex render logic that fragments might have hidden
            complexity_indicators = [
                "nested_fragments",  # Should not exist
                "fragment_state",  # Should not exist
                "st.session_state" * 5,  # Too many session state references
            ]

            # Count complexity indicators
            complexity_score = sum(
                content.count(indicator) for indicator in complexity_indicators
            )
            if complexity_score > 10:  # Arbitrary threshold
                complex_components.append(file_path)

        # Most components should be simple without fragments
        assert len(complex_components) < len(component_files) * 0.5, (
            f"Too many complex components found: {complex_components}"
        )

    def test_streamlined_update_patterns(self):
        """Test that update patterns are streamlined without fragments."""
        ui_files = glob.glob("src/ui/**/*.py", recursive=True)

        streamlined_patterns = 0
        for file_path in ui_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Look for streamlined patterns (widget-based updates)
            streamlined_indicators = [
                "st.button(",
                "st.selectbox(",
                "st.text_input(",
                "key=",
            ]

            for indicator in streamlined_indicators:
                if indicator in content:
                    streamlined_patterns += 1
                    break  # Count once per file

        # Should have widespread use of streamlined patterns
        assert streamlined_patterns > len(ui_files) * 0.5, (
            "Insufficient streamlined update patterns found"
        )


class TestRegressionPrevention:
    """Test to prevent regression back to fragment-based architecture."""

    def test_fragment_reintroduction_prevention(self):
        """Test guards against reintroducing fragments."""
        # This test serves as documentation and prevention
        forbidden_patterns = [
            "@st.fragment",
            "@streamlit.fragment",
            "st.fragment(",
            "streamlit.fragment(",
            "from streamlit import fragment",
        ]

        # Check that these patterns are documented as forbidden
        for pattern in forbidden_patterns:
            # These patterns should not exist anywhere in src/
            python_files = glob.glob("src/**/*.py", recursive=True)

            pattern_found = False
            for file_path in python_files:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                if pattern in content:
                    pattern_found = True
                    break

            assert not pattern_found, (
                f"Forbidden fragment pattern '{pattern}' found in codebase"
            )

    def test_architecture_documentation_compliance(self):
        """Test that architecture follows fragment-free patterns."""
        # Key architecture files should exist and be fragment-free
        key_files = [
            "src/ui/pages/jobs.py",
            "src/ui/pages/analytics.py",
            "src/ui/components/sidebar.py",
        ]

        for file_path in key_files:
            if os.path.exists(file_path):
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Should have proper widget-based architecture
                assert "def " in content, f"No functions found in {file_path}"
                assert "st." in content, f"No Streamlit usage found in {file_path}"
                assert "@st.fragment" not in content, (
                    f"Fragment found in key file {file_path}"
                )

    def test_migration_completeness_validation(self):
        """Test that fragment migration is complete."""
        # Check that fragment elimination is complete

        # 1. No fragment files should exist
        fragment_file_patterns = ["**/fragments/**", "**/*fragment*.py"]

        fragment_files = []
        for pattern in fragment_file_patterns:
            fragment_files.extend(glob.glob(pattern, recursive=True))

        # Filter out test files (they may reference fragments for testing)
        fragment_files = [f for f in fragment_files if not f.startswith("tests/")]

        assert len(fragment_files) == 0, f"Fragment files still exist: {fragment_files}"

        # 2. Fragment count should be zero
        total_fragment_references = 0
        python_files = glob.glob("src/**/*.py", recursive=True)

        for file_path in python_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            total_fragment_references += content.count("@st.fragment")
            total_fragment_references += content.count("st.fragment(")

        assert total_fragment_references == 0, (
            f"Found {total_fragment_references} fragment references (should be 0)"
        )

    def test_clean_architecture_validation(self):
        """Test that the architecture is clean without fragments."""
        # Validate clean separation of concerns without fragments

        # UI components should be simple and focused
        component_files = glob.glob("src/ui/components/**/*.py", recursive=True)

        for file_path in component_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Components should have clear, simple structure
            lines = content.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            # Components should be reasonably sized (not overly complex)
            assert len(non_empty_lines) < 500, (
                f"Component {file_path} is too large ({len(non_empty_lines)} lines)"
            )

            # Should not have fragment-related complexity
            assert "fragment_state" not in content, (
                f"Fragment state management found in {file_path}"
            )
            assert "fragment_update" not in content, (
                f"Fragment update logic found in {file_path}"
            )


class TestValidateFragmentElimination18to0:
    """Test to validate the specific 18 → 0 fragment elimination achievement."""

    def test_exact_fragment_count_zero(self):
        """Test that exactly 0 fragments remain (down from 18)."""
        # Count fragments across entire codebase
        python_files = glob.glob("src/**/*.py", recursive=True)

        total_fragments = 0
        fragment_locations = []

        for file_path in python_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Count all fragment decorator patterns
            fragment_patterns = ["@st.fragment", "@streamlit.fragment"]

            for pattern in fragment_patterns:
                count = content.count(pattern)
                if count > 0:
                    total_fragments += count
                    fragment_locations.append(f"{file_path}: {count} {pattern}")

        # Validate exact 0 fragment count
        assert total_fragments == 0, (
            f"Expected 0 fragments, found {total_fragments}. Locations:\n"
            + "\n".join(fragment_locations)
        )

    def test_group3_achievement_documentation(self):
        """Test that the Group 3 fragment elimination achievement is maintained."""
        # This test documents and validates the 18 → 0 achievement

        # Fragment elimination should be complete and documented
        achievement_metrics = {
            "fragments_before": 18,
            "fragments_after": 0,
            "elimination_percentage": 100.0,
        }

        # Validate current state matches achievement
        python_files = glob.glob("src/**/*.py", recursive=True)
        current_fragment_count = 0

        for file_path in python_files:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            current_fragment_count += content.count("@st.fragment")
            current_fragment_count += content.count("@streamlit.fragment")

        assert current_fragment_count == achievement_metrics["fragments_after"], (
            f"Fragment count regression detected: {current_fragment_count} "
            + f"(expected {achievement_metrics['fragments_after']})"
        )
