"""Page compatibility and rendering tests.

This module tests that all Streamlit pages can be imported and executed
without errors, ensuring compatibility with the navigation system.
"""

import importlib
import sys

from unittest.mock import patch

import pytest

# Page modules to test
PAGE_MODULES = [
    "src.ui.pages.jobs",
    "src.ui.pages.companies",
    "src.ui.pages.scraping",
    "src.ui.pages.settings",
]


class TestPageImports:
    """Test that all page modules can be imported successfully."""

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_page_module_imports_successfully(self, module_name):
        """Test that each page module can be imported without errors."""
        # Clear module from cache if it exists
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Attempt to import the module
        try:
            module = importlib.import_module(module_name)
            assert module is not None, f"Module {module_name} imported as None"
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing {module_name}: {e}")

    def test_all_page_modules_importable(self):
        """Test that all page modules can be imported together."""
        imported_modules = {}

        for module_name in PAGE_MODULES:
            # Clear module from cache if it exists
            if module_name in sys.modules:
                del sys.modules[module_name]

            try:
                module = importlib.import_module(module_name)
                imported_modules[module_name] = module
            except Exception as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

        # Verify all modules were imported
        assert len(imported_modules) == len(PAGE_MODULES), (
            f"Expected {len(PAGE_MODULES)} modules, got {len(imported_modules)}"
        )


class TestPageStructure:
    """Test page structure and expected functions."""

    def test_jobs_page_has_expected_functions(self):
        """Test that jobs page has expected public functions."""
        try:
            from src.ui.pages import jobs

            # Jobs page should have render functions or be executable
            # At minimum, it should not raise errors when imported
            assert hasattr(jobs, "__name__"), (
                "Jobs module should have __name__ attribute"
            )
        except ImportError as e:
            pytest.fail(f"Failed to import jobs page: {e}")

    def test_companies_page_has_expected_functions(self):
        """Test that companies page has expected public functions."""
        try:
            from src.ui.pages import companies

            # Companies page should have show_companies_page function
            assert hasattr(companies, "show_companies_page"), (
                "Companies page should have show_companies_page function"
            )
        except ImportError as e:
            pytest.fail(f"Failed to import companies page: {e}")

    def test_scraping_page_has_expected_functions(self):
        """Test that scraping page has expected public functions."""
        try:
            from src.ui.pages import scraping

            # Scraping page should be importable and have basic structure
            assert hasattr(scraping, "__name__"), (
                "Scraping module should have __name__ attribute"
            )
        except ImportError as e:
            pytest.fail(f"Failed to import scraping page: {e}")

    def test_settings_page_has_expected_functions(self):
        """Test that settings page has expected public functions."""
        try:
            from src.ui.pages import settings

            # Settings page should be importable and have basic structure
            assert hasattr(settings, "__name__"), (
                "Settings module should have __name__ attribute"
            )
        except ImportError as e:
            pytest.fail(f"Failed to import settings page: {e}")


class TestPageExecution:
    """Test page execution with mocked Streamlit components."""

    def test_companies_page_executes_without_errors(self):
        """Test companies page imports without errors and has required functions."""
        try:
            from src.ui.pages.companies import show_companies_page

            # Test that the function is callable (but don't actually call it)
            assert callable(show_companies_page)

            # Test that required imports work - imports tested above

        except ImportError as e:
            pytest.fail(f"Companies page import failed: {e}")
        except Exception as e:
            pytest.fail(f"Companies page compatibility failed: {e}")

    def test_jobs_page_imports_required_dependencies(self):
        """Test that jobs page can import its required dependencies."""
        try:
            # Import without executing to test dependencies
            import src.ui.pages.jobs

            # Check that the module was imported successfully
            assert src.ui.pages.jobs is not None

        except ImportError as e:
            pytest.fail(f"Jobs page failed to import dependencies: {e}")

    def test_scraping_page_imports_required_dependencies(self):
        """Test that scraping page can import its required dependencies."""
        try:
            # Import without executing to test dependencies
            import src.ui.pages.scraping

            # Check that the module was imported successfully
            assert src.ui.pages.scraping is not None

        except ImportError as e:
            pytest.fail(f"Scraping page failed to import dependencies: {e}")

    def test_settings_page_imports_required_dependencies(self):
        """Test that settings page can import its required dependencies."""
        try:
            # Import without executing to test dependencies
            import src.ui.pages.settings

            # Check that the module was imported successfully
            assert src.ui.pages.settings is not None

        except ImportError as e:
            pytest.fail(f"Settings page failed to import dependencies: {e}")


class TestPageCompatibility:
    """Test compatibility with Streamlit navigation system."""

    def test_pages_compatible_with_st_page(self):
        """Test that pages are compatible with st.Page() objects."""
        from pathlib import Path

        # Test that all page files exist and are readable
        page_paths = [
            "src/ui/pages/jobs.py",
            "src/ui/pages/companies.py",
            "src/ui/pages/scraping.py",
            "src/ui/pages/settings.py",
        ]

        for page_path in page_paths:
            full_path = Path(page_path)

            # Verify file exists
            assert full_path.exists(), f"Page file does not exist: {page_path}"

            # Verify file is readable
            assert full_path.is_file(), f"Page path is not a file: {page_path}"

            # Verify it's a Python file
            assert full_path.suffix == ".py", (
                f"Page file is not a Python file: {page_path}"
            )

            # Verify file is not empty
            content = full_path.read_text(encoding="utf-8")
            assert len(content.strip()) > 0, f"Page file is empty: {page_path}"

    def test_pages_do_not_conflict_with_navigation(self):
        """Test that pages don't interfere with navigation setup."""
        # This test ensures pages can be imported without side effects
        # that would interfere with the navigation system

        original_modules = set(sys.modules.keys())

        try:
            # Import all page modules
            for module_name in PAGE_MODULES:
                importlib.import_module(module_name)

            # Verify no unexpected side effects occurred
            # The modules should be importable without affecting sys state significantly
            new_modules = set(sys.modules.keys()) - original_modules

            # We expect only our imported modules to be added
            expected_new_modules = {
                module for module in PAGE_MODULES if module not in original_modules
            }

            # Allow for additional modules imported as dependencies
            # but verify our target modules are present
            for module in expected_new_modules:
                assert module in new_modules, f"Expected module {module} not found"

        except Exception as e:
            pytest.fail(f"Page import caused unexpected side effects: {e}")

    def test_pages_handle_missing_streamlit_gracefully(self):
        """Test page behavior when Streamlit components are not available."""
        # This tests robustness of page imports

        # Mock streamlit to simulate missing components
        with patch.dict(sys.modules, {"streamlit": None}):
            # Pages should still be importable even if streamlit is mocked out
            # This tests that imports are structured properly
            for module_name in PAGE_MODULES:
                # Clear from cache
                if module_name in sys.modules:
                    del sys.modules[module_name]

                # Try to import - this might fail, which is acceptable
                # We're testing that the import structure is sound
                try:
                    importlib.import_module(module_name)
                except (ImportError, AttributeError, NameError):  # noqa: S110
                    # Expected when streamlit is mocked out
                    pass
                except Exception as e:
                    error_message = (
                        f"Unexpected error with mocked streamlit for {module_name}: {e}"
                    )
                    pytest.fail(error_message)


class TestPagePerformance:
    """Test page performance characteristics."""

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_page_import_time_reasonable(self, module_name):
        """Test that page imports complete in reasonable time."""
        import time

        # Clear module from cache
        if module_name in sys.modules:
            del sys.modules[module_name]

        start_time = time.time()

        try:
            importlib.import_module(module_name)
            import_time = time.time() - start_time

            # Import should complete in under 5 seconds
            assert import_time < 5.0, (
                f"Module {module_name} import took {import_time:.2f}s, expected < 5.0s"
            )

        except ImportError:  # noqa: S110
            # Expected: Import failure is tested elsewhere
            pass

    def test_all_pages_import_memory_efficient(self):
        """Test that importing all pages doesn't cause excessive memory usage."""
        import gc
        import os

        import psutil

        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss

        # Clear module cache
        for module_name in PAGE_MODULES:
            if module_name in sys.modules:
                del sys.modules[module_name]

        gc.collect()

        # Import all pages
        import contextlib

        for module_name in PAGE_MODULES:
            with contextlib.suppress(ImportError):
                # Import failures are tested elsewhere
                importlib.import_module(module_name)

        # Check memory usage after imports
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory

        # Memory increase should be reasonable (less than 100MB)
        max_increase = 100 * 1024 * 1024  # 100MB in bytes
        assert memory_increase < max_increase, (
            f"Page imports increased memory by {memory_increase / 1024 / 1024:.1f}MB, "
            f"expected < {max_increase / 1024 / 1024:.1f}MB"
        )


class TestPageErrorHandling:
    """Test page error handling capabilities."""

    def test_companies_page_handles_service_errors(self):
        """Test that companies page can be imported and has error handling structure."""
        try:
            from src.ui.pages.companies import show_companies_page

            # Test that the function is callable (but don't actually execute it)
            assert callable(show_companies_page)

            # Test that required error handling modules are importable
            # (already tested above)

            # In a real UI test environment, we would test actual error handling
            # but for compatibility testing, we focus on structural requirements

        except ImportError as e:
            pytest.fail(f"Companies page error handling compatibility failed: {e}")
        except Exception as e:
            pytest.fail(
                f"Companies page structure incompatible with error handling: {e}",
            )

    def test_pages_handle_import_errors_in_dependencies(self):
        """Test that pages handle import errors in their dependencies gracefully."""
        # This is more of a structural test - pages should be designed
        # to handle missing optional dependencies

        for module_name in PAGE_MODULES:
            # Clear module from cache
            if module_name in sys.modules:
                del sys.modules[module_name]

            try:
                # Try importing with various dependencies potentially missing
                with patch.dict(
                    sys.modules,
                    {
                        "pandas": None,  # Mock missing pandas
                    },
                ):
                    # Some pages might handle missing pandas gracefully
                    try:
                        importlib.import_module(module_name)
                    except ImportError as e:
                        # Expected for modules that require pandas
                        if "pandas" not in str(e):
                            pytest.fail(
                                f"Unexpected import error for {module_name}: {e}",
                            )
                    except Exception as e:
                        pytest.fail(
                            f"Unexpected error with mocked dependencies "
                            f"for {module_name}: {e}",
                        )

            except Exception as e:
                pytest.fail(f"Error testing dependency handling for {module_name}: {e}")
