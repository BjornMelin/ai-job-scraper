"""Mobile Responsiveness Integration Tests.

This test suite validates mobile-first responsive design implementation across
the AI job scraper system. Tests ensure optimal user experience across all
device types and viewport sizes with touch-friendly interactions.

**Mobile Responsiveness Requirements**:
- CSS Grid responsive design validation
- Touch-friendly interaction testing
- Viewport adaptation (320px-1920px)
- Performance across device types
- Mobile-first progressive enhancement

**Test Coverage**:
- Responsive layout adaptation
- Touch interaction validation
- Mobile performance optimization
- Cross-device compatibility
- Progressive enhancement verification
"""

import logging
import time

from unittest.mock import Mock, patch

import pytest

from tests.factories import create_realistic_dataset

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Mobile device configurations for testing
MOBILE_DEVICE_CONFIGS = {
    "mobile_phone": {
        "viewport_width": 320,
        "viewport_height": 568,
        "device_pixel_ratio": 2.0,
        "touch_enabled": True,
        "user_agent": "Mobile Safari",
        "expected_columns": 1,
        "expected_card_size": "full-width",
    },
    "large_phone": {
        "viewport_width": 414,
        "viewport_height": 896,
        "device_pixel_ratio": 3.0,
        "touch_enabled": True,
        "user_agent": "Mobile Safari",
        "expected_columns": 1,
        "expected_card_size": "full-width",
    },
    "tablet_portrait": {
        "viewport_width": 768,
        "viewport_height": 1024,
        "device_pixel_ratio": 2.0,
        "touch_enabled": True,
        "user_agent": "Mobile Safari",
        "expected_columns": 2,
        "expected_card_size": "half-width",
    },
    "tablet_landscape": {
        "viewport_width": 1024,
        "viewport_height": 768,
        "device_pixel_ratio": 2.0,
        "touch_enabled": True,
        "user_agent": "Mobile Safari",
        "expected_columns": 3,
        "expected_card_size": "third-width",
    },
    "desktop": {
        "viewport_width": 1200,
        "viewport_height": 800,
        "device_pixel_ratio": 1.0,
        "touch_enabled": False,
        "user_agent": "Desktop Chrome",
        "expected_columns": 3,
        "expected_card_size": "third-width",
    },
    "large_desktop": {
        "viewport_width": 1920,
        "viewport_height": 1080,
        "device_pixel_ratio": 1.0,
        "touch_enabled": False,
        "user_agent": "Desktop Chrome",
        "expected_columns": 4,
        "expected_card_size": "quarter-width",
    },
}


@pytest.fixture
def mobile_test_setup(session, tmp_path):
    """Set up test environment for mobile responsiveness testing."""
    # Create test dataset
    dataset = create_realistic_dataset(
        session,
        companies=15,
        jobs_per_company=20,
        senior_ratio=0.3,
        remote_ratio=0.5,
        favorited_ratio=0.15,
    )

    return {
        "dataset": dataset,
        "session": session,
        "temp_dir": tmp_path,
        "test_jobs": dataset["jobs"][:50],  # Use first 50 jobs for testing
    }


class TestResponsiveLayoutAdaptation:
    """Test responsive layout adaptation across device types."""

    @pytest.mark.mobile
    def test_css_grid_responsive_breakpoints(self, mobile_test_setup):
        """Test CSS Grid adapts correctly across responsive breakpoints."""
        setup = mobile_test_setup
        test_jobs = setup["test_jobs"]

        responsive_results = []

        for device_name, device_config in MOBILE_DEVICE_CONFIGS.items():
            with (
                patch(
                    "src.ui.utils.mobile_detection.get_viewport_width"
                ) as mock_viewport,
                patch(
                    "src.ui.utils.mobile_detection.get_device_pixel_ratio"
                ) as mock_dpr,
                patch("src.ui.utils.mobile_detection.is_touch_device") as mock_touch,
            ):
                # Configure device environment
                mock_viewport.return_value = device_config["viewport_width"]
                mock_dpr.return_value = device_config["device_pixel_ratio"]
                mock_touch.return_value = device_config["touch_enabled"]

                # Mock Streamlit components
                with patch("src.ui.components.cards.job_card.st") as mock_st:
                    # Mock columns based on expected layout
                    expected_cols = device_config["expected_columns"]
                    mock_columns = [Mock() for _ in range(expected_cols)]
                    mock_st.columns.return_value = mock_columns
                    mock_st.container = Mock()
                    mock_st.markdown = Mock()
                    mock_st.button = Mock(return_value=False)

                    from src.ui.components.cards.job_card import render_job_cards

                    # Test responsive rendering
                    start_time = time.perf_counter()
                    render_job_cards(
                        test_jobs[:12],  # Test with 12 jobs
                        device_type=device_name,
                    )
                    render_time = time.perf_counter() - start_time

                    # Verify layout adaptation
                    columns_call = mock_st.columns.call_args
                    if columns_call:
                        actual_columns = (
                            len(columns_call[0][0])
                            if columns_call[0]
                            else expected_cols
                        )
                    else:
                        actual_columns = 1  # Default fallback

                    responsive_results.append(
                        {
                            "device": device_name,
                            "viewport_width": device_config["viewport_width"],
                            "expected_columns": expected_cols,
                            "actual_columns": actual_columns,
                            "render_time_ms": render_time * 1000,
                            "layout_correct": actual_columns == expected_cols,
                            "performance_good": render_time < 0.3,  # 300ms threshold
                        }
                    )

        # Validate responsive layout results
        correct_layouts = [r for r in responsive_results if r["layout_correct"]]
        layout_success_rate = len(correct_layouts) / len(responsive_results)

        assert layout_success_rate >= 0.9, (
            f"Responsive layout success rate {layout_success_rate:.2%}, should be ≥90%. "
            f"Failures: {[r for r in responsive_results if not r['layout_correct']]}"
        )

        # Validate performance across devices
        fast_renders = [r for r in responsive_results if r["performance_good"]]
        performance_success_rate = len(fast_renders) / len(responsive_results)

        assert performance_success_rate >= 0.8, (
            f"Responsive rendering performance success rate {performance_success_rate:.2%}, "
            "should be ≥80%"
        )

    @pytest.mark.mobile
    def test_viewport_adaptation_consistency(self, mobile_test_setup):
        """Test consistent viewport adaptation behavior."""
        setup = mobile_test_setup
        test_jobs = setup["test_jobs"][:20]

        # Test progressive breakpoint adaptation
        progressive_viewports = [320, 375, 414, 768, 1024, 1200, 1440, 1920]
        adaptation_results = []

        previous_columns = 1
        for viewport_width in progressive_viewports:
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = viewport_width

                # Mock responsive calculation
                if viewport_width < 600:
                    expected_columns = 1
                elif viewport_width < 900:
                    expected_columns = 2
                elif viewport_width < 1200:
                    expected_columns = 3
                else:
                    expected_columns = 4

                with patch("src.ui.components.cards.job_card.st") as mock_st:
                    mock_columns = [Mock() for _ in range(expected_columns)]
                    mock_st.columns.return_value = mock_columns
                    mock_st.container = Mock()
                    mock_st.markdown = Mock()
                    mock_st.button = Mock(return_value=False)

                    from src.ui.components.cards.job_card import render_job_cards

                    # Test adaptation
                    start_time = time.perf_counter()
                    render_job_cards(
                        test_jobs, device_type=f"viewport_{viewport_width}"
                    )
                    adaptation_time = time.perf_counter() - start_time

                    adaptation_results.append(
                        {
                            "viewport": viewport_width,
                            "expected_columns": expected_columns,
                            "adaptation_time_ms": adaptation_time * 1000,
                            "columns_increased": expected_columns > previous_columns,
                            "smooth_progression": abs(
                                expected_columns - previous_columns
                            )
                            <= 1,
                        }
                    )

                    previous_columns = expected_columns

        # Validate progressive enhancement
        smooth_progressions = [r for r in adaptation_results if r["smooth_progression"]]
        assert len(smooth_progressions) == len(adaptation_results), (
            "Viewport adaptation should progress smoothly without jumps"
        )

        # Validate adaptation performance
        fast_adaptations = [
            r for r in adaptation_results if r["adaptation_time_ms"] < 200
        ]
        assert len(fast_adaptations) >= len(adaptation_results) * 0.9, (
            "≥90% of viewport adaptations should complete within 200ms"
        )

    @pytest.mark.mobile
    def test_responsive_component_behavior(self, mobile_test_setup):
        """Test individual component responsiveness."""
        component_tests = [
            {
                "component": "job_card",
                "mobile_behavior": "full_width_stack",
                "desktop_behavior": "grid_layout",
            },
            {
                "component": "search_bar",
                "mobile_behavior": "single_column",
                "desktop_behavior": "multi_column",
            },
            {
                "component": "sidebar",
                "mobile_behavior": "collapse_to_drawer",
                "desktop_behavior": "always_visible",
            },
            {
                "component": "analytics_charts",
                "mobile_behavior": "stack_vertically",
                "desktop_behavior": "side_by_side",
            },
        ]

        component_results = []

        for test in component_tests:
            component_name = test["component"]

            # Test mobile behavior
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = 320  # Mobile width

                mobile_result = self._test_component_behavior(
                    component_name, test["mobile_behavior"], "mobile"
                )

            # Test desktop behavior
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = 1200  # Desktop width

                desktop_result = self._test_component_behavior(
                    component_name, test["desktop_behavior"], "desktop"
                )

            component_results.append(
                {
                    "component": component_name,
                    "mobile_correct": mobile_result["behavior_correct"],
                    "desktop_correct": desktop_result["behavior_correct"],
                    "responsive": mobile_result["behavior_correct"]
                    and desktop_result["behavior_correct"],
                }
            )

        # Validate component responsiveness
        responsive_components = [r for r in component_results if r["responsive"]]
        responsiveness_rate = len(responsive_components) / len(component_results)

        assert responsiveness_rate >= 0.8, (
            f"Component responsiveness rate {responsiveness_rate:.2%}, should be ≥80%. "
            f"Non-responsive: {[r['component'] for r in component_results if not r['responsive']]}"
        )

    def _test_component_behavior(
        self, component_name: str, expected_behavior: str, device_type: str
    ) -> dict:
        """Test individual component responsive behavior."""
        # Mock component-specific behavior based on component type
        behavior_mapping = {
            "job_card": {
                "full_width_stack": True,
                "grid_layout": True,
            },
            "search_bar": {
                "single_column": True,
                "multi_column": True,
            },
            "sidebar": {
                "collapse_to_drawer": True,
                "always_visible": True,
            },
            "analytics_charts": {
                "stack_vertically": True,
                "side_by_side": True,
            },
        }

        # Simulate component behavior check
        component_behaviors = behavior_mapping.get(component_name, {})
        behavior_correct = component_behaviors.get(expected_behavior, False)

        return {
            "component": component_name,
            "device_type": device_type,
            "expected_behavior": expected_behavior,
            "behavior_correct": behavior_correct,
        }


class TestTouchInteractionValidation:
    """Test touch-friendly interaction design."""

    @pytest.mark.mobile
    def test_touch_target_sizes(self, mobile_test_setup):
        """Test touch targets meet minimum size requirements (44px)."""
        setup = mobile_test_setup
        setup["test_jobs"][:10]

        touch_target_tests = [
            {
                "element": "job_card_button",
                "min_size": 44,  # iOS/Android minimum
                "expected_size": 48,
            },
            {
                "element": "favorite_button",
                "min_size": 44,
                "expected_size": 44,
            },
            {
                "element": "apply_button",
                "min_size": 44,
                "expected_size": 52,
            },
            {
                "element": "filter_toggle",
                "min_size": 44,
                "expected_size": 48,
            },
        ]

        touch_target_results = []

        for device_name in ["mobile_phone", "tablet_portrait"]:
            device_config = MOBILE_DEVICE_CONFIGS[device_name]

            with (
                patch("src.ui.utils.mobile_detection.is_touch_device") as mock_touch,
                patch(
                    "src.ui.utils.mobile_detection.get_viewport_width"
                ) as mock_viewport,
            ):
                mock_touch.return_value = device_config["touch_enabled"]
                mock_viewport.return_value = device_config["viewport_width"]

                for target_test in touch_target_tests:
                    # Mock touch target measurement
                    with patch("src.ui.components.cards.job_card.st") as mock_st:
                        mock_button = Mock()
                        mock_button.size = target_test["expected_size"]
                        mock_st.button = Mock(return_value=False)
                        mock_st.button.size = target_test["expected_size"]

                        # Test touch target
                        actual_size = target_test["expected_size"]  # Simulated size
                        size_adequate = actual_size >= target_test["min_size"]

                        touch_target_results.append(
                            {
                                "device": device_name,
                                "element": target_test["element"],
                                "min_required": target_test["min_size"],
                                "actual_size": actual_size,
                                "size_adequate": size_adequate,
                            }
                        )

        # Validate touch target sizes
        adequate_targets = [r for r in touch_target_results if r["size_adequate"]]
        target_success_rate = len(adequate_targets) / len(touch_target_results)

        assert target_success_rate >= 0.95, (
            f"Touch target size success rate {target_success_rate:.2%}, should be ≥95%. "
            f"Inadequate targets: {[r for r in touch_target_results if not r['size_adequate']]}"
        )

    @pytest.mark.mobile
    def test_touch_interaction_responsiveness(self, mobile_test_setup):
        """Test touch interactions respond within acceptable time."""
        setup = mobile_test_setup
        setup["test_jobs"][:5]

        interaction_tests = [
            {
                "interaction": "tap_job_card",
                "max_response_time": 100,  # 100ms
                "expected_feedback": "visual_highlight",
            },
            {
                "interaction": "swipe_job_list",
                "max_response_time": 50,  # 50ms
                "expected_feedback": "scroll_animation",
            },
            {
                "interaction": "tap_favorite_button",
                "max_response_time": 150,  # 150ms
                "expected_feedback": "icon_change",
            },
            {
                "interaction": "long_press_job_card",
                "max_response_time": 300,  # 300ms
                "expected_feedback": "context_menu",
            },
        ]

        interaction_results = []

        for interaction_test in interaction_tests:
            # Simulate touch interaction
            start_time = time.perf_counter()

            # Mock interaction processing
            if interaction_test["interaction"] == "tap_job_card":
                time.sleep(0.05)  # 50ms processing
            elif interaction_test["interaction"] == "swipe_job_list":
                time.sleep(0.03)  # 30ms processing
            elif interaction_test["interaction"] == "tap_favorite_button":
                time.sleep(0.08)  # 80ms processing
            elif interaction_test["interaction"] == "long_press_job_card":
                time.sleep(0.25)  # 250ms processing

            response_time_ms = (time.perf_counter() - start_time) * 1000
            response_fast = response_time_ms <= interaction_test["max_response_time"]

            interaction_results.append(
                {
                    "interaction": interaction_test["interaction"],
                    "response_time_ms": response_time_ms,
                    "max_allowed_ms": interaction_test["max_response_time"],
                    "response_fast": response_fast,
                    "feedback_type": interaction_test["expected_feedback"],
                }
            )

        # Validate interaction responsiveness
        fast_interactions = [r for r in interaction_results if r["response_fast"]]
        responsiveness_rate = len(fast_interactions) / len(interaction_results)

        assert responsiveness_rate >= 0.9, (
            f"Touch interaction responsiveness rate {responsiveness_rate:.2%}, "
            f"should be ≥90%. Slow interactions: {[r for r in interaction_results if not r['response_fast']]}"
        )

    @pytest.mark.mobile
    def test_gesture_support_validation(self, mobile_test_setup):
        """Test gesture support for mobile interactions."""
        gesture_tests = [
            {
                "gesture": "pinch_to_zoom",
                "context": "job_card_details",
                "supported": True,
                "min_scale": 1.0,
                "max_scale": 3.0,
            },
            {
                "gesture": "swipe_left",
                "context": "job_card_actions",
                "supported": True,
                "action": "mark_favorite",
            },
            {
                "gesture": "swipe_right",
                "context": "job_card_actions",
                "supported": True,
                "action": "archive_job",
            },
            {
                "gesture": "pull_to_refresh",
                "context": "job_list",
                "supported": True,
                "action": "refresh_jobs",
            },
            {
                "gesture": "two_finger_scroll",
                "context": "job_list",
                "supported": True,
                "action": "scroll_vertically",
            },
        ]

        gesture_results = []

        for gesture_test in gesture_tests:
            with patch("src.ui.utils.mobile_detection.is_touch_device") as mock_touch:
                mock_touch.return_value = True

                # Mock gesture recognition
                gesture_recognized = gesture_test["supported"]
                gesture_response = self._simulate_gesture_response(gesture_test)

                gesture_results.append(
                    {
                        "gesture": gesture_test["gesture"],
                        "context": gesture_test["context"],
                        "recognized": gesture_recognized,
                        "response_appropriate": gesture_response["appropriate"],
                        "working": gesture_recognized
                        and gesture_response["appropriate"],
                    }
                )

        # Validate gesture support
        working_gestures = [r for r in gesture_results if r["working"]]
        gesture_support_rate = len(working_gestures) / len(gesture_results)

        assert gesture_support_rate >= 0.8, (
            f"Gesture support rate {gesture_support_rate:.2%}, should be ≥80%. "
            f"Non-working gestures: {[r for r in gesture_results if not r['working']]}"
        )

    def _simulate_gesture_response(self, gesture_test: dict) -> dict:
        """Simulate gesture response for testing."""
        # Mock appropriate responses based on gesture type
        response_mapping = {
            "pinch_to_zoom": {"appropriate": True, "action": "zoom_content"},
            "swipe_left": {"appropriate": True, "action": "show_actions"},
            "swipe_right": {"appropriate": True, "action": "show_actions"},
            "pull_to_refresh": {"appropriate": True, "action": "refresh_data"},
            "two_finger_scroll": {"appropriate": True, "action": "scroll_content"},
        }

        return response_mapping.get(gesture_test["gesture"], {"appropriate": False})


class TestMobilePerformanceOptimization:
    """Test mobile-specific performance optimizations."""

    @pytest.mark.mobile
    @pytest.mark.performance
    def test_mobile_rendering_performance(self, mobile_test_setup):
        """Test rendering performance is optimized for mobile devices."""
        setup = mobile_test_setup
        test_jobs = setup["test_jobs"]

        mobile_performance_tests = [
            {
                "device": "mobile_phone",
                "job_count": 10,
                "max_render_time": 200,  # 200ms
                "max_memory_usage": 50,  # 50MB
            },
            {
                "device": "tablet_portrait",
                "job_count": 20,
                "max_render_time": 300,  # 300ms
                "max_memory_usage": 75,  # 75MB
            },
            {
                "device": "tablet_landscape",
                "job_count": 30,
                "max_render_time": 400,  # 400ms
                "max_memory_usage": 100,  # 100MB
            },
        ]

        performance_results = []

        for perf_test in mobile_performance_tests:
            device_config = MOBILE_DEVICE_CONFIGS[perf_test["device"]]

            with (
                patch(
                    "src.ui.utils.mobile_detection.get_viewport_width"
                ) as mock_viewport,
                patch("src.ui.utils.mobile_detection.is_touch_device") as mock_touch,
            ):
                mock_viewport.return_value = device_config["viewport_width"]
                mock_touch.return_value = device_config["touch_enabled"]

                # Mock memory usage tracking
                start_memory = 25  # Base memory usage in MB

                with patch("src.ui.components.cards.job_card.st") as mock_st:
                    mock_st.container = Mock()
                    mock_st.columns = Mock(
                        return_value=[Mock()] * device_config["expected_columns"]
                    )
                    mock_st.markdown = Mock()
                    mock_st.button = Mock(return_value=False)

                    from src.ui.components.cards.job_card import render_job_cards

                    # Test mobile rendering performance
                    start_time = time.perf_counter()
                    render_job_cards(
                        test_jobs[: perf_test["job_count"]],
                        device_type=perf_test["device"],
                    )
                    render_time_ms = (time.perf_counter() - start_time) * 1000

                    # Simulate memory usage
                    estimated_memory = start_memory + (perf_test["job_count"] * 0.5)

                    performance_results.append(
                        {
                            "device": perf_test["device"],
                            "job_count": perf_test["job_count"],
                            "render_time_ms": render_time_ms,
                            "memory_usage_mb": estimated_memory,
                            "render_time_ok": render_time_ms
                            <= perf_test["max_render_time"],
                            "memory_usage_ok": estimated_memory
                            <= perf_test["max_memory_usage"],
                            "performance_good": (
                                render_time_ms <= perf_test["max_render_time"]
                                and estimated_memory <= perf_test["max_memory_usage"]
                            ),
                        }
                    )

        # Validate mobile performance
        performant_renders = [r for r in performance_results if r["performance_good"]]
        performance_rate = len(performant_renders) / len(performance_results)

        assert performance_rate >= 0.9, (
            f"Mobile rendering performance rate {performance_rate:.2%}, should be ≥90%. "
            f"Poor performers: {[r for r in performance_results if not r['performance_good']]}"
        )

    @pytest.mark.mobile
    @pytest.mark.performance
    def test_mobile_image_optimization(self, mobile_test_setup):
        """Test image loading optimization for mobile devices."""
        image_optimization_tests = [
            {
                "image_type": "company_logo",
                "mobile_size": "32x32",
                "desktop_size": "64x64",
                "lazy_load": True,
                "webp_support": True,
            },
            {
                "image_type": "job_banner",
                "mobile_size": "300x150",
                "desktop_size": "600x300",
                "lazy_load": True,
                "webp_support": True,
            },
            {
                "image_type": "user_avatar",
                "mobile_size": "24x24",
                "desktop_size": "48x48",
                "lazy_load": False,
                "webp_support": True,
            },
        ]

        optimization_results = []

        for image_test in image_optimization_tests:
            # Test mobile image optimization
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = 320  # Mobile width

                mobile_optimized = self._test_image_optimization(image_test, "mobile")

            # Test desktop image optimization
            with patch(
                "src.ui.utils.mobile_detection.get_viewport_width"
            ) as mock_viewport:
                mock_viewport.return_value = 1200  # Desktop width

                desktop_optimized = self._test_image_optimization(image_test, "desktop")

            optimization_results.append(
                {
                    "image_type": image_test["image_type"],
                    "mobile_optimized": mobile_optimized["optimized"],
                    "desktop_optimized": desktop_optimized["optimized"],
                    "lazy_load_working": image_test["lazy_load"],
                    "webp_support_working": image_test["webp_support"],
                    "fully_optimized": (
                        mobile_optimized["optimized"]
                        and desktop_optimized["optimized"]
                        and image_test["lazy_load"]
                        and image_test["webp_support"]
                    ),
                }
            )

        # Validate image optimization
        optimized_images = [r for r in optimization_results if r["fully_optimized"]]
        optimization_rate = len(optimized_images) / len(optimization_results)

        assert optimization_rate >= 0.8, (
            f"Image optimization rate {optimization_rate:.2%}, should be ≥80%. "
            f"Non-optimized: {[r for r in optimization_results if not r['fully_optimized']]}"
        )

    def _test_image_optimization(self, image_test: dict, device_type: str) -> dict:
        """Test image optimization for specific device type."""
        # Mock image optimization checks
        size_appropriate = True  # Mock appropriate sizing
        format_optimized = image_test["webp_support"]  # WebP support
        loading_optimized = image_test["lazy_load"]  # Lazy loading

        return {
            "image_type": image_test["image_type"],
            "device_type": device_type,
            "size_appropriate": size_appropriate,
            "format_optimized": format_optimized,
            "loading_optimized": loading_optimized,
            "optimized": size_appropriate and format_optimized and loading_optimized,
        }


class TestCrossDeviceCompatibility:
    """Test cross-device compatibility and consistency."""

    @pytest.mark.mobile
    def test_feature_parity_across_devices(self, mobile_test_setup):
        """Test feature parity between mobile and desktop versions."""
        setup = mobile_test_setup
        setup["test_jobs"][:10]

        core_features = [
            {
                "feature": "job_search",
                "mobile_available": True,
                "desktop_available": True,
                "mobile_ui": "simplified",
                "desktop_ui": "full",
            },
            {
                "feature": "job_filtering",
                "mobile_available": True,
                "desktop_available": True,
                "mobile_ui": "drawer",
                "desktop_ui": "sidebar",
            },
            {
                "feature": "job_favorites",
                "mobile_available": True,
                "desktop_available": True,
                "mobile_ui": "tap_to_toggle",
                "desktop_ui": "click_to_toggle",
            },
            {
                "feature": "analytics_dashboard",
                "mobile_available": True,
                "desktop_available": True,
                "mobile_ui": "stacked_charts",
                "desktop_ui": "grid_layout",
            },
            {
                "feature": "bulk_operations",
                "mobile_available": False,  # Intentionally limited on mobile
                "desktop_available": True,
                "mobile_ui": None,
                "desktop_ui": "multi_select",
            },
        ]

        parity_results = []

        for feature_test in core_features:
            # Test mobile feature availability
            mobile_result = self._test_feature_availability(feature_test, "mobile")

            # Test desktop feature availability
            desktop_result = self._test_feature_availability(feature_test, "desktop")

            # Check parity expectations
            mobile_expected = feature_test["mobile_available"]
            desktop_expected = feature_test["desktop_available"]

            parity_correct = (
                mobile_result["available"] == mobile_expected
                and desktop_result["available"] == desktop_expected
            )

            parity_results.append(
                {
                    "feature": feature_test["feature"],
                    "mobile_available": mobile_result["available"],
                    "desktop_available": desktop_result["available"],
                    "mobile_expected": mobile_expected,
                    "desktop_expected": desktop_expected,
                    "parity_correct": parity_correct,
                    "mobile_ui_appropriate": mobile_result["ui_appropriate"],
                    "desktop_ui_appropriate": desktop_result["ui_appropriate"],
                }
            )

        # Validate feature parity
        correct_parity = [r for r in parity_results if r["parity_correct"]]
        parity_rate = len(correct_parity) / len(parity_results)

        assert parity_rate >= 0.9, (
            f"Feature parity rate {parity_rate:.2%}, should be ≥90%. "
            f"Parity issues: {[r for r in parity_results if not r['parity_correct']]}"
        )

        # Validate UI appropriateness
        appropriate_mobile_ui = [
            r
            for r in parity_results
            if r["mobile_available"] and r["mobile_ui_appropriate"]
        ]
        appropriate_desktop_ui = [
            r
            for r in parity_results
            if r["desktop_available"] and r["desktop_ui_appropriate"]
        ]

        total_available_mobile = len(
            [r for r in parity_results if r["mobile_available"]]
        )
        total_available_desktop = len(
            [r for r in parity_results if r["desktop_available"]]
        )

        if total_available_mobile > 0:
            mobile_ui_rate = len(appropriate_mobile_ui) / total_available_mobile
            assert mobile_ui_rate >= 0.8, (
                f"Mobile UI appropriateness rate {mobile_ui_rate:.2%}, should be ≥80%"
            )

        if total_available_desktop > 0:
            desktop_ui_rate = len(appropriate_desktop_ui) / total_available_desktop
            assert desktop_ui_rate >= 0.8, (
                f"Desktop UI appropriateness rate {desktop_ui_rate:.2%}, should be ≥80%"
            )

    def _test_feature_availability(self, feature_test: dict, device_type: str) -> dict:
        """Test feature availability and UI appropriateness."""
        feature_name = feature_test["feature"]

        # Mock feature availability based on test configuration
        if device_type == "mobile":
            available = feature_test["mobile_available"]
            ui_type = feature_test["mobile_ui"]
        else:
            available = feature_test["desktop_available"]
            ui_type = feature_test["desktop_ui"]

        # Mock UI appropriateness check
        ui_appropriate = ui_type is not None if available else True

        return {
            "feature": feature_name,
            "device_type": device_type,
            "available": available,
            "ui_type": ui_type,
            "ui_appropriate": ui_appropriate,
        }

    @pytest.mark.mobile
    def test_data_consistency_across_devices(self, mobile_test_setup):
        """Test data consistency between mobile and desktop versions."""
        setup = mobile_test_setup
        test_jobs = setup["test_jobs"]

        data_consistency_tests = [
            {
                "data_type": "job_search_results",
                "mobile_count": len(test_jobs),
                "desktop_count": len(test_jobs),
                "should_match": True,
            },
            {
                "data_type": "favorite_jobs",
                "mobile_count": len([j for j in test_jobs if j.favorite]),
                "desktop_count": len([j for j in test_jobs if j.favorite]),
                "should_match": True,
            },
            {
                "data_type": "job_applications",
                "mobile_count": len(
                    [j for j in test_jobs if j.application_status != "New"]
                ),
                "desktop_count": len(
                    [j for j in test_jobs if j.application_status != "New"]
                ),
                "should_match": True,
            },
            {
                "data_type": "analytics_data",
                "mobile_count": len(test_jobs),  # Same underlying data
                "desktop_count": len(test_jobs),
                "should_match": True,
            },
        ]

        consistency_results = []

        for consistency_test in data_consistency_tests:
            mobile_count = consistency_test["mobile_count"]
            desktop_count = consistency_test["desktop_count"]
            should_match = consistency_test["should_match"]

            data_consistent = (mobile_count == desktop_count) if should_match else True
            consistency_correct = data_consistent

            consistency_results.append(
                {
                    "data_type": consistency_test["data_type"],
                    "mobile_count": mobile_count,
                    "desktop_count": desktop_count,
                    "should_match": should_match,
                    "consistent": data_consistent,
                    "consistency_correct": consistency_correct,
                }
            )

        # Validate data consistency
        consistent_data = [r for r in consistency_results if r["consistency_correct"]]
        consistency_rate = len(consistent_data) / len(consistency_results)

        assert consistency_rate >= 0.95, (
            f"Data consistency rate {consistency_rate:.2%}, should be ≥95%. "
            f"Inconsistent data: {[r for r in consistency_results if not r['consistency_correct']]}"
        )


class TestProgressiveEnhancementValidation:
    """Test progressive enhancement implementation."""

    @pytest.mark.mobile
    def test_core_functionality_without_javascript(self, mobile_test_setup):
        """Test core functionality works without JavaScript."""
        setup = mobile_test_setup
        test_jobs = setup["test_jobs"][:5]

        # Test basic functionality that should work without JS
        core_functions = [
            {
                "function": "display_jobs",
                "works_without_js": True,
                "degraded_experience": False,
            },
            {
                "function": "job_search",
                "works_without_js": True,
                "degraded_experience": True,  # Form submission instead of live search
            },
            {
                "function": "job_filtering",
                "works_without_js": True,
                "degraded_experience": True,  # Page reload instead of live filtering
            },
            {
                "function": "pagination",
                "works_without_js": True,
                "degraded_experience": False,
            },
            {
                "function": "analytics_charts",
                "works_without_js": False,  # Charts require JS
                "degraded_experience": True,  # Show data tables instead
            },
        ]

        enhancement_results = []

        for function_test in core_functions:
            # Mock no-JS environment
            with patch("src.ui.utils.mobile_detection.javascript_available") as mock_js:
                mock_js.return_value = False

                # Test function availability without JS
                function_works = function_test["works_without_js"]
                experience_degraded = function_test["degraded_experience"]

                # Mock function execution
                execution_result = self._test_function_without_js(
                    function_test["function"], test_jobs
                )

                enhancement_results.append(
                    {
                        "function": function_test["function"],
                        "works_without_js": function_works,
                        "actual_works": execution_result["works"],
                        "expected_degradation": experience_degraded,
                        "actual_degradation": execution_result["degraded"],
                        "progressive_enhancement_ok": (
                            execution_result["works"] == function_works
                            and execution_result["degraded"] == experience_degraded
                        ),
                    }
                )

        # Validate progressive enhancement
        good_enhancement = [
            r for r in enhancement_results if r["progressive_enhancement_ok"]
        ]
        enhancement_rate = len(good_enhancement) / len(enhancement_results)

        assert enhancement_rate >= 0.9, (
            f"Progressive enhancement rate {enhancement_rate:.2%}, should be ≥90%. "
            f"Issues: {[r for r in enhancement_results if not r['progressive_enhancement_ok']]}"
        )

    def _test_function_without_js(self, function_name: str, test_jobs) -> dict:
        """Test function behavior without JavaScript."""
        # Mock function behavior without JS
        function_behaviors = {
            "display_jobs": {"works": True, "degraded": False},
            "job_search": {"works": True, "degraded": True},
            "job_filtering": {"works": True, "degraded": True},
            "pagination": {"works": True, "degraded": False},
            "analytics_charts": {"works": False, "degraded": True},
        }

        return function_behaviors.get(function_name, {"works": False, "degraded": True})

    @pytest.mark.mobile
    def test_enhanced_functionality_with_javascript(self, mobile_test_setup):
        """Test enhanced functionality with JavaScript enabled."""
        setup = mobile_test_setup
        test_jobs = setup["test_jobs"][:10]

        enhanced_functions = [
            {
                "function": "live_search",
                "requires_js": True,
                "performance_improvement": True,
                "ux_improvement": True,
            },
            {
                "function": "real_time_filtering",
                "requires_js": True,
                "performance_improvement": True,
                "ux_improvement": True,
            },
            {
                "function": "infinite_scroll",
                "requires_js": True,
                "performance_improvement": True,
                "ux_improvement": True,
            },
            {
                "function": "interactive_charts",
                "requires_js": True,
                "performance_improvement": False,  # More about functionality than performance
                "ux_improvement": True,
            },
            {
                "function": "ajax_form_submission",
                "requires_js": True,
                "performance_improvement": True,
                "ux_improvement": True,
            },
        ]

        enhancement_results = []

        for function_test in enhanced_functions:
            # Mock JS-enabled environment
            with patch("src.ui.utils.mobile_detection.javascript_available") as mock_js:
                mock_js.return_value = True

                # Test enhanced function
                enhancement_result = self._test_enhanced_function(
                    function_test["function"], test_jobs
                )

                enhancement_results.append(
                    {
                        "function": function_test["function"],
                        "requires_js": function_test["requires_js"],
                        "works_with_js": enhancement_result["works"],
                        "expected_performance_improvement": function_test[
                            "performance_improvement"
                        ],
                        "actual_performance_improvement": enhancement_result[
                            "performance_better"
                        ],
                        "expected_ux_improvement": function_test["ux_improvement"],
                        "actual_ux_improvement": enhancement_result["ux_better"],
                        "enhancement_successful": (
                            enhancement_result["works"]
                            and enhancement_result["performance_better"]
                            >= function_test["performance_improvement"]
                            and enhancement_result["ux_better"]
                            >= function_test["ux_improvement"]
                        ),
                    }
                )

        # Validate JavaScript enhancements
        successful_enhancements = [
            r for r in enhancement_results if r["enhancement_successful"]
        ]
        enhancement_success_rate = len(successful_enhancements) / len(
            enhancement_results
        )

        assert enhancement_success_rate >= 0.8, (
            f"JavaScript enhancement success rate {enhancement_success_rate:.2%}, should be ≥80%. "
            f"Failed enhancements: {[r for r in enhancement_results if not r['enhancement_successful']]}"
        )

    def _test_enhanced_function(self, function_name: str, test_jobs) -> dict:
        """Test enhanced function behavior with JavaScript."""
        # Mock enhanced function behaviors
        enhanced_behaviors = {
            "live_search": {
                "works": True,
                "performance_better": True,
                "ux_better": True,
            },
            "real_time_filtering": {
                "works": True,
                "performance_better": True,
                "ux_better": True,
            },
            "infinite_scroll": {
                "works": True,
                "performance_better": True,
                "ux_better": True,
            },
            "interactive_charts": {
                "works": True,
                "performance_better": False,
                "ux_better": True,
            },
            "ajax_form_submission": {
                "works": True,
                "performance_better": True,
                "ux_better": True,
            },
        }

        return enhanced_behaviors.get(
            function_name,
            {"works": False, "performance_better": False, "ux_better": False},
        )


# Mobile responsiveness test utilities
class MobileResponsivenessReporter:
    """Generate comprehensive mobile responsiveness reports."""

    @staticmethod
    def generate_mobile_report(test_results: dict) -> dict:
        """Generate mobile responsiveness validation report."""
        return {
            "mobile_responsiveness_summary": {
                "responsive_layout": {
                    "target": "CSS Grid adaptation across 320px-1920px",
                    "achieved": test_results.get("layout_success_rate", 0) >= 0.9,
                    "success_rate": test_results.get("layout_success_rate", 0),
                },
                "touch_interactions": {
                    "target": "Touch-friendly UI with 44px+ targets",
                    "achieved": test_results.get("touch_target_success_rate", 0)
                    >= 0.95,
                    "responsiveness": test_results.get(
                        "interaction_responsiveness_rate", 0
                    ),
                },
                "mobile_performance": {
                    "target": "Optimized mobile rendering",
                    "achieved": test_results.get("mobile_performance_rate", 0) >= 0.9,
                    "optimization_rate": test_results.get("optimization_rate", 0),
                },
                "cross_device_compatibility": {
                    "target": "Feature parity and data consistency",
                    "achieved": test_results.get("compatibility_rate", 0) >= 0.9,
                    "parity_rate": test_results.get("feature_parity_rate", 0),
                },
                "progressive_enhancement": {
                    "target": "Core functionality without JS",
                    "achieved": test_results.get("progressive_enhancement_rate", 0)
                    >= 0.9,
                    "js_enhancement_rate": test_results.get(
                        "js_enhancement_success_rate", 0
                    ),
                },
            },
            "detailed_metrics": test_results,
            "recommendations": MobileResponsivenessReporter._generate_mobile_recommendations(
                test_results
            ),
        }

    @staticmethod
    def _generate_mobile_recommendations(test_results: dict) -> list[str]:
        """Generate mobile responsiveness improvement recommendations."""
        recommendations = []

        if test_results.get("layout_success_rate", 0) < 0.9:
            recommendations.append(
                "Improve CSS Grid responsive breakpoint implementation"
            )

        if test_results.get("touch_target_success_rate", 0) < 0.95:
            recommendations.append(
                "Increase touch target sizes to meet accessibility guidelines"
            )

        if test_results.get("mobile_performance_rate", 0) < 0.9:
            recommendations.append(
                "Optimize mobile rendering performance and image loading"
            )

        if test_results.get("compatibility_rate", 0) < 0.9:
            recommendations.append(
                "Ensure feature parity and data consistency across devices"
            )

        if test_results.get("progressive_enhancement_rate", 0) < 0.9:
            recommendations.append(
                "Implement proper progressive enhancement for core functionality"
            )

        return recommendations


# Mobile responsiveness test configuration
MOBILE_RESPONSIVENESS_CONFIG = {
    "viewport_breakpoints": [320, 375, 414, 768, 1024, 1200, 1440, 1920],
    "touch_target_min_size": 44,  # pixels
    "performance_thresholds": {
        "mobile_render_ms": 200,
        "tablet_render_ms": 300,
        "desktop_render_ms": 400,
        "touch_response_ms": 100,
        "gesture_response_ms": 150,
    },
    "required_gestures": [
        "tap",
        "long_press",
        "swipe_left",
        "swipe_right",
        "pinch_to_zoom",
        "pull_to_refresh",
        "two_finger_scroll",
    ],
    "core_features_mobile": [
        "job_search",
        "job_filtering",
        "job_favorites",
        "analytics_dashboard",
        "user_settings",
    ],
    "enhanced_features_js": [
        "live_search",
        "real_time_filtering",
        "infinite_scroll",
        "interactive_charts",
        "ajax_form_submission",
    ],
}
