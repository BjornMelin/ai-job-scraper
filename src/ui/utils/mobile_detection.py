"""Mobile and viewport detection utilities for responsive UI components.

This module provides reliable mobile detection and viewport utilities that replace
the problematic st.get_option("client.toolbarMode") approach with modern JavaScript
solutions using the matchMedia API and CSS viewport queries.

Key features:
- Reliable mobile/tablet/desktop detection
- Real-time viewport size monitoring
- Touch capability detection
- Orientation change handling
- Performance optimized with caching
"""

import logging

from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def get_viewport_info() -> dict[str, Any]:
    """Get comprehensive viewport information using JavaScript matchMedia API.

    This function provides reliable device detection replacing the unreliable
    st.get_option("client.toolbarMode") approach with modern viewport queries.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - is_mobile: Boolean indicating if device is mobile (≤640px)
            - is_tablet: Boolean indicating if device is tablet (641-1024px)
            - is_desktop: Boolean indicating if device is desktop (≥1025px)
            - viewport_width: Current viewport width in pixels
            - viewport_height: Current viewport height in pixels
            - is_portrait: Boolean indicating portrait orientation
            - is_landscape: Boolean indicating landscape orientation
            - is_touch_device: Boolean indicating touch capability
            - device_pixel_ratio: Device pixel ratio for high-DPI displays
            - preferred_color_scheme: 'dark' or 'light' based on user preference

    Example:
        >>> viewport = get_viewport_info()
        >>> if viewport["is_mobile"]:
        ...     render_mobile_layout()
        >>> else:
        ...     render_desktop_layout()
    """
    # JavaScript code for comprehensive viewport detection
    viewport_js = """
    <script>
    // Comprehensive viewport detection using modern matchMedia API
    function getViewportInfo() {
        const viewport = {
            // Viewport dimensions
            viewport_width: window.innerWidth,
            viewport_height: window.innerHeight,

            // Device type detection using standard breakpoints
            is_mobile: window.matchMedia('(max-width: 640px)').matches,
            is_tablet: window.matchMedia('(min-width: 641px) and (max-width: 1024px)').matches,
            is_desktop: window.matchMedia('(min-width: 1025px)').matches,

            // Orientation detection
            is_portrait: window.matchMedia('(orientation: portrait)').matches,
            is_landscape: window.matchMedia('(orientation: landscape)').matches,

            // Touch capability detection
            is_touch_device: 'ontouchstart' in window || navigator.maxTouchPoints > 0,

            // Display characteristics
            device_pixel_ratio: window.devicePixelRatio || 1,

            // User preferences
            preferred_color_scheme: window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
            prefers_reduced_motion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
            prefers_high_contrast: window.matchMedia('(prefers-contrast: high)').matches
        };

        // Store in session storage for Streamlit access
        sessionStorage.setItem('viewport_info', JSON.stringify(viewport));

        // Also dispatch custom event for real-time updates
        window.dispatchEvent(new CustomEvent('viewportChanged', { detail: viewport }));

        return viewport;
    }

    // Initial detection
    const viewportInfo = getViewportInfo();

    // Update on resize with debounce for performance
    let resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(getViewportInfo, 100);
    });

    // Update on orientation change
    window.addEventListener('orientationchange', function() {
        setTimeout(getViewportInfo, 100);
    });

    console.log('Viewport Info:', viewportInfo);
    </script>
    """

    # Inject JavaScript into page
    st.html(viewport_js, height=0)

    # Provide fallback values for server-side rendering
    return {
        "is_mobile": False,
        "is_tablet": False,
        "is_desktop": True,
        "viewport_width": 1920,
        "viewport_height": 1080,
        "is_portrait": False,
        "is_landscape": True,
        "is_touch_device": False,
        "device_pixel_ratio": 1.0,
        "preferred_color_scheme": "light",
        "prefers_reduced_motion": False,
        "prefers_high_contrast": False,
    }


@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid repeated calculations
def get_responsive_columns(total_items: int, default_desktop: int = 3) -> int:
    """Calculate optimal number of columns for responsive grid layout.

    This function determines the best column count based on viewport size
    and content amount, ensuring optimal user experience across all devices.

    Args:
        total_items: Total number of items to display
        default_desktop: Default columns for desktop view (default: 3)

    Returns:
        int: Optimal number of columns for current viewport

    Example:
        >>> columns = get_responsive_columns(len(jobs))
        >>> st.columns(columns)
    """
    viewport = get_viewport_info()

    # Mobile: Always 1 column for optimal readability
    if viewport.get("is_mobile", False):
        return 1

    # Tablet: 2 columns with consideration for item count
    if viewport.get("is_tablet", False):
        return min(2, total_items)

    # Desktop: Flexible columns based on content and screen size
    # Adjust columns based on total items to avoid sparse layouts
    if total_items <= 2:
        return min(total_items, 2)
    if total_items <= 6:
        return min(total_items, default_desktop)
    return default_desktop


def get_responsive_card_config() -> dict[str, Any]:
    """Get responsive configuration for job card display.

    Returns configuration optimized for current viewport including
    grid settings, pagination, and performance options.

    Returns:
        Dict[str, Any]: Configuration dictionary containing:
            - cards_per_page: Optimal number of cards per page
            - show_descriptions: Whether to show full descriptions
            - use_compact_mode: Whether to use compact card layout
            - enable_lazy_loading: Whether to enable lazy loading
            - grid_gap: CSS grid gap size
            - card_height: Recommended card height

    Example:
        >>> config = get_responsive_card_config()
        >>> st.markdown(
        ...     f"<style>.job-cards-container {{ gap: {config['grid_gap']}; }}</style>"
        ... )
    """
    viewport = get_viewport_info()

    # Mobile configuration - optimized for touch and limited screen space
    if viewport.get("is_mobile", False):
        return {
            "cards_per_page": 10,  # Fewer cards for faster loading on mobile
            "show_descriptions": True,  # Show descriptions for better context
            "use_compact_mode": False,  # Full cards for better readability
            "enable_lazy_loading": True,  # Essential for mobile performance
            "grid_gap": "1rem",
            "card_height": "240px",
            "font_size_multiplier": 1.0,
            "touch_friendly": True,
            "show_hover_effects": False,  # No hover on touch devices
        }

    # Tablet configuration - balanced approach
    if viewport.get("is_tablet", False):
        return {
            "cards_per_page": 20,
            "show_descriptions": True,
            "use_compact_mode": False,
            "enable_lazy_loading": True,
            "grid_gap": "1.5rem",
            "card_height": "260px",
            "font_size_multiplier": 1.05,
            "touch_friendly": True,
            "show_hover_effects": True,
        }

    # Desktop configuration - full feature set
    return {
        "cards_per_page": 30,  # More cards for larger screens
        "show_descriptions": True,
        "use_compact_mode": False,
        "enable_lazy_loading": False,  # Less critical on desktop
        "grid_gap": "2rem",
        "card_height": "280px",
        "font_size_multiplier": 1.1,
        "touch_friendly": False,
        "show_hover_effects": True,
    }


def inject_responsive_css() -> None:
    """Inject responsive CSS that adapts to current viewport characteristics.

    This function adds CSS custom properties based on the detected viewport
    to enable fine-tuned responsive behavior beyond standard media queries.
    """
    viewport = get_viewport_info()

    # Generate CSS custom properties based on viewport
    css_props = f"""
    <style>
    :root {{
        --viewport-width: {viewport.get("viewport_width", 1920)}px;
        --viewport-height: {viewport.get("viewport_height", 1080)}px;
        --is-mobile: {1 if viewport.get("is_mobile", False) else 0};
        --is-tablet: {1 if viewport.get("is_tablet", False) else 0};
        --is-desktop: {1 if viewport.get("is_desktop", True) else 0};
        --is-touch: {1 if viewport.get("is_touch_device", False) else 0};
        --device-pixel-ratio: {viewport.get("device_pixel_ratio", 1)};
        --preferred-color-scheme: {viewport.get("preferred_color_scheme", "light")};
    }}

    /* Conditional styles based on viewport properties */
    .responsive-container {{
        --columns: calc(var(--is-mobile) * 1 + var(--is-tablet) * 2 + var(--is-desktop) * 3);
        --gap: calc(var(--is-mobile) * 1rem + var(--is-tablet) * 1.5rem + var(--is-desktop) * 2rem);
        --padding: calc(var(--is-mobile) * 0.5rem + var(--is-tablet) * 1rem + var(--is-desktop) * 1rem);
    }}

    /* Touch-optimized styles */
    .touch-target {{
        min-height: calc(44px * var(--is-touch, 1) + 32px * (1 - var(--is-touch, 0)));
        min-width: calc(44px * var(--is-touch, 1) + 32px * (1 - var(--is-touch, 0)));
    }}
    </style>
    """

    st.html(css_props, height=0)
    logger.debug("Responsive CSS injected with viewport properties")


def get_device_type() -> str:
    """Get simplified device type string for UI logic.

    Returns:
        str: One of 'mobile', 'tablet', or 'desktop'

    Example:
        >>> device = get_device_type()
        >>> if device == "mobile":
        ...     columns = 1
        >>> elif device == 'tablet':
        ...     columns = 2
        >>> else:
        ...     columns = 3
    """
    viewport = get_viewport_info()

    if viewport.get("is_mobile", False):
        return "mobile"
    if viewport.get("is_tablet", False):
        return "tablet"
    return "desktop"


def is_mobile_device() -> bool:
    """Simple mobile detection for backward compatibility.

    Returns:
        bool: True if current device is mobile (≤640px)

    Example:
        >>> if is_mobile_device():
        ...     st.info("Mobile-optimized view")
    """
    return get_viewport_info().get("is_mobile", False)


def get_grid_columns_css() -> str:
    """Generate CSS Grid columns property based on current viewport.

    Returns:
        str: CSS grid-template-columns value optimized for current device

    Example:
        >>> css = f"grid-template-columns: {get_grid_columns_css()};"
        >>> st.html(f"<style>.cards-grid {{ {css} }}</style>")
    """
    device = get_device_type()

    if device == "mobile":
        return "repeat(auto-fill, minmax(min(100%, 280px), 1fr))"
    if device == "tablet":
        return "repeat(auto-fill, minmax(320px, 1fr))"
    return "repeat(auto-fill, minmax(350px, 1fr))"


# Pre-inject responsive CSS on module import for immediate availability
if hasattr(st, "html"):
    try:
        inject_responsive_css()
    except Exception as e:
        logger.debug("Could not inject responsive CSS on import: %s", e)
