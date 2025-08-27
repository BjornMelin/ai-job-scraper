"""Native Components Testing Framework.

This package provides comprehensive testing for Streamlit native components,
focusing on functionality preservation, performance validation, and integration
testing during library optimization migrations.

Streams:
- Stream A: Progress components (st.status, st.progress, st.toast)
- Stream B: Caching performance (st.cache_data, st.cache_resource)
- Stream C: Fragment behavior (st.fragment, run_every, scoped reruns)
"""

__version__ = "1.0.0"

from tests.native_components.framework import (
    NativeComponentTester,
    NativeComponentValidator,
    PerformanceBenchmark,
    StreamValidationResults,
)

__all__ = [
    "NativeComponentTester",
    "NativeComponentValidator",
    "PerformanceBenchmark",
    "StreamValidationResults",
]
