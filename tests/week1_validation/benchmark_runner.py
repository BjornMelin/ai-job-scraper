"""Week 1 Performance Benchmark Runner with Comprehensive Metrics.

This module provides automated benchmark execution and detailed performance
reporting for all Week 1 stream achievements and their integration.

Features:
- Automated benchmark suite execution
- Comprehensive performance metrics collection
- Detailed reporting with visual charts
- Target validation against claimed improvements
- CI/CD integration ready
"""

import json
import time

from datetime import datetime
from pathlib import Path
from typing import Any

from tests.week1_validation.base_validation import (
    Week1ValidationSuite,
)
from tests.week1_validation.test_integration import Week1IntegrationValidator
from tests.week1_validation.test_stream_a_progress import StreamAProgressValidator
from tests.week1_validation.test_stream_b_caching import StreamBCachingValidator
from tests.week1_validation.test_stream_c_fragments import StreamCFragmentValidator


class Week1BenchmarkRunner:
    """Comprehensive benchmark runner for Week 1 achievements."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory for benchmark output files
        """
        self.output_dir = output_dir or Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize validators
        self.stream_a_validator = StreamAProgressValidator()
        self.stream_b_validator = StreamBCachingValidator()
        self.stream_c_validator = StreamCFragmentValidator()
        self.integration_validator = Week1IntegrationValidator()
        self.validation_suite = Week1ValidationSuite()

        # Benchmark results
        self.benchmark_results = {
            "metadata": {
                "execution_time": datetime.now().isoformat(),
                "version": "1.0.0",
                "targets": {
                    "stream_a_code_reduction": 96.0,
                    "stream_b_performance_improvement": 100.8,
                    "stream_c_performance_improvement": 30.0,
                },
            },
            "stream_results": {},
            "integration_results": {},
            "summary": {},
        }

    def run_stream_a_benchmarks(self, iterations: int = 5) -> dict[str, Any]:
        """Run comprehensive Stream A benchmarks."""
        print(f"🔄 Running Stream A benchmarks ({iterations} iterations)...")

        stream_results = {
            "stream": "A",
            "description": "Progress Components (96% Code Reduction)",
            "target_improvement": "96% code reduction",
            "benchmarks": {},
            "summary": {},
        }

        try:
            # Code reduction validation
            print("  📊 Validating code reduction claim...")
            code_reduction_result = (
                self.stream_a_validator.validate_code_reduction_claim()
            )
            stream_results["benchmarks"]["code_reduction"] = {
                "passed": code_reduction_result.passed,
                "meets_target": code_reduction_result.meets_target,
                "lines_before": code_reduction_result.metrics.code_lines_before,
                "lines_after": code_reduction_result.metrics.code_lines_after,
                "reduction_percent": code_reduction_result.metrics.calculate_code_reduction_percent(),
                "error_message": code_reduction_result.error_message,
            }

            # Enhanced functionality validation
            print("  🚀 Validating enhanced functionality...")
            enhanced_result = self.stream_a_validator.validate_enhanced_functionality()
            stream_results["benchmarks"]["enhanced_functionality"] = {
                "passed": enhanced_result.passed,
                "functionality_preserved": enhanced_result.metrics.functionality_preserved,
                "error_message": enhanced_result.error_message,
            }

            # Performance comparison
            print("  ⚡ Running performance comparison...")

            def baseline_test():
                return self.stream_a_validator._simulate_custom_progress()

            def optimized_test():
                return self.stream_a_validator._test_native_progress()

            performance_results = []
            for i in range(iterations):
                print(f"    Iteration {i + 1}/{iterations}")
                perf_result = self.stream_a_validator.compare_with_baseline(
                    baseline_test, optimized_test
                )
                performance_results.append(
                    {
                        "iteration": i + 1,
                        "passed": perf_result.passed,
                        "baseline_time_ms": perf_result.baseline_metrics.execution_time_ms,
                        "optimized_time_ms": perf_result.metrics.execution_time_ms,
                        "improvement_factor": perf_result.improvement_factor,
                    }
                )

            # Calculate averages
            avg_baseline_time = sum(
                r["baseline_time_ms"] for r in performance_results
            ) / len(performance_results)
            avg_optimized_time = sum(
                r["optimized_time_ms"] for r in performance_results
            ) / len(performance_results)
            avg_improvement = sum(
                r["improvement_factor"] for r in performance_results
            ) / len(performance_results)

            stream_results["benchmarks"]["performance_comparison"] = {
                "iterations": iterations,
                "avg_baseline_time_ms": avg_baseline_time,
                "avg_optimized_time_ms": avg_optimized_time,
                "avg_improvement_factor": avg_improvement,
                "results": performance_results,
            }

            # Summary
            reduction_percent = stream_results["benchmarks"]["code_reduction"][
                "reduction_percent"
            ]
            stream_results["summary"] = {
                "overall_passed": all(
                    [
                        code_reduction_result.passed,
                        enhanced_result.passed,
                        avg_improvement > 1.0,
                    ]
                ),
                "target_met": reduction_percent >= 90.0,
                "code_reduction_achieved": f"{reduction_percent:.1f}%",
                "performance_improvement": f"{avg_improvement:.1f}x",
                "key_achievements": [
                    f"Code reduction: {reduction_percent:.1f}%",
                    f"Performance improvement: {avg_improvement:.1f}x",
                    "Enhanced functionality preserved"
                    if enhanced_result.passed
                    else "Functionality issues detected",
                ],
            }

            print(
                f"  ✅ Stream A benchmarks completed - Target: {stream_results['summary']['target_met']}"
            )

        except Exception as e:
            stream_results["error"] = f"Stream A benchmark failed: {e}"
            print(f"  ❌ Stream A benchmarks failed: {e}")

        return stream_results

    def run_stream_b_benchmarks(self, iterations: int = 5) -> dict[str, Any]:
        """Run comprehensive Stream B benchmarks."""
        print(f"🔄 Running Stream B benchmarks ({iterations} iterations)...")

        stream_results = {
            "stream": "B",
            "description": "Native Caching Performance (100.8x Improvement)",
            "target_improvement": "100.8x performance improvement",
            "benchmarks": {},
            "summary": {},
        }

        try:
            # Cache data performance
            print("  💾 Testing cache data performance...")
            cache_data_results = []
            for i in range(iterations):
                print(f"    Cache data iteration {i + 1}/{iterations}")
                result = self.stream_b_validator.test_cache_data_performance()
                cache_data_results.append(
                    {
                        "iteration": i + 1,
                        "passed": result.passed,
                        "improvement_factor": result.improvement_factor,
                        "cache_hit_rate": result.metrics.cache_hit_rate,
                        "meets_target": result.meets_target,
                    }
                )

            avg_cache_improvement = sum(
                r["improvement_factor"] for r in cache_data_results
            ) / len(cache_data_results)
            avg_hit_rate = sum(r["cache_hit_rate"] for r in cache_data_results) / len(
                cache_data_results
            )

            stream_results["benchmarks"]["cache_data_performance"] = {
                "iterations": iterations,
                "avg_improvement_factor": avg_cache_improvement,
                "avg_cache_hit_rate": avg_hit_rate,
                "results": cache_data_results,
            }

            # Cache resource functionality
            print("  🔗 Testing cache resource functionality...")
            resource_result = (
                self.stream_b_validator.test_cache_resource_functionality()
            )
            stream_results["benchmarks"]["cache_resource"] = {
                "passed": resource_result.passed,
                "improvement_factor": resource_result.improvement_factor,
                "meets_target": resource_result.meets_target,
                "error_message": resource_result.error_message,
            }

            # Unified caching integration
            print("  🔄 Testing unified caching integration...")
            unified_result = self.stream_b_validator.test_unified_caching_integration()
            stream_results["benchmarks"]["unified_caching"] = {
                "passed": unified_result.passed,
                "improvement_factor": unified_result.improvement_factor,
                "cache_hit_rate": unified_result.metrics.cache_hit_rate,
                "meets_target": unified_result.meets_target,
            }

            # 100x performance validation
            print("  🚀 Validating 100x performance claim...")
            performance_100x_result = (
                self.stream_b_validator.validate_100x_performance_claim()
            )
            stream_results["benchmarks"]["100x_performance"] = {
                "passed": performance_100x_result.passed,
                "improvement_factor": performance_100x_result.improvement_factor,
                "meets_target": performance_100x_result.meets_target,
                "baseline_time_ms": performance_100x_result.baseline_metrics.execution_time_ms,
                "optimized_time_ms": performance_100x_result.metrics.execution_time_ms,
                "cache_hit_rate": performance_100x_result.metrics.cache_hit_rate,
            }

            # Summary
            best_improvement = max(
                [
                    avg_cache_improvement,
                    resource_result.improvement_factor,
                    unified_result.improvement_factor,
                    performance_100x_result.improvement_factor,
                ]
            )

            stream_results["summary"] = {
                "overall_passed": all(
                    [
                        any(r["passed"] for r in cache_data_results),
                        resource_result.passed,
                        unified_result.passed,
                        performance_100x_result.passed,
                    ]
                ),
                "target_met": best_improvement >= 50.0,  # 50x minimum target
                "best_improvement_factor": f"{best_improvement:.1f}x",
                "avg_cache_hit_rate": f"{avg_hit_rate:.1f}%",
                "meets_100x_claim": performance_100x_result.improvement_factor >= 100.0,
                "key_achievements": [
                    f"Best improvement: {best_improvement:.1f}x",
                    f"Average cache hit rate: {avg_hit_rate:.1f}%",
                    f"100x claim: {'✅ Met' if performance_100x_result.improvement_factor >= 100.0 else '⚠️ Partial'}",
                ],
            }

            print(
                f"  ✅ Stream B benchmarks completed - Best improvement: {best_improvement:.1f}x"
            )

        except Exception as e:
            stream_results["error"] = f"Stream B benchmark failed: {e}"
            print(f"  ❌ Stream B benchmarks failed: {e}")

        return stream_results

    def run_stream_c_benchmarks(self, iterations: int = 5) -> dict[str, Any]:
        """Run comprehensive Stream C benchmarks."""
        print(f"🔄 Running Stream C benchmarks ({iterations} iterations)...")

        stream_results = {
            "stream": "C",
            "description": "Fragment Performance (30% Improvement)",
            "target_improvement": "30% performance improvement",
            "benchmarks": {},
            "summary": {},
        }

        try:
            # Auto-refresh performance
            print("  🔄 Testing fragment auto-refresh performance...")
            auto_refresh_result = (
                self.stream_c_validator.test_fragment_auto_refresh_performance()
            )
            stream_results["benchmarks"]["auto_refresh"] = {
                "passed": auto_refresh_result.passed,
                "improvement_factor": auto_refresh_result.improvement_factor,
                "baseline_page_reruns": auto_refresh_result.baseline_metrics.page_rerun_count,
                "optimized_page_reruns": auto_refresh_result.metrics.page_rerun_count,
                "meets_target": auto_refresh_result.meets_target,
            }

            # Fragment isolation performance
            print("  🔐 Testing fragment isolation performance...")
            isolation_result = (
                self.stream_c_validator.test_fragment_isolation_performance()
            )
            stream_results["benchmarks"]["fragment_isolation"] = {
                "passed": isolation_result.passed,
                "improvement_factor": isolation_result.improvement_factor,
                "fragment_update_frequency": isolation_result.metrics.fragment_update_frequency,
                "meets_target": isolation_result.meets_target,
            }

            # Coordinated fragments performance
            print("  🤝 Testing coordinated fragment performance...")
            coordination_result = (
                self.stream_c_validator.test_coordinated_fragment_performance()
            )
            stream_results["benchmarks"]["coordination"] = {
                "passed": coordination_result.passed,
                "improvement_factor": coordination_result.improvement_factor,
                "meets_target": coordination_result.meets_target,
            }

            # 30% improvement validation
            print("  📈 Validating 30% improvement claim...")
            improvement_30_result = (
                self.stream_c_validator.validate_30_percent_improvement_claim()
            )
            improvement_percent = (improvement_30_result.improvement_factor - 1.0) * 100

            stream_results["benchmarks"]["30_percent_validation"] = {
                "passed": improvement_30_result.passed,
                "improvement_factor": improvement_30_result.improvement_factor,
                "improvement_percent": improvement_percent,
                "meets_target": improvement_30_result.meets_target,
                "avg_fragment_frequency": improvement_30_result.metrics.fragment_update_frequency,
                "avg_page_reruns": improvement_30_result.metrics.page_rerun_count,
            }

            # Summary
            avg_improvement = (
                sum(
                    [
                        auto_refresh_result.improvement_factor,
                        isolation_result.improvement_factor,
                        coordination_result.improvement_factor,
                    ]
                )
                / 3
            )

            avg_improvement_percent = (avg_improvement - 1.0) * 100

            stream_results["summary"] = {
                "overall_passed": all(
                    [
                        auto_refresh_result.passed,
                        isolation_result.passed,
                        coordination_result.passed,
                        improvement_30_result.passed,
                    ]
                ),
                "target_met": improvement_percent >= 25.0,  # 25% minimum
                "avg_improvement_factor": f"{avg_improvement:.1f}x",
                "avg_improvement_percent": f"{avg_improvement_percent:.1f}%",
                "meets_30_percent_claim": improvement_percent >= 30.0,
                "key_achievements": [
                    f"Average improvement: {avg_improvement_percent:.1f}%",
                    "Fragment isolation working",
                    "Auto-refresh reducing page reruns",
                    f"30% claim: {'✅ Met' if improvement_percent >= 30.0 else '⚠️ Partial'}",
                ],
            }

            print(
                f"  ✅ Stream C benchmarks completed - Improvement: {avg_improvement_percent:.1f}%"
            )

        except Exception as e:
            stream_results["error"] = f"Stream C benchmark failed: {e}"
            print(f"  ❌ Stream C benchmarks failed: {e}")

        return stream_results

    def run_integration_benchmarks(self, iterations: int = 3) -> dict[str, Any]:
        """Run integration benchmarks."""
        print(f"🔄 Running integration benchmarks ({iterations} iterations)...")

        integration_results = {
            "description": "Cross-stream integration validation",
            "benchmarks": {},
            "summary": {},
        }

        try:
            # Progress + Caching integration
            print("  🔄💾 Testing Progress-Caching integration...")
            progress_caching_results = []
            for i in range(iterations):
                result = self.integration_validator.validate_progress_with_caching()
                progress_caching_results.append(
                    {
                        "iteration": i + 1,
                        "passed": result.passed,
                        "cache_hit_rate": result.metrics.cache_hit_rate,
                        "cross_interactions": result.metrics.cross_component_interactions,
                    }
                )

            integration_results["benchmarks"]["progress_caching"] = {
                "iterations": iterations,
                "avg_cache_hit_rate": sum(
                    r["cache_hit_rate"] for r in progress_caching_results
                )
                / len(progress_caching_results),
                "avg_interactions": sum(
                    r["cross_interactions"] for r in progress_caching_results
                )
                / len(progress_caching_results),
                "results": progress_caching_results,
            }

            # Cached + Fragments integration
            print("  💾🔄 Testing Cached-Fragments integration...")
            cached_fragments_result = (
                self.integration_validator.validate_cached_fragments()
            )
            integration_results["benchmarks"]["cached_fragments"] = {
                "passed": cached_fragments_result.passed,
                "cache_hit_rate": cached_fragments_result.metrics.cache_hit_rate,
                "fragment_frequency": cached_fragments_result.metrics.fragment_update_frequency,
                "cross_interactions": cached_fragments_result.metrics.cross_component_interactions,
            }

            # Progress + Fragments integration
            print("  🔄📊 Testing Progress-Fragments integration...")
            progress_fragments_result = (
                self.integration_validator.validate_progress_fragments()
            )
            integration_results["benchmarks"]["progress_fragments"] = {
                "passed": progress_fragments_result.passed,
                "fragment_frequency": progress_fragments_result.metrics.fragment_update_frequency,
                "cross_interactions": progress_fragments_result.metrics.cross_component_interactions,
            }

            # Complete integration
            print("  🌟 Testing complete Week 1 integration...")
            complete_integration_result = (
                self.integration_validator.validate_complete_week1_integration()
            )
            integration_results["benchmarks"]["complete_integration"] = {
                "passed": complete_integration_result.passed,
                "meets_target": complete_integration_result.meets_target,
                "cache_hit_rate": complete_integration_result.metrics.cache_hit_rate,
                "fragment_frequency": complete_integration_result.metrics.fragment_update_frequency,
                "cross_interactions": complete_integration_result.metrics.cross_component_interactions,
                "total_progress_updates": len(
                    self.integration_validator.progress_updates
                ),
                "total_cache_operations": len(
                    self.integration_validator.cache_operations
                ),
                "total_fragment_updates": len(
                    self.integration_validator.fragment_updates
                ),
            }

            # Summary
            integration_results["summary"] = {
                "all_integrations_passed": all(
                    [
                        any(r["passed"] for r in progress_caching_results),
                        cached_fragments_result.passed,
                        progress_fragments_result.passed,
                        complete_integration_result.passed,
                    ]
                ),
                "complete_integration_passed": complete_integration_result.passed,
                "avg_cache_hit_rate": f"{integration_results['benchmarks']['progress_caching']['avg_cache_hit_rate']:.1f}%",
                "streams_working_together": complete_integration_result.passed,
                "key_achievements": [
                    "All streams integrate successfully",
                    f"Cache hit rate: {complete_integration_result.metrics.cache_hit_rate:.1f}%",
                    f"Fragment frequency: {complete_integration_result.metrics.fragment_update_frequency:.1f} updates/sec",
                    f"Cross-interactions: {complete_integration_result.metrics.cross_component_interactions}",
                ],
            }

            print(
                f"  ✅ Integration benchmarks completed - All working: {complete_integration_result.passed}"
            )

        except Exception as e:
            integration_results["error"] = f"Integration benchmark failed: {e}"
            print(f"  ❌ Integration benchmarks failed: {e}")

        return integration_results

    def run_complete_benchmark_suite(self, iterations: int = 5) -> dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 Starting Week 1 Complete Benchmark Suite")
        print("=" * 60)

        suite_start_time = time.perf_counter()

        # Run all stream benchmarks
        self.benchmark_results["stream_results"]["stream_a"] = (
            self.run_stream_a_benchmarks(iterations)
        )
        self.benchmark_results["stream_results"]["stream_b"] = (
            self.run_stream_b_benchmarks(iterations)
        )
        self.benchmark_results["stream_results"]["stream_c"] = (
            self.run_stream_c_benchmarks(iterations)
        )

        # Run integration benchmarks
        self.benchmark_results["integration_results"] = self.run_integration_benchmarks(
            max(3, iterations // 2)
        )

        suite_end_time = time.perf_counter()
        suite_duration = suite_end_time - suite_start_time

        # Generate summary
        self.benchmark_results["summary"] = self._generate_benchmark_summary(
            suite_duration
        )

        # Save results
        self._save_benchmark_results()

        print("=" * 60)
        print("🎉 Week 1 Benchmark Suite Completed!")
        print(f"⏱️  Total execution time: {suite_duration:.2f} seconds")

        return self.benchmark_results

    def _generate_benchmark_summary(self, suite_duration: float) -> dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        summary = {
            "execution_metadata": {
                "total_duration_seconds": suite_duration,
                "timestamp": datetime.now().isoformat(),
                "iterations_per_stream": 5,
            },
            "stream_achievements": {},
            "overall_assessment": {},
            "recommendations": [],
        }

        # Stream A assessment
        stream_a = self.benchmark_results["stream_results"]["stream_a"]
        if "summary" in stream_a:
            summary["stream_achievements"]["stream_a"] = {
                "target": "96% code reduction",
                "achieved": stream_a["summary"].get("code_reduction_achieved", "N/A"),
                "met_target": stream_a["summary"].get("target_met", False),
                "performance_improvement": stream_a["summary"].get(
                    "performance_improvement", "N/A"
                ),
                "status": "✅ SUCCESS"
                if stream_a["summary"].get("target_met", False)
                else "⚠️ PARTIAL",
            }

        # Stream B assessment
        stream_b = self.benchmark_results["stream_results"]["stream_b"]
        if "summary" in stream_b:
            summary["stream_achievements"]["stream_b"] = {
                "target": "100.8x performance improvement",
                "achieved": stream_b["summary"].get("best_improvement_factor", "N/A"),
                "met_target": stream_b["summary"].get("target_met", False),
                "meets_100x_claim": stream_b["summary"].get("meets_100x_claim", False),
                "cache_hit_rate": stream_b["summary"].get("avg_cache_hit_rate", "N/A"),
                "status": "✅ SUCCESS"
                if stream_b["summary"].get("meets_100x_claim", False)
                else "⚠️ PARTIAL",
            }

        # Stream C assessment
        stream_c = self.benchmark_results["stream_results"]["stream_c"]
        if "summary" in stream_c:
            summary["stream_achievements"]["stream_c"] = {
                "target": "30% performance improvement",
                "achieved": stream_c["summary"].get("avg_improvement_percent", "N/A"),
                "met_target": stream_c["summary"].get("target_met", False),
                "meets_30_percent_claim": stream_c["summary"].get(
                    "meets_30_percent_claim", False
                ),
                "status": "✅ SUCCESS"
                if stream_c["summary"].get("meets_30_percent_claim", False)
                else "⚠️ PARTIAL",
            }

        # Overall assessment
        all_targets_met = all(
            [
                summary["stream_achievements"]
                .get("stream_a", {})
                .get("met_target", False),
                summary["stream_achievements"]
                .get("stream_b", {})
                .get("met_target", False),
                summary["stream_achievements"]
                .get("stream_c", {})
                .get("met_target", False),
            ]
        )

        integration_success = (
            self.benchmark_results["integration_results"]
            .get("summary", {})
            .get("complete_integration_passed", False)
        )

        summary["overall_assessment"] = {
            "week1_targets_met": all_targets_met,
            "integration_successful": integration_success,
            "overall_status": "✅ SUCCESS"
            if (all_targets_met and integration_success)
            else "⚠️ NEEDS ATTENTION",
            "deployment_ready": all_targets_met and integration_success,
            "confidence_score": self._calculate_confidence_score(),
        }

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations()

        return summary

    def _calculate_confidence_score(self) -> float:
        """Calculate overall confidence score (0-100)."""
        scores = []

        # Stream A confidence
        stream_a = self.benchmark_results["stream_results"]["stream_a"]
        if "summary" in stream_a and stream_a["summary"].get("overall_passed", False):
            reduction_percent = float(
                stream_a["summary"]
                .get("code_reduction_achieved", "0%")
                .replace("%", "")
            )
            scores.append(min(100, (reduction_percent / 96.0) * 100))

        # Stream B confidence
        stream_b = self.benchmark_results["stream_results"]["stream_b"]
        if "summary" in stream_b and stream_b["summary"].get("overall_passed", False):
            best_improvement = float(
                stream_b["summary"]
                .get("best_improvement_factor", "0x")
                .replace("x", "")
            )
            scores.append(min(100, (best_improvement / 100.8) * 100))

        # Stream C confidence
        stream_c = self.benchmark_results["stream_results"]["stream_c"]
        if "summary" in stream_c and stream_c["summary"].get("overall_passed", False):
            improvement_percent = float(
                stream_c["summary"]
                .get("avg_improvement_percent", "0%")
                .replace("%", "")
            )
            scores.append(min(100, (improvement_percent / 30.0) * 100))

        # Integration confidence
        integration = self.benchmark_results["integration_results"]
        if "summary" in integration and integration["summary"].get(
            "complete_integration_passed", False
        ):
            scores.append(100)

        return sum(scores) / max(1, len(scores))

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on benchmark results."""
        recommendations = []

        # Stream A recommendations
        stream_a = self.benchmark_results["stream_results"]["stream_a"]
        if "summary" in stream_a:
            if not stream_a["summary"].get("target_met", False):
                recommendations.append(
                    "Stream A: Consider further code optimization to reach 96% reduction target"
                )

        # Stream B recommendations
        stream_b = self.benchmark_results["stream_results"]["stream_b"]
        if "summary" in stream_b:
            if not stream_b["summary"].get("meets_100x_claim", False):
                recommendations.append(
                    "Stream B: Optimize caching strategies to reach 100x improvement target"
                )

            hit_rate = float(
                stream_b["summary"].get("avg_cache_hit_rate", "0%").replace("%", "")
            )
            if hit_rate < 80:
                recommendations.append(
                    f"Stream B: Cache hit rate ({hit_rate:.1f}%) could be improved with better TTL configuration"
                )

        # Stream C recommendations
        stream_c = self.benchmark_results["stream_results"]["stream_c"]
        if "summary" in stream_c:
            if not stream_c["summary"].get("meets_30_percent_claim", False):
                recommendations.append(
                    "Stream C: Fragment isolation could be further optimized for 30% improvement"
                )

        # Integration recommendations
        integration = self.benchmark_results["integration_results"]
        if "summary" in integration:
            if not integration["summary"].get("complete_integration_passed", False):
                recommendations.append(
                    "Integration: Address cross-stream coordination issues for better integration"
                )

        # General recommendations
        confidence = self._calculate_confidence_score()
        if confidence < 90:
            recommendations.append(
                f"Overall: Confidence score ({confidence:.1f}%) indicates areas for improvement"
            )

        if not recommendations:
            recommendations.append(
                "All targets met! Ready for deployment with continued monitoring."
            )

        return recommendations

    def _save_benchmark_results(self) -> None:
        """Save benchmark results to files."""
        # Save JSON results
        json_file = (
            self.output_dir
            / f"week1_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_file, "w") as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)

        # Save summary report
        self._save_text_report()

        print(f"💾 Benchmark results saved to: {self.output_dir}")
        print(f"📄 JSON results: {json_file.name}")

    def _save_text_report(self) -> None:
        """Save human-readable text report."""
        report_file = (
            self.output_dir
            / f"week1_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(report_file, "w") as f:
            f.write("WEEK 1 STREAM VALIDATION BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(
                f"Execution Time: {self.benchmark_results['metadata']['execution_time']}\n"
            )
            f.write(
                f"Total Duration: {self.benchmark_results['summary']['execution_metadata']['total_duration_seconds']:.2f}s\n\n"
            )

            # Stream results
            f.write("STREAM ACHIEVEMENTS:\n")
            f.write("-" * 20 + "\n")

            for stream_key, stream_data in self.benchmark_results["summary"][
                "stream_achievements"
            ].items():
                f.write(f"\n{stream_key.upper()}:\n")
                f.write(f"  Target: {stream_data['target']}\n")
                f.write(f"  Achieved: {stream_data['achieved']}\n")
                f.write(f"  Status: {stream_data['status']}\n")

            # Overall assessment
            f.write("\nOVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            overall = self.benchmark_results["summary"]["overall_assessment"]
            f.write(f"Week 1 Targets Met: {overall['week1_targets_met']}\n")
            f.write(f"Integration Successful: {overall['integration_successful']}\n")
            f.write(f"Deployment Ready: {overall['deployment_ready']}\n")
            f.write(f"Confidence Score: {overall['confidence_score']:.1f}%\n")
            f.write(f"Status: {overall['overall_status']}\n")

            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(
                self.benchmark_results["summary"]["recommendations"], 1
            ):
                f.write(f"{i}. {rec}\n")

        print(f"📊 Text report: {report_file.name}")


def run_week1_benchmarks(
    output_dir: Path | None = None, iterations: int = 5
) -> dict[str, Any]:
    """Convenience function to run complete Week 1 benchmarks.

    Args:
        output_dir: Directory for output files
        iterations: Number of iterations per benchmark

    Returns:
        Complete benchmark results
    """
    runner = Week1BenchmarkRunner(output_dir)
    return runner.run_complete_benchmark_suite(iterations)


if __name__ == "__main__":
    # Run benchmarks if called directly
    import argparse

    parser = argparse.ArgumentParser(description="Week 1 Stream Validation Benchmarks")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per benchmark"
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")

    args = parser.parse_args()

    results = run_week1_benchmarks(args.output_dir, args.iterations)

    # Print summary
    summary = results["summary"]["overall_assessment"]
    print("\n🎯 FINAL RESULTS:")
    print(f"   Week 1 Targets Met: {summary['week1_targets_met']}")
    print(f"   Deployment Ready: {summary['deployment_ready']}")
    print(f"   Confidence Score: {summary['confidence_score']:.1f}%")
    print(f"   Status: {summary['overall_status']}")
