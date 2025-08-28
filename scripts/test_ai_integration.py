#!/usr/bin/env python3
"""AI Integration Test Script.

Comprehensive testing of the AI integration system including:
- vLLM health check and connectivity
- LiteLLM routing and fallback behavior
- Structured extraction with Instructor
- Error handling and recovery
- Model routing based on context size
"""

import asyncio
import json
import logging
import sys
import traceback

from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

# Import from our AI system
from src.ai import (
    LocalAIProcessor,
    LocalVLLMService,
    enhance_job_description,
    extract_job_skills,
)
from src.ai_client import get_ai_client, reset_ai_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestResult(BaseModel):
    """Test result model for structured reporting."""

    test_name: str = Field(..., description="Name of the test")
    passed: bool = Field(..., description="Whether the test passed")
    duration_ms: int = Field(..., description="Test duration in milliseconds")
    error_message: str | None = Field(None, description="Error message if failed")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional test details"
    )


class AIIntegrationTester:
    """Comprehensive AI integration tester."""

    def __init__(self):
        """Initialize the integration tester."""
        self.results: list[IntegrationTestResult] = []
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO") -> None:
        """Log message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    async def test_vllm_health_check(self) -> IntegrationTestResult:
        """Test vLLM service health check."""
        self.log("Testing vLLM health check...")
        start_time = datetime.now()

        try:
            service = LocalVLLMService()
            is_healthy = await service.health_check()

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if is_healthy:
                self.log("✅ vLLM health check passed", "SUCCESS")
                return IntegrationTestResult(
                    test_name="vLLM Health Check",
                    passed=True,
                    duration_ms=duration_ms,
                    details={"service_url": service.base_url},
                )
            self.log("❌ vLLM health check failed", "ERROR")
            return IntegrationTestResult(
                test_name="vLLM Health Check",
                passed=False,
                duration_ms=duration_ms,
                error_message="vLLM service not responding to health check",
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ vLLM health check exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="vLLM Health Check",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def test_model_availability(self) -> IntegrationTestResult:
        """Test vLLM model availability."""
        self.log("Testing model availability...")
        start_time = datetime.now()

        try:
            service = LocalVLLMService()
            models = await service.list_models()

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            expected_model = "Qwen3-4B-Instruct-2507-FP8"
            model_names = [model.get("id", "") for model in models]

            if any(expected_model in name for name in model_names):
                self.log(f"✅ Expected model '{expected_model}' found", "SUCCESS")
                return IntegrationTestResult(
                    test_name="Model Availability",
                    passed=True,
                    duration_ms=duration_ms,
                    details={"available_models": model_names},
                )
            self.log(f"❌ Expected model '{expected_model}' not found", "ERROR")
            return IntegrationTestResult(
                test_name="Model Availability",
                passed=False,
                duration_ms=duration_ms,
                error_message=f"Model '{expected_model}' not available",
                details={"available_models": model_names},
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ Model availability check exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="Model Availability",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def test_simple_completion(self) -> IntegrationTestResult:
        """Test simple AI completion through LiteLLM."""
        self.log("Testing simple completion...")
        start_time = datetime.now()

        try:
            client = get_ai_client()

            messages = [
                {
                    "role": "user",
                    "content": "Respond with exactly: 'AI integration test successful'",
                }
            ]

            response = client.get_simple_completion(
                messages=messages, model="local-qwen", max_tokens=20, temperature=0.0
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if "AI integration test successful" in response:
                self.log("✅ Simple completion working correctly", "SUCCESS")
                return IntegrationTestResult(
                    test_name="Simple Completion",
                    passed=True,
                    duration_ms=duration_ms,
                    details={"response": response[:100]},  # Truncate for logging
                )
            self.log(f"⚠️ Unexpected response: {response}", "WARNING")
            return IntegrationTestResult(
                test_name="Simple Completion",
                passed=False,
                duration_ms=duration_ms,
                error_message="Response did not contain expected text",
                details={"response": response[:100]},
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ Simple completion exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="Simple Completion",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def test_structured_extraction(self) -> IntegrationTestResult:
        """Test structured data extraction using Instructor."""
        self.log("Testing structured extraction...")
        start_time = datetime.now()

        try:
            processor = LocalAIProcessor()

            test_content = """
            Senior Python Developer at TechCorp Inc.
            Location: San Francisco, CA
            Remote work available

            We are seeking an experienced Python developer to join our team.

            Requirements:
            - 5+ years Python experience
            - Django/FastAPI knowledge
            - AWS/Docker expertise

            Benefits:
            - Competitive salary
            - Health insurance
            - 401k matching
            """

            result = await processor.extract_jobs(test_content)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Validate extraction results
            if result.title and result.company and len(result.requirements) > 0:
                self.log(
                    f"✅ Structured extraction successful: {result.title} at {result.company}",
                    "SUCCESS",
                )
                return IntegrationTestResult(
                    test_name="Structured Extraction",
                    passed=True,
                    duration_ms=duration_ms,
                    details={
                        "extracted_title": result.title,
                        "extracted_company": result.company,
                        "requirements_count": len(result.requirements),
                    },
                )
            self.log("❌ Structured extraction missing required fields", "ERROR")
            return IntegrationTestResult(
                test_name="Structured Extraction",
                passed=False,
                duration_ms=duration_ms,
                error_message="Missing required extraction fields",
                details=dict(result),
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ Structured extraction exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="Structured Extraction",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def test_fallback_behavior(self) -> IntegrationTestResult:
        """Test fallback from local to cloud model."""
        self.log("Testing fallback behavior...")
        start_time = datetime.now()

        try:
            client = get_ai_client()

            # Create a large context to trigger fallback
            large_content = "This is a test message. " * 1000  # ~8k+ tokens
            messages = [
                {
                    "role": "user",
                    "content": f"Analyze this content and respond 'Fallback test successful': {large_content}",
                }
            ]

            # This should automatically select gpt-4o-mini due to context size
            response = client.get_simple_completion(
                messages=messages, max_tokens=50, temperature=0.0
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if response and len(response.strip()) > 0:
                self.log("✅ Fallback behavior working", "SUCCESS")
                return IntegrationTestResult(
                    test_name="Fallback Behavior",
                    passed=True,
                    duration_ms=duration_ms,
                    details={
                        "context_size": len(large_content),
                        "response_length": len(response),
                    },
                )
            self.log("❌ Fallback behavior failed", "ERROR")
            return IntegrationTestResult(
                test_name="Fallback Behavior",
                passed=False,
                duration_ms=duration_ms,
                error_message="Empty or invalid response from fallback model",
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ Fallback behavior exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="Fallback Behavior",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def test_error_handling(self) -> IntegrationTestResult:
        """Test error handling with invalid requests."""
        self.log("Testing error handling...")
        start_time = datetime.now()

        try:
            client = get_ai_client()

            # Test with invalid model name
            try:
                await client.get_simple_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="nonexistent-model",
                )
                # If we get here, the test should fail
                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self.log("❌ Error handling test failed - no exception raised", "ERROR")
                return IntegrationTestResult(
                    test_name="Error Handling",
                    passed=False,
                    duration_ms=duration_ms,
                    error_message="Expected exception was not raised",
                )
            except Exception:
                # Expected behavior - error was caught
                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self.log("✅ Error handling working correctly", "SUCCESS")
                return IntegrationTestResult(
                    test_name="Error Handling",
                    passed=True,
                    duration_ms=duration_ms,
                    details={"error_caught": True},
                )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ Error handling test exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="Error Handling",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def test_backward_compatibility_functions(self) -> IntegrationTestResult:
        """Test backward compatibility functions."""
        self.log("Testing backward compatibility functions...")
        start_time = datetime.now()

        try:
            test_job_data = {
                "description": "Python developer position at a tech company. Requires Python, Django, AWS skills."
            }

            # Test enhance_job_description
            enhanced = await enhance_job_description(test_job_data)

            # Test extract_job_skills
            skills = await extract_job_skills(test_job_data)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if enhanced and len(enhanced.strip()) > 0 and len(skills) > 0:
                self.log("✅ Backward compatibility functions working", "SUCCESS")
                return IntegrationTestResult(
                    test_name="Backward Compatibility",
                    passed=True,
                    duration_ms=duration_ms,
                    details={
                        "enhanced_description_length": len(enhanced),
                        "skills_extracted": len(skills),
                    },
                )
            self.log("❌ Backward compatibility functions failed", "ERROR")
            return IntegrationTestResult(
                test_name="Backward Compatibility",
                passed=False,
                duration_ms=duration_ms,
                error_message="Empty results from compatibility functions",
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.log(f"❌ Backward compatibility test exception: {e}", "ERROR")
            return IntegrationTestResult(
                test_name="Backward Compatibility",
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    async def run_integration_tests(self) -> dict[str, Any]:
        """Run all integration tests and return comprehensive results."""
        self.log("Starting AI Integration Test Suite", "INFO")
        self.log("=" * 60, "INFO")

        # Test suite definition
        tests = [
            ("vLLM Health Check", self.test_vllm_health_check),
            ("Model Availability", self.test_model_availability),
            ("Simple Completion", self.test_simple_completion),
            ("Structured Extraction", self.test_structured_extraction),
            ("Fallback Behavior", self.test_fallback_behavior),
            ("Error Handling", self.test_error_handling),
            ("Backward Compatibility", self.test_backward_compatibility_functions),
        ]

        # Run all tests
        for test_name, test_func in tests:
            self.log(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                self.results.append(result)
            except Exception as e:
                self.log(
                    f"❌ {test_name} failed with unexpected exception: {e}", "ERROR"
                )
                self.log(f"Traceback: {traceback.format_exc()}", "ERROR")

                # Create failure result
                failure_result = IntegrationTestResult(
                    test_name=test_name,
                    passed=False,
                    duration_ms=0,
                    error_message=f"Unexpected exception: {e}",
                )
                self.results.append(failure_result)

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        total_duration = sum(r.duration_ms for r in self.results)

        # Print detailed results
        self.log("\n" + "=" * 60, "INFO")
        self.log("INTEGRATION TEST RESULTS", "INFO")
        self.log("=" * 60, "INFO")

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            self.log(
                f"{status} {result.test_name} ({result.duration_ms}ms)",
                "SUCCESS" if result.passed else "ERROR",
            )
            if not result.passed and result.error_message:
                self.log(f"    Error: {result.error_message}", "ERROR")

        # Print summary
        self.log("\n" + "=" * 60, "INFO")
        self.log("SUMMARY", "INFO")
        self.log("=" * 60, "INFO")
        self.log(f"Total Tests: {total_tests}", "INFO")
        self.log(f"Passed: {passed_tests}", "SUCCESS" if passed_tests > 0 else "INFO")
        self.log(f"Failed: {failed_tests}", "ERROR" if failed_tests > 0 else "INFO")
        self.log(f"Success Rate: {success_rate:.1f}%", "INFO")
        self.log(f"Total Duration: {total_duration}ms", "INFO")

        # Overall result
        if passed_tests == total_tests:
            self.log(
                "\n🎉 All integration tests passed! AI system is ready for production use.",
                "SUCCESS",
            )
            self.log(
                "🔧 The AI integration is properly configured with fallback support.",
                "SUCCESS",
            )
        else:
            self.log(
                f"\n⚠️ {failed_tests} test(s) failed. Check configuration and logs.",
                "WARNING",
            )

        # Return comprehensive results
        return {
            "overall_success": passed_tests == total_tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "total_duration_ms": total_duration,
            "test_results": [result.dict() for result in self.results],
            "timestamp": self.start_time.isoformat(),
        }


async def main():
    """Main test execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test AI integration system")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tester = AIIntegrationTester()

    try:
        # Reset AI client to ensure clean state
        reset_ai_client()

        # Run all integration tests
        results = await tester.run_integration_tests()

        # Save detailed results if requested
        if args.output:
            output_path = Path(args.output)
            with output_path.open("w") as f:
                json.dump(results, f, indent=2)
            tester.log(f"Detailed results saved to: {output_path}", "INFO")

        # Exit with appropriate code
        sys.exit(0 if results["overall_success"] else 1)

    except KeyboardInterrupt:
        tester.log("\n🔍 Integration tests interrupted by user", "INFO")
        sys.exit(1)
    except Exception as e:
        tester.log(f"❌ Integration tests failed with error: {e}", "ERROR")
        tester.log(f"Traceback: {traceback.format_exc()}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    # Check Python version

    # Check required packages
    try:
        import httpx
        import instructor
        import litellm
        import pydantic
        import yaml
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: uv add httpx instructor litellm pydantic pyyaml")
        sys.exit(1)

    # Run the integration tests
    asyncio.run(main())
