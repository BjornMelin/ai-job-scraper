#!/usr/bin/env python3
"""vLLM Server Validation Script
Tests server connectivity, model loading, and OpenAI API compatibility
Based on ADR-010 specifications.
"""

import asyncio
import json
import os
import sys
import time

from typing import Any

import httpx

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of a validation test."""

    test_name: str
    passed: bool
    message: str
    duration_ms: float | None = None
    details: dict[str, Any] | None = None


class JobExtraction(BaseModel):
    """Test schema for structured output validation."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str | None = Field(None, description="Job location")
    description: str = Field(..., description="Job description")
    requirements: list[str] = Field(
        default_factory=list, description="Job requirements"
    )


class VLLMValidator:
    """Comprehensive vLLM server validator."""

    def __init__(
        self, base_url: str = "http://localhost:8000", api_key: str | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "test-key")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.results: list[ValidationResult] = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    async def run_validation(self) -> bool:
        """Run all validation tests."""
        self.log("Starting vLLM server validation...")

        tests = [
            self.test_health_endpoint,
            self.test_models_endpoint,
            self.test_basic_completion,
            self.test_structured_output,
            self.test_concurrent_requests,
            self.test_performance_benchmark,
            self.test_openai_compatibility,
        ]

        total_tests = len(tests)
        passed_tests = 0

        for test in tests:
            try:
                result = await test()
                self.results.append(result)

                if result.passed:
                    passed_tests += 1
                    self.log(f"✅ {result.test_name}: {result.message}", "PASS")
                else:
                    self.log(f"❌ {result.test_name}: {result.message}", "FAIL")

                if result.duration_ms:
                    self.log(f"   Duration: {result.duration_ms:.2f}ms")

            except Exception as e:
                error_result = ValidationResult(
                    test_name=test.__name__,
                    passed=False,
                    message=f"Test failed with exception: {e!s}",
                )
                self.results.append(error_result)
                self.log(f"❌ {test.__name__}: {e!s}", "ERROR")

        # Print summary
        self.print_summary(passed_tests, total_tests)

        return passed_tests == total_tests

    async def test_health_endpoint(self) -> ValidationResult:
        """Test server health endpoint."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=10.0)
                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return ValidationResult(
                        test_name="Health Endpoint",
                        passed=True,
                        message="Server is healthy",
                        duration_ms=duration_ms,
                    )
                return ValidationResult(
                    test_name="Health Endpoint",
                    passed=False,
                    message=f"Health check failed with status {response.status_code}",
                )

        except Exception as e:
            return ValidationResult(
                test_name="Health Endpoint",
                passed=False,
                message=f"Health check failed: {e!s}",
            )

    async def test_models_endpoint(self) -> ValidationResult:
        """Test models API endpoint."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/v1/models", headers=self.headers, timeout=15.0
                )
                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("data", [])

                    if models:
                        model_names = [model["id"] for model in models]
                        return ValidationResult(
                            test_name="Models Endpoint",
                            passed=True,
                            message=f"Found {len(models)} model(s): {', '.join(model_names)}",
                            duration_ms=duration_ms,
                            details={"models": model_names},
                        )
                    return ValidationResult(
                        test_name="Models Endpoint",
                        passed=False,
                        message="No models found in response",
                    )
                return ValidationResult(
                    test_name="Models Endpoint",
                    passed=False,
                    message=f"Models endpoint failed with status {response.status_code}",
                )

        except Exception as e:
            return ValidationResult(
                test_name="Models Endpoint",
                passed=False,
                message=f"Models endpoint failed: {e!s}",
            )

    async def test_basic_completion(self) -> ValidationResult:
        """Test basic chat completion."""
        start_time = time.time()

        payload = {
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in exactly 3 words."},
            ],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0,
                )
                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    completion_data = response.json()
                    message = completion_data["choices"][0]["message"]["content"]

                    return ValidationResult(
                        test_name="Basic Completion",
                        passed=True,
                        message=f"Completion successful: '{message.strip()}'",
                        duration_ms=duration_ms,
                        details={"response": message},
                    )
                return ValidationResult(
                    test_name="Basic Completion",
                    passed=False,
                    message=f"Completion failed with status {response.status_code}: {response.text}",
                )

        except Exception as e:
            return ValidationResult(
                test_name="Basic Completion",
                passed=False,
                message=f"Completion failed: {e!s}",
            )

    async def test_structured_output(self) -> ValidationResult:
        """Test structured output generation."""
        start_time = time.time()

        job_content = """
        Software Engineer at TechCorp
        Location: San Francisco, CA
        We are looking for a skilled software engineer to join our team.
        Requirements: Python, JavaScript, 3+ years experience
        Benefits: Health insurance, 401k, flexible hours
        """

        payload = {
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": "Extract job information as JSON with fields: title, company, location, description, requirements (array), benefits (array).",
                },
                {"role": "user", "content": job_content.strip()},
            ],
            "max_tokens": 500,
            "temperature": 0.0,
            "extra_body": {"guided_json": JobExtraction.model_json_schema()},
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=45.0,
                )
                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    completion_data = response.json()
                    json_response = completion_data["choices"][0]["message"]["content"]

                    try:
                        # Validate JSON structure
                        parsed_json = json.loads(json_response)
                        job_data = JobExtraction(**parsed_json)

                        return ValidationResult(
                            test_name="Structured Output",
                            passed=True,
                            message=f"Valid structured output: {job_data.title} at {job_data.company}",
                            duration_ms=duration_ms,
                            details={"extracted_job": parsed_json},
                        )

                    except (json.JSONDecodeError, ValueError) as e:
                        return ValidationResult(
                            test_name="Structured Output",
                            passed=False,
                            message=f"Invalid JSON response: {e!s}",
                        )
                else:
                    return ValidationResult(
                        test_name="Structured Output",
                        passed=False,
                        message=f"Structured output failed with status {response.status_code}",
                    )

        except Exception as e:
            return ValidationResult(
                test_name="Structured Output",
                passed=False,
                message=f"Structured output failed: {e!s}",
            )

    async def test_concurrent_requests(self) -> ValidationResult:
        """Test concurrent request handling."""
        start_time = time.time()

        async def make_request(client: httpx.AsyncClient, i: int):
            payload = {
                "model": "Qwen3-4B-Instruct-2507-FP8",
                "messages": [
                    {"role": "user", "content": f"Count to {i} in one sentence."}
                ],
                "max_tokens": 30,
                "temperature": 0.1,
            }

            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=20.0,
            )
            return response.status_code == 200

        try:
            async with httpx.AsyncClient() as client:
                tasks = [
                    make_request(client, i) for i in range(1, 6)
                ]  # 5 concurrent requests
                results = await asyncio.gather(*tasks, return_exceptions=True)

                duration_ms = (time.time() - start_time) * 1000

                successful = sum(1 for r in results if r is True)
                total = len(tasks)

                if successful >= 4:  # Allow 1 failure
                    return ValidationResult(
                        test_name="Concurrent Requests",
                        passed=True,
                        message=f"Handled {successful}/{total} concurrent requests successfully",
                        duration_ms=duration_ms,
                    )
                return ValidationResult(
                    test_name="Concurrent Requests",
                    passed=False,
                    message=f"Only {successful}/{total} concurrent requests succeeded",
                )

        except Exception as e:
            return ValidationResult(
                test_name="Concurrent Requests",
                passed=False,
                message=f"Concurrent requests test failed: {e!s}",
            )

    async def test_performance_benchmark(self) -> ValidationResult:
        """Test basic performance metrics."""
        start_time = time.time()

        payload = {
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a brief summary of machine learning in exactly 50 words.",
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        }

        try:
            latencies = []

            async with httpx.AsyncClient() as client:
                for _ in range(3):  # 3 requests for average
                    req_start = time.time()
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=30.0,
                    )
                    req_duration = (time.time() - req_start) * 1000

                    if response.status_code == 200:
                        latencies.append(req_duration)

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                total_duration = (time.time() - start_time) * 1000

                return ValidationResult(
                    test_name="Performance Benchmark",
                    passed=True,
                    message=f"Average latency: {avg_latency:.2f}ms",
                    duration_ms=total_duration,
                    details={"latencies": latencies, "average": avg_latency},
                )
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                message="No successful requests for performance measurement",
            )

        except Exception as e:
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                message=f"Performance test failed: {e!s}",
            )

    async def test_openai_compatibility(self) -> ValidationResult:
        """Test OpenAI API compatibility."""
        start_time = time.time()

        try:
            # Test with OpenAI client format
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=f"{self.base_url}/v1"
            )

            response = await client.chat.completions.create(
                model="Qwen3-4B-Instruct-2507-FP8",
                messages=[
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer with just the number.",
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )

            duration_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content

            return ValidationResult(
                test_name="OpenAI Compatibility",
                passed=True,
                message=f"OpenAI client works: '{content.strip()}'",
                duration_ms=duration_ms,
                details={"response": content},
            )

        except ImportError:
            return ValidationResult(
                test_name="OpenAI Compatibility",
                passed=False,
                message="OpenAI library not available for compatibility test",
            )
        except Exception as e:
            return ValidationResult(
                test_name="OpenAI Compatibility",
                passed=False,
                message=f"OpenAI compatibility test failed: {e!s}",
            )

    def print_summary(self, passed: int, total: int):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status:<10} {result.test_name:<25} {result.message}")

        print("=" * 60)
        success_rate = (passed / total) * 100
        print(f"Results: {passed}/{total} tests passed ({success_rate:.1f}%)")

        if passed == total:
            print("🎉 All tests passed! vLLM server is ready for production.")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Check logs and configuration.")

        print("=" * 60)


async def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate vLLM server deployment")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument(
        "--timeout", type=int, default=60, help="Overall timeout in seconds"
    )

    args = parser.parse_args()

    validator = VLLMValidator(base_url=args.url, api_key=args.api_key)

    try:
        # Add overall timeout
        success = await asyncio.wait_for(
            validator.run_validation(), timeout=args.timeout
        )

        sys.exit(0 if success else 1)

    except TimeoutError:
        print(f"❌ Validation timed out after {args.timeout} seconds")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🔍 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Install required packages if missing
    try:
        import httpx
        import openai
        import pydantic
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: uv add httpx openai pydantic")
        sys.exit(1)

    asyncio.run(main())
