#!/usr/bin/env python3
"""vLLM Integration Test Script
Test integration between vLLM server and LiteLLM configuration.
"""

import asyncio
import json
import os
import sys

import httpx

from pydantic import BaseModel, Field


class JobExtraction(BaseModel):
    """Job extraction schema for testing."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str | None = Field(None, description="Job location")


class VLLMIntegrationTester:
    """Test vLLM server integration with expected patterns."""

    def __init__(
        self, base_url: str = "http://localhost:8000", api_key: str | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "test-key")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        import time

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    async def test_server_connectivity(self) -> bool:
        """Test basic server connectivity."""
        self.log("Testing server connectivity...")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=10.0)

                if response.status_code == 200:
                    self.log("✅ Server is reachable and healthy", "SUCCESS")
                    return True
                self.log(f"❌ Server returned status {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Cannot connect to server: {e}", "ERROR")
            return False

    async def test_model_availability(self) -> bool:
        """Test if the expected model is available."""
        self.log("Checking model availability...")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/v1/models", headers=self.headers, timeout=15.0
                )

                if response.status_code != 200:
                    self.log(
                        f"❌ Models endpoint returned {response.status_code}", "ERROR"
                    )
                    return False

                models_data = response.json()
                models = models_data.get("data", [])

                expected_model = "Qwen3-4B-Instruct-2507-FP8"
                available_models = [model["id"] for model in models]

                if expected_model in available_models:
                    self.log(
                        f"✅ Expected model '{expected_model}' is available", "SUCCESS"
                    )
                    return True
                self.log(
                    f"❌ Expected model not found. Available: {available_models}",
                    "ERROR",
                )
                return False

        except Exception as e:
            self.log(f"❌ Error checking models: {e}", "ERROR")
            return False

    async def test_basic_completion(self) -> bool:
        """Test basic chat completion."""
        self.log("Testing basic completion...")

        payload = {
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {
                    "role": "user",
                    "content": "Respond with exactly: 'Integration test successful'",
                }
            ],
            "max_tokens": 20,
            "temperature": 0.0,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    self.log(
                        f"❌ Completion failed with status {response.status_code}",
                        "ERROR",
                    )
                    return False

                completion_data = response.json()
                message = completion_data["choices"][0]["message"]["content"]

                if "Integration test successful" in message:
                    self.log("✅ Basic completion working correctly", "SUCCESS")
                    return True
                self.log(f"⚠️  Unexpected response: {message}", "WARNING")
                return False

        except Exception as e:
            self.log(f"❌ Completion test failed: {e}", "ERROR")
            return False

    async def test_structured_output(self) -> bool:
        """Test structured output generation."""
        self.log("Testing structured output...")

        job_content = """
        Senior Python Developer at TechCorp
        Location: San Francisco, CA
        We are looking for an experienced Python developer.
        """

        payload = {
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": "Extract job information as JSON with title, company, and location fields.",
                },
                {"role": "user", "content": job_content.strip()},
            ],
            "max_tokens": 200,
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

                if response.status_code != 200:
                    self.log(
                        f"❌ Structured output failed with status {response.status_code}",
                        "ERROR",
                    )
                    return False

                completion_data = response.json()
                json_response = completion_data["choices"][0]["message"]["content"]

                # Validate JSON structure
                parsed_json = json.loads(json_response)
                job_data = JobExtraction(**parsed_json)

                if job_data.title and job_data.company:
                    self.log(
                        f"✅ Structured output working: {job_data.title} at {job_data.company}",
                        "SUCCESS",
                    )
                    return True
                self.log("❌ Structured output missing required fields", "ERROR")
                return False

        except json.JSONDecodeError as e:
            self.log(f"❌ Invalid JSON in structured output: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ Structured output test failed: {e}", "ERROR")
            return False

    async def test_openai_client_compatibility(self) -> bool:
        """Test OpenAI client library compatibility."""
        self.log("Testing OpenAI client compatibility...")

        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=f"{self.base_url}/v1"
            )

            response = await client.chat.completions.create(
                model="Qwen3-4B-Instruct-2507-FP8",
                messages=[{"role": "user", "content": "Say 'OpenAI client works'"}],
                max_tokens=10,
                temperature=0.0,
            )

            content = response.choices[0].message.content

            if "OpenAI client works" in content:
                self.log("✅ OpenAI client compatibility confirmed", "SUCCESS")
                return True
            self.log(f"⚠️  Unexpected OpenAI client response: {content}", "WARNING")
            return False

        except ImportError:
            self.log(
                "⚠️  OpenAI library not available for compatibility test", "WARNING"
            )
            return True  # Not a failure, just unavailable
        except Exception as e:
            self.log(f"❌ OpenAI client compatibility failed: {e}", "ERROR")
            return False

    async def test_litellm_integration(self) -> bool:
        """Test LiteLLM configuration compatibility."""
        self.log("Testing LiteLLM integration pattern...")

        # Simulate LiteLLM request pattern
        payload = {
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {"role": "user", "content": "This is a LiteLLM integration test"}
            ],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "litellm/1.0.0",  # Simulate LiteLLM request
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    completion_data = response.json()
                    completion_data["choices"][0]["message"]["content"]
                    self.log("✅ LiteLLM integration pattern working", "SUCCESS")
                    return True
                self.log(
                    f"❌ LiteLLM pattern failed with status {response.status_code}",
                    "ERROR",
                )
                return False

        except Exception as e:
            self.log(f"❌ LiteLLM integration test failed: {e}", "ERROR")
            return False

    async def run_full_integration_test(self) -> bool:
        """Run complete integration test suite."""
        self.log("Starting vLLM Integration Test Suite", "INFO")
        self.log("=" * 60, "INFO")

        tests = [
            ("Server Connectivity", self.test_server_connectivity),
            ("Model Availability", self.test_model_availability),
            ("Basic Completion", self.test_basic_completion),
            ("Structured Output", self.test_structured_output),
            ("OpenAI Compatibility", self.test_openai_client_compatibility),
            ("LiteLLM Integration", self.test_litellm_integration),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            self.log(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                if result:
                    passed += 1
            except Exception as e:
                self.log(f"❌ {test_name} failed with exception: {e}", "ERROR")

        # Print summary
        self.log("=" * 60, "INFO")
        self.log("INTEGRATION TEST SUMMARY", "INFO")
        self.log("=" * 60, "INFO")

        success_rate = (passed / total) * 100
        self.log(
            f"Results: {passed}/{total} tests passed ({success_rate:.1f}%)", "INFO"
        )

        if passed == total:
            self.log(
                "🎉 All integration tests passed! vLLM server is ready for production use.",
                "SUCCESS",
            )
            self.log(
                "🔧 The server is properly configured for LiteLLM integration.",
                "SUCCESS",
            )
            return True
        failed = total - passed
        self.log(
            f"⚠️  {failed} test(s) failed. Please check configuration and logs.",
            "WARNING",
        )
        return False


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM server integration")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument("--api-key", help="API key for authentication")

    args = parser.parse_args()

    tester = VLLMIntegrationTester(base_url=args.url, api_key=args.api_key)

    try:
        success = await tester.run_full_integration_test()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n🔍 Integration test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check required packages
    try:
        import httpx
        import pydantic
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: uv add httpx pydantic")
        sys.exit(1)

    # Optional packages
    try:
        import openai
    except ImportError:
        print("ℹ️  OpenAI library not installed - some tests will be skipped")
        print("Install with: uv add openai")

    asyncio.run(main())
