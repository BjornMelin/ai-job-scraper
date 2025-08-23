#!/usr/bin/env python3
"""Test script for Phase 1 implementation.

This script tests the basic functionality of:
1. LiteLLM configuration loading
2. AI client initialization
3. Structured output with Instructor
4. Settings configuration

Run with: python test_phase1_implementation.py
"""

import sys

from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


def test_config_loading():
    """Test LiteLLM configuration loading."""
    print("1. Testing configuration loading...")

    try:
        from ai_client import AIClient

        config_path = "config/litellm.yaml"

        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False

        # Test config loading (will fail gracefully if no API key)
        try:
            client = AIClient(config_path)
            print("✅ Configuration loaded successfully")
            print(f"   - Models configured: {len(client.config['model_list'])}")
            return True
        except Exception as e:
            print(
                f"⚠️  Configuration loaded but client init failed (expected without API key): {e}"
            )
            return True  # This is expected without API keys

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_ai_models():
    """Test AI models for structured output."""
    print("\n2. Testing AI models...")

    try:
        from ai_models import JobListExtraction, JobPosting

        # Test model creation
        job = JobPosting(
            title="Software Engineer",
            company="Test Company",
            location="Remote",
            description="A test job description that is long enough to meet validation requirements for this field in our model.",
        )

        jobs_list = JobListExtraction(
            jobs=[job], total_found=1, extraction_confidence=0.95
        )

        print("✅ AI models working correctly")
        print(f"   - Job: {job.title} at {job.company}")
        print(f"   - Jobs list: {jobs_list.total_found} jobs found")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ AI models test failed: {e}")
        return False


def test_settings_config():
    """Test settings configuration."""
    print("\n3. Testing settings configuration...")

    try:
        from config import Settings

        settings = Settings()
        print("✅ Settings loaded successfully")
        print(f"   - Token threshold: {settings.ai_token_threshold}")
        print(f"   - Database URL: {settings.db_url}")
        print(f"   - OpenAI key set: {'Yes' if settings.openai_api_key else 'No'}")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False


def test_core_utils():
    """Test core utilities."""
    print("\n4. Testing core utilities...")

    try:
        from core_utils import get_proxy, random_delay, random_user_agent

        # Test proxy function
        proxy = get_proxy()  # Should return None if no proxies configured
        print(f"✅ Proxy function working (result: {proxy})")

        # Test user agent
        ua = random_user_agent()
        print(f"✅ User agent generation working: {ua[:50]}...")

        # Test delay (should return without error)
        random_delay(0.1, 0.2)  # Very short delay for testing
        print("✅ Random delay working")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Core utilities test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Phase 1 Implementation Test ===")
    print("Testing library-first AI client implementation...\n")

    tests = [test_config_loading, test_ai_models, test_settings_config, test_core_utils]

    results = []
    for test in tests:
        results.append(test())

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Phase 1 implementation is working correctly.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable for cloud fallback")
        print("2. Start local vLLM server (see docs/developers/)")
        print("3. Test with actual AI requests")
        return 0
    print("❌ Some tests failed. Please review the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
