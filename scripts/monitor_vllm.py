#!/usr/bin/env python3
"""vLLM Server Health Monitoring Script
Continuous monitoring of vLLM server health and performance.
"""

import asyncio
import json
import os
import time

from datetime import datetime

import httpx

from pydantic import BaseModel


class HealthMetrics(BaseModel):
    """Health metrics model."""

    timestamp: datetime
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    gpu_memory_used_mb: float | None = None
    gpu_utilization_percent: float | None = None
    active_requests: int | None = None
    queue_depth: int | None = None
    error_message: str | None = None


class VLLMMonitor:
    """vLLM server health monitor."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        check_interval: int = 30,
        history_limit: int = 100,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.check_interval = check_interval
        self.history: list[HealthMetrics] = []
        self.history_limit = history_limit
        self.headers = {}

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    async def check_health(self) -> HealthMetrics:
        """Perform comprehensive health check."""
        start_time = time.time()

        try:
            # Basic health check
            async with httpx.AsyncClient() as client:
                health_response = await client.get(
                    f"{self.base_url}/health", timeout=10.0
                )
                response_time_ms = (time.time() - start_time) * 1000

                if health_response.status_code != 200:
                    return HealthMetrics(
                        timestamp=datetime.now(),
                        status="unhealthy",
                        response_time_ms=response_time_ms,
                        error_message=f"Health check returned {health_response.status_code}",
                    )

                # Try to get detailed metrics
                gpu_memory = None
                gpu_util = None
                active_requests = None

                try:
                    # Try to get GPU metrics
                    gpu_memory, gpu_util = await self._get_gpu_metrics()
                except Exception as e:
                    self.log(f"GPU metrics unavailable: {e}", "DEBUG")

                try:
                    # Try to get server metrics
                    active_requests = await self._get_server_metrics()
                except Exception as e:
                    self.log(f"Server metrics unavailable: {e}", "DEBUG")

                # Determine status
                status = "healthy"
                if response_time_ms > 5000:  # 5 second threshold
                    status = "degraded"
                elif response_time_ms > 10000:  # 10 second threshold
                    status = "unhealthy"

                return HealthMetrics(
                    timestamp=datetime.now(),
                    status=status,
                    response_time_ms=response_time_ms,
                    gpu_memory_used_mb=gpu_memory,
                    gpu_utilization_percent=gpu_util,
                    active_requests=active_requests,
                )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthMetrics(
                timestamp=datetime.now(),
                status="unhealthy",
                response_time_ms=response_time_ms,
                error_message=str(e),
            )

    async def _get_gpu_metrics(self) -> tuple[float | None, float | None]:
        """Get GPU memory and utilization metrics."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used / (1024 * 1024)

            # Utilization info
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_info.gpu

            return memory_used_mb, gpu_util

        except ImportError:
            # pynvml not available
            return None, None
        except Exception:
            # Other GPU monitoring errors
            return None, None

    async def _get_server_metrics(self) -> int | None:
        """Get server-specific metrics."""
        try:
            async with httpx.AsyncClient() as client:
                # Try to get metrics from metrics endpoint
                response = await client.get(
                    f"{self.base_url.replace(':8000', ':8001')}/metrics", timeout=5.0
                )

                if response.status_code == 200:
                    # Parse Prometheus metrics for active requests
                    metrics_text = response.text
                    for line in metrics_text.split("\n"):
                        if "vllm_active_requests" in line and not line.startswith("#"):
                            # Extract value
                            value = float(line.split()[-1])
                            return int(value)

        except Exception:
            pass

        return None

    async def test_inference(self) -> bool:
        """Test basic inference capability."""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": "Qwen3-4B-Instruct-2507-FP8",
                    "messages": [
                        {"role": "user", "content": "Say 'OK' if you're working."}
                    ],
                    "max_tokens": 5,
                    "temperature": 0.0,
                }

                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0,
                )

                return response.status_code == 200

        except Exception:
            return False

    def add_to_history(self, metrics: HealthMetrics):
        """Add metrics to history and maintain limit."""
        self.history.append(metrics)
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def get_status_summary(self) -> dict:
        """Get current status summary."""
        if not self.history:
            return {"status": "unknown", "message": "No health checks performed yet"}

        latest = self.history[-1]
        recent_checks = self.history[-10:] if len(self.history) >= 10 else self.history

        # Calculate averages
        avg_response_time = sum(m.response_time_ms for m in recent_checks) / len(
            recent_checks
        )
        healthy_count = sum(1 for m in recent_checks if m.status == "healthy")
        success_rate = (healthy_count / len(recent_checks)) * 100

        return {
            "current_status": latest.status,
            "last_check": latest.timestamp.isoformat(),
            "response_time_ms": latest.response_time_ms,
            "avg_response_time_ms": avg_response_time,
            "success_rate_percent": success_rate,
            "gpu_memory_mb": latest.gpu_memory_used_mb,
            "gpu_utilization_percent": latest.gpu_utilization_percent,
            "active_requests": latest.active_requests,
            "total_checks": len(self.history),
            "error_message": latest.error_message,
        }

    def print_status(self, detailed: bool = False):
        """Print current status to console."""
        if not self.history:
            self.log("No health data available", "INFO")
            return

        summary = self.get_status_summary()
        latest = self.history[-1]

        # Status emoji
        status_emoji = {
            "healthy": "🟢",
            "degraded": "🟡",
            "unhealthy": "🔴",
            "unknown": "⚫",
        }

        emoji = status_emoji.get(summary["current_status"], "⚫")

        self.log(f"{emoji} Status: {summary['current_status'].upper()}", "STATUS")
        self.log(f"   Response Time: {latest.response_time_ms:.1f}ms", "STATUS")

        if latest.gpu_memory_used_mb:
            self.log(f"   GPU Memory: {latest.gpu_memory_used_mb:.1f}MB", "STATUS")

        if latest.gpu_utilization_percent:
            self.log(f"   GPU Utilization: {latest.gpu_utilization_percent}%", "STATUS")

        if latest.active_requests is not None:
            self.log(f"   Active Requests: {latest.active_requests}", "STATUS")

        if latest.error_message:
            self.log(f"   Error: {latest.error_message}", "ERROR")

        if detailed:
            self.log(
                f"   Success Rate: {summary['success_rate_percent']:.1f}%", "STATUS"
            )
            self.log(
                f"   Avg Response: {summary['avg_response_time_ms']:.1f}ms", "STATUS"
            )
            self.log(f"   Total Checks: {summary['total_checks']}", "STATUS")

    async def run_continuous_monitoring(self, duration_minutes: int | None = None):
        """Run continuous health monitoring."""
        self.log(f"Starting continuous monitoring (interval: {self.check_interval}s)")

        if duration_minutes:
            self.log(f"Will run for {duration_minutes} minutes")
            end_time = time.time() + (duration_minutes * 60)
        else:
            self.log("Running indefinitely (Ctrl+C to stop)")
            end_time = None

        try:
            while True:
                # Perform health check
                metrics = await self.check_health()
                self.add_to_history(metrics)

                # Print status
                self.print_status()

                # Test inference periodically (every 5 checks)
                if len(self.history) % 5 == 0:
                    inference_ok = await self.test_inference()
                    if not inference_ok:
                        self.log("⚠️  Inference test failed", "WARNING")

                # Check if we should stop
                if end_time and time.time() >= end_time:
                    break

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            self.log("Monitoring stopped by user", "INFO")

    async def save_metrics_report(self, filename: str | None = None):
        """Save metrics history to JSON file."""
        if not self.history:
            self.log("No metrics data to save", "WARNING")
            return

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vllm_metrics_{timestamp}.json"

        # Convert to JSON-serializable format
        data = {
            "monitoring_session": {
                "start_time": self.history[0].timestamp.isoformat(),
                "end_time": self.history[-1].timestamp.isoformat(),
                "total_checks": len(self.history),
                "check_interval_seconds": self.check_interval,
            },
            "summary": self.get_status_summary(),
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "status": m.status,
                    "response_time_ms": m.response_time_ms,
                    "gpu_memory_used_mb": m.gpu_memory_used_mb,
                    "gpu_utilization_percent": m.gpu_utilization_percent,
                    "active_requests": m.active_requests,
                    "error_message": m.error_message,
                }
                for m in self.history
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        self.log(f"Metrics report saved to {filename}", "INFO")


async def main():
    """Main monitoring function."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor vLLM server health")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--interval", type=int, default=30, help="Check interval in seconds"
    )
    parser.add_argument(
        "--duration", type=int, help="Duration in minutes (default: infinite)"
    )
    parser.add_argument("--save-report", help="Save report to file on exit")
    parser.add_argument("--once", action="store_true", help="Run single check and exit")

    args = parser.parse_args()

    monitor = VLLMMonitor(
        base_url=args.url, api_key=args.api_key, check_interval=args.interval
    )

    if args.once:
        # Single check
        monitor.log("Performing single health check...")
        metrics = await monitor.check_health()
        monitor.add_to_history(metrics)
        monitor.print_status(detailed=True)

        # Test inference
        inference_ok = await monitor.test_inference()
        if inference_ok:
            monitor.log("✅ Inference test passed", "SUCCESS")
        else:
            monitor.log("❌ Inference test failed", "ERROR")
    else:
        # Continuous monitoring
        try:
            await monitor.run_continuous_monitoring(args.duration)
        finally:
            if args.save_report:
                await monitor.save_metrics_report(args.save_report)


if __name__ == "__main__":
    try:
        import httpx
        import pydantic
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: uv add httpx pydantic")
        exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔍 Monitoring stopped")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        exit(1)
