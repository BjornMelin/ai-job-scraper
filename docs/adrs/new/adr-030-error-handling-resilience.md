# ADR-030: Error Handling & Resilience Strategy

## Status

**Decided** - August 2025

## Context

The AI Job Scraper architecture involves complex interactions between local LLM inference, web scraping, vector databases, and cloud fallback systems. With the hardware constraints of RTX 4090 laptop (16GB VRAM) and single active model architecture (ADR-029), robust error handling and resilience strategies are critical for production deployment.

### Error Categories Identified

#### 1. Hardware-Related Errors

- **VRAM Overflow**: Model loading exceeds 16GB limit
- **Thermal Throttling**: GPU temperature exceeds 87°C
- **Power Limiting**: Sustained load > 120W on laptop
- **CUDA Out of Memory**: Fragmented VRAM or memory leaks

#### 2. Model Loading/Switching Errors  

- **Model Download Failures**: Network issues, corrupted files
- **Quantization Errors**: AWQ/GPTQ loading failures
- **Model Switch Timeout**: Taking > 90 seconds to switch
- **vLLM Initialization Failures**: Engine startup problems

#### 3. Inference Errors

- **Generation Timeout**: Model taking too long to respond
- **Invalid Output**: Non-JSON or malformed responses
- **Context Length Exceeded**: Input longer than model's max context
- **Rate Limiting**: Too many concurrent requests

#### 4. Network/External Service Errors

- **Cloud API Failures**: GPT-5 API downtime or rate limits
- **Scraping Failures**: Target sites blocking requests
- **Vector DB Connectivity**: Qdrant connection issues
- **Redis Cache Failures**: Cache service unavailable

## Decision

### Multi-Layer Resilience Strategy

**Primary Principle**: Graceful degradation with minimal user impact

### 1. Hardware Protection Layer

```python
import torch
import psutil
import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

class HardwareProtectionManager:
    """Protect RTX 4090 laptop from hardware-related failures."""
    
    def __init__(self):
        self.vram_limit_gb = 15.0  # Leave 1GB buffer of 16GB
        self.temp_limit_c = 85     # Thermal protection
        self.power_limit_w = 120   # Sustained power limit
        self.monitoring_active = True
        
    @contextmanager
    def hardware_protection(self, operation_name: str):
        """Context manager for hardware-protected operations."""
        
        # Pre-operation checks
        if not self._check_hardware_safety():
            raise HardwareProtectionError(
                f"Hardware not safe for {operation_name}"
            )
        
        start_time = time.time()
        try:
            # Monitor during operation
            self._start_monitoring()
            yield
            
        except torch.cuda.OutOfMemoryError as e:
            self._handle_vram_overflow(e)
            raise VRAMOverflowError(f"VRAM overflow during {operation_name}")
            
        except Exception as e:
            logging.error(f"Hardware error in {operation_name}: {e}")
            raise
            
        finally:
            operation_time = time.time() - start_time
            self._log_operation_metrics(operation_name, operation_time)
            self._stop_monitoring()
    
    def _check_hardware_safety(self) -> bool:
        """Check if hardware is in safe state for operations."""
        
        try:
            # VRAM check
            vram_free = self._get_free_vram_gb()
            if vram_free < 2.0:  # Need 2GB minimum for safety
                logging.warning(f"Low VRAM: {vram_free:.1f}GB free")
                return False
            
            # Temperature check
            temp = self._get_gpu_temperature()
            if temp > self.temp_limit_c:
                logging.warning(f"High temperature: {temp}°C")
                return False
            
            # Power check
            power = self._get_gpu_power_draw()
            if power > self.power_limit_w:
                logging.warning(f"High power draw: {power}W")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Hardware check failed: {e}")
            return False
    
    def _handle_vram_overflow(self, error: torch.cuda.OutOfMemoryError):
        """Handle VRAM overflow with emergency cleanup."""
        
        logging.critical("VRAM overflow detected - performing emergency cleanup")
        
        try:
            # Emergency cleanup sequence
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Wait for cleanup
            time.sleep(3.0)
            
            # Verify cleanup
            vram_free = self._get_free_vram_gb()
            logging.info(f"Post-cleanup VRAM: {vram_free:.1f}GB free")
            
        except Exception as cleanup_error:
            logging.critical(f"Emergency cleanup failed: {cleanup_error}")

class HardwareProtectionError(Exception):
    """Raised when hardware is not safe for operation."""
    pass

class VRAMOverflowError(Exception):
    """Raised when VRAM overflow occurs."""
    pass
```

### 2. Model Management Resilience

```python
import asyncio
from enum import Enum
from typing import List, Optional
import traceback

class ModelSwitchStrategy(Enum):
    """Strategies for handling model switch failures."""
    RETRY = "retry"
    FALLBACK_SMALLER = "fallback_smaller" 
    FALLBACK_CLOUD = "fallback_cloud"
    EMERGENCY_RESTART = "emergency_restart"

class ResilientModelManager:
    """Model manager with comprehensive error handling."""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 5.0  # seconds
        self.fallback_models = [
            "qwen3-8b-base",
            "qwen3-4b-thinking-2507", 
            "qwen3-4b-instruct-2507"
        ]
        self.hardware_protection = HardwareProtectionManager()
        self.error_history = []
        
    async def safe_model_switch(
        self,
        target_model: str,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Switch models with comprehensive error handling."""
        
        for attempt in range(max_attempts):
            try:
                with self.hardware_protection.hardware_protection("model_switch"):
                    result = await self._attempt_model_switch(target_model)
                    
                    if result["success"]:
                        self._record_success(target_model, attempt + 1)
                        return result
                        
            except VRAMOverflowError as e:
                logging.error(f"VRAM overflow on attempt {attempt + 1}: {e}")
                strategy = self._determine_fallback_strategy("vram_overflow")
                
                if strategy == ModelSwitchStrategy.FALLBACK_SMALLER:
                    smaller_model = self._get_smaller_model(target_model)
                    if smaller_model:
                        logging.info(f"Falling back to smaller model: {smaller_model}")
                        return await self.safe_model_switch(smaller_model, max_attempts=1)
                
            except HardwareProtectionError as e:
                logging.error(f"Hardware protection triggered: {e}")
                await self._wait_for_hardware_recovery()
                
            except ModelLoadingError as e:
                logging.error(f"Model loading failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return await self._final_fallback_strategy(target_model, e)
                    
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                self._record_error(target_model, attempt + 1, e)
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
        
        # All attempts failed
        return await self._handle_complete_failure(target_model)
    
    async def _final_fallback_strategy(
        self,
        failed_model: str, 
        error: Exception
    ) -> Dict[str, Any]:
        """Handle complete local model failure."""
        
        logging.critical(f"All local model attempts failed for {failed_model}")
        
        # Try emergency fallback to smallest model
        emergency_model = "qwen3-4b-instruct-2507"
        if failed_model != emergency_model:
            try:
                return await self.safe_model_switch(emergency_model, max_attempts=1)
            except Exception as e:
                logging.critical(f"Emergency model failed: {e}")
        
        # Ultimate fallback to cloud
        return {
            "success": False,
            "active_model": None,
            "fallback_to_cloud": True,
            "error": str(error),
            "strategy": "cloud_fallback"
        }
    
    def _get_smaller_model(self, current_model: str) -> Optional[str]:
        """Get next smaller model in hierarchy."""
        
        model_hierarchy = [
            "qwen3-30b-a3b-instruct-2507",
            "qwen3-14b", 
            "qwen3-8b",
            "qwen3-4b-thinking-2507",
            "qwen3-4b-instruct-2507"
        ]
        
        try:
            current_index = model_hierarchy.index(current_model)
            if current_index < len(model_hierarchy) - 1:
                return model_hierarchy[current_index + 1]
        except ValueError:
            pass
            
        return "qwen3-4b-instruct-2507"  # Smallest fallback
    
    async def _wait_for_hardware_recovery(self, max_wait: float = 30.0):
        """Wait for hardware to return to safe operating conditions."""
        
        logging.info("Waiting for hardware recovery...")
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            if self.hardware_protection._check_hardware_safety():
                logging.info("Hardware recovered")
                return
                
            await asyncio.sleep(2.0)
        
        logging.warning("Hardware recovery timeout")
        raise HardwareRecoveryTimeoutError("Hardware did not recover in time")

class ModelLoadingError(Exception):
    """Raised when model loading fails."""
    pass

class HardwareRecoveryTimeoutError(Exception):
    """Raised when hardware doesn't recover in time."""
    pass
```

### 3. Inference Error Handling

```python
import json
import re
from typing import Any, Dict, Optional
import asyncio

class InferenceErrorHandler:
    """Handle inference-related errors with graceful recovery."""
    
    def __init__(self):
        self.timeout_seconds = 60
        self.max_retries = 3
        self.output_validators = {
            "json": self._validate_json_output,
            "structured": self._validate_structured_output
        }
        
    async def safe_inference(
        self,
        model: Any,
        prompt: str,
        expected_format: str = "json",
        max_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Perform inference with comprehensive error handling."""
        
        for attempt in range(self.max_retries):
            try:
                # Timeout protection
                result = await asyncio.wait_for(
                    self._attempt_inference(
                        model, prompt, max_tokens, temperature
                    ),
                    timeout=self.timeout_seconds
                )
                
                # Validate output format
                if self._validate_output(result, expected_format):
                    return {
                        "success": True,
                        "result": result,
                        "attempts": attempt + 1
                    }
                else:
                    logging.warning(f"Invalid output format on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        continue
                        
            except asyncio.TimeoutError:
                logging.error(f"Inference timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    # Reduce complexity for retry
                    max_tokens = min(max_tokens, 1024)
                    temperature = min(temperature + 0.1, 0.5)
                    continue
                    
            except Exception as e:
                logging.error(f"Inference error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        # All attempts failed
        return await self._handle_inference_failure(prompt, expected_format)
    
    async def _handle_inference_failure(
        self,
        prompt: str,
        expected_format: str
    ) -> Dict[str, Any]:
        """Handle complete inference failure."""
        
        logging.critical("All inference attempts failed")
        
        # Try with simplified prompt
        simplified_prompt = self._simplify_prompt(prompt)
        if simplified_prompt != prompt:
            logging.info("Attempting with simplified prompt")
            return await self.safe_inference(
                model=None,  # Will trigger cloud fallback
                prompt=simplified_prompt,
                expected_format=expected_format,
                max_tokens=512
            )
        
        # Return structured failure response
        return {
            "success": False,
            "result": None,
            "error": "inference_failure",
            "fallback_required": True
        }
    
    def _validate_output(self, output: str, expected_format: str) -> bool:
        """Validate output format."""
        
        validator = self.output_validators.get(expected_format)
        if validator:
            return validator(output)
        
        return True  # No validator, assume valid
    
    def _validate_json_output(self, output: str) -> bool:
        """Validate JSON output format."""
        
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                try:
                    json.loads(json_match.group())
                    return True
                except json.JSONDecodeError:
                    pass
        
        return False
```

### 4. Cloud Fallback Strategy

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class CloudFallbackManager:
    """Manage cloud API fallback with resilience."""
    
    def __init__(self):
        self.api_endpoints = {
            "gpt-5": "https://api.openai.com/v1/chat/completions",
            "gpt-5-mini": "https://api.openai.com/v1/chat/completions",
            "gpt-5-nano": "https://api.openai.com/v1/chat/completions"
        }
        self.rate_limits = {
            "gpt-5": {"requests_per_minute": 60, "tokens_per_minute": 300000},
            "gpt-5-mini": {"requests_per_minute": 120, "tokens_per_minute": 500000},
            "gpt-5-nano": {"requests_per_minute": 200, "tokens_per_minute": 1000000}
        }
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def cloud_inference(
        self,
        prompt: str,
        model_preference: str = "gpt-5-nano",
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Perform cloud inference with retry logic."""
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.api_endpoints[model_preference],
                    headers={"Authorization": f"Bearer {self._get_api_key()}"},
                    json={
                        "model": model_preference,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.1
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "result": response.json()["choices"][0]["message"]["content"],
                        "model": model_preference,
                        "source": "cloud"
                    }
                elif response.status_code == 429:  # Rate limited
                    raise CloudRateLimitError("API rate limit exceeded")
                else:
                    raise CloudAPIError(f"API error: {response.status_code}")
                    
            except httpx.TimeoutException:
                raise CloudTimeoutError("Cloud API timeout")
            except Exception as e:
                logging.error(f"Cloud inference failed: {e}")
                raise
```

### 5. Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, bypass
    HALF_OPEN = "half_open" # Testing recovery

class CircuitBreaker:
    """Circuit breaker for external dependencies."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitOpenError("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logging.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
```

## Implementation Plan

### Phase 1: Core Error Handling (Week 1)

1. Implement `HardwareProtectionManager`
2. Create `ResilientModelManager` with fallback strategies  
3. Setup basic logging and monitoring
4. Test VRAM overflow scenarios

### Phase 2: Inference Resilience (Week 1)

1. Implement `InferenceErrorHandler`
2. Create output validation systems
3. Setup timeout and retry mechanisms
4. Test with various failure modes

### Phase 3: Cloud Integration (Week 2)

1. Implement `CloudFallbackManager`
2. Setup circuit breakers for external services
3. Create rate limit handling
4. Integration testing

### Phase 4: Production Hardening (Week 2)

1. Comprehensive logging and alerting
2. Performance monitoring
3. Error analytics dashboard
4. Load testing and failure simulation

## Error Recovery Strategies

### Recovery Hierarchy

1. **Hardware Issues**:
   - Thermal → Reduce batch size, wait for cooling
   - VRAM → Emergency cleanup, switch to smaller model  
   - Power → Lower GPU utilization, switch to efficient model

2. **Model Issues**:
   - Loading failure → Retry with delay → Fallback to smaller model
   - Switch timeout → Cancel and retry → Emergency restart
   - Inference failure → Simplify prompt → Cloud fallback

3. **Network Issues**:
   - API timeout → Retry with backoff → Switch API endpoint
   - Rate limiting → Wait and retry → Use different model tier
   - Complete failure → Circuit breaker → Local-only mode

## Monitoring and Alerting

```yaml
# error_monitoring.yaml
error_tracking:
  log_level: INFO
  structured_logging: true
  error_aggregation: true
  
  categories:
    critical:
      - vram_overflow
      - hardware_failure
      - complete_model_failure
      
    warning:
      - model_switch_timeout
      - inference_timeout
      - cloud_fallback_triggered
      
    info:
      - successful_recovery
      - fallback_completed
      - circuit_breaker_state_change

alerting:
  immediate:
    - critical_errors
    - hardware_protection_triggered
    - emergency_fallback_activated
    
  daily_summary:
    - error_counts_by_type
    - recovery_success_rates
    - fallback_usage_statistics
```

## Performance Impact

### Error Handling Overhead

- **Hardware monitoring**: ~2% CPU overhead
- **Model switching safety checks**: ~5-10 seconds per switch
- **Inference validation**: ~10-50ms per request
- **Cloud fallback**: +2-5 seconds latency

### Recovery Times

- **VRAM overflow recovery**: 5-15 seconds
- **Model switch fallback**: 30-90 seconds
- **Cloud fallback activation**: 2-10 seconds
- **Circuit breaker recovery**: 60+ seconds

## Consequences

### Positive

- ✅ **System stability** - Graceful handling of hardware constraints
- ✅ **Data protection** - Prevents model/system crashes
- ✅ **User experience** - Transparent error recovery
- ✅ **Production readiness** - Comprehensive failure handling
- ✅ **Cost control** - Smart cloud fallback minimizes API usage

### Negative

- ❌ **Added complexity** - More code paths and edge cases
- ❌ **Performance overhead** - Monitoring and validation costs
- ❌ **Recovery latency** - Time to detect and recover from failures
- ❌ **Storage overhead** - Error logging and metrics storage

## Related Decisions

- **ADR-029**: Hardware-Aware Model Management (foundation for error handling)
- **ADR-020**: Hybrid LLM Strategy (cloud fallback implementation)
- **ADR-027**: Final Inference Stack (vLLM error scenarios)
- **ADR-032**: Monitoring & Observability (error tracking and alerting)

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [PyTorch CUDA Error Handling](https://pytorch.org/docs/stable/notes/cuda.html#error-checking)
- [vLLM Error Handling](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html)

## Review

- **Date**: August 2025
- **Reviewed by**: AI Engineering Team
- **Next Review**: October 2025
- **Implementation Priority**: **HIGH** - Critical for production deployment
