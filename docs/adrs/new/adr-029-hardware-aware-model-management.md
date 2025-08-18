# ADR-029: Hardware-Aware Model Management for RTX 4090 Laptop

## Status

**Decided** - August 2025

## Context

After comprehensive analysis of RTX 4090 laptop constraints (16GB VRAM), the current architecture's assumption of running multiple large LLMs simultaneously is **physically impossible**. This ADR addresses the critical gap in hardware-aware model management and establishes a **Single Active Model Architecture**.

### Hardware Reality Check

**RTX 4090 Laptop Constraints:**

- **Total VRAM**: 16GB GDDR6X
- **System Overhead**: ~1-2GB (OS, drivers, other processes)
- **Available for Models**: ~14-15GB maximum
- **Memory Bandwidth**: 576 GB/s (43% less than desktop RTX 4090)
- **Thermal Limit**: 87°C throttle, 120W sustained power

**Model Memory Requirements (with quantization):**

- **Qwen3-30B-A3B** (AWQ-4bit): ~15.5GB VRAM + overhead = **EXCEEDS 16GB**
- **Qwen3-14B** (AWQ-4bit): ~8.0GB VRAM ✓
- **Qwen3-8B** (AWQ-4bit): ~6.0GB VRAM ✓  
- **Qwen3-4B-Instruct-2507** (FP16): ~7.8GB VRAM ✓
- **Qwen3-4B-Thinking-2507** (Q5_K_M): ~4.5GB VRAM ✓

### Current Architecture Problems

1. **VRAM Overflow**: Multiple models loaded = 15.5GB + 8GB + overhead > 16GB = **SYSTEM CRASH**
2. **No Model Switching**: Current ADRs assume persistent model loading
3. **No Hardware Monitoring**: No thermal or VRAM monitoring
4. **No Graceful Degradation**: No fallback strategy for memory pressure

## Decision

### Single Active Model Architecture

**Primary Strategy**: Load only ONE model at a time with intelligent switching based on task requirements.

### Model Selection Strategy

```python
class ModelPriority(Enum):
    """Model selection based on capability and VRAM efficiency."""
    
    # Primary models (can fit with overhead)
    FAST_GENERAL = "qwen3-4b-instruct-2507"      # 7.8GB, 300-350 tok/s
    REASONING = "qwen3-4b-thinking-2507"         # 4.5GB, 300-330 tok/s  
    BALANCED = "qwen3-8b"                   # 6.0GB, 180-220 tok/s
    CAPABLE = "qwen3-14b"                   # 8.0GB, 140-160 tok/s
    
    # Maximum model (requires careful management)
    MAXIMUM = "qwen3-30b-a3b-instruct-2507"      # 15.5GB, 80-100 tok/s

class TaskComplexity(Enum):
    """Task complexity scoring for model selection."""
    
    SIMPLE = 0.1      # Basic extraction, formatting
    MODERATE = 0.4    # Standard job parsing, simple reasoning
    COMPLEX = 0.7     # Multi-step analysis, complex reasoning
    EXPERT = 0.9      # Advanced reasoning, edge cases
```

### Hardware-Aware Model Manager

```python
import torch
import psutil
import time
from typing import Optional, Dict, Any
from pathlib import Path
import logging

class RTX4090ModelManager:
    """Hardware-aware model management for RTX 4090 laptop."""
    
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.vram_threshold = 0.95  # 95% VRAM usage limit
        self.temp_threshold = 85    # °C thermal limit
        self.power_limit = 120      # Watts sustainable
        
        # Model switching cache
        self.model_cache_dir = Path("./models")
        self.switch_times = {}
        
        # Hardware monitoring
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """Initialize hardware monitoring."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.total_vram = torch.cuda.get_device_properties(0).total_memory
        else:
            raise RuntimeError("RTX 4090 not found or CUDA not available")
    
    def get_hardware_status(self) -> Dict[str, float]:
        """Get current hardware metrics."""
        try:
            # VRAM usage
            vram_used = torch.cuda.memory_allocated(0)
            vram_reserved = torch.cuda.memory_reserved(0)
            vram_free = self.total_vram - vram_reserved
            
            # Temperature (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except ImportError:
                temp = 0.0
                power = 0.0
            
            return {
                "vram_used_gb": vram_used / 1e9,
                "vram_free_gb": vram_free / 1e9,
                "vram_utilization": vram_used / self.total_vram,
                "temperature_c": temp,
                "power_draw_w": power,
                "system_ram_gb": psutil.virtual_memory().used / 1e9
            }
        except Exception as e:
            logging.warning(f"Hardware monitoring failed: {e}")
            return {"error": str(e)}
    
    def can_load_model(self, model_name: str) -> bool:
        """Check if model can safely load given current hardware state."""
        
        # Get model requirements
        model_config = MODEL_CONFIGS.get(model_name, {})
        required_vram_gb = model_config.get("vram_gb", 16.0)
        
        # Get current status
        status = self.get_hardware_status()
        
        # Check VRAM availability (leave 1GB buffer)
        available_vram = status.get("vram_free_gb", 0)
        if required_vram_gb > (available_vram - 1.0):
            logging.warning(f"Insufficient VRAM: need {required_vram_gb}GB, have {available_vram}GB")
            return False
        
        # Check temperature
        temp = status.get("temperature_c", 0)
        if temp > self.temp_threshold:
            logging.warning(f"Temperature too high: {temp}°C > {self.temp_threshold}°C")
            return False
        
        # Check power draw
        power = status.get("power_draw_w", 0)
        if power > self.power_limit:
            logging.warning(f"Power draw too high: {power}W > {self.power_limit}W")
            return False
        
        return True
    
    async def switch_model(
        self,
        target_model: str,
        force_switch: bool = False
    ) -> bool:
        """Switch to target model with safety checks."""
        
        # Skip if already loaded
        if self.current_model_name == target_model and not force_switch:
            return True
        
        logging.info(f"Switching model: {self.current_model_name} -> {target_model}")
        switch_start = time.time()
        
        try:
            # Step 1: Unload current model if exists
            if self.current_model is not None:
                await self._unload_current_model()
            
            # Step 2: Check if target model can be loaded
            if not force_switch and not self.can_load_model(target_model):
                return False
            
            # Step 3: Load new model
            success = await self._load_model(target_model)
            
            if success:
                switch_time = time.time() - switch_start
                self.switch_times[target_model] = switch_time
                logging.info(f"Model switch completed in {switch_time:.1f}s")
                return True
            else:
                logging.error(f"Failed to load model: {target_model}")
                return False
                
        except Exception as e:
            logging.error(f"Model switch failed: {e}")
            return False
    
    async def _unload_current_model(self):
        """Safely unload current model and free VRAM."""
        if self.current_model is None:
            return
        
        logging.info(f"Unloading model: {self.current_model_name}")
        
        # Delete model reference
        del self.current_model
        self.current_model = None
        self.current_model_name = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Wait for memory cleanup
        await asyncio.sleep(2.0)
        
        # Verify cleanup
        status = self.get_hardware_status()
        logging.info(f"Post-unload VRAM: {status.get('vram_used_gb', 0):.1f}GB used")
    
    async def _load_model(self, model_name: str) -> bool:
        """Load specific model with vLLM."""
        
        model_config = MODEL_CONFIGS.get(model_name)
        if not model_config:
            logging.error(f"Unknown model: {model_name}")
            return False
        
        try:
            from vllm import LLM
            
            # Configure vLLM for the model
            vllm_config = {
                "model": model_config["model_id"],
                "dtype": "half" if not model_config.get("quantization") else "auto",
                "max_model_len": model_config["context_length"],
                "gpu_memory_utilization": model_config.get("gpu_memory_utilization", 0.90),
                "enforce_eager": False,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "trust_remote_code": True,
            }
            
            # Add quantization if specified
            if model_config.get("quantization"):
                if "awq" in model_config["quantization"].lower():
                    vllm_config["quantization"] = "awq"
                elif "gptq" in model_config["quantization"].lower():
                    vllm_config["quantization"] = "gptq"
            
            # Load model
            logging.info(f"Loading {model_name} with vLLM...")
            self.current_model = LLM(**vllm_config)
            self.current_model_name = model_name
            
            # Verify loading
            status = self.get_hardware_status()
            logging.info(f"Model loaded. VRAM usage: {status.get('vram_used_gb', 0):.1f}GB")
            
            return True
            
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            return False
    
    def select_optimal_model(
        self,
        task_complexity: float,
        context_length: int,
        latency_sensitive: bool = True,
        privacy_required: bool = True
    ) -> str:
        """Select optimal model based on task requirements and hardware state."""
        
        # Get current hardware status
        status = self.get_hardware_status()
        available_vram = status.get("vram_free_gb", 0)
        temperature = status.get("temperature_c", 0)
        
        # Thermal throttling - prefer smaller models when hot
        if temperature > 80:
            if task_complexity < 0.5:
                return "qwen3-4b-thinking-2507"  # Smallest, coolest
            else:
                return "qwen3-8b-base"  # Balanced
        
        # VRAM-constrained selection
        if available_vram < 8:  # Low VRAM
            return "qwen3-4b-thinking-2507"  # 4.5GB
        elif available_vram < 10:  # Medium VRAM  
            return "qwen3-8b-base" if task_complexity < 0.7 else "qwen3-4b-thinking-2507"
        elif available_vram < 12:  # Good VRAM
            if task_complexity > 0.8:
                return "qwen3-14b"  # Maximum capability
            elif task_complexity > 0.4:
                return "qwen3-8b"
            else:
                return "qwen3-4b-instruct-2507"  # Fast
        else:  # Plenty of VRAM
            if task_complexity > 0.9 and available_vram > 14:
                return "qwen3-30b-a3b-instruct-2507"  # Maximum model
            elif task_complexity > 0.7:
                return "qwen3-14b"
            elif task_complexity > 0.4:
                return "qwen3-8b"
            else:
                return "qwen3-4b-instruct-2507"

# Model configurations
MODEL_CONFIGS = {
    "qwen3-4b-instruct-2507": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "quantization": None,
        "vram_gb": 7.8,
        "context_length": 262144,
        "gpu_memory_utilization": 0.85,
        "expected_tokens_sec": "300-350",
        "mode": "instruct"
    },
    "qwen3-4b-thinking-2507": {
        "model_id": "Qwen/Qwen3-4B-Thinking-2507", 
        "quantization": "Q5_K_M",
        "vram_gb": 4.5,
        "context_length": 262144,
        "gpu_memory_utilization": 0.80,
        "expected_tokens_sec": "300-330",
        "mode": "thinking"
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "quantization": "awq-4bit",
        "vram_gb": 6.0,
        "context_length": 131072,
        "gpu_memory_utilization": 0.85,
        "expected_tokens_sec": "180-220",
        "mode": "base_with_prompting"
    },
    "qwen3-14b": {
        "model_id": "Qwen/Qwen3-14B",
        "quantization": "awq-4bit", 
        "vram_gb": 8.0,
        "context_length": 131072,
        "gpu_memory_utilization": 0.85,
        "expected_tokens_sec": "140-160",
        "mode": "base_with_prompting"
    },
    "qwen3-30b-a3b-instruct-2507": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "quantization": "awq-4bit",
        "vram_gb": 15.5,
        "context_length": 262144, 
        "gpu_memory_utilization": 0.95,
        "expected_tokens_sec": "80-100",
        "mode": "instruct"
    }
}
```

### Intelligent Task Router

```python
class IntelligentModelRouter:
    """Route tasks to optimal models based on complexity and hardware."""
    
    def __init__(self, model_manager: RTX4090ModelManager):
        self.model_manager = model_manager
        self.task_history = []
        self.performance_metrics = {}
    
    def analyze_task_complexity(self, task_data: Dict[str, Any]) -> float:
        """Analyze task complexity to determine optimal model."""
        
        complexity_score = 0.0
        
        # Content length factor
        content_length = len(task_data.get("content", ""))
        if content_length > 50000:
            complexity_score += 0.3  # Very long content
        elif content_length > 20000:
            complexity_score += 0.2  # Long content
        elif content_length > 5000:
            complexity_score += 0.1  # Medium content
        
        # Task type factor  
        task_type = task_data.get("task_type", "")
        complexity_map = {
            "simple_extraction": 0.1,
            "standard_parsing": 0.3,
            "salary_analysis": 0.5,
            "complex_reasoning": 0.7,
            "multi_step_analysis": 0.8,
            "edge_case_handling": 0.9
        }
        complexity_score += complexity_map.get(task_type, 0.4)
        
        # Content complexity indicators
        content = task_data.get("content", "").lower()
        if "equity" in content or "stock options" in content:
            complexity_score += 0.2  # Complex compensation
        if "remote" in content and "hybrid" in content:
            complexity_score += 0.1  # Complex work arrangements
        if content.count("<table") > 2:
            complexity_score += 0.2  # Multiple tables
        
        return min(complexity_score, 1.0)
    
    async def route_task(self, task_data: Dict[str, Any]) -> str:
        """Route task to optimal model and switch if needed."""
        
        # Analyze task complexity
        complexity = self.analyze_task_complexity(task_data)
        
        # Select optimal model
        optimal_model = self.model_manager.select_optimal_model(
            task_complexity=complexity,
            context_length=len(task_data.get("content", "")),
            latency_sensitive=task_data.get("latency_sensitive", True),
            privacy_required=task_data.get("privacy_required", True)
        )
        
        # Switch to optimal model if needed
        switch_success = await self.model_manager.switch_model(optimal_model)
        
        if not switch_success:
            # Fallback strategy - try smaller model
            fallback_models = [
                "qwen3-8b",
                "qwen3-4b-thinking-2507", 
                "qwen3-4b-instruct-2507"
            ]
            
            for fallback in fallback_models:
                if await self.model_manager.switch_model(fallback):
                    logging.warning(f"Using fallback model: {fallback}")
                    return fallback
            
            # Ultimate fallback - cloud API
            logging.error("All local models failed, falling back to cloud")
            return "cloud-fallback"
        
        return optimal_model
```

## Implementation Plan

### Phase 1: Core Model Manager (Week 1)

1. Implement `RTX4090ModelManager` class
2. Setup hardware monitoring (VRAM, temperature, power)
3. Create model switching logic with safety checks
4. Test single model loading/unloading

### Phase 2: Intelligent Routing (Week 1)  

1. Implement `IntelligentModelRouter` class
2. Create task complexity analysis
3. Build fallback strategies
4. Integration testing

### Phase 3: Production Integration (Week 2)

1. Update inference service to use model manager
2. Implement graceful degradation
3. Performance optimization
4. Monitoring dashboard

## Performance Expectations

### Model Switching Times

- **4B ↔ 8B**: ~20-30 seconds
- **8B ↔ 14B**: ~30-45 seconds  
- **14B ↔ 30B**: ~45-60 seconds
- **Emergency unload**: ~5-10 seconds

### Hardware Utilization Targets

- **VRAM Usage**: 85-95% (leave buffer for switching)
- **Temperature**: <85°C sustained operation
- **Power Draw**: 100-120W continuous
- **Model Uptime**: 95%+ availability

### Task Distribution Estimate

- **Simple Tasks** (40%): Qwen3-4B-Instruct-2507
- **Standard Tasks** (35%): Qwen3-8B  
- **Complex Tasks** (20%): Qwen3-14B
- **Expert Tasks** (5%): Qwen3-30B-A3B or Cloud

## Monitoring and Alerting

```yaml
# monitoring_config.yaml
hardware_monitoring:
  interval_seconds: 5
  
  thresholds:
    vram_usage_critical: 0.95      # 95% VRAM usage
    vram_usage_warning: 0.85       # 85% VRAM usage  
    temperature_critical: 87       # °C
    temperature_warning: 80        # °C
    power_critical: 140            # Watts
    power_warning: 130             # Watts
  
  alerts:
    vram_overflow: "immediate"     # Immediate model switching
    thermal_throttle: "gradual"    # Reduce batch size first
    power_limit: "immediate"       # Switch to smaller model

model_switching:
  max_switch_time_seconds: 90      # Timeout for model switching
  switch_failure_retries: 3        # Retry attempts
  fallback_strategy: "cascade"     # 30B→14B→8B→4B→cloud
  
performance_tracking:
  track_switch_times: true
  track_model_efficiency: true
  track_task_routing_accuracy: true
```

## Consequences

### Positive

- ✅ **Eliminates VRAM overflow crashes** - Single model constraint prevents memory issues
- ✅ **Optimal hardware utilization** - Uses RTX 4090 laptop efficiently within thermal limits
- ✅ **Intelligent model selection** - Right model for each task complexity
- ✅ **Graceful degradation** - Cascading fallback prevents complete failures
- ✅ **Cost optimization** - Local processing for 90%+ of tasks
- ✅ **Hardware longevity** - Thermal and power management protects hardware

### Negative  

- ❌ **Model switching latency** - 30-60 second delays for model changes
- ❌ **Increased complexity** - More sophisticated orchestration required
- ❌ **Storage requirements** - Must store multiple quantized models (~50GB total)
- ❌ **No parallel processing** - Cannot run multiple model types simultaneously

### Trade-offs

- **Latency vs Cost**: 30-60s switching delay vs 90% cost savings
- **Complexity vs Reliability**: More complex system vs guaranteed memory safety
- **Storage vs Performance**: 50GB model storage vs local processing speed

## Related Decisions

- **ADR-020**: Hybrid LLM Strategy (needs capability mapping fixes)
- **ADR-027**: Final Inference Stack (implements this via vLLM)
- **ADR-019**: Local-First AI Integration (provides model selection foundation)
- **ADR-030**: Error Handling & Resilience (handles switching failures)

## References

- [RTX 4090 Laptop Memory Benchmarks](https://www.techpowerup.com/review/nvidia-geforce-rtx-4090-founders-edition/)
- [vLLM Memory Management](https://docs.vllm.ai/en/latest/models/engine_args.html#gpu-memory-utilization)
- [Qwen3 Model Memory Requirements](https://qwenlm.github.io/blog/qwen3/)
- [CUDA Memory Management Best Practices](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

## Review

- **Date**: August 2025
- **Reviewed by**: AI Engineering Team  
- **Next Review**: October 2025
- **Implementation Priority**: **CRITICAL** - Blocks deployment without this ADR
