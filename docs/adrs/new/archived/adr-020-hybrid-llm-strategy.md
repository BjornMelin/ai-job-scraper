# ADR-020: Hybrid LLM Strategy with GPT-5 and Local Models

## Status

**Updated** - August 2025 (Fixed LLM capability mapping and added single-model constraint)

## Context

With the release of GPT-5 (August 2025) and advanced local models like Qwen3, we need a strategic approach that optimizes for cost, performance, and privacy. Our RTX 4090 laptop (16GB VRAM) enables powerful local inference while GPT-5 offers state-of-the-art capabilities at competitive prices.

### GPT-5 Pricing (January 2025)

- **GPT-5**: $1.25/1M input, $10/1M output - 400K context
- **GPT-5-mini**: $0.25/1M input, $2/1M output - 256K context  
- **GPT-5-nano**: $0.05/1M input, $0.40/1M output - 128K context
- **Reasoning modes**: minimal, low, medium, high effort levels
- **Native tools**: Web search, file search, image generation built-in

### Local Model Capabilities (RTX 4090)

**✅ Available Models:**

- **Qwen3-4B-Instruct-2507**: Fast processing in FP16 (~7.8GB VRAM) ✓
- **Qwen3-4B-Thinking-2507**: Complex reasoning with Q5_K_M (~4.5GB VRAM) ✓
- **Qwen3-30B-A3B-Instruct-2507**: Maximum capability with AWQ 4-bit (~15.5GB VRAM) ✓

**🔄 Base Models (Instruct variants don't exist):**

- **Qwen3-8B (base)**: With AWQ 4-bit quantization (~6GB VRAM) + structured prompting
- **Qwen3-14B (base)**: With AWQ 4-bit quantization (~8GB VRAM) + structured prompting
- **Performance**: 300-350 tokens/s (4B), 180-220 tokens/s (8B), 140-160 tokens/s (14B)
- **Implementation**: See ADR-019 for Qwen3 model configurations and ADR-027 for inference stack

## Decision

### Primary Strategy: Local-First with Smart Cloud Fallback

**Rationale**: Maximize local GPU utilization while leveraging cloud for complex tasks

**⚠️ CRITICAL CONSTRAINT**: RTX 4090 laptop (16GB VRAM) can only run **ONE active model at a time**. Model switching required between tasks. See ADR-029 for single-model architecture implementation.

### Model Selection Matrix

| Task Type | Primary Model | Fallback | Rationale |
|-----------|--------------|----------|-----------|
| **Job Extraction** | Qwen3-8B (base) | GPT-5-nano | Local speed, structured prompting |
| **Complex Parsing** | Qwen3-14B (base) | GPT-5-mini | Maximum local capability |
| **Complex Reasoning** | Qwen3-30B-A3B-Thinking-2507 | GPT-5 | Maximum reasoning capability (85+ AIME score) |
| **Salary Analysis** | Qwen3-8B (base) | GPT-5-mini | Context + structured prompting |
| **Resume Matching** | Qwen3-Embedding | - | Privacy critical |
| **Summarization** | Qwen3-4B-Instruct-2507 | GPT-5-nano | Fast local processing |
| **Error Recovery** | - | GPT-5 | Complex edge cases |

### Decision Tree Implementation

```python
from enum import Enum
from typing import Optional, Dict, Any
import torch

class ModelSelection(Enum):
    LOCAL_SMALL = "qwen3-4b-instruct-2507"  # ✅ Available
    LOCAL_MEDIUM = "qwen3-8b"  # 🔄 Base model (Instruct doesn't exist) 
    LOCAL_LARGE = "qwen3-14b"  # 🔄 Base model (Instruct doesn't exist)
    LOCAL_THINKING = "qwen3-4b-thinking-2507"  # ✅ Available
    LOCAL_MAX = "qwen3-30b-a3b-instruct-2507"  # ✅ Available
    CLOUD_NANO = "gpt-5-nano"
    CLOUD_MINI = "gpt-5-mini"
    CLOUD_FULL = "gpt-5"

class HybridLLMRouter:
    """Intelligent routing between local and cloud models with single-model constraint."""
    
    def __init__(self):
        self.vram_available = torch.cuda.get_device_properties(0).total_memory
        self.current_active_model = None  # Only ONE model active at a time
        self.model_manager = None  # Reference to RTX4090ModelManager from ADR-029
        self.usage_stats = {"local": 0, "cloud": 0}
        
    def select_model(
        self,
        task_type: str,
        complexity: float,  # 0-1 score
        context_length: int,
        privacy_required: bool = False,
        latency_sensitive: bool = True
    ) -> ModelSelection:
        """Select optimal model based on task requirements."""
        
        # Privacy-critical tasks always local
        if privacy_required:
            if context_length > 131072:
                return ModelSelection.LOCAL_MAX  # 256K context
            elif context_length > 32768:
                return ModelSelection.LOCAL_LARGE
            elif complexity > 0.7:
                return ModelSelection.LOCAL_THINKING
            else:
                return ModelSelection.LOCAL_MEDIUM
        
        # Latency-sensitive prefer local
        if latency_sensitive and context_length < 8192:
            if complexity < 0.3:
                return ModelSelection.LOCAL_SMALL
            elif complexity < 0.6:
                return ModelSelection.LOCAL_MEDIUM
            else:
                return ModelSelection.LOCAL_THINKING
        
        # Complex reasoning tasks
        if complexity > 0.8:
            if self._can_run_locally(context_length):
                return ModelSelection.LOCAL_THINKING
            return ModelSelection.CLOUD_FULL
        
        # Long context tasks
        if context_length > 65536:
            return ModelSelection.CLOUD_MINI  # 256K context
        
        # Cost optimization for simple tasks
        if complexity < 0.4:
            return ModelSelection.CLOUD_NANO  # Cheapest option
        
        # Default to local medium
        return ModelSelection.LOCAL_MEDIUM
    
    def _can_run_locally(self, context_length: int) -> bool:
        """Check if context fits in VRAM."""
        # Rough estimate: 2 bytes per token average
        required_memory = context_length * 2 * 1024  # In bytes
        return required_memory < self.vram_available * 0.5
```

## Architecture

### 1. Model Loading Strategy

```python
class RTX4090ModelManager:
    """Optimized model management for RTX 4090."""
    
    def __init__(self):
        self.cache_dir = Path("./models")
        self.loaded_models = {}
        self.quantization_config = {
            "qwen3-14b": "AWQ-4bit",  # ~8.0GB VRAM (base model)
            "qwen3-8b": "AWQ-4bit",   # ~6.0GB VRAM (base model)
            "qwen3-4b-instruct-2507": None,  # FP16 ~7.8GB VRAM ✅ Available
            "qwen3-4b-thinking-2507": "Q5_K_M",  # ~4.5GB VRAM ✅ Available
            "qwen3-30b-a3b-instruct-2507": "AWQ-4bit"  # ~15.5GB VRAM ✅ Available
        }
    
    async def load_model(self, model_name: str):
        """Load model with optimal quantization."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        quant = self.quantization_config.get(model_name, "Q4_K_M")
        
        if "thinking" in model_name:
            # Special handling for thinking models
            model = await self._load_thinking_model(model_name, quant)
        else:
            model = await self._load_standard_model(model_name, quant)
        
        self.loaded_models[model_name] = model
        return model
    
    def unload_model(self, model_name: str):
        """Free VRAM by unloading model."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()
```

### 2. Fallback Mechanism

```python
class IntelligentFallback:
    """Smart fallback from local to cloud."""
    
    async def process_with_fallback(
        self,
        prompt: str,
        primary_model: str,
        fallback_model: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Try local first, fallback to cloud if needed."""
        
        # Try local model
        try:
            result = await self.local_inference(prompt, primary_model)
            if self._validate_output(result):
                return {"source": "local", "model": primary_model, **result}
        except (OutOfMemoryError, TimeoutError) as e:
            logger.warning(f"Local model failed: {e}")
        
        # Fallback to cloud
        try:
            result = await self.cloud_inference(prompt, fallback_model)
            return {"source": "cloud", "model": fallback_model, **result}
        except Exception as e:
            # Emergency fallback to GPT-5
            result = await self.cloud_inference(prompt, "gpt-5")
            return {"source": "cloud", "model": "gpt-5", **result}
```

### 3. Cost Optimization

```python
class CostOptimizer:
    """Minimize API costs while maintaining quality."""
    
    def __init__(self):
        self.cost_per_million = {
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5-nano": {"input": 0.05, "output": 0.40}
        }
        self.daily_budget = 10.0  # $10/day
        self.usage_today = 0.0
    
    def select_by_budget(self, tokens_needed: int) -> str:
        """Select model based on remaining budget."""
        remaining = self.daily_budget - self.usage_today
        
        for model in ["gpt-5-nano", "gpt-5-mini", "gpt-5"]:
            cost = self.estimate_cost(model, tokens_needed)
            if cost <= remaining * 0.1:  # Use max 10% of remaining
                return model
        
        return "local"  # Force local if budget exhausted
```

## Implementation Plan

### Phase 1: Local Model Setup (Day 1)

- Download **actual available** Qwen3 models:
  - ✅ Qwen3-4B-Instruct-2507
  - ✅ Qwen3-4B-Thinking-2507  
  - ✅ Qwen3-30B-A3B-Instruct-2507
  - 🔄 Qwen3-8B (base) + Qwen3-14B (base) with structured prompting
- Setup vLLM with Flash Attention 2
- Benchmark Qwen3 token generation speeds
- Implement structured prompting for base models

### Phase 2: GPT-5 Integration (Day 2)

- Update API clients for GPT-5 variants
- Implement reasoning_effort parameters
- Setup built-in tools (web search, etc.)
- Create cost tracking system

### Phase 3: Hybrid Router (Day 3)

- Implement intelligent routing logic
- Create fallback mechanisms
- Setup monitoring and logging
- Performance optimization

### Phase 4: Testing & Tuning (Day 4)

- A/B test local vs cloud accuracy
- Optimize quantization levels
- Fine-tune routing thresholds
- Load testing

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Qwen3-4B-Instruct-2507 speed | 300+ tokens/s | - |
| Qwen3-8B (base) speed | 180+ tokens/s | - |
| Qwen3-14B (base) speed | 140+ tokens/s | - |
| Cloud fallback rate | <10% | - |
| Monthly API cost | <$50 | - |
| Extraction accuracy | >95% | - |
| P95 latency | <500ms | - |

## Monitoring

```python
class HybridMonitor:
    """Track hybrid system performance."""
    
    def __init__(self):
        self.metrics = {
            "local_success_rate": 0.0,
            "cloud_fallback_rate": 0.0,
            "average_latency_ms": 0.0,
            "daily_cost_usd": 0.0,
            "vram_usage_gb": 0.0
        }
    
    async def log_inference(self, source: str, model: str, 
                           latency: float, tokens: int):
        """Log each inference for analysis."""
        # Update metrics
        # Send to monitoring dashboard
        pass
```

## Consequences

### Positive

- **90% cost reduction** vs cloud-only
- **10x faster inference** for most tasks
- **Complete privacy** for sensitive data
- **No rate limits** on local models
- **Predictable performance**

### Negative

- Initial model download (30GB+)
- VRAM limitations for largest models
- Slightly lower accuracy than GPT-5
- Complexity of hybrid system

## Migration Path

1. Start with cloud-only (GPT-5-nano)
2. Gradually introduce local models
3. Monitor accuracy and costs
4. Adjust routing thresholds
5. Optimize quantization levels

## Related Decisions

- **ADR-019**: Local-First AI Integration with Qwen3-2507 Models (model selection rationale)
- **ADR-027**: Final RTX 4090 Laptop Inference Stack Decision (implementation details)
- **ADR-028**: Structured Output Strategy for Job Extraction (output validation)

## References

- [GPT-5 Pricing and API](https://openai.com/api/pricing/)
- [Qwen3-2507 Model Cards](https://huggingface.co/Qwen)
- [Qwen3 Official Repo](https://github.com/QwenLM/Qwen3)
- [Qwen3-2507 Release Notes](https://qwenlm.github.io/blog/qwen3-2507/)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [AWQ Quantization](https://arxiv.org/abs/2306.00978)
