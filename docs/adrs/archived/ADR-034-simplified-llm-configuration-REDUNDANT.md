# ADR-034: Simplified LLM Configuration [REDUNDANT]

## ⚠️ **REDUNDANT WITH ADR-004**

**Archival Date:** August 23, 2025  
**Redundant With:** ADR-004 (Local AI Processing Architecture)  
**Reason:** 100% configuration overlap - ADR-004 server approach makes this obsolete

**For current LLM configuration, see:**

- **ADR-004**: Complete vLLM server configuration and deployment
- **Location**: `ai-job-scraper/docs/adrs/ADR-004-local-ai-integration.md`

---

## Redundancy Analysis

**Configuration Overlap Evidence:**

**ADR-034 Configuration:**

```python
LLM_CONFIG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "max_model_len": 8192,
    "quantization": "fp8",
    "gpu_memory_utilization": 0.9,
}
```

**ADR-004 Equivalent (Server Mode):**

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --quantization fp8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

## Why Made Redundant

**ADR-004's server approach supersedes all ADR-034 capabilities:**

- ✅ Same model configuration parameters
- ✅ Same optimization settings (FP8, memory utilization)  
- ✅ Server deployment handles all configuration complexity
- ✅ OpenAI client pattern abstracts configuration details
- ✅ Production deployment uses consistent server setup

## Migration Path

**All ADR-034 configuration is now handled by ADR-004:**

- **Model Selection**: Qwen/Qwen3-4B-Instruct-2507-FP8
- **Quantization**: FP8 with 8x memory reduction
- **Context**: 8K token optimization
- **GPU Utilization**: 90% with FP8 memory savings
- **Deployment**: Docker Compose server architecture

---

## Original Content [ARCHIVED FOR REFERENCE]

**Original Status:** Accepted  
**Original Version/Date:** v1.3 / 2025-08-23

### Key Features Now in ADR-004

- ✅ FP8 quantization configuration
- ✅ GPU memory optimization (90% utilization)
- ✅ 8K context window settings
- ✅ Production deployment patterns
- ✅ Environment variable management
- ✅ Performance validation metrics

**All functionality consolidated into ADR-004's comprehensive server-based architecture.**
