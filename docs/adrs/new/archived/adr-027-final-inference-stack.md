# ADR-027: Final RTX 4090 Laptop Inference Stack Decision

## Status

**Decided** - January 2025

## Context

After extensive research and benchmarking analysis, we need to select the optimal inference stack for our RTX 4090 laptop GPU (16GB VRAM, Ada Lovelace, Flash Attention 2 support) for the AI job scraper application.

## Hardware Constraints

- **GPU**: RTX 4090 Laptop (AD103, SM 8.9)
- **VRAM**: 16GB GDDR6
- **Power**: 100-120W sustainable, 150W burst
- **Thermal**: Throttles at 87°C
- **Flash Attention**: Version 2 only (RTX 4090 = Ada Lovelace - FA3 requires Hopper H100/H800)
- **Memory Bandwidth**: 576 GB/s (43% less than desktop RTX 4090)
- **Architecture**: Ada Lovelace - supports Flash Attention 2, TF32, but NOT Flash Attention 3

## Decision

### Primary Stack: vLLM + Flash Attention 2

> **Winner: vLLM v0.6.5+ with Flash Attention 2**

#### Rationale

1. **Best Balance**: Performance, features, ease of use
2. **Flash Attention 2**: Native support, well-tested on Ada Lovelace
3. **Quantization**: Excellent AWQ/GPTQ support
4. **Production Ready**: Battle-tested, active development
5. **Simple Setup**: pip installable, minimal configuration

### Configuration

```python
# Optimal vLLM Configuration for RTX 4090 Laptop
from vllm import LLM, SamplingParams
import os

# Environment setup
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # FA2 for Ada Lovelace
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model configuration
config = {
    "model": "Qwen/Qwen3-8B",  # Base model - instruct variant doesn't exist
    "quantization": "awq",  # AWQ-4bit more efficient than GPTQ
    "dtype": "half",
    "max_model_len": 131072,  # Qwen3-8B supports 131K context
    "gpu_memory_utilization": 0.85,  # Leave headroom for thermal
    "enforce_eager": False,  # Enable CUDA graphs
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 64,  # Reduced for laptop
    "trust_remote_code": True,
}

llm = LLM(**config)
```

## Performance Expectations

### Realistic Benchmarks (120W Power Limit)

| Model | Quantization | VRAM Usage | Tokens/sec | Context | Notes |
|-------|--------------|------------|------------|---------|-------|
| **Qwen3-4B-Instruct-2507** | FP16 | 7.8GB | 300-350 | 262K | Best speed (instruct) |
| **Qwen3-8B** (base) | AWQ-4bit | 6.0GB | 180-220 | 131K | **Recommended** + structured prompting |
| **Qwen3-14B** (base) | AWQ-4bit | 8.0GB | 140-160 | 131K | Max quality + structured prompting |
| **Qwen3-4B-Thinking-2507** | Q5_K_M | 4.5GB | 300-330 | 262K | Reasoning mode |

### Why vLLM Wins

1. **Performance**: 180-220 tok/s for Qwen3-8B models (job extraction sweet spot)
2. **Memory Efficiency**: PagedAttention reduces VRAM by 30%
3. **Flash Attention 2 Support**: Native FA2 support for Ada Lovelace architecture
4. **Thermal Friendly**: Works well within 100-120W power envelope
5. **Library Support**: First-class Pydantic/structured output integration
6. **RTX 4090 Optimization**: Excellent support for Ada Lovelace (SM 8.9)
7. **Qwen3 Compatibility**: Native support for Qwen3-2507 model series

## Flash Attention Support Clarification

### RTX 4090 Supports Flash Attention 2 ONLY

**Important**: RTX 4090 Laptop uses Ada Lovelace architecture (SM 8.9) which supports Flash Attention 2, NOT Flash Attention 3.

| Flash Attention Version | Required Architecture | Compute Capability | RTX 4090 Support |
|------------------------|----------------------|-------------------|------------------|
| Flash Attention 2 | Ampere+ | SM 8.0+ | ✅ Supported |
| Flash Attention 3 | **Hopper only** | **SM 9.0** | ❌ Not Supported |

### Why FA3 Doesn't Work on RTX 4090

Flash Attention 3 requires Hopper-specific features:

- WGMMA Instructions (Warpgroup Matrix Multiply-Accumulate)
- TMA (Tensor Memory Accelerator)
- Hopper thread block clusters

These are not available on Ada Lovelace, making FA2 the optimal choice for RTX 4090.

### Performance Impact

- Flash Attention 2: 2-4x speedup over standard attention
- Memory savings: 10-20x for long sequences
- Optimal for RTX 4090's memory bandwidth (576 GB/s)

## Alternatives Evaluated

### TensorRT-LLM (Not Recommended)

- ✅ 30-50% faster than vLLM
- ❌ Complex compilation per model
- ❌ Poor structured output support
- ❌ Difficult debugging
- **Verdict**: Overkill for job extraction

### llama.cpp (Backup Option)

- ✅ GGUF format flexibility
- ✅ CPU+GPU hybrid
- ❌ 40% slower than vLLM
- ❌ Limited structured output
- **Verdict**: Good for experimentation, not production

### ExLlamaV2 (Specialized Use)

- ✅ Best for GPTQ/EXL2 formats
- ✅ Good for long context
- ❌ Smaller ecosystem
- ❌ Less documentation
- **Verdict**: Consider if using GPTQ models

### Aphrodite Engine (Future Option)

- ✅ vLLM fork with optimizations
- ✅ Better LoRA support
- ❌ Less mature
- ❌ Smaller community
- **Verdict**: Watch for future

## Implementation Plan

### 1. Installation

```bash
# Create environment
uv venv
source .venv/bin/activate

# Install vLLM with Flash Attention 2
uv add vllm==0.6.5
uv add flash-attn --no-build-isolation

# Verify Flash Attention 2
python -c "from vllm import _custom_ops as ops; print('FA2 available')"
```

### 2. Model Selection

```python
# Download Qwen3 model
from huggingface_hub import snapshot_download

# PRIMARY: Base model with structured prompting
model_id = "Qwen/Qwen3-8B"  # Base model - Instruct variant doesn't exist  
snapshot_download(repo_id=model_id, cache_dir="./models")

# HIGH CAPABILITY: Base model for complex tasks
max_model_id = "Qwen/Qwen3-14B"  # Base model with structured prompting
snapshot_download(repo_id=max_model_id, cache_dir="./models")

# AVAILABLE: Download actual thinking model
thinking_model_id = "Qwen/Qwen3-4B-Thinking-2507"  # ✅ This exists
snapshot_download(repo_id=thinking_model_id, cache_dir="./models")

# AVAILABLE: Download actual instruct model
instruct_model_id = "Qwen/Qwen3-4B-Instruct-2507"  # ✅ This exists
snapshot_download(repo_id=instruct_model_id, cache_dir="./models")
```

### 3. Thermal Management

```python
# Set conservative power limit
import subprocess
subprocess.run(["nvidia-smi", "-pl", "120"])  # 120W for sustained operation

# Monitor temperature
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
if temp > 80:
    print("Warning: High temperature, reducing batch size")
```

### 4. Production Configuration

```yaml
# config.yaml
inference:
  engine: vllm
  model: Qwen/Qwen3-8B  # Base model with structured prompting
  quantization: awq  # More efficient than GPTQ
  max_batch_size: 16
  max_tokens: 2048
  temperature: 0.1  # Low for consistency
  power_limit: 120  # Watts
  gpu_memory_utilization: 0.85
  attention_backend: FLASH_ATTN
  # See ADR-019 for complete Qwen3 configuration
```

## Monitoring & Optimization

### Key Metrics to Track

- **Tokens/second**: Target 180+ for Qwen3-8B (base + AWQ), 300+ for Qwen3-4B models
- **GPU Temperature**: Keep below 80°C
- **VRAM Usage**: Stay under 14GB (of 16GB)
- **Power Draw**: 100-120W sustained
- **First Token Latency**: < 200ms
- **Context Length**: Utilize Qwen3's extended context (131K-262K tokens)

### Optimization Tips

1. **Use AWQ-4bit for all base models** - more efficient than GPTQ for Qwen3-8B and Qwen3-14B
2. **Implement structured prompting** for base models to achieve instruct-level performance
3. Enable prefix caching for repeated prompts
4. Batch similar-length requests
5. Use FP8 KV cache if supported
6. Profile with `vllm.profiler`
7. Leverage Qwen3's extended context windows

## Decision Rationale

vLLM is the clear winner because:

1. **Simplicity**: pip install and run
2. **Performance**: Excellent for Qwen3 models (180-350 tok/s)
3. **Reliability**: Production-tested by major companies
4. **Ecosystem**: Great structured output support
5. **Maintenance**: Active development, quick bug fixes
6. **Qwen3 Support**: Native compatibility with Qwen3-2507 series

## Consequences

### Positive

- ✅ Simple deployment and maintenance
- ✅ Excellent performance for Qwen3 models (4B-30B range)
- ✅ Native Flash Attention 2 support
- ✅ Great structured output integration
- ✅ Active community support

### Negative

- ❌ Not the absolute fastest (TensorRT-LLM is faster)
- ❌ Python overhead vs pure C++ solutions
- ❌ Requires careful thermal management on laptop

## Related Decisions

- **ADR-019**: Local-First AI Integration with Qwen3-2507 Models (model selection)
- **ADR-020**: Hybrid LLM Strategy with GPT-5 and Local Models (hybrid approach)
- **ADR-028**: Structured Output Strategy for Job Extraction (output validation)

## References

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [Flash Attention 3 Paper](https://tridao.me/publications/flash3/flash3.pdf) (Hopper-only)
- [AWQ Quantization](https://arxiv.org/abs/2306.00978)
- [NVIDIA Ada Lovelace Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [RTX 4090 Laptop Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)

## Review

- **Date**: January 2025
- **Reviewed by**: AI Engineering Team
- **Next Review**: March 2025
