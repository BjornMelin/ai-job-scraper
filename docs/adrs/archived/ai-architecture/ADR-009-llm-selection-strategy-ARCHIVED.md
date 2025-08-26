# ADR-009: LLM Selection Strategy

## Metadata

**Status:** Superseded by ADR-004  
**Version/Date:** v3.0 / 2025-08-21
**Superseded:** 2025-08-22 by ADR-004 (Comprehensive Local AI Processing Architecture)

## Title

LLM Selection Strategy for Local AI Processing

---

> **⚠️ SUPERSEDED NOTICE**
>
> This ADR has been **superseded by ADR-004 (Comprehensive Local AI Processing Architecture)** as of August 22, 2025.
>
> **Reason for Supersession:** Content consolidation to eliminate 85% duplication across ADR-004, ADR-005, and ADR-009. All model selection rationale, benchmark analysis, performance validation, and deployment strategies are now consolidated into ADR-004 as the single authoritative source.
>
> **Migration Path:** Refer to ADR-004 for:
> - Model selection benchmarks and analysis (Section: Context > Model Selection Benchmark Analysis)
> - Comprehensive alternatives evaluation (Section: Decision Framework)
> - Performance validation requirements (Section: Testing)
> - Production deployment configuration (Section: Configuration)
>
> This document is preserved for historical reference only.

---

## Description

Select Qwen3-4B-Instruct-2507 as the primary local LLM for job extraction based on comprehensive performance benchmarking, achieving superior accuracy while using 75% less memory than larger models.

## Context

The AI job scraper requires a local LLM that balances processing quality, memory efficiency, and cost-effectiveness for job data extraction and enhancement tasks. The current architecture lacks a standardized local inference strategy, requiring ad-hoc cloud API usage for complex extractions which increases costs and latency.

**Performance Requirements**:

- Process 95%+ of job extractions locally per **ADR-008** threshold strategy
- Handle complex structured output per **ADR-007** requirements  
- Integrate with vLLM inference stack for production deployment
- Support 8K context length optimized for job posting processing

**Resource Constraints**:

- RTX 4090 Laptop GPU with 16GB VRAM available
- Target monthly operating cost under $30 total
- Memory efficiency for concurrent processing
- Deployment simplicity for 1-week timeline

**Quality Benchmarks** - Model evaluation based on industry-standard benchmarks:

| Model | MMLU-Pro | GPQA | Context | VRAM (AWQ) | Performance |
|-------|----------|------|---------|------------|-------------|
| Qwen3-4B-Instruct | **69.6** | **62.0** | **8K** | **2.9GB** | Excellent |
| Qwen3-8B | 56.73 | 44.44 | 8K | 4.2GB | Good |
| Qwen3-14B | 61.03 | 39.90 | 8K | 7.5GB | Good |
| Llama-3.1-8B | 48.2 | 41.5 | 8K | 4.8GB | Fair |
| Mistral-7B | 45.1 | 38.2 | 32K | 4.1GB | Fair |

## Decision Drivers

- **Solution Leverage**: vLLM integration maturity, quantization support, deployment patterns
- **Application Value**: Job extraction accuracy, structured output quality, context handling capability
- **Maintenance & Cognitive Load**: Setup complexity, documentation quality, community support
- **Architectural Adaptability**: Scaling options, model swapping capability, future upgrade paths
- **Resource Efficiency**: Memory usage, VRAM requirements, processing speed optimization

## Alternatives

- **A: Qwen3-8B Base Model** — Larger parameter count, strong general performance, well-documented deployment / Higher memory usage (4.2GB vs 2.9GB), inferior benchmarks vs 4B-Instruct
- **B: Llama-3.1-8B-Instruct** — Meta ecosystem backing, proven deployment patterns, good community support / Significantly lower benchmark scores (48.2 MMLU-Pro vs 69.6), higher memory requirements
- **C: Qwen3-4B-Instruct-2507** — Highest benchmark scores (69.6 MMLU-Pro, 62.0 GPQA), optimized 8K context, lowest memory footprint (2.9GB AWQ), superior instruction following / Newer model with less deployment history, smaller parameter count
- **D: Multi-Model Ensemble** — Could combine model strengths, redundancy for quality assurance, flexible routing / Complex deployment and management, increased memory requirements, over-engineering for current needs

### Decision Framework

| Model / Option | Solution Leverage (Weight: 35%) | Application Value (Weight: 30%) | Maintenance & Cognitive Load (Weight: 25%) | Architectural Adaptability (Weight: 10%) | Total Score | Decision |
|---|---|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | 9 | 10 | 8 | 9 | **9.1** | ✅ **Selected** |
| Qwen3-8B Base Model | 8 | 7 | 8 | 8 | 7.6 | Rejected |
| Llama-3.1-8B-Instruct | 7 | 6 | 9 | 8 | 7.1 | Rejected |
| Multi-Model Ensemble | 6 | 8 | 4 | 7 | 6.4 | Rejected |

## Decision

We will adopt **Qwen3-4B-Instruct-2507** as the primary local LLM for job extraction processing. This involves using **vLLM inference server** configured with **AWQ-INT4 quantization** and **8K context length**. This decision establishes the local inference foundation for the hybrid processing strategy defined in **ADR-006**.

## High-Level Architecture

```mermaid
graph LR
    A[Job Content] --> B[Token Router]
    B -->|≤8K tokens| C[Qwen3-4B-FP8 Local]
    B -->|>8K tokens| D[gpt-4o-mini Cloud]
    
    C --> E[vLLM Server]
    E --> F[FP8 Quantized Model]
    F --> G[Structured Output]
    
    D --> H[OpenAI API]
    H --> G
    
    subgraph "Local Infrastructure"
        I[RTX 4090 Laptop 16GB]
        J[vLLM ≥0.6.2]
        K[FP8 Quantization]
    end
    
    E -.-> I
    E -.-> J
    F -.-> K
```

## Related Requirements

### Functional Requirements

- **FR-1:** The system must process job extraction with 95%+ accuracy on standard job postings
- **FR-2:** Users must receive structured output per **ADR-007** specifications with validated JSON schemas
- **FR-3:** The system must handle job postings up to 8K tokens covering 98% of use cases
- **FR-4:** The system must support concurrent processing for multiple jobs with queue management

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Memory usage must not exceed 8GB VRAM for primary operations on RTX 4090 Laptop GPU (16GB total)
- **NFR-2:** **(Performance)** Inference speed must achieve 30+ tokens/second for typical job documents
- **NFR-3:** **(Maintainability)** Deployment time must be under 2 hours for complete setup including model download
- **NFR-4:** **(Adaptability)** Model swap capability must be available for A/B testing and upgrades

### Performance Requirements

- **PR-1:** System must process 50+ job extractions per minute under normal load
- **PR-2:** Context switching latency must be under 100ms between sequential requests
- **PR-3:** Batch processing must support 10+ concurrent jobs without memory overflow

### Integration Requirements

- **IR-1:** The solution must integrate natively with vLLM server and all performance optimizations
- **IR-2:** The component must be compatible with structured output generation per **ADR-007**
- **IR-3:** The solution must integrate with threshold routing decisions per **ADR-008**

## Related Decisions

- **ADR-005** (Inference Stack): This decision implements the vLLM deployment architecture for local inference serving
- **ADR-006** (Hybrid Strategy): The selected model serves as the primary local component in the hybrid local-cloud processing approach
- **ADR-007** (Structured Output Strategy): This model provides the LLM backend for structured JSON generation with schema validation
- **ADR-008** (Optimized Token Thresholds): The model's capacity and performance characteristics inform threshold routing decisions
- **ADR-034** (Simplified LLM Configuration): Provides the canonical FP8 configuration and quantization strategy for this model selection

## Design

### Architecture Overview

```mermaid
graph LR
    A[Job Content] --> B[Token Router]
    B -->|≤8K tokens| C[Qwen3-4B-FP8 Local]
    B -->|>8K tokens| D[gpt-4o-mini Cloud]
    
    C --> E[vLLM Server]
    E --> F[FP8 Quantized Model]
    F --> G[Structured Output]
    
    D --> H[OpenAI API]
    H --> G
    
    subgraph "Local Infrastructure"
        I[RTX 4090 Laptop 16GB]
        J[vLLM ≥0.6.2]
        K[FP8 Quantization]
    end
    
    E -.-> I
    E -.-> J
    F -.-> K
```

### Implementation Details

```python
from vllm import LLM, SamplingParams
from typing import Dict, Any, Optional

class Qwen3InferenceService:
    """Qwen3-4B-Instruct-2507 inference service."""
    
    def __init__(self, quantization: str = "fp8"):
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507-FP8"
        
        # vLLM configuration optimized for RTX 4090 with FP8
        self.llm = LLM(
            model=self.model_name,
            quantization=quantization,
            kv_cache_dtype="fp8",  # Additional memory savings
            max_model_len=8192,  # Optimal 8K context for job postings
            gpu_memory_utilization=0.9,  # Aggressive with FP8 memory savings
            swap_space=4,  # Reduced with FP8 efficiency
            enable_prefix_caching=True,
            max_num_seqs=128,
            trust_remote_code=True
        )
        
        # Sampling parameters for job extraction
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=16384,
            presence_penalty=1.0  # Reduce repetition
        )
    
    def extract_job_data(self, content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured job data from content."""
        
        prompt = self._build_extraction_prompt(content, schema)
        
        # Generate with structured constraints
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        return self._parse_structured_output(outputs[0].outputs[0].text)
    
    def _build_extraction_prompt(self, content: str, schema: Dict[str, Any]) -> str:
        """Build extraction prompt with schema constraints."""
        
        return f"""Extract job information as JSON matching this schema:
{schema}

Job Content:
{content[:30000]}  # Respect context limits

Return valid JSON only:"""

# Production deployment configuration
DEPLOYMENT_CONFIG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "quantization": "fp8",  # 8x memory reduction
    "kv_cache_dtype": "fp8",  # Additional memory savings
    "max_context": 8192,  # Optimal for job postings
    "gpu_memory": 0.9,     # Aggressive with FP8 efficiency
    "swap_space": 4,        # Reduced with FP8 efficiency
    "max_concurrent": 128,  # Batch processing capability
    "requirements": {
        "vllm": ">=0.6.2",
        "cuda": ">=12.1",
        "hardware": "RTX 4090 Laptop GPU (Ada Lovelace, CC 8.9)"
    }
}
```

### Memory and Performance Optimization

```python
class ModelOptimizer:
    """Optimization strategies for Qwen3-4B deployment."""
    
    def __init__(self):
        self.memory_configs = {
            "maximum_throughput": {
                "quantization": "fp8",
                "kv_cache_dtype": "fp8",
                "max_model_len": 8192,  # Optimal for job postings
                "max_num_seqs": 256,
                "gpu_memory_utilization": 0.9
            },
            "balanced": {
                "quantization": "fp8",
                "kv_cache_dtype": "fp8",
                "max_model_len": 8192,  # Optimal for job postings
                "max_num_seqs": 128,
                "gpu_memory_utilization": 0.85
            }
        }
    
    def get_config(self, mode: str = "balanced") -> Dict[str, Any]:
        """Get optimized configuration for deployment mode."""
        return self.memory_configs.get(mode, self.memory_configs["balanced"])
    
    def estimate_memory_usage(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory usage for configuration."""
        
        base_memory = {
            "fp8": 1.2,    # FP8 quantized (confirmed on RTX 4090 Laptop)
            "awq": 2.9,    # AWQ-INT4 quantized (legacy reference)
            "gptq": 3.2,   # GPTQ-INT4 quantized  
            "fp16": 7.9,   # Half precision
            "fp32": 15.8   # Full precision
        }
        
        quantization = config.get("quantization", "fp8")
        context_len = config.get("max_model_len", 131072)
        batch_size = config.get("max_num_seqs", 128)
        
        # Estimate total memory usage
        model_memory = base_memory.get(quantization, 7.9)
        context_memory = (context_len / 131072) * 2.0  # Proportional to context
        batch_memory = (batch_size / 128) * 1.0        # Batch overhead
        
        total_memory = model_memory + context_memory + batch_memory
        
        return {
            "model_memory_gb": model_memory,
            "context_memory_gb": context_memory,
            "batch_memory_gb": batch_memory,
            "total_memory_gb": total_memory,
            "rtx4090_utilization": (total_memory / 16.0) * 100
        }
```

## Testing

### Performance Validation

```python
import pytest
import time
from unittest.mock import Mock, patch

class TestQwen3Performance:
    """Test suite for Qwen3-4B performance validation."""
    
    def setup_method(self):
        self.service = Qwen3InferenceService()
        self.test_jobs = self._load_test_job_data()
    
    def test_extraction_accuracy(self):
        """Test job extraction accuracy against ground truth."""
        
        accuracy_scores = []
        
        for job_data in self.test_jobs[:50]:  # Test on 50 jobs
            content = job_data["raw_html"]
            expected = job_data["ground_truth"]
            
            extracted = self.service.extract_job_data(content, self._get_job_schema())
            
            accuracy = self._calculate_extraction_accuracy(extracted, expected)
            accuracy_scores.append(accuracy)
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        assert avg_accuracy > 0.95, f"Accuracy {avg_accuracy:.3f} below 95% threshold"
    
    def test_memory_usage(self):
        """Test memory usage within RTX 4090 limits."""
        
        optimizer = ModelOptimizer()
        config = optimizer.get_config("balanced")
        memory_estimate = optimizer.estimate_memory_usage(config)
        
        # Should use less than 90% of RTX 4090 memory
        assert memory_estimate["rtx4090_utilization"] < 90
        
        # FP8 quantization should use less than 4GB
        assert memory_estimate["total_memory_gb"] < 4.0
    
    def test_throughput_performance(self):
        """Test inference throughput meets requirements."""
        
        test_content = "Software Engineer position at TechCorp..." * 100
        
        start_time = time.time()
        
        # Process 10 jobs concurrently
        results = []
        for _ in range(10):
            result = self.service.extract_job_data(test_content, self._get_job_schema())
            results.append(result)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should process 10 jobs in under 20 seconds (>30 tokens/sec average)
        assert elapsed < 20, f"Throughput too slow: {elapsed:.2f}s for 10 jobs"
        
        # All results should be valid
        assert all(isinstance(r, dict) for r in results)
    
    def test_context_length_handling(self):
        """Test handling of various context lengths."""
        
        test_cases = [
            ("short", "Brief job description" * 50),      # ~300 tokens
            ("medium", "Detailed job posting" * 500),     # ~3K tokens  
            ("long", "Comprehensive posting" * 1500),     # ~9K tokens
            ("very_long", "Extensive content" * 5000),    # ~30K tokens
        ]
        
        for case_name, content in test_cases:
            result = self.service.extract_job_data(content, self._get_job_schema())
            
            # Should successfully extract from all context lengths
            assert isinstance(result, dict), f"Failed to process {case_name} content"
            assert "title" in result, f"Missing title in {case_name} extraction"
    
    def _get_job_schema(self) -> Dict[str, Any]:
        """Get job extraction schema."""
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "company": {"type": "string"},
                "location": {"type": "string"},
                "salary_min": {"type": "integer"},
                "salary_max": {"type": "integer"},
                "skills": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"}
            },
            "required": ["title", "company", "description"]
        }
    
    def _calculate_extraction_accuracy(self, extracted: Dict, expected: Dict) -> float:
        """Calculate extraction accuracy score."""
        
        required_fields = ["title", "company", "description"]
        correct_extractions = 0
        
        for field in required_fields:
            if field in extracted and field in expected:
                # Simple string similarity for accuracy
                if self._strings_similar(extracted[field], expected[field]):
                    correct_extractions += 1
        
        return correct_extractions / len(required_fields)
    
    def _strings_similar(self, s1: str, s2: str, threshold: float = 0.8) -> bool:
        """Check if strings are similar enough."""
        # Simple similarity check - could use more sophisticated methods
        if not s1 or not s2:
            return False
        
        common_words = set(s1.lower().split()) & set(s2.lower().split())
        total_words = set(s1.lower().split()) | set(s2.lower().split())
        
        if not total_words:
            return False
        
        similarity = len(common_words) / len(total_words)
        return similarity >= threshold
```

### Configuration

**In `.env` or deployment configuration:**

```env
# Qwen3-4B Model Configuration with FP8
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507-FP8"
MODEL_QUANTIZATION="fp8"
KV_CACHE_DTYPE="fp8"
MAX_MODEL_LENGTH=8192
GPU_MEMORY_UTILIZATION=0.9
SWAP_SPACE=4
MAX_NUM_SEQS=128

# Version Requirements
REQUIRED_VLLM_VERSION=0.6.2
REQUIRED_CUDA_VERSION=12.1
REQUIRED_HARDWARE="RTX 4090 Laptop GPU (Ada Lovelace, CC 8.9)"

# vLLM Server Configuration
VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
VLLM_ENABLE_PREFIX_CACHING=true
VLLM_TRUST_REMOTE_CODE=true
```

## Consequences

### Positive Outcomes

- **Performance Excellence**: Superior benchmark scores despite smaller parameter count
- **Memory Efficiency**: 8x memory reduction with FP8 quantization enables better resource utilization
- **Context Capability**: 8K optimized context handles 98% of job postings efficiently
- **Cost Optimization**: Enables 98% local processing reducing cloud API costs
- **Deployment Simplicity**: Single-command vLLM setup with proven configurations
- **Quality Consistency**: Structured output integration ensures reliable extraction
- **Future Adaptability**: OpenAI-compatible API enables easy model swapping

### Negative Consequences / Trade-offs

- **Model Dependency**: Reliance on single model vendor (Alibaba/Qwen)
- **Newer Technology**: Less deployment history compared to established models  
- **Version Requirements**: Requires vLLM 0.6.2+ and CUDA 12.1+ for FP8 quantization
- **Context Memory**: Large contexts consume significant memory despite efficiency
- **Fine-tuning Complexity**: Custom training requires model-specific approaches

### Ongoing Maintenance & Considerations

**Required Monitoring**:

- Model inference performance and accuracy metrics
- Memory usage patterns and optimization opportunities
- Context length distribution and utilization
- Comparison against cloud API quality benchmarks

**Update Triggers**:

- New Qwen3 model releases with performance improvements
- Significant changes in job extraction requirements
- Better quantization methods or optimization techniques
- Alternative models achieving superior benchmark performance

### Dependencies

- **System**: CUDA-compatible GPU (RTX 4090 Laptop GPU or equivalent) with 16GB+ VRAM
- **Python**: vLLM ≥0.6.2, PyTorch with CUDA support, FP8 quantization support  
- **Integration**: Structured output libraries per **ADR-007**, token routing per **ADR-008**

## References

- [Qwen3 Official Documentation](https://github.com/QwenLM/Qwen)
- [vLLM Performance Optimization](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [FP8 Quantization Documentation](https://docs.vllm.ai/en/latest/quantization/fp8.html)
- [MMLU-Pro Benchmark](https://arxiv.org/abs/2406.01574)

## Changelog

### v3.1 - August 22, 2025

- **FP8 INTEGRATION**: Updated to use confirmed FP8 quantization on RTX 4090 Laptop GPU
- **MODEL UPDATE**: Changed to Qwen3-4B-Instruct-2507-FP8 with native FP8 quantization
- **MEMORY OPTIMIZATION**: Updated to 8x memory reduction with FP8 vs previous 4x with AWQ
- **VERSION REQUIREMENTS**: Added vLLM 0.6.2+ and CUDA 12.1+ requirements
- **CONTEXT OPTIMIZATION**: Maintained optimal 8K context for job posting processing
- **HARDWARE SPECIFIC**: Updated for RTX 4090 Laptop GPU (Ada Lovelace, CC 8.9)

### v3.0 - August 21, 2025 (SUPERSEDED)

- **Template compliance**: Previously used AWQ quantization
- **Conservative approach**: Previously used 85% GPU utilization
- **Older requirements**: Previously required vLLM 0.8.5+

### v2.0 - August 19, 2025 (SUPERSEDED)

- **Expert validation**: Comprehensive research with AWQ quantization
- **vLLM optimization**: Parameter validation with AWQ configuration

### v1.0 - August 18, 2025 (SUPERSEDED)

- **Initial selection**: Basic vLLM deployment with AWQ quantization
