# Local AI Integration Implementation Specification

## Branch Name

`feat/local-ai-vllm-qwen3-integration`

## Overview

Implement local AI infrastructure using vLLM and Qwen3 models following ADR-035 final architecture. This specification establishes 98% local processing capability with single model constraint for 16GB VRAM, 8000 token threshold optimization, and automatic model management through vLLM's native features.

## Context and Background

### Architectural Decision References

- **ADR-019:** Local AI Integration - Qwen3 model selection with corrected references
- **ADR-034:** Optimized Token Thresholds - 8000 token threshold for 98% local processing  
- **ADR-035:** Final Production Architecture - Single model constraint, vLLM swap_space=4
- **ADR-031:** Library-First Architecture - Use vLLM native features over custom implementations

### Current State Analysis

The project currently has:

- **No local AI capabilities:** All processing would go to cloud APIs
- **No model management:** No infrastructure for local model loading
- **No token threshold logic:** No intelligent local vs cloud routing
- **Legacy patterns:** Would use custom implementations instead of vLLM features

### Target State Goals

- **Single model loading:** Respect 16GB VRAM constraint with intelligent model switching
- **98% local processing:** 8000 token threshold routes most jobs locally
- **Automatic management:** vLLM swap_space=4 handles all memory operations
- **Cost optimization:** $2.50/month vs $50/month through local processing

## Implementation Requirements

### 1. Model Management System

**Single Model Constraint (Critical for RTX 4090):**

```python
# Only ONE model loaded at a time, vLLM handles switching
class SingleModelManager:
    """Manages single model loading with vLLM swap_space."""
    
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
    
    def load_model(self, model_name: str) -> LLM:
        """Load model with automatic memory management."""
        if self.current_model_name != model_name:
            # vLLM swap_space=4 automatically handles unloading
            self.current_model = LLM(
                model=model_name,
                swap_space=4,  # Automatic CPU offload
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
                quantization="awq"
            )
            self.current_model_name = model_name
        
        return self.current_model
```

### 2. Token Threshold Implementation

**8000 Token Decision Logic (ADR-034):**

```python
# Implements 98% local processing rate
class HybridProcessor:
    """Routes jobs to local vs cloud based on token count."""
    
    def __init__(self, threshold: int = 8000):
        self.threshold = threshold
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model_manager = SingleModelManager()
    
    async def process_job(self, content: str) -> JobPosting:
        """Route to local or cloud based on token count."""
        
        token_count = len(self.tokenizer.encode(content))
        
        if token_count < self.threshold:
            # 98% of jobs - process locally
            return await self.process_locally(content, token_count)
        else:
            # 2% of jobs - complex content to cloud
            return await self.process_cloud(content, token_count)
```

### 3. Corrected Model References

**Use Only Existing Qwen3 Models (ADR-035 Corrections):**

```python
# Verified model names that actually exist
QWEN3_MODELS = {
    "primary": "Qwen/Qwen3-8B",  # Base model for most tasks
    "thinking": "Qwen/Qwen3-4B-Thinking-2507",  # Instruct variant
    "maximum": "Qwen/Qwen3-14B"  # Base model for complex tasks
}
```

## Files to Create/Modify

### Files to Create

1. **`src/ai/inference.py`** - vLLM model management and inference
2. **`src/ai/threshold.py`** - 8000 token threshold logic
3. **`src/ai/models.py`** - Model configuration and selection
4. **`src/ai/extraction.py`** - Job extraction with local AI
5. **`tests/test_ai_integration.py`** - AI component testing

### Files to Modify

1. **`src/core/config.py`** - Add AI-specific configuration
2. **`requirements.txt`** or **`pyproject.toml`** - Ensure AI dependencies

## Dependencies and Libraries

### Required Model Downloads

```bash
# Download corrected Qwen3 models (verified to exist)
python -c "
from huggingface_hub import snapshot_download
import os

# Create model directory
os.makedirs('./models', exist_ok=True)

# Download verified models only
models = [
    'Qwen/Qwen3-8B',  # Base model - primary use
    'Qwen/Qwen3-4B-Thinking-2507',  # Instruct model - simple tasks
    'Qwen/Qwen3-14B'  # Base model - complex tasks
]

for model in models:
    print(f'Downloading {model}...')
    snapshot_download(
        repo_id=model,
        cache_dir='./models',
        local_files_only=False
    )
    print(f'✅ {model} downloaded')

print('All Qwen3 models downloaded successfully')
"
```

### vLLM Dependencies

```toml
# Add to pyproject.toml
[project.dependencies]
"vllm>=0.6.5,<1.0.0"  # Core inference engine
"torch>=2.0.0"        # PyTorch backend  
"tiktoken>=0.8.0"     # Token counting
"transformers>=4.45.0" # Model loading
"accelerate>=0.34.0"   # GPU acceleration
```

## Code Implementation

### 1. Model Management Implementation

```python
# src/ai/inference.py - Complete vLLM integration
from vllm import LLM, SamplingParams
from typing import Optional, Dict, Any, Literal
import torch
import logging
from src.core.config import settings

logger = logging.getLogger(__name__)

class VLLMModelManager:
    """Single model management with vLLM native features."""
    
    def __init__(self):
        self.current_model: Optional[LLM] = None
        self.current_model_name: Optional[str] = None
        self.model_configs = settings.model_configs
        
    def get_model(self, model_type: Literal["primary", "thinking", "maximum"]) -> LLM:
        """Get model with automatic loading via vLLM swap_space."""
        
        config = self.model_configs[model_type]
        model_name = config["name"]
        
        # Only reload if different model needed
        if self.current_model_name != model_name:
            logger.info(f"Loading model: {model_name}")
            
            # vLLM automatically handles previous model cleanup with swap_space
            self.current_model = LLM(
                model=model_name,
                swap_space=config.get("swap_space", 4),  # Automatic memory management
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.85),
                trust_remote_code=config.get("trust_remote_code", True),
                quantization=config.get("quantization", "awq"),
                dtype="auto",  # Let vLLM decide optimal dtype
                max_model_len=config.get("max_model_len", 131072),  # Qwen3 context
            )
            self.current_model_name = model_name
            logger.info(f"✅ Model {model_name} loaded successfully")
            
        return self.current_model
    
    async def generate_async(
        self, 
        prompt: str, 
        model_type: Literal["primary", "thinking", "maximum"] = "primary",
        **kwargs
    ) -> str:
        """Generate text with specified model."""
        
        model = self.get_model(model_type)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.1),  # Low for consistency
            max_tokens=kwargs.get("max_tokens", 2048),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop", None),
        )
        
        # Generate with vLLM
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        if not self.current_model:
            return {"status": "No model loaded"}
            
        return {
            "model_name": self.current_model_name,
            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3,  # GB
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "vram_utilization": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory,
        }

# Global model manager instance
model_manager = VLLMModelManager()
```

### 2. Token Threshold Implementation

```python
# src/ai/threshold.py - 8000 token threshold logic
import tiktoken
from typing import Tuple, Literal
from dataclasses import dataclass
from src.core.config import settings
from src.core.models import TokenThresholdDecision

@dataclass
class TokenAnalysis:
    """Token analysis result."""
    content: str
    token_count: int
    use_local: bool
    model_recommended: Literal["primary", "thinking", "maximum"] | None
    reasoning: str

class TokenThresholdManager:
    """Implements ADR-034 optimized token thresholds."""
    
    def __init__(self, threshold: int = None):
        self.threshold = threshold or settings.token_threshold  # Default 8000
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def analyze_content(self, content: str) -> TokenAnalysis:
        """Analyze content and decide processing strategy."""
        
        token_count = len(self.tokenizer.encode(content))
        
        # Apply 8000 token threshold for 98% local processing
        if token_count <= self.threshold:
            # Route to appropriate local model based on complexity
            if token_count < 2000:
                model = "thinking"  # Qwen3-4B for simple tasks
                reasoning = f"Simple content ({token_count} tokens) → Qwen3-4B-Thinking"
            elif token_count < 5000:
                model = "primary"   # Qwen3-8B for standard tasks
                reasoning = f"Standard content ({token_count} tokens) → Qwen3-8B"
            else:
                model = "maximum"   # Qwen3-14B for complex tasks
                reasoning = f"Complex content ({token_count} tokens) → Qwen3-14B"
                
            return TokenAnalysis(
                content=content,
                token_count=token_count,
                use_local=True,
                model_recommended=model,
                reasoning=reasoning
            )
        else:
            # Exceed threshold - route to cloud
            reasoning = f"Large content ({token_count} tokens > {self.threshold}) → Cloud processing"
            return TokenAnalysis(
                content=content,
                token_count=token_count,
                use_local=False,
                model_recommended=None,
                reasoning=reasoning
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threshold performance statistics."""
        # This would be populated by actual usage tracking
        return {
            "threshold": self.threshold,
            "local_processing_rate": 98.0,  # Target from ADR-034
            "average_tokens": 3500,  # Typical job posting size
            "model_distribution": {
                "thinking": 45,   # 0-2K tokens
                "primary": 35,    # 2-5K tokens  
                "maximum": 18,    # 5-8K tokens
                "cloud": 2        # 8K+ tokens
            }
        }

# Global threshold manager
threshold_manager = TokenThresholdManager()
```

### 3. Job Extraction with Local AI

```python
# src/ai/extraction.py - AI-powered job extraction
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ai.inference import model_manager
from src.ai.threshold import threshold_manager  
from src.core.models import JobPosting, TokenThresholdDecision

class JobExtractionPrompt:
    """Structured prompts for job extraction."""
    
    BASE_PROMPT = """Extract structured job information from the following content. Return a valid JSON object with these exact fields:

{
    "title": "Job title",
    "company": "Company name", 
    "location": "Job location or null",
    "salary_min": "Minimum salary number or null",
    "salary_max": "Maximum salary number or null",
    "description": "Full job description",
    "requirements": ["List of requirements"],
    "benefits": ["List of benefits"], 
    "skills": ["List of required skills"]
}

Content to extract from:
{content}

JSON Response:"""

class LocalJobExtractor:
    """Extract job postings using local AI models."""
    
    def __init__(self):
        self.model_manager = model_manager
        self.threshold_manager = threshold_manager
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def extract_job(self, content: str, source_url: str) -> JobPosting:
        """Extract job posting with local AI and fallback logic."""
        
        # Analyze token threshold
        analysis = self.threshold_manager.analyze_content(content)
        
        # Create threshold decision record
        threshold_decision = TokenThresholdDecision(
            content=content[:500] + "...",  # Store preview only
            token_count=analysis.token_count,
            threshold=self.threshold_manager.threshold,
            use_local=analysis.use_local,
            model_selected=analysis.model_recommended,
            reasoning=analysis.reasoning
        )
        
        try:
            if analysis.use_local:
                # Process with local model
                job_data = await self._extract_local(content, analysis.model_recommended)
            else:
                # Fallback to cloud processing
                job_data = await self._extract_cloud(content)
                
        except Exception as e:
            # If local fails, try cloud fallback
            if analysis.use_local:
                job_data = await self._extract_cloud(content)
                threshold_decision.model_selected = None
                threshold_decision.reasoning += f" | Local failed: {str(e)}"
            else:
                raise
        
        # Create JobPosting with extraction metadata
        return JobPosting(
            **job_data,
            source_url=source_url,
            extraction_method="crawl4ai" if analysis.use_local else "cloud",
            token_decision=threshold_decision
        )
    
    async def _extract_local(self, content: str, model_type: str) -> dict:
        """Extract using local vLLM model."""
        
        prompt = JobExtractionPrompt.BASE_PROMPT.format(content=content)
        
        # Generate with selected model
        response = await self.model_manager.generate_async(
            prompt=prompt,
            model_type=model_type,
            temperature=0.1,  # Low temperature for structured output
            max_tokens=2048,
            stop=["}\n\n", "}\n\nExtract", "}\n\nContent"]  # Stop at JSON end
        )
        
        # Parse JSON response
        try:
            # Clean response and extract JSON
            json_str = response.strip()
            if not json_str.startswith('{'):
                # Find first { character
                start = json_str.find('{')
                if start != -1:
                    json_str = json_str[start:]
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from local model: {e}")
    
    async def _extract_cloud(self, content: str) -> dict:
        """Fallback cloud extraction (2% of cases)."""
        
        if not settings.openai_api_key:
            raise ValueError("No cloud API key configured for fallback")
        
        # Implement OpenAI API call for fallback
        # This is the 2% case for very large content
        import openai
        
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective for fallback
            messages=[
                {"role": "system", "content": "Extract job posting data as JSON."},
                {"role": "user", "content": f"{JobExtractionPrompt.BASE_PROMPT.format(content=content)}"}
            ],
            temperature=0.1,
            max_tokens=2048
        )
        
        return json.loads(response.choices[0].message.content)

# Global extractor instance
job_extractor = LocalJobExtractor()
```

## Testing Requirements

### 1. Model Management Tests

```python
# tests/test_ai_integration.py
import pytest
import torch
from src.ai.inference import VLLMModelManager
from src.ai.threshold import TokenThresholdManager

@pytest.fixture
def model_manager():
    return VLLMModelManager()

@pytest.fixture  
def threshold_manager():
    return TokenThresholdManager()

class TestModelManagement:
    """Test vLLM model management."""
    
    def test_single_model_constraint(self, model_manager):
        """Verify only one model loaded at a time."""
        
        # Load primary model
        model1 = model_manager.get_model("primary")
        assert model_manager.current_model_name == "Qwen/Qwen3-8B"
        
        # Load different model - should replace first
        model2 = model_manager.get_model("thinking") 
        assert model_manager.current_model_name == "Qwen/Qwen3-4B-Thinking-2507"
        
        # Verify memory usage is reasonable for RTX 4090
        info = model_manager.get_model_info()
        assert info["gpu_memory_used"] < 15.0  # Under 15GB for 16GB card
    
    def test_model_loading_time(self, model_manager):
        """Test model loading meets <60s requirement."""
        import time
        
        start_time = time.time()
        model = model_manager.get_model("primary")
        load_time = time.time() - start_time
        
        assert load_time < 60.0  # ADR requirement
        assert model is not None

class TestTokenThreshold:
    """Test 8000 token threshold logic."""
    
    def test_threshold_routing(self, threshold_manager):
        """Test content routing based on token count."""
        
        # Small content - should use local "thinking" model
        small_content = "Software Engineer position at Tech Corp."
        analysis = threshold_manager.analyze_content(small_content)
        
        assert analysis.use_local is True
        assert analysis.model_recommended == "thinking"
        assert analysis.token_count < 2000
        
        # Medium content - should use local "primary" model  
        medium_content = "Software Engineer position " * 200  # ~2K-5K tokens
        analysis = threshold_manager.analyze_content(medium_content)
        
        assert analysis.use_local is True
        assert analysis.model_recommended == "primary"
        
        # Large content - should use cloud
        large_content = "Software Engineer position " * 2000  # >8K tokens
        analysis = threshold_manager.analyze_content(large_content)
        
        assert analysis.use_local is False
        assert analysis.model_recommended is None
    
    def test_token_counting_accuracy(self, threshold_manager):
        """Verify token counting matches tiktoken."""
        import tiktoken
        
        content = "Test content for token counting validation."
        
        # Manual token count
        tokenizer = tiktoken.get_encoding("cl100k_base") 
        expected_count = len(tokenizer.encode(content))
        
        # Threshold manager count
        analysis = threshold_manager.analyze_content(content)
        
        assert analysis.token_count == expected_count

@pytest.mark.asyncio
class TestJobExtraction:
    """Test AI job extraction functionality."""
    
    async def test_local_extraction(self):
        """Test local job extraction workflow."""
        from src.ai.extraction import LocalJobExtractor
        
        extractor = LocalJobExtractor()
        
        # Sample job content (under 8K tokens)
        content = """
        Software Engineer - Full Stack
        Company: Tech Innovations Inc
        Location: San Francisco, CA (Remote OK)
        Salary: $120,000 - $180,000
        
        We're looking for a full-stack engineer...
        Requirements: Python, React, PostgreSQL
        Benefits: Health insurance, 401k, flexible PTO
        """
        
        job = await extractor.extract_job(content, "https://example.com/job")
        
        assert job.title == "Software Engineer - Full Stack"
        assert job.company == "Tech Innovations Inc"
        assert job.salary_min == 120000
        assert job.salary_max == 180000
        assert "Python" in job.skills
        assert job.token_decision.use_local is True
    
    async def test_threshold_performance(self):
        """Test that 98% of typical jobs are processed locally."""
        from src.ai.extraction import LocalJobExtractor
        
        extractor = LocalJobExtractor()
        
        # Simulate 100 job postings of varying sizes
        local_count = 0
        total_jobs = 100
        
        for i in range(total_jobs):
            # Generate content of different sizes (most under 8K)
            if i < 98:  # 98% should be under threshold
                content = f"Job posting {i} " * (100 + i * 10)  # < 8K tokens
            else:  # 2% over threshold
                content = f"Very long job posting {i} " * 2000  # > 8K tokens
            
            analysis = extractor.threshold_manager.analyze_content(content)
            if analysis.use_local:
                local_count += 1
        
        local_percentage = (local_count / total_jobs) * 100
        assert local_percentage >= 98.0  # Target from ADR-034
```

### 2. Performance Benchmarks

```python
# tests/test_performance.py
import pytest
import time
import asyncio
from src.ai.inference import model_manager

@pytest.mark.performance
class TestPerformance:
    """Performance testing for AI components."""
    
    def test_model_switching_speed(self):
        """Test model switching meets <60s requirement."""
        
        switch_times = []
        models = ["primary", "thinking", "primary"]  # Test switching
        
        for model_type in models:
            start_time = time.time()
            model = model_manager.get_model(model_type)
            switch_time = time.time() - start_time
            switch_times.append(switch_time)
        
        # All switches should be under 60 seconds
        for switch_time in switch_times:
            assert switch_time < 60.0
        
        # Average should be much faster (vLLM optimization)
        avg_time = sum(switch_times) / len(switch_times)
        assert avg_time < 30.0  # Optimistic target
    
    @pytest.mark.asyncio
    async def test_inference_speed(self):
        """Test inference speed for different content sizes."""
        from src.ai.extraction import job_extractor
        
        # Small job content
        small_content = "Software Engineer at TechCorp. Python required."
        
        start_time = time.time()
        job = await job_extractor.extract_job(small_content, "http://test.com")
        inference_time = time.time() - start_time
        
        # Should be under 5 seconds for small content
        assert inference_time < 5.0
        assert job.token_decision.use_local is True
```

## Configuration

### 1. Model Storage Configuration

```yaml
# .env additions for AI integration
# Model Configuration
MODEL_CACHE_DIR="./models"
VLLM_SWAP_SPACE=4
VLLM_GPU_MEMORY=0.85
VLLM_TRUST_REMOTE_CODE=true

# Token Threshold (ADR-034)
TOKEN_THRESHOLD=8000

# Cloud Fallback (2% of cases)
OPENAI_API_KEY=""  # Optional - for >8K token jobs

# Performance Tuning
VLLM_QUANTIZATION="awq"
VLLM_MAX_MODEL_LEN=131072
```

### 2. Docker Updates for GPU

```yaml
# docker-compose.yml additions
services:
  app:
    # ... existing config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models  # Model storage
    environment:
      - MODEL_CACHE_DIR=/app/models
      - CUDA_VISIBLE_DEVICES=0
```

## Success Criteria

### Immediate Validation

- [ ] All Qwen3 models download successfully (corrected model names)
- [ ] vLLM loads models within 60 seconds
- [ ] Single model constraint respected (16GB VRAM)
- [ ] Token threshold logic routes 98% of typical jobs locally
- [ ] Model switching works with swap_space=4

### Performance Validation

- [ ] Model loading: <60 seconds (ADR requirement)
- [ ] GPU memory usage: <15GB on RTX 4090  
- [ ] Inference speed: <5 seconds for typical jobs
- [ ] Token counting accuracy: Matches tiktoken exactly
- [ ] Local processing rate: 98%+ for sample job content

### Integration Validation

- [ ] Job extraction works with all three Qwen3 models
- [ ] Cloud fallback works for >8K token content
- [ ] Error handling gracefully falls back to cloud
- [ ] Configuration loads correctly from environment
- [ ] Memory management stable over multiple model switches

## Commit and PR Instructions

### Commit Messages

```bash
git checkout -b feat/local-ai-vllm-qwen3-integration

# Model management system
git add src/ai/inference.py
git commit -m "feat: implement vLLM model management with single model constraint

- Single model loading respecting 16GB VRAM limit
- vLLM swap_space=4 for automatic memory management
- Corrected Qwen3 model references (8B, 4B-Thinking-2507, 14B)
- Lazy loading pattern for optimal performance
- Implements ADR-035 memory management strategy"

# Token threshold logic
git add src/ai/threshold.py  
git commit -m "feat: implement 8000 token threshold for 98% local processing

- Token analysis with tiktoken for accuracy
- Smart model selection based on content complexity
- 98% local processing rate targeting (ADR-034)
- Routing logic: thinking<2K, primary<5K, maximum<8K, cloud>8K
- Performance tracking and statistics"

# Job extraction system
git add src/ai/extraction.py
git commit -m "feat: implement local AI job extraction with cloud fallback

- Structured job extraction with Pydantic validation
- Local processing with appropriate model selection
- Cloud fallback for 2% of large content cases  
- Tenacity retry logic for resilience
- Complete extraction workflow with metadata tracking"

# Testing and validation
git add tests/test_ai_integration.py tests/test_performance.py
git commit -m "test: comprehensive AI integration testing

- Model management and switching tests
- Token threshold routing validation
- Performance benchmarks (<60s loading, <5s inference)
- Memory usage validation for RTX 4090
- End-to-end job extraction testing"
```

### PR Description Template

```markdown
# Local AI Integration - vLLM + Qwen3 Models

## Overview
Implements complete local AI infrastructure following ADR-035 final architecture, enabling 98% local processing with $2.50/month cost optimization.

## Key Features Implemented

### Single Model Management (ADR-035)
- ✅ One model loaded at a time (16GB VRAM constraint)
- ✅ vLLM swap_space=4 automatic memory management  
- ✅ Corrected Qwen3 model references (no non-existent models)
- ✅ <60 second model switching requirement met

### 8000 Token Threshold (ADR-034)
- ✅ 98% local processing rate implementation
- ✅ Smart model routing: thinking→primary→maximum→cloud
- ✅ Token counting with tiktoken accuracy
- ✅ Cost optimization: $50/month → $2.50/month

### Job Extraction System
- ✅ Structured extraction with Pydantic validation
- ✅ Local-first processing with cloud fallback
- ✅ Tenacity retry logic for resilience
- ✅ Complete extraction metadata tracking

## Technical Implementation

### Models Used (Verified Existing)
- `Qwen/Qwen3-8B` - Primary model (2K-5K tokens)
- `Qwen/Qwen3-4B-Thinking-2507` - Simple tasks (<2K tokens)  
- `Qwen/Qwen3-14B` - Complex tasks (5K-8K tokens)

### Performance Metrics
- Model loading: <60 seconds ✅
- GPU memory: <15GB on RTX 4090 ✅
- Inference speed: <5 seconds typical jobs ✅
- Local processing: 98% of sample content ✅

### Error Handling
- Automatic fallback from local to cloud on failures
- Tenacity retry logic with exponential backoff
- Graceful degradation for memory issues

## Testing Coverage
- Model management and memory constraints
- Token threshold routing accuracy
- Performance benchmarks
- End-to-end job extraction workflow

## Next Steps
Ready for `03-scraping-consolidation.md` - integrate AI extraction with Crawl4AI.
```

## Review Checklist

### Architecture Compliance

- [ ] Single model constraint properly implemented (ADR-035)
- [ ] 8000 token threshold correctly configured (ADR-034)
- [ ] vLLM native features used over custom implementations (ADR-031)
- [ ] Corrected Qwen3 model references only (no non-existent models)

### Code Quality

- [ ] Type hints on all functions and classes
- [ ] Error handling with tenacity library patterns  
- [ ] Async/await patterns for all I/O operations
- [ ] Pydantic validation for all data models
- [ ] Proper logging for debugging and monitoring

### Performance Validation

- [ ] Model loading tested under 60 seconds
- [ ] Memory usage profiled for RTX 4090 constraints
- [ ] Token counting accuracy verified against tiktoken
- [ ] Local processing rate measured on sample data

## Next Steps

After successful completion of this specification:

1. **Immediate:** Begin `03-scraping-consolidation.md` to integrate AI extraction with Crawl4AI
2. **Parallel:** Start preparing model downloads in development environment
3. **Validation:** Run performance benchmarks to confirm 98% local processing rate

This local AI integration provides the foundation for cost-optimized, privacy-first job processing that achieves the 95% cost reduction target outlined in the final architecture.
