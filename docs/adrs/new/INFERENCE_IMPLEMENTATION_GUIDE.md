# RTX 4090 Laptop Inference Implementation Guide

## Quick Start

This guide provides step-by-step instructions to set up the optimal inference stack for your RTX 4090 laptop GPU for the AI job scraper.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [vLLM Installation](#vllm-installation)
4. [Model Setup](#model-setup)
5. [Structured Output Configuration](#structured-output-configuration)
6. [Production Deployment](#production-deployment)
7. [RTX 4090 Laptop Optimization](#rtx-4090-laptop-optimization)
8. [Monitoring & Optimization](#monitoring--optimization)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4090 Laptop (16GB VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models
- **OS**: Ubuntu 22.04 or Windows 11 with WSL2

### Software Prerequisites
```bash
# Check CUDA version (should be 12.1+)
nvidia-smi

# Check Python version (should be 3.10-3.12)
python --version

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

### 1. Create Project Structure
```bash
mkdir -p ai-job-scraper/{models,configs,logs,data}
cd ai-job-scraper

# Initialize Python environment
uv init
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Configure Power & Thermal Management
```bash
# Set conservative power limit for sustained operation
sudo nvidia-smi -pl 120  # 120W power limit

# Enable persistence mode
sudo nvidia-smi -pm 1

# Check current settings
nvidia-smi -q -d POWER
```

### 3. Create pyproject.toml
```toml
[project]
name = "ai-job-scraper"
version = "1.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "vllm==0.6.5",
    "outlines>=0.1.0",
    "pydantic>=2.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pynvml>=11.5.0",
    "huggingface-hub>=0.20.0",
    "tenacity>=8.0.0",
    "structlog>=24.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.5.0",
]
```

## vLLM Installation

### 1. Install vLLM with Flash Attention 2
```bash
# Install vLLM
uv add vllm==0.6.5

# Install Flash Attention 2 (crucial for performance)
uv add flash-attn --no-build-isolation

# Verify installation
python -c "from vllm import LLM; print('vLLM installed successfully')"
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} installed')"
```

### 2. Test vLLM with Flash Attention 2
```python
# test_vllm.py
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

from vllm import LLM, SamplingParams

# Quick test with small model
llm = LLM(
    model="microsoft/Phi-3.5-mini-instruct",
    dtype="half",
    gpu_memory_utilization=0.5
)

prompt = "What is Flash Attention?"
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)
```

## Model Setup

### 1. Download Recommended Model
```python
# download_model.py
from huggingface_hub import snapshot_download
import os

# Primary model: Qwen3-8B-Instruct (best balance per ADR-019)
model_id = "Qwen/Qwen3-8B-Instruct"
cache_dir = "./models"

print(f"Downloading {model_id}...")
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    revision="main"
)

# Also download thinking model for complex reasoning
thinking_model_id = "Qwen/Qwen3-4B-Thinking-2507"
print(f"Downloading {thinking_model_id}...")
snapshot_download(
    repo_id=thinking_model_id,
    cache_dir=cache_dir,
    revision="main"
)

print(f"Models downloaded to {cache_dir}")
```

### 2. Create Model Configuration
```python
# src/inference/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceConfig:
    """RTX 4090 Laptop optimized configuration for Qwen3-2507 series"""
    model_path: str = "Qwen/Qwen3-8B-Instruct"  # Updated to Qwen3
    quantization: str = "gptq"  # GPTQ-8bit for Qwen3-8B per ADR-019
    dtype: str = "half"
    max_model_len: int = 131072  # Qwen3-8B supports 131K context
    gpu_memory_utilization: float = 0.85
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    trust_remote_code: bool = True
    
    # Thermal management
    power_limit_watts: int = 120
    max_temperature_c: int = 80
    
    # Flash Attention 2
    attention_backend: str = "FLASH_ATTN"
    
    def to_vllm_kwargs(self) -> dict:
        """Convert to vLLM initialization kwargs"""
        return {
            "model": self.model_path,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": False,  # Enable CUDA graphs
        }
```

### 3. Initialize Inference Engine
```python
# src/inference/engine.py
import os
from vllm import LLM, SamplingParams
from src.inference.config import InferenceConfig
import pynvml

class RTX4090InferenceEngine:
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self._setup_environment()
        self._init_monitoring()
        self.llm = self._init_model()
        
    def _setup_environment(self):
        """Configure environment for RTX 4090"""
        os.environ["VLLM_ATTENTION_BACKEND"] = self.config.attention_backend
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    def _init_monitoring(self):
        """Initialize GPU monitoring"""
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
    def _init_model(self) -> LLM:
        """Initialize vLLM with optimal settings"""
        print(f"Loading Qwen3 model: {self.config.model_path} (see ADR-019 for rationale)")
        return LLM(**self.config.to_vllm_kwargs())
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1
    ) -> str:
        """Generate text with thermal monitoring"""
        # Check temperature before generation
        temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0)
        if temp > self.config.max_temperature_c:
            print(f"Warning: GPU temperature {temp}°C exceeds limit")
            
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            top_k=20
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def get_gpu_stats(self) -> dict:
        """Get current GPU statistics"""
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        return {
            "temperature": pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0),
            "power_draw": pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000,
            "vram_used_gb": mem_info.used / 1e9,
            "vram_free_gb": mem_info.free / 1e9,
            "gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
        }
```

## Structured Output Configuration

### 1. Install Outlines
```bash
uv add outlines
```

### 2. Define Job Schema
```python
# src/schemas/job.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class JobType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"

class RemoteType(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"

class Salary(BaseModel):
    min: Optional[int] = Field(None, ge=0, le=1000000)
    max: Optional[int] = Field(None, ge=0, le=1000000)
    currency: str = Field("USD", pattern="^[A-Z]{3}$")
    period: str = Field("yearly", pattern="^(hourly|monthly|yearly)$")

class JobPosting(BaseModel):
    """Structured job posting with all relevant fields"""
    # Core fields
    title: str = Field(..., min_length=1, max_length=200)
    company: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    
    # Job details
    job_type: JobType = JobType.FULL_TIME
    remote_type: RemoteType = RemoteType.ONSITE
    experience_level: ExperienceLevel = ExperienceLevel.MID
    
    # Compensation
    salary: Optional[Salary] = None
    
    # Requirements
    skills: List[str] = Field(default_factory=list, max_items=20)
    experience_years_min: Optional[int] = Field(None, ge=0, le=50)
    experience_years_max: Optional[int] = Field(None, ge=0, le=50)
    education: Optional[str] = Field(None, max_length=100)
    
    # Description
    description: str = Field(..., min_length=10, max_length=5000)
    requirements: List[str] = Field(default_factory=list, max_items=30)
    responsibilities: List[str] = Field(default_factory=list, max_items=30)
    benefits: List[str] = Field(default_factory=list, max_items=20)
    
    # Metadata
    url: Optional[str] = None
    posted_date: Optional[str] = None  # ISO 8601
    deadline: Optional[str] = None  # ISO 8601
    job_id: Optional[str] = None
```

### 3. Create Structured Extractor
```python
# src/extraction/extractor.py
from outlines import models, generate
from src.schemas.job import JobPosting
from src.inference.engine import RTX4090InferenceEngine
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class JobExtractor:
    def __init__(self, engine: RTX4090InferenceEngine):
        self.engine = engine
        
        # Wrap vLLM with Outlines
        self.model = models.VLLM(engine.llm)
        
        # Create structured generator
        self.generator = generate.json(self.model, JobPosting)
        
    def extract(self, html_content: str, url: Optional[str] = None) -> JobPosting:
        """Extract job posting from HTML with guaranteed structure"""
        
        # Truncate content if too long
        max_content_length = 8000
        if len(html_content) > max_content_length:
            html_content = html_content[:max_content_length]
        
        prompt = self._build_prompt(html_content, url)
        
        try:
            # Generate with schema constraints
            logger.info(f"Extracting job from {url or 'content'}")
            job = self.generator(prompt, max_tokens=2048)
            
            # Post-process
            if url:
                job.url = url
                
            logger.info(f"Successfully extracted: {job.title} at {job.company}")
            return job
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
            
    def _build_prompt(self, html: str, url: Optional[str]) -> str:
        """Build extraction prompt"""
        return f"""You are a job posting extractor. Extract structured information from the HTML below.

Instructions:
1. Extract all relevant job information
2. Infer job type, remote status, and experience level from context
3. Parse salary information if present
4. Extract key skills, requirements, and benefits
5. Keep descriptions concise but complete

URL: {url or 'N/A'}

HTML Content:
{html}

Extract the job information into a structured JSON format."""

    def batch_extract(self, contents: List[tuple[str, str]]) -> List[JobPosting]:
        """Extract multiple jobs efficiently"""
        prompts = [self._build_prompt(html, url) for html, url in contents]
        return self.generator(prompts, max_tokens=2048)
```

## Production Deployment

### 1. Create FastAPI Service
```python
# src/api/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.inference.engine import RTX4090InferenceEngine
from src.extraction.extractor import JobExtractor
from src.schemas.job import JobPosting
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(title="AI Job Scraper API")

# Global instances
engine = None
extractor = None

class ExtractionRequest(BaseModel):
    html_content: str
    url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    gpu_temperature: float
    vram_used_gb: float
    power_draw_watts: float

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global engine, extractor
    
    logger.info("Starting inference engine...")
    engine = RTX4090InferenceEngine()
    extractor = JobExtractor(engine)
    logger.info("Engine ready")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    stats = engine.get_gpu_stats()
    return HealthResponse(
        status="healthy" if stats["temperature"] < 85 else "throttling",
        gpu_temperature=stats["temperature"],
        vram_used_gb=stats["vram_used_gb"],
        power_draw_watts=stats["power_draw"]
    )

@app.post("/extract", response_model=JobPosting)
async def extract_job(request: ExtractionRequest):
    """Extract job from HTML"""
    if not extractor:
        raise HTTPException(status_code=503, detail="Extractor not initialized")
    
    try:
        job = extractor.extract(request.html_content, request.url)
        return job
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_extract")
async def batch_extract(
    requests: List[ExtractionRequest],
    background_tasks: BackgroundTasks
):
    """Extract multiple jobs"""
    if not extractor:
        raise HTTPException(status_code=503, detail="Extractor not initialized")
    
    # Add to background task queue
    job_id = generate_job_id()
    background_tasks.add_task(
        process_batch_extraction,
        job_id,
        requests
    )
    
    return {"job_id": job_id, "status": "processing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Create Docker Configuration
```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync

# Expose port
EXPOSE 8000

# Set environment variables
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV CUDA_VISIBLE_DEVICES=0

# Run server
CMD [".venv/bin/python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  inference:
    build: .
    container_name: job-scraper-inference
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - VLLM_ATTENTION_BACKEND=FLASH_ATTN
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## RTX 4090 Laptop Optimization

### Hardware Specification Comparison

| Specification | RTX 4090 Desktop | RTX 4090 Laptop | Impact |
|--------------|------------------|-----------------|---------|
| CUDA Cores | 16,384 | 9,728 | ~40% fewer cores |
| Memory | 24GB GDDR6X | 16GB GDDR6 | 33% less VRAM |
| Memory Bandwidth | 1008 GB/s | 576 GB/s | 43% lower bandwidth |
| TGP | 450W | 80-150W | 67-82% less power |
| Boost Clock | 2.52 GHz | 2.04 GHz | Lower frequencies |

### Thermal Management

```python
# src/monitoring/thermal.py
import pynvml
import subprocess
import time

class ThermalManager:
    """Manage RTX 4090 Laptop thermals."""
    
    def __init__(self):
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Laptop-specific thresholds
        self.thermal_limits = {
            "optimal": 70,      # Best performance/longevity balance
            "warning": 80,      # Performance may degrade
            "throttle": 87,     # Thermal throttling begins
            "critical": 95      # System protection kicks in
        }
    
    def get_temperature(self):
        """Get current GPU temperature."""
        return pynvml.nvmlDeviceGetTemperature(self.gpu, 0)
    
    def get_power_draw(self):
        """Get current power consumption in watts."""
        return pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000
    
    def adjust_for_thermals(self):
        """Dynamically adjust power based on temperature."""
        temp = self.get_temperature()
        
        if temp > self.thermal_limits["warning"]:
            # Reduce power limit to cool down
            subprocess.run(["nvidia-smi", "-pl", "100"])
            return "reduced_power"
        elif temp < self.thermal_limits["optimal"]:
            # Can increase power for better performance
            subprocess.run(["nvidia-smi", "-pl", "140"])
            return "normal_power"
        
        return "maintaining"
```

### Power Management

```bash
# Power limit optimization for sustained workloads
# Conservative setting for sustained workloads
nvidia-smi -pl 100  # 100W - Cool and quiet

# Balanced performance
nvidia-smi -pl 120  # 120W - Good performance, manageable thermals

# Maximum performance (short bursts only)
nvidia-smi -pl 150  # 150W - Will thermal throttle quickly

# Check current power limit
nvidia-smi -q -d POWER
```

### Laptop-Optimized vLLM Configuration

```python
# src/inference/laptop_config.py
import os
from vllm import LLM

class LaptopOptimizedvLLM:
    """vLLM configuration optimized for RTX 4090 Laptop."""
    
    def __init__(self, model_path: str):
        # Laptop-specific environment variables
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # FA2
        os.environ["VLLM_USE_CUDA_GRAPH"] = "true"
        
        # Reduce memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        
        # Conservative GPU memory usage for thermal headroom
        self.config = {
            "model": model_path,
            "dtype": "half",
            "max_model_len": 16384,  # Reduced from 32768
            "gpu_memory_utilization": 0.85,  # Leave headroom
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 16384,  # Reduced for laptop
            "max_num_seqs": 128,  # Reduced from 256
            
            # Laptop-specific optimizations
            "disable_log_stats": True,  # Reduce overhead
            "disable_custom_all_reduce": True,
        }
        
        self.llm = LLM(**self.config)
```

### Model-Specific Settings for 16GB VRAM

```python
LAPTOP_MODEL_CONFIGS = {
    "qwen3-4b-instruct-2507": {
        "quantization": None,  # FP16 fits in 7.8GB
        "max_model_len": 262144,  # Full 262K context
        "gpu_memory_utilization": 0.5,
        "expected_tokens_sec": "300-350",
    },
    "qwen3-4b-thinking-2507": {
        "quantization": "Q5_K_M",  # Optimized for reasoning
        "max_model_len": 262144,  # Full 262K context
        "gpu_memory_utilization": 0.3,  # Uses 4.5GB
        "expected_tokens_sec": "300-330",
    },
    "qwen3-8b-instruct": {
        "quantization": "gptq",  # GPTQ 8-bit optimal
        "max_model_len": 131072,  # Full 131K context
        "gpu_memory_utilization": 0.5,  # Uses 7.9GB
        "expected_tokens_sec": "180-220",
    },
    "qwen3-14b-instruct": {
        "quantization": "awq",  # AWQ 4-bit for memory
        "max_model_len": 131072,  # Full 131K context
        "gpu_memory_utilization": 0.6,  # Uses 9.2GB
        "expected_tokens_sec": "140-160",
    },
    "qwen3-30b-a3b-instruct-2507": {
        "quantization": "awq",  # AWQ 4-bit required
        "max_model_len": 262144,  # Full 262K context
        "gpu_memory_utilization": 0.95,  # Uses 15.5GB
        "expected_tokens_sec": "80-100",
    }
}
```

### Performance Expectations (Realistic)

| Model | Quantization | Power Mode | Tokens/sec | Sustained | Notes |
|-------|--------------|------------|------------|-----------|--------|
| Qwen3-4B-Instruct-2507 | FP16 | 120W | 300-350 | Yes | Best speed |
| Qwen3-4B-Thinking-2507 | Q5_K_M | 120W | 300-330 | Yes | Complex reasoning |
| Qwen3-8B-Instruct | GPTQ-8bit | 120W | 180-220 | Yes | **Recommended** |
| Qwen3-14B-Instruct | AWQ-4bit | 120W | 140-160 | Yes | Max quality |
| Qwen3-30B-A3B-Instruct | AWQ-4bit | 120W | 80-100 | Thermal limit | Maximum capability |

### Cooling Best Practices

1. **Physical Setup**:
   - Use laptop cooling pad
   - Elevate laptop for airflow
   - Clean fans regularly
   - Avoid blocking vents

2. **Software Configuration**:
   ```bash
   # Set conservative power limit
   nvidia-smi -pl 120  # Sustainable 120W
   
   # Monitor continuously
   watch -n 1 nvidia-smi
   ```

3. **Thermal Monitoring Script**:
   ```python
   # monitor_gpu.py
   import pynvml
   import time
   
   pynvml.nvmlInit()
   gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
   
   while True:
       temp = pynvml.nvmlDeviceGetTemperature(gpu, 0)
       power = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000
       
       status = "🟢" if temp < 75 else "🟡" if temp < 85 else "🔴"
       print(f"\r{status} {temp}°C | {power:.0f}W", end="")
       
       if temp >= 87:
           print("\n⚠️  THERMAL THROTTLING!")
       
       time.sleep(1)
   ```

## Monitoring & Optimization

### 1. Create Monitoring Script
```python
# src/monitoring/monitor.py
import time
import pynvml
from datetime import datetime
import json
from pathlib import Path

class GPUMonitor:
    def __init__(self, log_dir: str = "./logs"):
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def collect_metrics(self):
        """Collect current GPU metrics"""
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature_c": pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0),
            "power_watts": pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000,
            "gpu_utilization": util.gpu,
            "memory_utilization": util.memory,
            "vram_used_gb": mem_info.used / 1e9,
            "vram_free_gb": mem_info.free / 1e9,
            "clock_mhz": pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, 0)
        }
    
    def monitor_continuous(self, interval: int = 5):
        """Monitor continuously and log metrics"""
        log_file = self.log_dir / f"gpu_metrics_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        
        print(f"Monitoring GPU... Logging to {log_file}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                metrics = self.collect_metrics()
                
                # Display
                print(f"\r🌡️ {metrics['temperature_c']}°C | "
                      f"⚡ {metrics['power_watts']:.0f}W | "
                      f"💾 {metrics['vram_used_gb']:.1f}GB | "
                      f"🔧 {metrics['gpu_utilization']}%", end="")
                
                # Log to file
                with open(log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                # Alert on high temperature
                if metrics['temperature_c'] > 85:
                    print("\n⚠️  WARNING: High temperature detected!")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            
if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.monitor_continuous()
```

### 2. Performance Optimization Script
```python
# src/optimization/benchmark.py
import time
from src.inference.engine import RTX4090InferenceEngine
from src.extraction.extractor import JobExtractor
import statistics

def benchmark_extraction(extractor: JobExtractor, sample_html: str, runs: int = 10):
    """Benchmark extraction performance with Qwen3 models"""
    times = []
    
    print(f"Running {runs} extraction benchmarks with Qwen3...")
    
    for i in range(runs):
        start = time.perf_counter()
        job = extractor.extract(sample_html)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}s - {job.title[:30]}...")
    
    print("\n=== Qwen3 Benchmark Results ===")
    print(f"Average: {statistics.mean(times):.2f}s")
    print(f"Median: {statistics.median(times):.2f}s")
    print(f"Min: {min(times):.2f}s")
    print(f"Max: {max(times):.2f}s")
    print(f"Std Dev: {statistics.stdev(times):.2f}s")
    
    # Calculate tokens/sec (approximate)
    avg_tokens = 500  # Approximate output tokens
    avg_time = statistics.mean(times)
    tokens_per_sec = avg_tokens / avg_time
    print(f"Estimated: {tokens_per_sec:.0f} tokens/sec")
    print(f"Expected for Qwen3-8B (base): 180-220 tokens/sec (see corrected ADR-019)")

def optimize_batch_size(engine: RTX4090InferenceEngine):
    """Find optimal batch size for your GPU"""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Test with dummy prompts
            prompts = ["Extract job information"] * batch_size
            
            start = time.perf_counter()
            outputs = engine.llm.generate(prompts, max_tokens=500)
            elapsed = time.perf_counter() - start
            
            throughput = batch_size / elapsed
            results[batch_size] = throughput
            
            print(f"Batch {batch_size}: {throughput:.2f} req/s")
            
        except Exception as e:
            print(f"Batch {batch_size}: Failed - {e}")
            break
    
    optimal = max(results, key=results.get)
    print(f"\nOptimal batch size: {optimal} ({results[optimal]:.2f} req/s)")
    return optimal
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```python
# Reduce memory usage
config = InferenceConfig(
    gpu_memory_utilization=0.75,  # Reduce from 0.85
    max_model_len=8192,  # Reduce from 16384
    max_num_seqs=32  # Reduce from 64
)
```

#### 2. Thermal Throttling
```bash
# Reduce power limit
sudo nvidia-smi -pl 100  # Reduce to 100W

# Improve cooling
# - Elevate laptop for better airflow
# - Use cooling pad
# - Clean fans and vents
```

#### 3. Flash Attention Not Working
```python
# Verify Flash Attention support
import torch
print(f"CUDA capability: {torch.cuda.get_device_capability()}")
# Should be (8, 9) for RTX 4090

# Reinstall with specific CUDA version
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

#### 4. Slow Token Generation
```python
# Enable optimizations
os.environ["VLLM_USE_CUDA_GRAPH"] = "true"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # RTX 4090 architecture

# Use better quantization
config.quantization = "awq"  # AWQ is faster than GPTQ
```

#### 5. Invalid JSON Output
```python
# Use stricter schema validation
from pydantic import ValidationError

try:
    job = extractor.extract(html)
except ValidationError as e:
    print(f"Schema validation failed: {e}")
    # Fall back to simpler schema
    job = simple_extractor.extract(html)
```

## Performance Tuning Checklist

- [ ] Power limit set to 120W for sustained operation
- [ ] Flash Attention 2 enabled and verified
- [ ] Using AWQ quantization for best speed
- [ ] GPU memory utilization at 85%
- [ ] Prefix caching enabled for repeated prompts
- [ ] CUDA graphs enabled (enforce_eager=False)
- [ ] Batch size optimized for workload
- [ ] Temperature monitoring active
- [ ] Logging configured for production
- [ ] Health checks implemented

## Next Steps

1. **Fine-tuning**: Consider fine-tuning Qwen3-8B (base) on job data with structured prompting
2. **Thinking Mode**: Leverage Qwen3-4B-Thinking-2507 (✅ available) for complex extractions
3. **Extended Context**: Utilize Qwen3's 131K-262K context windows
4. **Structured Prompting**: Develop robust prompting strategies for base models
4. **Multi-GPU**: Add second RTX 4090 for Qwen3-30B-A3B models
5. **Caching**: Implement Redis caching for repeated extractions
6. **Monitoring**: Set up Prometheus + Grafana dashboards
7. **Scaling**: Deploy multiple instances with load balancing

## Support & Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Outlines Documentation](https://github.com/outlines-dev/outlines)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [RTX 4090 Optimization Guide](https://developer.nvidia.com/rtx)

## License
MIT License - See LICENSE file for details