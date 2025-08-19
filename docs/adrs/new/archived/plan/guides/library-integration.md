# Library Integration Quick Reference Guide

## Overview

This guide provides quick reference for integrating the key libraries used in the AI Job Scraper following the library-first architecture approach from ADR-031.

## Core Libraries

### vLLM - Local AI Inference

**Purpose:** Local model management and inference with automatic memory handling

**Key Features:**

- `swap_space=4`: Automatic CPU/GPU memory management
- `gpu_memory_utilization=0.85`: Optimal VRAM usage
- Single model constraint for 16GB VRAM

**Basic Usage:**

```python
from vllm import LLM

# Initialize with memory management
model = LLM(
    model="Qwen/Qwen3-8B",
    swap_space=4,  # Automatic model swapping
    gpu_memory_utilization=0.85,
    trust_remote_code=True
)

# Generate text
outputs = model.generate(["prompt"], SamplingParams(temperature=0.1))
```

### Reflex - Real-Time UI Framework

**Purpose:** Modern web UI with native WebSocket support

**Key Features:**

- Native WebSocket updates via `yield`
- Mobile-responsive components
- Background task integration

**Basic Usage:**

```python
import reflex as rx

class State(rx.State):
    progress: float = 0.0
    
    async def update_progress(self):
        for i in range(100):
            self.progress = i
            yield  # Real-time WebSocket update

def progress_bar():
    return rx.progress(value=State.progress)
```

### Crawl4AI - AI-Powered Scraping

**Purpose:** Primary web scraping with built-in AI extraction

**Key Features:**

- Built-in anti-bot detection
- AI content extraction
- Smart caching

**Basic Usage:**

```python
from crawl4ai import AsyncWebCrawler

async def scrape_with_ai(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            extraction_strategy="llm",  # AI extraction
            anti_bot=True,  # Built-in stealth
            bypass_cache=False  # Smart caching
        )
        return result.extracted_data
```

### RQ - Background Task Queue

**Purpose:** Simple background job processing

**Key Features:**

- Native Redis integration
- Built-in retry logic
- Job scheduling

**Basic Usage:**

```python
from rq import Queue, job
import redis

# Setup queue
redis_conn = redis.Redis()
queue = Queue('scraping', connection=redis_conn)

# Define job
@job('scraping', timeout='30m')
def scrape_company(url: str):
    return scrape_jobs(url)

# Enqueue job
job = queue.enqueue(scrape_company, "https://company.com/jobs")
```

### Tenacity - Retry Logic

**Purpose:** Robust error handling with exponential backoff

**Key Features:**

- Exponential backoff
- Condition-based retries
- Built-in logging

**Basic Usage:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(ConnectionError)
)
async def resilient_operation():
    # Operation that might fail
    pass
```

## Integration Patterns

### Model + Threshold Integration

```python
# Combine vLLM with 8000 token threshold
class HybridProcessor:
    def __init__(self):
        self.threshold = 8000
        self.model_manager = VLLMModelManager()
        
    async def process(self, content: str):
        tokens = count_tokens(content)
        
        if tokens < self.threshold:
            # 98% of jobs - local processing
            model = self.model_manager.get_model("primary")
            return await local_processing(content, model)
        else:
            # 2% of jobs - cloud fallback  
            return await cloud_processing(content)
```

### Scraping + AI Integration

```python
# Combine Crawl4AI with local AI
async def scrape_with_local_ai(url: str):
    async with AsyncWebCrawler() as crawler:
        # Scrape content
        result = await crawler.arun(url=url)
        
        # Extract with local AI
        job = await job_extractor.extract_job(
            content=result.extracted_content,
            source_url=url
        )
        return job
```

### UI + Background Tasks Integration

```python
# Real-time progress updates
class AppState(rx.State):
    progress: float = 0.0
    
    @rx.background
    async def start_scraping(self, companies: list[str]):
        # Enqueue background task
        task_id = scraping_worker.enqueue_company_scraping(companies)
        
        # Subscribe to progress updates
        async for update in progress_tracker.subscribe_to_progress(task_id):
            self.progress = update['progress_percentage']
            yield  # Real-time UI update
```

## Best Practices

### Library-First Principles

1. **Use Native Features:** Prefer library capabilities over custom implementations
2. **Minimal Configuration:** Use library defaults, configure only when necessary
3. **Error Handling:** Use library error handling (Tenacity) over custom patterns
4. **Performance:** Leverage library optimizations (vLLM swap_space, Crawl4AI caching)

### Configuration Management

```python
# Centralized library configuration
class LibraryConfig:
    # vLLM settings
    vllm_swap_space: int = 4
    vllm_gpu_memory: float = 0.85
    
    # Crawl4AI settings
    crawl4ai_anti_bot: bool = True
    crawl4ai_timeout: int = 30
    
    # RQ settings
    rq_timeout: str = "30m"
    rq_retry_attempts: int = 3
    
    # Tenacity settings
    retry_max_attempts: int = 3
    retry_min_wait: int = 1
    retry_max_wait: int = 10
```

### Memory Management

```python
# Single model constraint pattern
class ModelManager:
    def __init__(self):
        self.current_model = None
        
    def get_model(self, model_type: str):
        # vLLM handles memory automatically
        if not self.current_model or self.needs_switch(model_type):
            self.current_model = LLM(
                model=self.get_model_name(model_type),
                swap_space=4,  # Automatic management
                gpu_memory_utilization=0.85
            )
        return self.current_model
```

## Common Integration Issues

### vLLM Memory Issues

**Problem:** Multiple models loaded simultaneously
**Solution:** Use single model constraint with swap_space=4

```python
# Wrong - multiple models
model1 = LLM("Qwen/Qwen3-8B")  
model2 = LLM("Qwen/Qwen3-14B")  # Memory overflow!

# Right - single model with swapping
manager = SingleModelManager()
model = manager.get_model("primary")  # Automatic swapping
```

### Reflex WebSocket Issues

**Problem:** State updates not appearing in UI
**Solution:** Always use `yield` for real-time updates

```python
# Wrong - no real-time update
def update_data(self):
    self.data = new_data

# Right - real-time WebSocket update  
def update_data(self):
    self.data = new_data
    yield  # Triggers WebSocket update
```

### Crawl4AI Timeout Issues

**Problem:** Scraping timeouts on slow sites
**Solution:** Configure appropriate timeouts and retries

```python
# Configure for reliability
async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(
        url=url,
        page_timeout=30000,  # 30 second timeout
        wait_for="[data-testid='job-card']",  # Wait for content
        anti_bot=True  # Handle anti-bot measures
    )
```

### RQ Job Failures

**Problem:** Background jobs failing silently
**Solution:** Use proper error handling with Tenacity

```python
@job('scraping', timeout='30m')
@retry(stop=stop_after_attempt(3))
def robust_scraping_job(url: str):
    try:
        return scrape_url(url)
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise  # Let Tenacity handle retries
```

## Performance Optimization

### Model Performance

- Use appropriate model for content size (4B for small, 8B for medium, 14B for large)
- Enable quantization for memory efficiency
- Monitor token processing speed (target: 180+ tokens/sec)

### Scraping Performance

- Use Crawl4AI caching to avoid duplicate requests
- Configure appropriate timeouts for different site types
- Use concurrent scraping with proper rate limiting

### UI Performance

- Batch state updates when possible
- Use pagination for large job lists
- Optimize component rendering with React patterns

### Background Task Performance

- Use appropriate queue priorities for different job types
- Monitor queue length and worker performance
- Configure worker scaling based on load

## Troubleshooting Guide

### Library Version Conflicts

```bash
# Check library versions
uv show vllm reflex crawl4ai rq tenacity

# Update to latest compatible versions
uv add "vllm>=0.6.5" "reflex>=0.6.0" "crawl4ai>=0.4.0"
```

### GPU Access Issues

```bash
# Test GPU access
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
docker run --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
```

### Library Import Errors

```python
# Common import patterns
try:
    from vllm import LLM
except ImportError:
    print("vLLM not available - using mock")
    LLM = MockLLM

try:
    import reflex as rx
except ImportError:
    print("Reflex not available")
    sys.exit(1)
```

This guide provides quick reference for all major library integrations. For detailed implementation examples, refer to the individual specification files (specs/01-07).
