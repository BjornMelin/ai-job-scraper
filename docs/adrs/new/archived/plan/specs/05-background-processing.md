# Background Processing Implementation Specification

## Branch Name

`feat/background-rq-tenacity-integration`

## Overview

Implement RQ-based background task processing with native retry logic using Tenacity library. This specification replaces custom background task patterns with library-first approach, integrates with Reflex UI for real-time progress updates, and provides robust error handling for scraping operations.

## Context and Background

### Architectural Decision References

- **ADR-031:** Library-First Architecture - Use RQ native features over custom task orchestration
- **ADR-035:** Final Production Architecture - RQ + Tenacity for background processing
- **ADR-030 (Superseded):** Complex error handling patterns replaced by Tenacity
- **Integration:** Reflex UI real-time updates from spec 04

### Current State Analysis

The project currently has:

- **Basic background task setup:** Simple RQ configuration
- **Custom error handling:** Manual retry logic and exception management
- **Limited progress tracking:** No real-time updates during tasks
- **Complex orchestration:** Custom task coordination patterns

### Target State Goals

- **Library-first processing:** Use RQ native capabilities and Tenacity retry patterns
- **Real-time integration:** Direct integration with Reflex WebSocket updates
- **Robust error handling:** Automatic retry with exponential backoff using Tenacity
- **Progress monitoring:** Live progress updates sent to UI

## Implementation Requirements

### 1. RQ Worker Configuration

**Native RQ Features (No Custom Orchestration):**

```python
# Use RQ native job management
class ScrapingWorker:
    """RQ worker with library-first approach."""
    
    def __init__(self):
        self.redis_conn = redis.from_url(settings.redis_url)
        self.queue = Queue('scraping', connection=self.redis_conn)
        
    @job('scraping', timeout='30m')  # RQ native timeout
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def scrape_company_task(self, company_url: str, task_id: str):
        """Task with native RQ and Tenacity features."""
        return unified_scraper.scrape(company_url)
```

### 2. Real-Time Progress Updates

**WebSocket Integration with Reflex:**

```python
# Direct integration with Reflex state updates
class ProgressTracker:
    """Track and broadcast scraping progress."""
    
    def __init__(self):
        self.redis_conn = redis.from_url(settings.redis_url)
        
    async def update_progress(self, task_id: str, progress: dict):
        """Update progress and notify UI via Redis pub/sub."""
        
        # Store progress in Redis
        await self.redis_conn.hset(f"task:{task_id}", mapping=progress)
        
        # Publish update for real-time UI
        await self.redis_conn.publish(f"progress:{task_id}", json.dumps(progress))
```

### 3. Tenacity Error Handling

**Replace Custom Error Patterns:**

```python
# Use Tenacity for all retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True
)
async def resilient_scraping(url: str):
    """Scraping with library-first error handling."""
    return await unified_scraper.scrape(url)
```

## Files to Create/Modify

### Files to Create

1. **`src/tasks/workers.py`** - RQ worker implementation with Tenacity
2. **`src/tasks/progress.py`** - Real-time progress tracking
3. **`src/tasks/scheduler.py`** - Periodic task scheduling  
4. **`src/tasks/monitoring.py`** - Task monitoring and health checks
5. **`tests/test_background_processing.py`** - Background task testing

### Files to Modify

1. **`src/core/config.py`** - Add RQ and Redis configuration
2. **`src/ui/state.py`** - Integrate with background tasks
3. **`docker-compose.yml`** - Add RQ worker service

### Files to Remove/Archive

1. **Custom task orchestration** - Replace with RQ patterns
2. **Manual retry logic** - Replace with Tenacity
3. **Complex error classes** - Use library exceptions

## Dependencies and Libraries

### Updated Dependencies

```toml
# Add to pyproject.toml - background processing
[project.dependencies]
"redis>=5.0.0,<6.0.0"           # Task queue backend
"rq>=1.16.0,<2.0.0"             # Simple job queue
"tenacity>=9.0.0,<10.0.0"       # Retry logic library
"rq-scheduler>=0.13.1,<1.0.0"   # Periodic task scheduling
```

## Code Implementation

### 1. RQ Worker with Tenacity Integration

```python
# src/tasks/workers.py - Complete RQ worker implementation
from rq import Worker, Queue, job
from rq.decorators import job as rq_job
import redis
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)

from src.core.config import settings
from src.core.models import JobPosting, ScrapingStrategy
from src.scraping.unified import unified_scraper
from src.tasks.progress import progress_tracker

logger = logging.getLogger(__name__)

class ScrapingWorker:
    """RQ-based scraping worker with Tenacity error handling."""
    
    def __init__(self):
        self.redis_conn = redis.from_url(settings.redis_url)
        self.queue = Queue('scraping', connection=self.redis_conn)
        self.high_queue = Queue('high', connection=self.redis_conn)  # Priority queue
        
        # Worker configuration
        self.worker = Worker(
            [self.queue, self.high_queue],
            connection=self.redis_conn,
            name=f"scraper-worker-{settings.environment}"
        )
    
    def enqueue_company_scraping(
        self, 
        companies: List[str], 
        task_id: str,
        priority: str = "normal"
    ) -> str:
        """Enqueue multi-company scraping task."""
        
        queue = self.high_queue if priority == "high" else self.queue
        
        job = queue.enqueue(
            scrape_companies_task,
            companies,
            task_id,
            job_timeout='30m',  # RQ native timeout
            retry=3,  # RQ native retry count
            job_id=task_id,
            description=f"Scraping {len(companies)} companies"
        )
        
        logger.info(f"✅ Enqueued scraping task: {task_id} for {len(companies)} companies")
        return job.id
    
    def enqueue_single_company(
        self, 
        company_url: str, 
        task_id: str = None
    ) -> str:
        """Enqueue single company scraping."""
        
        task_id = task_id or f"single-{datetime.now().isoformat()}"
        
        job = self.queue.enqueue(
            scrape_single_company_task,
            company_url,
            task_id,
            job_timeout='10m',
            job_id=f"{task_id}-{company_url.replace('/', '-')}"
        )
        
        return job.id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and progress."""
        
        try:
            job = self.queue.connection.get_job(task_id)
            
            if not job:
                return {"status": "not_found", "message": "Task not found"}
            
            # Get progress from Redis
            progress_data = self.redis_conn.hgetall(f"task:{task_id}")
            progress = {k.decode(): v.decode() for k, v in progress_data.items()} if progress_data else {}
            
            return {
                "status": job.get_status(),
                "progress": progress,
                "result": job.result,
                "error": str(job.exc_info) if job.exc_info else None,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None
            }
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"status": "error", "message": str(e)}
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel running task."""
        
        try:
            job = self.queue.connection.get_job(task_id)
            if job:
                job.cancel()
                logger.info(f"✅ Cancelled task: {task_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
            return False
    
    def start_worker(self):
        """Start the RQ worker."""
        logger.info("🚀 Starting RQ scraping worker...")
        self.worker.work(with_scheduler=True)

# RQ job functions with Tenacity retry logic
@rq_job('scraping', timeout='30m')  
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def scrape_companies_task(companies: List[str], task_id: str) -> Dict[str, Any]:
    """Main multi-company scraping task with Tenacity error handling."""
    
    logger.info(f"🎯 Starting scraping task {task_id} for {len(companies)} companies")
    
    results = {
        "task_id": task_id,
        "companies_processed": 0,
        "jobs_found": 0,
        "jobs": [],
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    try:
        for i, company in enumerate(companies):
            # Update progress
            progress = {
                "task_id": task_id,
                "current_company": company,
                "companies_processed": i,
                "total_companies": len(companies),
                "progress_percentage": (i / len(companies)) * 100,
                "jobs_found": results["jobs_found"],
                "status": "processing"
            }
            
            # Send progress update
            asyncio.run(progress_tracker.update_progress(task_id, progress))
            
            # Scrape company with individual retry logic
            try:
                company_jobs = asyncio.run(
                    scrape_single_company_with_retry(company)
                )
                
                # Add jobs to results
                for job in company_jobs:
                    job_dict = {
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "salary_min": job.salary_min,
                        "salary_max": job.salary_max,
                        "description": job.description[:500] + "...",
                        "skills": job.skills,
                        "source_url": job.source_url,
                        "extraction_method": job.extraction_method.value,
                        "scraped_at": datetime.now().isoformat()
                    }
                    results["jobs"].append(job_dict)
                
                results["jobs_found"] += len(company_jobs)
                logger.info(f"✅ {company}: Found {len(company_jobs)} jobs")
                
            except Exception as e:
                error_msg = f"Failed to scrape {company}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            results["companies_processed"] += 1
        
        # Final progress update
        final_progress = {
            "task_id": task_id,
            "current_company": "Completed",
            "companies_processed": len(companies),
            "total_companies": len(companies),
            "progress_percentage": 100.0,
            "jobs_found": results["jobs_found"],
            "status": "completed"
        }
        
        asyncio.run(progress_tracker.update_progress(task_id, final_progress))
        
        results["end_time"] = datetime.now().isoformat()
        logger.info(f"🎉 Task {task_id} completed: {results['jobs_found']} jobs from {results['companies_processed']} companies")
        
        return results
        
    except Exception as e:
        # Final error state
        error_progress = {
            "task_id": task_id,
            "current_company": "Failed",
            "status": "failed",
            "error": str(e)
        }
        
        asyncio.run(progress_tracker.update_progress(task_id, error_progress))
        logger.error(f"❌ Task {task_id} failed: {str(e)}")
        raise

@rq_job('scraping', timeout='10m')
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def scrape_single_company_task(company_url: str, task_id: str) -> Dict[str, Any]:
    """Single company scraping task."""
    
    logger.info(f"🎯 Scraping single company: {company_url}")
    
    try:
        jobs = asyncio.run(unified_scraper.scrape(company_url))
        
        result = {
            "company_url": company_url,
            "jobs_found": len(jobs),
            "jobs": [job.dict() for job in jobs],
            "scraped_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Single company scrape completed: {len(jobs)} jobs")
        return result
        
    except Exception as e:
        logger.error(f"❌ Single company scrape failed: {str(e)}")
        raise

# Helper function with individual retry logic
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def scrape_single_company_with_retry(company_url: str) -> List[JobPosting]:
    """Scrape single company with individual retry logic."""
    return await unified_scraper.scrape(company_url)

# Global worker instance
scraping_worker = ScrapingWorker()
```

### 2. Real-Time Progress Tracking

```python
# src/tasks/progress.py - Real-time progress tracking
import redis.asyncio as aioredis
import json
import logging
from typing import Dict, Any, AsyncGenerator
from datetime import datetime

from src.core.config import settings

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Real-time progress tracking with Redis pub/sub."""
    
    def __init__(self):
        self.redis_url = settings.redis_url
        self._redis_conn = None
    
    async def _get_redis_connection(self):
        """Get or create async Redis connection."""
        if not self._redis_conn:
            self._redis_conn = aioredis.from_url(self.redis_url)
        return self._redis_conn
    
    async def update_progress(self, task_id: str, progress: Dict[str, Any]):
        """Update task progress and broadcast to subscribers."""
        
        try:
            redis_conn = await self._get_redis_connection()
            
            # Add timestamp
            progress["timestamp"] = datetime.now().isoformat()
            
            # Store progress in Redis hash
            await redis_conn.hset(
                f"task:{task_id}",
                mapping={k: str(v) for k, v in progress.items()}
            )
            
            # Publish progress update for real-time subscribers
            await redis_conn.publish(
                f"progress:{task_id}",
                json.dumps(progress)
            )
            
            logger.debug(f"📊 Progress update sent for task {task_id}: {progress.get('progress_percentage', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to update progress for task {task_id}: {e}")
    
    async def get_progress(self, task_id: str) -> Dict[str, Any]:
        """Get current task progress."""
        
        try:
            redis_conn = await self._get_redis_connection()
            
            progress_data = await redis_conn.hgetall(f"task:{task_id}")
            
            if not progress_data:
                return {"status": "not_found"}
            
            # Convert bytes to strings and parse JSON values
            progress = {}
            for key, value in progress_data.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                
                # Try to parse numeric values
                try:
                    if key_str in ['progress_percentage', 'companies_processed', 'total_companies', 'jobs_found']:
                        progress[key_str] = float(value_str)
                    else:
                        progress[key_str] = value_str
                except ValueError:
                    progress[key_str] = value_str
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to get progress for task {task_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def subscribe_to_progress(self, task_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time progress updates."""
        
        try:
            redis_conn = await self._get_redis_connection()
            pubsub = redis_conn.pubsub()
            
            # Subscribe to progress channel
            await pubsub.subscribe(f"progress:{task_id}")
            
            logger.info(f"📡 Subscribed to progress updates for task {task_id}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        progress = json.loads(message['data'])
                        yield progress
                        
                        # Stop if task is completed or failed
                        if progress.get('status') in ['completed', 'failed']:
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse progress message: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to subscribe to progress for task {task_id}: {e}")
        finally:
            if pubsub:
                await pubsub.unsubscribe(f"progress:{task_id}")
    
    async def cleanup_old_progress(self, hours: int = 24):
        """Clean up old progress data."""
        
        try:
            redis_conn = await self._get_redis_connection()
            
            # Find old task keys
            keys = await redis_conn.keys("task:*")
            
            cleaned_count = 0
            for key in keys:
                # Get task timestamp
                timestamp_str = await redis_conn.hget(key, "timestamp")
                
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.decode())
                        if (datetime.now() - timestamp).total_seconds() > hours * 3600:
                            await redis_conn.delete(key)
                            cleaned_count += 1
                    except ValueError:
                        continue
            
            logger.info(f"🧹 Cleaned up {cleaned_count} old progress entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old progress data: {e}")

# Global progress tracker
progress_tracker = ProgressTracker()
```

### 3. Task Scheduler for Periodic Jobs

```python
# src/tasks/scheduler.py - Periodic task scheduling
from rq_scheduler import Scheduler
from datetime import datetime, timedelta
import redis
import logging

from src.core.config import settings
from src.tasks.workers import scraping_worker

logger = logging.getLogger(__name__)

class TaskScheduler:
    """Periodic task scheduling with rq-scheduler."""
    
    def __init__(self):
        self.redis_conn = redis.from_url(settings.redis_url)
        self.scheduler = Scheduler(connection=self.redis_conn)
    
    def schedule_daily_company_refresh(self, companies: list[str]):
        """Schedule daily refresh of company job listings."""
        
        # Schedule for 2 AM daily
        schedule_time = datetime.now().replace(hour=2, minute=0, second=0) + timedelta(days=1)
        
        job = self.scheduler.schedule(
            scheduled_time=schedule_time,
            func=scraping_worker.enqueue_company_scraping,
            args=[companies, f"daily-refresh-{schedule_time.strftime('%Y%m%d')}"],
            repeat=timedelta(days=1),  # Repeat daily
            timeout='2h'
        )
        
        logger.info(f"📅 Scheduled daily company refresh: {job.id}")
        return job.id
    
    def schedule_weekly_full_scrape(self, companies: list[str]):
        """Schedule weekly comprehensive scraping."""
        
        # Schedule for Sunday 1 AM
        next_sunday = datetime.now() + timedelta(days=(6 - datetime.now().weekday()))
        schedule_time = next_sunday.replace(hour=1, minute=0, second=0)
        
        job = self.scheduler.schedule(
            scheduled_time=schedule_time,
            func=scraping_worker.enqueue_company_scraping,
            args=[companies, f"weekly-full-{schedule_time.strftime('%Y%m%d')}", "high"],
            repeat=timedelta(weeks=1),  # Repeat weekly
            timeout='6h'
        )
        
        logger.info(f"📅 Scheduled weekly full scrape: {job.id}")
        return job.id
    
    def cancel_scheduled_job(self, job_id: str) -> bool:
        """Cancel scheduled job."""
        
        try:
            self.scheduler.cancel(job_id)
            logger.info(f"✅ Cancelled scheduled job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_scheduled_jobs(self) -> list[dict]:
        """Get all scheduled jobs."""
        
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "func": str(job.func),
                    "scheduled_time": job.meta.get('scheduled_time'),
                    "repeat": job.meta.get('repeat'),
                    "status": job.get_status()
                })
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get scheduled jobs: {e}")
            return []

# Global scheduler
task_scheduler = TaskScheduler()
```

## Testing Requirements

### 1. Background Task Tests

```python
# tests/test_background_processing.py
import pytest
import asyncio
import redis
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.tasks.workers import scraping_worker, scrape_companies_task
from src.tasks.progress import progress_tracker
from src.tasks.scheduler import task_scheduler

class TestScrapingWorker:
    """Test RQ scraping worker."""
    
    def test_worker_initialization(self):
        """Test worker initializes correctly."""
        worker = scraping_worker
        
        assert worker.redis_conn is not None
        assert worker.queue is not None
        assert worker.worker is not None
    
    def test_enqueue_company_scraping(self):
        """Test enqueuing company scraping task."""
        
        companies = ["test1.com", "test2.com"]
        task_id = "test-task-123"
        
        job_id = scraping_worker.enqueue_company_scraping(companies, task_id)
        
        assert job_id is not None
        assert job_id == task_id
    
    def test_task_status_retrieval(self):
        """Test getting task status."""
        
        # This would require a running Redis instance for integration testing
        status = scraping_worker.get_task_status("non-existent-task")
        assert status["status"] == "not_found"

@pytest.mark.asyncio
class TestProgressTracking:
    """Test real-time progress tracking."""
    
    async def test_progress_update(self):
        """Test progress update and retrieval.""" 
        
        task_id = "test-progress-123"
        progress_data = {
            "current_company": "test.com",
            "progress_percentage": 50.0,
            "jobs_found": 10,
            "status": "processing"
        }
        
        # Update progress
        await progress_tracker.update_progress(task_id, progress_data)
        
        # Retrieve progress
        retrieved = await progress_tracker.get_progress(task_id)
        
        assert retrieved["current_company"] == "test.com"
        assert retrieved["progress_percentage"] == 50.0
        assert retrieved["jobs_found"] == 10
    
    async def test_progress_subscription(self):
        """Test real-time progress subscription."""
        
        # This would test the pub/sub functionality
        # Requires async testing with Redis
        pass

@pytest.mark.integration
class TestBackgroundTasks:
    """Integration tests for background tasks."""
    
    @patch('src.scraping.unified.unified_scraper.scrape')
    async def test_scraping_task_execution(self, mock_scraper):
        """Test complete scraping task execution."""
        
        # Mock scraper response
        mock_jobs = [
            MagicMock(
                title="Test Job",
                company="Test Company",
                location="Remote",
                salary_min=100000,
                salary_max=150000,
                description="Test description",
                skills=["Python"],
                source_url="https://test.com/job",
                extraction_method="crawl4ai"
            )
        ]
        mock_scraper.return_value = mock_jobs
        
        # Execute task
        companies = ["test.com"]
        task_id = "integration-test-123"
        
        result = scrape_companies_task(companies, task_id)
        
        assert result["task_id"] == task_id
        assert result["companies_processed"] == 1
        assert result["jobs_found"] == 1
        assert len(result["jobs"]) == 1
    
    def test_error_handling_with_tenacity(self):
        """Test Tenacity retry logic for failed tasks."""
        
        # This would test that failed tasks are retried according to Tenacity configuration
        pass

class TestTaskScheduler:
    """Test periodic task scheduling.""" 
    
    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        scheduler = task_scheduler
        
        assert scheduler.redis_conn is not None
        assert scheduler.scheduler is not None
    
    def test_daily_scheduling(self):
        """Test scheduling daily tasks."""
        
        companies = ["test.com"]
        job_id = task_scheduler.schedule_daily_company_refresh(companies)
        
        assert job_id is not None
        
        # Clean up
        task_scheduler.cancel_scheduled_job(job_id)

@pytest.mark.performance
class TestBackgroundPerformance:
    """Performance tests for background processing."""
    
    def test_task_queue_performance(self):
        """Test task queue can handle high volume."""
        
        # Test enqueuing many tasks quickly
        task_ids = []
        
        for i in range(100):
            task_id = f"perf-test-{i}"
            job_id = scraping_worker.enqueue_single_company(f"test{i}.com", task_id)
            task_ids.append(job_id)
        
        assert len(task_ids) == 100
        
        # Clean up would be needed in real test
    
    async def test_progress_update_performance(self):
        """Test progress updates handle high frequency."""
        
        task_id = "perf-progress-test"
        
        # Send 100 rapid progress updates
        for i in range(100):
            progress = {
                "progress_percentage": i,
                "current_company": f"company{i}.com",
                "jobs_found": i * 2
            }
            await progress_tracker.update_progress(task_id, progress)
        
        # Should handle rapid updates without errors
        final_progress = await progress_tracker.get_progress(task_id)
        assert final_progress["progress_percentage"] == 99.0
```

### 2. Error Handling Tests

```python
# tests/test_error_handling.py
import pytest
from unittest.mock import patch, MagicMock
from tenacity import RetryError

from src.tasks.workers import scrape_single_company_with_retry

class TestTenacityIntegration:
    """Test Tenacity error handling."""
    
    @pytest.mark.asyncio
    @patch('src.scraping.unified.unified_scraper.scrape')
    async def test_successful_retry(self, mock_scraper):
        """Test successful retry after initial failure."""
        
        # Mock: fail first call, succeed second
        mock_scraper.side_effect = [
            ConnectionError("Network error"),
            [MagicMock()]  # Success on retry
        ]
        
        result = await scrape_single_company_with_retry("test.com")
        
        assert len(result) == 1
        assert mock_scraper.call_count == 2
    
    @pytest.mark.asyncio
    @patch('src.scraping.unified.unified_scraper.scrape')
    async def test_exhausted_retries(self, mock_scraper):
        """Test behavior when all retries are exhausted."""
        
        # Mock: always fail
        mock_scraper.side_effect = ConnectionError("Persistent error")
        
        with pytest.raises(RetryError):
            await scrape_single_company_with_retry("test.com")
        
        # Should attempt maximum retries
        assert mock_scraper.call_count == 2  # Original + 1 retry
    
    @pytest.mark.asyncio
    @patch('src.scraping.unified.unified_scraper.scrape')
    async def test_non_retryable_error(self, mock_scraper):
        """Test that non-retryable errors are not retried."""
        
        # Mock: non-retryable error
        mock_scraper.side_effect = ValueError("Invalid input")
        
        with pytest.raises(ValueError):
            await scrape_single_company_with_retry("test.com")
        
        # Should not retry for ValueError
        assert mock_scraper.call_count == 1
```

## Configuration

### 1. Background Processing Configuration

```python
# src/core/config.py additions
class BackgroundConfig(BaseModel):
    """Background processing configuration."""
    
    # RQ Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    queue_name: str = Field(default="scraping", description="Primary queue name")
    high_priority_queue: str = Field(default="high", description="High priority queue")
    
    # Worker Configuration
    worker_timeout: int = Field(default=1800, description="Worker timeout in seconds (30 min)")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Tenacity Configuration
    retry_min_wait: int = Field(default=4, description="Minimum retry wait time (seconds)")
    retry_max_wait: int = Field(default=10, description="Maximum retry wait time (seconds)")
    retry_multiplier: float = Field(default=1.0, description="Exponential backoff multiplier")
    
    # Progress Tracking
    progress_cleanup_hours: int = Field(default=24, description="Hours to keep progress data")
    
    # Scheduling
    daily_refresh_hour: int = Field(default=2, description="Hour for daily refresh (24h format)")
    weekly_refresh_day: int = Field(default=6, description="Day for weekly refresh (0=Monday)")

# Add to main settings
background = BackgroundConfig()
```

### 2. Environment Configuration

```bash
# .env additions for background processing
# Redis Configuration
REDIS_URL="redis://localhost:6379"
QUEUE_NAME="scraping"
HIGH_PRIORITY_QUEUE="high"

# Worker Settings
WORKER_TIMEOUT=1800
MAX_RETRIES=3

# Tenacity Settings
RETRY_MIN_WAIT=4
RETRY_MAX_WAIT=10
RETRY_MULTIPLIER=1.0

# Progress Tracking
PROGRESS_CLEANUP_HOURS=24

# Scheduling
DAILY_REFRESH_HOUR=2
WEEKLY_REFRESH_DAY=6
```

### 3. Docker Configuration

```yaml
# docker-compose.yml additions
services:
  worker:
    build: .
    command: python -m src.tasks.workers
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///data/jobs.db
    depends_on:
      - redis
      - app
    volumes:
      - ./data:/data
      - ./models:/models
    
  scheduler:
    build: .
    command: rq-scheduler --host redis --port 6379
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
```

## Success Criteria

### Immediate Validation

- [ ] RQ worker starts and connects to Redis successfully
- [ ] Tasks can be enqueued and executed without errors
- [ ] Tenacity retry logic works for failed operations
- [ ] Real-time progress updates sent to Redis pub/sub
- [ ] Task scheduling works for periodic jobs

### Integration Validation  

- [ ] Background tasks integrate with Reflex UI state
- [ ] Progress updates appear in real-time in UI
- [ ] Scraping tasks use unified scraper from spec 03
- [ ] Local AI processing works in background tasks
- [ ] Error handling provides appropriate feedback

### Performance Validation

- [ ] Task queue handles 100+ concurrent jobs
- [ ] Progress updates sent within 100ms
- [ ] Task completion notifications delivered instantly
- [ ] Worker memory usage stays stable during long runs
- [ ] Retry logic doesn't create infinite loops

## Commit and PR Instructions

### Commit Messages

```bash
git checkout -b feat/background-rq-tenacity-integration

# RQ worker implementation
git add src/tasks/workers.py
git commit -m "feat: implement RQ worker with Tenacity error handling

- Native RQ job queue with timeout and retry configuration
- Tenacity retry logic replacing custom error handling patterns
- Multi-company scraping task with progress tracking
- Background task integration with unified scraper
- Implements ADR-031 library-first approach for task management"

# Progress tracking
git add src/tasks/progress.py
git commit -m "feat: implement real-time progress tracking with Redis pub/sub

- Redis pub/sub for real-time progress updates
- Async progress tracking with WebSocket integration  
- Task progress storage and retrieval
- Integration ready for Reflex UI real-time updates
- Automatic cleanup of old progress data"

# Task scheduling
git add src/tasks/scheduler.py
git commit -m "feat: implement periodic task scheduling with rq-scheduler

- Daily company refresh scheduling
- Weekly comprehensive scraping
- Flexible job scheduling and cancellation
- Integration with RQ worker infrastructure
- Configurable scheduling parameters"

# Docker and configuration
git add docker-compose.yml src/core/config.py
git commit -m "feat: add background processing configuration and Docker setup

- RQ worker and scheduler Docker services
- Comprehensive background processing configuration
- Environment variable management
- Production-ready worker deployment
- Redis integration for task queue backend"
```

### PR Description Template

```markdown
# Background Processing - RQ + Tenacity Integration

## Overview
Implements robust background processing system using RQ job queue with Tenacity retry logic, enabling real-time progress tracking and seamless integration with Reflex UI.

## Key Features Implemented

### Library-First Background Processing (ADR-031)
- ✅ **RQ native features:** Job timeout, retry, priority queues
- ✅ **Tenacity error handling:** Exponential backoff, retry conditions  
- ✅ **No custom orchestration:** Uses library capabilities throughout
- ✅ **Redis integration:** Native pub/sub for real-time updates

### Real-Time Progress Integration
- ✅ **WebSocket ready:** Direct integration with Reflex UI state
- ✅ **Live progress updates:** Real-time scraping progress via Redis pub/sub
- ✅ **Task monitoring:** Complete task lifecycle tracking
- ✅ **Error reporting:** User-friendly error messages and retry status

### Robust Error Handling
- ✅ **Tenacity retry logic:** Automatic retry with exponential backoff
- ✅ **Connection resilience:** Handles network failures gracefully  
- ✅ **Task recovery:** Failed tasks automatically retried
- ✅ **Error classification:** Different retry strategies for different errors

### Production Features
- ✅ **Periodic scheduling:** Daily/weekly automated scraping
- ✅ **Priority queues:** High-priority vs normal task processing
- ✅ **Worker scaling:** Multiple worker support for high volume
- ✅ **Progress cleanup:** Automatic cleanup of old task data

## Architecture Benefits

### Complexity Elimination
- ❌ **Custom task orchestration:** Replaced with RQ native features
- ❌ **Manual retry logic:** Replaced with Tenacity library
- ❌ **Complex error classes:** Uses library exception patterns
- ❌ **Custom progress tracking:** Uses Redis pub/sub patterns

### Integration Points
- **Reflex UI:** Real-time updates via WebSocket integration
- **Unified Scraper:** Background tasks use consolidated scraping
- **Local AI:** Background processing integrates with vLLM models
- **Configuration:** Unified settings management

## Performance Characteristics
- **Task throughput:** 100+ concurrent jobs supported
- **Progress latency:** <100ms update delivery
- **Error recovery:** Automatic retry with exponential backoff
- **Memory efficiency:** Stable worker memory usage

## Testing Coverage
- RQ worker functionality and job execution
- Tenacity retry logic and error handling
- Real-time progress tracking and pub/sub
- Task scheduling and periodic jobs
- Performance and load testing

## Next Steps
Ready for `06-integration-testing.md` - end-to-end system validation.
```

## Review Checklist

### Architecture Compliance

- [ ] ADR-031 library-first approach implemented throughout
- [ ] RQ native features used (no custom task orchestration)
- [ ] Tenacity library used for all retry logic
- [ ] Real-time integration ready for Reflex UI

### Code Quality

- [ ] Type hints on all functions and classes
- [ ] Async patterns for I/O operations  
- [ ] Error handling uses library patterns (Tenacity)
- [ ] Logging provides appropriate debugging information

### Performance

- [ ] Task queue handles high volume efficiently
- [ ] Progress updates delivered with low latency
- [ ] Worker memory usage remains stable
- [ ] Retry logic prevents infinite loops

### Integration Readiness

- [ ] Compatible with unified scraping from spec 03
- [ ] Ready for Reflex UI integration from spec 04
- [ ] Prepared for production deployment (spec 07)
- [ ] Supports local AI processing from spec 02

## Next Steps

After successful completion of this specification:

1. **Immediate:** Begin `06-integration-testing.md` for complete system validation
2. **Testing:** Validate real-time progress updates with live scraping
3. **Performance:** Benchmark task queue under load

This background processing implementation provides the robust, scalable foundation needed for the 1-week deployment target while eliminating custom complexity through library-first architecture.
