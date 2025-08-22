# ADR-027: Local Task Management

## Title

Local Task Management with Reflex Async Patterns

## Version/Date

2.0 / August 19, 2025

## Status

**SUPERSEDED BY ADR-023** - Replaced by RQ/Redis background processing (81.5% weighted score validation)

## Description

**SUPERSEDED**: This ADR originally described simple async task management patterns for local development. Based on comprehensive research and expert consensus (GPT-5 8/10, Gemini-2.5-Pro 9/10), these patterns have been replaced by **RQ/Redis background job processing** in ADR-023, which provides:

- **3-5x performance improvement** through parallel processing
- **Robust error handling** with automatic retries and exponential backoff
- **Job persistence** that survives application crashes
- **Real-time progress tracking** integrated with Reflex UI
- **Production-ready patterns** with minimal infrastructure overhead

## Context (SUPERSEDED)

### Research-Driven Architecture Evolution

**Why Simple Async Patterns Were Replaced:**

1. **Performance Limitations**: Sequential processing vs 3-5x parallel improvement
2. **Error Handling**: Basic try/catch vs robust retry mechanisms with exponential backoff
3. **Reliability**: Memory-only state vs persistent job storage surviving crashes
4. **Scalability**: Single-process limitations vs distributed worker capabilities
5. **Infrastructure Overhead**: Assumptions about complexity were invalidated by research

**Research Validation Process:**

- **Systematic Analysis**: context7, tavily-search, firecrawl, clear-thought methodology
- **Expert Consensus**: GPT-5 (8/10) and Gemini-2.5-Pro (9/10) unanimous recommendation
- **Weighted Decision Framework**: 81.5% score for RQ/Redis vs 74% for simple async
- **Implementation Complexity**: Docker makes Redis setup "trivial" (expert assessment)

### Framework Integration

- **Reflex State**: Native state management for task tracking
- **Python AsyncIO**: Built-in async patterns for non-blocking operations
- **Simple Threading**: Basic thread pool for CPU-bound tasks if needed
- **Local Logging**: File-based logging for development debugging

## Decision (SUPERSEDED)

**ORIGINAL DECISION**: Use Simple Async Task Management for local development  
**CURRENT DECISION**: **Migrate to RQ/Redis Background Processing (ADR-023)**

**Research-Validated Replacement Architecture:**

Instead of the simple patterns shown below, implement the comprehensive RQ/Redis system detailed in **ADR-023**, which provides:

```python
# NEW: RQ/Redis background processing (ADR-023)
from src.services.job_queue_service import job_queue_service
from rq import Queue, Retry

class ScrapingState(rx.State):
    @rx.background
    async def start_parallel_scraping(self, companies: List[str]):
        # Enqueue parallel company scraping with robust error handling
        job_ids = job_queue_service.enqueue_company_scraping(
            companies=companies,
            user_id="local_user",
            with_ai_enhancement=True  # Qwen3-4B integration
        )
        
        # Real-time progress monitoring
        await self._monitor_background_jobs(job_ids, companies)
```

### Migration Path from Simple Async Patterns

1. **Replace TaskManager** → **RQ Queue Service** (job_queue_service)
2. **Replace in-memory state** → **Redis job persistence**
3. **Replace simple retry** → **Exponential backoff with Retry(max=3, interval=[10, 30, 60])**
4. **Replace direct execution** → **Specialized worker queues** (io_fetch, parse, ai_enrich, persist)
5. **Enhance progress tracking** → **Real-time job metadata updates**

## Original Simple Async Implementation (DEPRECATED)

*The following patterns are preserved for reference but should not be used in new development.*

### Basic Task Management (DEPRECATED)

```python
# src/services/task_service.py
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Simple task representation for local development."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None

class TaskManager:
    """Simple task manager for local development."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._progress_callbacks: Dict[str, List[Callable]] = {}
    
    def create_task(self, name: str, coro: Coroutine) -> str:
        """Create a new task."""
        task = Task(name=name)
        self.tasks[task.id] = task
        self._progress_callbacks[task.id] = []
        
        # Create asyncio task
        async def wrapped_task():
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                self._notify_progress(task.id)
                
                # Execute the coroutine
                result = await coro
                
                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                task.message = "Task completed successfully"
                task.completed_at = datetime.now()
                task.result = result
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.message = "Task was cancelled"
                task.completed_at = datetime.now()
                logger.info(f"Task {task.name} was cancelled")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.message = f"Task failed: {str(e)}"
                task.completed_at = datetime.now()
                logger.error(f"Task {task.name} failed: {e}")
                
            finally:
                self._notify_progress(task.id)
                # Clean up running task reference
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
        
        # Start the task
        asyncio_task = asyncio.create_task(wrapped_task())
        self.running_tasks[task.id] = asyncio_task
        
        logger.info(f"Created task: {name} ({task.id})")
        return task.id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            return True
        return False
    
    def update_progress(self, task_id: str, progress: float, message: str = ""):
        """Update task progress."""
        if task_id in self.tasks:
            self.tasks[task_id].progress = progress
            if message:
                self.tasks[task_id].message = message
            self._notify_progress(task_id)
    
    def add_progress_callback(self, task_id: str, callback: Callable):
        """Add progress callback for task."""
        if task_id not in self._progress_callbacks:
            self._progress_callbacks[task_id] = []
        self._progress_callbacks[task_id].append(callback)
    
    def _notify_progress(self, task_id: str):
        """Notify progress callbacks."""
        if task_id in self._progress_callbacks:
            for callback in self._progress_callbacks[task_id]:
                try:
                    callback(self.tasks[task_id])
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")

# Global task manager instance
task_manager = TaskManager()
```

### Reflex Task State Integration

```python
# src/state/task_state.py
import reflex as rx
import asyncio
from src.services.task_service import task_manager, Task, TaskStatus
from src.services.scraper_service import scraper_service
from typing import List, Dict, Optional

class TaskState(rx.State):
    """Task management state for Reflex UI."""
    
    # Current tasks
    active_tasks: List[Dict] = []
    completed_tasks: List[Dict] = []
    
    # Current task being monitored
    current_task_id: str = ""
    current_task_progress: float = 0.0
    current_task_message: str = ""
    current_task_status: str = "idle"
    
    def _task_to_dict(self, task: Task) -> Dict:
        """Convert task to dictionary for Reflex state."""
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message,
            "created_at": task.created_at.strftime("%H:%M:%S"),
            "error": task.error
        }
    
    def refresh_tasks(self):
        """Refresh task lists."""
        all_tasks = task_manager.get_all_tasks()
        
        # Separate active and completed tasks
        active = []
        completed = []
        
        for task in all_tasks:
            task_dict = self._task_to_dict(task)
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                active.append(task_dict)
            else:
                completed.append(task_dict)
        
        self.active_tasks = active
        self.completed_tasks = completed[-10:]  # Keep last 10 completed
    
    async def start_scraping_task(self, sources: List[str] = None):
        """Start a scraping task."""
        if self.current_task_status == "running":
            return
        
        sources = sources or ["indeed", "linkedin"]
        
        # Create the scraping coroutine
        async def scraping_task():
            total_sources = len(sources)
            jobs_found = 0
            
            for i, source in enumerate(sources):
                # Update progress
                base_progress = (i / total_sources) * 100
                task_manager.update_progress(
                    self.current_task_id,
                    base_progress,
                    f"Scraping {source}..."
                )
                
                # Scrape the source
                source_jobs = 0
                async for job_data in scraper_service.scrape_source(source):
                    source_jobs += 1
                    jobs_found += 1
                    
                    # Update progress within source
                    source_progress = min(base_progress + (source_jobs * 10), (i + 1) / total_sources * 100)
                    task_manager.update_progress(
                        self.current_task_id,
                        source_progress,
                        f"Found {jobs_found} jobs from {source}"
                    )
                    
                    # Small delay for realistic progress
                    await asyncio.sleep(0.1)
            
            return {"jobs_found": jobs_found, "sources": sources}
        
        # Create and start the task
        task_id = task_manager.create_task(
            f"Scrape jobs from {', '.join(sources)}",
            scraping_task()
        )
        
        self.current_task_id = task_id
        self.current_task_status = "running"
        
        # Set up progress callback
        def on_progress(task: Task):
            self.current_task_progress = task.progress
            self.current_task_message = task.message
            self.current_task_status = task.status.value
            self.refresh_tasks()
        
        task_manager.add_progress_callback(task_id, on_progress)
        
        # Start monitoring task
        await self._monitor_current_task()
    
    async def _monitor_current_task(self):
        """Monitor current task until completion."""
        while self.current_task_id:
            task = task_manager.get_task(self.current_task_id)
            if not task:
                break
            
            self.current_task_progress = task.progress
            self.current_task_message = task.message
            self.current_task_status = task.status.value
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
            
            yield  # Update UI
            await asyncio.sleep(0.5)  # Check every 500ms
        
        self.refresh_tasks()
        yield
    
    def cancel_current_task(self):
        """Cancel the current task."""
        if self.current_task_id:
            task_manager.cancel_task(self.current_task_id)
            self.current_task_status = "cancelled"
    
    def clear_completed_tasks(self):
        """Clear completed tasks."""
        self.completed_tasks = []
    
    def on_load(self):
        """Load initial task state."""
        self.refresh_tasks()
```

### Simple Background Operations

```python
# src/services/background_service.py
import asyncio
import logging
from typing import Any, Callable, Coroutine
from src.services.task_service import task_manager

logger = logging.getLogger(__name__)

class BackgroundService:
    """Simple background service for local development."""
    
    @staticmethod
    async def run_with_progress(
        coro: Coroutine,
        task_name: str,
        progress_callback: Callable[[float, str], None] = None
    ) -> Any:
        """Run a coroutine with progress tracking."""
        
        async def tracked_coro():
            try:
                if progress_callback:
                    progress_callback(0.0, "Starting...")
                
                result = await coro
                
                if progress_callback:
                    progress_callback(100.0, "Completed")
                
                return result
                
            except Exception as e:
                if progress_callback:
                    progress_callback(0.0, f"Failed: {str(e)}")
                raise
        
        return await tracked_coro()
    
    @staticmethod
    async def run_batch_operation(
        items: list,
        operation: Callable,
        task_name: str = "Batch Operation",
        progress_callback: Callable[[float, str], None] = None
    ) -> list:
        """Run a batch operation with progress tracking."""
        results = []
        total_items = len(items)
        
        for i, item in enumerate(items):
            try:
                result = await operation(item) if asyncio.iscoroutinefunction(operation) else operation(item)
                results.append(result)
                
                # Update progress
                progress = ((i + 1) / total_items) * 100
                message = f"Processed {i + 1}/{total_items} items"
                
                if progress_callback:
                    progress_callback(progress, message)
                
                # Small delay to prevent overwhelming
                if i % 10 == 0:  # Every 10 items
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Batch operation failed on item {i}: {e}")
                results.append(None)
        
        return results

# Global background service instance
background_service = BackgroundService()
```

### Task UI Components

```python
# src/components/task_components.py
import reflex as rx
from src.state.task_state import TaskState

def task_progress_bar(progress: float, message: str) -> rx.Component:
    """Simple progress bar component."""
    return rx.vstack(
        rx.progress(value=progress, width="100%"),
        rx.text(f"{progress:.1f}% - {message}", font_size="sm"),
        spacing="2"
    )

def current_task_display() -> rx.Component:
    """Display current task status."""
    return rx.cond(
        TaskState.current_task_status == "running",
        rx.vstack(
            rx.heading("Current Task", size="md"),
            task_progress_bar(TaskState.current_task_progress, TaskState.current_task_message),
            rx.button(
                "Cancel Task",
                on_click=TaskState.cancel_current_task,
                bg="red.500",
                color="white"
            ),
            spacing="3",
            width="100%"
        ),
        rx.text("No active tasks", color="gray.500")
    )

def task_list() -> rx.Component:
    """Display list of tasks."""
    return rx.vstack(
        rx.heading("Tasks", size="lg"),
        
        # Current task
        current_task_display(),
        
        # Active tasks
        rx.cond(
            TaskState.active_tasks,
            rx.vstack(
                rx.heading("Active Tasks", size="md"),
                rx.foreach(
                    TaskState.active_tasks,
                    lambda task: rx.hstack(
                        rx.text(task["name"]),
                        rx.text(f"{task['progress']:.1f}%"),
                        rx.text(task["status"]),
                        justify="between",
                        width="100%"
                    )
                )
            )
        ),
        
        # Completed tasks
        rx.cond(
            TaskState.completed_tasks,
            rx.vstack(
                rx.hstack(
                    rx.heading("Recent Tasks", size="md"),
                    rx.button(
                        "Clear",
                        on_click=TaskState.clear_completed_tasks,
                        size="sm"
                    ),
                    justify="between"
                ),
                rx.foreach(
                    TaskState.completed_tasks,
                    lambda task: rx.hstack(
                        rx.text(task["name"]),
                        rx.text(task["status"]),
                        rx.text(task["created_at"]),
                        justify="between",
                        width="100%"
                    )
                )
            )
        ),
        
        spacing="4",
        width="100%"
    )
```

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive Outcomes

- **Simple Implementation**: Basic async patterns without complex infrastructure
- **Real-Time Updates**: Native Reflex state updates for task progress
- **Local Development**: Perfect for development and testing scenarios
- **Easy Debugging**: Clear logging and simple task tracking
- **No Dependencies**: Uses built-in Python async without external services

### Negative Consequences

- **Development Only**: Not suitable for production-scale task processing
- **Limited Persistence**: Tasks don't survive application restarts
- **Simple Error Handling**: Basic error handling without sophisticated retry logic
- **No Distribution**: Single-process task execution only

### Risk Mitigation

- **Clear Documentation**: Simple patterns easy to understand and modify
- **Upgrade Path**: Clear migration to production task queues when needed
- **Error Logging**: Comprehensive logging for debugging
- **Graceful Cancellation**: Proper task cancellation handling

## Development Guidelines

### Task Creation

- Use `task_manager.create_task()` for background operations
- Implement progress updates with `task_manager.update_progress()`
- Handle cancellation gracefully with try/except blocks
- Add appropriate logging for debugging

### Reflex Integration

- Use async state methods for task management
- Implement progress callbacks for real-time updates
- Use yield statements for UI updates during async operations
- Monitor task completion with polling loops

### Error Handling

- Wrap task operations in try/except blocks
- Log errors appropriately for debugging
- Provide user feedback through task status
- Implement graceful degradation for failed operations

## Related ADRs (UPDATED)

### Superseded By

- **ADR-023**: Background Job Processing with RQ/Redis (comprehensive replacement)

### Integrated Into  

- **ADR-017**: Local Development Architecture (updated with RQ/Redis patterns)
- **ADR-009**: LLM Selection and Integration Strategy (AI processing via RQ workers)

### Migration References

- **ADR-025**: Local Database Setup (job result persistence patterns)
- **ADR-020**: Reflex Local Development (enhanced real-time UI integration)

## Migration Success Criteria

### Completed When

- [ ] RQ/Redis infrastructure deployed and operational
- [ ] All simple async patterns replaced with RQ job processing
- [ ] Real-time progress tracking integrated with Reflex UI
- [ ] Parallel company scraping achieves 3-5x performance improvement  
- [ ] Robust error handling with automatic retries functional
- [ ] Job persistence survives application restarts
- [ ] RQ Dashboard accessible for monitoring and debugging

### Performance Validation

- [ ] Company scraping throughput: 3-5x improvement vs original async patterns
- [ ] Job failure rate: <5% with automatic retries
- [ ] UI responsiveness: Real-time updates <1s latency
- [ ] System stability: Zero data loss during application restarts
- [ ] Resource efficiency: Redis memory usage <256MB

---

**SUPERSEDED STATUS**: This ADR is replaced by ADR-023 (RQ/Redis Background Processing) based on comprehensive research validation and expert consensus. The original simple async patterns are deprecated in favor of production-ready background job processing with 3-5x performance improvement.

**Migration Priority**: HIGH - Replace simple async patterns immediately with RQ/Redis implementation from ADR-023.
