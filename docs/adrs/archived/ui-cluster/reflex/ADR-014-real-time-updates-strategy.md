# ADR-014: Real-time Updates Strategy

## Status

**Accepted** - *Scope: Production architecture*

## Context

The AI Job Scraper requires real-time capabilities for several critical features:

- Live scraping progress updates showing current source, jobs found, and completion percentage
- Real-time notifications for new jobs matching user criteria
- Live application status updates
- Concurrent user sessions with independent state
- Background task progress monitoring

Users expect immediate feedback during long-running operations like scraping, and the UI must remain responsive while processing occurs on the server.

## Decision Drivers

1. **User Experience**: Immediate feedback for long-running operations
2. **Performance**: Minimal latency for updates
3. **Scalability**: Handle multiple concurrent scraping sessions
4. **Reliability**: Graceful handling of connection issues
5. **Resource Efficiency**: Optimize server and network resources
6. **Developer Experience**: Simple patterns for implementing real-time features

## Considered Options

### Option 1: Polling

Client polls server at regular intervals for updates.

**Pros:**

- Simple to implement
- Works through firewalls/proxies
- Stateless on server

**Cons:**

- High latency (depends on polling interval)
- Wastes resources on empty polls
- Not truly real-time
- Can overload server with many clients

### Option 2: Server-Sent Events (SSE)

One-way server-to-client communication.

**Pros:**

- Simple protocol
- Automatic reconnection
- Works over HTTP

**Cons:**

- One-way only (no client-to-server)
- Limited browser support for some features
- Connection limits in browsers
- Not suitable for bidirectional communication

### Option 3: WebSockets (Reflex Built-in)

Full-duplex communication channel.

**Pros:**

- True real-time bidirectional communication
- Low latency
- Built into Reflex framework
- Efficient for frequent updates
- Supports binary data

**Cons:**

- More complex than HTTP
- Requires connection management
- Potential proxy/firewall issues
- Stateful connections

### Option 4: GraphQL Subscriptions

Real-time data with GraphQL.

**Pros:**

- Strongly typed
- Flexible data fetching
- Good tooling

**Cons:**

- Additional complexity
- Requires GraphQL server setup
- Overkill for our use case
- Not native to Reflex

## Decision

**We will use Reflex's built-in WebSocket implementation for real-time updates.**

## Detailed Design

### WebSocket Architecture

```python
# Real-time state management
class RealtimeState(rx.State):
    """Base class for real-time features"""
    
    # Connection management
    ws_connected: bool = False
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5
    
    @rx.event
    def on_ws_connect(self):
        """Handle WebSocket connection"""
        self.ws_connected = True
        self.reconnect_attempts = 0
        return rx.toast.success("Connected to server")
    
    @rx.event
    def on_ws_disconnect(self):
        """Handle WebSocket disconnection"""
        self.ws_connected = False
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            # Exponential backoff
            delay = min(1000 * (2 ** self.reconnect_attempts), 30000)
            return rx.call_script(
                f"setTimeout(() => window.location.reload(), {delay})"
            )
```

### Scraping Progress Updates

```python
class ScrapingRealtimeState(RealtimeState):
    """Real-time scraping updates"""
    
    # Progress tracking
    scraping_active: bool = False
    total_sources: int = 0
    completed_sources: int = 0
    current_source: str = ""
    jobs_found: int = 0
    errors: list[str] = []
    log_buffer: list[LogEntry] = []
    
    def start_scraping(self, sources: list[str]):
        """Stream scraping progress in real-time"""
        self.scraping_active = True
        self.total_sources = len(sources)
        self.completed_sources = 0
        self.jobs_found = 0
        self.log_buffer = []
        yield
        
        for source in sources:
            # Update current source
            self.current_source = source
            self.emit_log(f"Starting {source}...", "info")
            yield
            
            try:
                # Perform scraping (sync or async handled by framework)
                jobs = scrape_source(source)  # Reflex handles async internally
                
                # Update progress
                self.jobs_found += len(jobs)
                self.completed_sources += 1
                self.emit_log(
                    f"Found {len(jobs)} jobs from {source}", 
                    "success"
                )
                yield
                
            except Exception as e:
                self.errors.append(str(e))
                self.emit_log(f"Error in {source}: {e}", "error")
                yield
        
        self.scraping_active = False
        self.emit_notification("Scraping complete!", "success")
    
    @rx.event
    def emit_log(self, message: str, level: str):
        """Add log entry with timestamp"""
        entry = LogEntry(
            timestamp=datetime.now(),
            message=message,
            level=level
        )
        self.log_buffer.append(entry)
        
        # Keep buffer size limited
        if len(self.log_buffer) > 100:
            self.log_buffer = self.log_buffer[-100:]
    
    @rx.var
    def progress_percentage(self) -> float:
        """Computed progress percentage"""
        if self.total_sources == 0:
            return 0
        return (self.completed_sources / self.total_sources) * 100
```

### Live Notifications

```python
class NotificationState(RealtimeState):
    """Real-time notification system"""
    
    notifications: list[Notification] = []
    unread_count: int = 0
    
    @rx.event(background=True)
    async def monitor_new_jobs(self):
        """Monitor for new jobs matching criteria"""
        while self.monitoring_active:
            # Check for new jobs
            new_jobs = await check_new_jobs_matching_filters()
            
            if new_jobs:
                async with self:
                    for job in new_jobs:
                        self.emit_notification(
                            f"New job: {job.title} at {job.company}",
                            "info",
                            action_url=f"/jobs/{job.id}"
                        )
            
            # Check every 30 seconds
            await asyncio.sleep(30)
    
    @rx.event
    def emit_notification(
        self, 
        message: str, 
        type: str = "info",
        action_url: str = None
    ):
        """Emit real-time notification"""
        notification = Notification(
            id=str(uuid4()),
            message=message,
            type=type,
            timestamp=datetime.now(),
            action_url=action_url,
            read=False
        )
        
        self.notifications.insert(0, notification)
        self.unread_count += 1
        
        # Show toast for important notifications
        if type in ["error", "warning"]:
            return rx.toast[type](message)
        
        # Limit notification history
        if len(self.notifications) > 50:
            self.notifications = self.notifications[:50]
```

### Optimistic Updates

```python
class OptimisticState(RealtimeState):
    """Optimistic update patterns"""
    
    @rx.event
    async def update_application_status(
        self, 
        job_id: str, 
        new_status: str
    ):
        """Optimistic update with rollback on failure"""
        # Store original state
        original_status = self.get_job_status(job_id)
        
        # Optimistic update
        self.set_job_status(job_id, new_status)
        
        try:
            # Persist to database
            await update_application_in_db(job_id, new_status)
            
            # Confirm update
            return rx.toast.success("Status updated successfully")
            
        except Exception as e:
            # Rollback on failure
            self.set_job_status(job_id, original_status)
            return rx.toast.error(f"Failed to update: {e}")
```

## Implementation Patterns

### 1. Connection Management

- Automatic reconnection with exponential backoff
- Connection status indicator in UI
- Graceful degradation when disconnected
- Queue updates during disconnection

### 2. Update Throttling

- Debounce rapid updates (100-300ms)
- Batch multiple updates
- Priority queue for critical updates
- Rate limiting per client

### 3. Progress Streaming

- Stream updates at regular intervals
- Buffer logs to prevent overflow
- Smooth progress bar animations
- Cancel/pause capabilities

### 4. Error Handling

- Retry failed WebSocket connections
- Fallback to polling if WebSockets fail
- Clear error messages to users
- Log errors for debugging

## Rationale

Reflex's built-in WebSocket support provides:

1. **Native Integration**: No additional libraries needed
2. **Automatic State Sync**: Framework handles state synchronization
3. **Per-Client Isolation**: Each user has independent state
4. **Background Tasks**: Async operations with UI updates
5. **Simple API**: Pythonic event handlers and state management

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive

- True real-time updates with minimal latency
- Responsive UI during long operations
- Efficient resource usage
- Built-in connection management
- Simple implementation patterns

### Negative

- Stateful connections require memory
- Potential scaling challenges with many concurrent users
- WebSocket proxy/firewall configuration needed
- Connection state management complexity

### Neutral

- Need monitoring for WebSocket health
- Client-side connection indicators required
- Testing real-time features more complex

## Performance Considerations

### Update Frequency

- Scraping progress: Every 100-500ms
- Notifications: Immediate
- Dashboard stats: Every 5 seconds
- Background sync: Every 30 seconds

### Resource Limits

- Max WebSocket connections: Configure in production
- Message size limits: 64KB default
- Buffer sizes: 100 messages for logs
- Timeout settings: 60 seconds for idle connections

## Testing Strategy

- Unit test state update logic
- Integration test WebSocket connections
- Load test concurrent connections
- Test reconnection scenarios
- Verify update throttling

## Related ADRs

- **ADR-012**: Reflex UI Framework Decision - Framework foundation for WebSocket support
- **ADR-013**: State Management Architecture - State patterns for real-time updates  
- **ADR-015**: Component Library Selection - UI components for progress display
- **ADR-016**: Routing and Navigation Design - State persistence during navigation
- **ADR-020**: Reflex Local Development - Development real-time patterns

## References

- [Reflex WebSocket Architecture](https://reflex.dev/blog/2024-03-21-reflex-architecture/)
- [Background Tasks Documentation](https://reflex.dev/docs/state/background-tasks/)
- [Real-time Streaming Example](https://github.com/reflex-dev/reflex-examples/tree/main/streaming)
- [WebSocket Best Practices](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
