# Reflex UI Architecture for AI Job Scraper

## Executive Summary

This document defines the comprehensive UI architecture for the AI Job Scraper application using the Reflex framework (v0.6+). The architecture leverages Reflex's pure Python approach to deliver a responsive, real-time web application with minimal maintenance overhead and maximum developer productivity.

## 1. Information Architecture

### 1.1 Core Entities

```yaml
entities:
  Job:
    - id: str
    - title: str
    - company: Company
    - location: str
    - salary_range: Optional[SalaryRange]
    - description: str
    - requirements: list[str]
    - posted_date: datetime
    - scraped_date: datetime
    - source: JobSource
    - application_status: ApplicationStatus
    - tags: list[Tag]
    
  Company:
    - id: str
    - name: str
    - website: Optional[str]
    - industry: Optional[str]
    - size: Optional[str]
    - rating: Optional[float]
    - jobs_count: int
    
  Application:
    - id: str
    - job_id: str
    - applied_date: datetime
    - status: ApplicationStatus
    - notes: Optional[str]
    - resume_version: Optional[str]
    
  ScrapingSession:
    - id: str
    - started_at: datetime
    - completed_at: Optional[datetime]
    - status: ScrapingStatus
    - jobs_found: int
    - errors: list[str]
```

### 1.2 Navigation Model

```python
navigation_structure = {
    "primary": [
        {"route": "/", "name": "Dashboard", "icon": "home"},
        {"route": "/jobs", "name": "Jobs", "icon": "briefcase"},
        {"route": "/companies", "name": "Companies", "icon": "building"},
        {"route": "/applications", "name": "Applications", "icon": "file-text"},
        {"route": "/scraping", "name": "Scraping", "icon": "download-cloud"},
    ],
    "secondary": [
        {"route": "/analytics", "name": "Analytics", "icon": "bar-chart"},
        {"route": "/settings", "name": "Settings", "icon": "settings"},
    ]
}
```

## 2. Layout System

### 2.1 Grid System

```python
# 8px base unit following Material Design principles
SPACING_SCALE = {
    "xs": "0.5rem",   # 4px
    "sm": "0.75rem",  # 6px
    "md": "1rem",     # 8px
    "lg": "1.5rem",   # 12px
    "xl": "2rem",     # 16px
    "2xl": "3rem",    # 24px
    "3xl": "4rem",    # 32px
}

BREAKPOINTS = {
    "mobile": "640px",
    "tablet": "768px",
    "desktop": "1024px",
    "wide": "1280px",
}
```

### 2.2 Responsive Layout Rules

```python
def responsive_layout():
    return rx.box(
        rx.cond(
            rx.breakpoint("mobile"),
            mobile_layout(),
            rx.cond(
                rx.breakpoint("tablet"),
                tablet_layout(),
                desktop_layout()
            )
        )
    )
```

## 3. Design Tokens

### 3.1 Typography

```json
{
  "typography": {
    "font_family": {
      "sans": "Inter, system-ui, -apple-system, sans-serif",
      "mono": "JetBrains Mono, monospace"
    },
    "font_size": {
      "xs": "0.75rem",
      "sm": "0.875rem",
      "base": "1rem",
      "lg": "1.125rem",
      "xl": "1.25rem",
      "2xl": "1.5rem",
      "3xl": "1.875rem",
      "4xl": "2.25rem"
    },
    "font_weight": {
      "normal": 400,
      "medium": 500,
      "semibold": 600,
      "bold": 700
    },
    "line_height": {
      "tight": 1.25,
      "normal": 1.5,
      "relaxed": 1.75
    }
  }
}
```

### 3.2 Color System

```json
{
  "colors": {
    "brand": {
      "primary": "#0066FF",
      "secondary": "#7C3AED",
      "accent": "#10B981"
    },
    "semantic": {
      "success": "#10B981",
      "warning": "#F59E0B",
      "error": "#EF4444",
      "info": "#3B82F6"
    },
    "neutral": {
      "50": "#FAFAFA",
      "100": "#F4F4F5",
      "200": "#E4E4E7",
      "300": "#D4D4D8",
      "400": "#A1A1AA",
      "500": "#71717A",
      "600": "#52525B",
      "700": "#3F3F46",
      "800": "#27272A",
      "900": "#18181B"
    }
  }
}
```

## 4. Component Architecture

### 4.1 Component Hierarchy

```python
# Base layout component
class AppLayout(rx.ComponentState):
    sidebar_open: bool = True
    
    @classmethod
    def get_component(cls, **props):
        return rx.hstack(
            sidebar(is_open=cls.sidebar_open),
            rx.box(
                navbar(),
                rx.container(
                    props.get("children", rx.fragment()),
                    max_width="1280px",
                    padding=SPACING_SCALE["lg"]
                ),
                flex=1
            ),
            width="100%",
            spacing="0"
        )

# Reusable job card component
class JobCard(rx.ComponentState):
    expanded: bool = False
    
    @rx.event
    def toggle_expand(self):
        self.expanded = not self.expanded
    
    @classmethod
    def get_component(cls, job: Job, **props):
        return rx.card(
            rx.vstack(
                rx.hstack(
                    rx.heading(job.title, size="lg"),
                    rx.badge(job.application_status),
                    justify="between"
                ),
                rx.text(job.company.name, color="gray"),
                rx.cond(
                    cls.expanded,
                    rx.vstack(
                        rx.text(job.description),
                        rx.hstack(*[rx.badge(tag) for tag in job.tags]),
                        spacing="md"
                    )
                ),
                rx.button(
                    rx.cond(cls.expanded, "Show Less", "Show More"),
                    on_click=cls.toggle_expand,
                    variant="ghost"
                ),
                spacing="md",
                width="100%"
            ),
            **props
        )
```

### 4.2 Core Components Selection

| Component | Reflex Component | Purpose | Props/State |
|-----------|-----------------|---------|-------------|
| Navigation | `rx.drawer` + `rx.tabs` | App navigation | `current_route`, `sidebar_open` |
| Job List | `rx.data_list` | Display jobs | `jobs`, `filters`, `sort_by` |
| Job Card | `rx.card` | Job display | `job`, `expanded`, `selected` |
| Filter Panel | `rx.accordion` | Filter controls | `active_filters` |
| Search Bar | `rx.input` + `rx.select` | Search/filter | `search_query`, `search_type` |
| Data Table | `rx.table` | Tabular data | `data`, `columns`, `pagination` |
| Progress | `rx.progress` | Scraping status | `value`, `status` |
| Modal | `rx.dialog` | Forms/details | `open`, `content` |
| Toast | `rx.toast` | Notifications | `message`, `type` |

## 5. State Management

### 5.1 Global State Architecture

```python
class AppState(rx.State):
    """Root application state"""
    
    # Authentication & Session
    user_id: str = ""
    session_token: str = ""
    
    # Navigation
    current_route: str = "/"
    sidebar_open: bool = True
    
    # Theme
    dark_mode: bool = False
    
    # Global notifications
    notifications: list[Notification] = []
    
    @rx.event
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        
    @rx.event
    def add_notification(self, message: str, type: str = "info"):
        notification = Notification(
            id=str(uuid4()),
            message=message,
            type=type,
            timestamp=datetime.now()
        )
        self.notifications.append(notification)
        
    @rx.var
    def theme_config(self) -> dict:
        return {
            "colorMode": "dark" if self.dark_mode else "light",
            "primaryColor": self.primary_color
        }
```

### 5.2 Domain-Specific State

```python
class JobState(AppState):
    """Job management state"""
    
    # Data
    jobs: list[Job] = []
    selected_job: Optional[Job] = None
    
    # Filters
    search_query: str = ""
    filter_company: str = ""
    filter_status: str = "all"
    filter_tags: list[str] = []
    
    # UI State
    loading: bool = False
    view_mode: str = "cards"  # cards | table
    sort_by: str = "posted_date"
    sort_order: str = "desc"
    
    # Pagination
    page: int = 1
    per_page: int = 20
    
    @rx.event(background=True)
    async def load_jobs(self):
        """Load jobs from database"""
        async with self:
            self.loading = True
            
        async with rx.session() as session:
            query = select(Job)
            
            # Apply filters
            if self.search_query:
                query = query.where(
                    or_(
                        Job.title.ilike(f"%{self.search_query}%"),
                        Job.description.ilike(f"%{self.search_query}%")
                    )
                )
            
            if self.filter_company:
                query = query.join(Company).where(
                    Company.name == self.filter_company
                )
            
            # Apply sorting
            sort_column = getattr(Job, self.sort_by)
            query = query.order_by(
                desc(sort_column) if self.sort_order == "desc" else asc(sort_column)
            )
            
            # Apply pagination
            query = query.offset((self.page - 1) * self.per_page).limit(self.per_page)
            
            jobs = session.exec(query).all()
            
        async with self:
            self.jobs = jobs
            self.loading = False
    
    @rx.var
    def filtered_jobs(self) -> list[Job]:
        """Computed var for filtered jobs"""
        filtered = self.jobs
        
        if self.filter_status != "all":
            filtered = [j for j in filtered if j.application_status == self.filter_status]
        
        if self.filter_tags:
            filtered = [j for j in filtered if any(tag in j.tags for tag in self.filter_tags)]
        
        return filtered
    
    @rx.var
    def total_pages(self) -> int:
        return math.ceil(len(self.filtered_jobs) / self.per_page)
```

### 5.3 Real-time Scraping State

```python
class ScrapingState(AppState):
    """Real-time scraping state management"""
    
    # Session
    current_session: Optional[ScrapingSession] = None
    is_scraping: bool = False
    
    # Progress
    total_sources: int = 0
    completed_sources: int = 0
    current_source: str = ""
    jobs_found: int = 0
    
    # Stream
    log_messages: list[LogMessage] = []
    
    @rx.event(background=True)
    async def start_scraping(self, sources: list[str], filters: dict):
        """Start scraping with real-time updates"""
        async with self:
            self.is_scraping = True
            self.total_sources = len(sources)
            self.completed_sources = 0
            self.jobs_found = 0
            self.log_messages = []
            
            # Create scraping session
            self.current_session = ScrapingSession(
                id=str(uuid4()),
                started_at=datetime.now(),
                status="running"
            )
        
        for source in sources:
            async with self:
                self.current_source = source
                self.add_log(f"Starting scrape of {source}...")
            
            # Simulate scraping (replace with actual scraper)
            jobs = await scrape_source(source, filters)
            
            async with self:
                self.jobs_found += len(jobs)
                self.completed_sources += 1
                self.add_log(f"Found {len(jobs)} jobs from {source}")
            
            # Stream updates every 100ms for smooth UI
            await asyncio.sleep(0.1)
        
        async with self:
            self.is_scraping = False
            self.current_session.completed_at = datetime.now()
            self.current_session.status = "completed"
            self.current_session.jobs_found = self.jobs_found
    
    @rx.event
    def add_log(self, message: str):
        log = LogMessage(
            timestamp=datetime.now(),
            message=message,
            level="info"
        )
        self.log_messages.append(log)
    
    @rx.var
    def progress_percentage(self) -> float:
        if self.total_sources == 0:
            return 0
        return (self.completed_sources / self.total_sources) * 100
```

## 6. Pages & Routing

### 6.1 Route Configuration

```python
# pages/__init__.py
from .dashboard import dashboard
from .jobs import jobs_page
from .companies import companies_page
from .applications import applications_page
from .scraping import scraping_page
from .settings import settings_page

# Route registration
app = rx.App()
app.add_page(dashboard, route="/", title="Dashboard")
app.add_page(jobs_page, route="/jobs", title="Browse Jobs")
app.add_page(lambda job_id: job_detail(job_id), route="/jobs/[job_id]", title="Job Details")
app.add_page(companies_page, route="/companies", title="Companies")
app.add_page(applications_page, route="/applications", title="My Applications")
app.add_page(scraping_page, route="/scraping", title="Scraping Center")
app.add_page(settings_page, route="/settings", title="Settings")
```

### 6.2 Dynamic Routing

```python
@rx.page(route="/jobs/[job_id]")
def job_detail():
    """Dynamic job detail page"""
    return rx.vstack(
        rx.heading(JobDetailState.job.title),
        rx.text(f"Job ID: {JobDetailState.job_id}"),
        job_detail_content(),
        on_mount=JobDetailState.load_job
    )

class JobDetailState(AppState):
    job_id: str = ""
    job: Optional[Job] = None
    
    @rx.event
    def load_job(self):
        # Access route params
        self.job_id = self.router.page.params.get("job_id", "")
        
        with rx.session() as session:
            self.job = session.exec(
                select(Job).where(Job.id == self.job_id)
            ).first()
```

## 7. Forms & Validation

### 7.1 Form Architecture

```python
class JobApplicationForm(rx.ComponentState):
    """Reusable job application form component"""
    
    # Form data
    cover_letter: str = ""
    resume_file: str = ""
    expected_salary: str = ""
    
    # Validation state
    errors: dict[str, str] = {}
    submitting: bool = False
    
    @rx.event
    def validate_field(self, field: str, value: str):
        """Real-time field validation"""
        if field == "cover_letter" and len(value) < 100:
            self.errors[field] = "Cover letter must be at least 100 characters"
        elif field == "expected_salary":
            try:
                salary = int(value)
                if salary < 0:
                    self.errors[field] = "Salary must be positive"
                else:
                    self.errors.pop(field, None)
            except ValueError:
                self.errors[field] = "Please enter a valid number"
        else:
            self.errors.pop(field, None)
    
    @rx.event
    async def submit_application(self, form_data: dict):
        """Handle form submission"""
        self.submitting = True
        
        # Validate all fields
        if not self.validate_all(form_data):
            self.submitting = False
            return rx.toast.error("Please fix validation errors")
        
        # Save to database
        with rx.session() as session:
            application = Application(**form_data)
            session.add(application)
            session.commit()
        
        self.submitting = False
        return rx.toast.success("Application submitted successfully!")
    
    @classmethod
    def get_component(cls, job_id: str, **props):
        return rx.form(
            rx.vstack(
                rx.form.field(
                    rx.form.label("Cover Letter"),
                    rx.text_area(
                        name="cover_letter",
                        value=cls.cover_letter,
                        on_change=cls.set_cover_letter,
                        on_blur=lambda: cls.validate_field("cover_letter", cls.cover_letter),
                        placeholder="Write your cover letter...",
                        rows=10
                    ),
                    rx.cond(
                        cls.errors.get("cover_letter"),
                        rx.form.message(
                            cls.errors["cover_letter"],
                            color="red"
                        )
                    )
                ),
                rx.form.field(
                    rx.form.label("Expected Salary"),
                    rx.input(
                        name="expected_salary",
                        value=cls.expected_salary,
                        on_change=cls.set_expected_salary,
                        on_blur=lambda: cls.validate_field("expected_salary", cls.expected_salary),
                        type="number"
                    )
                ),
                rx.form.submit(
                    rx.button(
                        "Submit Application",
                        loading=cls.submitting,
                        disabled=cls.submitting or bool(cls.errors),
                        width="100%"
                    )
                ),
                spacing="lg"
            ),
            on_submit=cls.submit_application,
            reset_on_submit=True
        )
```

## 8. Database Integration

### 8.1 Model Definitions

```python
# models.py
class Job(rx.Model, table=True):
    """Job model with SQLModel integration"""
    
    id: str = Field(primary_key=True, default_factory=lambda: str(uuid4()))
    title: str = Field(index=True)
    company_id: str = Field(foreign_key="company.id")
    company: Optional["Company"] = Relationship(back_populates="jobs")
    
    location: str
    remote_type: str = Field(default="onsite")  # onsite | remote | hybrid
    
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: str = Field(default="USD")
    
    description: str = Field(sa_column=Column(Text))
    requirements: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    posted_date: datetime = Field(index=True)
    scraped_date: datetime = Field(default_factory=datetime.now)
    expires_date: Optional[datetime] = None
    
    source_url: str
    source_name: str
    
    # Application tracking
    applications: list["Application"] = Relationship(back_populates="job")
    
    # Metadata
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class Company(rx.Model, table=True):
    """Company model"""
    
    id: str = Field(primary_key=True, default_factory=lambda: str(uuid4()))
    name: str = Field(unique=True, index=True)
    website: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    
    # Ratings
    glassdoor_rating: Optional[float] = None
    indeed_rating: Optional[float] = None
    
    # Relationships
    jobs: list["Job"] = Relationship(back_populates="company")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
```

### 8.2 Query Patterns

```python
class DatabaseService:
    """Database service patterns"""
    
    @staticmethod
    def search_jobs(query: str, filters: dict) -> list[Job]:
        """Advanced job search with filters"""
        with rx.session() as session:
            stmt = select(Job).join(Company)
            
            # Text search
            if query:
                search_filter = or_(
                    Job.title.ilike(f"%{query}%"),
                    Job.description.ilike(f"%{query}%"),
                    Company.name.ilike(f"%{query}%")
                )
                stmt = stmt.where(search_filter)
            
            # Location filter
            if filters.get("location"):
                stmt = stmt.where(Job.location.ilike(f"%{filters['location']}%"))
            
            # Salary range filter
            if filters.get("min_salary"):
                stmt = stmt.where(Job.salary_max >= filters["min_salary"])
            
            # Remote type filter
            if filters.get("remote_type"):
                stmt = stmt.where(Job.remote_type == filters["remote_type"])
            
            # Date range filter
            if filters.get("posted_after"):
                stmt = stmt.where(Job.posted_date >= filters["posted_after"])
            
            # Tags filter
            if filters.get("tags"):
                # Using PostgreSQL array operations
                stmt = stmt.where(
                    Job.tags.op("@>")(filters["tags"])
                )
            
            # Sorting
            sort_by = filters.get("sort_by", "posted_date")
            sort_order = filters.get("sort_order", "desc")
            
            sort_column = getattr(Job, sort_by)
            if sort_order == "desc":
                stmt = stmt.order_by(desc(sort_column))
            else:
                stmt = stmt.order_by(asc(sort_column))
            
            # Pagination
            page = filters.get("page", 1)
            per_page = filters.get("per_page", 20)
            stmt = stmt.offset((page - 1) * per_page).limit(per_page)
            
            return session.exec(stmt).all()
    
    @staticmethod
    def get_analytics() -> dict:
        """Get dashboard analytics"""
        with rx.session() as session:
            total_jobs = session.exec(select(func.count(Job.id))).one()
            total_companies = session.exec(select(func.count(Company.id))).one()
            
            # Jobs by status
            applications_by_status = session.exec(
                select(
                    Application.status,
                    func.count(Application.id)
                ).group_by(Application.status)
            ).all()
            
            # Recent activity
            recent_jobs = session.exec(
                select(Job)
                .order_by(desc(Job.scraped_date))
                .limit(10)
            ).all()
            
            return {
                "total_jobs": total_jobs,
                "total_companies": total_companies,
                "applications_by_status": dict(applications_by_status),
                "recent_jobs": recent_jobs
            }
```

## 9. Performance Optimization

### 9.1 Performance Budgets

| Metric | Target | Critical |
|--------|--------|----------|
| Initial Load | < 2s | < 4s |
| Time to Interactive | < 3s | < 5s |
| First Contentful Paint | < 1s | < 2s |
| State Update | < 100ms | < 300ms |
| Search Response | < 500ms | < 1s |
| Scraping Update | < 200ms | < 500ms |

### 9.2 Optimization Strategies

```python
class OptimizedState(rx.State):
    """Performance-optimized state patterns"""
    
    # Use rx.cached_var for expensive computations
    @rx.cached_var
    def expensive_analytics(self) -> dict:
        """Cached computed var - only recalculates when dependencies change"""
        return calculate_complex_analytics(self.jobs)
    
    # Debounced search
    search_query: str = ""
    _search_timer: Optional[asyncio.Task] = None
    
    @rx.event
    async def on_search_change(self, value: str):
        """Debounced search input"""
        self.search_query = value
        
        # Cancel previous timer
        if self._search_timer:
            self._search_timer.cancel()
        
        # Set new timer
        self._search_timer = asyncio.create_task(self._perform_search())
    
    async def _perform_search(self):
        """Actual search after debounce"""
        await asyncio.sleep(0.3)  # 300ms debounce
        await self.load_jobs()
    
    # Pagination for large lists
    @rx.var
    def paginated_jobs(self) -> list[Job]:
        """Return only current page of jobs"""
        start = (self.page - 1) * self.per_page
        end = start + self.per_page
        return self.filtered_jobs[start:end]
    
    # Lazy loading
    @rx.event(background=True)
    async def load_job_details(self, job_id: str):
        """Load details only when needed"""
        if job_id not in self._cached_details:
            details = await fetch_job_details(job_id)
            async with self:
                self._cached_details[job_id] = details
```

### 9.3 Caching Strategy

```python
class CacheManager:
    """Client-side caching using rx.LocalStorage"""
    
    @staticmethod
    def cache_key(entity: str, id: str) -> str:
        return f"cache_{entity}_{id}"
    
    @rx.event
    def save_to_cache(self, entity: str, id: str, data: dict):
        """Save to browser local storage"""
        key = self.cache_key(entity, id)
        rx.LocalStorage.set(key, json.dumps(data))
        rx.LocalStorage.set(f"{key}_timestamp", str(time.time()))
    
    @rx.event
    def get_from_cache(self, entity: str, id: str, max_age: int = 3600):
        """Get from cache if not expired"""
        key = self.cache_key(entity, id)
        timestamp = rx.LocalStorage.get(f"{key}_timestamp")
        
        if timestamp:
            age = time.time() - float(timestamp)
            if age < max_age:
                data = rx.LocalStorage.get(key)
                return json.loads(data) if data else None
        
        return None
```

## 10. Accessibility

### 10.1 WCAG 2.2 AA Compliance

```python
def accessible_job_card(job: Job):
    """Accessible job card component"""
    return rx.card(
        rx.vstack(
            rx.heading(
                job.title,
                as_="h2",
                id=f"job-title-{job.id}"
            ),
            rx.text(
                f"{job.company.name} • {job.location}",
                aria_label=f"Company: {job.company.name}, Location: {job.location}"
            ),
            rx.button(
                "View Details",
                aria_label=f"View details for {job.title} at {job.company.name}",
                aria_describedby=f"job-title-{job.id}"
            ),
            role="article",
            aria_label=f"Job listing for {job.title}"
        ),
        tabindex="0",
        on_key_down=handle_keyboard_navigation
    )
```

### 10.2 Keyboard Navigation Map

| Key | Action | Context |
|-----|--------|---------|
| Tab | Navigate forward | Global |
| Shift+Tab | Navigate backward | Global |
| Enter | Activate element | Buttons, links |
| Space | Toggle element | Checkboxes, buttons |
| Arrow Keys | Navigate within | Lists, menus |
| Escape | Close/cancel | Modals, dropdowns |
| / | Focus search | Global shortcut |
| ? | Show help | Global shortcut |

## 11. Internationalization

### 11.1 i18n Architecture

```python
class I18nState(rx.State):
    """Internationalization state"""
    
    locale: str = "en-US"
    translations: dict = {}
    
    @rx.event
    def load_translations(self):
        """Load translations for current locale"""
        self.translations = load_locale_file(self.locale)
    
    @rx.var
    def t(self) -> Callable:
        """Translation function"""
        def translate(key: str, **kwargs):
            text = self.translations.get(key, key)
            return text.format(**kwargs) if kwargs else text
        return translate
    
    @rx.var
    def format_number(self) -> Callable:
        """Locale-aware number formatting"""
        def format_num(value: float, style: str = "decimal"):
            locale_obj = Locale.parse(self.locale)
            if style == "currency":
                return format_currency(value, "USD", locale=locale_obj)
            elif style == "percent":
                return format_percent(value, locale=locale_obj)
            else:
                return format_decimal(value, locale=locale_obj)
        return format_num
```

## 12. Error Handling

### 12.1 Error States

```python
class ErrorBoundary(rx.ComponentState):
    """Error boundary component"""
    
    has_error: bool = False
    error_message: str = ""
    
    @rx.event
    def handle_error(self, error: str):
        self.has_error = True
        self.error_message = error
        # Log to monitoring service
        log_error_to_sentry(error)
    
    @rx.event
    def reset_error(self):
        self.has_error = False
        self.error_message = ""
    
    @classmethod
    def get_component(cls, **props):
        return rx.cond(
            cls.has_error,
            error_fallback(
                message=cls.error_message,
                on_retry=cls.reset_error
            ),
            props.get("children", rx.fragment())
        )
```

### 12.2 Loading & Empty States

```python
def data_display(state):
    """Data display with loading and empty states"""
    return rx.cond(
        state.loading,
        loading_skeleton(),
        rx.cond(
            state.jobs.length() == 0,
            empty_state(
                icon="inbox",
                title="No jobs found",
                description="Try adjusting your filters or search criteria",
                action=rx.button(
                    "Clear filters",
                    on_click=state.clear_filters
                )
            ),
            job_list(state.jobs)
        )
    )
```

## 13. Testing Strategy

### 13.1 Component Testing

```python
# tests/test_job_card.py
import pytest
from reflex.testing import ReflexTestClient

def test_job_card_expansion():
    """Test job card expand/collapse functionality"""
    with ReflexTestClient() as client:
        # Create component instance
        job_card = JobCard.create(job=sample_job)
        
        # Initial state
        assert job_card.State.expanded is False
        
        # Trigger expand
        client.trigger_event(job_card, "toggle_expand")
        assert job_card.State.expanded is True
        
        # Trigger collapse
        client.trigger_event(job_card, "toggle_expand")
        assert job_card.State.expanded is False
```

### 13.2 State Testing

```python
def test_job_filtering():
    """Test job filtering logic"""
    state = JobState()
    state.jobs = [
        Job(title="Python Developer", company=Company(name="TechCo")),
        Job(title="Data Scientist", company=Company(name="DataCorp")),
    ]
    
    # Test search filter
    state.search_query = "Python"
    filtered = state.filtered_jobs
    assert len(filtered) == 1
    assert filtered[0].title == "Python Developer"
```

## 14. Deployment Configuration

### 14.1 Production Settings

```python
# rxconfig.py
import reflex as rx
from decouple import config

config = rx.Config(
    app_name="ai_job_scraper",
    
    # Database
    db_url=config("DATABASE_URL", default="sqlite:///reflex.db"),
    
    # Redis for background tasks
    redis_url=config("REDIS_URL", default=None),
    
    # Frontend
    frontend_port=3000,
    frontend_host="0.0.0.0",
    
    # Backend
    backend_port=8000,
    backend_host="0.0.0.0",
    
    # Production optimizations
    telemetry_enabled=False,
    tailwind={"plugins": ["@tailwindcss/forms", "@tailwindcss/typography"]},
    
    # API settings
    api_url=config("API_URL", default="http://localhost:8000"),
    deploy_url=config("DEPLOY_URL", default="https://ai-job-scraper.app"),
)
```

### 14.2 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Initialize Reflex
RUN reflex init

# Build frontend
RUN reflex export --frontend-only --no-zip

# Expose ports
EXPOSE 3000 8000

# Start application
CMD ["reflex", "run", "--env", "prod"]
```

## 15. Monitoring & Analytics

### 15.1 Performance Monitoring

```python
class MonitoringState(rx.State):
    """Application monitoring"""
    
    @rx.event
    def track_event(self, event_name: str, properties: dict = None):
        """Track user events"""
        if not properties:
            properties = {}
        
        # Add context
        properties.update({
            "user_id": self.user_id,
            "session_id": self.router.session.session_id,
            "timestamp": datetime.now().isoformat(),
            "route": self.router.page.path
        })
        
        # Send to analytics service
        send_to_mixpanel(event_name, properties)
    
    @rx.event
    def log_performance(self, metric: str, value: float):
        """Log performance metrics"""
        metrics_data = {
            "metric": metric,
            "value": value,
            "timestamp": time.time()
        }
        
        # Send to monitoring service
        send_to_datadog(metrics_data)
```

## 16. Security Considerations

### 16.1 Security Patterns

```python
class SecurityState(rx.State):
    """Security-focused state patterns"""
    
    @rx.event
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input"""
        # Remove potential XSS
        cleaned = bleach.clean(user_input)
        # Remove SQL injection attempts
        cleaned = cleaned.replace("'", "''")
        return cleaned
    
    @rx.event
    def validate_file_upload(self, file_data: dict) -> bool:
        """Validate file uploads"""
        allowed_types = ["application/pdf", "text/plain"]
        max_size = 5 * 1024 * 1024  # 5MB
        
        if file_data["content_type"] not in allowed_types:
            raise ValueError("Invalid file type")
        
        if file_data["size"] > max_size:
            raise ValueError("File too large")
        
        return True
```

## 17. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebSocket disconnection | High | Implement reconnection logic with exponential backoff |
| State desync | Medium | Use optimistic updates with rollback on failure |
| Large data sets | Medium | Implement virtual scrolling and pagination |
| Browser compatibility | Low | Use Reflex's built-in polyfills |
| SEO limitations | Low | Use static generation for public pages |

## Conclusion

This architecture provides a robust, scalable foundation for the AI Job Scraper application using Reflex. It leverages Reflex's strengths in state management, real-time updates, and pure Python development while implementing best practices for performance, accessibility, and user experience.

The modular component architecture, combined with comprehensive state management and real-time capabilities, ensures the application can scale from MVP to production while maintaining code quality and developer productivity.
