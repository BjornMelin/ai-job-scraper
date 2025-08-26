"""AI Job Scraper - Reflex UI Implementation Stub.

This file demonstrates the core architecture patterns for the Reflex-based UI.
"""

import asyncio

from datetime import UTC, datetime
from typing import ClassVar
from uuid import uuid4

import reflex as rx

# ============================================================================
# Configuration
# ============================================================================

config = rx.Config(
    app_name="ai_job_scraper",
    db_url="sqlite:///reflex.db",
    frontend_port=3000,
    backend_port=8000,
)

# ============================================================================
# Design Tokens (from design-tokens.json)
# ============================================================================

COLORS = {
    "primary": "#3B82F6",
    "secondary": "#9333EA",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
}

SPACING = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "1rem",
    "lg": "1.5rem",
    "xl": "2rem",
}

# ============================================================================
# Models (SQLModel integration)
# ============================================================================


class Job(rx.Model, table=True):
    """Job model with SQLModel."""

    id: str
    title: str
    company_name: str
    location: str
    remote_type: str = "onsite"  # onsite | remote | hybrid
    salary_min: int | None = None
    salary_max: int | None = None
    description: str
    posted_date: datetime
    scraped_date: datetime = datetime.now(UTC)
    source_url: str
    application_status: str | None = None


class ScrapingSession(rx.Model, table=True):
    """Scraping session tracking."""

    id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"  # running | completed | failed
    jobs_found: int = 0


# ============================================================================
# Global Application State
# ============================================================================


class AppState(rx.State):
    """Root application state."""

    # Navigation
    current_route: str = "/"
    sidebar_open: bool = True

    # Theme
    dark_mode: bool = False

    # Notifications
    notifications: ClassVar[list[dict]] = []

    @rx.event
    def toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.sidebar_open = not self.sidebar_open

    @rx.event
    def toggle_theme(self):
        """Toggle dark/light mode."""
        self.dark_mode = not self.dark_mode

    @rx.event
    def add_notification(self, message: str, notification_type: str = "info"):
        """Add notification to queue."""
        notification = {
            "id": str(uuid4()),
            "message": message,
            "type": notification_type,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.notifications.insert(0, notification)

        # Keep only last 10 notifications
        if len(self.notifications) > 10:
            self.notifications = self.notifications[:10]

    @rx.var
    def theme_class(self) -> str:
        """Computed theme class."""
        return "dark" if self.dark_mode else "light"


# ============================================================================
# Job Management State
# ============================================================================


class JobState(AppState):
    """Job browsing and filtering state."""

    # Data
    jobs: ClassVar[list[Job]] = []
    selected_job: Job | None = None

    # Filters
    search_query: str = ""
    filter_company: str = ""
    filter_remote: str = "all"

    # UI State
    loading: bool = False
    view_mode: str = "cards"  # cards | table

    # Pagination
    page: int = 1
    per_page: int = 20

    @rx.event(background=True)
    async def load_jobs(self):
        """Load jobs from database with filters."""
        async with self:
            self.loading = True

        # Simulate database query
        await asyncio.sleep(1)

        # In real implementation, query database with filters
        # Database query would go here
        # session.exec(select(Job)...).all()

        async with self:
            self.loading = False
            self.add_notification(f"Loaded {len(self.jobs)} jobs", "success")

    @rx.event
    def select_job(self, job_id: str):
        """Select a job for detailed view."""
        self.selected_job = next((j for j in self.jobs if j.id == job_id), None)

    @rx.var
    def filtered_jobs(self) -> list[Job]:
        """Computed var for filtered jobs."""
        filtered = self.jobs

        if self.search_query:
            query = self.search_query.lower()
            filtered = [
                j
                for j in filtered
                if query in j.title.lower() or query in j.description.lower()
            ]

        if self.filter_company:
            filtered = [j for j in filtered if j.company_name == self.filter_company]

        if self.filter_remote != "all":
            filtered = [j for j in filtered if j.remote_type == self.filter_remote]

        return filtered

    @rx.var
    def total_pages(self) -> int:
        """Calculate total pages."""
        import math

        return max(1, math.ceil(len(self.filtered_jobs) / self.per_page))


# ============================================================================
# Real-time Scraping State
# ============================================================================


class ScrapingState(AppState):
    """Real-time scraping management."""

    # Session
    current_session: ScrapingSession | None = None
    is_scraping: bool = False

    # Progress
    total_sources: int = 0
    completed_sources: int = 0
    current_source: str = ""
    jobs_found: int = 0

    # Log
    log_messages: ClassVar[list[str]] = []

    @rx.event(background=True)
    async def start_scraping(self, sources: list[str]):
        """Start scraping with real-time updates."""
        async with self:
            self.is_scraping = True
            self.total_sources = len(sources)
            self.completed_sources = 0
            self.jobs_found = 0
            self.log_messages = []

            # Create session
            self.current_session = ScrapingSession(
                id=str(uuid4()), started_at=datetime.now(UTC), status="running"
            )
            self.add_log(f"Starting scraping session with {len(sources)} sources...")

        for source in sources:
            async with self:
                self.current_source = source
                self.add_log(f"Scraping {source}...")

            # Simulate scraping
            await asyncio.sleep(2)
            jobs_found = 15  # Simulated

            async with self:
                self.jobs_found += jobs_found
                self.completed_sources += 1
                self.add_log(f"✓ Found {jobs_found} jobs from {source}")

        async with self:
            self.is_scraping = False
            if self.current_session:
                self.current_session.completed_at = datetime.now(UTC)
                self.current_session.status = "completed"
                self.current_session.jobs_found = self.jobs_found

            self.add_notification(
                f"Scraping complete! Found {self.jobs_found} jobs", "success"
            )

    @rx.event
    def add_log(self, message: str):
        """Add log message."""
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")

        # Keep last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages = self.log_messages[-100:]

    @rx.event
    def stop_scraping(self):
        """Stop current scraping session."""
        self.is_scraping = False
        if self.current_session:
            self.current_session.status = "cancelled"
        self.add_notification("Scraping cancelled", "warning")

    @rx.var
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_sources == 0:
            return 0
        return (self.completed_sources / self.total_sources) * 100


# ============================================================================
# Component State (Reusable Components)
# ============================================================================


class JobCard(rx.ComponentState):
    """Reusable job card component with local state."""

    expanded: bool = False
    saved: bool = False

    @rx.event
    def toggle_expand(self):
        """Toggle expanded state of job card."""
        self.expanded = not self.expanded

    @rx.event
    def toggle_save(self):
        """Toggle saved state of job card."""
        self.saved = not self.saved

    @classmethod
    def get_component(cls, job: dict, **props):
        """Render job card component."""
        return rx.card(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.vstack(
                        rx.heading(job.get("title", ""), size="4"),
                        rx.text(job.get("company_name", ""), color="gray"),
                        align_items="start",
                    ),
                    rx.spacer(),
                    rx.icon_button(
                        rx.icon("bookmark" if cls.saved else "bookmark-outline"),
                        on_click=cls.toggle_save,
                        variant="ghost",
                    ),
                    width="100%",
                ),
                # Tags
                rx.hstack(
                    rx.badge(job.get("location", ""), variant="secondary"),
                    rx.badge(job.get("remote_type", ""), variant="outline"),
                    rx.cond(
                        job.get("salary_max"),
                        rx.badge(f"${job.get('salary_max', 0):,}", variant="success"),
                    ),
                ),
                # Description content
                rx.cond(
                    cls.expanded,
                    rx.vstack(
                        rx.divider(),
                        rx.text(job.get("description", "")[:200] + "..."),
                        rx.hstack(
                            rx.button(
                                "Apply",
                                color_scheme="green",
                                size="sm",
                            ),
                            rx.button(
                                "View Details",
                                variant="outline",
                                size="sm",
                            ),
                        ),
                        width="100%",
                    ),
                ),
                # Toggle button
                rx.button(
                    rx.cond(cls.expanded, "Show Less", "Show More"),
                    on_click=cls.toggle_expand,
                    variant="ghost",
                    size="sm",
                    width="100%",
                ),
                spacing=SPACING["md"],
                width="100%",
            ),
            **props,
        )


# Create reusable component
job_card = JobCard.create

# ============================================================================
# UI Components
# ============================================================================


def navbar():
    """Navigation bar component."""
    return rx.hstack(
        rx.heading("AI Job Scraper", size="6"),
        rx.spacer(),
        rx.hstack(
            rx.badge(
                rx.text(f"{len(JobState.filtered_jobs)} jobs"),
                variant="soft",
            ),
            rx.icon_button(
                rx.icon("moon" if AppState.dark_mode else "sun"),
                on_click=AppState.toggle_theme,
                variant="ghost",
            ),
            rx.icon_button(
                rx.icon("bell"),
                variant="ghost",
            ),
            spacing=SPACING["sm"],
        ),
        height="64px",
        padding=SPACING["lg"],
        border_bottom="1px solid",
        border_color="gray.200",
        width="100%",
    )


def sidebar():
    """Sidebar navigation."""
    nav_items = [
        {"name": "Dashboard", "icon": "home", "route": "/"},
        {"name": "Jobs", "icon": "briefcase", "route": "/jobs"},
        {"name": "Companies", "icon": "building", "route": "/companies"},
        {"name": "Applications", "icon": "file-text", "route": "/applications"},
        {"name": "Scraping", "icon": "download", "route": "/scraping"},
        {"name": "Settings", "icon": "settings", "route": "/settings"},
    ]

    return rx.drawer(
        rx.drawer.overlay(),
        rx.drawer.content(
            rx.vstack(
                rx.foreach(
                    nav_items,
                    lambda item: rx.link(
                        rx.hstack(
                            rx.icon(item["icon"]),
                            rx.text(item["name"]),
                            padding=SPACING["md"],
                            border_radius="md",
                            width="100%",
                            _hover={"bg": "gray.100"},
                        ),
                        href=item["route"],
                    ),
                ),
                padding=SPACING["lg"],
                width="240px",
                height="100vh",
                bg="gray.50",
            ),
        ),
        open=AppState.sidebar_open,
    )


def search_bar():
    """Search and filter bar."""
    return rx.hstack(
        rx.input(
            placeholder="Search jobs...",
            value=JobState.search_query,
            on_change=JobState.set_search_query,
            width="300px",
        ),
        rx.select(
            ["all", "remote", "hybrid", "onsite"],
            value=JobState.filter_remote,
            on_change=JobState.set_filter_remote,
            placeholder="Remote type",
        ),
        rx.button(
            "Search",
            on_click=JobState.load_jobs,
            loading=JobState.loading,
        ),
        spacing=SPACING["md"],
    )


def job_list():
    """Job listing with cards or table view."""
    return rx.cond(
        JobState.loading,
        rx.center(
            rx.spinner(size="lg"),
            min_height="400px",
        ),
        rx.cond(
            JobState.view_mode == "cards",
            rx.grid(
                rx.foreach(
                    JobState.filtered_jobs,
                    lambda job: job_card(job),
                ),
                columns=[1, 2, 3],  # Responsive columns
                spacing=SPACING["lg"],
                width="100%",
            ),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Title"),
                        rx.table.column_header_cell("Company"),
                        rx.table.column_header_cell("Location"),
                        rx.table.column_header_cell("Type"),
                        rx.table.column_header_cell("Actions"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        JobState.filtered_jobs,
                        lambda job: rx.table.row(
                            rx.table.cell(job["title"]),
                            rx.table.cell(job["company_name"]),
                            rx.table.cell(job["location"]),
                            rx.table.cell(job["remote_type"]),
                            rx.table.cell(
                                rx.button("View", size="sm", variant="outline")
                            ),
                        ),
                    ),
                ),
                width="100%",
            ),
        ),
    )


def scraping_dashboard():
    """Real-time scraping dashboard."""
    return rx.vstack(
        rx.heading("Scraping Center", size="5"),
        # Control panel
        rx.card(
            rx.vstack(
                rx.heading("Start New Scraping Session", size="4"),
                rx.checkbox_group(
                    ["LinkedIn", "Indeed", "Glassdoor", "AngelList"],
                    direction="row",
                ),
                rx.hstack(
                    rx.button(
                        "Start Scraping",
                        on_click=lambda: ScrapingState.start_scraping(
                            ["LinkedIn", "Indeed", "Glassdoor"]
                        ),
                        color_scheme="green",
                        loading=ScrapingState.is_scraping,
                        disabled=ScrapingState.is_scraping,
                    ),
                    rx.cond(
                        ScrapingState.is_scraping,
                        rx.button(
                            "Stop",
                            on_click=ScrapingState.stop_scraping,
                            color_scheme="red",
                        ),
                    ),
                ),
                spacing=SPACING["md"],
            ),
        ),
        # Progress
        rx.cond(
            ScrapingState.is_scraping,
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.text("Progress:"),
                        rx.text(
                            f"{ScrapingState.completed_sources}/"
                            f"{ScrapingState.total_sources} sources"
                        ),
                        rx.spacer(),
                        rx.text(f"{ScrapingState.jobs_found} jobs found"),
                    ),
                    rx.progress(value=ScrapingState.progress_percentage),
                    rx.text(
                        f"Current: {ScrapingState.current_source}",
                        size="sm",
                        color="gray",
                    ),
                    spacing=SPACING["sm"],
                ),
            ),
        ),
        # Log output
        rx.card(
            rx.vstack(
                rx.heading("Activity Log", size="4"),
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(
                            ScrapingState.log_messages,
                            lambda msg: rx.text(msg, font_family="mono", size="sm"),
                        ),
                        spacing="xs",
                    ),
                    height="300px",
                ),
            ),
        ),
        spacing=SPACING["lg"],
        width="100%",
    )


# ============================================================================
# Pages
# ============================================================================


def dashboard_page():
    """Main dashboard page."""
    return rx.vstack(
        navbar(),
        rx.hstack(
            sidebar(),
            rx.container(
                rx.vstack(
                    rx.heading("Dashboard", size="5"),
                    rx.grid(
                        rx.card(
                            rx.stat(
                                rx.stat.label("Total Jobs"),
                                rx.stat.number(len(JobState.jobs)),
                                rx.stat.help_text("↑ 12% from last week"),
                            ),
                        ),
                        rx.card(
                            rx.stat(
                                rx.stat.label("Applications"),
                                rx.stat.number("24"),
                                rx.stat.help_text("3 pending"),
                            ),
                        ),
                        rx.card(
                            rx.stat(
                                rx.stat.label("Companies"),
                                rx.stat.number("156"),
                                rx.stat.help_text("8 new this week"),
                            ),
                        ),
                        rx.card(
                            rx.stat(
                                rx.stat.label("Success Rate"),
                                rx.stat.number("18%"),
                                rx.stat.help_text("Industry avg: 12%"),
                            ),
                        ),
                        columns=[1, 2, 4],
                        spacing=SPACING["lg"],
                        width="100%",
                    ),
                    spacing=SPACING["xl"],
                ),
                max_width="1280px",
                padding=SPACING["xl"],
            ),
            width="100%",
            spacing="0",
        ),
        spacing="0",
        on_mount=JobState.load_jobs,
    )


def jobs_page():
    """Jobs browsing page."""
    return rx.vstack(
        navbar(),
        rx.hstack(
            sidebar(),
            rx.container(
                rx.vstack(
                    rx.heading("Browse Jobs", size="5"),
                    search_bar(),
                    rx.hstack(
                        rx.text(f"{len(JobState.filtered_jobs)} jobs found"),
                        rx.spacer(),
                        rx.radio_group(
                            ["cards", "table"],
                            value=JobState.view_mode,
                            on_change=JobState.set_view_mode,
                            direction="row",
                        ),
                    ),
                    job_list(),
                    # Pagination
                    rx.hstack(
                        rx.button(
                            "Previous",
                            on_click=lambda: JobState.set_page(
                                max(1, JobState.page - 1)
                            ),
                            disabled=JobState.page == 1,
                        ),
                        rx.text(f"Page {JobState.page} of {JobState.total_pages}"),
                        rx.button(
                            "Next",
                            on_click=lambda: JobState.set_page(
                                min(JobState.total_pages, JobState.page + 1)
                            ),
                            disabled=JobState.page == JobState.total_pages,
                        ),
                    ),
                    spacing=SPACING["lg"],
                ),
                max_width="1280px",
                padding=SPACING["xl"],
            ),
            width="100%",
            spacing="0",
        ),
        spacing="0",
    )


def scraping_page():
    """Scraping management page."""
    return rx.vstack(
        navbar(),
        rx.hstack(
            sidebar(),
            rx.container(
                scraping_dashboard(),
                max_width="1280px",
                padding=SPACING["xl"],
            ),
            width="100%",
            spacing="0",
        ),
        spacing="0",
    )


# ============================================================================
# App Configuration
# ============================================================================

# Create app
app = rx.App()

# Add pages
app.add_page(dashboard_page, route="/", title="Dashboard - AI Job Scraper")
app.add_page(jobs_page, route="/jobs", title="Browse Jobs - AI Job Scraper")
app.add_page(scraping_page, route="/scraping", title="Scraping Center - AI Job Scraper")

# Run app
if __name__ == "__main__":
    app.compile()
