# ADR-016: Routing and Navigation Design

## Status

**Accepted** - *Scope: Production architecture*

## Context

The AI Job Scraper requires a robust routing and navigation system to support:

- Multiple pages (Dashboard, Jobs, Companies, Applications, Scraping, Settings)
- Dynamic routes for job details, company profiles
- Deep linking for sharing specific jobs
- Browser history management
- Protected routes (future authentication)
- URL state persistence for filters and search
- Mobile-responsive navigation patterns

Users need intuitive navigation that maintains state across page transitions and supports common web patterns like back/forward buttons and bookmarkable URLs.

## Decision Drivers

1. **User Experience**: Intuitive navigation patterns
2. **State Persistence**: Maintain filter/search state in URLs
3. **Performance**: Fast page transitions
4. **SEO**: Support for meta tags and crawlability
5. **Mobile Support**: Responsive navigation
6. **Deep Linking**: Shareable URLs for specific content
7. **Developer Experience**: Simple route definition

## Considered Options

### Option 1: Single Page Application (SPA) Only

All content in one page with client-side routing.

**Pros:**

- No page reloads
- Fast transitions
- Simple state management

**Cons:**

- Poor SEO
- No deep linking
- Large initial bundle
- Poor accessibility

### Option 2: Multi-Page Application (MPA)

Traditional server-side routing with full page reloads.

**Pros:**

- Better SEO
- Simple mental model
- Works without JavaScript

**Cons:**

- Slower transitions
- State loss on navigation
- Poor user experience
- Not modern

### Option 3: Hybrid SPA/MPA with Reflex Pages

Use Reflex's page system with client-side navigation.

**Pros:**

- Fast client-side transitions
- URL-based routing
- Deep linking support
- State preservation options
- SEO possibilities

**Cons:**

- Complexity of hybrid approach
- Need to manage URL state
- Some SEO limitations

### Option 4: Custom Router Implementation

Build custom routing on top of Reflex.

**Pros:**

- Full control
- Exact feature match

**Cons:**

- Significant development effort
- Maintenance burden
- Reinventing the wheel
- Potential bugs

## Decision

**We will use Hybrid SPA/MPA with Reflex Pages, implementing URL state management for filters and search.**

## Detailed Design

### Route Structure

```python
# Route definitions
ROUTES = {
    # Static routes
    "/": "dashboard",
    "/jobs": "jobs_listing",
    "/companies": "companies_listing",
    "/applications": "applications_tracker",
    "/scraping": "scraping_center",
    "/analytics": "analytics_dashboard",
    "/settings": "user_settings",
    
    # Dynamic routes
    "/jobs/[job_id]": "job_detail",
    "/companies/[company_id]": "company_detail",
    "/applications/[application_id]": "application_detail",
    
    # Nested routes
    "/settings/profile": "profile_settings",
    "/settings/preferences": "preferences_settings",
    "/settings/notifications": "notification_settings",
}
```

### Navigation Implementation

```python
# Navigation state
class NavigationState(rx.State):
    """Global navigation state"""
    
    # Current location
    current_route: str = "/"
    breadcrumbs: list[dict] = []
    
    # Navigation UI state
    sidebar_open: bool = True
    mobile_menu_open: bool = False
    
    # Navigation history
    navigation_history: list[str] = []
    
    @rx.event
    def navigate_to(self, route: str, params: dict = None):
        """Programmatic navigation with state"""
        # Add to history
        self.navigation_history.append(self.current_route)
        
        # Build URL with params
        if params:
            query_string = urlencode(params)
            route = f"{route}?{query_string}"
        
        # Navigate
        return rx.redirect(route)
    
    @rx.event
    def on_route_change(self):
        """Handle route changes"""
        # Update current route
        self.current_route = self.router.page.path
        
        # Update breadcrumbs
        self.update_breadcrumbs()
        
        # Track navigation
        self.track_page_view()
    
    @rx.var
    def active_nav_item(self) -> str:
        """Determine active navigation item"""
        path = self.router.page.path
        
        # Match the most specific route
        if path.startswith("/jobs"):
            return "jobs"
        elif path.startswith("/companies"):
            return "companies"
        elif path.startswith("/applications"):
            return "applications"
        elif path.startswith("/scraping"):
            return "scraping"
        elif path.startswith("/settings"):
            return "settings"
        else:
            return "dashboard"
```

### URL State Management

```python
class URLStateManager(rx.State):
    """Manage application state in URLs"""
    
    # URL parameters
    url_params: dict = {}
    
    @rx.event
    def on_mount(self):
        """Parse URL parameters on page load"""
        # Get params from router
        params = self.router.page.params
        
        # Parse and apply filters
        if "search" in params:
            self.search_query = params["search"]
        
        if "filters" in params:
            self.parse_filters(params["filters"])
        
        if "sort" in params:
            self.sort_by = params["sort"]
        
        # Load data with filters
        return self.load_filtered_data()
    
    @rx.event
    def update_url_state(self, key: str, value: Any):
        """Update URL with state changes"""
        # Get current params
        params = dict(self.router.page.params)
        
        # Update specific param
        if value:
            params[key] = str(value)
        else:
            params.pop(key, None)
        
        # Update URL without navigation
        new_url = self.build_url(self.router.page.path, params)
        
        return rx.call_script(
            f"window.history.replaceState(null, '', '{new_url}')"
        )
    
    def build_url(self, path: str, params: dict) -> str:
        """Build URL with query parameters"""
        if params:
            query = urlencode(params)
            return f"{path}?{query}"
        return path
```

### Navigation Components

```python
# Sidebar navigation
def sidebar():
    return rx.drawer(
        rx.drawer.content(
            rx.vstack(
                # Logo
                rx.heading("AI Job Scraper", size="lg"),
                rx.divider(),
                
                # Primary navigation
                rx.vstack(
                    *[
                        nav_item(
                            route=item["route"],
                            name=item["name"],
                            icon=item["icon"],
                            active=NavigationState.active_nav_item == item["key"]
                        )
                        for item in PRIMARY_NAV_ITEMS
                    ],
                    spacing="2"
                ),
                
                rx.spacer(),
                
                # Secondary navigation
                rx.vstack(
                    *[
                        nav_item(
                            route=item["route"],
                            name=item["name"],
                            icon=item["icon"],
                            active=NavigationState.active_nav_item == item["key"]
                        )
                        for item in SECONDARY_NAV_ITEMS
                    ],
                    spacing="2"
                ),
                
                width="100%",
                height="100%",
                padding="4"
            ),
            width="240px",
            background_color=rx.color("gray", 2)
        ),
        open=NavigationState.sidebar_open
    )

def nav_item(route: str, name: str, icon: str, active: bool):
    return rx.link(
        rx.hstack(
            rx.icon(icon, size=20),
            rx.text(name, weight="medium" if active else "normal"),
            padding="2",
            border_radius="md",
            background_color=rx.cond(
                active,
                rx.color("blue", 3),
                "transparent"
            ),
            _hover={"background_color": rx.color("gray", 3)},
            width="100%"
        ),
        href=route,
        text_decoration="none"
    )

# Breadcrumbs
def breadcrumbs():
    return rx.hstack(
        rx.foreach(
            NavigationState.breadcrumbs,
            lambda crumb: rx.hstack(
                rx.link(
                    crumb["label"],
                    href=crumb["route"],
                    color=rx.color("gray", 11)
                ),
                rx.text("/", color=rx.color("gray", 8))
            )
        ),
        spacing="2"
    )

# Mobile navigation
def mobile_nav():
    return rx.drawer(
        rx.drawer.trigger(
            rx.icon_button(
                rx.icon("menu"),
                display=["flex", "flex", "none"]  # Hide on desktop
            )
        ),
        rx.drawer.content(
            mobile_menu_content(),
            width="80%"
        ),
        direction="left"
    )
```

### Dynamic Route Handling

```python
# Dynamic job detail page
@rx.page(route="/jobs/[job_id]", on_load=JobDetailState.load_job)
def job_detail():
    return app_layout(
        rx.vstack(
            breadcrumbs(),
            rx.cond(
                JobDetailState.loading,
                loading_skeleton(),
                job_detail_content()
            ),
            width="100%"
        )
    )

class JobDetailState(rx.State):
    """Job detail page state"""
    
    job_id: str = ""
    job: Optional[Job] = None
    loading: bool = True
    
    @rx.event
    async def load_job(self):
        """Load job details from route param"""
        # Get job ID from route
        self.job_id = self.router.page.params.get("job_id", "")
        
        if not self.job_id:
            return rx.redirect("/jobs")
        
        self.loading = True
        
        # Fetch job details
        with rx.session() as session:
            self.job = session.exec(
                select(Job).where(Job.id == self.job_id)
            ).first()
        
        if not self.job:
            return rx.redirect("/404")
        
        self.loading = False
        
        # Update page metadata
        self.update_meta_tags()
    
    def update_meta_tags(self):
        """Update page meta tags for SEO"""
        return rx.call_script(f"""
            document.title = '{self.job.title} - AI Job Scraper';
            document.querySelector('meta[property="og:title"]')
                ?.setAttribute('content', '{self.job.title}');
            document.querySelector('meta[property="og:description"]')
                ?.setAttribute('content', '{self.job.description[:150]}');
        """)
```

### Protected Routes (Future)

```python
def protected_route(page_func):
    """Decorator for protected routes"""
    def wrapper():
        return rx.cond(
            AuthState.is_authenticated,
            page_func(),
            rx.redirect("/login")
        )
    return wrapper

# Usage
@rx.page(route="/settings")
@protected_route
def settings_page():
    return settings_content()
```

## Navigation Patterns

### 1. Primary Navigation

- Persistent sidebar on desktop
- Collapsible drawer on mobile
- Active state indication
- Icon + text labels

### 2. Secondary Navigation

- Tab navigation for sub-pages
- Breadcrumb trail
- Back button for detail pages

### 3. Contextual Navigation

- Related links in content
- "See also" sections
- Quick actions

### 4. URL Patterns

```text
/                          # Dashboard
/jobs                      # Job listing
/jobs?search=python        # With search
/jobs?filters=remote,senior # With filters
/jobs/123                  # Job detail
/companies                 # Company listing
/companies/456             # Company detail
/applications              # Application tracker
/scraping                  # Scraping center
/settings                  # Settings
/settings/profile          # Nested route
```

## Rationale

The hybrid approach provides:

1. **Modern UX**: Fast client-side transitions
2. **Web Standards**: URLs reflect app state
3. **Shareability**: Deep links to specific content
4. **Flexibility**: Support for various navigation patterns
5. **Mobile-First**: Responsive navigation
6. **Future-Proof**: Can add authentication/authorization

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive

- Intuitive navigation for users
- Bookmarkable and shareable URLs
- Fast page transitions
- State persistence across navigation
- Mobile-responsive design
- Analytics-friendly

### Negative

- Complexity of URL state management
- Need to handle edge cases
- Browser compatibility considerations
- Testing navigation flows

### Neutral

- Need navigation documentation
- URL schema must be maintained
- Route guards for future features

## Implementation Checklist

- [ ] Define all routes and URL patterns
- [ ] Implement sidebar navigation component
- [ ] Create mobile navigation drawer
- [ ] Add breadcrumb component
- [ ] Implement URL state manager
- [ ] Add dynamic route handlers
- [ ] Create 404 page
- [ ] Add loading states for navigation
- [ ] Test browser back/forward
- [ ] Verify deep linking works
- [ ] Add navigation analytics

## Related ADRs

- **ADR-012**: Reflex UI Framework Decision - Foundation framework for routing capabilities
- **ADR-013**: State Management Architecture - URL state management integration  
- **ADR-014**: Real-time Updates Strategy - State preservation across navigation
- **ADR-015**: Component Library Selection - Navigation component implementations
- **ADR-020**: Reflex Local Development - Development navigation patterns

## Implementation Resources

- **URL State Patterns**: See `/docs/planning/url-state-management-patterns.md` for comprehensive URL state management patterns adapted from Streamlit prototyping for Reflex implementation

## References

- [Reflex Routing Documentation](https://reflex.dev/docs/pages/overview/)
- [Dynamic Routes in Reflex](https://reflex.dev/docs/pages/dynamic-routing/)
- [URL State Management Patterns](https://www.patterns.dev/posts/url-state/)
- [Navigation Best Practices](https://web.dev/navigation/)
- [Mobile Navigation Patterns](https://m3.material.io/foundations/navigation)
