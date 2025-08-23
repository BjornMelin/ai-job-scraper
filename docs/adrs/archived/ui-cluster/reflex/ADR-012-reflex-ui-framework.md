# ADR-012: Reflex UI Framework Decision

## Status

**Accepted** - *Implementation in progress*

**Current Reality**: Active migration from Streamlit to Reflex UI framework. Implementation follows component-based architecture as defined in this ADR and supporting ADRs (013, 014, 015, 016, 020).

## Context

The AI Job Scraper requires a modern web UI to enable users to browse jobs, track applications, manage companies, and monitor scraping activities. The current codebase uses Streamlit for basic UI, but we need a more robust solution that can scale with the application's growing requirements.

### Requirements

- Pure Python development (no JavaScript/TypeScript required)
- Real-time updates for scraping progress
- Rich component library for modern UI
- WebSocket support for live data streaming
- SQLModel integration for database operations
- Production-ready deployment options
- Minimal maintenance overhead
- Support for complex interactions and state management

### Current Challenges

- Streamlit limitations in complex UI patterns
- Need for real-time bidirectional communication
- Requirement for production-grade web application features
- Team expertise is primarily Python-focused

## Decision Drivers

1. **Developer Productivity**: Maintain Python-only development
2. **Real-time Capabilities**: Support WebSocket connections for live updates
3. **Component Ecosystem**: Rich set of pre-built components
4. **State Management**: Robust state management for complex interactions
5. **Database Integration**: Native SQLModel/SQLAlchemy support
6. **Production Readiness**: Suitable for deployment at scale
7. **Learning Curve**: Minimal for Python developers

## Considered Options

### Option 1: Continue with Streamlit

**Pros:**

- Already in use, no migration needed
- Simple for basic dashboards
- Good for prototyping

**Cons:**

- Limited to linear, top-down execution model
- No true client-side state management
- Limited component customization
- Poor support for complex interactions
- Not suitable for production web apps

### Option 2: Django + HTMX

**Pros:**

- Mature, battle-tested framework
- Great for server-side rendering
- Strong security features
- Excellent ORM

**Cons:**

- Requires HTML/JavaScript knowledge for advanced features
- Not pure Python for frontend
- More boilerplate code
- Steeper learning curve for full-stack features

### Option 3: FastAPI + React/Vue (Traditional Split)

**Pros:**

- Best-in-class performance
- Industry-standard architecture
- Maximum flexibility
- Rich ecosystem

**Cons:**

- Requires JavaScript/TypeScript expertise
- Two separate codebases to maintain
- Complex deployment
- Against our pure Python requirement

### Option 4: Reflex Framework

**Pros:**

- Pure Python for entire stack
- Built-in WebSocket support for real-time updates
- Rich component library (50+ components)
- Reactive state management
- Native SQLModel integration
- Single codebase deployment
- Built on proven tech (FastAPI backend, Next.js frontend)
- Component-based architecture with reusable functions
- Real-time updates with yield pattern

**Cons:**

- Relatively newer framework (but actively maintained)
- Smaller community compared to Django/FastAPI
- Some advanced customizations may require React knowledge
- SEO limitations for dynamic content

### Option 5: Dash

**Pros:**

- Pure Python
- Good for data visualization
- Plotly integration

**Cons:**

- Primarily for analytics dashboards
- Limited general-purpose components
- Callback-based model can become complex
- Not ideal for CRUD operations

## Decision

**We will adopt Reflex as our UI framework.**

## Rationale

Reflex best aligns with our requirements and constraints:

1. **Pure Python Development**: Entire application in Python, matching team expertise
2. **Real-time First**: WebSockets built-in, perfect for scraping progress updates
3. **Modern Architecture**: Component-based with proper state management
4. **Database Integration**: Native SQLModel support aligns with our stack
5. **Rapid Development**: Rich component library accelerates development
6. **Production Ready**: Built on FastAPI and Next.js, proven technologies
7. **Single Deployment**: One codebase, simplified DevOps

The framework's approach of compiling Python to React components gives us the best of both worlds: Python development experience with modern web UI capabilities.

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive

- Unified Python codebase reduces context switching
- Faster development with pre-built components
- Built-in real-time capabilities for scraping updates
- Simplified deployment and maintenance
- Lower barrier to entry for Python developers
- Strong type safety with Pydantic integration

### Negative

- Learning curve for Reflex-specific patterns
- Dependency on framework's continued development
- May need custom React components for advanced features
- Limited SEO capabilities for public-facing pages
- Smaller ecosystem compared to established frameworks

### Neutral

- Need to migrate from Streamlit (one-time effort)
- Documentation and examples still growing
- Performance characteristics need monitoring in production

## Implementation Plan

1. Create proof-of-concept with core features
2. Migrate database models to Reflex's rx.Model
3. Implement real-time scraping dashboard
4. Build job browsing and filtering UI
5. Add application tracking features
6. Deploy to staging environment
7. Performance testing and optimization
8. Production deployment

## Related ADRs

- **ADR-013**: State Management Architecture - Defines state patterns for Reflex application
- **ADR-014**: Real-time Updates Strategy - WebSocket implementation details
- **ADR-015**: Component Library Selection - UI component choices within Reflex
- **ADR-016**: Routing and Navigation Design - URL handling and page structure
- **ADR-020**: Reflex Local Development - Development-specific UI patterns

## References

- [Reflex Documentation](https://reflex.dev/docs/)
- [Reflex GitHub Repository](https://github.com/reflex-dev/reflex)
- [SQLModel Integration Guide](https://reflex.dev/docs/database/overview/)
- [WebSocket State Management](https://reflex.dev/blog/2024-03-21-reflex-architecture/)
