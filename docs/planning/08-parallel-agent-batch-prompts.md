# Parallel AI Agent Batch Execution - Operating Manual & Prompts

> *AI Job Scraper Modernization Project - 1-Week Sprint*  
> *Last Updated: August 2025*

## 🎯 Project Mission

Engineer and deliver a robust, feature-complete, modern job scraping and tracking application within **1 week**, by orchestrating parallel AI coding agents (including Claude Code, ChatGPT, etc.) using batch-based, dependency-aware, conflict-free parallel execution.

Transform the basic Streamlit job scraper into a modern, desktop-optimized job hunting platform with:

- Pinterest-style job grid layouts

- Real-time progress tracking with multi-level visualization

- Smart database synchronization with change detection

- Advanced filtering and search capabilities

- Application workflow tracking

- Analytics dashboard with trend visualization

- Performance optimization (sub-100ms search, 60fps animations)

## 🟢 Core Directives & Guiding Principles

### **Mission Statement**

Complete all 171 tasks in record time by dividing the full roadmap into groups ("batches") of truly independent tasks. Each group is executed in parallel by multiple AI subagents. The app must be robust, reliable, beautiful, performant, and maintainable for daily power-user job search.

### **Library-First Development**

Always use the most advanced, high-performance, and project-approved libraries and their built-in features. Avoid custom code unless there's a clear, justified gap. Priority libraries:

- **Streamlit 1.47.1+** with modern component ecosystem

- **streamlit-aggrid v1.1.7** for advanced data grids

- **streamlit-elements** for draggable dashboards and Material-UI

- **streamlit-shadcn-ui** for modern design system components

- **streamlit-lottie** for high-quality animations

- **Plotly** for interactive charts and analytics

- **SQLModel** with enhanced relationships and indexing

- **FastAPI patterns** for async operations where applicable

### **KISS / DRY / YAGNI Principles**

- **KISS (Keep It Simple)**: Prioritize simplicity and maintainability over clever solutions

- **DRY (Don't Repeat Yourself)**: Factor common logic into clear, reusable helpers

- **YAGNI (You Aren't Gonna Need It)**: Implement only what the current requirements explicitly need

- Advanced features are only implemented if directly valuable for the user/app and can be done with minimal complexity

### **Research & Validation Protocol**

For every design, architectural, or implementation decision, leverage these tools in order of preference:

1. **`exa:deep_researcher_start`** (PREFERRED for all research)
   - Use for: in-depth research, competitive analysis, feature planning, best practices, industry patterns
   - Default choice for any research task requiring comprehensive understanding

2. **`context7:resolve-library-id` → `context7:get-library-docs`**
   - Use for: official library documentation, latest APIs, code examples
   - Essential when implementing new libraries or features

3. **`firecrawl:firecrawl_extract`**
   - Use for: structured content from specific docs, Stack Overflow answers, UI/UX code examples
   - Only when you need specific code or implementation details from known sources
   - **NEVER use for deep research** - that's what exa is for

4. **`tavily:tavily-search`**
   - Use for: recent trends, region-specific issues, breaking news about libraries
   - When you need current information from the last few months

5. **`clear-thought:sequentialthinking`, `clear-thought:decisionframework`**
   - Use for: decomposing complex tasks, architectural decisions, trade-off analysis
   - When you need structured reasoning and decision documentation

### **Modern Python & Code Quality Standards**

All code must meet these requirements:

- **Type hints**: All functions must have complete type annotations

- **Docstrings**: Google-style docstrings for all public methods and classes

- **Linting**: Pass `ruff check . --fix` and `ruff format .`

- **Line length**: 88 characters maximum

- **Imports**: Organized with `ruff check . --select=["E","F","I","UP"]`

- **Modularity**: Split into maintainable modules following project conventions

- **Error handling**: Graceful error handling with user feedback

- **Performance**: Use `@st.cache_data` for expensive operations

### **Database-Driven Task Tracking**

Use the project's SQLite task database (`orchestration/database/implementation_tracker.db`) for all batch/subagent work:

**Query Operations:**

- Get active tasks for current batch

- Check task dependencies and prerequisites

- Retrieve task details, success criteria, and file assignments

- Monitor batch completion status

**Update Operations:**

- Log task progress with timestamps

- Record completion status and validation results

- Document any blockers, errors, or dependencies discovered

- Update time estimates vs actual completion

- Mark tasks complete and trigger next batch advancement

**Database Schema:**

```sql

-- Key tables for task coordination
tasks(id, title, description, component_area, phase, priority, status, ...)
execution_groups(id, group_name, day_range, prerequisite_groups, ...)
group_tasks(group_id, task_id, parallel_batch, estimated_agent_hours, ...)
task_progress(id, task_id, progress_percentage, notes, updated_by, ...)
```

### **Parallel Agent Execution Protocol**

**CRITICAL RULE: Run all subagents in parallel, not sequentially!**

**Batch Organization:**

- **Independent tasks**: Launch all subagents in the batch simultaneously

- **Dependent tasks**: Group into batches so only those with all prerequisites met run together

- **Conflict prevention**: Ensure no two agents work on the same files within a batch

- **Completion coordination**: Wait for ALL tasks in batch to complete before advancing

**Execution Flow:**

1. Query database for next available batch
2. Validate all prerequisites are met
3. Generate agent prompts for each task in batch
4. Launch all Task tool agents simultaneously
5. Monitor progress and handle blockers
6. Validate completion criteria
7. Mark batch complete and advance to next batch

**File Safety Rules:**

- Each agent works on distinct files/directories

- No shared file modifications within a batch

- Component-based isolation (UI/, services/, database/, etc.)

- Clear file ownership assignment per task

## 🟢 Parallel AI Agent Batch Execution Prompts & Operating Manual

### 🟩 Core Directives & Guiding Principles

- **Mission:**  
  Complete all tasks in record time by dividing the full roadmap into groups ("batches") of truly independent tasks. Each group is executed in parallel by multiple AI subagents.  
  The app must be robust, reliable, beautiful, performant, and maintainable for daily power-user job search.

- **Library-First:**  
  Always use the most advanced, high-performance, and project-approved libraries and their built-in features. Avoid custom code unless there's a clear, justified gap.

- **KISS / DRY / YAGNI:**  
  Prioritize simplicity and maintainability. Advanced features are only implemented if directly valuable for the user/app and can be done with minimal complexity.

- **Research & Validation:**  
  For every design, architectural, or implementation decision, leverage:
  - `exa:deep_researcher_start` (preferred for all in-depth/competitive research, feature planning, best practices)
  - `firecrawl:firecrawl_extract` (for structured content from docs, open-source, and Stack Overflow when specific code or UI details are needed)
  - `tavily:tavily-search` (for recent trends and region-specific issues)
  - `context7:resolve-library-id` → `context7:get-library-docs` (for official docs, latest APIs, code examples)
  - `clear-thought:sequentialthinking`, `clear-thought:decisionframework` (for decomposing complex tasks and clear, justified trade-off reasoning)
  **Never use firecrawl deep research unless exa deep research is not a good fit.**

- **Modern Python & Code Quality:**  
  All code must be typed, use Google-style docstrings, pass `ruff` linting (line-length 88, select=["E","F","I","UP"]), and be split into maintainable modules per your project conventions.

- **Database-Driven Task Tracking:**  
  Use the project's SQLite task database for all batch/subagent work:
  - Query for active tasks, their batch, and dependency status.
  - Update the DB with task progress, completion timestamps, agent logs, and any blockers or errors.
  - Mark tasks as completed when done and trigger advancement to the next independent batch as soon as the prior batch is 100% complete.

- **Parallel Agent Execution:**  
  **Run all subagents in parallel, not sequentially!**
  - **Independent tasks:** launch all subagents in the batch at the same time.
  - **Dependent tasks:** group into batches so only those with all prerequisites met are run together.
  - **Parallel-group execution:** within each group, run all subagents simultaneously to maximize efficiency.
  - This ensures maximum speed and zero resource conflicts.

---

## 🟢 Example Prompts for Each Batch/Subagent

For each batch, use a prompt like the following (modify numbers/tasks as needed):

---

### **Batch X (e.g., Batch 1A): Parallel Execution Start**

> **Prompt for the orchestrator/lead agent:**

```text
"Launch Batch X with N parallel subagents.  
Query the SQLite task database for all active tasks in Batch X.  
Spin up N Task tool agents, assigning one agent per independent task.  
Each agent must:

- Work only on its assigned task and associated files/directories.

- Always leverage the most advanced, appropriate library and exa:deep_researcher_start for any research, planning, or code examples.

- Use firecrawl or tavily only as needed for documentation/code or recent trends, but default to exa for deep/competitive/industry research.

- Reference official docs via context7 whenever using a new library or API.

- Use clear-thought tools for decomposing, reasoning, and decision-making on complex tasks.

- Log all progress, blockers, code diffs, and completion to the task database.

- Mark the task as completed in the DB when finished.

- Wait for all other tasks in the batch to finish before proceeding to the next batch.

Continue launching batches in parallel in this manner until all tasks are complete."
```

---

### **Per-Subagent Task Prompt Example**

> **Prompt for an individual subagent (replace with the actual task from the DB):**

```text
"You are an expert AI Coding Agent. Your task:  

- Query the task database for your assigned task description, dependencies, and any related files or prior logs.

- Complete the task to project standards (typed code, docstrings, ruff linting, modularity, etc.).

- Always leverage exa:deep_researcher_start for feature research, competitive analysis, or best practice comparison.  

- Use firecrawl or tavily for structured documentation, UI/UX code, or recent trends when needed, but only if exa is not optimal.

- Use context7 to retrieve official library docs for any APIs or features you are implementing.

- Use clear-thought tools for task decomposition or any complex reasoning.

- Update your progress, logs, blockers, and completion in the task database.

- Mark the task as complete when done, and notify the orchestrator/lead agent.

- **TASK INSTRUCTIONS:**  
  [Paste the complete core prompt and specific instructions for the subagent's assigned task here.]
```

---

## 🟢 Task Monitoring, Completion, and Orchestration

- The orchestrator or lead agent should:
  - Monitor the status of all tasks and batches via continuous queries to the SQLite task database.
  - Automatically trigger the next parallel batch as soon as all tasks in the current batch are completed.
  - Ensure subagents never work on conflicting files or overlapping code—tasks must be independent within each batch.
  - Update and log any blockers, delays, or cross-batch dependencies in the database for audit and troubleshooting.
  - At sprint end, produce a final status and metrics report (time to complete, agent utilization, blockers encountered, etc.).

---

## 🟢 What to Include in the Planning File

- All batch prompts for each of your task groups (e.g., "Launch Batch 1A", "Launch Batch 2A", etc.), ready to be copy-pasted for orchestration.

- Subagent task prompts for any individual task, referencing DB-driven assignment.

- Clear tool usage instructions, decision frameworks, code quality rules, and task tracking/monitoring methods.

- A section for logging progress, issues, and sprint completion reporting.

---

**With this, you can copy, customize, and run agent prompts for any batch, with automated, database-driven tracking and maximum speed via parallel execution.**

## 📋 Detailed Batch Execution Plans

### **Day 1-2: Foundation & Independent Setup**

#### **Batch 1A: Environment & Dependencies (5 tasks, 1.2h total)**

**Orchestrator Prompt:**

```text
Launch Batch 1A with 5 parallel subagents for environment setup.
Query SQLite DB for tasks in execution_groups where group_name='1A'.
Launch Task tool agents for:
1. Install Modern Streamlit Dependencies
2. Create UI Architecture Directory Structure  
3. Set Up Modern CSS Theme System
4. Configure API Key Management
5. Create Development Environment Setup

Each agent works independently on separate configuration areas.
No file conflicts expected - each handles different setup domains.
```

**Sample Individual Task Prompt (for Task 1 - Dependencies):**

```text
PARALLEL EXECUTION - BATCH 1A - AGENT 1 of 5

You are Agent 1 working on "Install Modern Streamlit Dependencies" in parallel with 4 other agents.

TASK: Install and configure all modern Streamlit component libraries
WORKING DIRECTORY: /home/bjorn/repos/ai-job-scraper
OTHER AGENTS WORKING ON: Directory structure, CSS themes, API config, dev environment

CRITICAL RULES:

- Work ONLY on dependency installation and pyproject.toml

- Use exa:deep_researcher_start to research latest compatible versions

- Use context7 for official library documentation

- Do NOT modify other files agents are working on

- Complete within 15 minutes

- Signal completion with "TASK [ID] COMPLETE" 

- Update task database with progress and completion

TASK DETAILS:
Research and install these modern Streamlit libraries:

- streamlit-aggrid v1.1.7+ (advanced data grids)

- streamlit-elements (Material-UI components)

- streamlit-shadcn-ui (modern design system)

- streamlit-lottie (animations)

- plotly>=5.0.0 (interactive charts)

SUCCESS CRITERIA:

- All libraries installed and importable

- Version compatibility verified

- pyproject.toml updated with proper dependencies

- Test imports successful
```

#### **Batch 1B: Architecture & State (3 tasks, 0.8h total)**

**Prerequisites:** Batch 1A complete

**Focus:** Core architectural patterns and state management

### **Day 3-4: Core Implementation**

#### **Batch 2A: Database & Services (10 tasks, 2.5h total)**

**Focus:** Enhanced database models, services, and optimization

#### **Batch 2B: UI Pages & Components (3 tasks, 1.0h total)**  

**Focus:** Multi-page navigation and basic page structure

#### **Batch 2C: Progress & Background (7 tasks, 2.0h total)**

**Focus:** Real-time progress tracking and background task system

### **Day 5-6: Advanced Features**

#### **Batch 3A: Analytics & Reporting (8 tasks, 3.2h total)**

**Focus:** Plotly visualizations and dashboard analytics

#### **Batch 3B: Search & Filtering (11 tasks, 4.5h total)**

**Focus:** Advanced search capabilities and smart filtering

#### **Batch 3C: Applications & Workflow (5 tasks, 2.5h total)**

**Focus:** Application tracking and workflow management

### **Day 7: Integration & Polish**

#### **Batch 4A: Testing & Quality (15 tasks, 5.8h total)**

**Focus:** Comprehensive testing suite and validation

#### **Batch 4B: UI Polish & Documentation (12 tasks, 5.0h total)**

**Focus:** Final polish, documentation, and deployment preparation

## 🔧 Database Integration Commands

### **Query Current Batch:**

```python
from task_manager import TaskManager
tm = TaskManager()

# Get next batch to execute
current_batch = tm.get_next_execution_batch()
tasks_in_batch = tm.get_tasks_in_batch(current_batch['id'])
```

### **Update Task Progress:**

```python

# Mark task as in progress
tm.update_task_status(task_id, 'in_progress', 'Agent started work')

# Log completion
tm.update_task_status(task_id, 'completed', 'Task completed successfully')
tm.log_task_completion(task_id, agent_name='Claude-Agent-1')
```

### **Batch Completion Check:**

```python

# Check if all tasks in batch are complete
batch_complete = tm.is_batch_complete(batch_id)
if batch_complete:
    next_batch = tm.advance_to_next_batch()
```

## 📊 Progress Monitoring & Reporting

### **Real-time Status Queries:**

```python

# Get sprint overview
stats = tm.get_sprint_stats()

# Returns: total_tasks, completed_tasks, current_batch, estimated_completion

# Get agent utilization
agent_stats = tm.get_agent_utilization()

# Returns: active_agents, tasks_per_agent, completion_rates

# Get blockers and issues
blockers = tm.get_current_blockers()

# Returns: blocked_tasks, dependency_issues, agent_conflicts
```

### **Final Sprint Report:**

```python

# Generate completion report
final_report = tm.generate_sprint_report()

# Includes: total_time, tasks_completed, agent_efficiency, blockers_resolved
```

## ✅ Success Criteria & Validation

### **Per-Task Validation:**

- Code passes ruff linting and formatting

- All imports resolve correctly

- Type hints are complete and accurate

- Docstrings follow Google style

- Success criteria met per task definition

### **Per-Batch Validation:**

- All tasks in batch completed successfully

- No file conflicts or merge issues

- Integration tests pass for batch components

- Dependencies for next batch are satisfied

### **Sprint Completion Criteria:**

- All 171 tasks marked complete in database

- Full application runs without errors

- Performance targets met (sub-100ms search, 60fps animations)

- All features functional per UI requirements

- Documentation complete and deployment ready

## 🚀 Sprint Execution Commands

### **Initialize Sprint:**

```bash

# Prepare for parallel execution
python3 orchestration/parallel_orchestrator.py --initialize

# Validate task database and dependencies
python3 orchestration/task_manager.py --validate-sprint
```

### **Launch Batch Execution:**

```bash

# Start next available batch
python3 orchestration/parallel_orchestrator.py --execute-next-batch

# Monitor progress
python3 orchestration/task_manager.py --monitor

# Handle blockers
python3 orchestration/task_manager.py --resolve-blockers
```

### **Sprint Completion:**

```bash

# Validate all tasks complete
python3 orchestration/task_manager.py --validate-completion

# Generate final report
python3 orchestration/task_manager.py --sprint-report

# Deploy application
python3 deploy.py --production
```

---

**This comprehensive operating manual provides everything needed to execute the 1-week AI Job Scraper modernization sprint using parallel AI agent coordination, database-driven task tracking, and conflict-free batch execution.**

> **Ready to launch the most efficient development sprint ever executed! 🚀**
