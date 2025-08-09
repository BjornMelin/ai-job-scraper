# AI Job Scraper - Detailed Implementation Plan (V1.1)

## Introduction

This plan details the tasks for the **V1.1 "Power User" Upgrade**. It assumes a fully functional V1.0 application is complete and serves as the baseline. These tasks are designed to be implemented sequentially to add significant value for analytics, filtering, and user experience.

---

## 🔮 V1.1: The "Power User" Upgrade

**Goal**: To enhance the application with features that provide deeper insights and greater control, transforming it from a simple tracker into a powerful analysis tool.

### **T2.1: Implement Analytics & Insights Dashboard**

- **Release**: V1.1

- **Priority**: **High**

- **Status**: **PENDING**

- **Prerequisites**: A functional V1.0 with a populated database.

- **Related Requirements**: `UI-ANALYTICS-01`

- **Libraries**: `plotly==5.18.0`, `pandas==2.3.1`

- **Description**: This task involves creating a new service to process analytics data and building a new dashboard page with interactive Plotly charts to visualize job trends and application funnels.

- **Architecture Diagram**:

  ```mermaid
  graph TD
      subgraph "UI Layer"
          A[Dashboard Page] -- Renders --> B[Plotly Charts];
      end
      subgraph "Service Layer"
          C[Analytics Service] -- Queries --> D[Database];
      end
      subgraph "Data Layer"
          D;
      end
      A -- Requests Data --> C;
  ```

- **Sub-tasks & Instructions**:
  - **T2.1.1: Create Analytics Service**:
    - **Instructions**: Create a new file `src/services/analytics_service.py`. Inside, create an `AnalyticsService` class.
    - **Instructions**: Implement a method `get_job_trends(days: int = 30)`. This method should query the `JobSQL` table, group jobs by their `posted_date` (day-level precision), and return a Pandas DataFrame with two columns: `date` and `job_count`.
    - **Instructions**: Implement a method `get_application_funnel()`. This method should query the `JobSQL` table, group by `application_status`, count the jobs in each status, and return a DataFrame with `status` and `count`.
    - **Success Criteria**: Both service methods return correctly structured DataFrames when called.
  - **T2.1.2: Build the Dashboard UI with Plotly**:
    - **Instructions**: In `src/ui/pages/dashboard.py`, replace the V1.0 `st.metric` placeholders.
    - **Instructions**: Call `AnalyticsService.get_job_trends()` and pass the resulting DataFrame to `plotly.express.line()` to create a line chart. Render it using `st.plotly_chart(fig, use_container_width=True)`.
    - **Instructions**: Call `AnalyticsService.get_application_funnel()` and pass the DataFrame to `plotly.express.funnel()` to create a funnel chart. Render it using `st.plotly_chart()`.
    - **Instructions**: Add titles and labels to the charts to make them easily understandable.
    - **Success Criteria**: The dashboard page displays two interactive charts: a line chart showing job posting volume over time and a funnel chart showing the distribution of application statuses. The charts are responsive and update when the underlying data changes.

### **T2.2: Implement Advanced Job Filtering**

- **Release**: V1.1

- **Priority**: **High**

- **Status**: **PENDING**

- **Prerequisites**: `T1.5` (Core Job Browser), `T2.1` (for data to analyze)

- **Related Requirements**: `UI-JOBS-05`

- **Libraries**: `streamlit==1.47.1`

- **Description**: This task enhances the existing job filter panel by adding new, more powerful controls for filtering by salary range and the date a job was posted.

- **Sub-tasks & Instructions**:
  - **T2.2.1: Update Job Service for Advanced Queries**:
    - **Instructions**: Open `src/services/job_service.py` and modify the `get_filtered_jobs` method signature to accept new optional parameters: `salary_min: int | None = None`, `salary_max: int | None = None`, and `posted_after: datetime | None = None`.
    - **Instructions**: In the method body, add new filtering logic to the SQLAlchemy query. For salary, use `func.json_extract(JobSQL.salary, "$[0]") >= salary_min` and `func.json_extract(JobSQL.salary, "$[1]") <= salary_max`. Note that `salary` is a JSON column.
    - **Instructions**: For the date filter, add `JobSQL.posted_date >= posted_after` to the query.
    - **Success Criteria**: The `get_filtered_jobs` method can now correctly filter the database based on salary and date inputs.
  - **T2.2.2: Update Filter UI in Sidebar**:
    - **Instructions**: In `src/ui/pages/jobs.py`, locate the filter section (likely in `st.sidebar`).
    - **Instructions**: Add a `st.slider("Salary Range ($k)", ...)` widget. This slider should have two thumbs to select a min/max range.
    - **Instructions**: Add a `st.date_input("Posted After")` widget to allow users to select a minimum posting date.
    - **Instructions**: Ensure the values from these new widgets are passed to the `JobService.get_filtered_jobs` call.
    - **Success Criteria**: The sidebar now contains sliders for salary and a date input for posting date. Modifying these controls correctly filters the job grid in real-time.

### **T2.3: Upgrade to Modal Job Details View**

- **Release**: V1.1

- **Priority**: **Medium**

- **Status**: **✅ COMPLETED** (PR #26)

- **Prerequisites**: `T1.5` (Core Job Browser)

- **Related Requirements**: `UI-JOBS-06`, `NFR-UX-01`

- **Libraries**: `streamlit-elements==0.1.0`

- **Description**: Replace the V1.0 `st.expander` job detail view with a professional, non-blocking modal overlay. This significantly improves the user experience by keeping the user in the context of the job grid.

- **Architecture Diagram**:

  ```mermaid
  graph TD
      A[Job Card "View Details" Click] --> B{Update Session State};
      B -- "st.session_state.modal_job_id = job.id" --> C[Trigger UI Rerun];
      C --> D[Jobs Page Renders];
      D -- "Checks session state" --> E[Job Detail Modal Component];
      E -- "Is open=True" --> F[Render Modal Overlay];
  ```

- **Sub-tasks & Instructions**:
  - **T2.3.1: Create the Job Detail Modal Component**: ✅ **COMPLETED**
    - **Instructions**: Create a new file `src/ui/components/modals/job_detail.py`.
    - **Instructions**: Inside, create a `JobDetailModal` class. Use `streamlit_elements` to build the modal. The core will be `with mui.Modal(open=is_open, onClose=handle_close):`. The `is_open` variable will be controlled by `st.session_state`.
    - **Instructions**: The content of the modal should be the same as the old expander: full description, notes text area, and application status controls.
    - **Success Criteria**: A reusable modal component is created that can display job details when activated.
  - **T2.3.2: Integrate Modal into the Jobs Page**: ✅ **COMPLETED**
    - **Instructions**: In `src/ui/pages/jobs.py`, instantiate the `JobDetailModal` at the top of the render function.
    - **Instructions**: Modify the "View Details" button on the `job_card`. Its `on_click` callback should now set `st.session_state.modal_job_id = job.id`.
    - **Instructions**: At the end of the `jobs.py` render function, call the modal's render method, passing it the job data corresponding to the ID in the session state.
    - **Instructions**: **Remove the old `st.expander` logic completely.**
    - **Success Criteria**: Clicking "View Details" on a job card now opens a modal overlay displaying that job's information. The modal can be closed, returning the user to the job grid.
