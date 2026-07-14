# Use Job Tracker

**Content type:** Reference

Job Tracker organizes job-search work across three pages. This reference explains every page and the persisted workflow.

## Review jobs

The **Jobs** page filters collected jobs by workflow stage. Each new job starts in **Inbox**.

The stage filter uses five canonical values:

- **Inbox**: jobs that still need an initial decision
- **Saved**: jobs worth returning to before applying
- **Applied**: submitted applications
- **Interviews**: active interview processes
- **Closed**: rejected, withdrawn, declined, or completed opportunities

Use **Search jobs** to match role titles, descriptions, companies, or locations. Use **Companies** to narrow the view to derived company facets. Select **Starred only** to show your marked jobs.

Open **Review and update** on a job to change its stage, starred marker, or notes. Select **Save changes** to persist all three fields in one transaction.

The original posting opens in a new browser destination. Job Tracker keeps the stored job if the external posting later disappears.

## Manage saved searches

The **Searches** page owns every collection configuration. Companies are never collection inputs.

A saved search contains:

- Name and keyword query
- Location
- One or more job boards
- Remote-only preference
- Employment type
- Results limit
- Enabled state

Select **Run now** to collect jobs for one saved search. Job Tracker does not run searches on a timer.

Each card displays the latest run state:

- `never`: the search has not run
- `running`: provider work is active
- `succeeded`: the provider returned successfully, including zero-result runs
- `partial`: valid jobs were stored, but one or more provider rows failed validation
- `failed`: the provider or persistence step failed
- `cancelled`: the run was cancelled

The card also records provider rows seen, new jobs, duration, completion time, and the latest error or validation warning. Disabling or deleting a saved search does not delete collected jobs.

## Read insights

The **Insights** page calculates read-only summaries from the same SQLite database as **Jobs**.

The page includes:

- Total jobs, companies, and saved searches
- Counts for every workflow stage
- Listing activity over the past 30 days
- Salary coverage and averages over the past 90 days
- Company facets with active-job counts and latest listing dates

Company facets appear only after a collected job references that company. You cannot add, edit, enable, or delete companies in the interface.

## Understand saved job fields

A repeated collection run can update provider-owned job details, including title, description, location, posted date, and salary. It preserves your workflow stage, starred marker, notes, and archive state.

The job URL identifies a repeated posting. A new URL creates a new job even when the title and company match an existing record.
