# ADR-013: Smart Database Synchronization Engine

## Title

Implementation of a Smart Synchronization Engine for Safe and Intelligent Database Updates

## Version/Date

1.0 / August 7, 2025

## Status

Accepted (Implemented)

## Context

The application requires a robust mechanism to update the local database with newly scraped job data without losing user-generated information (like favorites, notes, or application status). A simple "delete-and-insert" approach would be destructive and lead to a poor user experience. The system needs to intelligently handle new, updated, and stale job postings.

## Related Requirements

* `DB-SYNC-01`: Intelligent Job Synchronization

* `DB-SYNC-02`: Content-Based Change Detection

* `DB-SYNC-03`: User Data Preservation

* `DB-SYNC-04`: Soft Deletion (Archiving)

## Decision

We will implement a dedicated service class, `SmartSyncEngine`, responsible for all database write operations originating from the scraper. This engine will use a content-hashing strategy to detect changes and will follow specific rules to preserve user data.

### Synchronization Logic

1. **Content Hashing:** For each scraped job, a content hash (MD5) will be generated from its core, non-user-editable fields (title, description, location, company, etc.). This hash serves as a fingerprint for the job's content.
2. **New Jobs:** If a job's link does not exist in the database, it is inserted as a new record.
3. **Updated Jobs:** If a job's link exists, its new content hash is compared to the stored hash.
    * If the hashes are the same, the job is skipped, but its `last_seen` timestamp is updated.
    * If the hashes differ, the record is updated with the new scraped data, but **only non-user-editable fields are changed**. Fields like `favorite`, `notes`, and `application_status` are preserved.
4. **Stale Jobs:** After a scrape, the engine will identify jobs in the database that were not present in the latest scrape.
    * If a stale job has user data (`favorite` is true, `notes` are not empty, or `application_status` is not 'New'), it is "soft-deleted" by setting its `archived` flag to `True`.
    * If a stale job has no user data, it is permanently deleted from the database to save space.

## Design

The implementation resides in `src/services/database_sync.py`.

```python
class SmartSyncEngine:
    def sync_jobs(self, jobs: list[JobSQL]) -> dict[str, int]:
        # ...
    
    def _generate_content_hash(self, job: JobSQL) -> str:
        # ...
        
    def _update_existing_job(self, existing: JobSQL, new_job: JobSQL) -> str:
        # Preserves user data
        # ...

    def _handle_stale_jobs(self, session: Session, current_links: set[str]) -> dict[str, int]:
        # Implements archive/delete logic
        # ...
```

## Consequences

* **Positive:**
  * User data is protected from being overwritten by subsequent scrapes.
  * The database remains up-to-date with the latest job information.
  * The use of content hashing is efficient for detecting changes.
  * Archiving prevents the loss of application history for jobs that are no longer listed.

* **Negative:**
  * The synchronization logic is more complex than a simple overwrite.
  * The database will grow over time due to archived jobs (though a cleanup mechanism for very old archived jobs is possible).

* **Mitigations:** The logic is encapsulated within a single, well-tested service, making it maintainable. A separate, optional cleanup utility can be created to purge old archived jobs if database size becomes a concern.
