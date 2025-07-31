#!/usr/bin/env python3
"""Quick script to check database contents."""

from database import SessionLocal
from models import JobSQL

session = SessionLocal()
try:
    total_jobs = session.query(JobSQL).count()
    print(f"Total jobs in database: {total_jobs}")

    if total_jobs > 0:
        print("\nRecent jobs:")
        for job in session.query(JobSQL).limit(10):
            print(f"- {job.title} at {job.company} (link: {job.link[:50]}...)")
    else:
        print("No jobs found in database.")

finally:
    session.close()
