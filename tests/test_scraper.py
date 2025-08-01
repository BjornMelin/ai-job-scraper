"""Tests for the AI Job Scraper functionality in src/scraper.py."""

import datetime

from unittest.mock import patch

import pytest

from sqlmodel import select

from src.models import CompanySQL, JobSQL
from src.scraper import scrape_all, update_db


@pytest.mark.asyncio
async def test_update_db_new_jobs(temp_db):
    """Test updating database with new jobs."""
    jobs = [
        JobSQL(
            company="New Co",
            title="AI Eng",
            description="AI role",
            link="https://new.co/job1",
            location="Remote",
            posted_date=datetime.datetime.now(),
            salary="$100k-150k",
        ),
    ]

    update_db(jobs)

    result = (await temp_db.exec(select(JobSQL))).all()
    assert len(result) == 1
    assert result[0].salary == (100000, 150000)


@pytest.mark.asyncio
async def test_update_db_upsert_and_delete(temp_db):
    """Test upsert and delete stale jobs."""
    # Add existing job
    existing = JobSQL(
        company="Exist Co",
        title="Old Title",
        description="Old desc",
        link="https://exist.co/job",
        location="Old Loc",
        posted_date=datetime.datetime.now() - datetime.timedelta(days=1),
        salary=(80000, 120000),
        favorite=True,  # User field to preserve
    )
    temp_db.add(existing)
    await temp_db.commit()

    # Stale job
    stale = JobSQL(
        company="Stale Co",
        title="Stale Job",
        description="To delete",
        link="https://stale.co/job",
        location="Stale",
        salary=(None, None),
    )
    temp_db.add(stale)
    await temp_db.commit()

    # New jobs data
    new_jobs = [
        JobSQL(
            company="Exist Co",
            title="Updated Title",
            description="Updated desc",
            link="https://exist.co/job",
            location="Updated Loc",
            posted_date=datetime.datetime.now(),
            salary="$90k-130k",
        ),
        JobSQL(
            company="New Co",
            title="New Job",
            description="New desc",
            link="https://new.co/job",
            location="New Loc",
            salary=(None, None),
        ),
    ]

    update_db(new_jobs)

    all_jobs = (await temp_db.exec(select(JobSQL))).all()
    assert len(all_jobs) == 2  # Stale deleted

    updated = next(j for j in all_jobs if j.link == "https://exist.co/job")
    assert updated.title == "Updated Title"  # Updated
    assert updated.salary == (90000, 130000)
    assert updated.favorite is True  # Preserved

    new_job = next(j for j in all_jobs if j.link == "https://new.co/job")
    assert new_job.title == "New Job"


@pytest.mark.asyncio
@patch("src.scraper.scrape_job_boards")
@patch("src.scraper.load_active_companies")
@patch("src.scraper.graph.invoke")
async def test_scrape_all_workflow(
    mock_graph_invoke, mock_load_companies, mock_scrape_boards, temp_db
):
    """Test full scrape_all workflow with mocks."""
    mock_load_companies.return_value = [
        CompanySQL(name="Mock Co", url="https://mock.co", active=True)
    ]

    mock_graph_invoke.return_value = {
        "normalized_jobs": [
            JobSQL(
                company="Mock Co",
                title="AI Engineer",
                description="AI role",
                link="https://mock.co/job1",
                location="Remote",
                salary=(None, None),
            )
        ]
    }

    mock_scrape_boards.return_value = [
        {
            "title": "ML Engineer",
            "company": "Board Co",
            "description": "ML role",
            "job_url": "https://board.co/job2",
            "location": "Office",
            "date_posted": datetime.datetime.now(),
            "min_amount": 100000,
            "max_amount": 150000,
        }
    ]

    scrape_all()

    all_jobs = (await temp_db.exec(select(JobSQL))).all()
    assert len(all_jobs) == 2
    assert any(j.title == "AI Engineer" for j in all_jobs)
    assert any(
        j.title == "ML Engineer" and j.salary == (100000, 150000) for j in all_jobs
    )


@pytest.mark.asyncio
@patch("src.scraper.scrape_job_boards")
@patch("src.scraper.load_active_companies")
@patch("src.scraper.graph.invoke")
async def test_scrape_all_filtering(
    mock_graph_invoke, mock_load_companies, mock_scrape_boards, temp_db
):
    """Test relevance filtering in scrape_all."""
    mock_load_companies.return_value = []
    mock_scrape_boards.return_value = [
        {
            "title": "AI Engineer",
            "company": "Co",
            "description": "Desc",
            "job_url": "url1",
            "location": "Loc",
            "date_posted": None,
            "min_amount": None,
            "max_amount": None,
        },
        {
            "title": "Sales Manager",
            "company": "Co",
            "description": "Desc",
            "job_url": "url2",
            "location": "Loc",
            "date_posted": None,
            "min_amount": None,
            "max_amount": None,
        },
    ]

    scrape_all()

    all_jobs = (await temp_db.exec(select(JobSQL))).all()
    assert len(all_jobs) == 1
    assert all_jobs[0].title == "AI Engineer"
