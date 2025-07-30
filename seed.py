"""Database seeding script for initial company data.

This module populates the database with initial company information
including names and career page URLs for major AI companies.
"""

from database import SessionLocal
from models import CompanySQL

SITES = {
    "anthropic": "https://careers.anthropic.com/jobs",
    "openai": "https://openai.com/careers",
    "deepmind": "https://deepmind.google/careers/jobs",
    "xai": "https://x.ai/careers",
    "meta": "https://ai.meta.com/careers",
    "microsoft": "https://jobs.careers.microsoft.com/global/en/search?l=en_us&pg=1&pgSz=20&o=Relevance&flt=WorkplaceTypes_2&flt=LocationCountry_1&flt=JobCategory_Artificial%20Intelligence_15",
    "nvidia": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
}


def seed_companies() -> None:
    """Seed the database with initial company data.

    Adds major AI companies and their career page URLs to the database.
    Only adds companies that don't already exist to avoid duplicates.
    All seeded companies are marked as active by default.

    Note:
        Safe to run multiple times - existing companies are not duplicated.
        Uses database transactions with rollback on error.

    """
    session = SessionLocal()
    try:
        for name, url in SITES.items():
            if not session.query(CompanySQL).filter_by(name=name).first():
                session.add(CompanySQL(name=name, url=url, active=True))
        session.commit()
        print("Seeded companies.")
    except Exception as e:
        session.rollback()
        print(f"Seed failed: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_companies()
