"""Database seeding script for initial company data.

This module populates the database with initial company information
including names and career page URLs for major AI companies.
"""

from database import SessionLocal
from models import CompanySQL

SITES = {
    "anthropic": "https://www.anthropic.com/jobs?team=4002061008&office=4001218008",
    "openai": "https://openai.com/careers/search/?c=ab2b9da4-24a4-47df-8bed-1ed5a39c7036%2C86a66e6f-8ddc-493d-b71f-2f6f6d2769a6%2Ce2a6a756-466b-4b91-be68-bb0c96102de1%2C6dd4a467-446d-4093-8d57-d4633a571123%2Cd36236ec-fb74-49bd-bd3f-9d8365e2e2cb%2C18ad45e4-3e90-44b9-abc6-60b2df26b03e%2C27c9a852-c401-450e-9480-d3b507b8f64a%2C0f06f916-a404-414f-813f-6ac7ff781c61%2C3345bedf-45ec-4ae1-ad44-b0affc79bcb5%2C0c0f1511-91d1-4317-a68a-52ec2f849450%2C224d99ae-26ec-4751-8af3-ed7d104b60a2%2Cfb2b77c5-5f20-4a93-a1c4-c3d640d88e04%2Cec712b2d-1b07-4d50-a27f-7d1153e0a5df%2Caee065f0-5fa5-437d-9506-67c186cbfbbd%2C8cb35b37-f31f-4167-84ca-ba789cf36142%2Cf32f653e-df5a-407a-ab39-901459f5f6c1%2C68998f96-ac93-45a9-aa44-dda4adf7a47b%2C795ae415-f19a-41c9-8acd-b1b8c08c4896%2C0df0672c-86c0-46ee-b3dd-3cf63adb5b08&l=bbd9f7fe-aae5-476a-9108-f25aea8f6cd2",
    "deepmind": "https://deepmind.google/about/careers/?location=mountain-view-california-us",
    "xai": "https://x.ai/careers/open-roles?dept=4024733007",
    "meta": "https://www.metacareers.com/jobs?teams[0]=Artificial%20Intelligence&teams[1]=Research&roles[0]=Full%20time%20employment&offices[0]=Menlo%20Park%2C%20CA",
    "microsoft": "https://jobs.careers.microsoft.com/global/en/search?d=Data%20Science&d=Applied%20Sciences&d=Research%20Sciences&exp=Experienced%20professionals&rt=Individual%20Contributor&et=Full-Time&et=Full-time&ws=Up%20to%20100%25%20work%20from%20home&ws=Up%20to%2050%25%20work%20from%20home&l=en_us&pg=1&pgSz=20&o=Relevance&flt=true",
    "nvidia": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite?q=Agent%20OR%20AI%20OR%20AI%20Engineer%20OR%20Machine%20Learning%20OR%20Deep%20Learning%20OR%20CUDA&timeType=5509c0b5959810ac0029943377d47364&workerSubType=0c40f6bd1d8f10adf6dae161b1844a15&locationHierarchy1=2fcb99c455831013ea52fb338f2932d8&jobFamilyGroup=0c40f6bd1d8f10ae43ffc8817cf47e8e&jobFamilyGroup=0c40f6bd1d8f10ae43ffaefd46dc7e78&jobFamilyGroup=0c40f6bd1d8f10ae43ffbd1459047e84",
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
            company = session.query(CompanySQL).filter_by(name=name).first()
            if company:
                # Update existing company URL if it has changed
                if company.url != url:
                    print(f"Updating {name} URL: {company.url} -> {url}")
                    company.url = url
            else:
                # Add new company
                print(f"Adding new company: {name}")
                session.add(CompanySQL(name=name, url=url, active=True))
        session.commit()
        print("✅ Company database updated successfully.")
    except Exception as e:
        session.rollback()
        print(f"❌ Seed failed: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_companies()
