"""Companies management page for the AI Job Scraper application.

This module provides the Streamlit UI for managing company records, including
adding new companies and toggling their active status for scraping.
"""

import logging

import streamlit as st

from src.services.company_service import CompanyService

logger = logging.getLogger(__name__)


def show_companies_page() -> None:
    """Display the companies management page.

    Provides functionality to:
    - Add new companies with name and URL
    - View all companies in a organized list
    - Toggle active status for each company using toggles
    """
    st.title("Company Management")
    st.markdown("Manage companies for job scraping")

    # Add new company section
    with st.expander("➕ Add New Company", expanded=False), st.form("add_company_form"):
        st.markdown("### Add a New Company")

        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input(
                "Company Name",
                placeholder="e.g., TechCorp",
                help="Enter the company name (must be unique)",
            )

        with col2:
            company_url = st.text_input(
                "Careers URL",
                placeholder="e.g., https://techcorp.com/careers",
                help="Enter the company's careers page URL",
            )

        submit_button = st.form_submit_button("Add Company", type="primary")

        if submit_button:
            if not company_name or not company_name.strip():
                st.error("❌ Company name is required")
            elif not company_url or not company_url.strip():
                st.error("❌ Company URL is required")
            else:
                try:
                    company = CompanyService.add_company(
                        name=company_name.strip(), url=company_url.strip()
                    )
                    st.success(f"✅ Successfully added company: {company.name}")
                    logger.info(f"User added new company: {company.name}")
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                    logger.warning(f"Failed to add company due to validation: {e}")
                except Exception as e:
                    st.error("❌ Failed to add company. Please try again.")
                    logger.error(f"Failed to add company: {e}", exc_info=True)

    # Display all companies
    st.markdown("### Companies")

    try:
        companies = CompanyService.get_all_companies()

        if not companies:
            st.info("📝 No companies found. Add your first company above!")
            return

        # Display companies in a clean grid layout
        for company in companies:
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.markdown(f"**{company.name}**")
                    st.markdown(f"🔗 [{company.url}]({company.url})")

                with col2:
                    # Display company statistics
                    if company.last_scraped:
                        last_scraped_str = company.last_scraped.strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        st.markdown(f"📅 Last scraped: {last_scraped_str}")
                    else:
                        st.markdown("📅 Never scraped")

                    if company.scrape_count > 0:
                        success_rate = f"{company.success_rate:.1%}"
                        scrape_text = (
                            f"📊 Scrapes: {company.scrape_count} | "
                            f"Success: {success_rate}"
                        )
                        st.markdown(scrape_text)
                    else:
                        st.markdown("📊 No scraping history")

                with col3:
                    # Active toggle - this is the key requirement UI-COMP-02
                    active_status = st.toggle(
                        "Active",
                        value=company.active,
                        key=f"company_active_{company.id}",
                        help=f"Toggle scraping for {company.name}",
                    )

                    # Handle toggle change
                    if active_status != company.active:
                        try:
                            new_status = CompanyService.toggle_company_active(
                                company.id
                            )
                            if new_status:
                                st.success(f"✅ Enabled scraping for {company.name}")
                            else:
                                st.info(f"⏸️ Disabled scraping for {company.name}")
                            logger.info(
                                f"User toggled {company.name} active status to "
                                f"{new_status}"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to update {company.name} status")
                            logger.error(
                                f"Failed to toggle company status: {e}", exc_info=True
                            )

    except Exception as e:
        st.error("❌ Failed to load companies. Please refresh the page.")
        logger.error(f"Failed to load companies: {e}", exc_info=True)

    # Show summary statistics
    try:
        active_companies = CompanyService.get_active_companies()
        total_companies = len(companies)
        active_count = len(active_companies)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Companies", total_companies)

        with col2:
            st.metric("Active Companies", active_count)

        with col3:
            inactive_count = total_companies - active_count
            st.metric("Inactive Companies", inactive_count)

    except Exception as e:
        logger.error(f"Failed to load company statistics: {e}", exc_info=True)


if __name__ == "__main__":
    # For testing the page standalone
    show_companies_page()
