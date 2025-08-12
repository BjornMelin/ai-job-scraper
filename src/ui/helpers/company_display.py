"""Company display UI helpers.

Helper functions for rendering company information and statistics.
"""

import streamlit as st

from src.schemas import Company


def render_company_info(company: Company) -> None:
    """Render company name and URL."""
    st.markdown(f"**{company.name}**")
    st.markdown(f"🔗 [{company.url}]({company.url})")


def render_company_statistics(company: Company) -> None:
    """Render company scraping statistics and last scraped date."""
    # Display last scraped date
    if company.last_scraped:
        last_scraped_str = company.last_scraped.strftime("%Y-%m-%d %H:%M")
        st.markdown(f"📅 Last scraped: {last_scraped_str}")
    else:
        st.markdown("📅 Never scraped")

    # Display scraping statistics
    if company.scrape_count > 0:
        success_rate = f"{company.success_rate:.1%}"
        scrape_text = f"📊 Scrapes: {company.scrape_count} | Success: {success_rate}"
        st.markdown(scrape_text)
    else:
        st.markdown("📊 No scraping history")


def render_company_toggle(company: Company, toggle_callback) -> None:
    """Render company active toggle with callback."""
    st.toggle(
        "Active",
        value=company.active,
        key=f"company_active_{company.id}",
        help=f"Toggle scraping for {company.name}",
        on_change=toggle_callback,
        args=(company.id,),
    )


def render_company_card(company: Company, toggle_callback) -> None:
    """Render a complete company card with info, stats, and toggle."""
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            render_company_info(company)

        with col2:
            render_company_statistics(company)

        with col3:
            render_company_toggle(company, toggle_callback)


def render_company_card_with_delete(
    company: Company, toggle_callback, delete_callback
) -> None:
    """Render a company card with info, stats, toggle, and delete button."""
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

        with col1:
            render_company_info(company)

        with col2:
            render_company_statistics(company)

        with col3:
            render_company_toggle(company, toggle_callback)

        with col4:
            # Create a unique key for the confirmation checkbox
            confirm_key = f"delete_confirm_{company.id}"

            # Show confirmation checkbox if delete button was clicked
            if st.session_state.get(f"show_delete_confirm_{company.id}", False):
                st.checkbox(
                    "Confirm?",
                    key=confirm_key,
                    help="Check to confirm deletion",
                    on_change=delete_callback,
                    args=(company.id,),
                )
                # Add cancel button
                if st.button("Cancel", key=f"cancel_delete_{company.id}"):
                    st.session_state[f"show_delete_confirm_{company.id}"] = False
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            # Show delete button
            elif st.button(
                "🗑️ Delete",
                key=f"delete_btn_{company.id}",
                help=f"Delete {company.name} and all associated jobs",
                type="secondary",
            ):
                st.session_state[f"show_delete_confirm_{company.id}"] = True
                st.rerun()
