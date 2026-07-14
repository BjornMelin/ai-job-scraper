"""Saved-search definitions and manual runs."""

from __future__ import annotations

import asyncio
import logging

import streamlit as st
from pydantic import ValidationError

from src.models.job_models import JobSite, JobType, SavedSearchRunStatus
from src.schemas import SavedSearch, SavedSearchCreate, SavedSearchUpdate
from src.scraping.scrape_all import SavedSearchRunInProgressError, run_saved_search
from src.services.saved_search_service import saved_search_service
from src.ui.design import empty_state, page_intro, relative_time, sentence_case

logger = logging.getLogger(__name__)
JOB_TYPES = (None, *JobType)
CREATE_SEARCH_OPEN_KEY = "create-search-open"
CREATE_SEARCH_RESET_KEY = "create-search-reset"
CREATE_SEARCH_ERROR_KEY = "create-search-error"
CREATE_SEARCH_WIDGET_KEYS = (
    "create-search-name",
    "create-search-query",
    "create-search-location",
    "create-search-sites",
    "create-search-remote-only",
    "create-search-job-type",
    "create-search-results-limit",
)


def _job_type_label(value: JobType | None) -> str:
    return "Any employment type" if value is None else sentence_case(value.value)


def _validation_message(error: ValidationError) -> str:
    labels = {
        "name": "Name",
        "query": "keywords",
        "location": "location",
        "sites": "at least one job board",
    }
    invalid = {str(item["loc"][0]) for item in error.errors() if item["loc"]}
    fields = [label for field, label in labels.items() if field in invalid]
    if len(fields) == 1:
        return f"{fields[0].capitalize()} is required."
    return f"{', '.join(fields[:-1])} and {fields[-1]} are required."


def _create_search() -> None:
    if st.session_state.pop(CREATE_SEARCH_RESET_KEY, False):
        for key in CREATE_SEARCH_WIDGET_KEYS:
            st.session_state.pop(key, None)

    with st.expander(
        "New saved search",
        expanded=st.session_state.get(CREATE_SEARCH_OPEN_KEY, False),
    ):
        with st.form("create-search"):
            name = st.text_input(
                "Name",
                placeholder="Remote ML roles",
                key="create-search-name",
            )
            query = st.text_input(
                "Keywords",
                placeholder="machine learning engineer",
                key="create-search-query",
            )
            location = st.text_input(
                "Location",
                value="United States",
                key="create-search-location",
            )
            sites = st.multiselect(
                "Job boards",
                list(JobSite),
                default=[JobSite.LINKEDIN, JobSite.INDEED],
                format_func=lambda value: sentence_case(value.value),
                key="create-search-sites",
            )
            remote_only = st.checkbox(
                "Remote only",
                key="create-search-remote-only",
            )
            job_type = st.selectbox(
                "Employment type",
                JOB_TYPES,
                format_func=_job_type_label,
                key="create-search-job-type",
            )
            results_limit = st.number_input(
                "Results per run",
                min_value=1,
                max_value=1000,
                value=50,
                key="create-search-results-limit",
            )
            submitted = st.form_submit_button(
                "Create saved search",
                type="primary",
            )

        if error_message := st.session_state.get(CREATE_SEARCH_ERROR_KEY):
            st.error(error_message)

        if submitted:
            try:
                saved_search_service.create(
                    SavedSearchCreate(
                        name=name,
                        query=query,
                        location=location,
                        sites=sites,
                        remote_only=remote_only,
                        job_type=job_type,
                        results_limit=results_limit,
                    )
                )
            except ValidationError as error:
                st.session_state[CREATE_SEARCH_OPEN_KEY] = True
                st.session_state[CREATE_SEARCH_ERROR_KEY] = _validation_message(error)
                st.rerun()
            except Exception:
                logger.exception("Could not create saved search")
                st.session_state[CREATE_SEARCH_OPEN_KEY] = True
                st.session_state[CREATE_SEARCH_ERROR_KEY] = (
                    "The saved search could not be created. "
                    "Check the fields and use a unique name."
                )
                st.rerun()
            else:
                st.session_state.pop(CREATE_SEARCH_ERROR_KEY, None)
                st.session_state[CREATE_SEARCH_OPEN_KEY] = False
                st.session_state[CREATE_SEARCH_RESET_KEY] = True
                st.session_state["searches-notice"] = "Saved search created."
                st.rerun()


def _run(search: SavedSearch) -> SavedSearch | None:
    with st.status(f"Running {search.name}", expanded=True) as status:
        st.write("Collecting matching jobs from the selected boards.")
        completed = asyncio.run(run_saved_search(search.id))
        if completed is None:
            status.update(label="Saved search no longer exists", state="error")
            return None
        if completed.last_run_status is SavedSearchRunStatus.SUCCEEDED:
            status.update(label=f"{search.name} finished", state="complete")
        elif completed.last_run_status is SavedSearchRunStatus.PARTIAL:
            status.update(
                label=f"{search.name} finished with skipped rows",
                state="complete",
            )
        else:
            status.update(label=f"{search.name} failed", state="error")
        return completed


def _edit_search(search: SavedSearch) -> None:
    with st.expander("Edit search"):
        with st.form(f"edit-search-{search.id}"):
            name = st.text_input("Name", value=search.name)
            query = st.text_input("Keywords", value=search.query)
            location = st.text_input("Location", value=search.location)
            sites = st.multiselect(
                "Job boards",
                list(JobSite),
                default=search.sites,
                format_func=lambda value: sentence_case(value.value),
            )
            remote_only = st.checkbox("Remote only", value=search.remote_only)
            job_type = st.selectbox(
                "Employment type",
                JOB_TYPES,
                index=JOB_TYPES.index(search.job_type),
                format_func=_job_type_label,
            )
            results_limit = st.number_input(
                "Results per run",
                min_value=1,
                max_value=1000,
                value=search.results_limit,
            )
            if st.form_submit_button("Save search"):
                try:
                    updated = saved_search_service.update(
                        search.id,
                        SavedSearchUpdate(
                            name=name,
                            query=query,
                            location=location,
                            sites=sites,
                            remote_only=remote_only,
                            job_type=job_type,
                            results_limit=results_limit,
                        ),
                    )
                except Exception:
                    logger.exception("Could not update saved search %s", search.id)
                    st.error(
                        "The saved search could not be updated. "
                        "Check the fields and use a unique name."
                    )
                else:
                    if updated is None:
                        st.error("The saved search no longer exists.")
                    else:
                        st.session_state["searches-notice"] = "Saved search updated."
                        st.rerun()

        with st.expander("Delete search"):
            st.write("This removes the search definition. Collected jobs stay in Jobs.")
            if st.button("Delete permanently", key=f"delete-search-{search.id}"):
                try:
                    deleted = saved_search_service.delete(search.id)
                except Exception:
                    logger.exception("Could not delete saved search %s", search.id)
                    st.error("The saved search could not be deleted. Try again.")
                else:
                    if not deleted:
                        st.error("The saved search no longer exists.")
                    else:
                        st.session_state["searches-notice"] = "Saved search deleted."
                        st.rerun()


def _render_search(search: SavedSearch) -> None:
    with st.container(border=True, key=f"search-card-{search.id}"):
        displayed_search = search
        st.subheader(search.name, anchor=False)
        scope = f"{search.query} · {search.location}"
        if search.remote_only:
            scope += " · remote only"
        st.markdown(scope)
        st.caption(
            f"{', '.join(sentence_case(site.value) for site in search.sites)} · "
            f"up to {search.results_limit} results"
        )

        enabled = st.checkbox(
            "Enabled", value=search.enabled, key=f"enabled-{search.id}"
        )
        if enabled != search.enabled:
            try:
                saved_search_service.update(
                    search.id,
                    SavedSearchUpdate(enabled=enabled),
                )
            except Exception:
                logger.exception("Could not change saved search %s", search.id)
                st.error("The saved search could not be changed. Try again.")
                enabled = search.enabled

        if st.button(
            "Run now",
            key=f"run-search-{search.id}",
            type="primary",
            disabled=not enabled,
            icon=":material/play_arrow:",
        ):
            try:
                completed = _run(search)
            except SavedSearchRunInProgressError:
                st.warning("This saved search is already running in another session.")
            except Exception:
                logger.exception("Could not run saved search %s", search.id)
                st.error("The search could not be run. Review the logs and try again.")
            else:
                if (
                    completed
                    and completed.last_run_status is SavedSearchRunStatus.SUCCEEDED
                ):
                    displayed_search = completed
                    st.success(
                        f"Found {completed.jobs_seen} jobs; "
                        f"{completed.jobs_new} were new."
                    )
                elif completed:
                    displayed_search = completed

        status = sentence_case(displayed_search.last_run_status.value)
        st.markdown(
            f"**Last run:** {status} · {relative_time(displayed_search.last_run_at)}"
        )
        if displayed_search.duration_ms is not None:
            st.caption(
                f"{displayed_search.jobs_seen} seen · "
                f"{displayed_search.jobs_new} new · "
                f"{displayed_search.duration_ms / 1000:.1f}s"
            )
        if (
            displayed_search.last_run_status is SavedSearchRunStatus.PARTIAL
            and displayed_search.last_error
        ):
            st.warning(displayed_search.last_error)
        elif displayed_search.last_error:
            st.error(displayed_search.last_error)
        elif displayed_search.last_run_status is SavedSearchRunStatus.FAILED:
            st.error("The search did not complete.")
        _edit_search(search)


def render_searches_page() -> None:
    """Render the only supported job-collection configuration."""
    page_intro(
        "Collection",
        "Saved searches",
        "Define repeatable searches, then run each one when you want fresh results.",
    )
    if notice := st.session_state.pop("searches-notice", None):
        st.success(notice)
    _create_search()

    try:
        searches = saved_search_service.list()
    except Exception:
        logger.exception("Could not load saved searches")
        st.error("Saved searches could not be loaded. Check the database and retry.")
        return

    if not searches:
        empty_state(
            "Create your first search",
            "Add a focused role and location above. Runs happen only when you start them.",
        )
        return

    for search in searches:
        _render_search(search)


render_searches_page()
