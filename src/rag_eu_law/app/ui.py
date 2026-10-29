"""Streamlit interface. Run with `uv run streamlit run src/rag_eu_law/app/ui.py`."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from rag_eu_law.app.auth import AuthenticatedUser, Authenticator
from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.app.history import HistoryService, RecordedAnswer
from rag_eu_law.domain.constants import (
    DEFAULT_USER,
    HISTORY_PAGE_SIZE,
    RATING_DOWN,
    RATING_UP,
)
from rag_eu_law.domain.exceptions import AuthenticationError, RagError
from rag_eu_law.domain.models import Regulation, RunRecord, SearchFilter

ALL_REGULATIONS = "All"
LAST_RUN_KEY = "last_run"
RATED_KEY = "rated_run_id"
RATING_LABELS = {RATING_UP: "👍", RATING_DOWN: "👎", None: "-"}


AUTH_USER_KEY = "auth_user"


@st.cache_resource
def load_container() -> Container:
    """Built once per server process, not on every Streamlit rerun."""
    return Container(Settings())


def configure_page() -> None:
    """Must be the first Streamlit call of a run, so `main` does it before building anything."""
    st.set_page_config(page_title="EU law Q&A", layout="wide")


def main(container_factory: Callable[[], Container]) -> None:
    """Entry point: sign in first when the Supabase store is configured.

    The factory runs after the page is configured, so it may show a spinner while it works.
    """
    configure_page()
    container = container_factory()
    authenticator = container.authenticator
    if authenticator is None:
        render(container.history_service)
        return
    user: AuthenticatedUser | None = st.session_state.get(AUTH_USER_KEY)
    if user is None:
        _login_form(authenticator)
    else:
        render(container.history_service_for(user.run_store), user)


def _login_form(authenticator: Authenticator) -> None:
    st.title("Sign in")
    with st.form("login"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            st.session_state[AUTH_USER_KEY] = authenticator.sign_in(email, password)
        except AuthenticationError as error:
            st.error(str(error))
        else:
            st.rerun()


def _sign_out(user: AuthenticatedUser) -> None:
    user.sign_out()
    st.session_state.clear()
    st.rerun()


def render(service: HistoryService, signed_in: AuthenticatedUser | None = None) -> None:
    st.title("GDPR and EU AI Act: cited answers")
    user, search_filter = _sidebar(service, signed_in)
    ask_tab, history_tab = st.tabs(["Ask", "History"])
    with ask_tab:
        _ask_tab(service, user, search_filter)
    with history_tab:
        _history_tab(service)


def _sidebar(
    service: HistoryService, signed_in: AuthenticatedUser | None
) -> tuple[str, SearchFilter | None]:
    with st.sidebar:
        st.header("Options")
        if signed_in is None:
            user = st.text_input("Your name", value=DEFAULT_USER, key="user")
        else:
            user = signed_in.email
            st.caption(f"Signed in as {user}")
            if st.button("Sign out", key="sign_out"):
                _sign_out(signed_in)
        choice = st.selectbox(
            "Regulation", [ALL_REGULATIONS, *(r.value for r in Regulation)], key="regulation"
        )
        article = st.text_input("Only article number (optional)", key="article").strip()
        st.caption(f"Model: {service.model}")
        st.caption(f"Index version: {service.index_version}")
    regulation = None if choice == ALL_REGULATIONS else Regulation(choice)
    search_filter = SearchFilter(regulation=regulation, article=article or None)
    return user.strip() or DEFAULT_USER, search_filter if search_filter.constraints() else None


def _ask_tab(service: HistoryService, user: str, search_filter: SearchFilter | None) -> None:
    question = st.text_area("Your question", key="question", height=100)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        _run_question(service, question.strip(), user, search_filter)
    last: RecordedAnswer | None = st.session_state.get(LAST_RUN_KEY)
    if last is not None:
        _show_answer(service, last)


def _run_question(
    service: HistoryService, question: str, user: str, search_filter: SearchFilter | None
) -> None:
    try:
        with st.spinner("Searching the texts and writing an answer..."):
            st.session_state[LAST_RUN_KEY] = service.ask(question, user, search_filter)
    except RagError as error:
        st.session_state.pop(LAST_RUN_KEY, None)
        st.error(str(error))


def _show_answer(service: HistoryService, recorded: RecordedAnswer) -> None:
    answer = recorded.answer
    if recorded.record.refused:
        st.warning(answer.answer)
    else:
        st.markdown(answer.answer)
    if answer.citations:
        st.markdown("**Citations**")
        st.markdown("\n".join(f"- `{citation.format()}`" for citation in answer.citations))
    with st.expander(f"Retrieved sources ({len(answer.retrieved_chunks)})"):
        for item in answer.retrieved_chunks:
            st.markdown(f"**{item.chunk.heading}**, p. {item.chunk.page_start}")
            st.text(item.chunk.text)
    st.caption(f"{answer.latency_ms / 1000:.1f} s. {answer.disclaimer}")
    _rating_controls(service, recorded.record)


def _rating_controls(service: HistoryService, record: RunRecord) -> None:
    note = st.text_input("Reviewer note (optional)", key=f"note-{record.run_id}")
    up, down, status = st.columns([1, 1, 6])
    for column, rating in ((up, RATING_UP), (down, RATING_DOWN)):
        if column.button(RATING_LABELS[rating], key=f"rate-{rating}-{record.run_id}"):
            service.rate(record.run_id, rating, note)
            st.session_state[RATED_KEY] = record.run_id
    if st.session_state.get(RATED_KEY) == record.run_id:
        status.success("Rating saved")


def _history_tab(service: HistoryService) -> None:
    only_rated = st.checkbox("Only rated runs", key="only_rated")
    runs = [r for r in service.recent(HISTORY_PAGE_SIZE) if r.rating is not None or not only_rated]
    if not runs:
        st.info("No runs yet.")
        return
    st.dataframe([_summary_row(run) for run in runs], use_container_width=True, hide_index=True)
    chosen = st.selectbox(
        "Open a run",
        runs,
        format_func=lambda run: f"{run.created_at}  {run.question[:70]}",
        key="open_run",
    )
    if chosen is not None:
        _show_run(chosen)


def _summary_row(run: RunRecord) -> dict[str, str | float]:
    return {
        "time": run.created_at,
        "user": run.user,
        "question": run.question,
        "refused": "yes" if run.refused else "no",
        "rating": RATING_LABELS[run.rating],
        "model": run.model,
        "seconds": round(run.latency_ms / 1000, 1),
    }


def _show_run(run: RunRecord) -> None:
    st.markdown(run.answer)
    st.caption(f"Citations: {', '.join(run.citations) or 'none'}")
    st.caption(f"Index version {run.index_version}. Note: {run.note or '-'}")
    st.caption("Retrieved: " + ", ".join(run.retrieved_chunk_ids))


if __name__ == "__main__":
    main(load_container)
