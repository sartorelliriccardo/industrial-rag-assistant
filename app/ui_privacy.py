"""Who may use the app and when the documents go away.

Optional password, the per-session document store, deletion of uploaded files (including
Streamlit's own copy) and the idle auto-delete. The rules themselves are in
rag_assistant.session and rag_assistant.workspace and are unit-tested there.
"""

import os
import time
from uuid import uuid4

import streamlit as st
from streamlit import runtime as st_runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

from rag_assistant.i18n import t
from rag_assistant.session import check_interval_seconds, idle_expired, passwords_match
from rag_assistant.workspace import Workspace, get_store

# uploaded documents are dropped after this many idle minutes (0 disables it)
IDLE_MINUTES = float(os.environ.get("DOC_IDLE_MINUTES", "15"))


def configured_password() -> str:
    """Optional access password: APP_PASSWORD in the environment or in Streamlit secrets."""
    value = os.environ.get("APP_PASSWORD", "")
    if not value:
        try:
            value = str(st.secrets.get("APP_PASSWORD", ""))
        except Exception:  # no secrets file configured
            value = ""
    return value


def require_password() -> None:
    """Stop the script until the right password is typed. Does nothing if none is set."""
    expected = configured_password()
    if not expected or st.session_state.get("authenticated"):
        return
    typed = st.text_input("Password", type="password", key="app_password")
    if typed and passwords_match(expected, typed):
        st.session_state.authenticated = True
        st.rerun()
    if typed:
        st.error("Password errata / Wrong password")
    st.stop()


def session_is_active(session_id: str) -> bool:
    return st_runtime.get_instance().is_active_session(session_id)


def current_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is not None:
        return ctx.session_id
    return st.session_state.setdefault("fallback_session_id", uuid4().hex)


# One store for the whole process: documents live here, not in Streamlit's session state,
# so they are discarded a few seconds after the browser tab goes away (see workspace.py).
store = get_store(session_is_active)


def workspace() -> Workspace:
    return store.get_or_create(current_session_id())


def discard_upload_buffers() -> None:
    """Delete Streamlit's own server-side copy of the files uploaded in this session.

    Streamlit keeps every uploaded file in memory until the session is cleaned up, even
    after the widget is reset. We already hold the bytes we need in our workspace.
    """
    try:
        st_runtime.get_instance().uploaded_file_mgr.remove_session_files(current_session_id())
    except Exception:  # best effort: the session cleanup will still remove them eventually
        pass


def clear_documents(language: str, message_key: str = "cleared") -> None:
    """Drop every uploaded document from memory and reset the upload widget."""
    ws = workspace()
    if ws.pipeline is not None:
        ws.pipeline.clear()
    ws.files.clear()
    discard_upload_buffers()
    st.session_state.question = ""
    st.session_state.upload_gen = st.session_state.get("upload_gen", 0) + 1
    st.toast(t(language, message_key, minutes=f"{IDLE_MINUTES:g}"))


@st.fragment(run_every=check_interval_seconds(IDLE_MINUTES))
def idle_guard(lang: str) -> None:
    """Drop the documents if nobody touched the app for IDLE_MINUTES.

    Runs by itself every few seconds while the page is open (a fragment re-run does not
    count as user activity). Once the page is closed nothing runs here: the session is then
    discarded by Streamlit.
    """
    if workspace().files and idle_expired(
        st.session_state.get("last_active", time.time()), time.time(), IDLE_MINUTES
    ):
        clear_documents(lang, "idle_cleared")
        st.rerun()
