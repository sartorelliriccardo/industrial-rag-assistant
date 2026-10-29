"""Streamlit UI (Italian / English): upload PDFs, ask questions, inspect the evidence.

This file only assembles the page; each part lives in a file next to it:
  ui_sidebar       settings and upload          ui_assistant     ask and read the answer
  ui_walkthrough   step by step                 ui_architecture  the flows drawn
  ui_privacy       password, session store, deletion
  ui_state         what every tab receives
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
try:
    import rag_assistant  # noqa: F401
except ImportError:  # running without `pip install -e .` (e.g. Streamlit Cloud)
    sys.path.insert(0, str(ROOT / "src"))
if str(APP_DIR) not in sys.path:  # the ui_*.py files sit next to this script
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from rag_assistant.i18n import t
from ui_architecture import render_architecture
from ui_assistant import render_assistant
from ui_privacy import IDLE_MINUTES, clear_documents, idle_guard, require_password
from ui_sidebar import process_uploads, render_sidebar
from ui_state import open_workspace
from ui_walkthrough import render_walkthrough

st.set_page_config(page_title="Industrial RAG Assistant", page_icon="🏭", layout="wide")

require_password()
st.session_state.last_active = time.time()  # every full run is a user action

# ---------------------------------------------------------------- settings, documents
side = render_sidebar()
ctx = open_workspace(side.lang, side.search, side.provider_name, side.api_key, side.model)
process_uploads(ctx, side.uploads)  # indexes new PDFs, then reruns the script
side.clear_slot.button(
    t(ctx.lang, "clear_all"), on_click=clear_documents, args=(ctx.lang,), disabled=not ctx.files
)

# ---------------------------------------------------------------- page
st.title(f"🏭 {t(ctx.lang, 'title')}")
st.caption(t(ctx.lang, "subtitle"))

tab_assistant, tab_about, tab_walk, tab_arch = st.tabs(
    [t(ctx.lang, key) for key in ("tab_assistant", "tab_about", "tab_walk", "tab_arch")]
)
with tab_assistant:
    render_assistant(ctx)
with tab_about:
    st.markdown(t(ctx.lang, "about_md"))
with tab_walk:
    render_walkthrough(ctx)
with tab_arch:
    render_architecture(ctx.lang)

if IDLE_MINUTES > 0:
    idle_guard(ctx.lang)
