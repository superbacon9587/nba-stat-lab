"""NBA Stat Lab: Streamlit entry point.

Run locally:  .venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Hosts install requirements.txt but not the nbalab package itself.
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="NBA Stat Lab", page_icon="🏀", layout="wide", initial_sidebar_state="collapsed")

try:  # copy hosted secrets (st.secrets) into the environment before the parser loads
    from nbalab.deploy.secrets import export_secrets_to_env  # noqa: E402

    export_secrets_to_env()
except ImportError:
    pass

from ui import components as C  # noqa: E402

ask = st.Page("views/ask.py", title="Ask", icon=":material/chat:", default=True)
explore = st.Page("views/explore.py", title="Explore variables", icon=":material/insights:")
methodology = st.Page("views/methodology.py", title="Methodology", icon=":material/menu_book:")
about = st.Page("views/about.py", title="About", icon=":material/person:")

nav = st.navigation([ask, explore, methodology, about], position="top")
C.inject_css()
nav.run()
