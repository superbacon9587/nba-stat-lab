"""Deploy smoke test: a throwaway page that checks the hosting plumbing.

It checks the hosting plumbing end to end without computing any statistics:
data download and load, the API key from secrets, and parser routing under
the rate limits. The calls marked KEEP are the ones the real app (app/streamlit_app.py) needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Hosts install requirements.txt but not this project, so import nbalab from src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from nbalab.deploy.fetch import DataUnavailable, data_manifest, ensure_data  # noqa: E402
from nbalab.deploy.rate_limit import app_limiter, route_question, session_limiter  # noqa: E402
from nbalab.deploy.secrets import anthropic_api_key  # noqa: E402

st.set_page_config(page_title="NBA Stat Lab", page_icon="🏀", layout="wide")


@st.cache_resource(show_spinner="Downloading NBA data (first start only)...")
def _data_dir() -> Path:
    return ensure_data()  # KEEP: download-on-first-start


@st.cache_resource(show_spinner="Loading tables...")
def _query_data(directory: str):
    from nbalab.query.data import load_query_data

    return load_query_data(Path(directory))  # KEEP: one load per server process


st.title("NBA Stat Lab")
st.caption("Deployment check. The full app is not built yet.")

try:
    data_dir = _data_dir()
except DataUnavailable as exc:
    st.error(str(exc))
    st.stop()

manifest = data_manifest(data_dir)
cols = st.columns(3)
cols[0].metric("Data source", data_dir.name)
cols[1].metric("First season", manifest.get("start_season", "n/a"))
cols[2].metric("AI parser", "on" if anthropic_api_key() else "off (rule-based)")

try:
    qd = _query_data(str(data_dir))
    st.success(
        f"Loaded {len(qd.player_games):,} player-games and {len(qd.team_games):,} team-games, "
        f"seasons {qd.player_games['season'].min()}-{qd.latest_season}. "
        f"On-court table: {'yes' if qd.on_court is not None else 'no'}."
    )
except Exception as exc:  # surface loader problems on the smoke page instead of a blank app
    st.error(f"Data loaded from disk but the query layer could not read it: {exc}")

question = st.text_input("Try a question (routing only, nothing is sent to the API yet)")
if question:
    route = route_question()  # KEEP: call once per uncached question, before parsing
    if route.parser == "llm":
        st.info("This question would go to the AI parser.")
    else:
        st.warning(route.reason)
    st.caption(
        f"AI questions left: {session_limiter().remaining()} this session, "
        f"{app_limiter().remaining()} for the whole app this hour."
    )
