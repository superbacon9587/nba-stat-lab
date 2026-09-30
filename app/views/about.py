"""About the author. Fill in the placeholders below."""

from __future__ import annotations

import streamlit as st

from ui import components as C

# ---- edit these ------------------------------------------------------------
NAME = "Your Name"
TAGLINE = "Data scientist · sports analytics"
GITHUB_URL = "https://github.com/your-username/nba-stat-lab"
LINKEDIN_URL = "https://www.linkedin.com/in/your-profile"
EMAIL = ""  # optional
# ------------------------------------------------------------------------------

C.header("views/methodology.py")
st.markdown(f"## {NAME}")
st.markdown(f"<div class='nbl-note'>{TAGLINE}</div>", unsafe_allow_html=True)
st.write("")
c1, c2, c3 = st.columns([1, 1, 3])
c1.link_button("GitHub", GITHUB_URL, icon=":material/code:", width="stretch")
c2.link_button("LinkedIn", LINKEDIN_URL, icon=":material/work:", width="stretch")
if EMAIL:
    c3.markdown(f"[{EMAIL}](mailto:{EMAIL})")

st.markdown("#### About this project")
st.markdown(
    "NBA Stat Lab answers plain-English basketball questions with real statistics. It combines:\n\n"
    "- an **LLM parser** (Claude with tool use) that only translates questions into a typed query,\n"
    "- a **split engine** with Welch t-tests, bootstrap intervals and empirical Bayes shrinkage,\n"
    "- **league-wide fixed-effects regressions** with player-clustered standard errors,\n"
    "- a **projection model** with count distributions, prediction intervals and copula-linked combos, "
    "backtested on held-out seasons.\n\n"
    "**Stack:** Python 3.11 · pandas · DuckDB · statsmodels · scikit-learn · LightGBM · SciPy · Plotly · "
    "Streamlit · Anthropic SDK · pytest."
)
st.caption("Not affiliated with the NBA. Not betting advice.")
