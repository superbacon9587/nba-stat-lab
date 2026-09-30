"""Explore variables: how one variable moves a stat for a typical player, league-wide."""

from __future__ import annotations

import streamlit as st

from nbalab.query import schema as s
from nbalab.query.stats import PLAYER_STATS
from ui import backend, components as C, explore

FEATURED = [
    ("defender_height", "Defender height"), ("rest_days", "Rest days"), ("day_of_week", "Day of week"),
    ("venue", "Venue (arena)"), ("altitude", "Altitude"), ("denver", "Playing in Denver"), ("month", "Month"),
]
STATS = ["points", "rebounds", "assists", "threes", "pra", "ts_pct", "minutes", "turnovers"]

C.header("views/methodology.py")
st.markdown("### Explore variables")
st.markdown("<div class='nbl-note'>Pick a variable and see how much it moves a stat for a <i>typical</i> player, "
            "estimated across every player in the league at once. Each player is only compared with their own games.</div>",
            unsafe_allow_html=True)

available = backend.effect_variables()
featured = [(k, v) for k, v in FEATURED if k in available]
others = [(k, k.replace("_", " ").capitalize()) for k in available if k not in dict(featured)]

c1, c2, c3 = st.columns([2, 1.3, 1.3])
with c1:
    options = featured + others
    var = st.selectbox("Variable", [k for k, _ in options], format_func=dict(options).get, key="ex_var")
    st.caption(available[var].unit)
with c2:
    stat = st.selectbox("Stat", STATS, format_func=lambda k: PLAYER_STATS[k].label, key="ex_stat")
    positions = st.multiselect("Positions", ["G", "F", "C"], placeholder="All positions", key="ex_pos")
with c3:
    all_seasons = st.toggle("All seasons since 1996", value=False, key="ex_all",
                            help="Default is the last 5 seasons, which loads in under a second. "
                                 "All 30 seasons takes a few seconds the first time, then it's cached.")
    control_minutes = st.toggle("Hold minutes fixed", value=False, key="ex_min",
                                help="On: the effect for the same minutes played. Off: also counts effects "
                                     "that act through playing time.")

filters: list = [] if all_seasons else [s.LastNSeasonsFilter(n=5)]
if positions:
    filters.append(s.PlayerPositionFilter(positions=positions))
q = s.StatQuery(stats=[stat], mode="variable_effect", effect_variable=var, filters=filters)
st.divider()
explore.render_effect(q, heading=True, control_minutes=control_minutes)
