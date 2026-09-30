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


# ------------------------------------------------------------------ season timing
st.divider()
st.markdown("### Season timing")
st.markdown("<div class='nbl-note'>Does a team or player change after the All-Star break, after a date, between "
            "groups of months, or in the last games before the playoffs? Regular season only; every team at once "
            "with league-adjusted changes and multiple-testing control, or one player with per-36 rates.</div>",
            unsafe_allow_html=True)
from nbalab.query.stats import TEAM_STATS  # noqa: E402
from ui import periods  # noqa: E402

TEAM_TIMING_STATS = ["team_score", "opponent_score", "net_rating", "off_rating", "def_rating", "pace",
                     "threes_attempted", "three_pct", "win_pct", "margin"]
lo_season, hi_season = backend.season_bounds()
t1, t2, t3, t4 = st.columns([1.1, 1.3, 1.3, 1.6])
who = t1.segmented_control("Who", ["Every team", "A player"], default="Every team", key="st_who")
kind = t2.selectbox("Split point", list(periods.KIND_LABELS), format_func=periods.KIND_LABELS.get, key="st_kind")
split_args: dict = {"kind": kind}
if kind == "custom_date":
    split_args["date"] = t2.text_input("Date (MM-DD)", "02-20", key="st_date")
elif kind == "month_groups":
    split_args["before_months"] = t2.multiselect("Before", list(periods.MONTHS), [10, 11, 12, 1],
                                                 format_func=periods.MONTHS.get, key="st_bm")
    split_args["after_months"] = t2.multiselect("After", list(periods.MONTHS), [2, 3, 4],
                                                format_func=periods.MONTHS.get, key="st_am")
elif kind == "last_n_before_playoffs":
    split_args["n_games"] = t2.slider("Last N games", 5, 40, 20, key="st_n")
seasons = t4.slider("Seasons (starting year)", lo_season, hi_season, (hi_season - 2, hi_season), key="st_seasons")
if who == "A player":
    players = backend.player_options()
    pid = t3.selectbox("Player", players["personId"], format_func=backend.player_label, index=None,
                       placeholder="Type a name", key="st_player")
    pstat = t3.selectbox("Stat", STATS, format_func=lambda k: PLAYER_STATS[k].label, key="st_pstat")
    subject_type, ids, stats = "player", [int(pid)] if pid is not None else [], [pstat]
else:
    tstat = t3.selectbox("Stat", TEAM_TIMING_STATS, format_func=lambda k: TEAM_STATS[k].label, key="st_tstat")
    subject_type, ids, stats = "team", [], [tstat]
try:
    timing_q = s.StatQuery(subject_type=subject_type, subject_ids=ids, stats=stats, mode="period",
                           period_split=s.PeriodSplit(**split_args),
                           filters=[s.SeasonRangeFilter(start=seasons[0], end=seasons[1])])
except ValueError as exc:
    st.warning(f"Pick a valid split: {exc}")
else:
    if who == "A player" and not ids:
        st.info("Pick a player, or switch to 'Every team'. Leave the player empty on the Ask page "
                "('who improves most after the break') to rank every player.")
    else:
        periods.render(timing_q, key="explore_timing")
