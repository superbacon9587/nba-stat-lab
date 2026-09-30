"""League-wide variable effects (the ``variable_effect`` regression), shared by the
Explore page and by league-wide questions on the Ask page."""

from __future__ import annotations

import calendar
import math

import pandas as pd
import streamlit as st

from nbalab.query import schema as s
from nbalab.query.defenders import SOURCE_LABELS
from nbalab.query.stats import stat_catalog
from nbalab.viz import charts
from ui import backend, components as C

PLOT_CONFIG = {"displaylogo": False}
CONTROL_LABELS = {"minutes": "minutes played", "is_home": "home/away", "rest_days": "rest days",
                  "opp_def_rating": "opponent defensive rating", "def_height": "defender height",
                  "opp_avg_height": "opponent's minutes-weighted average height"}
NOMINAL = {"venue", "defender_position"}
EXTRA_SOURCE_LABELS = {"excluded_no_individual_defender": "Left out: no individual defender known for the game"}


def level_label(term: str, variable: str) -> str:
    """'day_of_week=Monday' -> 'Monday', 'month=02' -> 'Feb', plain terms unchanged."""
    level = term.split("=", 1)[1] if "=" in term else term
    if variable == "month" and level.isdigit():
        return calendar.month_abbr[int(level)]
    return level


def render_effect(q: s.StatQuery, heading: bool = False, control_minutes: bool = True) -> None:
    with st.spinner("Fitting the league-wide regression…"):
        try:
            res = backend.variable_effect(q.model_dump_json(), control_minutes)
        except ValueError as exc:
            st.error(str(exc))
            return
    catalog = stat_catalog(q.subject_type)
    for stat in q.stats:
        table = res.per_stat[stat]
        coefs = table.coefficients[table.coefficients["is_variable"]].copy()
        label = catalog[stat].label
        name = res.variable.replace("_", " ")
        if heading:
            st.markdown(f"### How {name} affects {label.lower()}, league-wide")
        pop = ", ".join(res.population) if res.population else "all players"
        st.caption(f"{pop} · {table.n_obs:,} games from {table.n_subjects:,} "
                   f"{'teams' if q.subject_type == 'team' else 'players'} · average {label.lower()} {table.stat_mean:.1f}")
        if len(coefs) == 1:
            _single_effect(coefs.iloc[0], res.unit, label)
        elif len(coefs) > 1:
            coefs["label"] = [level_label(t, res.variable) for t in coefs["term"]]
            if res.variable in NOMINAL:  # no natural order: rank by effect
                coefs = coefs.sort_values("coef", ascending=False)
            ref = res.unit.replace("vs ", "") if res.unit.startswith("vs ") else None
            fig = charts.coefficient_chart(coefs.rename(columns={"coef": "estimate", "ci_low": "lo", "ci_high": "hi"}),
                                           label, reference=ref or res.unit)
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
            st.caption(f"Each row: change in {label.lower()} for a typical player, {res.unit}. "
                       "Whiskers are 95% intervals with player-clustered standard errors. Faded = could be zero.")
        if res.variable in ("back_to_back", "rest_days"):
            from ui.results import SURVIVORSHIP_NOTE

            st.warning(SURVIVORSHIP_NOTE + " Turn on 'Hold minutes fixed' to compare per minute played.",
                       icon=":material/airline_seat_recline_normal:")
        _method_box(res, table, control_minutes)


def _single_effect(row: pd.Series, unit: str, label: str) -> None:
    coef, lo, hi, p = float(row["coef"]), float(row["ci_low"]), float(row["ci_high"]), float(row["p_value"])
    cls = "nbl-up" if coef > 0 else "nbl-down" if coef < 0 else ""
    sig = "statistically clear" if p < 0.05 else "not statistically clear (could be zero)"
    C.cards([
        C.card(f"Effect on {label.lower()}", f"{'▲' if coef > 0 else '▼'} {coef:+.2f}", unit, cls="hero", value_cls=cls),
        C.card("95% interval", f"{lo:+.2f} to {hi:+.2f}", "player-clustered standard errors"),
        C.card("p-value", f"{p:.3g}" if not math.isnan(p) else "–", sig),
    ])


def _method_box(res, table, control_minutes: bool) -> None:
    with st.expander("How this was estimated", icon=":material/functions:"):
        controls = ", ".join(CONTROL_LABELS.get(c, c) for c in res.controls) or "none"
        st.markdown(
            f"**Method:** {res.method}.  \n"
            "Each player is compared **with their own games** (player fixed effects), so stars and bench players "
            "don't get mixed up. The number is how much the stat moves for the same player when this variable "
            f"changes, holding fixed: {controls}.  \n"
            f"Within-player R² = {table.r2_within:.3f}. Most game-to-game variation is noise, "
            "which is expected for single games."
        )
        if control_minutes:
            st.caption("Minutes are held fixed, so effects that work *through* playing time "
                       "(e.g. fewer minutes on back-to-backs) are excluded. Turn that off to include them.")
        else:
            st.caption("This is the total effect, including any change in playing time. Turn on "
                       "'Hold minutes fixed' to see the effect per minute played.")
        for src, n in (res.defender_sources or {}).items():
            st.caption(f"Defender data: {SOURCE_LABELS.get(src) or EXTRA_SOURCE_LABELS.get(src, src)}: {n:,} games")
        for n in res.notes:
            st.caption(n)
