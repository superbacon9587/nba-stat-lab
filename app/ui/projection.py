"""Projection section: expected value, nested prediction intervals, P(over)/P(under),
the probability-vs-line curve, joint probability for combos, and live line/minutes sliders.

Every number comes from ``nbalab.models``. Moving the line only re-reads the
model's distribution (``p_over_at``). Moving projected minutes re-runs the
projection (cached per minutes value).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import streamlit as st

from nbalab.query import schema as s
from nbalab.query.engine import SplitResult
from nbalab.query.stats import stat_catalog
from nbalab.viz import charts
from ui import backend, components as C

PLOT_CONFIG = {"displaylogo": False}
LEVELS = (90, 95, 99)


def render(query_json: str, split: SplitResult, stat: str) -> None:
    q = s.StatQuery.model_validate_json(query_json)
    label = stat_catalog(q.subject_type)[stat].label
    C.section("Projection for the next game", "Forward-looking model",
              "A model's forecast for one future game in this context, as opposed to the historical averages above.")
    ctx = q.projection.context if q.projection else s.ProjectionContext()
    minutes = None
    if q.subject_type == "player":
        mcol, _ = st.columns([2, 3])
        with mcol:
            override = st.toggle("Override projected minutes", value=ctx.projected_minutes is not None,
                                 key=f"pm_on_{stat}")
            if override:
                minutes = st.slider("Projected minutes", 5.0, 48.0, float(ctx.projected_minutes or 34.0), 0.5,
                                    key=f"pm_{stat}")
    res = backend.project(query_json, minutes)
    sp = _stat_projection(res, stat)
    if sp is None:
        _fallback(res, split, stat, q)
        return
    line = _line_slider(sp, q, stat, label)
    _summary(sp, line, label)
    _charts(sp, line, label)
    _joint(res, q, stat)
    _contributions(sp, label)
    _warnings(sp, res)


def _stat_projection(res: Any, stat: str) -> Any:
    if res is None or isinstance(res, Exception):
        return None
    stats = getattr(res, "stats", None)
    return stats.get(stat) if isinstance(stats, dict) else None


def _line_slider(sp: Any, q: s.StatQuery, stat: str, label: str) -> float:
    lo99, hi99 = sp.pi[99]
    default = (q.projection.lines.get(stat) if q.projection else None)
    if default is None:
        default = math.floor(sp.mean) + 0.5
    lo = max(0.0, math.floor(min(lo99, default) - 2))
    hi = math.ceil(max(hi99, default) + 2)
    return st.slider(f"{label} line", float(lo), float(hi), float(default), 0.5, key=f"line_{stat}_{hash(q.model_dump_json())}",
                     help="Drag to see the over/under probabilities for any line. Whole-number lines can push.")


def _summary(sp: Any, line: float, label: str) -> None:
    p_over, p_under, p_push = sp.probabilities_at(line) if hasattr(sp, "probabilities_at") else (
        sp.p_over_at(line), sp.p_under_at(line), sp.p_push_at(line))
    whole = float(line).is_integer()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(C.big_number("Expected " + label.lower(), f"{sp.mean:.1f}"), unsafe_allow_html=True)
    c1.caption(f"median {sp.median:.1f}" + (f" · {sp.minutes_mean:.1f} projected min" if getattr(sp, "minutes_mean", None) else ""))
    c2.markdown(C.big_number(f"P(over {line:g})", f"{p_over:.0%}", "nbl-up" if p_over > 0.5 else ""), unsafe_allow_html=True)
    c3.markdown(C.big_number(f"P(under {line:g})", f"{p_under:.0%}", "nbl-down" if p_under > 0.5 else ""), unsafe_allow_html=True)
    c4.markdown(C.big_number(f"P(push at {line:g})", f"{p_push:.0%}" if whole else "0%"), unsafe_allow_html=True)
    c4.caption("lands exactly on the line" if whole else "no push on a .5 line")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Range for this one game** (prediction interval)", help=_tooltip_ci_vs_pi())
        st.markdown("  \n".join(f"**{lvl}%** of games: {sp.pi[lvl][0]:.0f} – {sp.pi[lvl][1]:.0f}"
                                 for lvl in LEVELS if lvl in sp.pi))
    ci = getattr(sp, "ci_mean", {}) or {}
    with r2:
        st.markdown("**Range for the average** (confidence interval)", help=_tooltip_ci_vs_pi())
        st.markdown("  \n".join(f"**{lvl}%** CI of the long-run average: {ci[lvl][0]:.1f} – {ci[lvl][1]:.1f}"
                                 for lvl in LEVELS if lvl in ci) or "not available")


def _tooltip_ci_vs_pi() -> str:
    return ("Prediction interval = where this ONE game's box score will likely land. It is wide because a single "
            "game swings a lot (shooting luck, minutes, foul trouble). A 95% prediction interval should contain "
            "the real result in about 95 of 100 games.\n\nConfidence interval = where the player's true long-run "
            "AVERAGE in this context sits. It is narrow because it only reflects how precisely we know the average. "
            "Compare a betting line with the prediction interval and P(over), not with the confidence interval.")


def _charts(sp: Any, line: float, label: str) -> None:
    st.markdown("**Where the game will likely land**")
    with st.container():
        ints = {lvl / 100: tuple(sp.pi[lvl]) for lvl in LEVELS if lvl in sp.pi}
        st.plotly_chart(charts.interval_chart(sp.mean, ints, line, label, sp.median), width="stretch",
                        config=PLOT_CONFIG)
        st.caption("Nested bands: 90% (darkest), 95%, 99% (lightest) prediction intervals. Yellow = the line.")
    left, _ = st.columns([3, 1])
    with left:
        st.markdown("**P(over) for every line**")
        lo, hi = sp.pi[99]
        grid = np.arange(max(0.0, math.floor(lo) - 0.5), math.ceil(hi) + 1.0, 0.5)
        curve = sp.over_curve(grid)
        st.plotly_chart(charts.probability_curve(curve["line"].to_numpy(), curve["p_over"].to_numpy(), line, label),
                        width="stretch", config=PLOT_CONFIG)


def _joint(res: Any, q: s.StatQuery, stat: str) -> None:
    lines = dict(q.projection.lines) if q.projection else {}
    if len(lines) < 2 or not hasattr(res, "joint_probability"):
        return
    catalog = stat_catalog(q.subject_type)
    st.markdown("**Combo: all overs hit together**")
    cols = st.columns(len(lines) + 1)
    chosen = {}
    for c, (k, v) in zip(cols, lines.items()):
        chosen[k] = c.number_input(f"{catalog[k].label} over", value=float(v), step=1.0, key=f"combo_{k}")
    p_joint = res.joint_probability({k: (">", v) for k, v in chosen.items()})
    indep = float(np.prod([res.stats[k].p_over_at(v) for k, v in chosen.items() if k in res.stats]))
    cols[-1].markdown(C.big_number("P(all over)", f"{p_joint:.0%}"), unsafe_allow_html=True)
    cols[-1].caption(f"If the stats were unrelated: {indep:.0%}. The gap comes from correlation between "
                     "them in this player's history (Gaussian copula, 10,000 simulated games).")


def _contributions(sp: Any, label: str) -> None:
    contribs = getattr(sp, "contributions", None)
    if not contribs:
        return
    from nbalab.viz.projection import contribution_waterfall

    with st.expander("What moves the projection", icon=":material/stacked_bar_chart:", expanded=False):
        labels = [c.label + (" (proxy)" if getattr(c, "is_proxy", False) else "") for c in contribs]
        fig = contribution_waterfall(labels, [c.value for c in contribs], getattr(sp, "neutral_mean", sp.mean), label)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption("Starts from a neutral game for this player (league-average opponent, 50/50 home/away, 1 day of "
                   "rest, recent minutes, form at its longer-run level) and adds each part of the context. These are "
                   "Shapley values, so the steps add up exactly to the projection.")


def _warnings(sp: Any, res: Any) -> None:
    notes = list(getattr(res, "warnings", []) or []) + list(getattr(sp, "warnings", []) or [])
    model = getattr(sp, "model_name", None)
    if model:
        notes.append(f"Model: {model}. Backtest and calibration are on the Methodology page.")
    for n in notes:
        st.caption(n)


def _fallback(res: Any, split: SplitResult, stat: str, q: s.StatQuery) -> None:
    """No model available: say why, and show the historical hit rate (clearly labeled) instead."""
    if isinstance(res, Exception):
        st.warning(f"The projection model couldn't run for this question: {res}", icon=":material/error:")
    else:
        st.info("The projection model isn't available yet, so only the historical view is shown.",
                icon=":material/hourglass_empty:")
    hr = split.stats[stat].hit_rates
    line = q.projection.lines.get(stat) if q.projection else None
    if hr and line is not None:
        a, b = st.columns(2)
        a.markdown(C.big_number(f"Past games over {line:g} (this split)", f"{hr['split']['over']:.0%}"
                                if hr['split']['n'] else "–"), unsafe_allow_html=True)
        a.caption(f"{int(hr['split']['n'])} games")
        b.markdown(C.big_number(f"Past games over {line:g} (baseline)", f"{hr['baseline']['over']:.0%}"),
                   unsafe_allow_html=True)
        b.caption(f"{int(hr['baseline']['n'])} games. A historical frequency, not a forecast.")
