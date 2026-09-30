"""Plotly figure builders. Every function takes plain numbers or DataFrames and
returns a ``go.Figure``. None of them compute statistics: they only draw what
the query engine and the models already computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nbalab.viz import theme as T
from nbalab.viz.trend import TrendFit

T.register()


def _fmt(x: float, digits: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


# ----------------------------------------------------------------- variable effects


@dataclass(frozen=True)
class EffectRow:
    """One bar in the "how each variable affects this stat" chart.

    ``diff`` is (split average - baseline average) for this one filter alone,
    ``lo``/``hi`` its 95% interval, ``n`` the games in that split.
    """

    label: str
    diff: float
    lo: float
    hi: float
    n: int
    p_value: float = math.nan
    combined: bool = False

    @property
    def significant(self) -> bool:
        """True when the 95% interval does not cross zero."""
        return not (math.isnan(self.lo) or math.isnan(self.hi)) and (self.lo > 0 or self.hi < 0)


def effect_chart(rows: Sequence[EffectRow], baseline: float, stat_label: str, digits: int = 1,
                 show_n: bool = True) -> go.Figure:
    """Horizontal bars of each filter's effect vs the baseline, plus the combined effect.

    Every bar starts at zero (= the baseline). Green bars point right (above
    baseline), red bars point left (below). Whiskers are 95% intervals. Faded
    bars have intervals that cross zero: not distinguishable from the baseline.
    The combined bar sits last, below a divider, because the individual effects
    overlap and do not simply add up.
    """
    rows = list(rows)
    finite = [abs(r.diff) for r in rows if not math.isnan(r.diff)]
    if finite and max(finite) < 1 and digits < 2:
        digits = 2  # small effects: one decimal would print "-0.0"
    singles = [r for r in rows if not r.combined]
    combined = [r for r in rows if r.combined]
    ordered = singles + combined
    labels = [r.label for r in ordered]
    y = [f"<b>{r.label}</b>" if r.combined else r.label for r in ordered]

    colors, opac = [], []
    for r in ordered:
        colors.append(T.signed_color(r.diff))
        opac.append(1.0 if r.significant else 0.45)

    diffs = [0.0 if math.isnan(r.diff) else r.diff for r in ordered]
    err_plus = [max(r.hi - r.diff, 0) if not math.isnan(r.hi) else 0 for r in ordered]
    err_minus = [max(r.diff - r.lo, 0) if not math.isnan(r.lo) else 0 for r in ordered]
    text = [
        f"{'▲' if r.diff > 0 else '▼' if r.diff < 0 else '•'} {_fmt(r.diff, digits, True)}  "
        + (f"<span style='color:{T.MUTED}'>n={r.n}</span>" if show_n and r.n else "")
        for r in ordered
    ]
    hover = [
        f"<b>{r.label}</b><br>vs baseline: {_fmt(r.diff, digits, True)}"
        f"<br>95% interval: {_fmt(r.lo, digits, True)} to {_fmt(r.hi, digits, True)}"
        f"<br>stat in this split: {_fmt(baseline + r.diff, digits)}"
        f"<br>games: {r.n}"
        + (f"<br>p-value: {r.p_value:.3f}" if not math.isnan(r.p_value) else "")
        + ("" if r.significant else "<br><i>interval crosses 0: could be noise</i>")
        for r in ordered
    ]

    fig = go.Figure(
        go.Bar(
            x=diffs, y=y, orientation="h",
            marker=dict(color=colors, opacity=opac, cornerradius=4, line=dict(width=0)),
            error_x=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus,
                         color=T.INK_2, thickness=1.5, width=6),
            hovertext=hover, hoverinfo="text", width=0.55,
        )
    )
    # value labels sit just past the end of each whisker, on the bar's side, so they never overlap a mark
    for yv, d, ep, em, txt in zip(y, diffs, err_plus, err_minus, text):
        right = d >= 0
        fig.add_annotation(x=d + ep if right else d - em, y=yv, text=txt, showarrow=False,
                           xanchor="left" if right else "right", xshift=8 if right else -8,
                           font=dict(color=T.INK, size=13))
    span = max([abs(v) for v in err_plus + err_minus] + [abs(d) for d in diffs] + [1e-9])
    ext = max((abs(d) + max(p, m)) for d, p, m in zip(diffs, err_plus, err_minus)) if ordered else 1
    pad = ext * 0.75 + span * 0.05
    fig.add_vline(x=0, line=dict(color=T.INK_2, width=1.5))
    fig.add_annotation(
        x=0, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
        text=f"baseline {_fmt(baseline, digits)}", font=dict(color=T.INK_2, size=12),
    )
    if combined and singles:
        fig.add_hline(y=len(singles) - 0.5, line=dict(color=T.AXIS, width=1))
    fig.update_layout(
        height=max(180, 70 + 52 * len(ordered)),
        showlegend=False,
        xaxis=dict(title=f"{stat_label}: difference from baseline", range=[-ext - pad, ext + pad],
                   zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False, showline=False, categoryorder="array",
                   categoryarray=y, tickfont=dict(color=T.INK, size=13)),
        margin=dict(l=16, r=32, t=40, b=48),
    )
    del labels
    return fig


# ------------------------------------------------------------------- distribution


def distribution_chart(split: pd.Series, everything: pd.Series, stat_label: str,
                       split_label: str = "This split", all_label: str = "All games") -> go.Figure:
    """Overlaid histograms (as % of games) of the stat in the split vs all games.

    Percent-of-games (not counts) keeps a 30-game split comparable with a
    900-game baseline. Vertical lines mark each group's average.
    """
    s = pd.to_numeric(split, errors="coerce").dropna()
    a = pd.to_numeric(everything, errors="coerce").dropna()
    both = pd.concat([s, a])
    fig = go.Figure()
    if both.empty:
        return fig
    lo, hi = float(both.min()), float(both.max())
    is_int = bool(np.all(np.equal(np.mod(both.to_numpy(), 1), 0)))
    width = max(1.0, math.ceil((hi - lo) / 30)) if is_int else max((hi - lo) / 30, 1e-6)
    bins = dict(start=lo - (0.5 if is_int else 0), end=hi + width, size=width)
    fig.add_trace(go.Histogram(x=a, xbins=bins, histnorm="percent", name=f"{all_label} (n={len(a)})",
                               marker=dict(color=T.MUTED, line=dict(color=T.SURFACE, width=1)), opacity=0.55,
                               hovertemplate="%{x}: %{y:.1f}% of games<extra>" + all_label + "</extra>"))
    fig.add_trace(go.Histogram(x=s, xbins=bins, histnorm="percent", name=f"{split_label} (n={len(s)})",
                               marker=dict(color=T.SERIES[0], line=dict(color=T.SURFACE, width=1)), opacity=0.8,
                               hovertemplate="%{x}: %{y:.1f}% of games<extra>" + split_label + "</extra>"))
    for vals, color, name in ((a, T.INK_2, "all"), (s, T.SERIES[0], "split")):
        if len(vals):
            m = float(vals.mean())
            fig.add_vline(x=m, line=dict(color=color, width=2))
            fig.add_annotation(x=m, y=1, yref="paper", yanchor="bottom", showarrow=False,
                               text=f"avg {m:.1f}", font=dict(color=T.INK_2, size=12),
                               xshift=-28 if name == "all" and len(s) and m < s.mean() else 28)
    fig.update_layout(barmode="overlay", height=340, xaxis_title=stat_label, yaxis_title="% of games",
                      bargap=0.04, margin=dict(l=16, r=24, t=64, b=48))
    return fig


# ------------------------------------------------------------------------ trend


def trend_chart(dates: pd.Series, values: pd.Series, fit: TrendFit | None, stat_label: str,
                in_split: pd.Series | None = None, rolling: pd.Series | None = None) -> go.Figure:
    """Stat per game over time: faint dots per game, split games highlighted,
    a 10-game rolling average, and the linear trend with its 95% confidence band."""
    d = pd.to_datetime(dates)
    v = pd.to_numeric(values, errors="coerce")
    fig = go.Figure()
    mask = np.asarray(in_split, dtype=bool) if in_split is not None else np.zeros(len(v), dtype=bool)
    fig.add_trace(go.Scatter(x=d[~mask], y=v[~mask], mode="markers", name="Other games",
                             marker=dict(color=T.MUTED, size=5, opacity=0.35),
                             hovertemplate="%{x|%b %d, %Y}: %{y}<extra></extra>"))
    if mask.any():
        fig.add_trace(go.Scatter(x=d[mask], y=v[mask], mode="markers", name="Games in this split",
                                 marker=dict(color=T.SERIES[0], size=8, line=dict(color=T.SURFACE, width=2)),
                                 hovertemplate="%{x|%b %d, %Y}: %{y}<extra>split</extra>"))
    if rolling is not None:
        rx, ry = break_at_gaps(d, pd.to_numeric(rolling, errors="coerce"))
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", name="10-game average", connectgaps=False,
                                 line=dict(color=T.SERIES[1], width=2), hoverinfo="skip"))
    if fit is not None:
        fig.add_trace(go.Scatter(x=pd.concat([fit.dates, fit.dates[::-1]]),
                                 y=np.concatenate([fit.upper, fit.lower[::-1]]), fill="toself",
                                 fillcolor="rgba(195,194,183,0.14)", line=dict(width=0),
                                 name="95% confidence band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fit.dates, y=fit.fitted, mode="lines", name="Linear trend",
                                 line=dict(color=T.INK, width=2),
                                 hovertemplate="trend: %{y:.1f}<extra></extra>"))
    fig.update_layout(height=360, yaxis_title=stat_label, hovermode="closest",
                      margin=dict(l=16, r=24, t=64, b=40))
    return fig


def break_at_gaps(dates: pd.Series, values: pd.Series, max_gap_days: int = 60) -> tuple[list, list]:
    """Insert a blank point wherever consecutive games are more than ``max_gap_days`` apart,
    so a line chart does not draw a flat bridge across the off-season."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    v = pd.Series(values).reset_index(drop=True)
    xs: list = []
    ys: list = []
    for i in range(len(d)):
        if i and (d[i] - d[i - 1]).days > max_gap_days:
            xs.append(d[i - 1] + (d[i] - d[i - 1]) / 2)
            ys.append(None)
        xs.append(d[i])
        ys.append(None if pd.isna(v[i]) else float(v[i]))
    return xs, ys


# ------------------------------------------------------------------- projection


def interval_chart(mean: float, intervals: dict[float, tuple[float, float]], line: float | None,
                   stat_label: str, median: float | None = None) -> go.Figure:
    """Nested prediction intervals on a number line: 99% (widest, faintest),
    95%, 90% (narrowest, strongest), the expected value, and the betting line."""
    fig = go.Figure()
    shades = {0.99: 0.18, 0.95: 0.32, 0.90: 0.55}
    heights = {0.99: 0.36, 0.95: 0.26, 0.90: 0.16}
    for lvl in sorted(intervals, reverse=True):
        lo, hi = intervals[lvl]
        h = heights.get(round(lvl, 2), 0.2)
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=-h, y1=h, line=dict(width=0),
                      fillcolor=f"rgba(57,135,229,{shades.get(round(lvl, 2), 0.3)})")
        fig.add_annotation(x=hi, y=h, text=f"{int(round(lvl * 100))}%", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=T.MUTED, size=11), xshift=2)
    fig.add_trace(go.Scatter(x=[mean], y=[0], mode="markers", name=f"Expected {mean:.1f}",
                             marker=dict(color=T.INK, size=12, line=dict(color=T.SURFACE, width=2)),
                             hovertemplate=f"expected value {mean:.2f}<extra></extra>"))
    if median is not None and not math.isnan(median):
        fig.add_trace(go.Scatter(x=[median], y=[0], mode="markers", name=f"Median {median:.1f}",
                                 marker=dict(color=T.INK_2, size=9, symbol="diamond"),
                                 hovertemplate=f"median {median:.2f}<extra></extra>"))
    if line is not None:
        fig.add_vline(x=line, line=dict(color=T.WARN, width=2))
        fig.add_annotation(x=line, y=-0.42, text=f"line {line:g}", showarrow=False, yanchor="top",
                           xanchor="left", xshift=4, font=dict(color=T.WARN, size=13))
    lo_all = min(v[0] for v in intervals.values()) if intervals else mean
    hi_all = max(v[1] for v in intervals.values()) if intervals else mean
    if line is not None:
        lo_all, hi_all = min(lo_all, line), max(hi_all, line)
    pad = (hi_all - lo_all) * 0.08 + 0.5
    fig.update_layout(height=230, xaxis=dict(title=stat_label, range=[lo_all - pad, hi_all + pad], showgrid=True),
                      yaxis=dict(visible=False, range=[-0.75, 0.6]), margin=dict(l=16, r=24, t=40, b=40),
                      legend=dict(x=1, xanchor="right", y=1.02, yanchor="bottom"))
    return fig


def probability_curve(lines: np.ndarray, p_over: np.ndarray, current_line: float | None, stat_label: str) -> go.Figure:
    """P(over) for every line from low to high, with the chosen line marked."""
    fig = go.Figure(go.Scatter(x=lines, y=np.asarray(p_over) * 100, mode="lines", name="P(over)",
                               line=dict(color=T.SERIES[0], width=2, shape="hv"),
                               fill="tozeroy", fillcolor="rgba(57,135,229,0.10)",
                               hovertemplate="line %{x:g}: P(over) %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line=dict(color=T.AXIS, width=1))
    if current_line is not None:
        p = float(np.interp(current_line, lines, p_over)) * 100
        fig.add_trace(go.Scatter(x=[current_line], y=[p], mode="markers+text", text=[f"{p:.0f}%"],
                                 textposition="top right", textfont=dict(color=T.INK, size=13),
                                 marker=dict(color=T.WARN, size=11, line=dict(color=T.SURFACE, width=2)),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=300, showlegend=False, xaxis_title=f"{stat_label} line",
                      yaxis=dict(title="P(over)", range=[0, 102], ticksuffix="%"),
                      margin=dict(l=16, r=24, t=32, b=40))
    return fig


# --------------------------------------------------------- league-wide variable effect


def coefficient_chart(df: pd.DataFrame, stat_label: str, reference: str | None = None,
                      label_col: str = "label", est_col: str = "estimate", lo_col: str = "lo",
                      hi_col: str = "hi") -> go.Figure:
    """Dot-and-whisker chart of regression effects (one row per variable level),
    each relative to ``reference`` (the level every other level is compared with)."""
    rows = [
        EffectRow(str(r[label_col]), float(r[est_col]), float(r[lo_col]), float(r[hi_col]),
                  int(r["n"]) if "n" in df and pd.notna(r["n"]) else 0,
                  float(r["p_value"]) if "p_value" in df and pd.notna(r["p_value"]) else math.nan)
        for _, r in df.iterrows()
    ]
    fig = effect_chart(rows, 0.0, stat_label, digits=2, show_n=any(r.n for r in rows))
    fig.layout.annotations = fig.layout.annotations[:-1]  # drop the "baseline" label; keep value labels
    if reference:
        fig.add_annotation(x=0, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"vs {reference}", font=dict(color=T.INK_2, size=12))
    fig.update_layout(xaxis_title=f"Change in {stat_label} for a typical player")
    return fig


def calibration_chart(predicted: Sequence[float], observed: Sequence[float], counts: Sequence[int] | None = None) -> go.Figure:
    """Predicted P(over) per bucket vs the share that actually went over.
    Points on the diagonal mean the probabilities can be taken at face value."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             line=dict(color=T.MUTED, width=1)))
    size = [8 + 10 * (c / max(counts)) for c in counts] if counts else 10
    fig.add_trace(go.Scatter(x=np.asarray(predicted) * 100, y=np.asarray(observed) * 100,
                             mode="lines+markers", name="Model",
                             line=dict(color=T.SERIES[0], width=2),
                             marker=dict(size=size, color=T.SERIES[0], line=dict(color=T.SURFACE, width=2)),
                             hovertemplate="predicted %{x:.0f}% → actual %{y:.0f}%<extra></extra>"))
    fig.update_layout(height=360, xaxis=dict(title="Predicted P(over)", ticksuffix="%", range=[0, 100]),
                      yaxis=dict(title="Actual over rate", ticksuffix="%", range=[0, 100]))
    return fig
