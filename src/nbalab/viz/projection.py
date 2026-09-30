"""Plotly builders for the projection section. They draw what ``nbalab.models``
computed and compute nothing themselves (the trend bands come from
``nbalab.models.trend``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nbalab.viz import theme as T
from nbalab.viz.charts import calibration_chart

__all__ = ["outcome_chart", "contribution_waterfall", "trend_chart", "calibration_chart"]

T.register()


def outcome_chart(support: np.ndarray, pmf: np.ndarray, pi: dict[int, tuple[float, float]], line: float | None,
                  stat_label: str, mean: float | None = None) -> go.Figure:
    """Probability of every possible outcome. Bars inside the 90% prediction interval are
    solid, the 90-99% tails lighter, and bars above the line are marked as "over"."""
    support = np.asarray(support)
    pmf = np.asarray(pmf)
    lo99, hi99 = pi.get(99, (support[0], support[-1]))
    keep = (support >= lo99 - 2) & (support <= hi99 + 2)
    x, y = support[keep], pmf[keep] * 100
    lo90, hi90 = pi.get(90, (lo99, hi99))
    inside = (x >= lo90) & (x <= hi90)
    color = np.where(inside, T.SERIES[0], "rgba(57,135,229,0.35)")
    if line is not None:
        color = np.where(x > line, np.where(inside, T.SERIES[2], "rgba(25,158,112,0.35)"), color)
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=list(color), name="P(outcome)",
                           hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    if line is not None:
        fig.add_vline(x=line, line=dict(color=T.WARN, width=2))
        fig.add_annotation(x=line, y=1, yref="paper", text=f"line {line:g}", showarrow=False, xanchor="left",
                           xshift=4, font=dict(color=T.WARN, size=12))
    if mean is not None:
        fig.add_vline(x=mean, line=dict(color=T.INK_2, width=1, dash="dot"))
    fig.update_layout(height=300, showlegend=False, bargap=0.08, xaxis_title=stat_label,
                      yaxis=dict(title="Probability", ticksuffix="%"), margin=dict(l=16, r=24, t=32, b=40))
    return fig


def contribution_waterfall(labels: Sequence[str], values: Sequence[float], neutral: float, stat_label: str,
                           digits: int = 1) -> go.Figure:
    """Neutral-game projection, each context group's push up or down, and the final projection."""
    vals = [float(v) for v in values]
    total = neutral + sum(vals)
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", *["relative"] * len(vals), "total"],
        x=["Neutral game", *labels, "Projection"], y=[neutral, *vals, total],
        text=[f"{neutral:.{digits}f}", *[f"{v:+.{digits}f}" for v in vals], f"{total:.{digits}f}"],
        textposition="outside", connector=dict(line=dict(color=T.AXIS, width=1)),
        increasing=dict(marker=dict(color=T.UP)), decreasing=dict(marker=dict(color=T.DOWN)),
        totals=dict(marker=dict(color=T.SERIES[0])),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    lo = min(neutral, total, *(neutral + np.cumsum(vals))) if vals else neutral
    hi = max(neutral, total, *(neutral + np.cumsum(vals))) if vals else neutral
    pad = (hi - lo) * 0.6 + 0.5
    fig.update_layout(height=340, showlegend=False, yaxis=dict(title=stat_label, range=[max(lo - pad, 0), hi + pad]),
                      margin=dict(l=16, r=24, t=32, b=60))
    return fig


def trend_chart(dates: pd.Series, values: pd.Series, bands: pd.DataFrame | None, stat_label: str,
                projection: tuple[float, float, float] | None = None) -> go.Figure:
    """Game-by-game values with the OLS trend, its 95% CI band (the average) and the wider
    95% PI band (single games), extended into the future. ``projection`` = (mean, pi_lo, pi_hi)
    for the next game from the model, drawn after the last game."""
    fig = go.Figure()
    if bands is not None:
        x = list(bands["date"]) + list(bands["date"][::-1])
        fig.add_trace(go.Scatter(x=x, y=list(bands["pi_hi"]) + list(bands["pi_lo"][::-1]), fill="toself",
                                 fillcolor="rgba(57,135,229,0.10)", line=dict(width=0), name="95% range for one game",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=list(bands["ci_hi"]) + list(bands["ci_lo"][::-1]), fill="toself",
                                 fillcolor="rgba(57,135,229,0.30)", line=dict(width=0), name="95% CI of the trend",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bands["date"], y=bands["fitted"], mode="lines", name="Trend",
                                 line=dict(color=T.SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=dates, y=values, mode="markers", name=stat_label,
                             marker=dict(color=T.INK_2, size=5, opacity=0.7),
                             hovertemplate="%{x|%b %d, %Y}: %{y}<extra></extra>"))
    if projection is not None and len(dates):
        nxt = pd.to_datetime(pd.Series(dates)).max() + pd.Timedelta(days=2)
        mean, lo, hi = projection
        fig.add_trace(go.Scatter(x=[nxt], y=[mean], mode="markers", name="Model projection (95% PI)",
                                 marker=dict(color=T.WARN, size=11, symbol="diamond"),
                                 error_y=dict(type="data", symmetric=False, array=[hi - mean], arrayminus=[mean - lo],
                                              color=T.WARN, thickness=2)))
    fig.update_layout(height=340, yaxis_title=stat_label, margin=dict(l=16, r=24, t=32, b=40),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return fig
