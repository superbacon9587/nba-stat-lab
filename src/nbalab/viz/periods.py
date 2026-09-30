"""Charts for period comparisons (before vs after a split point). They draw what
:mod:`nbalab.query.periods` computed and compute nothing themselves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nbalab.viz import theme as T
from nbalab.viz.teams import accent

T.register()


def dumbbell_chart(table: pd.DataFrame, stat_label: str, abbrevs: dict[int, str | None],
                   higher_is_better: bool = True) -> go.Figure:
    """One row per team: before dot (hollow), after dot (team color), a line between them.
    Rows are sorted by change, biggest improvement on top."""
    t = table.sort_values("change", ascending=higher_is_better).reset_index(drop=True)
    fig = go.Figure()
    for r in t.itertuples():
        color = accent(abbrevs.get(int(r.teamId)))
        fig.add_trace(go.Scatter(x=[r.before, r.after], y=[r.team, r.team], mode="lines",
                                 line=dict(color=color, width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=t["before"], y=t["team"], mode="markers", name="Before",
                             marker=dict(size=10, color=T.SURFACE, line=dict(color=T.INK_2, width=2)),
                             customdata=t["teamId"], hovertemplate="%{y}<br>before %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t["after"], y=t["team"], mode="markers", name="After",
                             marker=dict(size=12, color=[accent(abbrevs.get(int(i))) for i in t["teamId"]],
                                         line=dict(color=T.INK, width=1)),
                             customdata=t["teamId"], text=[f"{c:+.2f}" for c in t["change"]],
                             hovertemplate="%{y}<br>after %{x:.1f} (change %{text})<extra></extra>"))
    fig.update_layout(height=max(360, 26 * len(t) + 80), xaxis_title=stat_label,
                      yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), margin=dict(l=16, r=24, t=40, b=40))
    return fig


def change_bars(table: pd.DataFrame, col: str, lo: str, hi: str, stat_label: str, abbrevs: dict[int, str | None],
                higher_is_better: bool = True, verdict_col: str | None = None) -> go.Figure:
    """Diverging bars of the change with 95% whiskers; faded bars could be noise (per the verdict column)."""
    t = table.sort_values(col, ascending=higher_is_better).reset_index(drop=True)
    faded = (t[verdict_col] != "likely real") if verdict_col else pd.Series(False, index=t.index)
    colors = [accent(abbrevs.get(int(i))) for i in t["teamId"]]
    fig = go.Figure(go.Bar(
        x=t[col], y=t["team"], orientation="h", customdata=t["teamId"],
        marker=dict(color=colors, opacity=np.where(faded, 0.4, 1.0), line=dict(width=0)),
        error_x=dict(type="data", symmetric=False, array=(t[hi] - t[col]).clip(lower=0),
                     arrayminus=(t[col] - t[lo]).clip(lower=0), color=T.INK_2, thickness=1.2, width=4),
        hovertemplate="%{y}<br>change %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=T.INK_2, width=1))
    fig.update_layout(height=max(360, 26 * len(t) + 80), xaxis_title=f"Change in {stat_label} (after - before)",
                      yaxis=dict(autorange="reversed", tickfont=dict(size=12)), showlegend=False,
                      margin=dict(l=16, r=24, t=24, b=40))
    return fig


def monthly_trend_chart(monthly: pd.DataFrame, stat_label: str, subject: str) -> go.Figure:
    """Month-by-month average with a 95% band, the league curve scaled to this subject's level,
    and the playoffs as a separate point."""
    reg = monthly[monthly["month"] != "Playoffs"]
    post = monthly[monthly["month"] == "Playoffs"]
    fig = go.Figure()
    x = list(reg["month"]) + list(reg["month"][::-1])
    fig.add_trace(go.Scatter(x=x, y=list(reg["ci_high"]) + list(reg["ci_low"][::-1]), fill="toself",
                             fillcolor="rgba(57,135,229,0.18)", line=dict(width=0), name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=reg["month"], y=reg["value"], mode="lines+markers", name=subject,
                             line=dict(color=T.SERIES[0], width=2.5), customdata=reg["games"],
                             hovertemplate="%{x}: %{y:.1f} (%{customdata} games)<extra></extra>"))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["league_scaled"], mode="lines", name="League trend (scaled)",
                             line=dict(color=T.MUTED, width=1.5, dash="dash"),
                             hovertemplate="league pattern at this level: %{y:.1f}<extra></extra>"))
    if len(post):
        p = post.iloc[0]
        fig.add_trace(go.Scatter(x=["Playoffs"], y=[p["value"]], mode="markers", name="Playoffs",
                                 marker=dict(size=12, color=T.WARN, symbol="diamond"),
                                 error_y=dict(type="data", symmetric=False, array=[p["ci_high"] - p["value"]],
                                              arrayminus=[p["value"] - p["ci_low"]], color=T.WARN),
                                 hovertemplate=f"Playoffs: %{{y:.1f}} ({int(p['games'])} games)<extra></extra>"))
    fig.update_layout(height=340, yaxis_title=stat_label, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      margin=dict(l=16, r=24, t=40, b=40))
    return fig


def seasonal_change_chart(seasonal: pd.DataFrame, stat_label: str, higher_is_better: bool = True,
                          col: str = "diff") -> go.Figure:
    """The change after the split, season by season, with 95% whiskers: consistent pattern or one-off?"""
    t = seasonal.sort_values("season")
    labels = [f"{s}-{str(s + 1)[-2:]}" for s in t["season"]]
    better = (t[col] > 0) == higher_is_better
    fig = go.Figure(go.Bar(
        x=labels, y=t[col], marker=dict(color=np.where(better, T.UP, T.DOWN), line=dict(width=0)),
        error_y=dict(type="data", array=1.96 * t["se"].fillna(0), color=T.INK_2, thickness=1.2, width=4),
        customdata=np.stack([t["n_before"], t["n_after"]], axis=1),
        hovertemplate="%{x}: %{y:+.2f} (%{customdata[0]} games before, %{customdata[1]} after)<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=T.INK_2, width=1))
    fig.update_layout(height=300, yaxis_title=f"Change in {stat_label}", showlegend=False,
                      margin=dict(l=16, r=24, t=24, b=40))
    return fig
