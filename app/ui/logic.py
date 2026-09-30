"""Presentation logic that sits between the engines and the page: confidence
badges and the per-filter effects behind the "how each variable affects this
stat" chart. No Streamlit imports, so everything here is unit tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nbalab.query import schema as s
from nbalab.query.data import QueryData
from nbalab.query.engine import SplitResult, split_for_subject
from nbalab.query.filters import describe_filter
from nbalab.viz.charts import EffectRow

HIGH_N, MEDIUM_N = 25, 10
HIGH_P, MEDIUM_P = 0.05, 0.20
MIN_N_FOR_INTERVAL = 5  # below this a bootstrap interval is falsely precise, so none is shown


@dataclass(frozen=True)
class Badge:
    level: str  # "high" | "medium" | "low"
    reason: str


def confidence_badge(n: int, p_value: float) -> Badge:
    """How much to trust that the split really differs from the baseline.

    - **high**: at least 25 games and p < 0.05. Enough games, and a gap that
      would be unusual if the split were just a random slice of the baseline.
    - **low**: fewer than 10 games, no p-value, or p >= 0.20. Either too few
      games to tell, or a gap that luck explains easily.
    - **medium**: everything in between.

    A "low" badge on a large sample means "no real difference found", not
    "bad data".
    """
    if n < MEDIUM_N:
        return Badge("low", f"only {n} games")
    if p_value is None or math.isnan(p_value):
        return Badge("low", "no significance test possible")
    if p_value >= MEDIUM_P:
        return Badge("low", f"gap is within normal noise (p = {p_value:.2f})")
    if n >= HIGH_N and p_value < HIGH_P:
        return Badge("high", f"{n} games, p = {p_value:.3f}")
    if n < HIGH_N:
        return Badge("medium", f"a real-looking gap but only {n} games (p = {p_value:.3f})")
    return Badge("medium", f"suggestive gap (p = {p_value:.2f})")


def single_filter_queries(query: s.StatQuery) -> list[tuple[s.Filter, s.StatQuery]]:
    """One split query per non-scope filter, each keeping the season scope.

    Running each filter alone shows its own effect vs. the baseline, before
    other filters are stacked on top.
    """
    q = query.with_projection_context() if query.mode == "projection" else query
    out = []
    for f in q.split_filters:
        single = q.model_copy(update={"filters": [*q.scope_filters, f], "mode": "split",
                                      "group_by": None, "projection": None})
        out.append((f, single))
    return out


def effect_rows(result: SplitResult, stat: str, data: QueryData) -> list[EffectRow]:
    """Each filter's individual effect on ``stat`` plus the combined effect.

    The combined row is the engine's own split (all filters stacked). It is
    left out when there is only one filter, because it would repeat that bar.
    """
    rows: list[EffectRow] = []
    pairs = single_filter_queries(result.query)
    for f, single in pairs:
        r = split_for_subject(single, result.subject_id, data)
        st = r.stats[stat]
        lo, hi = (st.ci_low, st.ci_high) if st.split.n >= MIN_N_FOR_INTERVAL else (math.nan, math.nan)
        rows.append(EffectRow(describe_filter(f, data), st.diff, lo, hi, st.split.n, st.p_value))
    if len(pairs) >= 2:
        st = result.stats[stat]
        lo, hi = (st.ci_low, st.ci_high) if st.split.n >= MIN_N_FOR_INTERVAL else (math.nan, math.nan)
        rows.append(EffectRow("All filters combined", st.diff, lo, hi, st.split.n, st.p_value, combined=True))
    return rows


def format_value(x: float | None, is_ratio: bool = False, digits: int = 1) -> str:
    """Display format: '27.4', '48.1%', or an en dash for missing."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}%" if is_ratio else f"{x:.{digits}f}"
