"""Plain-text rendering of engine results for the CLI."""

from __future__ import annotations

import math

import pandas as pd

from nbalab.query.engine import SplitResult, StatResult
from nbalab.query.variable_effect import VariableEffectResult


def num(x: float | None, digits: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def pval(p: float) -> str:
    if p is None or math.isnan(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def verdict(r: StatResult) -> str:
    """Plain-English read of significance (p < 0.05 and CI excluding 0)."""
    if math.isnan(r.p_value):
        return "not testable"
    if r.p_value < 0.05 and not (r.ci_low <= 0 <= r.ci_high):
        return "statistically significant"
    return "not statistically significant"


def season_span(seasons: tuple[int, int] | None) -> str:
    if not seasons:
        return "no seasons"
    a, b = seasons
    return f"{a}-{str(a + 1)[-2:]} to {b}-{str(b + 1)[-2:]}"


def render_split(r: SplitResult) -> str:
    lines = []
    filt = ", ".join(r.filters_described) or "no filters"
    lines.append(f"== {r.subject_name}: {filt}")
    lines.append(f"   seasons {season_span(r.seasons)} | split n={r.n_games} games | baseline n={r.n_baseline} "
                 f"| confidence: {r.confidence.upper()}")
    only_grouped = r.groups is not None and r.n_games == r.n_baseline and not r.query.split_filters
    for s in ([] if only_grouped else r.stats.values()):
        kind = "pooled %" if s.is_ratio else "per game"
        lines.append(f"\n   {s.label} ({kind})")
        lines.append(f"     split     mean {num(s.split.value)}  median {num(s.split.median, 1)}  "
                     f"std {num(s.split.std)}  per36 {num(s.split.per36)}")
        lines.append(f"     baseline  mean {num(s.baseline.value)}  median {num(s.baseline.median, 1)}  "
                     f"std {num(s.baseline.std)}  per36 {num(s.baseline.per36)}")
        lines.append(f"     difference {num(s.diff, signed=True)} ({num(s.diff_pct, 1, signed=True)}%)  "
                     f"z={num(s.z_score)}  Welch p={pval(s.p_value)}  "
                     f"95% CI [{num(s.ci_low, signed=True)}, {num(s.ci_high, signed=True)}]  -> {verdict(s)}")
        lines.append(f"     shrunk estimate {num(s.shrunk)} (raw {num(s.split.value)}, "
                     f"weight on split data {num(s.shrink_weight)})")
        if s.hit_rates:
            h = s.hit_rates
            lines.append(f"     line: over {num(100 * h['split']['over'], 0)}% in split, "
                         f"{num(100 * h['baseline']['over'], 0)}% in baseline")
    if only_grouped:
        for s in r.stats.values():
            lines.append(f"   overall {s.label}: {num(s.baseline.value)} (median {num(s.baseline.median, 1)}, "
                         f"std {num(s.baseline.std)}, per36 {num(s.baseline.per36)})")
    if r.groups is not None and not r.groups.table.empty:
        lines.append(f"\n   By {r.groups.variable} (diff vs overall; p_holm corrects for testing every level):")
        t = r.groups.table.rename(columns={"p_value_holm": "p_holm"})
        cols = ["level", "stat", "n", "value", "shrunk", "diff", "p_value", "p_holm", "ci_low", "ci_high",
                "confidence"]
        lines.extend("     " + ln for ln in _table(t[cols]).splitlines())
    if r.defender_note:
        lines.append(f"\n   {r.defender_note}")
    for note in r.notes:
        lines.append(f"   Note: {note}")
    for w in r.warnings:
        lines.append(f"   ! {w}")
    return "\n".join(lines)


def render_effect(r: VariableEffectResult) -> str:
    lines = [f"== Effect of {r.variable} ({r.unit}) on a typical {r.subject_type}",
             f"   {r.method}", f"   population: {', '.join(r.population)}",
             f"   controls: {', '.join(r.controls) or 'none'}"]
    for stat, t in r.per_stat.items():
        lines.append(f"\n   {stat}: n={t.n_obs:,} games, {t.n_subjects:,} {r.subject_type}s, "
                     f"mean {t.stat_mean:.2f}, within R^2 {t.r2_within:.3f}")
        lines.extend("     " + ln for ln in _table(t.coefficients.drop(columns=["is_variable"])).splitlines())
    for n in r.notes:
        lines.append(f"   Note: {n}")
    return "\n".join(lines)


def _table(df: pd.DataFrame) -> str:
    def fmt(x: object) -> str:
        if isinstance(x, float):
            return "n/a" if math.isnan(x) else f"{x:.3f}" if abs(x) < 1 else f"{x:.2f}"
        return str(x)

    return df.to_string(index=False, formatters={c: fmt for c in df.columns})
