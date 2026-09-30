"""Split engine: a subject's stats under the query's filters vs. their own baseline.

For each subject in a :class:`StatQuery`:

1. **Baseline**: the subject's games in the query's season scope
   (``season_range`` / ``last_n_seasons``), with no other filter.
2. **Split**: the baseline games that pass every other filter (AND).
3. For each stat: mean, median, std, n, per-36 rate for both; the difference
   (raw, %, z-score); a Welch t-test p-value and a 95% bootstrap CI for the
   difference; and an empirical Bayes *shrunk* estimate that pulls small
   samples toward the baseline (see :mod:`nbalab.query.inference`).
4. Sample-size labels: fewer than 10 games gets a warning. Fewer than 25 is
   "low confidence". When stacked filters leave fewer than 5 games, the result
   is still returned but flagged, with a suggestion of which filter to drop.
5. With ``group_by``, the same numbers for every value of the variable.

The shrinkage needs tau^2, how much true averages really differ between
splits. With ``group_by`` it is estimated from those groups. Otherwise it is
estimated from the subject's per-opponent averages in the baseline, the most
common kind of split and a stable reference for how much a player's output
moves with context.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from nbalab.query import inference as inf
from nbalab.query import schema as s
from nbalab.query.data import QueryData, load_query_data
from nbalab.query.defenders import SOURCE_LABELS, shared_floor_summary, summarize_sources
from nbalab.query.filters import FilterImpact, apply_filters, describe_filter, scope_mask
from nbalab.query.frames import subject_frame
from nbalab.query.stats import GROUPABLE_VARIABLES, StatDef, stat_catalog

SMALL_SAMPLE_WARNING_N = 10
LOW_CONFIDENCE_N = 25
STACKED_FILTER_MIN_N = 5
MIN_GROUP_N_FOR_PRIOR = 3


# ---------------------------------------------------------------------- results


@dataclass
class SampleStats:
    """Descriptive numbers for one stat over one set of games.

    ``value`` is the headline: the mean for counting stats, the pooled
    made/attempted ratio for percentage stats.
    """

    n: int
    mean: float
    median: float
    std: float
    value: float
    per36: float | None


@dataclass
class StatResult:
    """Split vs. baseline for one stat."""

    stat: str
    label: str
    is_ratio: bool
    split: SampleStats
    baseline: SampleStats
    diff: float
    diff_pct: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    shrunk: float
    shrink_weight: float
    hit_rates: dict[str, dict[str, float]] | None = None


@dataclass
class GroupResult:
    """One row per (level of the group_by variable, stat)."""

    variable: str
    table: pd.DataFrame


@dataclass
class SplitResult:
    """Everything the UI needs for one subject."""

    subject_type: str
    subject_id: int
    subject_name: str
    query: s.StatQuery
    games: pd.DataFrame
    baseline_games: pd.DataFrame
    stats: dict[str, StatResult]
    n_games: int
    n_baseline: int
    confidence: str
    warnings: list[str] = field(default_factory=list)
    stacked_filter_flag: bool = False
    suggestion: str | None = None
    filter_impacts: list[FilterImpact] = field(default_factory=list)
    filters_described: list[str] = field(default_factory=list)
    defender_sources: dict[str, int] = field(default_factory=dict)
    defender_note: str | None = None
    career_note: str | None = None
    groups: GroupResult | None = None
    seasons: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)
    shared_floor: dict[str, float] | None = None

    def to_dict(self, include_games: bool = False) -> dict:
        """JSON-friendly dict (games tables left out unless asked)."""
        out = {
            k: v for k, v in asdict(self).items()
            if k not in {"games", "baseline_games", "query", "groups", "stats", "filter_impacts"}
        }
        out["query"] = self.query.model_dump()
        out["stats"] = {k: asdict(v) for k, v in self.stats.items()}
        out["filter_impacts"] = [asdict(i) for i in self.filter_impacts]
        if self.groups is not None:
            out["groups"] = {"variable": self.groups.variable,
                             "table": self.groups.table.to_dict(orient="records")}
        if include_games:
            out["games"] = self.games.to_dict(orient="records")
        return out


# ----------------------------------------------------------------------- engine


def run_query(query: s.StatQuery, data: QueryData | None = None):
    """Dispatch on ``query.mode``. Split and projection return ``list[SplitResult]``
    (one per subject). variable_effect returns a ``VariableEffectResult``."""
    data = data or load_query_data()
    if query.mode == "variable_effect":
        from nbalab.query.variable_effect import run_variable_effect

        return run_variable_effect(query, data)
    return run_split(query, data)


def run_split(query: s.StatQuery, data: QueryData | None = None) -> list[SplitResult]:
    """One :class:`SplitResult` per subject id."""
    data = data or load_query_data()
    if query.mode == "projection":
        query = query.with_projection_context()
    return [split_for_subject(query, sid, data) for sid in query.subject_ids]


def split_for_subject(query: s.StatQuery, subject_id: int, data: QueryData) -> SplitResult:
    frame = subject_frame(data, query.subject_type, subject_id)
    baseline = frame[scope_mask(query.scope_filters, frame, data.latest_season)]
    outcome = apply_filters(query.split_filters, baseline, data, query.subject_type)
    split = baseline[outcome.mask]
    rest = baseline[~outcome.mask]

    catalog = stat_catalog(query.subject_type)
    lines = query.projection.lines if query.projection else {}
    stats: dict[str, StatResult] = {}
    for name in query.stats:
        tau2 = prior_for(catalog[name], baseline, query.group_by)
        stats[name] = compare(catalog[name], split, rest, baseline, tau2, line=lines.get(name))

    if query.subject_type == "team":  # the name the franchise had in the latest season shown (Sonics vs Thunder)
        name = data.team_name(subject_id, int(baseline["season"].max()) if len(baseline) else None)
    else:
        name = data.player_name(subject_id)
    result = SplitResult(
        subject_type=query.subject_type,
        subject_id=subject_id,
        subject_name=name,
        query=query,
        games=split,
        baseline_games=baseline,
        stats=stats,
        n_games=len(split),
        n_baseline=len(baseline),
        confidence=confidence_label(len(split)),
        filter_impacts=outcome.impacts,
        filters_described=[describe_filter(f, data) for f in query.filters],
        seasons=(int(baseline["season"].min()), int(baseline["season"].max())) if len(baseline) else None,
    )
    add_sample_warnings(result, query)
    add_defender_info(result, query, outcome.defender_source, split)
    relabel_named_defenders(result, query, data)
    add_shared_floor(result, query, data)
    add_data_start_note(result, data)
    if query.group_by:
        result.groups = group_split(query, split, baseline, catalog)
    if query.mode == "projection":
        result.notes.append(
            "Historical view of the projection context. The forward-looking projection "
            "(expected value, prediction intervals, over/under probabilities) comes from nbalab.models."
        )
    return result


def compare(
    stat: StatDef,
    split: pd.DataFrame,
    rest: pd.DataFrame,
    baseline: pd.DataFrame,
    tau2: float,
    line: float | None = None,
) -> StatResult:
    """All split-vs-baseline numbers for one stat."""
    sp, bl = sample_stats(stat, split), sample_stats(stat, baseline)
    diff, pct = inf.difference(sp.value, bl.value)
    split_v, rest_v = stat.per_game(split).to_numpy(), stat.per_game(rest).to_numpy()
    if stat.is_ratio:
        ci = inf.bootstrap_diff_ci(
            stat.value(split).to_numpy(), stat.value(rest).to_numpy(),
            split_den=stat.denominator(split).to_numpy(), rest_den=stat.denominator(rest).to_numpy(),
        )
        ci = (ci[0] * stat.scale, ci[1] * stat.scale)
    else:
        ci = inf.bootstrap_diff_ci(split_v, rest_v)
    weight = inf.shrinkage_weight(sp.n, bl.std**2 if bl.n > 1 else math.nan, tau2)
    return StatResult(
        stat=stat.name,
        label=stat.label,
        is_ratio=stat.is_ratio,
        split=sp,
        baseline=bl,
        diff=diff,
        diff_pct=pct,
        z_score=inf.z_score(sp.value, bl.value, bl.std, sp.n),
        p_value=inf.welch_p_value(split_v, rest_v),
        ci_low=ci[0],
        ci_high=ci[1],
        shrunk=inf.shrink(sp.value, bl.value, weight) if sp.n else math.nan,
        shrink_weight=weight,
        hit_rates=hit_rates(stat, split, baseline, line) if line is not None else None,
    )


def sample_stats(stat: StatDef, games: pd.DataFrame) -> SampleStats:
    """Mean/median/std/n of per-game values, the headline value, and the per-36 rate."""
    summ = inf.summarize(stat.per_game(games).to_numpy())
    rate = None
    if stat.per36 and "minutes" in games:
        rate = inf.per36(float(stat.per_game(games).sum()), float(games["minutes"].astype("float64").sum()))
    return SampleStats(summ.n, summ.mean, summ.median, summ.std, stat.pooled(games) if len(games) else math.nan, rate)


def prior_for(stat: StatDef, baseline: pd.DataFrame, group_by: str | None) -> float:
    """tau^2 for shrinkage: from the group_by groups if given, else per-opponent averages."""
    column = GROUPABLE_VARIABLES[group_by] if group_by else "opponentTeamId"
    if column not in baseline or len(baseline) < 2:
        return math.nan
    values = stat.per_game(baseline)
    groups = values.groupby(baseline[column].astype("object"), observed=True).agg(["mean", "count"])
    groups = groups[groups["count"] >= 1]
    return inf.prior_variance(groups["mean"].to_numpy(), groups["count"].to_numpy(), float(values.var(ddof=1)))


def hit_rates(stat: StatDef, split: pd.DataFrame, baseline: pd.DataFrame, line: float) -> dict[str, dict[str, float]]:
    """Share of past games over / under / exactly on a betting line, in the split and baseline."""
    out = {}
    for label, games in (("split", split), ("baseline", baseline)):
        v = stat.per_game(games).dropna()
        n = len(v)
        out[label] = {
            "n": float(n),
            "over": float((v > line).mean()) if n else math.nan,
            "under": float((v < line).mean()) if n else math.nan,
            "push": float((v == line).mean()) if n else math.nan,
        }
    return out


def confidence_label(n: int) -> str:
    """"high" for 25+ games, "low" for 10-24, "very low" under 10."""
    if n >= LOW_CONFIDENCE_N:
        return "high"
    if n >= SMALL_SAMPLE_WARNING_N:
        return "low"
    return "very low"


def add_sample_warnings(result: SplitResult, query: s.StatQuery) -> None:
    """Sample-size warning, low-confidence label, and the stacked-filter flag plus suggestion."""
    n = result.n_games
    if n == 0:
        result.warnings.append("No games match these filters.")
    elif n < SMALL_SAMPLE_WARNING_N:
        result.warnings.append(f"Sample size warning: only {n} games. Treat the raw split as anecdotal; "
                               "the shrunk estimate is the better guess.")
    elif n < LOW_CONFIDENCE_N:
        result.warnings.append(f"Low confidence: {n} games (fewer than {LOW_CONFIDENCE_N}).")

    if n < STACKED_FILTER_MIN_N and len(result.filter_impacts) >= 2:
        result.stacked_filter_flag = True
        drop, enough = suggest_drop(result.filter_impacts)
        result.suggestion = (
            f"Stacked filters leave only {n} games. Consider dropping "
            + (f"the least important filter that fixes this, '{drop.description}'"
               if enough else f"'{drop.description}' (no single filter gets back to "
               f"{STACKED_FILTER_MIN_N} games; this one recovers the most)")
            + f", which would leave {drop.n_if_dropped} games."
        )
        result.warnings.append(result.suggestion)


def suggest_drop(impacts: list[FilterImpact]) -> tuple[FilterImpact, bool]:
    """Which filter to suggest dropping when the stack leaves too few games.

    Prefer filters whose removal alone gets back to at least
    ``STACKED_FILTER_MIN_N`` games, and among those the least important one.
    If no single filter does that, suggest the one that recovers the most games
    (least important on ties). Returns (filter, whether it reaches the minimum).
    """
    enough = [i for i in impacts if i.n_if_dropped >= STACKED_FILTER_MIN_N]
    if enough:
        return min(enough, key=lambda i: (i.importance, -i.n_if_dropped)), True
    return max(impacts, key=lambda i: (i.n_if_dropped, -i.importance)), False


def add_defender_info(result: SplitResult, query: s.StatQuery, decided_by: pd.Series | None, split: pd.DataFrame) -> None:
    """Record which defender data source(s) answered the defender part of the question."""
    has_def_filter = any(f.type in s.DEFENDER_FILTER_TYPES for f in query.filters)
    grouped_by_def = bool(query.group_by and query.group_by.startswith("defender_"))
    if not (has_def_filter or grouped_by_def) or query.subject_type != "player":
        return
    source = decided_by if decided_by is not None else result.baseline_games.get("def_source")
    if source is None:
        return
    in_split = source.loc[source.index.intersection(split.index)]
    counts = summarize_sources(in_split if len(in_split) else source)
    result.defender_sources = counts
    parts = [f"{SOURCE_LABELS[k]} for {v} game{'s' if v != 1 else ''}" for k, v in counts.items()]
    note = "Defender data: " + "; ".join(parts) + "."
    if "official_matchups" not in counts:
        note += " No official matchup data covers these games, so this is not a true 'guarded by' split."
    result.defender_note = note


PROXY_WORDING: dict[str, str] = {
    "shared_floor_time": "games where {subject} and {other} shared the floor",
    "same_game": "games where {other} played for the opponent",
}


def relabel_named_defenders(result: SplitResult, query: s.StatQuery, data: QueryData) -> None:
    """Describe a named-defender filter by what the data really measured.

    "defended by Y" is only true for official matchup data. When most games
    came from a proxy, say what the proxy is instead.
    """
    counts = result.defender_sources
    if not counts:
        return
    main = max(counts, key=counts.get)
    if main not in PROXY_WORDING:
        return
    for i, f in enumerate(query.filters):
        if isinstance(f, s.DefenderPlayerFilter):
            result.filters_described[i] = PROXY_WORDING[main].format(
                subject=result.subject_name, other=data.player_name(f.person_id))


def career_started_before_data(person_id: int, data: QueryData) -> str | None:
    """A short reason if the player's career began before the first season in the data, else None.

    Drafted players: drafted before that season. Undrafted: already 24+ in their
    first game here (rookies are rarely that old). Returns e.g. "drafted in 1985".
    """
    pl = data.players.loc[data.players["personId"].eq(person_id)]
    pg = data.player_games
    first_data = int(pg["season"].min())
    if pl.empty or int(pl["first_season"].iloc[0]) > first_data:
        return None
    draft = pl["draftYear"].iloc[0] if "draftYear" in pl else None
    if draft is not None and pd.notna(draft) and 0 < int(draft) < first_data:
        return f"drafted in {int(draft)}"
    ages = pg.loc[pg["personId"].eq(person_id), "age"]
    if (draft is None or pd.isna(draft) or int(draft) <= 0) and len(ages) and float(ages.min()) >= 24:
        return f"already {float(ages.min()):.0f} at the first game in the data"
    return None


def add_data_start_note(result: SplitResult, data: QueryData) -> None:
    """Flag careers that started before the dataset does: their early seasons are missing."""
    if result.subject_type != "player":
        return
    why = career_started_before_data(result.subject_id, data)
    if why:
        first = int(data.player_games["season"].min())
        result.career_note = (f"{result.subject_name}'s career began before {first}-{str(first + 1)[-2:]} ({why}). "
                              f"Stats before {first}-{str(first + 1)[-2:]} are not included, so career numbers "
                              "here cover only part of the career.")
        result.notes.append(result.career_note)


def add_shared_floor(result: SplitResult, query: s.StatQuery, data: QueryData) -> None:
    """For a named defender / opponent, add per-36 rates over only the minutes both were on the floor."""
    if query.subject_type != "player":
        return
    named = [f for f in query.filters if isinstance(f, (s.DefenderPlayerFilter, s.OpponentPlayerOnCourtFilter))]
    if not named:
        return
    summary = shared_floor_summary(result.games, named[0].person_id, data)
    if summary is not None:
        result.shared_floor = summary
        result.notes.append(
            f"While {data.player_name(named[0].person_id)} was on the floor ({summary['shared_minutes']:.0f} shared "
            f"minutes over {summary['games']:.0f} games): {summary['points_per36_shared']:.1f} pts/36 vs "
            f"{summary['points_per36_same_games']:.1f} pts/36 over all his minutes in those games. "
            "This is shared floor time, not a defensive assignment."
        )


def group_split(query: s.StatQuery, split: pd.DataFrame, baseline: pd.DataFrame, catalog: dict[str, StatDef]) -> GroupResult:
    """Split vs. baseline for every value of ``query.group_by`` within the filtered games.

    Each level is compared to the whole baseline, so the rows read as "on
    Tuesdays he averages X, Y above his normal". p-values compare the level's
    games to all other baseline games. ``p_value_holm`` adjusts them for
    testing every level at once (see :func:`inference.holm_adjust`).
    """
    column = GROUPABLE_VARIABLES[query.group_by]
    rows = []
    levels = split[column].dropna().astype("object")
    for level in sort_levels(levels.unique(), query.group_by):
        in_level = split[split[column].astype("object") == level]
        rest = baseline.drop(index=in_level.index)
        for name in query.stats:
            stat = catalog[name]
            r = compare(stat, in_level, rest, baseline, prior_for(stat, baseline, query.group_by))
            rows.append({
                "level": level, "stat": name, "n": r.split.n, "value": r.split.value,
                "median": r.split.median, "std": r.split.std, "per36": r.split.per36,
                "baseline": r.baseline.value, "diff": r.diff, "diff_pct": r.diff_pct,
                "z_score": r.z_score, "p_value": r.p_value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "shrunk": r.shrunk, "shrink_weight": r.shrink_weight,
                "confidence": confidence_label(r.split.n),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_value_holm"] = table.groupby("stat")["p_value"].transform(
            lambda p: pd.Series(inf.holm_adjust(p.to_numpy()), index=p.index)
        )
    return GroupResult(query.group_by, table)


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def sort_levels(levels: np.ndarray, variable: str) -> list:
    """Natural order for known variables (weekdays, rest buckets), else sorted."""
    levels = list(levels)
    if variable == "day_of_week":
        return [d for d in DAY_ORDER if d in levels]
    if variable == "rest_days":
        order = ["0", "1", "2", "3+"]
        return [r for r in order if r in levels]
    try:
        return sorted(levels)
    except TypeError:
        return sorted(levels, key=str)
