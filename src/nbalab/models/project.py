"""Public API: project one future game for a player or team.

    from nbalab.models import load_bundle, project
    res = project(query, data, bundle, projected_minutes=None)
    res.stats["points"].mean, .pi[95], .p_over_at(27.5), .contributions
    res.joint_probability({"points": (">", 27.5), "assists": (">", 6.5)})

How a player projection is built (every number comes from here, not the LLM):

1. The player's history plus a placeholder "next game" row go through the
   same feature code as training, so the next game sees every game played so far.
2. The minutes model gives 25 equally likely minutes scenarios (or the
   user's override).
3. The chosen rate model gives the expected stat per minute; scenario x rate
   is the expected stat in that scenario. A named defender (proxy) scales it.
4. The count distribution (negative binomial or Poisson) is mixed over the
   scenarios. Mean, median, intervals and P(over/under/push) are read from
   that exact pmf, with no simulation.
5. Joint probabilities and combo stats (PRA, stocks) come from 10,000
   simulated games that share the minutes draw and a Gaussian copula.
6. Exact Shapley values split (projection - neutral game) across context groups.

If home/away is not given, the projection is a 50/50 mix of both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from nbalab.models import copula
from nbalab.models.calibration import ProbabilityCalibrator, calibrated_over_under_push, node_interval
from nbalab.models.config import APPLY_DEFENDER_PROXY, MC_DRAWS, SEED
from nbalab.models.contributions import Contribution, all_coalitions, shapley_values
from nbalab.models.defender import LABEL as DEFENDER_LABEL
from nbalab.models.defender import defender_effects
from nbalab.models.distributions import count_pmf, pmf_moments, prediction_interval, quantile, support_for
from nbalab.models.features import (
    EWM_HALFLIVES, PLAYER_COUNT_COLS, TEAM_COUNT_COLS, add_team_history, attach_opponent_allowed, ewm_effective_n,
    finish_team_frame, future_team_ratings, last_starters_size, opponent_starter_size, player_frame_from,
    positional_defense_cumulative, positional_defense_totals, prepare_player_games, prepare_team_games,
)
from nbalab.models.predict import conditional_pit
from nbalab.models.quantile import MINUTES_FEATURE, QuantileModel
from nbalab.models.registry import Bundle
from nbalab.query.data import QueryData
from nbalab.query.schema import ProjectionContext, StatQuery

LEVELS: tuple[int, ...] = (90, 95, 99)
APP_HISTORY_SEASONS = 3
SMALL_SAMPLE_GAMES = 20
REGULAR_SEASON_NOTE = ("Projected as a regular-season game. Playoff games run longer minutes for starters, so treat "
                       "a playoff projection as a floor on minutes.")
FORM_HALFLIFE = max(EWM_HALFLIVES)

# query stat name -> model stat, and combo stats built from simulated draws
PLAYER_STATS: dict[str, str] = {
    "points": "points", "assists": "assists", "rebounds": "reboundsTotal", "threes": "threePointersMade",
    "steals": "steals", "blocks": "blocks", "turnovers": "turnovers",
}
PLAYER_COMBOS: dict[str, tuple[str, ...]] = {
    "pra": ("points", "rebounds", "assists"), "pr": ("points", "rebounds"), "pa": ("points", "assists"),
    "ra": ("rebounds", "assists"), "stocks": ("steals", "blocks"),
}
TEAM_STATS: dict[str, str] = {"team_score": "teamScore", "assists": "assists", "rebounds": "reboundsTotal",
                              "threes": "threePointersMade"}
MODEL_NAMES = {"glm": "Poisson GLM rate", "lgbm": "LightGBM Poisson rate", "quantile": "LightGBM quantile regression"}
GROUP_ORDER = ("opponent", "home", "rest", "form", "minutes", "defender")

TOOLTIPS: dict[str, str] = {
    "ci_mean": ("Confidence interval of the average: how precisely we know this player's true average in this "
                "context. It is narrow because it only reflects estimation uncertainty (recent games are an "
                "effective sample of about 20-30 games, plus uncertainty in projected minutes)."),
    "pi": ("Prediction interval: where this one game's box score is likely to land. It is wide because a single "
           "game swings a lot (hot/cold shooting, minutes, foul trouble). For count stats it holds at least the "
           "stated probability; the backtest reports the real coverage."),
    "ci_vs_pi": ("A 95% CI of 26.1-28.4 means we're fairly sure of his typical output. A 95% PI of 14-41 is the "
                 "range for tonight. The PI is the one to compare with a betting line."),
    "p_over": ("Probability the stat finishes strictly above the line. A whole-number line can also push (land exactly "
               "on it). For most stats these probabilities pass through a calibration map fitted on the 2024-25 season "
               "and checked on 2025-26, so a stated 70% happens about 70% of the time."),
    "joint": ("Probability that all the chosen overs hit in the same game, from 10,000 simulated games. Stats share "
              "one minutes draw per game and a Gaussian copula fitted to this player's history (shrunk toward the "
              "league), so correlated stats are not treated as independent."),
    "contributions": ("How much each part of the context moves the expected value away from a neutral game "
                      "(league-average opponent, 50/50 home/away, 1 day of rest, recent minutes, form at the "
                      "longer-run level). Shapley values, so they add up exactly."),
    "defender": DEFENDER_LABEL,
}


# ------------------------------------------------------------------ results


@dataclass
class StatProjection:
    """The projected distribution of one stat in one future game."""

    stat: str
    mean: float
    median: float
    ci_mean: dict[int, tuple[float, float]]
    pi: dict[int, tuple[float, float]]
    support: np.ndarray
    pmf: np.ndarray
    line: float | None = None
    contributions: list[Contribution] = field(default_factory=list)
    neutral_mean: float = math.nan
    n_games: int = 0
    minutes_mean: float | None = None
    minutes_pi: dict[int, tuple[float, float]] | None = None
    model_name: str = ""
    warnings: list[str] = field(default_factory=list)
    calibrator: ProbabilityCalibrator | None = None  # applied to over/under/push only

    def __post_init__(self) -> None:
        self._cdf = np.cumsum(self.pmf)

    def _cdf_at(self, k: float) -> float:
        """P(X <= k) for integer k."""
        if k < self.support[0]:
            return 0.0
        return float(self._cdf[min(int(k) - int(self.support[0]), len(self._cdf) - 1)])

    def _raw_survival(self, k: float) -> float:
        """Raw P(X > k)."""
        return 1.0 - self._cdf_at(k)

    def probabilities_at(self, line: float, calibrated: bool = True) -> tuple[float, float, float]:
        """(P(over), P(under), P(push)) for one line; they always sum to 1."""
        s_line, s_below = self._raw_survival(math.floor(line)), self._raw_survival(math.ceil(line) - 1)
        if not calibrated or self.calibrator is None:
            return s_line, 1.0 - s_below, max(s_below - s_line, 0.0)
        o, u, p = calibrated_over_under_push(self.calibrator, np.array([s_line]), np.array([s_below]))
        return float(o[0]), float(u[0]), max(float(p[0]), 0.0)

    def p_over_at(self, line: float, calibrated: bool = True) -> float:
        return self.probabilities_at(line, calibrated)[0]

    def p_under_at(self, line: float, calibrated: bool = True) -> float:
        return self.probabilities_at(line, calibrated)[1]

    def p_push_at(self, line: float, calibrated: bool = True) -> float:
        return 0.0 if not float(line).is_integer() else self.probabilities_at(line, calibrated)[2]

    @property
    def p_over(self) -> float | None:
        return None if self.line is None else self.p_over_at(self.line)

    @property
    def p_under(self) -> float | None:
        return None if self.line is None else self.p_under_at(self.line)

    @property
    def p_push(self) -> float | None:
        if self.line is None or not float(self.line).is_integer():
            return None
        return self.p_push_at(self.line)

    def over_curve(self, lines: np.ndarray) -> pd.DataFrame:
        lines = np.asarray(lines, dtype=float)
        return pd.DataFrame({
            "line": lines,
            "p_over": [self.p_over_at(x) for x in lines],
            "p_under": [self.p_under_at(x) for x in lines],
            "p_push": [self.p_push_at(x) for x in lines],
        })


@dataclass
class ProjectionResult:
    subject_type: str
    subject_id: int
    stats: dict[str, StatProjection]
    draws: pd.DataFrame
    context: dict
    warnings: list[str] = field(default_factory=list)
    tooltips: dict[str, str] = field(default_factory=lambda: dict(TOOLTIPS))
    _joint_cache: dict = field(default_factory=dict, repr=False)

    def joint_probability(self, conditions: dict[str, copula.Condition]) -> float:
        """P(every condition holds), e.g. {"points": (">", 28.5), "assists": (">", 6.5)}."""
        key = tuple(sorted((k, v[0], float(v[1])) for k, v in conditions.items()))
        if key not in self._joint_cache:
            missing = [k for k in conditions if k not in self.draws]
            if missing:
                raise KeyError(f"no simulated draws for {missing}; available: {list(self.draws.columns)}")
            self._joint_cache[key] = copula.joint_probability(self.draws, conditions)
        return self._joint_cache[key]


# ------------------------------------------------------------------ cached league tables


@dataclass
class _Tables:
    player_base: pd.DataFrame
    pos_cumulative: pd.DataFrame
    starter_sizes: pd.DataFrame
    team_base: pd.DataFrame


_TABLES: dict[tuple[int, int], _Tables] = {}


def league_tables(data: QueryData) -> _Tables:
    """Recent-season lookup tables, built once per loaded dataset."""
    key = (id(data.player_games), id(data.team_games))
    if key not in _TABLES:
        first = data.latest_season - APP_HISTORY_SEASONS + 1
        base = prepare_player_games(data.player_games, first)
        _TABLES[key] = _Tables(base, positional_defense_cumulative(positional_defense_totals(base)),
                               opponent_starter_size(base), prepare_team_games(data.team_games, first))
    return _TABLES[key]


# ------------------------------------------------------------------ future rows


def rest_context(ctx: ProjectionContext) -> tuple[float, bool]:
    """(rest days, back-to-back) for the future game; unknown means one day of rest."""
    if ctx.back_to_back:
        return 0.0, True
    if ctx.rest_days is not None:
        return float(ctx.rest_days), ctx.rest_days == 0
    return 1.0, False


def home_value(ctx: ProjectionContext, team_id: int) -> float | None:
    if ctx.home_away is not None:
        return 1.0 if ctx.home_away == "home" else 0.0
    if ctx.venue_team_id is not None:
        return 1.0 if ctx.venue_team_id == team_id else 0.0
    return None


def player_rows(data: QueryData, bundle: Bundle, person_id: int, ctx: ProjectionContext
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(history with features, one future row with features)."""
    t = league_tables(data)
    hist = t.player_base[t.player_base["personId"].eq(person_id)]
    if hist.empty:
        raise ValueError(f"{data.player_name(person_id)} has no games in the last {APP_HISTORY_SEASONS} seasons")
    last = hist.iloc[-1]
    rest, b2b = rest_context(ctx)
    when = last["gameDateTimeEst"] + pd.Timedelta(days=rest + 1)
    opp = ctx.opponent_team_id
    fut = last.to_dict() | {
        "gameId": -1, "gameDateTimeEst": when, "game_date": when.normalize(), "opponentTeamId": opp if opp else -1,
        "minutes": np.nan, **{c: np.nan for c in PLAYER_COUNT_COLS}, "starter": False, "startingPosition": None,
        "age": float(last["age"]) + (rest + 1) / 365.25 if pd.notna(last["age"]) else np.nan,
        "team_rest_days": rest, "team_is_back_to_back": b2b, "opp_rest_days": 1.0, "opp_is_back_to_back": False,
        "home": 0.5, "is_playoff": 0.0,  # projections are for a regular-season game (see REGULAR_SEASON_NOTE)
    }
    fut |= future_team_ratings(data.team_games, int(last["teamId"]), opp, int(last["season"]), when)
    pos = str(last["position_group"])
    size = last_starters_size(t.player_base, opp, pos) if opp else (np.nan, np.nan)
    neutral = bundle.neutral.get("starter_size", {}).get(str(_pos_code(pos)), {})
    fut["opp_starter_height"] = size[0] if np.isfinite(size[0]) else neutral.get("opp_starter_height", np.nan)
    fut["opp_starter_weight"] = size[1] if np.isfinite(size[1]) else neutral.get("opp_starter_weight", np.nan)
    rows = pd.concat([hist, pd.DataFrame([fut])], ignore_index=True)
    for c in ("home", "team_is_back_to_back", "opp_is_back_to_back", "starter"):
        rows[c] = rows[c].astype(float)
    frame = player_frame_from(rows, t.pos_cumulative, t.starter_sizes, data.team_games)
    return frame.iloc[:-1].reset_index(drop=True), frame.iloc[[-1]].reset_index(drop=True)


def _pos_code(pos: str) -> int:
    return {"G": 0, "F": 1, "C": 2}.get(pos, 1)


def team_rows(data: QueryData, team_id: int, ctx: ProjectionContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(history with features, one future row with features) for a team."""
    t = league_tables(data)
    base = t.team_base
    own = base[base["teamId"].eq(team_id)]
    if own.empty:
        raise ValueError(f"{data.team_name(team_id)} has no games in the last {APP_HISTORY_SEASONS} seasons")
    last = own.iloc[-1]
    rest, b2b = rest_context(ctx)
    when = last["gameDateTimeEst"] + pd.Timedelta(days=rest + 1)
    opp = ctx.opponent_team_id
    blank = {c: np.nan for c in (*TEAM_COUNT_COLS, *(f"allowed_{c}" for c in TEAM_COUNT_COLS))}
    common = {"gameId": -1, "gameDateTimeEst": when, "game_date": when.normalize(), "season": int(last["season"]),
              "is_playoff": 0.0, **blank}
    fut = last.to_dict() | common | {"opponentTeamId": opp if opp else -1, "team_rest_days": rest,
                                     "team_is_back_to_back": b2b, "opp_rest_days": 1.0, "opp_is_back_to_back": False,
                                     "home": 0.5}
    extra = [fut]
    if opp:
        opp_last = base[base["teamId"].eq(opp)]
        if len(opp_last):
            extra.append(opp_last.iloc[-1].to_dict() | common | {"teamId": opp, "opponentTeamId": team_id})
    rows = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    rows = rows.sort_values(["teamId", "gameDateTimeEst", "gameId"], kind="stable").reset_index(drop=True)
    hist_all = attach_opponent_allowed(add_team_history(rows))
    is_fut = hist_all["gameId"].eq(-1) & hist_all["teamId"].eq(team_id)
    r = future_team_ratings(data.team_games, team_id, opp, int(last["season"]), when)
    for k, v in r.items():
        col = k.replace("team_pre_", "pre_")
        hist_all.loc[is_fut, col if k.startswith("team_") else k] = v
    if not opp:  # league-average opponent: mean of every team's latest "allowed" form
        cols = [c for c in hist_all.columns if c.startswith("opp_allowed_")]
        latest = hist_all[hist_all["gameId"].ne(-1)].groupby("teamId").tail(1)
        for c in cols:
            hist_all.loc[is_fut, c] = latest[c.removeprefix("opp_")].mean()
    keep = hist_all["teamId"].eq(team_id)
    frame = finish_team_frame(hist_all[keep].reset_index(drop=True), data.team_games)
    return frame.iloc[:-1].reset_index(drop=True), frame.iloc[[-1]].reset_index(drop=True)


# ------------------------------------------------------------------ scenarios and neutral games


def scenarios(future: pd.DataFrame, home: float | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Future rows to average over: one row if home/away is known, else home and away at 50/50."""
    if home is not None:
        return future.assign(home=home), np.ones(1)
    return pd.concat([future.assign(home=1.0), future.assign(home=0.0)], ignore_index=True), np.array([0.5, 0.5])


def neutral_overrides(subject: str, row: pd.Series, bundle: Bundle) -> dict[str, dict[str, float]]:
    """Column values of the neutral game, per context group (home is handled as a 50/50 mix)."""
    if subject == "player":
        size = bundle.neutral.get("starter_size", {}).get(str(int(row.get("pos_code", 1) or 1)), {})
        opponent = {c: 1.0 for c in row.index if c.endswith("_opp_pos")}
        opponent |= {"opp_drtg_rel": 1.0, "opp_ortg_rel": 1.0, "opp_pace_rel": 1.0, "opp_net_rel": 0.0,
                     "opp_rest_days": 1.0, "opp_is_back_to_back": 0.0, **size}
        form = {f"{c}_rate_ewm5": row[f"{c}_rate_ewm20"] for c in PLAYER_COUNT_COLS}
        form |= {"minutes_ewm5": row["minutes_ewm20"], "minutes_last5": row["minutes_last20"],
                 "minutes_last10": row["minutes_last20"], "minutes_trend": 0.0}
    else:
        opponent = {"opp_drtg_rel": 1.0, "opp_ortg_rel": 1.0, "opp_pace_rel": 1.0,
                    "opp_rest_days": 1.0, "opp_is_back_to_back": 0.0}
        form = {f"{c}_ewm5": row[f"{c}_ewm20"] for c in TEAM_COUNT_COLS}
    rest = {"team_rest_days": 1.0, "team_is_back_to_back": 0.0}
    return {"opponent": opponent, "rest": rest, "form": form}


def coalition_rows(rows: pd.DataFrame, row_w: np.ndarray, groups: list[str], overrides: dict[str, dict[str, float]]
                   ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Every coalition's rows stacked: (frame, coalition index, weight within coalition)."""
    frames, idx, weights = [], [], []
    for i, coal in enumerate(all_coalitions(groups)):
        base, w = rows, row_w
        if "home" in groups and "home" not in coal:
            base, w = scenarios(rows.iloc[[0]], None)
        base = base.copy()
        for g, cols in overrides.items():
            if g in groups and g not in coal:
                for c, v in cols.items():
                    if c in base:
                        base[c] = v
        frames.append(base)
        idx.extend([i] * len(base))
        weights.extend(list(w))
    return pd.concat(frames, ignore_index=True), np.asarray(idx), np.asarray(weights)


# ------------------------------------------------------------------ per-stat distributions


@dataclass
class _Dist:
    support: np.ndarray
    node_pmfs: np.ndarray  # (nodes, support)
    node_w: np.ndarray
    alpha: float


def stat_distribution(model, rows: pd.DataFrame, node_values: np.ndarray, node_row: np.ndarray,
                      node_w: np.ndarray, multiplier: float, row_w: np.ndarray) -> _Dist:
    """Per-node pmfs of one stat. Count models: NB(node minutes x rate); quantile: its own pmf per row."""
    if isinstance(model, QuantileModel):
        exp_m = _row_minutes(node_values, node_row, node_w, len(rows))
        f = rows.assign(**{MINUTES_FEATURE: exp_m}) if model.use_minutes else rows
        top = float(np.max(model.predict_quantiles(f)))
        support = np.arange(int(top * 1.5) + 15)
        pmfs = model.pmf(f, support)
        return _Dist(support, pmfs[node_row], node_w, 0.0)
    rate = model.predict_rate(rows) * multiplier
    means = node_values * rate[node_row]
    support = support_for(means, model.alpha)
    return _Dist(support, count_pmf(means, model.alpha, support), node_w, model.alpha)


def _row_minutes(node_values: np.ndarray, node_row: np.ndarray, node_w: np.ndarray, n_rows: int) -> np.ndarray:
    """Expected minutes of each scenario row (weighted mean of its nodes)."""
    out = np.zeros(n_rows)
    for r in range(n_rows):
        m = node_row == r
        out[r] = np.average(node_values[m], weights=node_w[m])
    return out


def coalition_means(model, frame: pd.DataFrame, exp_minutes: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    """Expected stat for each stacked coalition row."""
    if isinstance(model, QuantileModel):
        f = frame.assign(**{MINUTES_FEATURE: exp_minutes}) if model.use_minutes else frame
        top = float(np.max(model.predict_quantiles(f)))
        support = np.arange(int(top * 1.5) + 15)
        return model.pmf(f, support) @ support
    rate = model.predict_rate(frame) * multiplier
    return rate * exp_minutes if model.use_minutes else rate


# ------------------------------------------------------------------ main entry points


def project(query: StatQuery, data: QueryData, bundle: Bundle, projected_minutes: float | None = None
            ) -> ProjectionResult:
    """Projection for the first subject of a query (stats = query stats plus any line stats)."""
    if not query.subject_ids:
        raise ValueError("a projection needs a player or team")
    proj = query.projection
    ctx = proj.context if proj else ProjectionContext()
    lines = dict(proj.lines) if proj else {}
    stats = list(dict.fromkeys([*query.stats, *lines]))
    if query.subject_type == "team":
        return project_team(query.subject_ids[0], stats, ctx, lines, data, bundle)
    return project_player(query.subject_ids[0], stats, ctx, lines, data, bundle, projected_minutes)


def project_player(person_id: int, stats: list[str], context: ProjectionContext, lines: dict[str, float],
                   data: QueryData, bundle: Bundle, projected_minutes: float | None = None) -> ProjectionResult:
    override = projected_minutes if projected_minutes is not None else context.projected_minutes
    hist, fut = player_rows(data, bundle, person_id, context)
    rows, row_w = scenarios(fut, home_value(context, int(fut["teamId"].iloc[0])))
    notes: list[str] = []
    mults = _defender_multipliers(person_id, context, data, bundle, notes)
    return _assemble("player", person_id, stats, lines, hist, rows, row_w, bundle, override, mults, notes,
                     context, data)


def project_team(team_id: int, stats: list[str], context: ProjectionContext, lines: dict[str, float],
                 data: QueryData, bundle: Bundle) -> ProjectionResult:
    hist, fut = team_rows(data, team_id, context)
    rows, row_w = scenarios(fut, home_value(context, team_id))
    notes = ["Team projections have no minutes or defender stage."] if context.defender_person_id else []
    return _assemble("team", team_id, stats, lines, hist, rows, row_w, bundle, None, {}, notes, context, data)


def _defender_multipliers(person_id: int, ctx: ProjectionContext, data: QueryData, bundle: Bundle,
                          notes: list[str]) -> dict[str, float]:
    if ctx.defender_person_id is None:
        return {}
    name = data.player_name(ctx.defender_person_id)
    if data.on_court is None or not bundle.defender_tau2:
        notes.append(f"Defender {name}: no on-court data available, so no adjustment was applied.")
        return {}
    eff = defender_effects(data.on_court, data.player_games, person_id, ctx.defender_person_id, None,
                           bundle.defender_tau2)
    if not eff:
        notes.append(f"Defender {name}: the two have never shared the floor in the data, so no adjustment was applied.")
        return {}
    e = next(iter(eff.values()))
    shifts = ", ".join(f"{k} {100 * (v.multiplier - 1):+.1f}%" for k, v in eff.items())
    applied = "applied" if APPLY_DEFENDER_PROXY else (
        "NOT applied to the projection, because in the backtest this proxy made predictions worse")
    notes.append(f"Defender {name}: {DEFENDER_LABEL}. From {e.on_minutes:.0f} shared minutes over {e.n_games} games, "
                 f"shrunk toward no effect, the on/off data suggests {shifts}; {applied}.")
    return {s: e.multiplier for s, e in eff.items()} if APPLY_DEFENDER_PROXY else {}


def _minutes_nodes(bundle: Bundle, subject: str, rows: pd.DataFrame, row_w: np.ndarray, override: float | None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattened (node minutes, node row index, node weight) across scenario rows."""
    if subject == "team":
        return np.ones(len(rows)), np.arange(len(rows)), row_w / row_w.sum()
    nodes, w = bundle.minutes.distribution(rows, override)
    k = nodes.shape[1]
    return nodes.ravel(), np.repeat(np.arange(len(rows)), k), np.repeat(row_w, k) * np.tile(w, len(rows))


def _assemble(subject: str, sid: int, stats: list[str], lines: dict[str, float], hist: pd.DataFrame,
              rows: pd.DataFrame, row_w: np.ndarray, bundle: Bundle, override: float | None,
              mults: dict[str, float], notes: list[str], ctx: ProjectionContext, data: QueryData) -> ProjectionResult:
    sm = bundle.subject(subject)
    mapping = PLAYER_STATS if subject == "player" else TEAM_STATS
    combos = PLAYER_COMBOS if subject == "player" else {}
    wanted = [s for s in stats if s in mapping or s in combos]
    skipped = [s for s in stats if s not in wanted]
    warnings = [*notes, REGULAR_SEASON_NOTE]
    if skipped:
        warnings.append(f"No projection model for {', '.join(skipped)}; only the historical view is shown.")
    base = list(dict.fromkeys([c for s in wanted for c in (combos.get(s) or (s,))]))
    node_values, node_row, node_w = _minutes_nodes(bundle, subject, rows, row_w, override)
    dists = {q: stat_distribution(sm.model(mapping[q]), rows, node_values, node_row, node_w,
                                  mults.get(mapping[q], 1.0), row_w) for q in base}
    contribs, neutral = _contributions(subject, base, mapping, rows, row_w, bundle, override, mults, ctx, data)
    draws = _simulate(subject, base, mapping, dists, hist, bundle, node_values)
    for name, parts in combos.items():
        if all(p in draws for p in parts):
            draws[name] = draws[list(parts)].sum(axis=1)
    minutes_mean = float(np.average(node_values, weights=node_w)) if subject == "player" else None
    minutes_pi = _minutes_pi(node_values, node_w) if subject == "player" else None
    small = len(hist) < SMALL_SAMPLE_GAMES
    out: dict[str, StatProjection] = {}
    for q in wanted:
        parts = combos.get(q) or (q,)
        if q in combos:
            support, pmf = _pmf_from_draws(draws[q].to_numpy())
            mean = float(sum(_dist_mean(dists[p]) for p in parts))
        else:
            d = dists[q]
            support, pmf = d.support, d.node_w @ d.node_pmfs
            pmf = pmf / pmf.sum()
            mean = _dist_mean(d)
        sp_warn = []
        if small:
            sp_warn.append(f"Small sample: only {len(hist)} recent games, so the projection leans on league patterns.")
        model_kinds = sorted({sm.chosen[mapping[p]] for p in parts})
        cal = None if q in combos else _calibrator(bundle, subject, mapping[q])
        out[q] = StatProjection(
            stat=q, mean=mean, median=quantile(support, pmf, 0.5),
            ci_mean=_ci_mean(subject, parts, mapping, dists, hist, bundle, override, node_values, node_w),
            pi={lvl: prediction_interval(support, pmf, lvl / 100) for lvl in LEVELS},
            support=support, pmf=pmf, line=lines.get(q),
            contributions=_combine_contributions(contribs, parts), neutral_mean=float(sum(neutral[p] for p in parts)),
            n_games=len(hist), minutes_mean=minutes_mean, minutes_pi=minutes_pi,
            model_name=_model_name(model_kinds, [dists[p].alpha for p in parts], q in combos) + _cal_note(cal),
            warnings=sp_warn, calibrator=cal,
        )
    context = {"projected_minutes_override": override, **ctx.model_dump()}
    return ProjectionResult(subject, sid, out, draws, context, warnings)


def _calibrator(bundle: Bundle, subject: str, stat: str) -> ProbabilityCalibrator | None:
    cal = ProbabilityCalibrator.from_dict(bundle.calibration.get(subject, {}).get(stat))
    return None if cal.kind == "identity" else cal


def _cal_note(cal: ProbabilityCalibrator | None) -> str:
    if cal is None:
        return ""
    push = "; push probabilities are the raw model's" if cal.raw_push else ""
    return f"; over/under probabilities recalibrated ({cal.kind.capitalize()} scaling fitted on 2024-25{push})"


def _dist_mean(d: _Dist) -> float:
    return float(d.node_w @ (d.node_pmfs @ d.support) / d.node_w.sum())


def _minutes_pi(values: np.ndarray, w: np.ndarray) -> dict[int, tuple[float, float]]:
    """Equal-tailed minutes range; scenarios are equally weighted, as in the backtest's coverage check."""
    out = {}
    for lvl in LEVELS:
        lo, hi = node_interval(values[None, :], lvl / 100)
        out[lvl] = (float(lo[0]), float(hi[0]))
    return out


def _pmf_from_draws(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(np.asarray(x, dtype=int))
    return np.arange(len(counts)), counts / counts.sum()


def _model_name(kinds: list[str], alphas: list[float], combo: bool) -> str:
    fam = " + ".join(MODEL_NAMES[k] for k in kinds)
    if kinds == ["quantile"]:
        dist = "piecewise-linear CDF from predicted quantiles"
    else:
        dist = "negative binomial" if any(a > 0 for a in alphas) else "Poisson"
        dist += " mixed over minutes scenarios"
    name = f"{fam} ({dist})"
    return name + ", combo from 10,000 simulated games" if combo else name


def _ci_mean(subject: str, parts: tuple[str, ...], mapping: dict[str, str], dists: dict[str, _Dist],
             hist: pd.DataFrame, bundle: Bundle, override: float | None, node_values: np.ndarray,
             node_w: np.ndarray) -> dict[int, tuple[float, float]]:
    """CI of the average: rate uncertainty plus minutes uncertainty (delta method, parts added as independent).

    relative SE of the rate ~ sqrt(var / n_eff) / mean. ``var`` is one game's
    variance *at fixed minutes* (NB: mean + alpha x mean^2), because minutes get
    their own term; n_eff is the effective number of games behind the model's
    longest recency-weighted form feature (half-life FORM_HALFLIFE games). The
    minutes term is sd(recent minutes) / sqrt(n_eff) / projected minutes.
    """
    total_mean, var_mean = 0.0, 0.0
    n_eff = max(ewm_effective_n(len(hist), FORM_HALFLIFE), 1.0)
    for p in parts:
        d = dists[p]
        mean = _dist_mean(d)
        if d.alpha > 0 or not isinstance(bundle.subject(subject).model(mapping[p]), QuantileModel):
            var = mean + d.alpha * mean**2
        else:
            _, var = pmf_moments(d.support, d.node_w @ d.node_pmfs / d.node_w.sum())
        rel = var / max(mean, 1e-9) ** 2 / n_eff
        if subject == "player" and override is None and len(hist) > 1:
            m_mean = float(np.average(node_values, weights=node_w))
            rel += (float(hist["minutes"].tail(20).std()) / math.sqrt(n_eff) / max(m_mean, 1e-9)) ** 2
        total_mean += mean
        var_mean += rel * mean**2
    se = math.sqrt(var_mean)
    out = {}
    for lvl in LEVELS:
        z = sps.norm.ppf(0.5 + lvl / 200)
        out[lvl] = (max(total_mean - z * se, 0.0), total_mean + z * se)
    return out


def _contributions(subject: str, base: list[str], mapping: dict[str, str], rows: pd.DataFrame, row_w: np.ndarray,
                   bundle: Bundle, override: float | None, mults: dict[str, float], ctx: ProjectionContext,
                   data: QueryData) -> tuple[dict[str, list[Contribution]], dict[str, float]]:
    """Shapley contributions per base stat, and the neutral-game mean."""
    sm = bundle.subject(subject)
    groups = ["opponent", "home", "rest", "form"]
    if subject == "player":
        groups.append("minutes")
        if mults:
            groups.append("defender")
    row0 = rows.iloc[0]
    frame, coal_idx, w = coalition_rows(rows, row_w, groups, neutral_overrides(subject, row0, bundle))
    coalitions = all_coalitions(groups)
    minutes_on = np.array(["minutes" in coalitions[i] for i in coal_idx])
    defender_on = np.array(["defender" in coalitions[i] for i in coal_idx])
    if subject == "player":
        nodes, nw = bundle.minutes.distribution(frame, override)
        exp_m = np.where(minutes_on, nodes @ nw / nw.sum(), _neutral_minutes(row0))
    else:
        exp_m = np.ones(len(frame))
    labels = _group_labels(subject, ctx, row0, data, override, exp_m[minutes_on][0] if minutes_on.any() else None)
    out, neutral = {}, {}
    for q in base:
        stat = mapping[q]
        mult = np.where(defender_on, mults.get(stat, 1.0), 1.0)
        means = coalition_means(sm.model(stat), frame, exp_m, mult)
        values = np.bincount(coal_idx, weights=means * w) / np.bincount(coal_idx, weights=w)
        phi = shapley_values(groups, {c: float(values[i]) for i, c in enumerate(coalitions)})
        neutral[q] = float(values[0])
        out[q] = [Contribution(labels[g], phi[g], g, g == "defender") for g in GROUP_ORDER if g in phi]
    return out, neutral


def _neutral_minutes(row: pd.Series) -> float:
    for c in ("minutes_last10", "minutes_ewm20", "minutes_std"):
        if pd.notna(row.get(c)):
            return float(row[c])
    return 24.0


def _group_labels(subject: str, ctx: ProjectionContext, row: pd.Series, data: QueryData,
                  override: float | None, exp_minutes: float | None) -> dict[str, str]:
    opp = data.team_name(ctx.opponent_team_id) if ctx.opponent_team_id else "league-average opponent"
    home = {None: "Home/away not set (50/50)", 1.0: "Home game", 0.0: "Away game"}[home_value(ctx, int(row["teamId"]))]
    rest = "Back-to-back" if row["team_is_back_to_back"] else f"{int(row['team_rest_days'])} day(s) of rest"
    labels = {"opponent": f"Opponent: {opp}", "home": home, "rest": rest, "form": "Recent form vs longer-run level"}
    if subject == "player":
        m = override if override is not None else exp_minutes
        labels["minutes"] = f"Projected minutes ({m:.1f} vs recent {_neutral_minutes(row):.1f})" if m else "Projected minutes"
        if ctx.defender_person_id:
            labels["defender"] = f"Defender: {data.player_name(ctx.defender_person_id)}"
    return labels


def _combine_contributions(contribs: dict[str, list[Contribution]], parts: tuple[str, ...]) -> list[Contribution]:
    """Shapley values are additive, so a combo's contribution is the sum of its parts'."""
    first = contribs[parts[0]]
    return [Contribution(c.label, float(sum(next(x.value for x in contribs[p] if x.group == c.group) for p in parts)),
                         c.group, c.is_proxy) for c in first]


def _simulate(subject: str, base: list[str], mapping: dict[str, str], dists: dict[str, _Dist], hist: pd.DataFrame,
              bundle: Bundle, node_values: np.ndarray) -> pd.DataFrame:
    """10,000 joint draws sharing one minutes scenario per draw, correlated by the copula."""
    rng = np.random.default_rng(SEED)
    sm = bundle.subject(subject)
    corr = _player_correlation(subject, base, mapping, hist, sm, rng)
    tables = {q: np.cumsum(dists[q].node_pmfs, axis=1) for q in base}
    w = dists[base[0]].node_w
    draws = copula.simulate(w, tables, corr, MC_DRAWS, rng, node_values if subject == "player" else None)
    return draws


def _player_correlation(subject: str, base: list[str], mapping: dict[str, str], hist: pd.DataFrame, sm,
                        rng: np.random.Generator) -> np.ndarray:
    """Subject's residual correlation (given minutes), shrunk toward the league matrix."""
    idx = [sm.corr_stats.index(mapping[q]) for q in base]
    league = sm.league_corr[np.ix_(idx, idx)]
    if len(base) < 2 or len(hist) < 3:
        return copula.nearest_correlation(league) if len(base) > 1 else np.eye(len(base))
    z = np.column_stack([
        copula.normal_scores(conditional_pit(sm.model(mapping[q]), hist, hist[mapping[q]].to_numpy(float), rng))
        for q in base
    ])
    own = copula.correlation(z)
    return copula.shrink_correlation(own, league, len(hist))
