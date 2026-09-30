"""League-wide effect of one variable on a stat, for a *typical* player.

Model (one regression per stat)::

    stat_ig = alpha_i + beta x variable_ig + gamma x controls_ig + error_ig

``alpha_i`` is a separate intercept for every player (a *fixed effect*). It
absorbs everything stable about the player (talent, role, era), so beta is
estimated only from how each player's own games move when the variable
changes. That is what "how much does X move the stat for a typical player"
means. Without fixed effects, a variable that simply correlates with *who*
plays (stars start more playoff games) would be mistaken for an effect.

Fixed effects are fit with the *within* transformation: subtract each player's
own average from the stat and from every regressor, then run ordinary least
squares. This gives exactly the same beta as adding one dummy per player, but
without a 2,800-column matrix.

Standard errors are **clustered by player**: one player's games are not
independent (hot streaks, injuries), and clustering lets errors correlate
freely within a player. Confidence intervals are coef +/- t x SE, and p-values
test beta = 0.

Default controls: minutes, home, rest days (capped at 3), opponent pre-game
defensive rating, and the opponent's minutes-weighted average height. When
defender height/weight/position is the *studied* variable, only games with an
individual defender (official matchup or opposing same-position starter) are
used, so the variable means the same thing on every row. Holding minutes fixed means the
effect is "at the same playing time". Pass ``control_minutes=False`` to include
effects that work by changing minutes (a back-to-back might cut minutes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nbalab.query import schema as s
from nbalab.query.data import QueryData, load_query_data
from nbalab.query.defenders import (
    SOURCE_LABELS,
    defender_attributes,
    from_team_average,
    opponent_rows,
    summarize_sources,
)
from nbalab.query.filters import describe_filter, filter_mask, scope_mask
from nbalab.query.stats import stat_catalog

REST_CAP = 3
# A control missing in more than this share of games is dropped rather than
# silently discarding those games from the regression.
MAX_CONTROL_MISSING = 0.2


@dataclass(frozen=True)
class VariableSpec:
    """How to turn one named variable into regressor columns.

    ``kind`` is "numeric" (one slope), "binary" (0/1, effect of being 1), or
    "categorical" (dummies against ``reference``).
    """

    name: str
    column: str
    kind: str
    unit: str = ""
    reference: str | None = None
    needs_defender: bool = False
    replaces_controls: tuple[str, ...] = ()
    coding: str = "treatment"


VARIABLES: dict[str, VariableSpec] = {
    v.name: v
    for v in [
        VariableSpec("home_away", "is_home", "binary", "home vs away", replaces_controls=("is_home",)),
        VariableSpec("back_to_back", "is_b2b", "binary", "2nd night of back-to-back vs not",
                     replaces_controls=("rest_days",)),
        VariableSpec("rest_days", "rest_days", "numeric", "per extra rest day (capped at 3)",
                     replaces_controls=("rest_days",)),
        VariableSpec("rest_bucket", "rest_bucket", "categorical", "vs 1 day of rest", reference="1",
                     replaces_controls=("rest_days",)),
        VariableSpec("day_of_week", "day_of_week", "categorical", "vs Wednesday", reference="Wednesday"),
        VariableSpec("month", "month_label", "categorical", "vs January", reference="01"),
        VariableSpec("week_of_season", "week_of_season", "numeric", "per week into the season"),
        VariableSpec("playoffs", "is_playoff", "binary", "playoff vs regular season"),
        VariableSpec("defender_height", "def_height", "numeric", "per inch of defender height",
                     needs_defender=True, replaces_controls=("opp_avg_height",)),
        VariableSpec("defender_weight", "def_weight", "numeric", "per 10 lb of defender weight",
                     needs_defender=True, replaces_controls=("opp_avg_height",)),
        VariableSpec("defender_position", "def_position", "categorical", "vs a guard defender",
                     reference="G", needs_defender=True, replaces_controls=("opp_avg_height",)),
        VariableSpec("opponent_def_rating", "opp_def_rating", "numeric",
                     "per +1 opponent defensive rating (worse defense)", replaces_controls=("opp_def_rating",)),
        VariableSpec("opponent_pace", "opp_pace", "numeric", "per +1 opponent pace"),
        VariableSpec("minutes", "minutes", "numeric", "per extra minute", replaces_controls=("minutes",)),
        VariableSpec("starter", "is_starter", "binary", "started vs came off the bench"),
        VariableSpec("age", "age", "numeric", "per year of age"),
        VariableSpec("venue", "venue_city", "categorical", "vs the average arena (home court held fixed)",
                     coding="deviation"),
        VariableSpec("altitude", "altitude_kft", "numeric", "per 1,000 ft of arena elevation"),
        VariableSpec("denver", "is_denver", "binary", "game in Denver vs elsewhere"),
    ]
}

# Approximate arena elevation in feet, keyed by the home franchise's city that
# season (``venue_city``). Reference data for the altitude variable.
VENUE_ELEVATION_FT: dict[str, float] = {
    "Atlanta": 1050, "Boston": 20, "Brooklyn": 30, "Charlotte": 750, "Chicago": 600, "Cleveland": 650,
    "Dallas": 430, "Denver": 5280, "Detroit": 600, "Golden State": 50, "Houston": 80, "Indiana": 715,
    "Los Angeles": 300, "Memphis": 340, "Miami": 10, "Milwaukee": 620, "Minnesota": 830, "New Jersey": 30,
    "New Orleans": 5, "New York": 30, "Oklahoma City": 1200, "Orlando": 100, "Philadelphia": 40,
    "Phoenix": 1090, "Portland": 50, "Sacramento": 30, "San Antonio": 650, "Seattle": 180, "Toronto": 250,
    "Utah": 4230, "Vancouver": 10, "Washington": 30,
}
# Aliases so group_by names work as effect variables too.
VARIABLE_ALIASES = {"defender_height_bucket": "defender_height", "defender_weight_bucket": "defender_weight",
                    "home": "home_away", "b2b": "back_to_back"}

PLAYER_CONTROLS = ("minutes", "is_home", "rest_days", "opp_def_rating", "opp_avg_height")
# Sources that name one defender. Mixing them with the roster-average proxy in
# one regressor would confound defender size with starter vs bench role
# (starters face one opposing starter; bench rows get the roster average).
INDIVIDUAL_SOURCES = ("official_matchups", "same_position_starter")
TEAM_CONTROLS = ("is_home", "rest_days", "opp_def_rating")


@dataclass
class EffectTable:
    """Regression output for one stat."""

    stat: str
    coefficients: pd.DataFrame  # term, coef, std_err, ci_low, ci_high, p_value, is_variable
    n_obs: int
    n_subjects: int
    r2_within: float
    stat_mean: float


@dataclass
class VariableEffectResult:
    variable: str
    unit: str
    subject_type: str
    method: str
    controls: list[str]
    population: list[str]
    per_stat: dict[str, EffectTable]
    defender_sources: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variable": self.variable, "unit": self.unit, "subject_type": self.subject_type,
            "method": self.method, "controls": self.controls, "population": self.population,
            "defender_sources": self.defender_sources, "notes": self.notes,
            "per_stat": {
                k: {"n_obs": t.n_obs, "n_subjects": t.n_subjects, "r2_within": t.r2_within,
                    "stat_mean": t.stat_mean, "coefficients": t.coefficients.to_dict(orient="records")}
                for k, t in self.per_stat.items()
            },
        }


# ------------------------------------------------------------------------ design


def resolve_variable(name: str) -> VariableSpec:
    key = VARIABLE_ALIASES.get(name, name)
    if key not in VARIABLES:
        raise ValueError(f"unknown effect variable {name!r}; valid: {sorted(VARIABLES)}")
    return VARIABLES[key]


def population(query: s.StatQuery, data: QueryData) -> pd.DataFrame:
    """Rows the regression runs on: every subject (or the listed ones) under the query's filters.

    Filters that only make sense for one subject (a named defender or teammate)
    are rejected here, because a league-wide model has no single subject for them.
    """
    team = query.subject_type == "team"
    df = data.team_games if team else data.player_games
    id_col = "teamId" if team else "personId"
    if query.subject_ids:
        df = df[df[id_col].isin(query.subject_ids)]
    df = df.copy()
    df["home_away"] = np.where(df["home"].astype("float64") == 1, "home", "away")
    df["back_to_back"] = df["team_is_back_to_back"].astype(bool)
    df["day_of_week"] = df["day_of_week"].astype(str)
    mask = scope_mask(query.scope_filters, df, data.latest_season)
    for f in query.split_filters:
        if f.type in {"defender_player", "teammate", "opponent_player_on_court", "last_n_games",
                      "defender_height_range", "defender_weight_range", "defender_position"}:
            raise ValueError(f"filter {f.type!r} is not supported in variable_effect mode")
        mask &= filter_mask(f, df, data, query.subject_type).mask.fillna(False)
    return df[mask].reset_index(drop=True)


def design_columns(df: pd.DataFrame, data: QueryData, need_defender: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add every numeric column a variable or control may use. Returns (df, defender source counts)."""
    out = df.copy()
    out["is_home"] = (out["home"].astype("float64") == 1).astype(float)
    out["is_b2b"] = out["team_is_back_to_back"].astype(float)
    out["rest_days"] = out["team_rest_days"].astype("float64").clip(upper=REST_CAP)
    out["rest_bucket"] = out["rest_days"].map(lambda r: None if np.isnan(r) else ("3+" if r >= 3 else str(int(r))))
    out["month_label"] = out["month"].astype(int).map(lambda m: f"{m:02d}")
    out["is_playoff"] = out["game_type"].astype(str).eq("playoffs").astype(float)
    out["opp_def_rating"] = out["opp_pre_def_rating"].astype("float64")
    out["opp_pace"] = out["opp_pre_pace"].astype("float64")
    out["week_of_season"] = out["week_of_season"].astype("float64")
    out["venue_city"] = out["venue_city"].astype("object")
    out["altitude_kft"] = out["venue_city"].map(VENUE_ELEVATION_FT).astype("float64") / 1000.0
    out["is_denver"] = out["venue_city"].eq("Denver").astype(float)
    sources: dict[str, int] = {}
    if "personId" in out:
        out["minutes"] = out["minutes"].astype("float64")
        out["is_starter"] = out["starter"].astype(float)
        out["age"] = out["age"].astype("float64")
        if need_defender:
            opp = opponent_rows(out, data)
            attrs = defender_attributes(out, data, opp)
            individual = attrs["def_source"].isin(INDIVIDUAL_SOURCES)
            # As the studied variable: only games with an individual defender.
            out["def_height"] = attrs["def_height"].astype("float64").where(individual)
            out["def_weight"] = (attrs["def_weight"].astype("float64") / 10.0).where(individual)
            out["def_position"] = attrs["def_position"].where(individual)
            # As a control: one consistent measure for every game.
            team_avg = from_team_average(out, opp).set_index(["gameId", "personId"])["def_height"]
            keys = pd.MultiIndex.from_frame(out[["gameId", "personId"]])
            out["opp_avg_height"] = team_avg.reindex(keys).to_numpy(dtype="float64")
            sources = summarize_sources(attrs["def_source"].where(individual, "excluded_no_individual_defender"))
    return out, sources


def regressors(df: pd.DataFrame, spec: VariableSpec, controls: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Design matrix (variable terms first) and the names of the variable's own terms."""
    if spec.kind == "categorical":
        levels = df[spec.column].dropna().astype(str)
        ref = spec.reference if spec.reference in set(levels) else levels.value_counts().idxmax()
        if spec.coding == "deviation":
            ref = sorted(set(levels))[-1]
        dummies = pd.get_dummies(df[spec.column].astype("object"), prefix=spec.name, prefix_sep="=", dtype=float)
        ref_col = dummies.pop(f"{spec.name}={ref}") if f"{spec.name}={ref}" in dummies else None
        if spec.coding == "deviation" and ref_col is not None:
            dummies = dummies.sub(ref_col, axis=0)  # the reference level is coded -1 everywhere
        dummies[df[spec.column].isna().to_numpy()] = np.nan
        var_terms = sorted(dummies.columns)
        X = pd.concat([dummies[var_terms], df[controls]], axis=1)
    else:
        X = pd.concat([df[[spec.column]].rename(columns={spec.column: spec.name}), df[controls]], axis=1)
        var_terms = [spec.name]
    return X, var_terms


def within(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract each group's own mean from every column (the fixed-effects transformation)."""
    return frame - frame.groupby(groups.to_numpy()).transform("mean")


def fit_fixed_effects(y: pd.Series, X: pd.DataFrame, groups: pd.Series, level: float = 0.95):
    """OLS on within-transformed data with standard errors clustered by ``groups``.

    Rows with any missing value are dropped *before* demeaning so each group's
    mean is computed on the rows actually used. Groups with a single game carry
    no within-player information and drop out automatically (all zeros).
    """
    ok = y.notna() & X.notna().all(axis=1)
    y, X, g = y[ok], X[ok], groups[ok]
    keep = g.map(g.value_counts()) > 1
    y, X, g = y[keep], X[keep], g[keep]
    y_w, X_w = within(y.to_frame(), g).iloc[:, 0], within(X, g)
    varying = [c for c in X_w.columns if X_w[c].abs().max() > 1e-12]
    model = sm.OLS(y_w.to_numpy(), X_w[varying].to_numpy())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    ci = fit.conf_int(alpha=1 - level)
    cov = pd.DataFrame(fit.cov_params(), index=varying, columns=varying)
    table = pd.DataFrame({
        "term": varying,
        "coef": fit.params,
        "std_err": fit.bse,
        "ci_low": ci[:, 0],
        "ci_high": ci[:, 1],
        "p_value": fit.pvalues,
    })
    ss_res = float(np.sum(fit.resid ** 2))
    ss_tot = float(np.sum(y_w.to_numpy() ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return table, int(len(y)), int(g.nunique()), r2, float(y.mean()), cov


def add_deviation_reference(
    table: pd.DataFrame, cov: pd.DataFrame, var_terms: list[str], spec: VariableSpec, df: pd.DataFrame,
    level: float = 0.95,
) -> pd.DataFrame:
    """Add the omitted level's row under deviation (sum-to-zero) coding.

    With deviation coding every level's coefficient is its gap from the
    average level, and the gaps sum to zero. So the omitted level's effect is
    minus the sum of the others. Its standard error comes from the full
    covariance matrix: Var(-sum b) = 1' V 1.
    """
    from scipy import stats as sps

    terms = [t for t in var_terms if t in cov.index]
    ref = sorted(set(df[spec.column].dropna().astype(str)))[-1]
    coef = -float(table.loc[table["term"].isin(terms), "coef"].sum())
    se = float(np.sqrt(cov.loc[terms, terms].to_numpy().sum()))
    z = sps.norm.ppf(0.5 + level / 2)
    row = {"term": f"{spec.name}={ref}", "coef": coef, "std_err": se, "ci_low": coef - z * se,
           "ci_high": coef + z * se, "p_value": float(2 * sps.norm.sf(abs(coef / se))) if se > 0 else np.nan,
           "is_variable": True}
    out = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    first = out[out["is_variable"]].sort_values("term")
    return pd.concat([first, out[~out["is_variable"]]], ignore_index=True)


# ----------------------------------------------------------------------- runner


def run_variable_effect(
    query: s.StatQuery, data: QueryData | None = None, control_minutes: bool = True
) -> VariableEffectResult:
    """Fit the fixed-effects regression for each requested stat."""
    data = data or load_query_data()
    spec = resolve_variable(query.effect_variable or query.group_by or "")
    team = query.subject_type == "team"
    base_controls = list(TEAM_CONTROLS if team else PLAYER_CONTROLS)
    if not control_minutes and "minutes" in base_controls:
        base_controls.remove("minutes")
    controls = [c for c in base_controls if c not in spec.replaces_controls and c != spec.column]
    if spec.needs_defender and team:
        raise ValueError(f"{spec.name} applies to players only")

    pop = population(query, data)
    need_def = spec.needs_defender or "opp_avg_height" in controls
    df, sources = design_columns(pop, data, need_def)
    notes: list[str] = []
    sparse = [c for c in controls if df[c].isna().mean() > MAX_CONTROL_MISSING]
    if sparse:
        controls = [c for c in controls if c not in sparse]
        notes.append(f"Dropped controls missing in more than {MAX_CONTROL_MISSING:.0%} of games: {', '.join(sparse)}.")
    groups = df["teamId" if team else "personId"]
    X, var_terms = regressors(df, spec, controls)
    catalog = stat_catalog(query.subject_type)

    per_stat = {}
    for name in query.stats:
        y = catalog[name].per_game(df)
        table, n, n_subj, r2, mean, cov = fit_fixed_effects(y, X, groups)
        table["is_variable"] = table["term"].isin(var_terms)
        if spec.coding == "deviation":
            table = add_deviation_reference(table, cov, var_terms, spec, df)
        per_stat[name] = EffectTable(name, table.reset_index(drop=True), n, n_subj, r2, mean)

    if spec.needs_defender:
        notes.append(
            "Only games with an individual defender are used (official matchup, else the opposing "
            "same-position starter, so starters only): "
            + "; ".join(f"{SOURCE_LABELS.get(k, k.replace('_', ' '))}: {v:,} games" for k, v in sources.items())
        )
    elif "opp_avg_height" in controls:
        notes.append("Defender size is controlled with the opponent's minutes-weighted average height (a proxy).")
    if "minutes" in controls:
        notes.append("Minutes are held fixed, so effects are 'at the same playing time'.")
    return VariableEffectResult(
        variable=spec.name,
        unit=spec.unit,
        subject_type=query.subject_type,
        method=("OLS with " + ("team" if team else "player")
                + " fixed effects (within transformation), standard errors clustered by "
                + ("team" if team else "player")),
        controls=controls,
        population=[describe_filter(f, data) for f in query.filters] or ["all games in the dataset"],
        per_stat=per_stat,
        defender_sources=sources,
        notes=notes,
    )
