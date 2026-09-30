"""Model features for player and team projections.

Every feature on a game row uses only what was known before tip-off:

- **Form** is an exponentially weighted moving average (EWMA) of *previous*
  games. Half-life h means a game h games ago counts half as much as the most
  recent one. Per-minute rates are the ratio of two EWMAs (stat / minutes), so
  a 40-minute game counts more than a 10-minute one.
- **Season to date** is the per-minute rate over earlier games this season.
- **Opponent positional defense** is how much the opponent has allowed to
  players of this position group (per minute) this season, relative to the
  league, shrunk toward 1.0 early in the season when the sample is small.
- **Opponent starter size** is the height and weight of the opponent's
  starter(s) at the player's position in that game. Starting lineups are
  announced before tip-off, so this counts as pre-game information.

No player or team ids are model features. The same builders produce the row
for a *future* game: a placeholder row with unknown stats is appended to the
history, and its lagged features then summarize every game played so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbalab.data.features import cumulative_team_ratings, lagged_rolling_mean, pregame_team_ratings
from nbalab.models.config import PLAYER_STATS, TEAM_STATS

EWM_HALFLIVES: tuple[int, ...] = (5, 20)
PLAYER_VOLUME_COLS: tuple[str, ...] = ("fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted")
PLAYER_COUNT_COLS: tuple[str, ...] = (*PLAYER_STATS, *PLAYER_VOLUME_COLS)
TEAM_COUNT_COLS: tuple[str, ...] = (*TEAM_STATS, "fieldGoalsAttempted", "threePointersAttempted", "possessions")
POSITION_CODES: dict[str, int] = {"G": 0, "F": 1, "C": 2}
POS_PSEUDO_MINUTES = 500.0  # prior strength of the league rate in positional defense
MIN_TEAM_SCORE = 50  # team rows below this are data errors (forfeits, placeholders)

PLAYER_BASE_COLUMNS: tuple[str, ...] = (
    "personId", "gameId", "teamId", "opponentTeamId", "gameDateTimeEst", "game_date", "season",
    "is_playoff", "home", "starter", "startingPosition", "position_group", "age", "heightInches",
    "bodyWeightLbs", "team_rest_days", "team_is_back_to_back", "opp_rest_days", "opp_is_back_to_back",
    "opp_pre_off_rating", "opp_pre_def_rating", "opp_pre_pace", "minutes", *PLAYER_COUNT_COLS,
)


def player_feature_names(stats: tuple[str, ...] = PLAYER_STATS) -> list[str]:
    """Columns the player rate and minutes models read, in a fixed order."""
    form = [f"{c}_rate_ewm{h}" for c in PLAYER_COUNT_COLS for h in EWM_HALFLIVES]
    std = [f"{s}_rate_std" for s in stats]
    opp_pos = [f"{s}_opp_pos" for s in stats]
    names = [
        *form, *std, *opp_pos, *MINUTES_FEATURES,
        "home", "heightInches", "bodyWeightLbs", "pos_code",
        "opp_drtg_rel", "opp_ortg_rel", "opp_pace_rel", "team_ortg_rel", "team_drtg_rel", "team_pace_rel",
        "opp_rest_days", "opp_is_back_to_back", "opp_starter_height", "opp_starter_weight",
    ]
    return list(dict.fromkeys(names))


MINUTES_FEATURES: tuple[str, ...] = (
    "minutes_ewm5", "minutes_ewm20", "minutes_last5", "minutes_last10", "minutes_last20",
    "minutes_trend", "minutes_std", "starter_share10", "games_std", "player_days_since_last_game",
    "days_into_season", "team_rest_days", "team_is_back_to_back", "is_playoff", "age",
    "team_net_rel", "opp_net_rel",
)


# ------------------------------------------------------------------ generic helpers


def lagged_ewm_mean(df: pd.DataFrame, by: str, cols: list[str], halflife: float) -> pd.DataFrame:
    """EWMA of each column over *previous* rows of the same group (current row excluded).

    Missing values (a future game's unknown stats) are skipped, not treated as 0.
    ``df`` must be sorted by ``by`` and then by time.
    """
    shifted = df.groupby(by, sort=False)[cols].shift(1)
    ewm = shifted.groupby(df[by], sort=False).ewm(halflife=halflife, ignore_na=True).mean()
    return ewm.reset_index(level=0, drop=True).reindex(df.index)


def lagged_group_sum(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Sum of each column over earlier rows of the same group (0 for the first row)."""
    vals = df[cols].fillna(0)  # fill first: a future game's unknown stats must add 0, not reset the sum
    return vals.groupby([df[k] for k in keys], sort=False).cumsum() - vals


def ewm_effective_n(n_games: int, halflife: float) -> float:
    """Effective sample size of an EWMA over ``n_games``: (sum w)^2 / sum w^2.

    With a half-life of 10 games the most recent ~30 games carry nearly all the
    weight, so even a 500-game career behaves like about 29 independent games.
    """
    if n_games <= 0:
        return 0.0
    w = 0.5 ** (np.arange(n_games) / halflife)
    return float(w.sum() ** 2 / (w**2).sum())


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


# ------------------------------------------------------------------ league context


def league_rating_means(team_games: pd.DataFrame) -> pd.DataFrame:
    """Per-season league average offensive rating and pace (the scale for relative ratings)."""
    tg = team_games[team_games["possessions"].gt(0)]
    out = tg.groupby("season").agg(league_rtg=("offensiveRating", "mean"), league_pace=("pace", "mean"))
    return out.reset_index()


def season_openers(team_games: pd.DataFrame) -> pd.Series:
    """First game date of each season, indexed by season."""
    return pd.to_datetime(team_games.groupby("season")["game_date"].min())


def add_relative_ratings(df: pd.DataFrame, league: pd.DataFrame, pairs: dict[str, str]) -> pd.DataFrame:
    """Divide pre-game ratings by the season's league average (1.0 = average team).

    ``pairs`` maps output column -> input column. Pace columns are divided by
    league pace, rating columns by league rating.
    """
    out = df.merge(league, on="season", how="left")
    for new, src in pairs.items():
        scale = out["league_pace"] if "pace" in src else out["league_rtg"]
        out[new] = out[src] / scale
    return out.drop(columns=["league_rtg", "league_pace"])


# ------------------------------------------------------------------ player history


def add_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """Form, season-to-date and role features from each player's earlier games.

    ``df`` must be sorted by personId then gameDateTimeEst.
    """
    out = df.copy()
    cols = ["minutes", *PLAYER_COUNT_COLS]
    for h in EWM_HALFLIVES:
        ewm = lagged_ewm_mean(out, "personId", cols, h)
        out[f"minutes_ewm{h}"] = ewm["minutes"]
        for c in PLAYER_COUNT_COLS:
            out[f"{c}_rate_ewm{h}"] = safe_ratio(ewm[c], ewm["minutes"])
    for w in (5, 10, 20):
        out[f"minutes_last{w}"] = lagged_rolling_mean(out, "personId", "minutes", w)
    out["minutes_trend"] = out["minutes_last5"] - out["minutes_last20"]
    out["starter_share10"] = lagged_rolling_mean(out.assign(_s=out["starter"].astype(float)), "personId", "_s", 10)
    sums = lagged_group_sum(out, ["personId", "season"], ["minutes", *PLAYER_STATS])
    out["games_std"] = out.groupby(["personId", "season"], sort=False).cumcount()
    out["minutes_std"] = safe_ratio(sums["minutes"], out["games_std"].astype(float))
    for s in PLAYER_STATS:
        out[f"{s}_rate_std"] = safe_ratio(sums[s], sums["minutes"])
    dates = out["game_date"].dt.normalize()
    out["player_days_since_last_game"] = dates.groupby(out["personId"], sort=False).diff().dt.days
    return out


# ------------------------------------------------------------------ opponent context


def positional_defense_totals(pg: pd.DataFrame, stats: tuple[str, ...] = PLAYER_STATS) -> pd.DataFrame:
    """What each defense allowed to each position group, one row per (team, game, position).

    ``defTeamId`` is the team that was defending (the players' opponent).
    """
    g = pg.dropna(subset=["position_group"])
    agg = g.groupby(["opponentTeamId", "season", "gameDateTimeEst", "position_group"], observed=True)[
        ["minutes", *stats]
    ].sum()
    return agg.reset_index().rename(columns={"opponentTeamId": "defTeamId"})


def positional_defense_cumulative(totals: pd.DataFrame, stats: tuple[str, ...] = PLAYER_STATS) -> pd.DataFrame:
    """Season-to-date allowed-per-minute factors *including* each game, per (defense, position).

    factor = shrunk team rate / league rate, where the shrunk rate adds
    ``POS_PSEUDO_MINUTES`` of league-average play. A defense with no games yet
    is exactly 1.0 (league average); after a few hundred minutes its own
    record dominates. The league rate is the season's rate so far.
    """
    t = totals.sort_values("gameDateTimeEst").copy()
    team_cum = t.groupby(["defTeamId", "season", "position_group"], sort=False)[["minutes", *stats]].cumsum()
    league_day = t.groupby(["season", "position_group", "gameDateTimeEst"])[["minutes", *stats]].sum()
    league_cum = league_day.groupby(level=["season", "position_group"]).cumsum()
    league_cum = league_cum.reindex(pd.MultiIndex.from_frame(t[["season", "position_group", "gameDateTimeEst"]]))
    out = t[["defTeamId", "season", "position_group", "gameDateTimeEst"]].copy()
    for s in stats:
        league_rate = league_cum[s].to_numpy() / league_cum["minutes"].to_numpy()
        shrunk = (team_cum[s].to_numpy() + POS_PSEUDO_MINUTES * league_rate) / (
            team_cum["minutes"].to_numpy() + POS_PSEUDO_MINUTES
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{s}_opp_pos"] = np.where(league_rate > 0, shrunk / league_rate, 1.0)
    return out


def positional_defense_asof(keys: pd.DataFrame, cumulative: pd.DataFrame,
                            stats: tuple[str, ...] = PLAYER_STATS) -> pd.DataFrame:
    """Look up each row's opponent-vs-position factors from games strictly before it.

    ``keys`` needs opponentTeamId, season, position_group, gameDateTimeEst.
    Missing history (first game of a season) is 1.0, i.e. league average.
    """
    cols = [f"{s}_opp_pos" for s in stats]
    left = keys[["opponentTeamId", "season", "position_group", "gameDateTimeEst"]].reset_index()
    left = left.rename(columns={"opponentTeamId": "defTeamId"}).dropna(subset=["position_group"])
    left = left.sort_values("gameDateTimeEst")
    right = cumulative.sort_values("gameDateTimeEst")
    for df in (left, right):
        df["position_group"] = df["position_group"].astype(str)
    merged = pd.merge_asof(left, right, on="gameDateTimeEst", by=["defTeamId", "season", "position_group"],
                           allow_exact_matches=False, direction="backward")
    out = pd.DataFrame(1.0, index=keys.index, columns=cols)
    out.loc[merged["index"].to_numpy(), cols] = merged[cols].fillna(1.0).to_numpy()
    return out


def opponent_starter_size(pg: pd.DataFrame) -> pd.DataFrame:
    """Average height and weight of each team's starters at each starting position, per game.

    Keyed by (gameId, teamId, position) so a player can look up the *opponent's*
    starter(s) at his position.
    """
    st = pg[pg["starter"].astype(bool)].dropna(subset=["startingPosition"])
    agg = st.groupby(["gameId", "teamId", st["startingPosition"].astype(str)], observed=True).agg(
        opp_starter_height=("heightInches", "mean"), opp_starter_weight=("bodyWeightLbs", "mean")
    )
    return agg.reset_index().rename(columns={"teamId": "opponentTeamId", "startingPosition": "position_group"})


def attach_opponent_starters(df: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """Join the opponent's same-position starter size onto each player row (by position group)."""
    keys = df[["gameId", "opponentTeamId"]].assign(position_group=df["position_group"].astype(str))
    joined = keys.merge(sizes.astype({"position_group": str}), on=["gameId", "opponentTeamId", "position_group"],
                        how="left")
    out = df.copy()
    for c in ("opp_starter_height", "opp_starter_weight"):
        out[c] = keep_given(df, c, joined[c].to_numpy(dtype=float))
    return out


def keep_given(df: pd.DataFrame, col: str, looked_up: np.ndarray) -> np.ndarray:
    """Looked-up values, except where ``df`` already has a value (a future game's known context)."""
    if col not in df:
        return looked_up
    given = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(given), looked_up, given)


def own_team_ratings(df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Join the player's own team pre-game ratings (pre_off/def_rating, pre_pace) by game."""
    cols = {"pre_off_rating": "team_pre_off_rating", "pre_def_rating": "team_pre_def_rating", "pre_pace": "team_pre_pace"}
    own = team_games[["gameId", "teamId", *cols]].rename(columns=cols)
    out = df.drop(columns=[c for c in cols.values() if c in df]).merge(own, on=["gameId", "teamId"], how="left")
    for c in cols.values():
        out[c] = keep_given(df, c, out[c].to_numpy(dtype=float))
    return out


def add_context_features(df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Relative ratings, projected margin proxies, calendar and position code."""
    out = add_relative_ratings(df, league_rating_means(team_games), {
        "opp_drtg_rel": "opp_pre_def_rating", "opp_ortg_rel": "opp_pre_off_rating", "opp_pace_rel": "opp_pre_pace",
        "team_ortg_rel": "team_pre_off_rating", "team_drtg_rel": "team_pre_def_rating", "team_pace_rel": "team_pre_pace",
    })
    out["team_net_rel"] = out["team_ortg_rel"] - out["team_drtg_rel"]
    out["opp_net_rel"] = out["opp_ortg_rel"] - out["opp_drtg_rel"]
    openers = season_openers(team_games)
    out["days_into_season"] = (out["game_date"].dt.normalize() - out["season"].map(openers)).dt.days
    out["pos_code"] = out["position_group"].astype(str).map(POSITION_CODES)
    for c in ("home", "is_playoff", "team_is_back_to_back", "opp_is_back_to_back"):
        out[c] = out[c].astype(float)
    return out


# ------------------------------------------------------------------ frames


def prepare_player_games(pg: pd.DataFrame, min_season: int) -> pd.DataFrame:
    """Rows the models learn from: games actually played (minutes > 0), sorted by player and time."""
    cols = [c for c in PLAYER_BASE_COLUMNS if c in pg.columns]
    out = pg.loc[pg["season"].ge(min_season) & pg["minutes"].gt(0), cols].copy()
    out["minutes"] = out["minutes"].astype(float)
    for c in PLAYER_COUNT_COLS:
        out[c] = out[c].astype(float)
    return out.sort_values(["personId", "gameDateTimeEst", "gameId"]).reset_index(drop=True)


def build_player_frame(pg: pd.DataFrame, team_games: pd.DataFrame, min_season: int) -> pd.DataFrame:
    """Full leak-free training frame, one row per player-game played since ``min_season``."""
    base = prepare_player_games(pg, min_season)
    cum = positional_defense_cumulative(positional_defense_totals(base))
    return player_frame_from(base, cum, opponent_starter_size(base), team_games)


def player_frame_from(rows: pd.DataFrame, pos_cumulative: pd.DataFrame, starter_sizes: pd.DataFrame,
                      team_games: pd.DataFrame) -> pd.DataFrame:
    """Every feature step for ``rows`` (sorted player games), given the league-wide lookup tables.

    ``rows`` may be one player's history plus a placeholder future game.
    """
    out = add_player_history(rows)
    out = pd.concat([out, positional_defense_asof(out, pos_cumulative)], axis=1)
    out = attach_opponent_starters(out, starter_sizes)
    out = own_team_ratings(out, team_games)
    return add_context_features(out, team_games)


# ------------------------------------------------------------------ team frames


def prepare_team_games(tg: pd.DataFrame, min_season: int) -> pd.DataFrame:
    """Team-game rows with valid scores, each paired with what the opponent did that game."""
    t = tg.loc[tg["season"].ge(min_season) & tg["teamScore"].ge(MIN_TEAM_SCORE)].copy()
    opp = t[["gameId", "teamId", *TEAM_COUNT_COLS]].rename(
        columns={"teamId": "opponentTeamId", **{c: f"allowed_{c}" for c in TEAM_COUNT_COLS}}
    )
    t = t.merge(opp, on=["gameId", "opponentTeamId"], how="inner")
    for c in (*TEAM_COUNT_COLS, *(f"allowed_{c}" for c in TEAM_COUNT_COLS)):
        t[c] = t[c].astype(float)
    return t.sort_values(["teamId", "gameDateTimeEst", "gameId"]).reset_index(drop=True)


def team_feature_names(stats: tuple[str, ...] = TEAM_STATS) -> list[str]:
    """Columns the team models read."""
    own = [f"{c}_ewm{h}" for c in TEAM_COUNT_COLS for h in EWM_HALFLIVES]
    allowed = [f"opp_allowed_{c}_ewm{h}" for c in TEAM_COUNT_COLS for h in EWM_HALFLIVES]
    std = [f"{s}_std" for s in stats]
    return [
        *own, *allowed, *std, "home", "is_playoff", "team_rest_days", "team_is_back_to_back",
        "opp_rest_days", "opp_is_back_to_back", "days_into_season",
        "opp_drtg_rel", "opp_ortg_rel", "opp_pace_rel", "team_ortg_rel", "team_drtg_rel", "team_pace_rel",
    ]


def add_team_history(t: pd.DataFrame) -> pd.DataFrame:
    """Team form (own stats and stats allowed) and season-to-date averages from earlier games."""
    out = t.copy()
    allowed = [f"allowed_{c}" for c in TEAM_COUNT_COLS]
    for h in EWM_HALFLIVES:
        ewm = lagged_ewm_mean(out, "teamId", [*TEAM_COUNT_COLS, *allowed], h)
        for c in [*TEAM_COUNT_COLS, *allowed]:
            out[f"{c}_ewm{h}"] = ewm[c]
    sums = lagged_group_sum(out, ["teamId", "season"], list(TEAM_STATS))
    n = out.groupby(["teamId", "season"], sort=False).cumcount().astype(float)
    for s in TEAM_STATS:
        out[f"{s}_std"] = safe_ratio(sums[s], n)
    return out


def attach_opponent_allowed(t: pd.DataFrame) -> pd.DataFrame:
    """Each row gets the *opponent's* pre-game form at allowing each stat."""
    cols = [f"allowed_{c}_ewm{h}" for c in TEAM_COUNT_COLS for h in EWM_HALFLIVES]
    opp = t[["gameId", "teamId", *cols]].rename(columns={"teamId": "opponentTeamId", **{c: f"opp_{c}" for c in cols}})
    return t.merge(opp, on=["gameId", "opponentTeamId"], how="left")


def build_team_frame(tg: pd.DataFrame, min_season: int) -> pd.DataFrame:
    """Full leak-free team training frame."""
    return finish_team_frame(attach_opponent_allowed(add_team_history(prepare_team_games(tg, min_season))), tg)


def finish_team_frame(t: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Relative ratings, calendar and numeric flags (shared by training and future rows)."""
    t = t.rename(columns={"pre_off_rating": "team_pre_off_rating", "pre_def_rating": "team_pre_def_rating",
                          "pre_pace": "team_pre_pace"})
    t = add_relative_ratings(t, league_rating_means(tg), {
        "opp_drtg_rel": "opp_pre_def_rating", "opp_ortg_rel": "opp_pre_off_rating", "opp_pace_rel": "opp_pre_pace",
        "team_ortg_rel": "team_pre_off_rating", "team_drtg_rel": "team_pre_def_rating", "team_pace_rel": "team_pre_pace",
    })
    t["days_into_season"] = (t["game_date"].dt.normalize() - t["season"].map(season_openers(tg))).dt.days
    for c in ("home", "is_playoff", "team_is_back_to_back", "opp_is_back_to_back"):
        t[c] = t[c].astype(float)
    return t


# ------------------------------------------------------------------ future games


def future_team_ratings(team_games: pd.DataFrame, team_id: int, opp_id: int | None, season: int,
                        when: pd.Timestamp) -> dict[str, float]:
    """Pre-game season-to-date ratings for the subject's team and the opponent at ``when``."""
    cum = cumulative_team_ratings(team_games[team_games["season"].between(season - 1, season)])
    cum = cum.astype({"teamId": "int64", "season": "int64"})
    ids = [team_id] + ([opp_id] if opp_id is not None else [])
    keys = pd.DataFrame({"gameId": -1, "teamId": ids, "season": season, "gameDateTimeEst": when})
    keys = keys.astype({"teamId": "int64", "season": "int64", "gameDateTimeEst": cum["gameDateTimeEst"].dtype})
    r = pregame_team_ratings(keys, cum).set_index("teamId")
    out = {"team_pre_off_rating": r.at[team_id, "std_off_rating"], "team_pre_def_rating": r.at[team_id, "std_def_rating"],
           "team_pre_pace": r.at[team_id, "std_pace"]}
    if opp_id is not None:
        out |= {"opp_pre_off_rating": r.at[opp_id, "std_off_rating"], "opp_pre_def_rating": r.at[opp_id, "std_def_rating"],
                "opp_pre_pace": r.at[opp_id, "std_pace"]}
    else:  # league-average opponent
        lg = league_rating_means(team_games).set_index("season").loc[season]
        out |= {"opp_pre_off_rating": lg["league_rtg"], "opp_pre_def_rating": lg["league_rtg"],
                "opp_pre_pace": lg["league_pace"]}
    return {k: float(v) for k, v in out.items()}


def last_starters_size(pg: pd.DataFrame, team_id: int, position: str) -> tuple[float, float]:
    """Height/weight of the team's starter(s) at ``position`` in its most recent game."""
    t = pg[pg["teamId"].eq(team_id)]
    if t.empty:
        return float("nan"), float("nan")
    last = t[t["gameDateTimeEst"].eq(t["gameDateTimeEst"].max())]
    st = last[last["starter"].astype(bool) & last["startingPosition"].astype(str).eq(position)]
    return float(st["heightInches"].mean()), float(st["bodyWeightLbs"].mean())
