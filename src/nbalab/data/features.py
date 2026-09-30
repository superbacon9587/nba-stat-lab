"""Feature builders shared by the player-game and team-game tables.

The rule throughout: a feature attached to a game may only use information
available *before* tip-off. Rolling averages are shifted by one game and
season-to-date ratings are looked up strictly before the game's timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_rolling_mean(
    df: pd.DataFrame, by: str, value: str, window: int
) -> pd.Series:
    """Average of ``value`` over the previous ``window`` rows of the same group.

    The current row is never included: the series is shifted by one within the
    group before averaging. Early rows with fewer than ``window`` prior games
    average whatever history exists (e.g. the 3rd game averages games 1-2);
    the very first row of a group has no history and is NaN. ``df`` must already
    be sorted by ``by`` and then by time.
    """
    shifted = df.groupby(by, sort=False)[value].shift(1)
    rolled = shifted.groupby(df[by], sort=False).rolling(window, min_periods=1).mean()
    return rolled.reset_index(level=0, drop=True).reindex(df.index)


def add_lagged_rolling_means(
    df: pd.DataFrame,
    by: str,
    stats: tuple[str, ...],
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """Add ``<stat>_last<N>`` columns: the mean of each stat over the previous N games."""
    out = df.copy()
    for stat in stats:
        for window in windows:
            out[f"{stat}_last{window}"] = lagged_rolling_mean(out, by, stat, window)
    return out


def days_since_previous(df: pd.DataFrame, by: list[str], date_col: str) -> pd.Series:
    """Calendar days between each row's date and the previous row's date in the same group.

    NaN for the first row of each group. ``df`` must be sorted by ``by`` then date.
    """
    dates = df[date_col].dt.normalize()
    return dates.groupby([df[c] for c in by], sort=False).diff().dt.days


def rest_features(days_since: pd.Series, prefix: str) -> pd.DataFrame:
    """Turn "days since last game" into rest days and a back-to-back flag.

    Playing on consecutive days is 1 day since the last game, 0 rest days, and a
    back-to-back. The first game of a season has unknown rest (NaN, not a B2B).
    """
    return pd.DataFrame(
        {
            f"{prefix}days_since_last_game": days_since,
            f"{prefix}rest_days": days_since - 1,
            f"{prefix}is_back_to_back": days_since.eq(1),
        },
        index=days_since.index,
    )


def cumulative_team_ratings(team_games: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date offensive rating, defensive rating and pace *including* each game.

    Ratings are possession-weighted, the standard way to combine games:
    offensive rating = 100 x total points scored / total possessions, and
    defensive rating = 100 x total points allowed / total possessions (a team's
    and its opponent's possessions in a game are near-identical). Pace is the
    plain average of per-game pace (possessions per 48 minutes). Games without
    advanced stats are skipped. Needs columns: teamId, season, gameDateTimeEst,
    teamScore, opponentScore, possessions, pace.
    """
    cols = ["teamId", "season", "gameDateTimeEst", "teamScore", "opponentScore", "possessions", "pace"]
    g = team_games.loc[team_games["possessions"].gt(0), cols].sort_values(
        ["teamId", "season", "gameDateTimeEst"]
    )
    grp = g.groupby(["teamId", "season"], sort=False)
    poss = grp["possessions"].cumsum()
    return pd.DataFrame(
        {
            "teamId": g["teamId"],
            "season": g["season"],
            "gameDateTimeEst": g["gameDateTimeEst"],
            "std_off_rating": 100 * grp["teamScore"].cumsum() / poss,
            "std_def_rating": 100 * grp["opponentScore"].cumsum() / poss,
            "std_pace": grp["pace"].cumsum() / (grp.cumcount() + 1),
            "std_games": grp.cumcount() + 1,
        }
    )


def pregame_team_ratings(keys: pd.DataFrame, cumulative: pd.DataFrame) -> pd.DataFrame:
    """Each team's season-to-date ratings from games strictly *before* each game.

    ``keys`` has one row per (gameId, teamId) with season and gameDateTimeEst.
    For a team's first game of a season there is no current-season history, so
    the team's full previous-season values are used instead (``std_games`` = 0
    marks these). Expansion teams in their first game stay NaN.
    """
    rating_cols = ["std_off_rating", "std_def_rating", "std_pace"]
    left = keys[["gameId", "teamId", "season", "gameDateTimeEst"]].sort_values("gameDateTimeEst")
    right = cumulative.sort_values("gameDateTimeEst")
    asof = pd.merge_asof(
        left,
        right,
        on="gameDateTimeEst",
        by=["teamId", "season"],
        allow_exact_matches=False,
        direction="backward",
    )
    prior_final = (
        cumulative.sort_values("gameDateTimeEst")
        .groupby(["teamId", "season"], as_index=False)
        .last()[["teamId", "season", *rating_cols]]
        .assign(season=lambda d: d["season"] + 1)
    )
    asof = asof.merge(prior_final, on=["teamId", "season"], how="left", suffixes=("", "_prev"))
    for col in rating_cols:
        asof[col] = asof[col].fillna(asof[f"{col}_prev"])
    asof["std_games"] = asof["std_games"].fillna(0).astype("int64")
    return asof[["gameId", "teamId", *rating_cols, "std_games"]]


def calendar_features(dates: pd.Series, season_opener: pd.Series) -> pd.DataFrame:
    """Day of week, month, and week of season for each game date.

    Week of season counts 7-day blocks from the season's first regular-season
    game (opening night is week 1). Playoff games continue the count.
    """
    dates = pd.to_datetime(dates)
    days_in = (dates.dt.normalize() - pd.to_datetime(season_opener).dt.normalize()).dt.days
    return pd.DataFrame(
        {
            "day_of_week": dates.dt.day_name().astype("category"),
            "day_of_week_num": dates.dt.dayofweek.astype("int8"),
            "month": dates.dt.month.astype("int8"),
            "week_of_season": (np.floor_divide(days_in, 7) + 1).astype("Int16"),
        },
        index=dates.index,
    )
