"""Catalog of the stats a query can ask for, and how to compute each one per game.

There are two kinds of stat:

- **Counting stats** (points, assists, PRA, ...). The per-game value is read
  or summed from the box score. The average over a split is the ordinary mean
  of the per-game values. The per-36-minute rate is total stat / total minutes x 36.
- **Ratio stats** (FG%, TS%, ...). The per-game value is made / attempted,
  which is undefined in games with zero attempts. The honest summary over many
  games is the **pooled** ratio (total made / total attempted), because it
  weights each game by its attempts. The mean of per-game percentages is also
  reported (the t-test runs on it), but the pooled number is the headline.
  Per-36 rates do not apply to ratios.

True shooting % = points / (2 x (FGA + 0.44 x FTA)). The 0.44 turns free-throw
attempts into possessions, because and-ones and technicals mean not every pair
of free throws ends a possession.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Column = Callable[[pd.DataFrame], pd.Series]


def col(name: str) -> Column:
    return lambda df: df[name].astype("float64")


def total(*names: str) -> Column:
    return lambda df: sum(df[n].astype("float64") for n in names)


def ts_attempts(fga: str, fta: str) -> Column:
    return lambda df: 2.0 * (df[fga].astype("float64") + 0.44 * df[fta].astype("float64"))


@dataclass(frozen=True)
class StatDef:
    """How to compute one stat from a games table.

    ``value`` gives the per-game number for counting stats and the numerator for
    ratios. ``denominator`` is set only for ratio stats. ``per36`` says if a
    per-36-minute rate makes sense.
    """

    name: str
    label: str
    value: Column
    denominator: Column | None = None
    per36: bool = True
    scale: float = 1.0
    higher_is_better: bool = True  # False for turnovers, fouls, defensive rating, points allowed

    @property
    def is_ratio(self) -> bool:
        return self.denominator is not None

    def per_game(self, df: pd.DataFrame) -> pd.Series:
        """One value per game (NaN for a ratio stat with zero attempts)."""
        v = self.value(df)
        if self.denominator is None:
            return v * self.scale
        d = self.denominator(df)
        return (v / d.where(d > 0)) * self.scale

    def pooled(self, df: pd.DataFrame) -> float:
        """Total numerator / total denominator for ratio stats, else the plain mean."""
        if self.denominator is None:
            return float(self.per_game(df).mean()) if len(df) else float("nan")
        d = float(self.denominator(df).sum())
        return float(self.value(df).sum()) / d * self.scale if d > 0 else float("nan")


PLAYER_STATS: dict[str, StatDef] = {
    s.name: s
    for s in [
        StatDef("points", "Points", col("points")),
        StatDef("assists", "Assists", col("assists")),
        StatDef("rebounds", "Rebounds", col("reboundsTotal")),
        StatDef("offensive_rebounds", "Offensive rebounds", col("reboundsOffensive")),
        StatDef("defensive_rebounds", "Defensive rebounds", col("reboundsDefensive")),
        StatDef("threes", "3-pointers made", col("threePointersMade")),
        StatDef("threes_attempted", "3-pointers attempted", col("threePointersAttempted")),
        StatDef("steals", "Steals", col("steals")),
        StatDef("blocks", "Blocks", col("blocks")),
        StatDef("stocks", "Steals + blocks", total("steals", "blocks")),
        StatDef("turnovers", "Turnovers", col("turnovers"), higher_is_better=False),
        StatDef("fouls", "Personal fouls", col("foulsPersonal"), higher_is_better=False),
        StatDef("minutes", "Minutes", col("minutes"), per36=False),
        StatDef("fgm", "Field goals made", col("fieldGoalsMade")),
        StatDef("fga", "Field goals attempted", col("fieldGoalsAttempted")),
        StatDef("ftm", "Free throws made", col("freeThrowsMade")),
        StatDef("fta", "Free throws attempted", col("freeThrowsAttempted")),
        StatDef("plus_minus", "Plus/minus", col("plusMinusPoints")),
        StatDef("pra", "Points + rebounds + assists", total("points", "reboundsTotal", "assists")),
        StatDef("pr", "Points + rebounds", total("points", "reboundsTotal")),
        StatDef("pa", "Points + assists", total("points", "assists")),
        StatDef("ra", "Rebounds + assists", total("reboundsTotal", "assists")),
        StatDef("fg_pct", "FG%", col("fieldGoalsMade"), col("fieldGoalsAttempted"), False, 100.0),
        StatDef("three_pct", "3P%", col("threePointersMade"), col("threePointersAttempted"), False, 100.0),
        StatDef("ft_pct", "FT%", col("freeThrowsMade"), col("freeThrowsAttempted"), False, 100.0),
        StatDef(
            "efg_pct", "eFG%",
            lambda df: df["fieldGoalsMade"].astype("float64") + 0.5 * df["threePointersMade"].astype("float64"),
            col("fieldGoalsAttempted"), False, 100.0,
        ),
        StatDef("ts_pct", "TS%", col("points"), ts_attempts("fieldGoalsAttempted", "freeThrowsAttempted"), False, 100.0),
    ]
}

TEAM_STATS: dict[str, StatDef] = {
    s.name: s
    for s in [
        StatDef("team_score", "Team points", col("teamScore"), per36=False),
        StatDef("opponent_score", "Opponent points", col("opponentScore"), per36=False, higher_is_better=False),
        StatDef("total_points", "Total points (both teams)", total("teamScore", "opponentScore"), per36=False),
        StatDef("margin", "Point margin", lambda df: df["teamScore"].astype("float64") - df["opponentScore"].astype("float64"), per36=False),
        StatDef("assists", "Assists", col("assists"), per36=False),
        StatDef("rebounds", "Rebounds", col("reboundsTotal"), per36=False),
        StatDef("threes", "3-pointers made", col("threePointersMade"), per36=False),
        StatDef("threes_attempted", "3-pointers attempted", col("threePointersAttempted"), per36=False),
        StatDef("steals", "Steals", col("steals"), per36=False),
        StatDef("blocks", "Blocks", col("blocks"), per36=False),
        StatDef("turnovers", "Turnovers", col("turnovers"), per36=False, higher_is_better=False),
        StatDef("pace", "Pace", col("pace"), per36=False),
        StatDef("possessions", "Possessions", col("possessions"), per36=False),
        StatDef("off_rating", "Offensive rating", col("offensiveRating"), per36=False),
        StatDef("def_rating", "Defensive rating", col("defensiveRating"), per36=False, higher_is_better=False),
        StatDef("net_rating", "Net rating", col("netRating"), per36=False),
        StatDef("win_pct", "Win %", col("win"), per36=False, scale=100.0),
        StatDef("fg_pct", "FG%", col("fieldGoalsMade"), col("fieldGoalsAttempted"), False, 100.0),
        StatDef("three_pct", "3P%", col("threePointersMade"), col("threePointersAttempted"), False, 100.0),
        StatDef("ft_pct", "FT%", col("freeThrowsMade"), col("freeThrowsAttempted"), False, 100.0),
        StatDef("ts_pct", "TS%", col("teamScore"), ts_attempts("fieldGoalsAttempted", "freeThrowsAttempted"), False, 100.0),
    ]
}

# Everyday spellings that map to a canonical stat name. Keys are normalized
# with :func:`_norm` (lowercase, only letters, digits and %).
STAT_ALIASES: dict[str, str] = {
    "pts": "points", "ppg": "points", "point": "points", "scoring": "points",
    "ast": "assists", "apg": "assists", "assist": "assists", "dimes": "assists",
    "reb": "rebounds", "rpg": "rebounds", "rebound": "rebounds", "boards": "rebounds",
    "3pm": "threes", "3s": "threes", "threepointersmade": "threes", "3pointers": "threes", "three": "threes",
    "3pa": "threes_attempted",
    "stl": "steals", "spg": "steals", "blk": "blocks", "bpg": "blocks",
    "tov": "turnovers", "to": "turnovers", "min": "minutes", "mpg": "minutes", "mins": "minutes",
    "pts+reb+ast": "pra", "p+r+a": "pra", "pointsreboundsassists": "pra",
    "fg%": "fg_pct", "fgpct": "fg_pct", "fieldgoal%": "fg_pct",
    "3p%": "three_pct", "3pt%": "three_pct", "ft%": "ft_pct",
    "ts%": "ts_pct", "ts": "ts_pct", "trueshooting": "ts_pct", "efg%": "efg_pct",
    "+/-": "plus_minus", "plusminus": "plus_minus",
    "teamscore": "team_score", "teampoints": "team_score",
    "opponentscore": "opponent_score", "pointsallowed": "opponent_score",
    "total": "total_points", "ortg": "off_rating", "drtg": "def_rating",
    "win%": "win_pct", "winpct": "win_pct", "winpercentage": "win_pct", "wins": "win_pct", "record": "win_pct",
    "netrtg": "net_rating", "nrtg": "net_rating",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9%+/_-]", "", name.lower()).replace("_", "")


def stat_catalog(subject_type: str) -> dict[str, StatDef]:
    return TEAM_STATS if subject_type == "team" else PLAYER_STATS


def stat_names_for(subject_type: str) -> set[str]:
    return set(stat_catalog(subject_type))


def canonical_stat(name: str, subject_type: str = "player") -> str:
    """Map a user spelling ("PRA", "FG%", "ppg") to the catalog name.

    For a team, "points" means the team's score.
    """
    catalog = stat_catalog(subject_type)
    if name in catalog:
        return name
    key = _norm(name)
    by_key = {_norm(k): k for k in catalog}
    target = by_key.get(key) or STAT_ALIASES.get(key) or STAT_ALIASES.get(key.rstrip("s"))
    if subject_type == "team" and target == "points":
        target = "team_score"
    if target not in catalog:
        raise ValueError(f"unknown {subject_type} stat {name!r}; valid: {sorted(catalog)}")
    return target


# Variables a split can be grouped by. Values are the engine column they read.
GROUPABLE_VARIABLES: dict[str, str] = {
    "day_of_week": "day_of_week",
    "month": "month",
    "week_of_season": "week_of_season",
    "season": "season_label",
    "home_away": "home_away",
    "opponent_team": "opponent_name",
    "venue": "venue_city",
    "rest_days": "rest_bucket",
    "back_to_back": "back_to_back",
    "playoffs": "playoff_label",
    "starter": "starter",
    "defender_height_bucket": "def_height_bucket",
    "defender_weight_bucket": "def_weight_bucket",
    "defender_position": "def_position",
}


def rest_bucket(rest_days: pd.Series) -> pd.Series:
    """0, 1, 2, 3+ days of rest (season openers have no rest value)."""
    r = rest_days.astype("float64")
    out = np.where(r >= 3, "3+", r.fillna(-1).astype(int).astype(str))
    return pd.Series(np.where(r.isna(), None, out), index=rest_days.index, dtype="object")


def height_bucket(inches: pd.Series) -> pd.Series:
    """Two-inch bands labeled in feet and inches, e.g. ``6'6"-6'7"``."""
    h = inches.astype("float64")
    lo = (np.floor(h / 2) * 2)

    def fmt(x: float) -> str:
        return f"{int(x) // 12}'{int(x) % 12}\""

    labels = [None if np.isnan(v) else f"{fmt(v)}-{fmt(v + 1)}" for v in lo]
    return pd.Series(labels, index=inches.index, dtype="object")


def weight_bucket(lbs: pd.Series) -> pd.Series:
    """20-lb bands, e.g. ``200-219 lb``."""
    w = lbs.astype("float64")
    lo = np.floor(w / 20) * 20
    labels = [None if np.isnan(v) else f"{int(v)}-{int(v) + 19} lb" for v in lo]
    return pd.Series(labels, index=lbs.index, dtype="object")
