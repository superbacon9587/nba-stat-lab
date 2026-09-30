"""Default betting lines when the prompt asks for a projection but gives none.

A sportsbook line sits near what the player usually does, and ends in .5 so
there is no push. The default here is the subject's average over their most
recent games (20 by default), moved to the nearest number ending in .5:
26.7 -> 26.5, 26.2 -> 26.5, 25.9 -> 25.5. Ratio stats (FG%) use the pooled
percentage over those games (total made / total attempted).

This is computed from the processed tables by ordinary code. It is only a
starting point: the app shows it as an editable assumption.
"""

from __future__ import annotations

import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import pandas as pd

from nbalab.data.config import DATA_DIR
from nbalab.query.stats import stat_catalog

PROCESSED_DIR = DATA_DIR / "processed"


class LineProvider(Protocol):
    def __call__(self, subject_type: str, subject_id: int, stat: str, today: date) -> float | None: ...


def half_point_line(value: float) -> float:
    """Nearest number ending in .5: ``26.7 -> 26.5``, ``25.9 -> 25.5``."""
    return float(math.floor(value) + 0.5)


def recent_value(games: pd.DataFrame, subject_type: str, stat: str, n_games: int) -> float | None:
    """The stat over the ``n_games`` most recent rows (mean, or pooled for ratios)."""
    if games.empty:
        return None
    recent = games.sort_values("game_date").tail(n_games)
    value = stat_catalog(subject_type)[stat].pooled(recent)
    return None if math.isnan(value) else value


class RecentAverageLines:
    """Line provider backed by ``player_games`` / ``team_games`` parquet files."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR, n_games: int = 20) -> None:
        self.processed_dir = Path(processed_dir)
        self.n_games = n_games

    def __call__(self, subject_type: str, subject_id: int, stat: str, today: date) -> float | None:
        games = _subject_games(self.processed_dir, subject_type, subject_id, today)
        if games is None:
            return None
        value = recent_value(games, subject_type, stat, self.n_games)
        return None if value is None else half_point_line(value)


@lru_cache(maxsize=64)
def _subject_games(processed_dir: Path, subject_type: str, subject_id: int, today: date) -> pd.DataFrame | None:
    """One subject's games played before ``today``, or ``None`` if the table is not built."""
    name, key = ("team_games", "teamId") if subject_type == "team" else ("player_games", "personId")
    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, filters=[(key, "==", subject_id)])
    if "game_date" in df:
        df = df[pd.to_datetime(df["game_date"]) < pd.Timestamp(today)]
    return df
