"""Paths and tunable settings for the data pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("NBALAB_DATA_DIR", PROJECT_ROOT / "data"))
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# Game types kept for analysis. Preseason and All-Star games are excluded by default.
INCLUDED_GAME_TYPES: tuple[str, ...] = ("regular", "nba_cup", "play_in", "playoffs")


@dataclass(frozen=True)
class BuildConfig:
    """Settings for one pipeline run.

    ``start_season`` is the starting calendar year of the first season kept
    (1996 means the 1996-97 season onward).
    """

    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    start_season: int = 1996
    included_game_types: tuple[str, ...] = INCLUDED_GAME_TYPES
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    player_rolling_stats: tuple[str, ...] = ("points", "assists", "reboundsTotal", "minutes")
    team_rolling_stats: tuple[str, ...] = (
        "teamScore",
        "opponentScore",
        "offensiveRating",
        "defensiveRating",
        "pace",
    )
    extra_schedule_files: tuple[str, ...] = field(
        default=("LeagueSchedule24_25.csv", "LeagueSchedule25_26.csv")
    )
