"""Cleaning helpers: game types, seasons, minutes, positions.

Every function here is pure (DataFrame/Series in, DataFrame/Series out) so it
can be unit tested without the raw files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw gameType label (lowercased, stripped) -> normalized game type.
GAME_TYPE_LABELS: dict[str, str] = {
    "regular season": "regular",
    "playoffs": "playoffs",
    "play-in tournament": "play_in",
    "preseason": "preseason",
    "pre season": "preseason",
    "all-star game": "all_star",
    "nba emirates cup": "nba_cup",
    "emirates nba cup": "nba_cup",
    "nba cup": "nba_cup",
    "in-season-knockout": "nba_cup",
    "in-season tournament": "nba_cup",
}

# First digit of the 8-digit gameId -> normalized game type (official NBA encoding).
GAME_TYPE_BY_ID_PREFIX: dict[int, str] = {
    1: "preseason",
    2: "regular",
    3: "all_star",
    4: "playoffs",
    5: "play_in",
    6: "nba_cup",  # NBA Cup championship game; does not count in regular-season stats
}

# Seasons start in October. The one exception inside the data is the 2020 bubble,
# whose Finals ran into October 2020 but belong to 2019-20; the gameId handles it.
SEASON_START_MONTH = 10


def normalize_game_type_label(label: pd.Series) -> pd.Series:
    """Map messy raw ``gameType`` text to a normalized value (NaN if unknown/missing)."""
    return label.astype("string").str.strip().str.lower().map(GAME_TYPE_LABELS)


def game_type_from_id(game_id: pd.Series) -> pd.Series:
    """Read the game type from the first digit of the 8-digit gameId."""
    prefix = (game_id.astype("int64") // 10_000_000).astype("int64")
    return prefix.map(GAME_TYPE_BY_ID_PREFIX)


def resolve_game_type(game_id: pd.Series, *labels: pd.Series) -> pd.Series:
    """Combine the gameId prefix with raw text labels into one normalized game type.

    The gameId prefix decides the family (preseason, playoffs, ...). Text labels
    are only consulted for regular-season ids, because NBA Cup group/knockout
    games share the regular-season prefix and are only identifiable by label.
    Labels are tried in order; the first non-null normalized label wins.
    """
    by_id = game_type_from_id(game_id)
    by_label = pd.Series(pd.NA, index=game_id.index, dtype="object")
    for label in labels:
        by_label = by_label.fillna(normalize_game_type_label(label))
    is_cup = (by_id == "regular") & (by_label == "nba_cup")
    return by_id.where(~is_cup, "nba_cup")


def season_from_game_id(game_id: pd.Series) -> pd.Series:
    """Starting year of the season, read from digits 2-3 of the gameId.

    ``22400123`` -> 2024 (the 2024-25 season). Two-digit years 46-99 are 1900s.
    """
    yy = (game_id.astype("int64") // 100_000) % 100
    return (yy + np.where(yy >= 46, 1900, 2000)).astype("int64")


def season_from_date(dates: pd.Series) -> pd.Series:
    """Starting year of the season a game date falls in.

    NBA seasons span two calendar years: October-December games belong to the
    season starting that year, January-September games to the season that
    started the previous year (a January 2025 game is in 2024-25).
    """
    dates = pd.to_datetime(dates)
    return (dates.dt.year - (dates.dt.month < SEASON_START_MONTH).astype("int64")).astype("int64")


def season_label(start_year: pd.Series | int) -> pd.Series | str:
    """Format a season's starting year as ``"2024-25"``."""
    if isinstance(start_year, (int, np.integer)):
        return f"{start_year}-{(start_year + 1) % 100:02d}"
    return start_year.astype("int64").map(lambda y: f"{y}-{(y + 1) % 100:02d}")


def parse_minutes(raw: pd.Series) -> pd.Series:
    """Parse minutes played from decimal strings (``"24.5"``) or clock strings (``"22:29"``)."""
    text = raw.astype("string").str.strip()
    decimal = pd.to_numeric(text, errors="coerce")
    clock = text.str.extract(r"^(\d+):(\d{1,2})$").astype("float64")
    from_clock = clock[0] + clock[1] / 60.0
    return decimal.fillna(from_clock).astype("float64")


def position_group(
    guard: pd.Series,
    forward: pd.Series,
    center: pd.Series,
    starting_mode: pd.Series | None = None,
) -> pd.Series:
    """Collapse the guard/forward/center flags into one group: ``G``, ``F`` or ``C``.

    Players with exactly one flag get that group. Dual-position players (G-F or
    F-C) take the position they most often started at when known, otherwise the
    bigger of the two positions (G-F -> F, F-C -> C). Players with no flags fall
    back to their most common starting position, else NaN.
    """
    g, f, c = (s.fillna(0).astype(bool) for s in (guard, forward, center))
    bigger = pd.Series(
        np.select([c, f, g], ["C", "F", "G"], default=None), index=guard.index, dtype="object"
    )
    n_flags = g.astype(int) + f.astype(int) + c.astype(int)
    if starting_mode is None:
        return bigger
    mode = starting_mode.reindex(guard.index)
    flag_for_mode = pd.Series(
        np.select([mode == "G", mode == "F", mode == "C"], [g, f, c], default=False),
        index=guard.index,
    )
    use_mode = (n_flags == 0) | ((n_flags > 1) & flag_for_mode)
    return mode.where(use_mode, bigger)


def age_in_years(birth_date: pd.Series, on_date: pd.Series) -> pd.Series:
    """Age in fractional years on a given date (days / 365.25).

    ``1900-01-01`` is a placeholder in ``Players.csv`` and is treated as unknown.
    """
    birth = pd.to_datetime(birth_date, errors="coerce")
    birth = birth.where(birth > pd.Timestamp("1900-01-01"))
    return (pd.to_datetime(on_date) - birth).dt.days / 365.25
