"""Turn relative time phrases into NBA seasons.

Seasons are named by their *starting* calendar year everywhere in this
project: season 2024 is 2024-25. A regular season opens in October and the
Finals end in June, so:

- the **current season** is the one that started most recently (from October
  on, this calendar year's; before that, last year's). Between July and
  September it is the season that just finished.
- the **last completed season** is the most recent one whose Finals are over
  (from July on, the season that started last calendar year).

"Last 5 years" means the last five *completed* seasons, so a half-played
season never sneaks in and makes the window look shorter than it is.
"""

from __future__ import annotations

from datetime import date

SEASON_START_MONTH = 10  # October: opening night
SEASON_END_MONTH = 6  # June: the Finals


def current_season(today: date) -> int:
    """Starting year of the season in progress, or the one just finished in the offseason."""
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def last_completed_season(today: date) -> int:
    """Starting year of the most recent season whose Finals are over."""
    return today.year - 1 if today.month > SEASON_END_MONTH else today.year - 2


def last_n_completed(n: int, today: date) -> tuple[int, int]:
    """(start, end) starting years of the last ``n`` completed seasons, inclusive."""
    end = last_completed_season(today)
    return end - n + 1, end


def season_label(start_year: int) -> str:
    """``2024`` -> ``"2024-25"``."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"
