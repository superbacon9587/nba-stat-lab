"""Which processed tables and columns the deployed app ships with.

This is the single list to edit when the app starts reading a new column.
``tests/test_deploy.py`` fails if the query code names a column of a shipped
table that is missing here, so drift gets caught before a deploy.

What is left out, and why:

- ``on_court_players`` and ``on_court_quality``: diagnostics from the
  play-by-play reconstruction. Only the pipeline reads them.
- ``on_court``: team ids, game type, the duplicate seconds column, and rows
  the loader discards as unreliable.
- ``matchups``: every column except the four the loader reads (plus season).
- ``player_games`` / ``team_games``: the ``*_days_since_last_game`` helper
  columns the rest-day features were derived from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa


@dataclass(frozen=True)
class TableSpec:
    """One shipped table: the columns kept, optional lossless-enough downcasts,
    sort order, and a boolean column whose false rows are dropped."""

    columns: tuple[str, ...]
    downcast: dict[str, pa.DataType] = field(default_factory=dict)
    sort_by: tuple[str, ...] = ()
    keep_if_true: str | None = None


_IDENTITY_AND_CALENDAR: tuple[str, ...] = (
    "gameId", "opponentTeamId", "gameDateTimeEst", "season", "season_label", "game_date", "game_type",
    "is_playoff", "is_play_in", "day_of_week", "day_of_week_num", "month", "week_of_season",
    "venue_team_id", "venue_city", "arena_name", "arena_city", "home", "win",
)
_SCHEDULE_CONTEXT: tuple[str, ...] = (
    "team_game_num", "team_rest_days", "team_is_back_to_back",
    "opp_rest_days", "opp_is_back_to_back",
    "opp_pre_off_rating", "opp_pre_def_rating", "opp_pre_pace", "opp_pre_games",
)
_BOX_SHOOTING: tuple[str, ...] = (
    "assists", "reboundsTotal", "reboundsOffensive", "reboundsDefensive", "steals", "blocks",
    "turnovers", "foulsPersonal", "fieldGoalsMade", "fieldGoalsAttempted",
    "threePointersMade", "threePointersAttempted", "freeThrowsMade", "freeThrowsAttempted",
)


def _rolling(stats: tuple[str, ...], windows: tuple[int, ...] = (5, 10, 20)) -> tuple[str, ...]:
    return tuple(f"{s}_last{w}" for s in stats for w in windows)


TABLES: dict[str, TableSpec] = {
    "player_games": TableSpec(
        columns=(
            "personId", "teamId", *_IDENTITY_AND_CALENDAR, *_SCHEDULE_CONTEXT,
            "starter", "startingPosition", "minutes", "points", *_BOX_SHOOTING, "plusMinusPoints",
            "heightInches", "bodyWeightLbs", "age", "position_group",
            "player_game_num", "player_days_since_last_game",
            *_rolling(("points", "assists", "reboundsTotal", "minutes")),
        ),
        sort_by=("personId", "gameDateTimeEst"),
    ),
    "team_games": TableSpec(
        columns=(
            "teamId", *_IDENTITY_AND_CALENDAR, *_SCHEDULE_CONTEXT,
            "teamScore", "opponentScore", *_BOX_SHOOTING,
            "offensiveRating", "defensiveRating", "netRating", "pace", "possessions",
            "effectiveFieldGoalPercentage", "trueShootingPercentage", "teamTurnoverPercentage",
            "offensiveReboundPercentage", "freeThrowAttemptRate",
            "opponentEffectiveFieldGoalPercentage", "opponentTurnoverPercentage",
            "opponentOffensiveReboundPercentage", "opponentFreeThrowAttemptRate",
            "pre_off_rating", "pre_def_rating", "pre_pace", "pre_games",
            *_rolling(("teamScore", "opponentScore", "offensiveRating", "defensiveRating", "pace")),
        ),
        sort_by=("teamId", "gameDateTimeEst"),
    ),
    "players": TableSpec(
        columns=(
            "personId", "firstName", "lastName", "full_name", "aliases", "position",
            "position_group", "heightInches", "bodyWeightLbs", "birthDate",
            "first_season", "last_season", "games", "last_team_id", "draftYear",
        ),
    ),
    "teams": TableSpec(
        columns=("teamId", "abbrev", "city", "name", "full_name", "aliases", "name_history"),
    ),
    # 7M+ rows, so ids go to int32 (all NBA ids are < 2^31) and minutes to float32.
    # nbalab.query.data.normalize_on_court casts back to int64/float64 on load,
    # and drops unreliable rows, so they are dropped here too.
    "on_court": TableSpec(
        columns=(
            "gameId", "season", "personId", "opponentPersonId", "shared_floor_minutes", "reliable",
            "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "ast",
        ),
        downcast={
            "gameId": pa.int32(), "personId": pa.int32(), "opponentPersonId": pa.int32(),
            "shared_floor_minutes": pa.float32(),
        },
        sort_by=("gameId", "personId", "opponentPersonId"),
        keep_if_true="reliable",
    ),
    "matchups": TableSpec(
        columns=("gameId", "season", "offPersonId", "defPersonId", "partialPossessions"),
    ),
    # Season timing (nbalab.data.calendar). Small; the app rebuilds the calendar from
    # team_games (without the All-Star Game dates) if these are missing.
    "season_calendar": TableSpec(
        columns=("season", "break_last_before", "break_first_after", "all_star_date", "all_star_date_known",
                 "source", "regular_season_start", "regular_season_end", "playoffs_start"),
    ),
    "conferences": TableSpec(columns=("season", "teamId", "conference")),
}

# Tables the app can run without (the query layer skips them when absent).
OPTIONAL_TABLES: frozenset[str] = frozenset({"on_court", "matchups", "season_calendar", "conferences"})
