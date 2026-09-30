"""Turn each :class:`~nbalab.query.schema.Filter` into a row mask on a subject frame.

Filters stack with AND. :func:`filter_mask` handles one filter.
:func:`apply_filters` combines them and also reports, per filter, how many
games survive it alone and how many would survive without it. The engine uses
those counts to suggest which filter to drop when the stack is too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nbalab.query import schema as s
from nbalab.query.data import QueryData
from nbalab.query.defenders import named_defender_mask, opponent_on_court_mask, same_game_mask

# Higher = more central to the question. The suggestion to drop a filter picks
# the lowest. Identity filters ("vs the Spurs", "guarded by X") define the question.
# Calendar filters are usually incidental.
FILTER_IMPORTANCE: dict[str, int] = {
    "defender_player": 100, "opponent_team": 95, "opponent_player_on_court": 90, "teammate": 85,
    "defender_height_range": 80, "defender_weight_range": 75, "defender_position": 70,
    "player_position": 68, "venue": 65, "home_away": 60, "playoffs": 55, "last_n_games": 50,
    "back_to_back": 45, "rest_days": 40, "month": 30, "week_of_season": 25, "day_of_week": 20,
    "min_minutes": 10, "season_range": 200, "last_n_seasons": 200,
}


@dataclass
class Masked:
    """A filter's row mask, plus the defender source per row when a defender filter decided it."""

    mask: pd.Series
    source: pd.Series | None = None


@dataclass
class FilterImpact:
    """How restrictive one filter is within the stack."""

    filter_type: str
    description: str
    n_alone: int
    n_if_dropped: int
    importance: int


@dataclass
class FilterOutcome:
    mask: pd.Series
    impacts: list[FilterImpact]
    defender_source: pd.Series | None


def scope_mask(filters: list[s.Filter], frame: pd.DataFrame, latest_season: int) -> pd.Series:
    """Rows inside the season scope (season_range / last_n_seasons). Defines the baseline."""
    mask = pd.Series(True, index=frame.index)
    for f in filters:
        if isinstance(f, s.SeasonRangeFilter):
            mask &= frame["season"].between(f.start, f.end)
        elif isinstance(f, s.LastNSeasonsFilter):
            mask &= frame["season"] > latest_season - f.n
    return mask


def filter_mask(f: s.Filter, frame: pd.DataFrame, data: QueryData, subject_type: str) -> Masked:
    """Row mask for one non-scope filter. ``last_n_games`` is handled in :func:`apply_filters`."""
    if isinstance(f, s.OpponentTeamFilter):
        return Masked(frame["opponentTeamId"].isin(f.team_ids))
    if isinstance(f, s.OpponentPlayerOnCourtFilter):
        if subject_type == "team":
            return Masked(same_game_mask(frame, f.person_id, data))
        mask, src = opponent_on_court_mask(frame, f.person_id, data, f.min_shared_minutes)
        return Masked(mask)
    if isinstance(f, s.DefenderPlayerFilter):
        mask, src = named_defender_mask(frame, f.person_id, data, f.min_partial_possessions, f.min_shared_minutes)
        return Masked(mask, src)
    if isinstance(f, s.DefenderHeightRangeFilter):
        return Masked(frame["def_height"].between(f.min_inches, f.max_inches), frame["def_source"])
    if isinstance(f, s.DefenderWeightRangeFilter):
        return Masked(frame["def_weight"].between(f.min_lbs, f.max_lbs), frame["def_source"])
    if isinstance(f, s.DefenderPositionFilter):
        return Masked(frame["def_position"].isin(f.positions), frame["def_source"])
    if isinstance(f, s.TeammateFilter):
        return Masked(teammate_mask(frame, f, data, subject_type))
    if isinstance(f, s.PlayerPositionFilter):
        return Masked(frame["position_group"].isin(f.positions))
    if isinstance(f, s.VenueFilter):
        return Masked(venue_mask(frame, f))
    if isinstance(f, s.HomeAwayFilter):
        return Masked(frame["home_away"].eq(f.value))
    if isinstance(f, s.DayOfWeekFilter):
        return Masked(frame["day_of_week"].isin(f.days))
    if isinstance(f, s.MonthFilter):
        return Masked(frame["month"].isin(f.months))
    if isinstance(f, s.WeekOfSeasonFilter):
        return Masked(frame["week_of_season"].astype("float64").between(f.min_week, f.max_week))
    if isinstance(f, s.RestDaysFilter):
        return Masked(frame["team_rest_days"].astype("float64").between(f.min_days, f.max_days))
    if isinstance(f, s.BackToBackFilter):
        return Masked(frame["back_to_back"].eq(f.value))
    if isinstance(f, s.PlayoffsFilter):
        playoffs = frame["game_type"].astype(str).eq("playoffs")
        return Masked(playoffs if f.value else ~playoffs & ~frame["game_type"].astype(str).eq("play_in"))
    if isinstance(f, s.MinMinutesFilter):
        return Masked(frame["minutes"].astype("float64") >= f.minutes)
    if isinstance(f, (s.SeasonRangeFilter, s.LastNSeasonsFilter, s.LastNGamesFilter)):
        return Masked(pd.Series(True, index=frame.index))
    raise TypeError(f"unhandled filter {f!r}")


def teammate_mask(frame: pd.DataFrame, f: s.TeammateFilter, data: QueryData, subject_type: str) -> pd.Series:
    """"in": the teammate played for the subject's team in that game.
    "out": he was on that team that season (played for it at least once) but not in this game.
    """
    pg = data.player_games
    mate = pg.loc[pg["personId"] == f.person_id, ["gameId", "teamId", "season"]]
    played = pd.Series(list(zip(frame["gameId"], frame["teamId"])), index=frame.index).isin(
        set(zip(mate["gameId"], mate["teamId"]))
    )
    if f.status == "in":
        return played
    on_roster = pd.Series(list(zip(frame["teamId"], frame["season"])), index=frame.index).isin(
        set(zip(mate["teamId"], mate["season"]))
    )
    return on_roster & ~played


def venue_mask(frame: pd.DataFrame, f: s.VenueFilter) -> pd.Series:
    """Home team's building by team id, or a city match on the franchise city or arena city."""
    mask = pd.Series(True, index=frame.index)
    if f.team_id is not None:
        mask &= frame["venue_team_id"].eq(f.team_id)
    if f.city:
        city = f.city.strip().lower()
        by_team = frame["venue_city"].astype(str).str.lower().eq(city)
        by_arena = frame["arena_city"].astype(str).str.lower().eq(city)
        mask &= by_team | by_arena
    return mask


def last_n_mask(frame: pd.DataFrame, candidates: pd.Series, n: int) -> pd.Series:
    """The ``n`` most recent rows among ``candidates`` (frame must be sorted oldest first)."""
    idx = frame.index[candidates.to_numpy()][-n:]
    return pd.Series(frame.index.isin(idx), index=frame.index)


def apply_filters(
    filters: list[s.Filter], frame: pd.DataFrame, data: QueryData, subject_type: str
) -> FilterOutcome:
    """AND together all split filters on a baseline frame.

    ``last_n_games`` applies last: "last 10 games vs the Spurs" means the 10
    most recent Spurs games.
    """
    masks: list[tuple[s.Filter, Masked]] = [
        (f, filter_mask(f, frame, data, subject_type)) for f in filters
    ]
    last_n = [f for f in filters if isinstance(f, s.LastNGamesFilter)]

    def combine(skip: s.Filter | None = None) -> pd.Series:
        m = pd.Series(True, index=frame.index)
        for f, masked in masks:
            if f is not skip:
                m &= masked.mask.fillna(False).astype(bool)
        for f in last_n:
            if f is not skip:
                m = last_n_mask(frame, m, f.n)
        return m

    impacts = []
    for f, masked in masks:
        alone = last_n_mask(frame, pd.Series(True, index=frame.index), f.n) if isinstance(
            f, s.LastNGamesFilter) else masked.mask.fillna(False)
        impacts.append(FilterImpact(
            f.type, describe_filter(f, data), int(alone.sum()), int(combine(skip=f).sum()),
            FILTER_IMPORTANCE.get(f.type, 50),
        ))
    sources = [m.source for f, m in masks if m.source is not None]
    return FilterOutcome(combine(), impacts, sources[0] if sources else None)


def describe_filter(f: s.Filter, data: QueryData) -> str:
    """Plain-English description of one filter."""
    if isinstance(f, s.OpponentTeamFilter):
        return "vs " + " / ".join(data.team_name(t) for t in f.team_ids)
    if isinstance(f, s.OpponentPlayerOnCourtFilter):
        return f"with {data.player_name(f.person_id)} on the floor for the opponent"
    if isinstance(f, s.DefenderPlayerFilter):
        return f"defended by {data.player_name(f.person_id)}"
    if isinstance(f, s.DefenderHeightRangeFilter):
        return f"defender {inches_label(f.min_inches)} to {inches_label(f.max_inches)}"
    if isinstance(f, s.DefenderWeightRangeFilter):
        return f"defender {f.min_lbs:.0f}-{f.max_lbs:.0f} lb"
    if isinstance(f, s.DefenderPositionFilter):
        return "defender position " + "/".join(f.positions)
    if isinstance(f, s.TeammateFilter):
        return f"{data.player_name(f.person_id)} {'playing' if f.status == 'in' else 'out'}"
    if isinstance(f, s.PlayerPositionFilter):
        return "players at " + "/".join(f.positions)
    if isinstance(f, s.VenueFilter):
        return "at " + (f"{data.team_name(f.team_id)} home" if f.team_id is not None else str(f.city))
    if isinstance(f, s.HomeAwayFilter):
        return f"{f.value} games"
    if isinstance(f, s.DayOfWeekFilter):
        return "on " + "/".join(f.days)
    if isinstance(f, s.MonthFilter):
        return "in month " + "/".join(str(m) for m in f.months)
    if isinstance(f, s.WeekOfSeasonFilter):
        return f"season weeks {f.min_week}-{f.max_week}"
    if isinstance(f, s.SeasonRangeFilter):
        return f"seasons {f.start}-{str(f.start + 1)[-2:]} to {f.end}-{str(f.end + 1)[-2:]}"
    if isinstance(f, s.LastNSeasonsFilter):
        return f"last {f.n} seasons"
    if isinstance(f, s.LastNGamesFilter):
        return f"last {f.n} games"
    if isinstance(f, s.RestDaysFilter):
        return f"{f.min_days}-{f.max_days} rest days"
    if isinstance(f, s.BackToBackFilter):
        return "back-to-backs" if f.value else "not back-to-backs"
    if isinstance(f, s.PlayoffsFilter):
        return "playoffs only" if f.value else "regular season only"
    if isinstance(f, s.MinMinutesFilter):
        return f"{f.minutes:g}+ minutes"
    return f.type


def inches_label(inches: float) -> str:
    """79 -> 6'7\"."""
    return f"{int(inches) // 12}'{inches - 12 * (int(inches) // 12):g}\""
