"""Catch impossible questions before they run, and offer one-click fixes.

A query can ask for games that cannot exist: a named defender who never
played for the opponent, a projection where the "defender" is on a different
team (or the subject's own team), a team playing itself. Running it would
return 0 games or a misleading projection. :func:`find_conflicts` explains
the problem in plain words and builds the corrected queries, so the app can
show buttons like "Use the Lakers as the opponent" or "Drop the defender".

Works on any :class:`StatQuery`, whether it came from the rule parser, the
LLM parser, or the manual form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nbalab.query import schema as s
from nbalab.query.data import QueryData


@dataclass(frozen=True)
class Fix:
    label: str
    query: s.StatQuery


@dataclass
class Conflict:
    message: str
    fixes: list[Fix] = field(default_factory=list)


def season_scope(q: s.StatQuery, latest_season: int) -> tuple[int, int]:
    """(first, last) season the query covers; every season in the data when unscoped."""
    lo, hi = 0, latest_season
    for f in q.scope_filters:
        if isinstance(f, s.SeasonRangeFilter):
            lo, hi = max(lo, f.start), min(hi, f.end)
        elif isinstance(f, s.LastNSeasonsFilter):
            lo = max(lo, latest_season - f.n + 1)
    return lo, hi


def teams_by_season(data: QueryData, person_id: int) -> pd.DataFrame:
    """One row per (season, team) the player appeared for, with his last game date there."""
    pg = data.player_games
    g = pg.loc[pg["personId"].eq(person_id), ["season", "teamId", "gameDateTimeEst"]]
    return g.groupby(["season", "teamId"], as_index=False)["gameDateTimeEst"].max()


def current_team(data: QueryData, person_id: int) -> int | None:
    """The team of the player's most recent game."""
    t = teams_by_season(data, person_id)
    return None if t.empty else int(t.sort_values("gameDateTimeEst")["teamId"].iloc[-1])


def teams_in_scope(data: QueryData, person_id: int, lo: int, hi: int) -> list[int]:
    """Teams the player played for within seasons lo..hi, most recent first."""
    t = teams_by_season(data, person_id)
    t = t[t["season"].between(lo, hi)].sort_values("gameDateTimeEst", ascending=False)
    return list(dict.fromkeys(int(x) for x in t["teamId"]))


def _drop_filters(q: s.StatQuery, types: set[str], person_id: int | None = None) -> s.StatQuery:
    keep = [f for f in q.filters
            if not (f.type in types and (person_id is None or getattr(f, "person_id", None) == person_id))]
    return q.model_copy(update={"filters": keep})


def _with_context(q: s.StatQuery, **changes) -> s.StatQuery:
    ctx = q.projection.context.model_copy(update=changes)
    return q.model_copy(update={"projection": q.projection.model_copy(update={"context": ctx})})


def _with_opponent_filter(q: s.StatQuery, team_id: int) -> s.StatQuery:
    rest = [f for f in q.filters if f.type != "opponent_team"]
    return q.model_copy(update={"filters": [*rest, s.OpponentTeamFilter(team_ids=[team_id])]})


def find_conflicts(q: s.StatQuery, data: QueryData) -> list[Conflict]:
    """Every impossible combination in the query, each with suggested fixes."""
    if q.mode == "variable_effect" or not q.subject_ids:
        return []
    out: list[Conflict] = []
    out += _team_plays_itself(q, data)
    if q.subject_type == "player":
        out += _projection_conflicts(q, data)
        out += _split_defender_conflicts(q, data)
    return out


def _team_plays_itself(q: s.StatQuery, data: QueryData) -> list[Conflict]:
    out = []
    opp_filters = [f for f in q.filters if isinstance(f, s.OpponentTeamFilter)]
    subject_teams = set(q.subject_ids) if q.subject_type == "team" else set()
    if q.subject_type == "player" and q.projection is not None:
        cur = current_team(data, q.subject_ids[0])
        subject_teams = {cur} if cur is not None else set()
    for f in opp_filters:
        bad = subject_teams & set(f.team_ids)
        if bad and q.subject_type == "team":
            out.append(Conflict(f"A team cannot play itself: the {data.team_name(next(iter(bad)))} are both the "
                                "subject and the opponent.", [Fix("Drop the opponent", _drop_filters(q, {"opponent_team"}))]))
    if q.projection is not None and q.projection.context.opponent_team_id in subject_teams:
        t = q.projection.context.opponent_team_id
        who = "The team" if q.subject_type == "team" else data.player_name(q.subject_ids[0])
        out.append(Conflict(f"{who} plays for the {data.team_name(t)}, so they cannot be the opponent.",
                            [Fix("Drop the opponent", _with_context(q, opponent_team_id=None))]))
    return out


def _projection_conflicts(q: s.StatQuery, data: QueryData) -> list[Conflict]:
    """Next-game context: the defender must play for the opponent, and not for the subject's team."""
    if q.projection is None or q.projection.context.defender_person_id is None:
        return []
    ctx = q.projection.context
    d = ctx.defender_person_id
    name = data.player_name(d)
    d_team = current_team(data, d)
    subject_team = current_team(data, q.subject_ids[0])
    drop = Fix("Drop the defender", _with_context(q, defender_person_id=None))
    if d_team is None:
        return [Conflict(f"{name} has no games in the data, so he can't be the defender.", [drop])]
    if d_team == subject_team:
        return [Conflict(f"{name} is a teammate of {data.player_name(q.subject_ids[0])} (both on the "
                         f"{data.team_name(d_team)}), not a defender.", [drop])]
    if ctx.opponent_team_id is not None and ctx.opponent_team_id != d_team:
        use = Fix(f"Use the {data.team_name(d_team)} as the opponent", _with_context(q, opponent_team_id=d_team))
        return [Conflict(f"{name} plays for the {data.team_name(d_team)}, not the "
                         f"{data.team_name(ctx.opponent_team_id)}, so he can't defend in this game.", [use, drop])]
    return []


def _split_defender_conflicts(q: s.StatQuery, data: QueryData) -> list[Conflict]:
    """Historical split: a named defender must have played for the opponent in the seasons asked about."""
    opp = next((f for f in q.filters if isinstance(f, s.OpponentTeamFilter)), None)
    ctx_opp = q.projection.context.opponent_team_id if q.projection is not None else None
    opp_ids = set(opp.team_ids) if opp else ({ctx_opp} if ctx_opp is not None else set())
    if not opp_ids:
        return []
    lo, hi = season_scope(q, data.latest_season)
    out = []
    for f in q.filters:
        if not isinstance(f, s.DefenderPlayerFilter):
            continue
        name = data.player_name(f.person_id)
        teams = teams_in_scope(data, f.person_id, lo, hi)
        if set(teams) & opp_ids:
            continue
        opp_names = " / ".join(data.team_name(t) for t in sorted(opp_ids))
        fixes = []
        if teams:
            fixes.append(Fix(f"Use the {data.team_name(teams[0])} as the opponent", _with_opponent_filter(q, teams[0])))
        fixes.append(Fix("Drop the defender", _drop_filters(q, {"defender_player"}, f.person_id)))
        if opp is not None:
            fixes.append(Fix("Drop the opponent", _drop_filters(q, {"opponent_team"})))
        where = "in the seasons asked about" if (lo, hi) != (0, data.latest_season) else "in the data"
        out.append(Conflict(f"{name} never played for the {opp_names} {where}, so no game matches both "
                            "filters.", fixes))
    return out
