"""Two-way mapping between a :class:`StatQuery` and the flat fields of the manual form.

The prompt box, the chips and the manual form all edit the same StatQuery.
Each edit goes StatQuery -> form fields -> change one field -> StatQuery.
Filters the form has no widget for (teammate in/out, weight range, ...) pass
through untouched, so hand-tweaking never throws away part of a parsed query.

In projection mode the game context (opponent, venue, home/away, defender)
lives in ``projection.context`` rather than in the filter list. The engine
turns the context back into filters for the historical view.
"""

from __future__ import annotations

from typing import Any

from nbalab.query import schema as s
from nbalab.query.stats import stat_names_for

HANDLED_FILTERS: frozenset[str] = frozenset({
    "opponent_team", "defender_player", "defender_height_range", "venue", "home_away",
    "day_of_week", "season_range", "last_n_seasons", "back_to_back", "playoffs", "last_n_games",
    "player_position",
})
FIELDS: tuple[str, ...] = (
    "subject_type", "subject_id", "stats", "opponents", "defender_id", "height_on", "height_range",
    "venue_team", "home_away", "days", "season_range", "back_to_back", "playoffs", "last_n_games",
    "group_by", "line_on", "line", "minutes_on", "minutes", "effect_variable", "positions",
)


def default_form(first_season: int, last_season: int) -> dict[str, Any]:
    """An empty form: no subject, points, every season in the data."""
    return {
        "subject_type": "player", "subject_id": None, "stats": ["points"], "opponents": [],
        "defender_id": None, "height_on": False, "height_range": (77, 80), "venue_team": None,
        "home_away": "any", "days": [], "season_range": (first_season, last_season),
        "back_to_back": "any", "playoffs": "any", "last_n_games": 0, "group_by": None,
        "line_on": False, "line": None, "minutes_on": False, "minutes": 34.0,
        "effect_variable": None, "positions": [],
    }


def query_to_form(q: s.StatQuery, first_season: int, last_season: int) -> dict[str, Any]:
    """Flatten a StatQuery into form fields."""
    f = default_form(first_season, last_season)
    f["subject_type"] = "league" if q.mode == "variable_effect" and not q.subject_ids else q.subject_type
    f["subject_id"] = q.subject_ids[0] if q.subject_ids else None
    f["stats"] = list(q.stats)
    f["group_by"] = q.group_by
    f["effect_variable"] = q.effect_variable
    for flt in q.filters:
        if isinstance(flt, s.OpponentTeamFilter):
            f["opponents"] = list(flt.team_ids)
        elif isinstance(flt, s.DefenderPlayerFilter):
            f["defender_id"] = flt.person_id
        elif isinstance(flt, s.DefenderHeightRangeFilter):
            f["height_on"], f["height_range"] = True, (int(flt.min_inches), int(flt.max_inches))
        elif isinstance(flt, s.VenueFilter) and flt.team_id is not None:
            f["venue_team"] = flt.team_id
        elif isinstance(flt, s.HomeAwayFilter):
            f["home_away"] = flt.value
        elif isinstance(flt, s.DayOfWeekFilter):
            f["days"] = list(flt.days)
        elif isinstance(flt, s.SeasonRangeFilter):
            f["season_range"] = (max(flt.start, first_season), min(flt.end, last_season))
        elif isinstance(flt, s.LastNSeasonsFilter):
            f["season_range"] = (max(last_season - flt.n + 1, first_season), last_season)
        elif isinstance(flt, s.BackToBackFilter):
            f["back_to_back"] = "only" if flt.value else "exclude"
        elif isinstance(flt, s.PlayoffsFilter):
            f["playoffs"] = "only" if flt.value else "exclude"
        elif isinstance(flt, s.LastNGamesFilter):
            f["last_n_games"] = flt.n
        elif isinstance(flt, s.PlayerPositionFilter):
            f["positions"] = list(flt.positions)
    if q.projection is not None:
        ctx = q.projection.context
        if ctx.opponent_team_id is not None:
            f["opponents"] = [ctx.opponent_team_id]
        if ctx.venue_team_id is not None:
            f["venue_team"] = ctx.venue_team_id
        if ctx.home_away is not None:
            f["home_away"] = ctx.home_away
        if ctx.defender_person_id is not None:
            f["defender_id"] = ctx.defender_person_id
        if ctx.projected_minutes is not None:
            f["minutes_on"], f["minutes"] = True, float(ctx.projected_minutes)
        if q.projection.lines:
            f["line_on"] = True
            first = q.stats[0] if q.stats[0] in q.projection.lines else next(iter(q.projection.lines))
            f["line"] = float(q.projection.lines[first])
    return f


def form_to_query(f: dict[str, Any], base: s.StatQuery | None, first_season: int, last_season: int) -> s.StatQuery:
    """Build a StatQuery from form fields. Raises ``ValueError`` if the result is invalid.

    ``base`` supplies what the form cannot show: extra subjects, unhandled
    filters, and lines for secondary stats in a combo projection.
    """
    league = f["subject_type"] == "league"
    subject_type = league_subject_type(f["stats"], base) if league else f["subject_type"]
    projecting = bool(f["line_on"]) and not league
    filters: list[s.Filter] = []
    context: dict[str, Any] = {}

    if f["opponents"]:
        if projecting and len(f["opponents"]) == 1:
            context["opponent_team_id"] = int(f["opponents"][0])
        else:
            filters.append(s.OpponentTeamFilter(team_ids=[int(t) for t in f["opponents"]]))
    if f["defender_id"] is not None and subject_type == "player":
        if projecting:
            context["defender_person_id"] = int(f["defender_id"])
        else:
            filters.append(s.DefenderPlayerFilter(person_id=int(f["defender_id"])))
    if f["height_on"] and subject_type == "player":
        lo, hi = f["height_range"]
        filters.append(s.DefenderHeightRangeFilter(min_inches=float(lo), max_inches=float(hi)))
    if f["venue_team"] is not None:
        if projecting:
            context["venue_team_id"] = int(f["venue_team"])
        else:
            filters.append(s.VenueFilter(team_id=int(f["venue_team"])))
    if f["home_away"] in ("home", "away"):
        if projecting:
            context["home_away"] = f["home_away"]
        else:
            filters.append(s.HomeAwayFilter(value=f["home_away"]))
    if f["days"]:
        filters.append(s.DayOfWeekFilter(days=list(f["days"])))
    start, end = (int(x) for x in f["season_range"])
    if (start, end) != (first_season, last_season):
        filters.append(s.SeasonRangeFilter(start=start, end=end))
    if f["back_to_back"] in ("only", "exclude"):
        filters.append(s.BackToBackFilter(value=f["back_to_back"] == "only"))
    if f["playoffs"] in ("only", "exclude"):
        filters.append(s.PlayoffsFilter(value=f["playoffs"] == "only"))
    if f["last_n_games"]:
        filters.append(s.LastNGamesFilter(n=int(f["last_n_games"])))
    if f["positions"] and subject_type == "player":
        filters.append(s.PlayerPositionFilter(positions=list(f["positions"])))

    kept = [flt for flt in (base.filters if base else []) if flt.type not in HANDLED_FILTERS]
    if subject_type == "team":
        kept = [flt for flt in kept if flt.type not in s.PLAYER_ONLY_FILTER_TYPES]
    subject_ids = _subject_ids(f, base, subject_type, league)

    projection = None
    if projecting:
        if f["minutes_on"] and subject_type == "player":
            context["projected_minutes"] = float(f["minutes"])
        lines = dict(base.projection.lines) if base and base.projection else {}
        lines = {k: v for k, v in lines.items() if k in f["stats"]}
        if f["line"] is not None:
            lines[f["stats"][0]] = float(f["line"])
        projection = s.Projection(lines=lines, context=s.ProjectionContext(**context))

    if base is not None and base.mode == "period" and not league:
        # Period comparisons keep their split point; only the season scope applies to them.
        scope = [flt for flt in [*filters, *kept] if flt.type in s.SCOPE_FILTER_TYPES]
        return s.StatQuery(subject_type=subject_type, subject_ids=subject_ids, stats=list(f["stats"]),
                           filters=scope, mode="period", period_split=base.period_split)
    mode = "variable_effect" if league else ("projection" if projecting else "split")
    effect_variable = (f["effect_variable"] or f["group_by"]) if league else None
    return s.StatQuery(
        subject_type=subject_type, subject_ids=subject_ids, stats=list(f["stats"]),
        filters=[*filters, *kept], group_by=f["group_by"] if not league else None,
        effect_variable=effect_variable, projection=projection, mode=mode,
    )


def league_subject_type(stats: list[str], base: s.StatQuery | None = None) -> str:
    """Subject type of a league-wide question, which names no player or team.

    Keep the current query's type while its stats still exist for it (league-wide team
    assists stay a team question). Otherwise the stats decide: "team" when every stat is
    team-only (team_score, pace, net_rating, ...), else "player".
    """
    if base is not None and not base.subject_ids and stats and set(stats) <= stat_names_for(base.subject_type):
        return base.subject_type
    team_only = stat_names_for("team") - stat_names_for("player")
    return "team" if stats and all(x in team_only for x in stats) else "player"


def _subject_ids(f: dict[str, Any], base: s.StatQuery | None, subject_type: str, league: bool) -> list[int]:
    """The form's subject first, then any extra subjects from the parsed query of the same kind."""
    if league or f["subject_id"] is None:
        return []
    ids = [int(f["subject_id"])]
    if base and base.subject_type == subject_type:
        ids += [i for i in base.subject_ids[1:] if i not in ids]
    return ids
