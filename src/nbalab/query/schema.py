"""The structured query: what the LLM parser produces and the engines consume.

A :class:`StatQuery` is a plain description of a question ("Curry's assists vs
the Spurs"). It contains only ids, stat names, and filter parameters. It never
holds any computed numbers. All computation happens in :mod:`nbalab.query.engine`
and :mod:`nbalab.query.variable_effect`.

Filters are a *discriminated union*: every filter carries a ``type`` string
that names its kind, so the JSON ``{"type": "home_away", "value": "home"}``
parses straight into :class:`HomeAwayFilter`.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nbalab.query.stats import GROUPABLE_VARIABLES, canonical_stat, stat_names_for

SubjectType = Literal["player", "team"]
Mode = Literal["split", "variable_effect", "projection", "period"]
Position = Literal["G", "F", "C"]
DayName = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class _Filter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------- who is on the floor


class OpponentTeamFilter(_Filter):
    """Games against any of these opponent teamIds."""

    type: Literal["opponent_team"] = "opponent_team"
    team_ids: list[int] = Field(min_length=1)


class OpponentPlayerOnCourtFilter(_Filter):
    """Games where this opponent player was on the floor at the same time as the subject.

    Uses shared floor time from ``on_court.parquet`` when available, otherwise
    falls back to "the opponent player also played in the game".
    """

    type: Literal["opponent_player_on_court"] = "opponent_player_on_court"
    person_id: int
    min_shared_minutes: float = Field(default=1.0, ge=0)


class DefenderPlayerFilter(_Filter):
    """Games where this player defended the subject. See :mod:`nbalab.query.defenders`
    for which data source can actually answer that."""

    type: Literal["defender_player"] = "defender_player"
    person_id: int
    min_partial_possessions: float = Field(default=5.0, ge=0)
    min_shared_minutes: float = Field(default=10.0, ge=0)


class DefenderHeightRangeFilter(_Filter):
    """Primary defender between ``min_inches`` and ``max_inches`` tall (inclusive)."""

    type: Literal["defender_height_range"] = "defender_height_range"
    min_inches: float
    max_inches: float

    @model_validator(mode="after")
    def _ordered(self) -> "DefenderHeightRangeFilter":
        if self.min_inches > self.max_inches:
            raise ValueError("min_inches must be <= max_inches")
        return self


class DefenderWeightRangeFilter(_Filter):
    """Primary defender between ``min_lbs`` and ``max_lbs`` (inclusive)."""

    type: Literal["defender_weight_range"] = "defender_weight_range"
    min_lbs: float
    max_lbs: float

    @model_validator(mode="after")
    def _ordered(self) -> "DefenderWeightRangeFilter":
        if self.min_lbs > self.max_lbs:
            raise ValueError("min_lbs must be <= max_lbs")
        return self


class DefenderPositionFilter(_Filter):
    """Primary defender's position group is one of these (G/F/C)."""

    type: Literal["defender_position"] = "defender_position"
    positions: list[Position] = Field(min_length=1)


class TeammateFilter(_Filter):
    """Teammate played ("in") or did not play ("out") while on the subject's team that season."""

    type: Literal["teammate"] = "teammate"
    person_id: int
    status: Literal["in", "out"]


class PlayerPositionFilter(_Filter):
    """Subject player's position group is one of these. Mostly for league-wide
    ``variable_effect`` queries ("how do back-to-backs affect all guards")."""

    type: Literal["player_position"] = "player_position"
    positions: list[Position] = Field(min_length=1)


# -------------------------------------------------------------------- where and when


class VenueFilter(_Filter):
    """Games played in this home team's building (``team_id``) or city (``city``)."""

    type: Literal["venue"] = "venue"
    team_id: int | None = None
    city: str | None = None

    @model_validator(mode="after")
    def _one_given(self) -> "VenueFilter":
        if self.team_id is None and not self.city:
            raise ValueError("venue needs team_id or city")
        return self


class HomeAwayFilter(_Filter):
    type: Literal["home_away"] = "home_away"
    value: Literal["home", "away"]


class DayOfWeekFilter(_Filter):
    type: Literal["day_of_week"] = "day_of_week"
    days: list[DayName] = Field(min_length=1)


class MonthFilter(_Filter):
    type: Literal["month"] = "month"
    months: list[Annotated[int, Field(ge=1, le=12)]] = Field(min_length=1)


class WeekOfSeasonFilter(_Filter):
    type: Literal["week_of_season"] = "week_of_season"
    min_week: int = Field(default=1, ge=1)
    max_week: int = Field(default=99, ge=1)


class SeasonRangeFilter(_Filter):
    """Seasons by starting year, inclusive: 2020..2024 means 2020-21 through 2024-25."""

    type: Literal["season_range"] = "season_range"
    start: int
    end: int


class LastNSeasonsFilter(_Filter):
    """The last ``n`` seasons in the dataset (not the player's last ``n``)."""

    type: Literal["last_n_seasons"] = "last_n_seasons"
    n: int = Field(ge=1)


class LastNGamesFilter(_Filter):
    """The subject's most recent ``n`` games within the other filters' season scope."""

    type: Literal["last_n_games"] = "last_n_games"
    n: int = Field(ge=1)


class RestDaysFilter(_Filter):
    """Full days off before the game (0 = back-to-back)."""

    type: Literal["rest_days"] = "rest_days"
    min_days: int = Field(default=0, ge=0)
    max_days: int = Field(default=99, ge=0)


class BackToBackFilter(_Filter):
    type: Literal["back_to_back"] = "back_to_back"
    value: bool = True


class PlayoffsFilter(_Filter):
    """``True`` keeps playoff games only; ``False`` drops playoff and play-in games."""

    type: Literal["playoffs"] = "playoffs"
    value: bool = True


class MinMinutesFilter(_Filter):
    type: Literal["min_minutes"] = "min_minutes"
    minutes: float = Field(ge=0)


Filter = Annotated[
    Union[
        OpponentTeamFilter,
        OpponentPlayerOnCourtFilter,
        DefenderPlayerFilter,
        DefenderHeightRangeFilter,
        DefenderWeightRangeFilter,
        DefenderPositionFilter,
        TeammateFilter,
        PlayerPositionFilter,
        VenueFilter,
        HomeAwayFilter,
        DayOfWeekFilter,
        MonthFilter,
        WeekOfSeasonFilter,
        SeasonRangeFilter,
        LastNSeasonsFilter,
        LastNGamesFilter,
        RestDaysFilter,
        BackToBackFilter,
        PlayoffsFilter,
        MinMinutesFilter,
    ],
    Field(discriminator="type"),
]

# Filters that set the *scope* (which seasons count). The baseline keeps these;
# every other filter defines the split.
SCOPE_FILTER_TYPES: frozenset[str] = frozenset({"season_range", "last_n_seasons"})
DEFENDER_FILTER_TYPES: frozenset[str] = frozenset(
    {"defender_player", "defender_height_range", "defender_weight_range", "defender_position"}
)
PLAYER_ONLY_FILTER_TYPES: frozenset[str] = DEFENDER_FILTER_TYPES | {"player_position", "min_minutes"}


# ----------------------------------------------------------------------- projection


class ProjectionContext(BaseModel):
    """The future game being projected. Every field is optional."""

    model_config = ConfigDict(extra="forbid")

    opponent_team_id: int | None = None
    venue_team_id: int | None = None
    defender_person_id: int | None = None
    home_away: Literal["home", "away"] | None = None
    rest_days: int | None = None
    back_to_back: bool | None = None
    projected_minutes: float | None = None


class Projection(BaseModel):
    """Betting-style lines per stat (``{"points": 27.5}``) and the game context."""

    model_config = ConfigDict(extra="forbid")

    lines: dict[str, float] = Field(default_factory=dict)
    context: ProjectionContext = Field(default_factory=ProjectionContext)


# ------------------------------------------------------------------------ periods

PeriodKind = Literal["all_star_break", "custom_date", "month_groups", "last_n_before_playoffs"]


class PeriodSplit(BaseModel):
    """Where to cut each season into "before" and "after" for a period comparison.

    - ``all_star_break`` (default): before = regular-season games up to the break,
      after = games after it (see :mod:`nbalab.data.calendar` for how the break is found).
    - ``custom_date``: ``date`` as "MM-DD" (e.g. "02-20"); before = earlier games that season.
    - ``month_groups``: ``before_months`` vs ``after_months`` (calendar month numbers).
    - ``last_n_before_playoffs``: after = the last ``n_games`` regular-season games,
      before = the rest of that regular season.

    Regular season only by default (NBA Cup group games count as regular season);
    ``include_playoffs`` adds playoff games to "after".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PeriodKind = "all_star_break"
    date: str | None = Field(default=None, pattern=r"^(0?[1-9]|1[0-2])-(0?[1-9]|[12][0-9]|3[01])$")
    before_months: list[int] = Field(default_factory=list)
    after_months: list[int] = Field(default_factory=list)
    n_games: int = Field(default=20, ge=1, le=82)
    include_playoffs: bool = False

    @model_validator(mode="after")
    def _check(self) -> "PeriodSplit":
        if self.kind == "custom_date" and not self.date:
            raise ValueError("custom_date needs date as MM-DD")
        if self.kind == "month_groups":
            if not self.before_months or not self.after_months:
                raise ValueError("month_groups needs before_months and after_months")
            if set(self.before_months) & set(self.after_months):
                raise ValueError("a month cannot be both before and after")
            if any(not 1 <= m <= 12 for m in (*self.before_months, *self.after_months)):
                raise ValueError("months are 1-12")
        return self

    def describe(self) -> str:
        if self.kind == "all_star_break":
            return "before vs after the All-Star break"
        if self.kind == "custom_date":
            return f"before vs after {self.date}"
        if self.kind == "month_groups":
            names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
            return (" / ".join(names[m - 1] for m in self.before_months) + " vs "
                    + " / ".join(names[m - 1] for m in self.after_months))
        return f"last {self.n_games} regular-season games vs the rest"


# --------------------------------------------------------------------------- query


class StatQuery(BaseModel):
    """A fully structured stat question.

    ``mode``:
      - ``split``: the subject's stat under the filters vs. their own baseline.
        With ``group_by``, one row per value of that variable.
      - ``variable_effect``: league-wide regression of how ``effect_variable``
        moves the stat for a typical player (``subject_ids`` may be empty).
      - ``projection``: the split for the projection context, plus historical
        hit rates for the lines. The forward-looking model lives in
        :mod:`nbalab.models`.
    """

    model_config = ConfigDict(extra="forbid")

    subject_type: SubjectType = "player"
    subject_ids: list[int] = Field(default_factory=list)
    stats: list[str] = Field(min_length=1)
    filters: list[Filter] = Field(default_factory=list)
    group_by: str | None = None
    effect_variable: str | None = None
    projection: Projection | None = None
    period_split: PeriodSplit | None = None
    mode: Mode = "split"

    @model_validator(mode="before")
    @classmethod
    def _canonical_stats(cls, data: Any) -> Any:
        """Accept everyday stat spellings ("PRA", "FG%", "ppg") and store catalog names."""
        if not isinstance(data, dict):
            return data
        kind = data.get("subject_type", "player")
        data = dict(data)
        if isinstance(data.get("stats"), list):
            data["stats"] = [canonical_stat(s, kind) if isinstance(s, str) else s for s in data["stats"]]
        proj = data.get("projection")
        if isinstance(proj, dict) and isinstance(proj.get("lines"), dict):
            lines = {canonical_stat(k, kind): v for k, v in proj["lines"].items()}
            data["projection"] = {**proj, "lines": lines}
        return data

    @model_validator(mode="after")
    def _check(self) -> "StatQuery":
        valid = stat_names_for(self.subject_type)
        bad = [s for s in self.stats if s not in valid]
        if bad:
            raise ValueError(f"unknown {self.subject_type} stat(s) {bad}; valid: {sorted(valid)}")
        if self.projection:
            bad_lines = [s for s in self.projection.lines if s not in valid]
            if bad_lines:
                raise ValueError(f"projection line for unknown stat(s) {bad_lines}")
        if self.mode in ("split", "projection") and not self.subject_ids:
            raise ValueError(f"mode={self.mode!r} needs at least one subject id")
        if self.mode == "variable_effect" and not (self.effect_variable or self.group_by):
            raise ValueError("variable_effect mode needs effect_variable (or group_by)")
        if self.mode == "period" and self.period_split is None:
            object.__setattr__(self, "period_split", PeriodSplit())
        if self.mode != "period" and self.period_split is not None:
            raise ValueError("period_split only applies to mode='period'")
        if self.mode == "projection" and self.projection is None:
            raise ValueError("projection mode needs a projection block")
        if self.subject_type == "team":
            bad_f = [f.type for f in self.filters if f.type in PLAYER_ONLY_FILTER_TYPES]
            if bad_f:
                raise ValueError(f"filters {bad_f} only apply to player queries")
        if self.group_by is not None and self.group_by not in GROUPABLE_VARIABLES:
            raise ValueError(f"unknown group_by {self.group_by!r}; valid: {sorted(GROUPABLE_VARIABLES)}")
        return self

    @property
    def scope_filters(self) -> list[Filter]:
        return [f for f in self.filters if f.type in SCOPE_FILTER_TYPES]

    @property
    def split_filters(self) -> list[Filter]:
        return [f for f in self.filters if f.type not in SCOPE_FILTER_TYPES]

    def with_projection_context(self) -> "StatQuery":
        """Turn the projection context into ordinary split filters (for the historical view)."""
        if self.projection is None:
            return self
        ctx = self.projection.context
        extra: list[Filter] = []
        if ctx.opponent_team_id is not None:
            extra.append(OpponentTeamFilter(team_ids=[ctx.opponent_team_id]))
        if ctx.venue_team_id is not None:
            extra.append(VenueFilter(team_id=ctx.venue_team_id))
        if ctx.defender_person_id is not None and self.subject_type == "player":
            extra.append(DefenderPlayerFilter(person_id=ctx.defender_person_id))
        if ctx.home_away is not None:
            extra.append(HomeAwayFilter(value=ctx.home_away))
        if ctx.back_to_back is not None:
            extra.append(BackToBackFilter(value=ctx.back_to_back))
        elif ctx.rest_days is not None:
            extra.append(RestDaysFilter(min_days=ctx.rest_days, max_days=ctx.rest_days))
        have = {f.type for f in self.filters}
        merged = [*self.filters, *(f for f in extra if f.type not in have)]
        return self.model_copy(update={"filters": merged})
