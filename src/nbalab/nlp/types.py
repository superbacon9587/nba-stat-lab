"""Data types shared by the prompt parsers.

:class:`QueryDraft` is what a parser reads off the prompt *before* any lookup:
names exactly as written ("Steph", "the Spurs", "Philly"), heights as written
("6'7"), and relative dates left relative ("last 5 years"). Both the Claude
parser and the rule-based fallback produce a draft, and one deterministic
builder turns it into a :class:`~nbalab.query.schema.StatQuery`. That keeps
entity resolution, date math, and defaults in tested code, never in the LLM.

:class:`Assumption` and :class:`UnresolvedItem` are what the app shows as
editable chips before running the query.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nbalab.query.schema import DayName, Mode, Position, SubjectType

Comparison = Literal["about", "at_least", "at_most"]


class DraftLine(BaseModel):
    """A betting-style line as written: ``{"stat": "points", "line": 27.5}``."""

    model_config = ConfigDict(extra="forbid")

    stat: str
    line: float


class QueryDraft(BaseModel):
    """A prompt read into fields, with nothing looked up or computed yet.

    Every name field holds text as the user wrote it. Seasons are starting
    years (2024 = 2024-25). ``last_n_seasons`` and ``relative_season`` stay
    relative so the builder can apply today's date.
    """

    model_config = ConfigDict(extra="forbid")

    subject_type: SubjectType = "player"
    subjects: list[str] = Field(default_factory=list)
    subject_positions: list[Position] = Field(default_factory=list)
    stats: list[str] = Field(default_factory=list)
    mode: Mode = "split"
    group_by: str | None = None
    effect_variable: str | None = None

    opponent_teams: list[str] = Field(default_factory=list)
    opponent_players: list[str] = Field(default_factory=list)
    defenders: list[str] = Field(default_factory=list)
    defender_height: str | None = None
    defender_height_comparison: Comparison | None = None
    defender_weight: str | None = None
    defender_weight_comparison: Comparison | None = None
    defender_positions: list[Position] = Field(default_factory=list)
    teammates_in: list[str] = Field(default_factory=list)
    teammates_out: list[str] = Field(default_factory=list)

    venue: str | None = None
    home_away: Literal["home", "away"] | None = None
    days_of_week: list[DayName] = Field(default_factory=list)
    months: list[int] = Field(default_factory=list)
    last_n_seasons: int | None = None
    relative_season: Literal["this", "last"] | None = None
    season_start: int | None = None
    season_end: int | None = None
    last_n_games: int | None = None
    back_to_back: bool | None = None
    rest_days_min: int | None = None
    rest_days_max: int | None = None
    playoffs: bool | None = None
    min_minutes: float | None = None

    lines: list[DraftLine] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    """One possible match for a name, for the user to pick from."""

    model_config = ConfigDict(frozen=True)

    id: int
    kind: Literal["player", "team"]
    name: str
    detail: str = ""
    score: float = 0.0


class Assumption(BaseModel):
    """Something the parser filled in or interpreted. ``field`` names the chip."""

    field: str
    message: str
    value: Any = None


class UnresolvedItem(BaseModel):
    """Something the user should confirm or fix before the query runs.

    ``kind``:
      - ``ambiguous``: several matches. ``chosen_id`` is the one used for now.
      - ``not_found``: no match for the name.
      - ``conflict``: the pieces contradict each other (a defender who does not
        play for the opponent).
      - ``unsupported``: the parser understood it but the query cannot express it.
    """

    kind: Literal["ambiguous", "not_found", "conflict", "unsupported"]
    field: str
    text: str
    message: str
    candidates: list[Candidate] = Field(default_factory=list)
    chosen_id: int | None = None
