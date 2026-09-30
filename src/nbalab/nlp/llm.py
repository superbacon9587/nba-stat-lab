"""Claude-backed prompt parser: English question -> :class:`QueryDraft` via tool use.

Claude only *translates*. It is given one tool, ``record_query``, whose input
schema is the draft (with ``strict: true`` so the arguments always match the
schema), plus the valid stats, grouping variables, and today's date. It writes
names exactly as the user did and leaves relative dates relative. Resolving
names to ids, turning "last 5 years" into seasons, and every number shown to
the user happen afterwards in deterministic code (:mod:`nbalab.nlp.build`).

Any failure (no key, no credit, network, refusal, no tool call, bad JSON)
raises :class:`LLMUnavailable`, and the caller falls back to the rule parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Protocol

import anthropic
from pydantic import ValidationError

from nbalab.nlp.types import QueryDraft
from nbalab.query.stats import GROUPABLE_VARIABLES, PLAYER_STATS, TEAM_STATS

TOOL_NAME = "record_query"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    """The Claude parser could not produce a draft; use the rule parser instead."""


@dataclass(frozen=True)
class LLMSettings:
    """``model`` is any current Claude model id. ``effort`` trades depth for speed;
    reading a question into fields is a light task, so it defaults to ``low``.
    ``fallbacks`` lets the API retry a declined request on another model.
    ``strict`` asks the API to guarantee schema-valid tool arguments; the draft is
    validated locally either way."""

    model: str = "claude-opus-5-5"
    effort: str = "low"
    max_tokens: int = 8000
    timeout_s: float = 60.0
    max_retries: int = 2
    fallbacks: bool = True
    strict: bool = True


class MessagesClient(Protocol):
    """The slice of ``anthropic.Anthropic`` used here (lets tests pass a fake)."""

    @property
    def beta(self) -> Any: ...


# ------------------------------------------------------------------------ schema

def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _array(items: dict[str, Any], description: str) -> dict[str, Any]:
    return {"type": "array", "items": items, "description": description}


STAT_NAMES = sorted(set(PLAYER_STATS) | set(TEAM_STATS))
POSITIONS = {"type": "string", "enum": ["G", "F", "C"]}
DAY_ENUM = {"type": "string", "enum": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
COMPARISON = _nullable({"type": "string", "enum": ["about", "at_least", "at_most"]})
VARIABLE = _nullable({"type": "string", "enum": sorted(GROUPABLE_VARIABLES)})
NAMES = {"type": "string"}


def draft_schema() -> dict[str, Any]:
    """JSON schema for :class:`QueryDraft`. Every key is required (strict mode);
    unused scalars are ``null`` and unused lists are ``[]``."""
    props: dict[str, Any] = {
        "subject_type": {"type": "string", "enum": ["player", "team"]},
        "subjects": _array(NAMES, "Players or teams the question is about, as written. Empty for league-wide questions."),
        "subject_positions": _array(POSITIONS, "For league-wide questions about a position ('all guards')."),
        "stats": _array({"type": "string", "enum": STAT_NAMES}, "Stats asked about."),
        "mode": {"type": "string", "enum": ["split", "variable_effect", "projection"]},
        "group_by": VARIABLE,
        "effect_variable": VARIABLE,
        "opponent_teams": _array(NAMES, "Opposing teams, as written."),
        "opponent_players": _array(NAMES, "Opposing players on the floor (not described as the defender), as written."),
        "defenders": _array(NAMES, "Players named as the subject's defender, as written."),
        "defender_height": _nullable({"type": "string", "description": "Defender height exactly as written, e.g. 6'7."}),
        "defender_height_comparison": COMPARISON,
        "defender_weight": _nullable({"type": "string", "description": "Defender weight as written, e.g. 250 lbs."}),
        "defender_weight_comparison": COMPARISON,
        "defender_positions": _array(POSITIONS, "Defender position group."),
        "teammates_in": _array(NAMES, "Teammates who must be playing, as written."),
        "teammates_out": _array(NAMES, "Teammates who must be out, as written."),
        "venue": _nullable({"type": "string", "description": "Place of the game as written (city, arena)."}),
        "home_away": _nullable({"type": "string", "enum": ["home", "away"]}),
        "days_of_week": _array(DAY_ENUM, "Days of week filter."),
        "months": _array({"type": "integer"}, "Month numbers 1-12."),
        "last_n_seasons": _nullable({"type": "integer", "description": "'last N years/seasons' -> N. Do not convert to years."}),
        "relative_season": _nullable({"type": "string", "enum": ["this", "last"]}),
        "season_start": _nullable({"type": "integer", "description": "Starting year of first season (2019 = 2019-20)."}),
        "season_end": _nullable({"type": "integer", "description": "Starting year of last season."}),
        "last_n_games": _nullable({"type": "integer"}),
        "back_to_back": _nullable({"type": "boolean"}),
        "rest_days_min": _nullable({"type": "integer"}),
        "rest_days_max": _nullable({"type": "integer"}),
        "playoffs": _nullable({"type": "boolean", "description": "true = playoffs only, false = regular season only."}),
        "min_minutes": _nullable({"type": "number"}),
        "lines": _array({
            "type": "object",
            "properties": {"stat": {"type": "string", "enum": STAT_NAMES}, "line": {"type": "number"}},
            "required": ["stat", "line"], "additionalProperties": False,
        }, "Betting-style lines the user gave ('exceeds 27.5 points')."),
        "notes": _array({"type": "string"}, "Short notes on anything you had to interpret."),
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def record_tool(strict: bool = True) -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Record the structured reading of the user's NBA stat question.",
        "strict": strict,
        "input_schema": draft_schema(),
    }


# ------------------------------------------------------------------------ prompt

SYSTEM_PROMPT = f"""You translate a plain-English NBA stats question into a structured query by calling the \
{TOOL_NAME} tool exactly once. You never answer the question and never state any statistic.

Rules:
- Copy player, team, and place names exactly as the user wrote them ("Steph", "the Spurs", "Philly", \
"LeBron"). Do not expand, correct, or look them up; separate code resolves names.
- Pronouns ("he", "his") refer to the subject.
- subject_type is "team" when the question is about a team's own numbers.
- Seasons are named by starting year: 2023 means 2023-24. For "last N years/seasons" set last_n_seasons=N \
and leave season_start/season_end null. "Since 2020" -> season_start=2020. "This season" / "last season" -> \
relative_season.
- mode:
  - "split": a stat under conditions ("Curry's assists vs the Spurs"), optionally broken down with group_by \
("at home vs away" -> group_by=home_away).
  - "variable_effect": how one variable moves a stat ("how do back-to-backs affect scoring"). Set \
effect_variable. If a single subject is named, still use variable_effect; code turns it into a split.
  - "projection": a future game, expected values, chances, over/under, "how might he perform".
- A player described as the subject's defender ("guarded by", "his defender is", "X as a defender") goes in \
defenders, not opponent_players. A height or weight of the defender goes in defender_height / defender_weight \
as written; use the comparison field for "taller than" (at_least) or "under" (at_most), else "about".
- A city or arena ("in San Francisco", "at MSG") goes in venue as written. Only set home_away when the user \
says home/away (or at home/on the road) explicitly; code infers it from the venue.
- lines: only numbers the user gave as thresholds ("exceeds 27.5 points" -> points 27.5, "30+ points" -> \
points 29.5). Never invent a line.
- stats must come from this list: {", ".join(STAT_NAMES)}. For a team, "points" / "ppg" means team_score. \
If no stat is named, leave stats empty.
- group_by / effect_variable must come from: {", ".join(sorted(GROUPABLE_VARIABLES))}.
- Use notes for anything you had to guess. Leave unused fields null or [].

Examples (only the non-empty fields are shown; include every field in your call):
Q: what is Steph Curry's apg against the spurs
A: subject_type=player, subjects=["Steph Curry"], stats=["assists"], mode=split, opponent_teams=["the spurs"]
Q: how do back to backs affect scoring for all guards
A: subjects=[], subject_positions=["G"], stats=["points"], mode=variable_effect, effect_variable=back_to_back
Q: Knicks points at home vs away on Sundays since 2020
A: subject_type=team, subjects=["Knicks"], stats=["team_score"], mode=split, group_by=home_away, \
days_of_week=["Sunday"], season_start=2020
Q: Curry vs Philly in San Francisco, his defender is LeBron. Chances he exceeds 27.5 points?
A: subjects=["Curry"], stats=["points"], mode=projection, opponent_teams=["Philly"], defenders=["LeBron"], \
venue="San Francisco", lines=[{{"stat": "points", "line": 27.5}}]
"""


def user_message(prompt: str, today: date) -> str:
    return f"Today's date: {today.isoformat()}\n\nQuestion: {prompt.strip()}"


# ------------------------------------------------------------------------ call

def make_client(api_key: str, settings: LLMSettings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=settings.timeout_s, max_retries=settings.max_retries)


def request_kwargs(prompt: str, today: date, settings: LLMSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "tools": [record_tool(settings.strict)],
        "tool_choice": {"type": "auto"},
        "output_config": {"effort": settings.effort},
        "messages": [{"role": "user", "content": user_message(prompt, today)}],
    }
    if settings.fallbacks:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def tool_input(response: Any) -> dict[str, Any]:
    """The ``record_query`` arguments from a response, or :class:`LLMUnavailable`."""
    if response.stop_reason == "refusal":
        raise LLMUnavailable("Claude declined the request")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("Claude ran out of output tokens")
    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            data = block.input
            return json.loads(data) if isinstance(data, str) else dict(data)
    raise LLMUnavailable("Claude did not call the record_query tool")


def is_schema_rejection(error: anthropic.BadRequestError) -> bool:
    text = str(error.message).lower()
    return "schema" in text or "strict" in text


def create(prompt: str, today: date, client: MessagesClient, settings: LLMSettings) -> Any:
    """One API call. If the API rejects the strict schema, retry once without ``strict``
    (the tool input is still validated against :class:`QueryDraft` locally)."""
    try:
        return client.beta.messages.create(**request_kwargs(prompt, today, settings))
    except anthropic.BadRequestError as e:
        if not (settings.strict and is_schema_rejection(e)):
            raise
        return client.beta.messages.create(**request_kwargs(prompt, today, replace(settings, strict=False)))


def parse_with_claude(prompt: str, today: date, client: MessagesClient, settings: LLMSettings) -> QueryDraft:
    """Ask Claude for a draft. Raises :class:`LLMUnavailable` on any failure."""
    try:
        response = create(prompt, today, client, settings)
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("the Anthropic API key was rejected") from e
    except anthropic.PermissionDeniedError as e:
        raise LLMUnavailable("the API key lacks permission for this model") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailable("rate limited by the Anthropic API") from e
    except anthropic.BadRequestError as e:
        raise LLMUnavailable(f"request rejected ({e.message})") from e
    except anthropic.APIStatusError as e:
        raise LLMUnavailable(f"Anthropic API error {e.status_code}") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("could not reach the Anthropic API") from e
    try:
        return QueryDraft.model_validate(tool_input(response))
    except (ValidationError, json.JSONDecodeError) as e:
        raise LLMUnavailable(f"Claude's tool call did not match the schema: {e}") from e
