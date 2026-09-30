"""Claude-backed prompt parser: English question -> :class:`QueryDraft` via tool use.

Claude only *translates*. It is given one tool, ``record_query``, whose input
schema is the draft (validated locally against :class:`QueryDraft`; the API's
strict mode can't compile a schema this size), plus the valid stats, grouping
variables, and today's date. It writes
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


# Model capabilities the request depends on (see request_kwargs).
NO_EFFORT_MODELS: frozenset[str] = frozenset({"claude-haiku-4-5", "claude-sonnet-4-5"})  # effort errors there
FALLBACK_MODELS: frozenset[str] = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})
NO_FORCED_TOOL_MODELS: frozenset[str] = frozenset({"claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5",
                                                   "claude-sonnet-5-5"})  # tool_choice any/tool -> 400


@dataclass(frozen=True)
class LLMSettings:
    """``model`` is any current Claude model id; the default is Claude Haiku 4.5, since
    reading a question into fields is a light, latency-sensitive task. ``effort``
    trades depth for speed on models that support it (not Haiku 4.5, where it is
    left out). ``fallbacks`` lets the API retry a declined request on another model
    (only sent to models that support it). ``strict`` asks the API to guarantee
    schema-valid tool arguments (off by default, see below); the draft is always validated locally."""

    model: str = "claude-haiku-4-5"
    effort: str = "low"
    max_tokens: int = 8000
    timeout_s: float = 60.0
    max_retries: int = 2
    fallbacks: bool = True
    # Off: this 33-field schema is too large for the API's strict-mode grammar compiler
    # ("compiled grammar is too large"), so a strict request is always rejected. Drafts are
    # validated against QueryDraft locally instead, and anything invalid falls back to the rules.
    strict: bool = False


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
# Stat names are listed in the system prompt and checked in code (canonical_stat), not as a schema
# enum: a ~60-name enum used twice makes the strict-mode grammar too large for the API to compile.
STAT = {"type": "string"}


# Strict tool schemas allow at most 16 union-typed (nullable) parameters, so three
# fields use an explicit "no value" enum member instead of null; from_tool_input maps them back.
SENTINELS: dict[str, dict[str, Any]] = {
    "home_away": {"any": None},
    "relative_season": {"none": None},
    "back_to_back": {"yes": True, "no": False, "any": None},
}
MAX_UNION_PARAMS = 16


NULL_STRINGS = frozenset({"null", "none", "nil", "n/a", ""})


def from_tool_input(data: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments -> QueryDraft fields.

    Maps the sentinel enum values back to None/bool, and treats a *string*
    "null"/"None"/"" in a nullable field as missing: without strict mode the
    model occasionally writes the word instead of JSON null.
    """
    out = dict(data)
    nullable = {k for k, f in QueryDraft.model_fields.items() if f.default is None}
    for key in list(out):
        if isinstance(out[key], str) and out[key].strip().lower() in NULL_STRINGS and (key in nullable or key in SENTINELS):
            out[key] = None
    for key, mapping in SENTINELS.items():
        if key in out and out[key] in mapping:
            out[key] = mapping[out[key]]
    return out


def draft_schema() -> dict[str, Any]:
    """JSON schema for :class:`QueryDraft`. Every key is required (strict mode);
    unused scalars are ``null`` (or the field's "any"/"none" value) and unused lists are ``[]``."""
    props: dict[str, Any] = {
        "subject_type": {"type": "string", "enum": ["player", "team"]},
        "subjects": _array(NAMES, "Players or teams the question is about, as written. Empty for league-wide questions."),
        "subject_positions": _array(POSITIONS, "For league-wide questions about a position ('all guards')."),
        "stats": _array(STAT, "Stats asked about, from the stat list in the instructions."),
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
        "home_away": {"type": "string", "enum": ["home", "away", "any"], "description": "'any' unless the user said home/away."},
        "days_of_week": _array(DAY_ENUM, "Days of week filter."),
        "months": _array({"type": "integer"}, "Month numbers 1-12."),
        "last_n_seasons": _nullable({"type": "integer", "description": "'last N years/seasons' -> N. Do not convert to years."}),
        "relative_season": {"type": "string", "enum": ["this", "last", "none"]},
        "season_start": _nullable({"type": "integer", "description": "Starting year of first season (2019 = 2019-20)."}),
        "season_end": _nullable({"type": "integer", "description": "Starting year of last season."}),
        "last_n_games": _nullable({"type": "integer"}),
        "back_to_back": {"type": "string", "enum": ["yes", "no", "any"], "description": "Second night of a back-to-back."},
        "rest_days_min": _nullable({"type": "integer"}),
        "rest_days_max": _nullable({"type": "integer"}),
        "playoffs": _nullable({"type": "boolean", "description": "true = playoffs only, false = regular season only."}),
        "min_minutes": _nullable({"type": "number"}),
        "lines": _array({
            "type": "object",
            "properties": {"stat": STAT, "line": {"type": "number"}},
            "required": ["stat", "line"], "additionalProperties": False,
        }, "Betting-style lines the user gave ('exceeds 27.5 points')."),
        "period_kind": {"type": "string", "enum": ["none", "all_star_break", "custom_date", "month_groups",
                                                   "last_n_before_playoffs"],
                        "description": "Before-vs-after questions; 'none' otherwise."},
        "period_date": {"type": "string", "description": "MM-DD for custom_date (after Feb 20 -> 02-20), else ''."},
        "before_months": _array({"type": "integer"}, "month_groups: month numbers of the first period."),
        "after_months": _array({"type": "integer"}, "month_groups: month numbers of the second period."),
        "period_n_games": {"type": "integer", "description": "last_n_before_playoffs: N (0 = default 20)."},
        "leaderboard": {"type": "boolean", "description": "'Who improves most': rank every player."},
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
- "More/less at home", "more on the road", "home vs away": the user is comparing, so set group_by=home_away \
and leave home_away="any". Only set home_away="home"/"away" when the question is about those games alone \
("his road games", "at home this season"). The same holds for other comparisons: "on back-to-backs vs rested" \
-> group_by=back_to_back; "by month" -> group_by=month.
- A player's opponent can be named by city, nickname, abbreviation or arena ("Philly", "the Sixers", "PHI", \
"at MSG"). "At <team or city>" means that team is the opponent AND the game is in its building: fill both \
opponent_teams and venue with the words as written.
- Shooting questions name a percentage stat: "shoot from three" / "3-point percentage" -> three_pct, "shoot" \
or "field goal percentage" -> fg_pct, "free throws" percentage -> ft_pct, "efficient" -> ts_pct. "Threes" or \
"3s made" alone means the count (threes).
- Rest: "on 2+ days of rest" -> rest_days_min=2; "on zero rest" or "second night of a back-to-back" -> \
back_to_back=yes; "not on a back-to-back" -> back_to_back=no.
- Combined stats keep their own names: points+rebounds+assists -> pra, points+rebounds -> pr, points+assists \
-> pa, rebounds+assists -> ra, steals+blocks -> stocks.
- Before-vs-after questions set period_kind (and nothing else changes how games are split): "after the All-Star \
break" / "second half of the season" / "post-break" -> all_star_break; "after Feb 20" -> custom_date with \
period_date="02-20"; "months 10,11,12,1 vs 2,3,4" -> month_groups with before_months / after_months; "last N games \
before the playoffs" -> last_n_before_playoffs with period_n_games=N; "leading up to the playoffs" / "down the \
stretch" -> last_n_before_playoffs with period_n_games=20. "Teams" with no team named -> subject_type=team and \
subjects=[] (every team is compared). "Who improves most" / "which players" -> subject_type=player, \
leaderboard=true, subjects=[] ("who" means players unless the user says "which team"). Do not also set \
group_by, months or last_n_games for these.
- Leave every season field (last_n_seasons, relative_season, season_start, season_end) empty unless the user \
names a time span; code picks the default seasons.
- Use notes for anything you had to guess. Leave unused fields null, [], "any" or "none".

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
Q: What is Stephen Curry's ppg against LeBron as a defender
A: subjects=["Stephen Curry"], stats=["points"], mode=split, defenders=["LeBron"]
Q: How might Curry perform against a 6'7 defender
A: subjects=["Curry"], mode=projection, defender_height="6'7", defender_height_comparison=about \
(no stat named, so stats=[])
Q: Jokic rebounds against defenders taller than 6-10 this season
A: subjects=["Jokic"], stats=["rebounds"], mode=split, defender_height="6-10", defender_height_comparison=at_least, \
relative_season=this
Q: Nikola Jokic points at OKC, line 28.5
A: subjects=["Nikola Jokic"], stats=["points"], mode=projection, opponent_teams=["OKC"], venue="OKC", \
lines=[{{"stat": "points", "line": 28.5}}]   ("line N" is a betting line for the stat named; "at <team>" names \
both the opponent and the venue)
Q: Lakers team threes at home, line 13
A: subject_type=team, subjects=["Lakers"], stats=["threes"], mode=projection, home_away=home, \
lines=[{{"stat": "threes", "line": 13}}]
Q: Anthony Edwards threes on back-to-backs, o/u 3.5
A: subjects=["Anthony Edwards"], stats=["threes"], mode=projection, back_to_back=yes, \
lines=[{{"stat": "threes", "line": 3.5}}]
Q: How does rest affect Jayson Tatum's rebounds
A: subjects=["Jayson Tatum"], stats=["rebounds"], mode=variable_effect, effect_variable=rest_days
Q: Tatum points with 2 or more days of rest
A: subjects=["Tatum"], stats=["points"], mode=split, rest_days_min=2
Q: Kobe Bryant ppg vs the Celtics in the playoffs
A: subjects=["Kobe Bryant"], stats=["points"], mode=split, opponent_teams=["the Celtics"], playoffs=true
Q: Seattle SuperSonics points per game in 2005
A: subject_type=team, subjects=["Seattle SuperSonics"], stats=["team_score"], mode=split, season_start=2004, \
season_end=2004   (a single year names the season that ends in it: 2005 -> 2004-05)
Q: Luka points when Kyrie is out, last 20 games
A: subjects=["Luka"], stats=["points"], mode=split, teammates_out=["Kyrie"], last_n_games=20
Q: Giannis rebounds in games he played at least 30 minutes, last 3 seasons
A: subjects=["Giannis"], stats=["rebounds"], mode=split, min_minutes=30, last_n_seasons=3
Q: Celtics pace by month last season
A: subject_type=team, subjects=["Celtics"], stats=["pace"], mode=split, group_by=month, relative_season=last
Q: Brunson assists vs the Heat when Bam Adebayo is on the floor
A: subjects=["Brunson"], stats=["assists"], mode=split, opponent_teams=["the Heat"], \
opponent_players=["Bam Adebayo"]   (on the floor, not described as his defender)
Q: Will Wembanyama get 4+ blocks against Golden State tomorrow?
A: subjects=["Wembanyama"], stats=["blocks"], mode=projection, opponent_teams=["Golden State"], \
lines=[{{"stat": "blocks", "line": 3.5}}]
Q: how do teams change after the all star break
A: subject_type=team, subjects=[], stats=[], period_kind=all_star_break
Q: compare team ppg from months 10,11,12,1 to months 2,3,4,5,6
A: subject_type=team, subjects=[], stats=["team_score"], period_kind=month_groups, before_months=[10, 11, 12, 1], \
after_months=[2, 3, 4, 5, 6]
Q: Lakers net rating last 20 games before playoffs
A: subject_type=team, subjects=["Lakers"], stats=["net_rating"], period_kind=last_n_before_playoffs, period_n_games=20
Q: who improves most after the break
A: subject_type=player, subjects=[], stats=[], period_kind=all_star_break, leaderboard=true
Q: Celtics after the all star break
A: subject_type=team, subjects=["Celtics"], stats=[], period_kind=all_star_break   (no season named: seasons empty)
Q: Harden PRA on Mondays and Tuesdays since 2022, regular season only
A: subjects=["Harden"], stats=["pra"], mode=split, days_of_week=["Monday", "Tuesday"], season_start=2022, \
playoffs=false
"""


def user_message(prompt: str, today: date) -> str:
    return f"Today's date: {today.isoformat()}\n\nQuestion: {prompt.strip()}"


# ------------------------------------------------------------------------ call

def make_client(api_key: str, settings: LLMSettings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=settings.timeout_s, max_retries=settings.max_retries)


def request_kwargs(prompt: str, today: date, settings: LLMSettings) -> dict[str, Any]:
    """One parse request. The tools + system prefix is identical on every call and ends
    with a cache breakpoint, so repeat questions read it from the prompt cache; the
    per-question part (date, question) comes after the breakpoint."""
    forced = settings.model not in NO_FORCED_TOOL_MODELS
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "tools": [record_tool(settings.strict)],
        "tool_choice": {"type": "tool", "name": TOOL_NAME} if forced else {"type": "auto"},
        "messages": [{"role": "user", "content": user_message(prompt, today)}],
    }
    if settings.model not in NO_EFFORT_MODELS:
        kwargs["output_config"] = {"effort": settings.effort}
    if settings.fallbacks and settings.model in FALLBACK_MODELS:
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
        return QueryDraft.model_validate(from_tool_input(tool_input(response)))
    except (ValidationError, json.JSONDecodeError) as e:
        raise LLMUnavailable(f"Claude's tool call did not match the schema: {e}") from e
