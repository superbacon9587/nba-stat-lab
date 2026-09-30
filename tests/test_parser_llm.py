"""The Claude path, with a fake client: request shape, schema, fallback, caching, chip edits."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from nbalab.nlp import parser as parser_mod
from nbalab.nlp.llm import FALLBACK_BETA, TOOL_NAME, LLMSettings, LLMUnavailable, draft_schema
from nbalab.nlp.parser import ParserConfig, PromptParser, parse_prompt, swap_entity
from nbalab.nlp.types import DraftLine, QueryDraft
from nbalab.query.stats import GROUPABLE_VARIABLES, PLAYER_STATS
from tests.nlp_fixture import fixture_index
from tests.test_parser_prompts import CASES, TODAY, canon, stub_lines
from nbalab.query.schema import StatQuery

EXAMPLE_5 = CASES[4][0]
EXAMPLE_5_DRAFT = QueryDraft(
    subjects=["Curry"], stats=["points", "assists"], mode="projection", opponent_teams=["Philly"],
    defenders=["LeBron"], venue="San Francisco",
    lines=[DraftLine(stat="points", line=27.5), DraftLine(stat="assists", line=5.5)],
)


def tool_response(draft: QueryDraft | dict[str, Any], stop_reason: str = "tool_use") -> SimpleNamespace:
    data = draft.model_dump() if isinstance(draft, QueryDraft) else draft
    block = SimpleNamespace(type="tool_use", name=TOOL_NAME, input=data)
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="thinking"), block])


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: records calls, returns or raises what it is given."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        out = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(out, Exception):
            raise out
        return out


def make_parser(client: FakeClient | None, config: ParserConfig | None = None) -> PromptParser:
    return PromptParser(config, index=fixture_index(), client=client, line_provider=stub_lines)


def api_error(status: int, message: str) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cls = {400: anthropic.BadRequestError, 401: anthropic.AuthenticationError}.get(status, anthropic.APIStatusError)
    return cls(message, response=httpx2.Response(status, request=request), body=None)


# ------------------------------------------------------------------ happy path


def test_llm_draft_builds_the_expected_query() -> None:
    client = FakeClient(tool_response(EXAMPLE_5_DRAFT))
    result = make_parser(client).parse(EXAMPLE_5, today=TODAY)
    assert result.source == "llm"
    assert canon(result.query) == canon(StatQuery.model_validate(CASES[4][1]))
    assert [(u.kind, u.field, u.chosen_id) for u in result.unresolved] == [("conflict", "defender", 2544)]


def test_llm_and_rules_agree_on_example_5() -> None:
    llm = make_parser(FakeClient(tool_response(EXAMPLE_5_DRAFT))).parse(EXAMPLE_5, today=TODAY)
    rules = make_parser(None).parse(EXAMPLE_5, backend="rules", today=TODAY)
    assert canon(llm.query) == canon(rules.query)


def test_request_gives_schema_stats_and_date() -> None:
    client = FakeClient(tool_response(EXAMPLE_5_DRAFT))
    make_parser(client).parse(EXAMPLE_5, today=TODAY)
    [kw] = client.calls
    assert kw["model"] == LLMSettings().model
    [tool] = kw["tools"]
    assert tool["name"] == TOOL_NAME and tool["strict"] is False  # schema too large for strict mode
    assert kw["model"] == "claude-haiku-4-5"
    assert kw["tool_choice"] == {"type": "tool", "name": TOOL_NAME}  # Haiku 4.5 accepts a forced tool call
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}  # stable prefix is cached
    system = kw["system"][0]["text"]
    assert all(s in system for s in PLAYER_STATS) and all(v in system for v in GROUPABLE_VARIABLES)
    assert "2026-09-29" not in system  # the date lives after the cache breakpoint
    assert "2026-09-29" in kw["messages"][0]["content"]
    assert EXAMPLE_5 in kw["messages"][0]["content"]
    assert "output_config" not in kw  # effort is not supported on Haiku 4.5
    assert "fallbacks" not in kw and "betas" not in kw  # server-side fallbacks are for newer models


def test_request_shape_follows_the_model() -> None:
    from dataclasses import replace

    from nbalab.nlp.llm import request_kwargs

    opus = request_kwargs("q", TODAY, replace(LLMSettings(), model="claude-opus-5-5"))
    assert opus["tool_choice"] == {"type": "auto"}  # forced tool_choice is rejected there
    assert opus["output_config"] == {"effort": "low"}
    assert opus["betas"] == [FALLBACK_BETA] and opus["fallbacks"] == "default"


def test_schema_is_strict_and_matches_draft() -> None:
    schema = draft_schema()
    assert set(schema["properties"]) == set(QueryDraft.model_fields)
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False


def test_model_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBALAB_PARSER_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("NBALAB_HEIGHT_TOLERANCE", "2")
    cfg = ParserConfig.from_env()
    assert cfg.llm.model == "claude-sonnet-5-5" and cfg.build.height_tolerance_in == 2.0
    client = FakeClient(tool_response(QueryDraft(subjects=["Curry"], stats=["points"], defender_height="6'7")))
    result = make_parser(client, cfg).parse("x", today=TODAY)
    assert client.calls[0]["model"] == "claude-sonnet-5-5"
    [f] = result.query.filters
    assert (f.min_inches, f.max_inches) == (77, 81)


# -------------------------------------------------------------------- fallback


@pytest.mark.parametrize("outcome", [
    api_error(400, "Your credit balance is too low"),
    api_error(401, "invalid x-api-key"),
    api_error(529, "overloaded"),
    anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
    tool_response(EXAMPLE_5_DRAFT, stop_reason="refusal"),
    SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Sure!")]),
    tool_response({"subjects": "not a list"}),
], ids=["no-credit", "bad-key", "overloaded", "network", "refusal", "no-tool-call", "bad-json"])
def test_any_llm_failure_falls_back_to_rules(outcome: Any) -> None:
    result = make_parser(FakeClient(outcome)).parse(EXAMPLE_5, today=TODAY)
    assert result.source == "rules"
    assert canon(result.query) == canon(StatQuery.model_validate(CASES[4][1]))
    assert result.assumptions[0].field == "parser" and "unavailable" in result.assumptions[0].message


def test_strict_schema_rejection_retries_without_strict() -> None:
    client = FakeClient(api_error(400, "tools.0.input_schema: strict schema too complex"),
                        tool_response(EXAMPLE_5_DRAFT))
    cfg = ParserConfig(llm=LLMSettings(strict=True))  # opt back in to strict: a rejection retries without it
    result = make_parser(client, cfg).parse(EXAMPLE_5, today=TODAY)
    assert result.source == "llm"
    assert [c["tools"][0]["strict"] for c in client.calls] == [True, False]


def test_schema_stays_within_strict_union_limit() -> None:
    import json

    from nbalab.nlp.llm import MAX_UNION_PARAMS, from_tool_input

    props = draft_schema()["properties"]
    unions = [k for k, v in props.items() if "anyOf" in json.dumps(v)]
    assert len(unions) <= MAX_UNION_PARAMS
    back = from_tool_input({"home_away": "any", "relative_season": "none", "back_to_back": "yes"})
    assert back == {"home_away": None, "relative_season": None, "back_to_back": True}
    assert QueryDraft.model_validate({**QueryDraft().model_dump(), **back}).back_to_back is True


def test_backend_llm_raises_instead_of_falling_back() -> None:
    with pytest.raises(LLMUnavailable):
        make_parser(FakeClient(api_error(400, "Your credit balance is too low"))).parse(EXAMPLE_5, backend="llm")


def test_no_key_uses_rules_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser_mod, "load_api_key", lambda: None)
    p = make_parser(None)
    result = p.parse(EXAMPLE_5, today=TODAY)
    assert result.source == "rules" and "No Anthropic API key" in result.assumptions[0].message
    with pytest.raises(LLMUnavailable):
        p.parse(EXAMPLE_5, backend="llm", today=TODAY)


def test_backend_rules_never_calls_claude() -> None:
    client = FakeClient(tool_response(EXAMPLE_5_DRAFT))
    result = make_parser(client).parse(EXAMPLE_5, backend="rules", today=TODAY)
    assert client.calls == [] and result.source == "rules"
    assert not any(a.field == "parser" for a in result.assumptions)


# ---------------------------------------------------------------------- cache


def test_identical_prompts_are_cached() -> None:
    client = FakeClient(tool_response(EXAMPLE_5_DRAFT))
    p = make_parser(client)
    first = p.parse(EXAMPLE_5, today=TODAY)
    second = p.parse("  " + EXAMPLE_5.replace(" ", "   ") + "\n", today=TODAY)
    assert len(client.calls) == 1
    assert canon(first.query) == canon(second.query)
    first.assumptions.clear()  # callers get copies; the cache is not affected
    assert p.parse(EXAMPLE_5, today=TODAY).assumptions


def test_cache_key_includes_date_and_backend() -> None:
    client = FakeClient(tool_response(EXAMPLE_5_DRAFT))
    p = make_parser(client)
    p.parse(EXAMPLE_5, today=TODAY)
    p.parse(EXAMPLE_5, today=date(2026, 11, 1))
    p.parse(EXAMPLE_5, backend="rules", today=TODAY)
    assert len(client.calls) == 2


def test_fallback_after_api_failure_is_not_cached() -> None:
    client = FakeClient(api_error(529, "overloaded"), tool_response(EXAMPLE_5_DRAFT))
    p = make_parser(client)
    assert p.parse(EXAMPLE_5, today=TODAY).source == "rules"
    assert p.parse(EXAMPLE_5, today=TODAY).source == "llm"


# ------------------------------------------------------------------ public api


def test_result_unpacks_to_three() -> None:
    result = make_parser(None).parse("Curry apg vs the Spurs", backend="rules", today=TODAY)
    query, assumptions, unresolved = result
    assert query is result.query and assumptions is result.assumptions and unresolved is result.unresolved


def test_parse_prompt_uses_given_parser() -> None:
    p = make_parser(None)
    assert parse_prompt("Tatum rebounds", backend="rules", today=TODAY, parser=p).query.subject_ids == [1628369]


def test_swap_entity_replaces_everywhere() -> None:
    query = make_parser(None).parse(EXAMPLE_5, backend="rules", today=TODAY).query
    swapped = swap_entity(query, 2544, 1630178)  # user picks Tyrese Maxey as the defender
    assert swapped.projection.context.defender_person_id == 1630178
    swapped = swap_entity(swapped, 201939, 203552)  # and Seth Curry as the subject
    assert swapped.subject_ids == [203552]
    assert swapped.projection.context.opponent_team_id == 1610612755


def test_claude_period_draft_builds_team_period_query() -> None:
    """The bug prompt through the Claude path: a team month-group comparison, never a player query."""
    draft = QueryDraft(subject_type="team", stats=["team_score"], period_kind="month_groups",
                       before_months=[10, 11, 12, 1], after_months=[2, 3, 4, 5, 6])
    client = FakeClient(tool_response({**draft.model_dump(), "home_away": "any", "relative_season": "none",
                                       "back_to_back": "any"}))
    q = make_parser(client).parse("compare team ppg from months 10,11,12,1 to months 2,3,4,5,6", today=TODAY).query
    assert q.mode == "period" and q.subject_type == "team" and q.stats == ["team_score"] and q.subject_ids == []
    assert q.period_split.before_months == [10, 11, 12, 1] and q.period_split.after_months == [2, 3, 4, 5, 6]


def test_string_null_from_the_model_is_treated_as_missing() -> None:
    from nbalab.nlp.llm import from_tool_input

    data = {**QueryDraft(subjects=["Lakers"], stats=["threes"]).model_dump(),
            "relative_season": "null", "venue": "None", "home_away": "home", "back_to_back": "any"}
    draft = QueryDraft.model_validate(from_tool_input(data))
    assert draft.relative_season is None and draft.venue is None and draft.home_away == "home"
    assert draft.back_to_back is None and draft.subjects == ["Lakers"]


def test_parser_data_dir_follows_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Hosted apps download the data elsewhere; the parser must read names from that folder."""
    monkeypatch.setenv("NBALAB_PARSER_DATA_DIR", str(tmp_path))
    assert ParserConfig.from_env().processed_dir == tmp_path
    monkeypatch.delenv("NBALAB_PARSER_DATA_DIR")
    assert ParserConfig.from_env().processed_dir.name == "processed"
