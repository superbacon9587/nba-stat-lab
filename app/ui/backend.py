"""Cached access to data, the parser, the split engine, and the projection models.

Caching choices:
- The processed tables are one large shared object that is never mutated, so
  they use ``st.cache_resource`` (``st.cache_data`` would deep-copy ~230 MB on
  every rerun). Small derived tables (dropdown options) use ``st.cache_data``.
- Query results are keyed by the query's JSON, so the same question never runs twice.
- Trained models use ``st.cache_resource``.

The parser (``nbalab.nlp``) and models (``nbalab.models``) are imported lazily so
the rest of the app keeps working when either is missing.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nbalab.query import schema as s
from nbalab.query.data import QueryData, load_query_data
from nbalab.query.engine import SplitResult, run_split

ROOT = Path(__file__).resolve().parents[2]


def _secrets_to_env() -> None:
    """Copy st.secrets (hosted deploys) into os.environ, where the parser looks for its key."""
    try:
        from nbalab.deploy.secrets import export_secrets_to_env
    except ImportError:
        return
    export_secrets_to_env()


# ------------------------------------------------------------------------- data


@st.cache_resource(show_spinner="Loading NBA data…")
def data() -> QueryData:
    """Processed tables from data/deploy/ (slim, hosted) or data/processed/ (local)."""
    try:
        from nbalab.deploy.fetch import ensure_data
    except ImportError:
        return load_query_data()
    return load_query_data(ensure_data())


def season_bounds() -> tuple[int, int]:
    pg = data().player_games
    return int(pg["season"].min()), int(pg["season"].max())


@st.cache_data(show_spinner=False)
def date_range() -> tuple[str, str]:
    d = pd.to_datetime(data().player_games["game_date"])
    return d.min().strftime("%b %d, %Y"), d.max().strftime("%b %d, %Y")


@st.cache_data(show_spinner=False)
def player_options() -> pd.DataFrame:
    """personId, label ("Stephen Curry · 2009–25 · 1229 g"), sorted by games played."""
    p = data().players[["personId", "full_name", "first_season", "last_season", "games", "last_team_id"]].copy()
    p = p.sort_values("games", ascending=False)
    last = p["last_season"].astype(int)
    p["label"] = (p["full_name"] + " · " + p["first_season"].astype(int).astype(str) + "–"
                  + ((last + 1) % 100).map("{:02d}".format) + " · " + p["games"].astype(int).astype(str) + " g")
    return p.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def team_options() -> pd.DataFrame:
    """teamId, abbrev, full_name, city, sorted by name."""
    return data().teams[["teamId", "abbrev", "full_name", "city"]].sort_values("full_name").reset_index(drop=True)


def player_label(pid: int | None) -> str:
    if pid is None:
        return "—"
    opts = player_options()
    row = opts.loc[opts["personId"] == pid, "label"]
    return str(row.iloc[0]) if len(row) else data().player_name(pid)


def team_label(tid: int | None) -> str:
    return "—" if tid is None else data().team_name(int(tid))


def team_abbrev(tid: int | None) -> str | None:
    t = team_options()
    row = t.loc[t["teamId"] == tid, "abbrev"]
    return str(row.iloc[0]) if len(row) else None


# ----------------------------------------------------------------------- parser


def parser_module():
    """The parser module, or None if it is not installed yet."""
    try:
        _secrets_to_env()
        return importlib.import_module("nbalab.nlp.parser")
    except Exception:
        return None


def has_llm_key() -> bool:
    """True when an Anthropic key is configured (same lookup the parser uses)."""
    try:
        from nbalab.deploy.secrets import anthropic_api_key
    except ImportError:
        _secrets_to_env()
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return anthropic_api_key() is not None


@st.cache_resource(show_spinner=False)
def _parse_cache() -> dict[tuple[str, str], Any]:
    """Parsed prompts shared by all sessions, keyed by (normalized prompt, parser used)."""
    return {}


def parse(prompt: str) -> tuple[Any, str]:
    """(ParseResult, note) for a prompt. The note explains a fallback to the rule-based parser.

    A prompt already parsed by the LLM is reused for free. Otherwise the deploy
    rate limiter picks LLM vs rules, spending quota only on a real LLM call.
    """
    mod = parser_module()
    if mod is None:
        raise RuntimeError("The prompt parser (nbalab.nlp.parser) is not available.")
    key = " ".join(prompt.lower().split())
    cache = _parse_cache()
    if (key, "llm") in cache:
        return cache[(key, "llm")], ""
    backend, note = "rules", ""
    try:
        from nbalab.deploy.rate_limit import route_question

        route = route_question()
        backend, note = route.parser, route.reason
    except ImportError:
        backend = "auto" if has_llm_key() else "rules"
    if (key, backend) in cache:
        return cache[(key, backend)], note
    result = mod.parse_prompt(prompt, backend=backend)
    cache[(key, getattr(result, "source", backend))] = result
    return result, note


def swap_entity(q: s.StatQuery, old_id: int, new_id: int) -> s.StatQuery:
    """Replace an entity id everywhere in the query (the parser owns this logic)."""
    return parser_module().swap_entity(q, old_id, new_id)


# ----------------------------------------------------------------------- engine


@st.cache_data(show_spinner=False, max_entries=128)
def split(query_json: str) -> list[SplitResult]:
    return run_split(s.StatQuery.model_validate_json(query_json), data())


@st.cache_data(show_spinner=False, max_entries=256)
def effects(query_json: str, subject_index: int, stat: str) -> list:
    from ui.logic import effect_rows

    result = split(query_json)[subject_index]
    return effect_rows(result, stat, data())


@st.cache_data(show_spinner=False, max_entries=64)
def variable_effect(query_json: str, control_minutes: bool = True) -> Any:
    from nbalab.query.variable_effect import run_variable_effect

    return run_variable_effect(s.StatQuery.model_validate_json(query_json), data(), control_minutes=control_minutes)


@st.cache_data(show_spinner=False, max_entries=128)
def conflicts(query_json: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Impossible combinations in the query: (message, [(fix label, fixed query json)])."""
    from nbalab.query.conflicts import find_conflicts

    q = s.StatQuery.model_validate_json(query_json)
    return [(c.message, [(f.label, f.query.model_dump_json()) for f in c.fixes]) for c in find_conflicts(q, data())]


# ----------------------------------------------------------------------- periods


@st.cache_data(show_spinner=False, max_entries=64)
def period_team_table(query_json: str, season: int | None) -> Any:
    from nbalab.query.periods import team_table

    return team_table(s.StatQuery.model_validate_json(query_json), data(), season)


@st.cache_data(show_spinner=False, max_entries=64)
def period_team_detail(query_json: str, team_id: int) -> Any:
    from nbalab.query.periods import team_detail

    return team_detail(s.StatQuery.model_validate_json(query_json), data(), team_id)


@st.cache_data(show_spinner=False, max_entries=64)
def period_player_detail(query_json: str) -> Any:
    from nbalab.query.periods import player_detail

    return player_detail(s.StatQuery.model_validate_json(query_json), data())


@st.cache_data(show_spinner=False, max_entries=64)
def period_leaderboard(query_json: str, min_games: int, per36: bool) -> Any:
    from nbalab.query.periods import player_leaderboard

    return player_leaderboard(s.StatQuery.model_validate_json(query_json), data(), min_games=min_games, per36=per36)


def effect_variables() -> dict[str, Any]:
    from nbalab.query.variable_effect import VARIABLES

    return VARIABLES


# ----------------------------------------------------------------------- models


def models_module():
    try:
        return importlib.import_module("nbalab.models")
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading projection models…")
def model_bundle() -> Any:
    mod = models_module()
    if mod is None or not hasattr(mod, "load_bundle"):
        return None
    try:
        return mod.load_bundle()
    except Exception as exc:  # models not trained yet
        return exc


def project(query_json: str, projected_minutes: float | None) -> Any:
    """Projection result, or an Exception / None explaining why there is none."""
    return _project(query_json, None if projected_minutes is None else round(float(projected_minutes), 1))


@st.cache_resource(show_spinner=False, max_entries=64)
def _project(query_json: str, projected_minutes: float | None) -> Any:
    mod, bundle = models_module(), model_bundle()
    if mod is None:
        return None
    if bundle is None or isinstance(bundle, Exception):
        return bundle
    try:
        return mod.project(s.StatQuery.model_validate_json(query_json), data(), bundle,
                           projected_minutes=projected_minutes)
    except Exception as exc:
        return exc


# ----------------------------------------------------------------------- report


def read_doc(name: str) -> str | None:
    path = ROOT / "docs" / name
    return path.read_text() if path.exists() else None


def doc_path(rel: str) -> Path:
    return ROOT / "docs" / rel
