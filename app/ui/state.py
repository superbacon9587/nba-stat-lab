"""Session state: the one StatQuery that the prompt box, the chips and the manual form share.

Every edit runs in a widget callback:
    current StatQuery -> form fields -> change one field -> new StatQuery -> refresh every widget.
Refreshing all widgets from the new query keeps the chips and the manual form
in step no matter which one the user touched.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import streamlit as st

from nbalab.nlp.build import TEAM_DEFAULT_SEASONS
from nbalab.query import schema as s
from nbalab.query.stats import stat_names_for
from ui import backend
from ui.formmodel import FIELDS, default_form, form_to_query, query_to_form

PREFIXES = ("form_", "chip_")


def init() -> None:
    """Create the session keys and widget defaults on first load."""
    ss = st.session_state
    ss.setdefault("sq", None)
    ss.setdefault("sq_error", None)
    ss.setdefault("parsed", None)
    ss.setdefault("ran", None)
    ss.setdefault("prompt", "")
    ss.setdefault("show_manual", False)
    if "form_subject_type" not in ss:
        _push(default_form(*backend.season_bounds()))


def query() -> s.StatQuery | None:
    return st.session_state.get("sq")


def current_form() -> dict[str, Any]:
    q = query()
    lo, hi = backend.season_bounds()
    return query_to_form(q, lo, hi) if q is not None else _form_from_widgets("form_")


def set_query(q: s.StatQuery | None) -> None:
    st.session_state["sq"] = q
    st.session_state["sq_error"] = None
    lo, hi = backend.season_bounds()
    _push(query_to_form(q, lo, hi) if q is not None else default_form(lo, hi))


def _push(form: dict[str, Any]) -> None:
    for prefix in PREFIXES:
        for field in FIELDS:
            st.session_state[prefix + field] = form[field]


def _form_from_widgets(prefix: str) -> dict[str, Any]:
    lo, hi = backend.season_bounds()
    f = default_form(lo, hi)
    for field in FIELDS:
        if prefix + field in st.session_state:
            f[field] = st.session_state[prefix + field]
    return f


def on_change(prefix: str, field: str) -> None:
    """Widget callback: apply one edited field to the shared query."""
    f = current_form()
    f[field] = st.session_state[prefix + field]
    if field == "subject_type":
        f = retarget_subject_type(f)
    apply_form(f)


def apply_form(f: dict[str, Any]) -> None:
    lo, hi = backend.season_bounds()
    if f["subject_id"] is None and f["subject_type"] != "league":
        _hold(f)  # nothing to query until a player or team is picked
        return
    try:
        q = form_to_query(f, query(), lo, hi)
    except ValueError as exc:
        _hold(f, keep_query=True)
        st.session_state["sq_error"] = _short_error(exc)
        return
    set_query(q)


def _hold(f: dict[str, Any], keep_query: bool = False) -> None:
    """Show the edited fields in every widget without changing (or with clearing) the query."""
    if not keep_query:
        st.session_state["sq"] = None
    st.session_state["sq_error"] = None
    for prefix in PREFIXES:
        for k, v in f.items():
            st.session_state[prefix + k] = v


def retarget_subject_type(f: dict[str, Any]) -> dict[str, Any]:
    """Switching player <-> team <-> league: clear the subject, keep stats that still exist."""
    f = dict(f)
    f["subject_id"] = None
    kind = "team" if f["subject_type"] == "team" else "player"
    valid = stat_names_for(kind)
    mapped = [("team_score" if (kind == "team" and x == "points") else
               "points" if (kind == "player" and x == "team_score") else x) for x in f["stats"]]
    f["stats"] = [x for x in mapped if x in valid] or (["team_score"] if kind == "team" else ["points"])
    f["line_on"] = f["line_on"] and f["subject_type"] != "league"
    if f["subject_type"] == "league" and not f["effect_variable"]:
        f["effect_variable"] = "home_away"
    lo, hi = backend.season_bounds()
    if kind == "team" and f["subject_type"] == "team" and tuple(f["season_range"]) == (lo, hi):
        f["season_range"] = (max(hi - TEAM_DEFAULT_SEASONS + 1, lo), hi)  # same default as the parser
    return f


def apply_fix(query_json: str) -> None:
    """One-click conflict fix: replace the query with the corrected one; the user runs it again."""
    set_query(s.StatQuery.model_validate_json(query_json))
    st.session_state["ran"] = None
    parsed = st.session_state.get("parsed")
    if parsed is not None:  # the parser's own warning about this conflict is now stale
        keep = [u for u in parsed.unresolved if not (u.kind == "conflict" and u.field in ("defender", "opponent"))]
        st.session_state["parsed"] = dataclasses.replace(parsed, unresolved=keep)


def _short_error(exc: Exception) -> str:
    msg = str(exc)
    return msg.split("\n")[1].strip() if "validation error" in msg and "\n" in msg else msg


# ------------------------------------------------------------------- prompt flow


def parse_current_prompt() -> None:
    """Callback for the Interpret button and the example prompts."""
    text = (st.session_state.get("prompt") or "").strip()
    if not text:
        return
    try:
        res, note = backend.parse(text)
    except Exception as exc:  # parser missing or crashed: tell the user, keep the old query
        st.session_state["parsed"] = None
        st.session_state["sq_error"] = f"Could not interpret the question: {exc}"
        return
    for k in [k for k in st.session_state if str(k).startswith("cand_")]:
        del st.session_state[k]
    st.session_state["parsed"] = res
    st.session_state["parse_note"] = note
    st.session_state["ran"] = None
    set_query(res.query)
    if res.query is None:
        st.session_state["sq_error"] = "Couldn't build a query from that question. See the notes below."


def use_example(text: str) -> None:
    st.session_state["prompt"] = text
    parse_current_prompt()
    if query() is not None:
        run()


def run() -> None:
    q = query()
    st.session_state["ran"] = q.model_dump_json() if q is not None else None


def remove_filter(index: int) -> None:
    q = query()
    if q is None:
        return
    filters = [f for i, f in enumerate(q.filters) if i != index]
    set_query(q.model_copy(update={"filters": filters}))


def clear_projection() -> None:
    q = query()
    if q is None:
        return
    set_query(q.with_projection_context().model_copy(update={"projection": None, "mode": "split"}))


def swap(old_id: int, new_id: int) -> None:
    q = query()
    if q is not None and old_id != new_id:
        set_query(backend.swap_entity(q, old_id, new_id))


def reset() -> None:
    st.session_state["prompt"] = ""
    st.session_state["parsed"] = None
    st.session_state["ran"] = None
    set_query(None)
