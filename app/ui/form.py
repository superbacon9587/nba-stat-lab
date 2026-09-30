"""Widgets that edit the shared StatQuery: the manual form and the editable chips.

Both call the same ``editor(field, prefix)``. The chip version of a field sits
in a popover, and the form version sits in the "Advanced / manual" expander.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from nbalab.query import schema as s
from nbalab.query.filters import describe_filter, inches_label
from nbalab.query.stats import GROUPABLE_VARIABLES, stat_catalog
from ui import backend, state
from ui.formmodel import HANDLED_FILTERS, default_form

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GROUP_LABELS = {
    "day_of_week": "Day of week", "month": "Month", "week_of_season": "Week of season", "season": "Season",
    "home_away": "Home / away", "opponent_team": "Opponent", "venue": "Arena", "rest_days": "Rest days",
    "back_to_back": "Back-to-back", "playoffs": "Playoffs vs regular season", "starter": "Starter vs bench",
    "defender_height_bucket": "Defender height", "defender_weight_bucket": "Defender weight",
    "defender_position": "Defender position",
}
TRI = {"any": "Any", "only": "Only", "exclude": "Exclude"}


def _kw(prefix: str, field: str) -> dict[str, Any]:
    return dict(key=prefix + field, on_change=state.on_change, args=(prefix, field))


def _stat_kind() -> str:
    return "team" if st.session_state.get("form_subject_type") == "team" else "player"


# --------------------------------------------------------------------- editors


def ed_subject_type(p: str) -> None:
    st.segmented_control("Subject", ["player", "team", "league"], selection_mode="single",
                         format_func={"player": "Player", "team": "Team", "league": "League-wide"}.get,
                         **_kw(p, "subject_type"))


def ed_subject_id(p: str) -> None:
    kind = st.session_state.get(p + "subject_type")
    if kind == "league":
        return
    if kind == "team":
        t = backend.team_options()
        opts = [None, *t["teamId"].astype(int).tolist()]
        st.selectbox("Team", opts, format_func=lambda i: "Pick a team…" if i is None else backend.team_label(i),
                     **_kw(p, "subject_id"))
    else:
        pl = backend.player_options()
        labels = dict(zip(pl["personId"].astype(int), pl["label"]))
        opts = [None, *labels]
        st.selectbox("Player", opts, format_func=lambda i: "Type to search any player…" if i is None else labels[i],
                     **_kw(p, "subject_id"))


def ed_stats(p: str) -> None:
    catalog = stat_catalog(_stat_kind())
    st.multiselect("Stats", list(catalog), format_func=lambda k: catalog[k].label, max_selections=4,
                   **_kw(p, "stats"))


def ed_opponents(p: str) -> None:
    t = backend.team_options()
    st.multiselect("Opponent", t["teamId"].astype(int).tolist(), format_func=backend.team_label,
                   placeholder="Any opponent", **_kw(p, "opponents"))


def ed_defender(p: str) -> None:
    pl = backend.player_options()
    labels = dict(zip(pl["personId"].astype(int), pl["label"]))
    st.selectbox("Defender (named player)", [None, *labels],
                 format_func=lambda i: "Any defender" if i is None else labels[i],
                 help="No box score records who guarded whom. Without official matchup data this uses "
                      "minutes both players shared on the floor, a labeled proxy.",
                 **_kw(p, "defender_id"))


def ed_height(p: str) -> None:
    st.toggle("Filter by defender height", **_kw(p, "height_on"))
    if st.session_state.get(p + "height_on"):
        st.slider("Defender height (inches)", 66, 90, format="%d in", **_kw(p, "height_range"))
        lo, hi = st.session_state.get(p + "height_range", (77, 80))
        st.caption(f"{inches_label(lo)} to {inches_label(hi)}")


def ed_venue(p: str) -> None:
    t = backend.team_options()
    st.selectbox("Venue (home arena of)", [None, *t["teamId"].astype(int).tolist()],
                 format_func=lambda i: "Any venue" if i is None else f"{backend.team_label(i)} arena",
                 **_kw(p, "venue_team"))


def ed_home_away(p: str) -> None:
    st.segmented_control("Home / away", ["any", "home", "away"], format_func=str.capitalize,
                         **_kw(p, "home_away"))


def ed_days(p: str) -> None:
    st.multiselect("Day of week", DAYS, placeholder="Any day", **_kw(p, "days"))


def ed_seasons(p: str) -> None:
    lo, hi = backend.season_bounds()
    st.slider("Seasons (starting year)", lo, hi, **_kw(p, "season_range"))
    a, b = st.session_state.get(p + "season_range", (lo, hi))
    st.caption(f"{a}-{str(a + 1)[-2:]} through {b}-{str(b + 1)[-2:]}")


def ed_b2b(p: str) -> None:
    st.segmented_control("Back-to-backs", list(TRI), format_func=TRI.get, **_kw(p, "back_to_back"))


def ed_playoffs(p: str) -> None:
    st.segmented_control("Playoff games", list(TRI), format_func=TRI.get, **_kw(p, "playoffs"))


def ed_last_n(p: str) -> None:
    st.number_input("Only the last N games (0 = all)", 0, 500, step=5, **_kw(p, "last_n_games"))


def ed_group_by(p: str) -> None:
    if st.session_state.get(p + "subject_type") == "league":
        vars_ = backend.effect_variables()
        st.selectbox("Variable (league-wide effect)", list(vars_),
                     format_func=lambda k: f"{k.replace('_', ' ').capitalize()}  ·  {vars_[k].unit}",
                     **_kw(p, "effect_variable"))
        st.multiselect("Players at position", ["G", "F", "C"], placeholder="All positions",
                       **_kw(p, "positions"))
        return
    opts = [None, *GROUPABLE_VARIABLES]
    st.selectbox("Split by (show every value)", opts,
                 format_func=lambda k: "No grouping" if k is None else GROUP_LABELS.get(k, k),
                 **_kw(p, "group_by"))


def ed_line(p: str) -> None:
    st.toggle("Project a future game with a betting line", **_kw(p, "line_on"),
              disabled=st.session_state.get(p + "subject_type") == "league")
    if st.session_state.get(p + "line_on"):
        st.number_input("Line (first stat)", min_value=0.0, max_value=400.0, step=0.5, format="%.1f",
                        placeholder="e.g. 27.5", **_kw(p, "line"))


def ed_minutes(p: str) -> None:
    if not st.session_state.get(p + "line_on") or st.session_state.get(p + "subject_type") != "player":
        return
    st.toggle("Override projected minutes", **_kw(p, "minutes_on"))
    if st.session_state.get(p + "minutes_on"):
        st.slider("Projected minutes", 5.0, 48.0, step=0.5, **_kw(p, "minutes"))


EDITORS: dict[str, Callable[[str], None]] = {
    "subject_id": ed_subject_id, "stats": ed_stats, "opponents": ed_opponents, "defender_id": ed_defender,
    "height_on": ed_height, "venue_team": ed_venue, "home_away": ed_home_away, "days": ed_days,
    "season_range": ed_seasons, "back_to_back": ed_b2b, "playoffs": ed_playoffs,
    "last_n_games": ed_last_n, "group_by": ed_group_by, "line_on": ed_line,
}


# ------------------------------------------------------------------ manual form


def manual_form() -> None:
    """Every filter as a widget, in three columns."""
    p = "form_"
    c0, c1 = st.columns([1, 2])
    with c0:
        ed_subject_type(p)
    with c1:
        ed_subject_id(p)
    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown("**What**")
        ed_stats(p)
        ed_group_by(p)
        ed_seasons(p)
        ed_last_n(p)
    with b:
        st.markdown("**Against whom**")
        ed_opponents(p)
        if st.session_state.get("form_subject_type") != "team":
            ed_defender(p)
            ed_height(p)
    with c:
        st.markdown("**Where and when**")
        ed_venue(p)
        ed_home_away(p)
        ed_days(p)
        ed_b2b(p)
        ed_playoffs(p)
    st.divider()
    l1, l2 = st.columns(2, gap="large")
    with l1:
        ed_line(p)
    with l2:
        ed_minutes(p)


# ------------------------------------------------------------------------ chips


def _reset_field(field: str) -> None:
    lo, hi = backend.season_bounds()
    f = state.current_form()
    f[field] = default_form(lo, hi)[field]
    if field == "line_on":
        f["line"], f["minutes_on"] = None, False
    state.apply_form(f)


def chip_items(q: s.StatQuery) -> list[tuple[str, str, bool]]:
    """(field, chip label, removable) for every active part of the query."""
    lo, hi = backend.season_bounds()
    f = state.current_form()
    d = backend.data()
    catalog = stat_catalog(q.subject_type)
    items: list[tuple[str, str, bool]] = []
    if f["subject_type"] == "league":
        items.append(("group_by", f"League-wide · {q.effect_variable or q.group_by}", False))
    else:
        name = d.team_name(f["subject_id"]) if f["subject_type"] == "team" else d.player_name(f["subject_id"])
        more = f" +{len(q.subject_ids) - 1}" if len(q.subject_ids) > 1 else ""
        items.append(("subject_id", f"{name}{more}", False))
    items.append(("stats", " + ".join(catalog[x].label for x in q.stats), False))
    if f["opponents"]:
        items.append(("opponents", "vs " + " / ".join(d.team_name(t) for t in f["opponents"]), True))
    if f["defender_id"] is not None:
        items.append(("defender_id", f"defender: {d.player_name(f['defender_id'])}", True))
    if f["height_on"]:
        a, b = f["height_range"]
        items.append(("height_on", f"defender {inches_label(a)}–{inches_label(b)}", True))
    if f["venue_team"] is not None:
        items.append(("venue_team", f"at {d.team_name(f['venue_team'])} arena", True))
    if f["home_away"] != "any":
        items.append(("home_away", f"{f['home_away']} games", True))
    if f["days"]:
        items.append(("days", "on " + "/".join(x[:3] for x in f["days"]), True))
    if tuple(f["season_range"]) != (lo, hi):
        a, b = f["season_range"]
        items.append(("season_range", f"seasons {a}-{str(a + 1)[-2:]} → {b}-{str(b + 1)[-2:]}", True))
    else:
        items.append(("season_range", f"all seasons {lo}–{hi + 1}", False))
    if f["back_to_back"] != "any":
        items.append(("back_to_back", "back-to-backs " + f["back_to_back"], True))
    if f["playoffs"] != "any":
        items.append(("playoffs", "playoffs " + f["playoffs"], True))
    if f["last_n_games"]:
        items.append(("last_n_games", f"last {f['last_n_games']} games", True))
    if f["group_by"] and f["subject_type"] != "league":
        items.append(("group_by", "split by " + GROUP_LABELS.get(f["group_by"], f["group_by"]).lower(), True))
    if f["line_on"]:
        lines = q.projection.lines if q.projection else {}
        txt = ", ".join(f"{catalog[k].label.lower()} {v:g}" for k, v in lines.items()) or "no line yet"
        items.append(("line_on", f"projection · line {txt}", True))
    return items


def chips(q: s.StatQuery) -> None:
    """The interpreted query as a row of editable chips (each opens a small editor)."""
    items = chip_items(q)
    cols = st.columns(min(len(items), 4))
    for i, (field, label, removable) in enumerate(items):
        with cols[i % len(cols)]:
            with st.popover(label, icon=":material/tune:", width="stretch"):
                EDITORS[field]("chip_")
                if removable:
                    st.button("Remove this filter", key=f"rm_{field}", icon=":material/close:",
                              on_click=_reset_field, args=(field,), type="tertiary")
    other = [(i, fl) for i, fl in enumerate(q.filters) if fl.type not in HANDLED_FILTERS]
    for i, fl in other:
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"`{describe_filter(fl, backend.data())}`")
        c2.button("Remove", key=f"rm_other_{i}", on_click=state.remove_filter, args=(i,), type="tertiary")
