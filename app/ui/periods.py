"""Period comparisons in the app: every team, one team, one player, or a player ranking,
before vs after a split point (the All-Star break by default).

Every number comes from :mod:`nbalab.query.periods`; this module only lays it out.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from nbalab.query import schema as s
from nbalab.query.stats import stat_catalog
from nbalab.viz import periods as V
from ui import backend, components as C

PLOT_CONFIG = {"displaylogo": False}
KIND_LABELS = {"all_star_break": "All-Star break", "custom_date": "Custom date", "month_groups": "Month groups",
               "last_n_before_playoffs": "Last N games before the playoffs"}
MONTHS = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun"}


def season_label(y: int) -> str:
    return f"{y}-{str(y + 1)[-2:]}"


def scope_text(seasons: list[int]) -> str:
    if not seasons:
        return "no seasons"
    return season_label(seasons[0]) if len(seasons) == 1 else f"{season_label(seasons[0])} → {season_label(seasons[-1])}"


def fmt(x: float, d: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


# ------------------------------------------------------------------ entry point


def render(q: s.StatQuery, key: str = "ask") -> None:
    if q.subject_type == "team" and not q.subject_ids:
        render_all_teams(q, key)
    elif q.subject_type == "team":
        render_team(q, q.subject_ids[0], key)
    elif q.subject_ids:
        render_player(q, key)
    else:
        render_leaderboard(q, key)


def split_editor(q: s.StatQuery, on_change, key: str) -> None:
    """Chip-style editor for the split point; ``on_change(new_split)`` applies it."""
    sp = q.period_split
    with st.popover(f"Split: {sp.describe()}", icon=":material/call_split:"):
        kind = st.selectbox("Split point", list(KIND_LABELS), index=list(KIND_LABELS).index(sp.kind),
                            format_func=KIND_LABELS.get, key=f"{key}_kind")
        new: dict[str, Any] = {"kind": kind}
        if kind == "custom_date":
            new["date"] = st.text_input("Date (MM-DD)", value=sp.date or "02-20", key=f"{key}_date")
        elif kind == "month_groups":
            new["before_months"] = st.multiselect("Before", list(MONTHS), default=sp.before_months or [10, 11, 12, 1],
                                                  format_func=MONTHS.get, key=f"{key}_bm")
            new["after_months"] = st.multiselect("After", list(MONTHS), default=sp.after_months or [2, 3, 4],
                                                 format_func=MONTHS.get, key=f"{key}_am")
        elif kind == "last_n_before_playoffs":
            new["n_games"] = st.slider("Last N games", 5, 40, sp.n_games, key=f"{key}_n")
        new["include_playoffs"] = st.toggle("Count playoff games as 'after'", value=sp.include_playoffs,
                                            key=f"{key}_po", disabled=kind == "month_groups")
        if st.button("Apply", key=f"{key}_apply", type="primary"):
            try:
                split = s.PeriodSplit(**new)
            except ValueError as exc:
                st.warning(f"Can't use that split: {exc}")
            else:
                on_change(split)
                st.rerun()


# ------------------------------------------------------------------ all teams


def render_all_teams(q: s.StatQuery, key: str) -> None:
    stat = stat_catalog("team")[q.stats[0]]
    qj = q.model_dump_json()
    base = backend.period_team_table(qj, None)
    seasons = base.seasons
    c1, c2, c3 = st.columns([1.4, 1.2, 1.2])
    choice = c1.selectbox("Seasons", ["average"] + seasons[::-1], key=f"{key}_season",
                          format_func=lambda v: f"Average of {scope_text(seasons)}" if v == "average" else season_label(v))
    adjusted = c2.toggle("League-adjusted", value=False, key=f"{key}_adj",
                         help="Subtract the league-wide change over the same periods, so a team that only moved "
                              "with the league shows about 0.")
    chart = c3.segmented_control("Chart", ["Dumbbell", "Change bars"], default="Dumbbell", key=f"{key}_chart")
    res = base if choice == "average" else backend.period_team_table(qj, int(choice))
    t = res.table
    if t.empty:
        st.info("No team has games on both sides of the split in these seasons.")
        return
    st.markdown(f"### Every team — {stat.label} · {q.period_split.describe()} · "
                f"{scope_text(res.seasons) if choice == 'average' else season_label(int(choice))}")
    league = res.league
    col, lo, hi, verdict, qcol = (("adj_change", "adj_ci_low", "adj_ci_high", "adj_verdict", "adj_q_value") if adjusted
                                  else ("change", "ci_low", "ci_high", "verdict", "q_value"))
    real = int((t[verdict] == "likely real").sum())
    C.cards([
        C.card("League change", fmt(league.diff, 2, True) if league else "–",
               f"{fmt(league.before)} → {fmt(league.after)} per game" if league else "", cls="hero"),
        C.card("Teams likely real", f"{real} / {len(t)}",
               "Benjamini-Hochberg q < 0.10" + (" · league-adjusted" if adjusted else "")),
        C.card("Games per team", f"{(t['games_before'] / t['seasons']).median():.0f} / "
               f"{(t['games_after'] / t['seasons']).median():.0f}", "median per season, before / after"),
        C.card("Seasons", f"{len(res.seasons)}", "equal weight per season" if len(res.seasons) > 1 else ""),
    ])
    st.caption("With 30 teams, 1-2 will look 'significant' at 5% by pure chance, so p-values are adjusted "
               "(Benjamini-Hochberg). 'Likely real' = q < 0.10; everything else could be noise.")
    abbrevs = {int(tid): backend.team_abbrev(int(tid)) for tid in t["teamId"]}
    if chart == "Change bars" or adjusted:
        fig = V.change_bars(t, col, lo, hi, stat.label, abbrevs, stat.higher_is_better, verdict)
    else:
        fig = V.dumbbell_chart(t, stat.label, abbrevs, stat.higher_is_better)
    event = st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, on_select="rerun", selection_mode="points",
                            key=f"{key}_fig_{choice}_{adjusted}_{chart}")
    picked = _picked_team(event)
    table = _team_display(t, adjusted, len(res.seasons) > 1)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"p": st.column_config.NumberColumn(format="%.3f"),
                                "q (BH)": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download CSV", t.to_csv(index=False).encode(), f"teams_{stat.name}_{q.period_split.kind}.csv",
                       "text/csv", icon=":material/download:", key=f"{key}_dl")
    names = dict(zip(t["teamId"], t["team"]))
    default = list(names).index(picked) + 1 if picked in names else 0
    team = st.selectbox("Open a team (or click one in the chart)", [None, *names], index=default,
                        format_func=lambda i: "—" if i is None else names[i], key=f"{key}_open")
    if team is not None:
        render_team(q.model_copy(update={"subject_ids": []}), int(team), f"{key}_detail", heading=names[team])
    notes_box(res.notes, res.split_sources, q)


def _picked_team(event: Any) -> int | None:
    try:
        pts = event.selection.points  # type: ignore[union-attr]
        return int(pts[0]["customdata"]) if pts else None
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _team_display(t: pd.DataFrame, adjusted: bool, multi: bool) -> pd.DataFrame:
    out = pd.DataFrame({
        "Team": t["team"], "Before": t["before"].round(1), "After": t["after"].round(1),
        "Change": t["change"].round(2), "% change": t["pct_change"].round(1),
        "vs league": t["adj_change"].round(2),
        "95% CI": [f"{a:+.1f} to {b:+.1f}" for a, b in zip(
            t["adj_ci_low" if adjusted else "ci_low"], t["adj_ci_high" if adjusted else "ci_high"])],
        "p": t["adj_p_value" if adjusted else "p_value"], "q (BH)": t["adj_q_value" if adjusted else "q_value"],
        "Verdict": t["adj_verdict" if adjusted else "verdict"],
        "Games before": t["games_before"], "Games after": t["games_after"],
    })
    if multi:
        col = "seasons_improved_vs_league" if adjusted else "seasons_improved"
        out["Consistency"] = [f"{a}/{b} seasons better" for a, b in zip(t[col], t["seasons"])]
    return out


# ------------------------------------------------------------------ one team


def render_team(q: s.StatQuery, team_id: int, key: str, heading: str | None = None) -> None:
    stat = stat_catalog("team")[q.stats[0]]
    qj = q.model_dump_json()
    base = backend.period_team_table(qj, None)
    row = base.table[base.table["teamId"] == team_id]
    name = heading or backend.data().team_name(team_id, max(base.seasons) if base.seasons else None)
    st.markdown(f"#### {name} — {stat.label} · {q.period_split.describe()} · {scope_text(base.seasons)}")
    if row.empty:
        st.info("No games on both sides of the split in these seasons.")
        return
    r = row.iloc[0]
    better = "nbl-up" if r["better"] == "better" else "nbl-down"
    C.cards([
        C.card("Before", fmt(r["before"]), f"{int(r['games_before'])} games", cls="hero"),
        C.card("After", fmt(r["after"]), f"{int(r['games_after'])} games"),
        C.card("Change", fmt(r["change"], 2, True), f"95% CI {fmt(r['ci_low'], 1, True)} to {fmt(r['ci_high'], 1, True)}",
               value_cls=better),
        C.card("vs league", fmt(r["adj_change"], 2, True), f"q = {r['adj_q_value']:.2f} · {r['adj_verdict']}"),
        C.card("Consistency", f"{int(r['seasons_improved'])}/{int(r['seasons'])}", "seasons better after the split"),
    ])
    detail = backend.period_team_detail(qj, team_id)
    if len(detail["seasonal"]) > 1:
        st.markdown("**Season by season**")
        st.plotly_chart(V.seasonal_change_chart(detail["seasonal"], stat.label, stat.higher_is_better),
                        width="stretch", config=PLOT_CONFIG, key=f"{key}_seasonal_{team_id}")
    if not detail["monthly"].empty:
        st.markdown("**Month by month**")
        st.plotly_chart(V.monthly_trend_chart(detail["monthly"], stat.label, name), width="stretch",
                        config=PLOT_CONFIG, key=f"{key}_monthly_{team_id}")
    if heading is None:
        notes_box(base.notes, base.split_sources, q)


# ------------------------------------------------------------------ one player


def render_player(q: s.StatQuery, key: str) -> None:
    stat = stat_catalog("player")[q.stats[0]]
    res = backend.period_player_detail(q.model_dump_json())
    name = backend.data().player_name(q.subject_ids[0])
    seasons = sorted(res.seasonal["season"].unique()) if not res.seasonal.empty else []
    st.markdown(f"### {name} — {stat.label} · {q.period_split.describe()} · {scope_text(seasons)}")
    if res.raw is None:
        st.info("Not enough games on both sides of the split (5+ each) in any season.")
        return
    raw, p36, mins = res.raw, res.per36, res.minutes

    def cls(c) -> str:
        return "nbl-up" if (c.diff > 0) == stat.higher_is_better else "nbl-down"

    cards = [
        C.card("Before → after", f"{fmt(raw.before)} → {fmt(raw.after)}", f"{raw.n_before} / {raw.n_after} games",
               cls="hero"),
        C.card("Change (raw)", fmt(raw.diff, 2, True), f"95% CI {fmt(raw.ci[0], 1, True)} to {fmt(raw.ci[1], 1, True)}"
               f"<br>p = {raw.p_value:.3f}", value_cls=cls(raw)),
    ]
    if p36 is not None:
        cards.append(C.card("Change per 36 min", fmt(p36.diff, 2, True),
                            f"{fmt(p36.before)} → {fmt(p36.after)} per 36<br>p = {p36.p_value:.3f}", value_cls=cls(p36)))
    if mins is not None:
        cards.append(C.card("Minutes", fmt(mins.diff, 1, True), f"{fmt(mins.before)} → {fmt(mins.after)} per game"))
    cards.append(C.card("Consistency", f"{raw.seasons_improved}/{raw.seasons}", "seasons better after the split"))
    C.cards(cards)
    st.caption("Seasons are weighted by precision; seasons with fewer than 5 games on a side are left out.")
    if not res.monthly.empty:
        st.markdown("**Month by month, into the playoffs**")
        st.plotly_chart(V.monthly_trend_chart(res.monthly, stat.label, name), width="stretch", config=PLOT_CONFIG,
                        key=f"{key}_pm")
        st.caption("Dashed line: the league-wide monthly pattern, rescaled to this player's level, to show whether "
                   "a dip or rise is his own or the whole league's.")
    if len(res.seasonal) > 1:
        st.markdown("**Post-split change, season by season** (consistent pattern or one-off?)")
        st.plotly_chart(V.seasonal_change_chart(res.seasonal, stat.label, stat.higher_is_better), width="stretch",
                        config=PLOT_CONFIG, key=f"{key}_ps")
    if res.last_n:
        rows = [{"Split": f"Last {n} regular-season games vs the rest", "Rest": round(c.before, 1),
                 f"Last {n}": round(c.after, 1), "Change": round(c.diff, 2),
                 "95% CI": f"{c.ci[0]:+.1f} to {c.ci[1]:+.1f}", "p": round(c.p_value, 3)} for n, c in res.last_n.items()]
        st.markdown("**Heading into the playoffs**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not res.seasonal.empty:
        with st.expander("Season table", icon=":material/table:"):
            show = res.seasonal.rename(columns={"before": "Before", "after": "After", "diff": "Change",
                                                "n_before": "Games before", "n_after": "Games after",
                                                "before36": "Before /36", "after36": "After /36", "diff36": "Change /36"})
            show["season"] = show["season"].map(season_label)
            st.dataframe(show.drop(columns=["personId", "se", "improved"], errors="ignore").round(2), hide_index=True,
                         width="stretch")
    notes_box(res.notes, {}, q)


# ------------------------------------------------------------------ player ranking


def render_leaderboard(q: s.StatQuery, key: str) -> None:
    stat = stat_catalog("player")[q.stats[0]]
    c1, c2 = st.columns(2)
    min_games = c1.slider("Minimum games on each side (per season)", 5, 30, 10, key=f"{key}_min")
    per36 = c2.toggle("Rank by per-36 change", value=False, key=f"{key}_p36",
                      help="Per-36 removes changes that come only from more or fewer minutes.")
    lb = backend.period_leaderboard(q.model_dump_json(), min_games, per36)
    seasons = [f.start for f in q.filters if isinstance(f, s.SeasonRangeFilter)]
    st.markdown(f"### Who changes most — {stat.label}{' per 36' if per36 else ''} · {q.period_split.describe()}"
                + (f" · {season_label(seasons[0])}" if len(seasons) == 1 else ""))
    if lb.empty:
        st.info("No player has enough games on both sides.")
        return
    st.caption("Ranked by the change after shrinkage toward the league-average change: a small, noisy sample is "
               "pulled toward the middle, so a hot 12-game stretch can't top the list on its own.")
    show = lb[["player", "before", "after", "change", "shrunk_change", "games_before", "games_after", "seasons"]]
    st.dataframe(show.rename(columns={"player": "Player", "before": "Before", "after": "After", "change": "Change",
                                      "shrunk_change": "Change (shrunk)", "games_before": "Games before",
                                      "games_after": "Games after", "seasons": "Seasons"}).round(2),
                 hide_index=True, width="stretch")
    notes_box(["Late-season changes often reflect role, not skill: players on teams out of the race get more "
               "minutes and shots after the break. Compare with the per-36 ranking."], {}, q)


# ------------------------------------------------------------------ notes


def notes_box(notes: list[str], sources: dict[int, str], q: s.StatQuery) -> None:
    C.section("Data notes", "Read before trusting")
    lines = [f"**Split:** {q.period_split.describe()}; regular-season games only"
             + ("" if not q.period_split.include_playoffs else ", plus playoff games as 'after'") + "."]
    if sources:
        from nbalab.data.calendar import SOURCE_LABELS

        by_source: dict[str, list[int]] = {}
        for yr, src in sources.items():
            by_source.setdefault(src, []).append(yr)
        for src, yrs in by_source.items():
            lines.append(f"**Break dates ({len(yrs)} season{'s' if len(yrs) != 1 else ''}):** {SOURCE_LABELS.get(src, src)}.")
    lines += [f"- {n}" for n in notes]
    lo, hi = backend.date_range()
    lines.append(f"**Dataset:** box scores {lo} → {hi}. Preseason and All-Star games are excluded.")
    with st.container(border=True):
        st.markdown("  \n".join(lines))
