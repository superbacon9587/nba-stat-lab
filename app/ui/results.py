"""The results page for split and projection queries, in the order the reader needs:
headline -> what drives it -> distribution -> the games themselves -> trend -> projection -> caveats.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from nbalab.query import schema as s
from nbalab.query.defenders import SOURCE_LABELS
from nbalab.query.engine import SplitResult, StatResult
from nbalab.query.stats import stat_catalog
from nbalab.viz import charts
from nbalab.viz import theme as T
from nbalab.viz.teams import accent
from nbalab.viz.trend import linear_trend, rolling_mean
from ui import backend, components as C, projection
from ui.form import GROUP_LABELS
from ui.logic import confidence_badge, format_value

MIN_N_FOR_INTERVAL = 5  # a bootstrap of 2-4 games gives a falsely precise interval
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def render(query_json: str) -> None:
    q = s.StatQuery.model_validate_json(query_json)
    if q.mode == "variable_effect":
        render_variable_effect(q)
        return
    if q.mode == "period":
        from ui import periods

        with st.spinner("Comparing before vs after…"):
            periods.render(q)
        return
    with st.spinner("Crunching the numbers…"):
        results = backend.split(query_json)
    if len(results) == 1:
        render_subject(results[0], 0, query_json)
        return
    tabs = st.tabs([r.subject_name for r in results])
    for i, (tab, r) in enumerate(zip(tabs, results)):
        with tab:
            render_subject(r, i, query_json)


# -------------------------------------------------------------------- one subject


def render_subject(res: SplitResult, idx: int, query_json: str) -> None:
    q = res.query
    catalog = stat_catalog(res.subject_type)
    stat = q.stats[0]
    if len(q.stats) > 1:
        stat = st.segmented_control("Stat", q.stats, default=q.stats[0], format_func=lambda k: catalog[k].label,
                                    key=f"statpick_{idx}_{hash(query_json)}") or q.stats[0]
    sr = res.stats[stat]
    C.accent_bar(accent(_subject_abbrev(res)))
    where = " · ".join(d for d in res.filters_described) or "all games"
    proxy = next((d for d in res.filters_described if d.startswith("games where")), None)
    span = f" · {_season_span(res)}" if res.seasons else ""
    st.markdown(f"### {res.subject_name} — {sr.label}{span}")
    if proxy:
        st.markdown(f"**In {proxy}** (shared floor time is a proxy: box scores never say who guarded whom)")
    st.caption(where)
    if res.career_note:
        st.info(res.career_note, icon=":material/history:")
    if is_fatigue_query(q):
        st.caption(SURVIVORSHIP_NOTE)

    grouped_only = res.groups is not None and not res.query.with_projection_context().split_filters
    if grouped_only:
        group_headline(res, stat)
    else:
        headline_cards(res, sr)
    if res.n_games == 0:
        st.warning("No games match every filter, so there is no split to show. "
                   + (res.suggestion or "Try removing a filter.")
                   + " Check the notes on the interpreted query above for a conflict (e.g. a defender who "
                   "never played for that opponent).", icon=":material/filter_alt_off:")
        if q.mode == "projection":
            projection.render(query_json, res, stat)
        data_notes(res)
        return
    if res.suggestion:
        st.warning(res.suggestion, icon=":material/filter_alt_off:")

    if grouped_only:
        group_section(res, stat, main=True)
    else:
        effects_section(res, idx, stat, query_json)
        if res.groups is not None:
            group_section(res, stat)
    if res.n_games < res.n_baseline:  # a split equal to the baseline has nothing to compare
        distribution_section(res, stat)
    game_log_section(res, stat)
    trend_section(res, stat)
    if q.mode == "projection":
        projection.render(query_json, res, stat)
    data_notes(res)


def _subject_abbrev(res: SplitResult) -> str | None:
    if res.subject_type == "team":
        return backend.team_abbrev(res.subject_id)
    g = res.baseline_games
    if g.empty:
        return None
    return backend.team_abbrev(int(g.sort_values("game_date")["teamId"].iloc[-1]))


def baseline_only_cards(res: SplitResult, sr: StatResult) -> None:
    """No split filters: the answer is the average itself, so there is no difference to test."""
    ratio = sr.is_ratio
    per36 = f"{sr.split.per36:.1f} per 36 min" if sr.split.per36 and not math.isnan(sr.split.per36) else ""
    C.cards([
        C.card("Average", format_value(sr.split.value, ratio), per36, cls="hero"),
        C.card("Games", f"{res.n_games}", _season_span(res)),
        C.card("Difference", "–", "no filters to compare with; this is the whole sample"),
        C.card("Sample size", f"{res.n_games}", "games"),
        C.card("Confidence", C.badge(confidence_badge(res.n_games, math.nan).level) if res.n_games < 25 else "n/a",
               "fewer than 25 games" if res.n_games < 25 else "a plain average; no comparison to test"),
    ])


def headline_cards(res: SplitResult, sr: StatResult) -> None:
    if not res.query.with_projection_context().split_filters and res.n_games == res.n_baseline:
        baseline_only_cards(res, sr)
        return
    ratio = sr.is_ratio
    fatigue = is_fatigue_query(res.query) and not ratio
    badge = confidence_badge(sr.split.n, sr.p_value)
    diff_cls = "nbl-up" if sr.diff > 0 else "nbl-down" if sr.diff < 0 else ""
    arrow = "▲" if sr.diff > 0 else "▼" if sr.diff < 0 else ""
    pct = "" if math.isnan(sr.diff_pct) else f"{sr.diff_pct:+.0f}% vs baseline"
    ci = (f"95% CI {sr.ci_low:+.1f} to {sr.ci_high:+.1f}"
          if not math.isnan(sr.ci_low) and sr.split.n >= MIN_N_FOR_INTERVAL else "too few games for an interval")
    shrunk = (f"shrunk estimate {format_value(sr.shrunk, ratio)}"
              if not math.isnan(sr.shrunk) and abs(sr.shrunk - sr.split.value) > 0.05 else "")
    p_txt = "" if math.isnan(sr.p_value) else f"p = {sr.p_value:.3f}"
    per36 = f"{sr.split.per36:.1f} per 36 min" if sr.split.per36 and not math.isnan(sr.split.per36) else ""
    base36 = (f"{sr.baseline.per36:.1f} per 36 min · " if fatigue and sr.baseline.per36
              and not math.isnan(sr.baseline.per36) else "")
    hero_sub = " · ".join(x for x in (per36, shrunk) if x) if fatigue else (shrunk or per36)
    C.cards([
        C.card("Under these conditions", format_value(sr.split.value, ratio), hero_sub, cls="hero"),
        C.card("Baseline", format_value(sr.baseline.value, ratio),
               f"{base36}{res.n_baseline} games · {_season_span(res)}"),
        C.card("Difference", f"{arrow} {format_value(sr.diff, ratio) if not ratio else f'{sr.diff:+.1f} pts'}"
               if not math.isnan(sr.diff) else "–", f"{pct}<br>{ci}", value_cls=diff_cls),
        C.card("Sample size", f"{res.n_games}", f"games in the split ({res.n_games / max(res.n_baseline, 1):.0%} of baseline)"),
        C.card("Confidence", C.badge(badge.level), f"{badge.reason}" + (f"<br>{p_txt}" if p_txt and p_txt not in badge.reason else "")),
    ])
    with st.expander("What do these numbers mean?", icon=":material/help:"):
        st.markdown(
            "- **Under these conditions**: the average in the games that match every filter "
            "(pooled made ÷ attempted for percentages). The **shrunk estimate** pulls small samples toward the "
            "baseline, since 12 games against one team is mostly noise. It's the better guess for a future game.\n"
            "- **Baseline**: the same player or team, same seasons, no other filters.\n"
            "- **Difference**: split minus baseline. The 95% interval comes from 2,000 bootstrap resamples. "
            "The p-value is a Welch t-test of these games vs. all the other baseline games.\n"
            f"- **Confidence**: high = 25+ games and p < 0.05; low = fewer than 10 games or p ≥ 0.20; "
            "medium in between. A *low* badge on a big sample means \"no real difference\", not bad data."
        )


def _season_span(res: SplitResult) -> str:
    if not res.seasons:
        return "no games"
    a, b = res.seasons
    return f"{a}-{str(a + 1)[-2:]}" if a == b else f"{a}-{str(a + 1)[-2:]} → {b}-{str(b + 1)[-2:]}"


# --------------------------------------------------------------------- sections


def effects_section(res: SplitResult, idx: int, stat: str, query_json: str) -> None:
    C.section("How each variable affects this stat", "Main result",
              "Each bar is one filter on its own vs. the baseline. The bold bar stacks all of them. "
              "Whiskers are 95% intervals. <b>Faded bars</b> have intervals that cross zero, so they could be noise.")
    rows = backend.effects(query_json, idx, stat)
    if not rows:
        st.info("No filters to compare yet. Add an opponent, venue, day, or defender to see its effect.",
                icon=":material/tune:")
        return
    sr = res.stats[stat]
    fig = charts.effect_chart(rows, sr.baseline.value, sr.label)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    if len(rows) > 1:
        st.caption("Individual effects don't add up to the combined bar. Filters overlap "
                   "(e.g. most games vs one team in one arena are also away games), and each bar is its own sample.")


def group_headline(res: SplitResult, stat: str) -> None:
    """Headline for "how does X affect Y": baseline, the highest and lowest level, and how sure we are."""
    t = res.groups.table[res.groups.table["stat"] == stat]
    sr = res.stats[stat]
    if t.empty:
        headline_cards(res, sr)
        return
    ratio = sr.is_ratio
    top, low = t.loc[t["diff"].idxmax()], t.loc[t["diff"].idxmin()]
    strongest = t.loc[t["p_value"].fillna(1).idxmin()]
    badge = confidence_badge(int(strongest["n"]), float(strongest["p_value"]))

    def level_card(title: str, r) -> str:
        cls = "nbl-up" if r["diff"] > 0 else "nbl-down" if r["diff"] < 0 else ""
        return C.card(title, f"{r['level']}", f"<span class='{cls}'>{'▲' if r['diff'] > 0 else '▼'} "
                      f"{r['diff']:+.2f}</span> · {format_value(r['value'], ratio)} avg · {int(r['n'])} games · "
                      f"p = {r['p_value']:.2f}")

    C.cards([
        C.card("Baseline", format_value(sr.baseline.value, ratio), f"{res.n_baseline} games · {_season_span(res)}",
               cls="hero"),
        level_card("Highest", top),
        level_card("Lowest", low),
        C.card("Spread", f"{top['diff'] - low['diff']:.2f}", "highest minus lowest level"),
        C.card("Confidence", C.badge(badge.level),
               f"strongest level: {strongest['level']}<br>{badge.reason}"),
    ])
    st.caption("With many levels, one of them will look unusual by chance. Treat single p-values near 0.05 with caution.")


def group_section(res: SplitResult, stat: str, main: bool = False) -> None:
    table = res.groups.table
    t = table[table["stat"] == stat]
    if t.empty:
        return
    var = GROUP_LABELS.get(res.groups.variable, res.groups.variable)
    C.section(f"How {var.lower()} affects this stat" if main else f"By {var.lower()}",
              "Main result" if main else "Every value of the variable",
              "Each bar compares games with that value to the overall baseline. Whiskers are 95% intervals. "
              "Faded bars could be noise.")
    rows = [charts.EffectRow(str(r.level), r.diff, r.ci_low, r.ci_high, int(r.n), r.p_value)
            for r in t.itertuples()]
    fig = charts.effect_chart(rows, float(t["baseline"].iloc[0]), res.stats[stat].label)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    t = t.assign(confidence=t["confidence"].map({"high": "25+ games", "low": "10-24 games", "very low": "under 10"})
                 .fillna(t["confidence"]))
    cols = ["level", "n", "value", *(["per36"] if is_fatigue_query(res.query) and "per36" in t else []),
            "shrunk", "diff", "ci_low", "ci_high", "p_value", "confidence"]
    show = t[cols].rename(columns={
        "level": var, "n": "Games", "value": "Average", "per36": "Per 36 min", "shrunk": "Shrunk", "diff": "vs baseline",
        "ci_low": "CI low", "ci_high": "CI high", "p_value": "p-value", "confidence": "Sample size"})
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ["Average", "Per 36 min", "Shrunk", "vs baseline", "CI low", "CI high"]}
                 | {"p-value": st.column_config.NumberColumn(format="%.3f")})


def distribution_section(res: SplitResult, stat: str) -> None:
    sd = stat_catalog(res.subject_type)[stat]
    C.section("Distribution", "Split vs. all games",
              "How often each value happened, as a share of games, so a small split and a large baseline compare fairly.")
    fig = charts.distribution_chart(sd.per_game(res.games), sd.per_game(res.baseline_games), sd.label,
                                    split_label="This split", all_label="All baseline games")
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


PLAYER_LOG = [("minutes", "MIN"), ("points", "PTS"), ("reboundsTotal", "REB"), ("assists", "AST"),
              ("threePointersMade", "3PM"), ("steals", "STL"), ("blocks", "BLK"), ("turnovers", "TOV"),
              ("fieldGoalsMade", "FGM"), ("fieldGoalsAttempted", "FGA"), ("plusMinusPoints", "+/-")]
TEAM_LOG = [("teamScore", "PTS"), ("opponentScore", "OPP"), ("reboundsTotal", "REB"), ("assists", "AST"),
            ("threePointersMade", "3PM"), ("turnovers", "TOV"), ("pace", "Pace"), ("offensiveRating", "ORtg"),
            ("defensiveRating", "DRtg")]


def game_log_table(res: SplitResult, stat: str) -> pd.DataFrame:
    """The split's games, newest first, with the requested stat up front."""
    g = res.games.sort_values("game_date", ascending=False)
    sd = stat_catalog(res.subject_type)[stat]
    out = pd.DataFrame({
        "Date": pd.to_datetime(g["game_date"]).dt.date,
        "Season": g["season_label"],
        "Opponent": g["opponent_name"] if "opponent_name" in g else g["opponentTeamId"],
        "H/A": g["home_away"].str[0].str.upper() if "home_away" in g else np.where(g["home"], "H", "A"),
        "W/L": np.where(g["win"].astype(bool), "W", "L"),
        "Type": g["game_type"],
    })
    out[sd.label] = sd.per_game(g).round(1).to_numpy()
    requested = out[sd.label].to_numpy()
    for col, label in (PLAYER_LOG if res.subject_type == "player" else TEAM_LOG):
        if col not in g or label in out:
            continue
        values = pd.to_numeric(g[col], errors="coerce").round(1).to_numpy()
        if np.allclose(values, requested, equal_nan=True):
            continue  # the requested stat is already the first stat column
        out[label] = values
    if res.defender_sources and "def_height" in g:
        out["Defender ht (in)"] = g["def_height"].round(1).to_numpy()
        out["Defender source"] = g["def_source"].to_numpy()
    return out.reset_index(drop=True)


def game_log_section(res: SplitResult, stat: str) -> None:
    C.section("Game log", f"{res.n_games} games", "Click a column header to sort.")
    table = game_log_table(res, stat)
    st.dataframe(table, hide_index=True, width="stretch", height=min(420, 38 + 35 * max(len(table), 1)))
    name = res.subject_name.lower().replace(" ", "_")
    st.download_button("Download CSV", table.to_csv(index=False).encode(), f"{name}_{stat}_split.csv",
                       "text/csv", icon=":material/download:", key=f"dl_{res.subject_id}_{stat}")


def trend_section(res: SplitResult, stat: str) -> None:
    sd = stat_catalog(res.subject_type)[stat]
    base = res.baseline_games.sort_values("game_date")
    values = sd.per_game(base)
    fit = linear_trend(base["game_date"], values)
    sub = ""
    if fit is not None:
        direction = "rising" if fit.slope_per_year > 0 else "falling"
        sig = "a clear trend" if fit.slope_p_value < 0.05 else "not a clear trend"
        p_txt = "p < 0.001" if fit.slope_p_value < 0.001 else f"p = {fit.slope_p_value:.3f}"
        sub = (f"Linear trend: {fit.slope_per_year:+.2f} per season ({direction}; {p_txt}, {sig}). "
               "The band is the 95% confidence band for the average trend line, not the range for single games.")
    C.section("Trend over time", "All baseline games", sub)
    fig = charts.trend_chart(base["game_date"], values, fit, sd.label,
                             in_split=base.index.isin(res.games.index) if len(res.games) < len(base) else None,
                             rolling=rolling_mean(values.reset_index(drop=True), 10).set_axis(values.index))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def data_notes(res: SplitResult) -> None:
    C.section("Data notes", "Read before trusting")
    lines = []
    if res.defender_note:
        lines.append(f"**Defender data:** {res.defender_note}")
        for src, n in res.defender_sources.items():
            lines.append(f"&nbsp;&nbsp;• {SOURCE_LABELS.get(src, src)}: {n} games")
    elif any(f.type in s.DEFENDER_FILTER_TYPES for f in res.query.filters):
        lines.append("**Defender data:** no defender information for these games.")
    else:
        lines.append("**Defender data:** not used in this question.")
    defender_involved = bool(res.defender_note) or any(f.type in s.DEFENDER_FILTER_TYPES for f in res.query.filters) or (
        res.query.projection is not None and res.query.projection.context.defender_person_id is not None)
    if defender_involved or res.query.mode == "projection":
        lines.append("**Named defenders in projections:** box scores never record who guarded whom, so any "
                     "named-defender effect is a *proxy* (shared floor time). It is shown as a note but "
                     "**not applied** to projected numbers: in the backtest it made predictions worse.")
    for w in res.warnings:
        if w != res.suggestion:
            lines.append(f"**Sample:** {w}")
    if res.n_games < 20 and len(res.baseline_games):
        lines.append(f"**Small sample:** only {res.n_games} games match. Averages this small are noisy; "
                     "lean on the shrunk estimate.")
    if res.query.mode == "projection":
        lines.append(f"**Calibration:** {calibration_note(res.subject_type)}")
    if is_fatigue_query(res.query):
        lines.append(f"**Rest and fatigue:** {SURVIVORSHIP_NOTE}")
    if res.n_games >= 25 and not res.warnings:
        lines.append(f"**Sample:** {res.n_games} games, enough for a stable average.")
    g = res.games if len(res.games) else res.baseline_games
    if len(g):
        d = pd.to_datetime(g["game_date"])
        lines.append(f"**Games in this split:** {d.min():%b %d, %Y} → {d.max():%b %d, %Y}")
    lo, hi = backend.date_range()
    lines.append(f"**Dataset:** box scores {lo} → {hi}. Preseason and All-Star games are excluded. "
                 "Teams are matched by franchise id, so the Sonics and Thunder are one franchise.")
    for n in res.notes:
        if "nbalab.models" not in n:
            lines.append(n)
    with st.container(border=True):
        st.markdown("  \n".join(lines))


FATIGUE_VARIABLES: frozenset[str] = frozenset({"back_to_back", "rest_days"})
SURVIVORSHIP_NOTE = ("Survivorship bias: players who sit out back-to-backs or short-rest games (rest, minor injuries, "
                     "load management) drop out of these games, so the ones who do play are the healthier, fresher ones. "
                     "Per-36-minute rates are shown next to raw averages because coaches also trim minutes on tired nights.")


def is_fatigue_query(q: s.StatQuery) -> bool:
    """Any rest or fatigue condition: a back-to-back/rest filter, grouping, effect variable or projection context."""
    if any(f.type in FATIGUE_VARIABLES for f in q.filters):
        return True
    if q.group_by in FATIGUE_VARIABLES or q.effect_variable in FATIGUE_VARIABLES:
        return True
    ctx = q.projection.context if q.projection else None
    return bool(ctx and (ctx.back_to_back is not None or ctx.rest_days is not None))


def calibration_note(subject_type: str) -> str:
    """Which projected stats have recalibrated over/under probabilities, read from the loaded bundle."""
    bundle = backend.model_bundle()
    cal = getattr(bundle, "calibration", None) or {}
    stats = cal.get(subject_type, {})
    names = {"points": "points", "assists": "assists", "reboundsTotal": "rebounds", "threePointersMade": "threes",
             "steals": "steals", "blocks": "blocks", "turnovers": "turnovers", "teamScore": "team points"}
    done = [names.get(k, k) for k, v in stats.items() if v.get("kind") != "identity"]
    if not stats:
        return "no calibration file loaded; over/under probabilities are the raw model's."
    kept = [names.get(k, k) for k, v in stats.items() if v.get("kind") == "identity"]
    return (f"over/under probabilities recalibrated on 2024-25 (checked on 2025-26) for {', '.join(done) or 'none'}; "
            f"raw model probabilities for {', '.join(kept) or 'none'}. Combo stats (PRA etc.) are not recalibrated.")


# --------------------------------------------------------------- league-wide mode


def render_variable_effect(q: s.StatQuery) -> None:
    from ui import explore

    hold = st.toggle("Hold minutes fixed", value=False, key=f"ve_min_{q.model_dump_json()}",
                     help="Off: the total effect, including changes in playing time. "
                          "On: the effect for the same minutes played.")
    explore.render_effect(q, heading=True, control_minutes=hold)
