"""Methodology: the data, the split engine, shrinkage, the model, and the backtest."""

from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

from nbalab.viz import charts
from ui import backend, components as C

C.header("views/methodology.py")
st.markdown("## How NBA Stat Lab works")
st.markdown(
    "<div class='nbl-note'>One rule runs through the whole app: <b>the AI never produces a number.</b> "
    "Claude only turns your question into a structured query (JSON). Every statistic, interval and "
    "probability is computed by deterministic, unit-tested Python.</div>", unsafe_allow_html=True)

lo, hi = backend.date_range()
d = backend.data()

with st.container(border=True):
    st.markdown("#### 1 · The pipeline")
    st.markdown(
        "**Question → parser → StatQuery → split engine / model → charts.**\n\n"
        "1. **Parser.** Claude reads the question and fills in a strict schema: subject, stats, filters, "
        "optional line. Names like *Steph*, *Philly* or *the Sonics* are matched to ids by fuzzy matching in our own "
        "code, not by the AI. With no API key, a rule-based parser does the same job.\n"
        "2. **You check it.** The interpreted query appears as chips plus a list of assumptions, and you can edit "
        "any of it before running.\n"
        "3. **Engines.** The split engine filters the games and runs the statistics. The projection model "
        "forecasts the next game."
    )

with st.container(border=True):
    st.markdown("#### 2 · The data")
    st.markdown(
        f"- Box scores for every regular-season, play-in and playoff game from **{lo} to {hi}**: "
        f"{len(d.player_games):,} player-games and {len(d.team_games):,} team-games for "
        f"{len(d.players):,} players.\n"
        "- Preseason and All-Star games are dropped. Game types with messy labels are normalized first.\n"
        "- Teams are always matched by franchise id, never by name, so the Seattle SuperSonics → OKC Thunder "
        "and New Jersey → Brooklyn Nets histories stay connected.\n"
        "- Rolling form features (last 5/10/20 games) and opponent ratings use only games *before* the one being "
        "described, so nothing leaks from the future.\n"
        "- **Defender data is the big limit.** Box scores never record who guarded whom. The app uses official NBA "
        "matchup data when it's available and says so. Otherwise it falls back to labeled proxies: minutes two players "
        "shared on the floor, the opponent's starter at the same position, or the opponent's minutes-weighted height. "
        "Every result names the source it used."
    )

with st.container(border=True):
    st.markdown("#### 3 · The split engine")
    st.markdown(
        "- **Baseline**: the player's (or team's) games in the chosen seasons. **Split**: the baseline games that "
        "pass every filter. Filters stack with AND.\n"
        "- **Difference**: split average − baseline average, also shown as a percent and a z-score.\n"
        "- **Is it real?** A Welch t-test compares the split games to the *other* baseline games. It doesn't assume "
        "equal variances, which matters when the split is small. The 95% interval for the difference comes from "
        "2,000 bootstrap resamples. Percentages like FG% are pooled (total makes ÷ total attempts), so a 1-for-1 night "
        "doesn't count as much as 10-for-20.\n"
        "- **Each variable's effect**: the main chart re-runs the split with one filter at a time. You see what "
        "*vs the Spurs* does on its own, what *on the road* does on its own, and then both together."
    )

with st.container(border=True):
    st.markdown("#### 4 · Shrinkage: why small samples get pulled toward normal")
    st.markdown(
        "A player who averaged 31 points in 8 games against one team probably didn't become a 31-point scorer "
        "against them. Some of that is luck. **Empirical Bayes shrinkage** blends the split average with the "
        "baseline:\n\n"
        "> shrunk = B × split average + (1 − B) × baseline, where B = τ² / (τ² + σ²/n)\n\n"
        "- σ²/n is how noisy an n-game average is.\n"
        "- τ² is how much a player's averages *really* differ between contexts. It's estimated from the data: "
        "how much more the player's per-opponent averages spread out than luck alone would produce.\n"
        "- With few games, or when contexts barely matter, B is small and the estimate stays near the baseline. "
        "With many games it trusts the split. Both the raw and the shrunk number are shown."
    )

with st.container(border=True):
    st.markdown("#### 5 · League-wide variable effects")
    st.markdown(
        "The *Explore variables* page fits one regression across every player: "
        "stat ~ variable + controls (minutes, home/away, rest, opponent defense, defender height) + a separate "
        "intercept for each player. That per-player intercept (a *fixed effect*) means each player is compared only "
        "with their own games, so the answer is \"how much does this move the stat for a typical player\". Standard "
        "errors are clustered by player, because one player's games aren't independent of each other."
    )

with st.container(border=True):
    st.markdown("#### 6 · The projection model")
    st.markdown(
        "- **Minutes first.** Playing time drives counting stats more than anything else, so the model projects "
        "minutes, then production given minutes. You can override projected minutes on the results page.\n"
        "- **A full distribution, not just an average.** Points are modeled with a negative binomial, which allows "
        "more spread than a Poisson. From the distribution come the median, the 90/95/99% **prediction intervals** "
        "(where one game will land) and P(over)/P(under)/P(push) for any line.\n"
        "- **Combos** (e.g. 27.5+ points *and* 5.5+ assists) aren't the product of the two probabilities, because "
        "the stats are correlated. A Gaussian copula fitted on the player's own history ties the two distributions "
        "together, and 10,000 simulated games give the joint probability.\n"
        "- **Benchmark.** The model has to beat a simple baseline: a recency-weighted average adjusted for opponent "
        "and home/away."
    )

# --------------------------------------------------------------------- backtest
st.markdown("#### 7 · Backtest: does it actually work?")
st.markdown("<div class='nbl-note'>Trained on seasons up to 2023-24, tested on 2024-25 and 2025-26. The split is by "
            "time, never random, just like real forecasting.</div>", unsafe_allow_html=True)


def _render_metrics() -> bool:
    raw = backend.read_doc("model_metrics.json")
    if not raw:
        return False
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if isinstance(m, dict):
        flat = pd.json_normalize(m, sep=" · ").T.reset_index()
        flat.columns = ["Metric", "Value"]
        flat["Value"] = flat["Value"].map(lambda v: f"{v:g}" if isinstance(v, (int, float)) else str(v))
        st.dataframe(flat, hide_index=True, width="stretch")
        return True
    return False


def _render_calibration() -> bool:
    raw = backend.doc_path("model_report_calibration.csv")
    if not raw.exists():
        return False
    cal = pd.read_csv(raw)
    stats = list(cal["stat"].unique()) if "stat" in cal else [None]
    pick = st.segmented_control("Calibration for", stats, default=stats[0]) if len(stats) > 1 else stats[0]
    sub = cal[cal["stat"] == pick] if pick is not None else cal
    st.plotly_chart(charts.calibration_chart(sub["bin_pred"], sub["bin_actual"],
                                             sub["n"].tolist() if "n" in sub else None),
                    width="stretch", config={"displaylogo": False})
    st.caption("Each dot is a bucket of predictions. On the diagonal = when the model says 60%, it happens 60% of the time.")
    return True


def _render_report() -> bool:
    text = backend.read_doc("model_report.md")
    if not text:
        return False
    with st.expander("Full model report (docs/model_report.md)", expanded=False):
        parts = re.split(r"(!\[[^\]]*\]\([^)]+\))", text)
        for part in parts:
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", part)
            if m:
                path = backend.doc_path(m.group(2))
                if path.exists():
                    st.image(str(path), caption=m.group(1) or None)
            elif part.strip():
                st.markdown(part)
    return True


shown_metrics = _render_metrics()
shown_cal = _render_calibration()
if not shown_cal:
    imgs = sorted(backend.doc_path("img").glob("calibration*.png")) if backend.doc_path("img").exists() else []
    if imgs:
        cols = st.columns(min(len(imgs), 2))
        for i, img in enumerate(imgs):
            cols[i % len(cols)].image(str(img), caption=img.stem.replace("_", " "))
        shown_cal = True
shown_report = _render_report()
if not (shown_metrics or shown_cal or shown_report):
    st.info("The backtest report hasn't been generated yet. Run the model evaluation to create "
            "docs/model_report.md and its calibration plots.", icon=":material/hourglass_empty:")

st.markdown("#### 8 · Honest limits")
st.markdown(
    "- No box score says who guarded whom. Defender questions use the best available source and say which one.\n"
    "- Small samples are everywhere: one player vs one team is often 20–40 games. The sample size is always shown, "
    "and small splits are shrunk.\n"
    "- Projections are statistical estimates for learning and exploration. **This is not betting advice.**"
)
