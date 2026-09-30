"""Page chrome: CSS, header, headline cards, badges and callouts."""

from __future__ import annotations

import html

import streamlit as st

from nbalab.viz import theme as T

CSS = f"""
<style>
:root {{
  --surface: {T.SURFACE}; --ink: {T.INK}; --ink2: {T.INK_2}; --muted: {T.MUTED};
  --ring: rgba(255,255,255,0.10); --up: {T.UP}; --down: {T.DOWN}; --warn: {T.WARN};
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.nbl-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
  border-bottom:1px solid var(--ring); padding-bottom:.9rem; margin-bottom:1.2rem; }}
.nbl-title {{ font-size:2.1rem; font-weight:700; color:var(--ink); line-height:1.1; margin:0; }}
.nbl-title span {{ color:#F58426; }}
.nbl-sub {{ color:var(--ink2); font-size:1rem; margin-top:.35rem; }}
.nbl-accent {{ height:3px; border-radius:2px; margin:-.2rem 0 1rem 0; }}
.nbl-cards {{ display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:.75rem; margin:.4rem 0 1rem; }}
@media (max-width: 900px) {{ .nbl-cards {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} }}
.nbl-card {{ background:var(--surface); border:1px solid var(--ring); border-radius:12px; padding:.9rem 1rem; }}
.nbl-card .lbl {{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }}
.nbl-card .val {{ color:var(--ink); font-size:2rem; font-weight:650; line-height:1.15; margin-top:.2rem; }}
.nbl-card .sub {{ color:var(--ink2); font-size:.82rem; margin-top:.25rem; }}
.nbl-card.hero {{ border-color:rgba(245,132,38,.55); }}
.nbl-up {{ color:var(--up) !important; }} .nbl-down {{ color:var(--down) !important; }}
.nbl-badge {{ display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .6rem; border-radius:999px;
  font-weight:650; font-size:.95rem; margin-top:.35rem; }}
.nbl-badge.high {{ background:rgba(12,163,12,.16); color:#4cd94c; }}
.nbl-badge.medium {{ background:rgba(250,178,25,.16); color:var(--warn); }}
.nbl-badge.low {{ background:rgba(208,59,59,.18); color:#ff7a7a; }}
.nbl-section {{ margin-top:1.6rem; }}
.nbl-section h3 {{ margin-bottom:.1rem; }}
.nbl-kicker {{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }}
.nbl-note {{ color:var(--ink2); font-size:.9rem; }}
.nbl-big {{ font-size:3.2rem; font-weight:700; line-height:1; color:var(--ink); }}
.nbl-big-lbl {{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }}
div[data-testid="stPopover"] button {{ border-radius:999px; }}
div[data-testid="stTextArea"] textarea {{ font-size:1.08rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(methodology_page) -> None:
    left, right = st.columns([5, 1.3], vertical_alignment="bottom")
    with left:
        st.markdown(
            "<div class='nbl-title'>NBA Stat <span>Lab</span></div>"
            "<div class='nbl-sub'>Ask any question about any NBA player or team since 1996. "
            "Get the split, whether it's real, and what to expect next game.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.page_link(methodology_page, label="How it works", icon=":material/menu_book:")
    st.markdown("<div style='border-bottom:1px solid rgba(255,255,255,.10);margin:.4rem 0 1.2rem'></div>",
                unsafe_allow_html=True)


def accent_bar(color: str) -> None:
    st.markdown(f"<div class='nbl-accent' style='background:{color}'></div>", unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", cls: str = "", value_cls: str = "") -> str:
    return (f"<div class='nbl-card {cls}'><div class='lbl'>{html.escape(label)}</div>"
            f"<div class='val {value_cls}'>{value}</div><div class='sub'>{sub}</div></div>")


def cards(items: list[str]) -> None:
    st.markdown("<div class='nbl-cards'>" + "".join(items) + "</div>", unsafe_allow_html=True)


def badge(level: str) -> str:
    icon = {"high": "●●●", "medium": "●●○", "low": "●○○"}.get(level, "")
    return f"<span class='nbl-badge {level}'>{icon} {level.capitalize()}</span>"


def section(title: str, kicker: str = "", note: str = "") -> None:
    parts = ["<div class='nbl-section'>"]
    if kicker:
        parts.append(f"<div class='nbl-kicker'>{html.escape(kicker)}</div>")
    parts.append(f"<h3>{html.escape(title)}</h3>")
    if note:
        parts.append(f"<div class='nbl-note'>{note}</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def big_number(label: str, value: str, color_cls: str = "") -> str:
    return (f"<div><div class='nbl-big-lbl'>{html.escape(label)}</div>"
            f"<div class='nbl-big {color_cls}'>{value}</div></div>")
