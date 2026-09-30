"""Ask page: question -> interpreted query (editable) -> results."""

from __future__ import annotations

import streamlit as st

from ui import backend, components as C, form, results, state

EXAMPLES: list[tuple[str, str]] = [
    ("Curry's assists vs the Spurs", "what is Steph Curry's apg against the spurs"),
    ("Celtics vs Lakers, last 5 years", "what is the avg ppg for the Boston Celtics in the last 5 years playing against the Lakers"),
    ("Curry vs a 6'7\" defender", "how might Steph Curry perform against a 6'7 defender"),
    ("Curry vs Philly: 27.5 pts & 5.5 ast?", "Curry is playing against Philly in San Francisco, his defender is LeBron. What is the "
     "expected points and assists for this game? What are the chances he exceeds 27.5 points and 5.5 assists?"),
    ("Tatum's rebounds by weekday", "how does day of week affect Jayson Tatum's rebounds"),
    ("Back-to-backs for all guards", "how do back to backs affect scoring for all guards"),
]



def _on_pick(key: str) -> None:
    """Swap the entity the parser guessed for the one the user picked."""
    new = st.session_state[key]
    old = st.session_state.get(key + "_current")
    if old is not None and new != old:
        state.swap(old, new)
    st.session_state[key + "_current"] = new


def candidate_picker(u) -> None:
    key = f"cand_{u.field}_{u.text}"
    st.session_state.setdefault(key + "_current", u.chosen_id)
    ids = [c.id for c in u.candidates]
    if u.chosen_id not in ids:
        ids = [u.chosen_id, *ids]
    names = {c.id: f"{c.name} · {c.detail}" for c in u.candidates}
    st.selectbox(f"Which one did you mean by “{u.text}”?", ids, index=ids.index(st.session_state[key + "_current"]),
                 format_func=lambda i: names.get(i, str(i)), key=key, on_change=_on_pick, args=(key,))


def conflict_box(q) -> None:
    """Impossible combinations, caught before running, each with one-click fixes."""
    for i, (message, fixes) in enumerate(backend.conflicts(q.model_dump_json())):
        st.warning(f"**This can't happen as asked.** {message}", icon=":material/warning:")
        cols = st.columns(max(len(fixes), 1) + 2)
        for j, (label, fixed) in enumerate(fixes):
            cols[j].button(label, key=f"fix_{i}_{j}", on_click=state.apply_fix, args=(fixed,), width="stretch",
                           icon=":material/build:")


state.init()
C.header("views/methodology.py")

# ------------------------------------------------------------------- question box
st.text_area("Ask a question", key="prompt", height=96,
             placeholder="e.g. How does Nikola Jokic rebound against the Lakers on the road since 2021?")
b1, b2, b3 = st.columns([1.2, 1, 5])
b1.button("Interpret", type="primary", icon=":material/auto_awesome:", on_click=state.parse_current_prompt,
          width="stretch")
b2.button("Clear", icon=":material/restart_alt:", on_click=state.reset, width="stretch")
if backend.parser_module() is not None and not backend.has_llm_key():
    b3.info("No ANTHROPIC_API_KEY set: questions are read by the rule-based parser. Every number is still "
            "computed by tested Python, never by the AI.", icon=":material/info:")

st.markdown("<div class='nbl-kicker' style='margin-top:.6rem'>Try one</div>", unsafe_allow_html=True)
cols = st.columns(3)
for i, (label, prompt) in enumerate(EXAMPLES):
    cols[i % 3].button(label, key=f"ex_{i}", on_click=state.use_example, args=(prompt,), width="stretch",
                       help=prompt)

# ------------------------------------------------------------------ manual form
with st.expander("Advanced / manual: set every filter by hand", icon=":material/tune:",
                 expanded=st.session_state.get("sq") is None and st.session_state.get("parsed") is not None):
    st.caption("This form and the question box edit the same query. Type a question, then fine-tune it here.")
    form.manual_form()

# --------------------------------------------------------- interpreted query
if st.session_state.get("sq_error"):
    st.error(st.session_state["sq_error"], icon=":material/error:")

q = state.query()
parsed = st.session_state.get("parsed")
if q is not None or parsed is not None:
    with st.container(border=True):
        st.markdown("<div class='nbl-kicker'>Here's how I read your question · click a chip to edit it</div>",
                    unsafe_allow_html=True)
        if q is not None:
            form.chips(q)
        if parsed is not None:
            if st.session_state.get("parse_note") and backend.has_llm_key():  # rate-limit fallbacks only
                st.info(st.session_state["parse_note"], icon=":material/info:")
            if parsed.assumptions:
                st.markdown("**Assumptions**")
                st.markdown("\n".join(f"- {a.message}" for a in parsed.assumptions))
            found = backend.conflicts(q.model_dump_json()) if q is not None else []
            for u in parsed.unresolved:
                if found and u.kind == "conflict" and u.field in ("defender", "opponent"):
                    continue  # the conflict box below explains it and offers fixes
                icon = {"conflict": ":material/warning:", "ambiguous": ":material/help:"}.get(u.kind, ":material/info:")
                st.warning(f"**{u.text}**: {u.message}", icon=icon)
                if u.candidates and u.chosen_id is not None:
                    candidate_picker(u)
            st.caption(f"Parsed by the {'Claude' if getattr(parsed, 'source', '') == 'llm' else 'rule-based'} parser.")
        if q is not None:
            conflict_box(q)
            stale = st.session_state.get("ran") not in (None, q.model_dump_json())
            st.button("Run" if not stale else "Run with the edited filters", type="primary",
                      icon=":material/play_arrow:", on_click=state.run)

# ------------------------------------------------------------------------ results
ran = st.session_state.get("ran")
if ran:
    st.divider()
    try:
        results.render(ran)
    except Exception as exc:  # keep the page alive and say what went wrong
        st.error(f"Something went wrong running this query: {exc}", icon=":material/bug_report:")
        with st.expander("Details"):
            st.exception(exc)
elif q is None and parsed is None:
    st.markdown("<div class='nbl-note' style='margin-top:2rem'>Works for any player or team since 1996-97: "
                "current stars, bench players, rookies, retired legends, and relocated franchises. "
                "Every number is computed by tested Python code. The AI only turns your words into a structured query.</div>",
                unsafe_allow_html=True)
