"""Live comparison of the Claude parser and the rule-based parser on the same prompts.

    .venv/bin/python scripts/compare_parsers.py [--out docs/parser_comparison.md]

Makes real API calls (one per prompt) with the key from .env / secrets / env.
The Claude parser runs with backend="llm", so a Claude failure is reported as
a failure instead of silently falling back to the rules. For each prompt it
records whether the two structured queries agree, the differences if not, the
latency, and the prompt-cache tokens from the API response (the first call
writes the cache, later calls should read it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbalab.nlp import llm  # noqa: E402
from nbalab.nlp.parser import default_parser  # noqa: E402

PROMPTS: list[str] = [
    "What is Stephen Curry's ppg against LeBron as a defender",
    "What is Steph Curry's apg against the Spurs",
    "What is the avg ppg for the Boston Celtics in the last 5 years playing against the Lakers",
    "How might Curry perform against a 6'7 defender",
    "Curry is playing against Philly in San Francisco, his defender is LeBron. Expected points and assists? "
    "Chances he exceeds 27.5 points and 5.5 assists?",
    "How does day of week affect Jayson Tatum's rebounds",
    "How do back-to-backs affect scoring for all guards",
    "Knicks points at home vs away on Sundays since 2020",
    "Nikola Jokic points at OKC, line 28.5",
    "Victor Wembanyama blocks vs the Warriors, line 3.5",
    "Kobe Bryant ppg vs the Celtics in the playoffs",
    "Seattle SuperSonics points per game in 2005",
    "Anthony Edwards threes on back-to-backs, line 3.5",
    "Lakers team threes at home, line 13",
    # phrasings not copied from the system prompt's examples
    "How many rebounds does Giannis average against the Celtics since 2021?",
    "Does Luka Doncic score more on the road?",
    "Tyrese Haliburton assists in his last 15 games",
    "Will Shai score 30+ points against Denver next game?",
    "Warriors points allowed at home this season",
    "How does Joel Embiid shoot from three when he has 2+ days of rest",
    # period comparisons
    "before vs after the all star break",
    "Celtics after the all star break",
    "how do teams change after the all star break",
    "Tatum's scoring leading up to the playoffs",
    "compare team ppg from months 10,11,12,1 to months 2,3,4,5,6",
    "who improves most after the break",
    "Lakers net rating last 20 games before playoffs",
]

# Haiku 4.5 prices per million tokens (input, output); cache writes 1.25x input, cache reads 0.1x input.
PRICE_IN, PRICE_OUT = 1.00, 5.00


def canon(q: Any) -> dict[str, Any] | None:
    if q is None:
        return None
    d = q.model_dump()
    d["filters"] = sorted(d["filters"], key=lambda f: json.dumps(f, sort_keys=True))
    return d


def diff(a: dict | None, b: dict | None) -> list[str]:
    if a is None or b is None:
        return [f"query: claude={'none' if a is None else 'ok'} rules={'none' if b is None else 'ok'}"]
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(f"{k}: claude={json.dumps(a.get(k), default=str)} | rules={json.dumps(b.get(k), default=str)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="optional markdown report path")
    args = ap.parse_args()

    usages: list[Any] = []
    real_create = llm.create

    def recording_create(prompt, today, client, settings):  # capture usage without changing behavior
        response = real_create(prompt, today, client, settings)
        usages.append(response.usage)
        return response

    llm.create = recording_create
    parser = default_parser()
    today = date.today()
    print(f"model: {parser.config.llm.model}  prompts: {len(PROMPTS)}  today: {today}\n")
    rows = []
    for prompt in PROMPTS:
        t0 = time.time()
        try:
            claude = parser.parse(prompt, backend="llm", today=today)
            err = None
        except Exception as exc:  # report, keep going
            claude, err = None, f"{type(exc).__name__}: {exc}"
        latency = time.time() - t0
        rules = parser.parse(prompt, backend="rules", today=today)
        u = usages[-1] if usages and err is None else None
        a, b = canon(claude.query) if claude else None, canon(rules.query)
        d = [err] if err else diff(a, b)
        rows.append({"prompt": prompt, "agree": not d, "diffs": d, "latency": latency, "usage": u,
                     "source": getattr(claude, "source", None)})
        mark = "AGREE " if not d else "DIFFER"
        cache = f"cache write {u.cache_creation_input_tokens} / read {u.cache_read_input_tokens}" if u else "no usage"
        print(f"{mark} {latency:4.1f}s  {cache:28s}  {prompt[:70]}")
        for line in d:
            print(f"         - {line}")

    ok = [r for r in rows if r["usage"] is not None]
    tin = sum(r["usage"].input_tokens for r in ok)
    tout = sum(r["usage"].output_tokens for r in ok)
    cw = sum(r["usage"].cache_creation_input_tokens or 0 for r in ok)
    cr = sum(r["usage"].cache_read_input_tokens or 0 for r in ok)
    cost = (tin * PRICE_IN + cw * PRICE_IN * 1.25 + cr * PRICE_IN * 0.1 + tout * PRICE_OUT) / 1e6
    no_cache = ((tin + cw + cr) * PRICE_IN + tout * PRICE_OUT) / 1e6
    agree = sum(r["agree"] for r in rows)
    summary = (f"\n{agree}/{len(rows)} prompts agree. Claude calls: {len(ok)} ok, {len(rows) - len(ok)} failed.\n"
               f"Tokens: {tin} uncached in, {cw} cache-write, {cr} cache-read, {tout} out.\n"
               f"Cost ~${cost:.4f} (would be ~${no_cache:.4f} without caching). "
               f"Median latency {sorted(r['latency'] for r in rows)[len(rows) // 2]:.1f}s.")
    print(summary)
    if args.out:
        lines = [f"# Parser comparison ({today}, {parser.config.llm.model})", "", summary.strip(), "",
                 "| Prompt | Agree | Latency | Cache read | Differences (claude vs rules) |", "|---|---|---|---|---|"]
        for r in rows:
            cr_ = r["usage"].cache_read_input_tokens if r["usage"] else "-"
            lines.append(f"| {r['prompt']} | {'yes' if r['agree'] else 'no'} | {r['latency']:.1f}s | {cr_} | "
                         + "<br>".join(x.replace("|", "\\|") for x in r["diffs"]) + " |")
        args.out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
