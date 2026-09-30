# NBA Stat Lab

![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black)
![statsmodels](https://img.shields.io/badge/statsmodels-4051B5)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude%20API-D97757?logo=anthropic&logoColor=white)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

**Ask a plain-English question about any NBA player or team since 1996. Get the stat split, a significance test, charts, and a calibrated over/under projection. The LLM does no math.**

🔗 **Live demo:** _coming soon_ &nbsp;·&nbsp; 📊 **Model report:** [`docs/model_report.md`](docs/model_report.md) _(pending)_

![Demo: asking a question and getting an answer](docs/img/demo.gif)
<!-- TODO: record docs/img/demo.gif of the app answering "What's Steph Curry's apg against the Spurs?" -->

---

## What it does

| You ask | You get |
|---|---|
| *"What's Steph Curry's apg against the Spurs?"* | Split vs. career baseline, sample size, Welch p-value, bootstrap CI, and a shrunk estimate for small samples |
| *"How does Curry score against 6'6"–6'8" defenders?"* | The same split through on-court defender data, with each proxy labeled by source |
| *"Will LeBron go over 25.5 points tonight?"* | P(over/under), plus 90 / 95 / 99% prediction intervals |

<!-- TODO: add screenshots once the Streamlit app is built -->
| Split vs. baseline | Defender split | Projection |
|---|---|---|
| ![split](docs/img/screenshot_split.png) | ![defender](docs/img/screenshot_defender.png) | ![projection](docs/img/screenshot_projection.png) |

## How it works

```mermaid
flowchart LR
    A["💬 Plain-English prompt"] --> B["LLM parser<br/>(Claude)"]
    B -->|JSON only| C["StatQuery<br/>(pydantic, validated)"]
    C --> D["Split engine<br/>DuckDB + pandas"]
    C --> E["Projection model<br/>NB / quantile + copula"]
    D --> F["Inference<br/>Welch · bootstrap · EB shrinkage"]
    F --> G["📈 Plotly charts + Streamlit UI"]
    E --> G
```

**Core rule:** the LLM only turns the question into a typed `StatQuery`. Name matching runs in our own code. Every number comes from deterministic, unit-tested Python.

## Data science highlights

- **Leakage-free rolling features.** Rolling averages are shifted one game within each player, so a row never sees its own game. Tests check that no row sees its own game or another player's games. Team ratings are cumulative *pre-game* values, and rest and calendar features are known before tip-off.
- **Empirical Bayes shrinkage.** Split means are pulled toward the baseline by `n / (n + σ²/τ²)`. τ² is estimated from how much real between-split variation exists, so a 12-game "vs. Spurs" split isn't taken at face value.
- **Valid significance tests.** Welch's t-test and a bootstrap CI compare the split against the *rest* of the baseline, not against a baseline that contains the split.
- **Distribution-aware projections.** Negative binomial models handle over-dispersed counts (assists, rebounds, 3PM), and quantile models give the intervals. *(in progress)*
- **Joint probabilities via Gaussian copula.** 10k Monte Carlo draws from the fitted marginals, linked by the player's residual correlations (e.g. P(25+ pts **and** 8+ ast)). *(in progress)*
- **Time-based backtest.** Train on earlier seasons, test on later ones, never shuffled. *(in progress)*

  | Metric | Result |
  |---|---|
  | Points MAE (model vs. rolling-average baseline) | `TBD` vs `TBD` |
  | Brier score, over/under (model vs. baseline) | `TBD` vs `TBD` |
  | Interval coverage (nominal 90 / 95 / 99%) | `TBD` / `TBD` / `TBD` |

- **Serious data cleaning.** 1.67M player rows, 18.7M play-by-play events, and 80 seasons of data. The pipeline normalizes 10 messy `gameType` labels through gameId encoding, recovers 100% of 105k missing team ids, and handles franchise moves by `teamId` (two different "Hornets"!). See [`docs/data_profile.md`](docs/data_profile.md).

## Limitations

- **Box scores don't record who guarded whom.** Defender splits use matchup data where it exists. Otherwise they fall back to on-court overlap or a same-position starter, and the UI always labels which source was used.
- **Small samples are the norm.** A player vs. one team is often 20–40 games. Sample size is always shown, and estimates are shrunk toward the baseline.
- **History ≠ destiny.** Projections can't account for injuries, lineup news, or anything after the data cutoff (June 2026; play-by-play ends before the 2026 playoffs).
- **Not betting advice.** Over/under lines are a way to express uncertainty. This is a portfolio project, not a wagering tool.

## Run it locally

```bash
uv sync                                   # Python 3.11 deps
.venv/bin/python -m nbalab.data.build     # raw CSVs -> data/processed/*.parquet
.venv/bin/python -m pytest                # test suite
```

Raw data (not included) goes in `data/raw/`. `ANTHROPIC_API_KEY` goes in `.env`.
