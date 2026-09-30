# NBA Stat Lab

A data science web app where a user types a plain-English question about NBA players or teams and gets back:

1. The historical stat split they asked for.
2. How far that split differs from the player's or team's baseline, and whether the difference is statistically meaningful.
3. Charts.
4. A forward-looking projection with the probability of going over or under a betting-style line, plus 90%, 95%, and 99% intervals.

## Core principle

**The LLM only translates the user's question into a structured query (JSON).** Every number shown to the user is computed by deterministic, tested Python code. The LLM never computes or invents statistics.

## Data

- All raw files live in `data/raw/`. **Never modify raw files.**
- Processed parquet files go in `data/processed/`.

| File | Contents |
|---|---|
| `PlayerStatistics.csv` | Player box scores, 1946 to June 2026 (~1.67M rows, 390MB). Key columns: `personId`, `gameId`, `gameDateTimeEst`, `playerteamId`, `opponentteamId`, `home`, `win`, `numMinutes`, `points`, `assists`, `reboundsTotal`, `steals`, `blocks`, `threePointersMade`, `turnovers`, `startingPosition`, `gameType`. |
| `PlayerStatisticsExtended.csv` | Advanced player box stats. **Inspect columns before using.** |
| `TeamStatistics.csv` | Team box scores. |
| `TeamStatisticsExtended.csv` | Offensive/defensive rating, pace, possessions, four factors. |
| `Games.csv` | One row per game: home/away team ids, scores, arena, officials. Arena columns are mostly empty, so derive venue from the home team. |
| `Players.csv` | Bio data: `heightInches` and `bodyWeightLbs` (~76% populated), guard/forward/center flags, `birthDate`, draft info. |
| `TeamHistories.csv` | Maps `teamId` to city, name, and abbreviation by season range. Franchises relocate and rename, so **always join on `teamId`, never team name.** |
| `LeagueSchedule24_25.csv`, `LeagueSchedule25_26.csv` | Schedules with `gameDay` (day of week) and `weekNumber`. |
| `PlayByPlay.parquet` | Play-by-play events. **Inspect schema before using.** |

### Cleaning rules

- `gameType` has messy values (e.g. "Preseason" vs "Pre Season", several NBA Cup spellings). Normalize them.
- Exclude preseason and All-Star games from all analysis by default.

## Known data limits (must be surfaced honestly in the UI)

- **No defender data in box scores.** Box scores do not record who guarded whom. "Player X vs defender Y" questions require matchup data (`data/raw/matchups` if present) or must fall back to a clearly labeled proxy.
- **Small samples are common.** A player vs one team may be only 20 to 40 games. Always show sample size, and shrink small-sample splits toward the baseline.

## Stack

Python 3.11, pandas, pyarrow, duckdb (queries), statsmodels, scikit-learn, lightgbm (modeling), scipy (distributions), plotly (charts), streamlit (UI), anthropic SDK (prompt parser), pytest (tests).

## Layout (src/ layout)

```
src/nbalab/data/      loading, cleaning, feature building
src/nbalab/query/     structured query schema + split engine
src/nbalab/nlp/       prompt -> query parser
src/nbalab/models/    projection models, intervals, probabilities
src/nbalab/viz/       chart builders
app/                  streamlit app
tests/
notebooks/            exploration and model evaluation
```

## Conventions

- Type hints everywhere.
- Small functions.
- Docstrings that explain the statistics in plain English.
- A test for every stat function.
- No hardcoded player names in logic.
- Secrets live in `.env`, which is listed in `.gitignore`.
- `data/` is in `.gitignore`, except for a small sample file used in tests.

## Commands

```bash
uv sync                                    # install deps into .venv (Python 3.11)
.venv/bin/python -m nbalab.data.build      # raw -> data/processed/*.parquet (--start-season 1996)
.venv/bin/python -m nbalab.data.make_sample  # regenerate data/sample/ test fixture
.venv/bin/python -m pytest                 # tests (processed-file tests skip if not built)
.venv/bin/python -m nbalab.models.train    # select on 2023-24, refit <=2023-24 -> models/<version>/ + models/LATEST (~15 min)
.venv/bin/python -m nbalab.models.calibrate # fit minutes widening + over/under calibration on 2024-25 only
.venv/bin/python -m nbalab.models.evaluate # backtest 2024-25/2025-26 -> docs/model_report.md, docs/img/, docs/model_metrics.json
.venv/bin/python -m nbalab.deploy.slim     # processed -> data/deploy/ (manifest columns, zstd)
.venv/bin/python -m nbalab.deploy.publish --repo <user>/nbalab-data  # upload to HF dataset
```

- Deployment (Streamlit Community Cloud + HF dataset) is in `docs/deploy.md`. When the app reads a new processed column, add it to `src/nbalab/deploy/manifest.py`.
- `.venv` is a symlink to `~/.venvs/nbalab`. The project sits in iCloud-synced `~/Desktop`, which marks `.pth` files hidden, and Python then skips them, breaking the editable install.
- Data findings and quirks are in `docs/data_profile.md`. Read it before touching the pipeline.
- LightGBM on macOS needs OpenMP: `brew install libomp` (Linux deploys get `libgomp1` from `packages.txt`).
- Projection API: `nbalab.models.{load_bundle, project}`. Unchosen quantile models go to gitignored `models_archive/` (backtest only). Season split and stat lists live in `src/nbalab/models/config.py`; the evaluate script refuses to run if training overlaps the test seasons.
