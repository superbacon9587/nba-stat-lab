# Deploying NBA Stat Lab (free public link)

Result: the app runs on **Streamlit Community Cloud** at `https://<your-name>.streamlit.app`,
and its data lives in a **Hugging Face dataset** that the app downloads on first start.
Code is on GitHub. Raw and processed data never touch git.

```
GitHub repo (code ~4 MB + models/ ~31 MB) ──▶ Streamlit Community Cloud ──first start──▶ HF dataset (parquet, ~77 MB)
                                   │  secrets: ANTHROPIC_API_KEY, NBALAB_DATA_REPO, HF_TOKEN
                                   └─ LLM parser (rate-limited) or rule-based parser (fallback)
```

## 1. Which host, and why

Measured on 2026-09-29:

| | Size on disk | In memory (pandas) |
|---|---:|---:|
| `data/processed/` (all 8 files) | 140 MB | |
| `data/deploy/` (what ships, 6 tables) | **77 MB** | peak ~1.4 GB, ~0.8 GB without `on_court` |

| | Streamlit Community Cloud | Hugging Face Spaces |
|---|---|---|
| Cost for a Streamlit app | **Free** | Docker/Gradio Spaces now need a **paid PRO plan** to create. Only static Spaces and ZeroGPU Gradio Spaces are free. |
| Memory | 690 MB to 2.7 GB | 16 GB (CPU Basic) |
| Large files | App repo must stay small. Git has a 100 MB per-file limit. | Git LFS/Xet built in |
| Secrets | `st.secrets` (TOML box in settings) | Environment variables |
| Sleeps after | 12 h without traffic (any visitor can wake it) | a period without traffic on free hardware |
| Deploy from | GitHub repo, redeploys on push | Its own git repo |

**Recommendation: Streamlit Community Cloud for the app, a Hugging Face *dataset* for the data.**
Spaces is the better host for large files and has far more RAM, but it's no longer free
for a Streamlit app. Hugging Face *datasets* are still free, so we use HF for exactly
the part it's best at (hosting the parquet files) and Streamlit Cloud for the app.

The tradeoff: 2.7 GB of RAM is enough (peak ~1.4 GB) but not roomy. If the app
restarts with "resource limits" errors, set `NBALAB_SKIP_ON_COURT = "1"` in secrets.
That drops the largest optional table, and on-court filters fall back to "both
played in the game" (the app labels this). If you later get HF PRO, the same code runs on a
Docker Space unchanged (secrets are read from env vars too).

Why not commit the 77 MB to GitHub? It's under the 100 MB file limit, but every data
refresh adds another ~77 MB to git history forever, and GitHub warns above 50 MB.
The HF dataset keeps the code repo at ~2 MB and lets you refresh data without a code commit.

## 2. What was added for deployment

| File | Purpose |
|---|---|
| `src/nbalab/deploy/manifest.py` | The tables and columns that ship. Edit here when the app reads a new column. |
| `src/nbalab/deploy/slim.py` | `data/processed` → `data/deploy`: manifest columns only, seasons ≥ `start_season`, unreliable on-court rows dropped, zstd level 19. |
| `src/nbalab/deploy/publish.py` | Uploads `data/deploy` to a (private by default) HF dataset. |
| `src/nbalab/deploy/fetch.py` | On app start: use `data/deploy/`, else download from HF, else use `data/processed/`. |
| `src/nbalab/deploy/secrets.py` | `st.secrets` → env var → `.env`. Placeholders count as missing. |
| `src/nbalab/deploy/rate_limit.py` | 20 LLM-parsed questions per session per hour, 200 per app per hour, then the rule-based parser. |
| `requirements.txt` | Pinned runtime deps, exported from `uv.lock`. |
| `packages.txt` | `libgomp1` (system library LightGBM needs on Linux). |
| `.streamlit/config.toml` | Dark theme, viewer-only toolbar, no stack traces shown to visitors. |
| `.streamlit/secrets.toml.example` | Template for local secrets and the Cloud secrets box. |
| `scripts/deploy_check_app.py` | Throwaway page that tests the hosting plumbing (data, secrets, routing). |
| `tests/test_deploy.py` | Tests for all of the above, including a check that the manifest ships every column the query code uses. |

The column slimming, per table:

| Table | Rows | Columns | Before → after |
|---|---:|---:|---:|
| `player_games` | 779,311 | 69 → 67 | 33.3 → 30.8 MB |
| `team_games` | 76,042 | 81 → 78 | 7.3 → 7.0 MB |
| `on_court` | 7.25M → 6.92M | 18 → 14 | 92.6 → 38.9 MB |
| `players`, `teams`, `matchups` | small | trimmed | < 0.3 MB |
| `on_court_players`, `on_court_quality` | | not shipped (pipeline diagnostics) | 13 MB → 0 |

`player_games` and `team_games` barely shrink: most of their columns are the rolling-form
and opponent-rating features that the projection model needs.

### How the rate limit protects your bill

- **Per session (20/hour):** stops one visitor from hammering the AI parser. Opening a new tab starts a new session, so this limit alone is easy to get around.
- **Per app (200/hour):** shared by every visitor. This is the real cap on spending.
- Cached repeat questions don't count. When either limit is hit, the question goes to the rule-based parser and the visitor sees why.
- **Hard backstop (do this, step 5):** a monthly spend limit in the Anthropic Console. The code limits reset when the app restarts; the Console limit does not.

Change the numbers without a code change by setting `NBALAB_LLM_PER_SESSION_PER_HOUR` / `NBALAB_LLM_PER_APP_PER_HOUR` in secrets.

## 3. Step by step

You need free accounts on GitHub, Hugging Face, and Streamlit Community Cloud
(sign in with GitHub), plus an Anthropic API key (optional; the app works without one).

### Step 1: build the slim data (local, ~10 s)

```bash
cd ~/Desktop/NBA_BOT
.venv/bin/python -m nbalab.data.build          # only if data/processed is stale
.venv/bin/python -m nbalab.deploy.slim         # -> data/deploy/ (~77 MB) + manifest.json
.venv/bin/python -m pytest tests/test_deploy.py
```

### Step 2: upload the data to a Hugging Face dataset

1. Create a token at <https://huggingface.co/settings/tokens> → **Write** access (used only on your laptop).
2. Log in and publish (the repo is created **private**):
   ```bash
   .venv/bin/hf auth login                     # paste the write token
   .venv/bin/python -m nbalab.deploy.publish --repo <hf-username>/nbalab-data
   ```
3. Create a second token for the app: **Fine-grained** → *Read access to contents of selected repos* → pick `<hf-username>/nbalab-data`. This one goes into Streamlit secrets. It can only read this dataset.

> Why private? The raw data comes from a third-party dataset with its own license. Keep
> the derived copy private unless you've checked that license allows redistribution, and
> then you can re-run publish with `--public` and drop `HF_TOKEN`.

### Step 3: put the code on GitHub

The project is a git repo on branch `main` with a first commit. `.gitignore` excludes
`data/` (except `data/sample/`), `models_archive/`, `archive/`, `archive.zip`, `.env`,
`.venv`, and `.streamlit/secrets.toml`. The trained projection models in `models/`
**are** committed: the app loads them at startup (see "Projection models" below).

Before pushing, check nothing private is tracked:

```bash
git ls-files | grep -E "data/raw|data/processed|data/deploy|models_archive|archive|\.env$|secrets\.toml$" \
  && echo "STOP: something private is tracked" || echo "OK: nothing private tracked"
```

Then create an **empty** repo at <https://github.com/new> (no README) and:

```bash
git remote add origin https://github.com/<github-username>/nba-stat-lab.git
git push -u origin main
```

(If you have the GitHub CLI: `gh repo create nba-stat-lab --public --source . --push`.)

### Step 4: deploy on Streamlit Community Cloud

1. <https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**.
2. Repository `<github-username>/nba-stat-lab`, branch `main`, main file path:
   **`app/streamlit_app.py`** (the real app). `scripts/deploy_check_app.py` is a lighter page
   that tests only the hosting plumbing, useful if the real app fails to start.
3. **App URL**: pick a subdomain, e.g. `nba-stat-lab` → `nba-stat-lab.streamlit.app`.
4. **Advanced settings**:
   - **Python version: 3.11.** This is required, because the project pins `<3.12`. It can only be changed by redeploying.
   - **Secrets**: paste (see `.streamlit/secrets.toml.example`):
     ```toml
     ANTHROPIC_API_KEY = "sk-ant-api03-..."
     NBALAB_DATA_REPO = "<hf-username>/nbalab-data"
     HF_TOKEN = "hf_...the READ-only token from step 2.3..."
     ```
5. **Deploy**. The first build installs requirements (a few minutes). The first page load then
   downloads ~77 MB from HF (seconds) and caches it on the instance.

### Step 5: cap spending in the Anthropic Console

In <https://console.anthropic.com> → **Settings → Limits**, set a monthly spend limit
you're comfortable with (e.g. $5–$10). Optionally create a dedicated API key for this app so you
can revoke it without touching other projects. If the key runs out of credit, the
parser call fails and the app falls back to the rule-based parser.

### Step 6: check the live app

On the live app (or the deploy-check page) confirm:

- "Data source: deploy" and "First season: 1996", and the load message shows ~779k player-games.
- "AI parser: on". If it's off, the key is missing or still the placeholder.
- Type a few questions: the counter drops from 20. Remove the key from secrets and the app
  says it's using the rule-based parser.
- Open the link in a private window (logged out of Streamlit) to see what visitors see.

Then share `https://<subdomain>.streamlit.app`.

### Projection models

The projection engine (`nbalab.models`) reads a trained bundle from `models/`:

```
models/
  LATEST                  # name of the bundle the app loads, e.g. v1.0.0-20260929
  v1.0.0-20260929/        # ~31 MB: minutes model, chosen rate models per stat, baselines,
                          # calibration.json, manifest.json (seasons, features, metrics)
models_archive/           # gitignored: unchosen quantile comparison models (backtest only)
```

Only `models/` ships. No single file is near GitHub's 100 MB limit. LightGBM needs
OpenMP: Streamlit Cloud gets it from `packages.txt` (`libgomp1`); on a Mac run
`brew install libomp` once.

**Retrain order.** Run these three in order, locally, after the data pipeline:

```bash
.venv/bin/python -m nbalab.models.train      # ~11 min: select on 2023-24, refit through 2023-24 -> models/<version>/, updates models/LATEST
.venv/bin/python -m nbalab.models.calibrate  # ~10 s: minutes widening + over/under calibration, fitted on 2024-25 only
.venv/bin/python -m nbalab.models.evaluate   # ~1.5 min: backtest 2024-25 / 2025-26 -> docs/model_report.md, docs/img/, docs/model_metrics.json
```

`train` writes a bundle with **no** calibration; `calibrate` adds it; `evaluate` scores the
calibrated bundle and needs `models_archive/` from the same `train` run (it refuses to run
without it, and refuses if training overlaps the test seasons). Then commit `models/` and
`docs/`, and push.

## 4. Updating

| Change | What to do |
|---|---|
| Code | `git push`. Streamlit Cloud redeploys automatically. |
| Models (new season, new features) | `nbalab.models.train` → `nbalab.models.calibrate` → `nbalab.models.evaluate`, commit `models/` and `docs/`, push. Move the season split in `src/nbalab/models/config.py` forward first. |
| Data (new games, rebuilt pipeline) | `nbalab.data.build` → `nbalab.deploy.slim` → `nbalab.deploy.publish --repo ...`, then **Reboot app** in Streamlit Cloud (the running instance keeps its old copy until reboot). |
| App reads a new column | Add it to `src/nbalab/deploy/manifest.py` (the test fails until you do), then re-slim and re-publish. |
| New dependency | `uv add <pkg>`, then regenerate `requirements.txt` (command at the top of that file). |
| Pin a data version | Set `NBALAB_DATA_REVISION` in secrets to an HF commit hash. |

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| "No data found..." | `NBALAB_DATA_REPO` missing or misspelled in secrets. |
| 401/403 or "Repository not found" on download | `HF_TOKEN` missing, or the fine-grained token wasn't granted this repo. |
| App restarts, "over resource limits" | Set `NBALAB_SKIP_ON_COURT = "1"` and reboot. |
| `ModuleNotFoundError: nbalab` | The entry file must add `src/` to `sys.path` (see the top of `scripts/deploy_check_app.py`). |
| LightGBM `libgomp.so.1` error | `packages.txt` must be in the repo root. |
| Build fails on a pinned package | The Python version isn't 3.11. Redeploy with 3.11. |
