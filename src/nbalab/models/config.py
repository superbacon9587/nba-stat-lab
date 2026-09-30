"""Settings for the projection models: which stats, which seasons, where bundles live."""

from __future__ import annotations

from nbalab.data.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
ARCHIVE_DIR = PROJECT_ROOT / "models_archive"  # gitignored: unchosen comparison models

PLAYER_STATS: tuple[str, ...] = (
    "points", "assists", "reboundsTotal", "threePointersMade", "steals", "blocks", "turnovers",
)
TEAM_STATS: tuple[str, ...] = ("teamScore", "assists", "reboundsTotal", "threePointersMade")

# Time-based split (season = starting year). Never shuffle.
TRAIN_START_SEASON = 2012
VALIDATION_SEASON = 2023  # model selection: fit <= 2022, score 2023-24
TRAIN_END_SEASON = 2023   # final refit: <= 2023-24
TEST_SEASONS: tuple[int, ...] = (2024, 2025)

INTERVAL_LEVELS: tuple[int, ...] = (90, 95, 99)
QUANTILES: tuple[float, ...] = (0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995)
MC_DRAWS = 10_000
SEED = 20260929

# The named-defender on/off proxy made points predictions worse in the 2024-26 backtest
# (docs/model_report.md), so it is reported as a note and not applied to the numbers.
APPLY_DEFENDER_PROXY = False

# Post-hoc calibration (nbalab.models.calibrate): fitted on this season only,
# judged on the next one. Both lie inside TEST_SEASONS, after TRAIN_END_SEASON.
CALIBRATION_SEASON = 2024
CALIBRATION_TEST_SEASON = 2025

# Manual overrides, decided after reading the 2025-26 check in docs/model_report.md (so for these
# two choices 2025-26 is no longer untouched):
# - player turnovers: Platt made 2025-26 Brier/ECE slightly worse -> no calibration.
# - team threes: calibrated over/under help, but calibrated pushes were badly off -> raw push.
CALIBRATION_FORCE_IDENTITY: tuple[tuple[str, str], ...] = (("player", "turnovers"),)
CALIBRATION_RAW_PUSH: tuple[tuple[str, str], ...] = (("team", "threePointersMade"),)
