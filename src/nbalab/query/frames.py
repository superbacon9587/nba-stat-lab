"""Build one subject's games table with every column filters and group-bys read.

Player subjects read ``player_games``; team subjects read ``team_games``. Both
get the same derived columns (``home_away``, ``rest_bucket``, ...). Player
frames also get the primary-defender attributes from :mod:`nbalab.query.defenders`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbalab.query.data import QueryData
from nbalab.query.defenders import defender_attributes, opponent_rows
from nbalab.query.stats import height_bucket, rest_bucket, weight_bucket


def subject_frame(data: QueryData, subject_type: str, subject_id: int) -> pd.DataFrame:
    """All games for one player or team, oldest first, with derived columns."""
    if subject_type == "team":
        df = data.team_games[data.team_games["teamId"] == subject_id].copy()
    else:
        df = data.player_games[data.player_games["personId"] == subject_id].copy()
    df = df.sort_values(["gameDateTimeEst", "gameId"]).reset_index(drop=True)
    return add_derived_columns(df, data, subject_type)


def add_derived_columns(df: pd.DataFrame, data: QueryData, subject_type: str) -> pd.DataFrame:
    """Readable labels for grouping plus defender attributes for players."""
    df["home_away"] = np.where(df["home"].astype("float64") == 1, "home", "away")
    df["opponent_name"] = [data.team_name(int(t), int(s)) for t, s in zip(df["opponentTeamId"], df["season"])]
    df["rest_bucket"] = rest_bucket(df["team_rest_days"])
    df["back_to_back"] = df["team_is_back_to_back"].astype(bool)
    df["playoff_label"] = np.select(
        [df["game_type"].astype(str).eq("playoffs"), df["game_type"].astype(str).eq("play_in")],
        ["playoffs", "play-in"], "regular season",
    )
    df["day_of_week"] = df["day_of_week"].astype(str)
    df["venue_city"] = df["venue_city"].astype("object")
    if subject_type == "player" and len(df):
        opp = opponent_rows(df, data)
        df[["def_person_id", "def_height", "def_weight", "def_position", "def_source"]] = (
            defender_attributes(df, data, opp)
        )
        df["def_height_bucket"] = height_bucket(df["def_height"])
        df["def_weight_bucket"] = weight_bucket(df["def_weight"])
    return df
