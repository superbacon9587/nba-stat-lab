"""Parsing BoxScoreMatchupsV3 responses and attaching defender bio."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from nbalab.data.matchups import build_matchups, cache_path, clock_to_seconds, parse_game


def stats(**kw) -> dict:
    base = {"matchupMinutes": "1:30", "partialPossessions": 4.5, "playerPoints": 5,
            "matchupFieldGoalsMade": 2, "matchupFieldGoalsAttempted": 4, "matchupAssists": 1}
    return {**base, **kw}


RESPONSE = {
    "boxScoreMatchups": {
        "gameId": "0022300061",
        "homeTeam": {"teamId": 1, "players": [
            {"personId": 10, "position": "G", "matchups": [
                {"personId": 20, "statistics": stats()},
                {"personId": 21, "statistics": stats(matchupMinutes="0:42", partialPossessions=1.0)},
            ]},
        ]},
        "awayTeam": {"teamId": 2, "players": [
            {"personId": 20, "position": "", "matchups": [{"personId": 10, "statistics": stats()}]},
        ]},
    }
}


def test_clock_to_seconds() -> None:
    assert clock_to_seconds("0:42") == 42.0
    assert clock_to_seconds("12:05") == 725.0
    assert pd.isna(clock_to_seconds(None))


def test_parse_game_offense_defense_orientation() -> None:
    df = parse_game(RESPONSE)
    assert len(df) == 3
    home_off = df[df["offPersonId"] == 10]
    assert set(home_off["defPersonId"]) == {20, 21}
    assert set(home_off["offTeamId"]) == {1} and set(home_off["defTeamId"]) == {2}
    row = home_off[home_off["defPersonId"] == 21].iloc[0]
    assert row["matchupSeconds"] == 42.0 and row["partialPossessions"] == 1.0
    assert pd.isna(df.loc[df["offPersonId"] == 20, "offStartingPosition"].iloc[0])


def test_parse_game_empty_response() -> None:
    assert parse_game({"boxScoreMatchups": {}}).empty


def test_build_matchups_joins_defender_bio(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "0022300061", 2023)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump(RESPONSE, f)
    bio = pd.DataFrame({"personId": [20], "heightInches": [81.0], "bodyWeightLbs": [250.0],
                        "position_group": ["F"], "position": ["F"]})
    df = build_matchups(tmp_path, bio)
    assert df["season"].unique().tolist() == [2023]
    assert df["gameType"].unique().tolist() == ["regular"]
    lebron_like = df[df["defPersonId"] == 20].iloc[0]
    assert (lebron_like["defHeightInches"], lebron_like["defPosition"]) == (81.0, "F")
    assert df.loc[df["defPersonId"] == 21, "defHeightInches"].isna().all()
    assert df["partialPossessions"].sum() == pytest.approx(10.0)
