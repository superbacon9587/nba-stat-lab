"""Play-by-play normalization, lineup reconstruction, and shared-floor-time aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from nbalab.data.lineups import reconstruct_game, resolve_name
from nbalab.data.on_court import pair_rows, player_rows
from nbalab.data.pbp import game_seconds, normalize_events, parse_clock

GAME = 21500001
HOME, AWAY = 100, 200
HOME_PLAYERS = [1, 2, 3, 4, 5, 6]  # 6 comes off the bench
AWAY_PLAYERS = [11, 12, 13, 14, 15]


def raw_row(**kw) -> dict:
    base = {
        "gameId": str(GAME), "period": 1, "clock": "PT12M00.00S", "actionNumber": 0,
        "orderNumber": None, "actionType": "", "subType": "", "description": "",
        "personId": "0", "playerName": None, "teamId": None, "shotResult": None,
        "shotValue": 0, "assistPersonId": None, "subsInPersonId": None,
    }
    return {**base, **kw}


def roster() -> pd.DataFrame:
    rows = []
    for pid in HOME_PLAYERS:
        rows.append((GAME, pid, HOME, "F", f"Home{pid}", pid != 6, 30.0))
    for pid in AWAY_PLAYERS:
        rows.append((GAME, pid, AWAY, "F", f"Away{pid}", True, 48.0))
    return pd.DataFrame(rows, columns=["gameId", "personId", "teamId", "firstName", "lastName", "starter", "minutes"])


def legacy_game() -> pd.DataFrame:
    """One legacy-format period: a made 3 by 1, a sub 'Home6 FOR Home1' at 6:00, a FT miss by 6."""
    return pd.DataFrame([
        raw_row(actionNumber=1, clock="PT11M00.00S", actionType="Made Shot", subType="Jump Shot",
                description="Home1 26' 3PT Jump Shot (3 PTS) (Home2 1 AST)", personId="1",
                playerName="Home1", teamId=str(HOME), shotResult="Made", shotValue=3),
        raw_row(actionNumber=2, clock="PT06M00.00S", actionType="Substitution",
                description="SUB: Home6 FOR Home1", personId="1", playerName="Home1", teamId=str(HOME)),
        raw_row(actionNumber=3, clock="PT03M00.00S", actionType="Free Throw", subType="Free Throw 1 of 2",
                description="MISS Home6 Free Throw 1 of 2", personId="6", playerName="Home6", teamId=str(HOME)),
        raw_row(actionNumber=4, clock="PT03M00.00S", actionType="Free Throw", subType="Free Throw 2 of 2",
                description="Home6 Free Throw 2 of 2 (1 PTS)", personId="6", playerName="Home6", teamId=str(HOME)),
        raw_row(actionNumber=5, clock="PT01M00.00S", actionType="Foul", subType="Technical",
                description="Away11 T.FOUL", personId="11", playerName="Away11", teamId=str(AWAY)),
    ])


def test_parse_clock_and_game_seconds() -> None:
    clock = pd.Series(["PT06M40.00S", "PT00M20.90S"])
    assert parse_clock(clock).tolist() == pytest.approx([400.0, 20.9])
    # Q2 with 12:00 left = 720 s elapsed; first OT with 0:00 left = 48 + 5 minutes.
    elapsed = game_seconds(pd.Series([2, 5]), pd.Series(["PT12M00.00S", "PT00M00.00S"]))
    assert elapsed.tolist() == [720.0, 3180.0]


def test_normalize_legacy_stats_subs_and_assist() -> None:
    ev = normalize_events(legacy_game())
    made = ev[(ev["personId"] == 1) & (ev["fga"] == 1)].iloc[0]
    assert (made["pts"], made["fg3m"], made["fg3a"]) == (3, 1, 1)
    assert ev.loc[ev["assistName"] == "Home2", "ast"].tolist() == [1]
    sub = ev[ev["kind"].isin(["sub_out", "sub_in"])]
    assert sub["kind"].tolist() == ["sub_out", "sub_in"]
    assert sub["inName"].iloc[1] == "Home6"
    fts = ev[ev["fta"] == 1]
    assert fts["ftm"].tolist() == [0, 1]
    # A technical foul is not evidence of being on the floor.
    assert not ((ev["personId"] == 11) & (ev["kind"] == "ref")).any()


def test_normalize_modern_subs_in_person_id() -> None:
    raw = pd.DataFrame([
        raw_row(orderNumber=10, clock="PT07M20.00S", actionType="substitution", subType="out",
                personId="1", teamId=str(HOME), subsInPersonId="6"),
    ])
    ev = normalize_events(raw)
    assert ev[["kind", "personId"]].values.tolist() == [["sub_out", 1], ["sub_in", 6]]


def test_resolve_name_prefers_bench_player() -> None:
    index = {HOME: {"morris": {7, 8}}}
    assert resolve_name("Morris", HOME, index, avoid={7}) == 8
    assert resolve_name("Morris", HOME, index) is None  # ambiguous
    assert resolve_name("Marc Morris", HOME, index, avoid={8}) == 7  # last-word fallback


def test_reconstruct_shared_floor_time_and_stats() -> None:
    rec = reconstruct_game(normalize_events(legacy_game()), roster())
    players = player_rows(rec).set_index("personId")
    # Only period 1 has events, so the game is 720 s long here.
    assert players.loc[1, "reconSeconds"] == pytest.approx(360.0)  # subbed out at 6:00
    assert players.loc[6, "reconSeconds"] == pytest.approx(360.0)
    assert players.loc[2, "reconSeconds"] == pytest.approx(720.0)
    assert players.loc[1, "pbp_pts"] == 3 and players.loc[6, "pbp_pts"] == 1
    assert players.loc[2, "pbp_ast"] == 1

    pairs = pair_rows(rec).set_index(["personId", "opponentPersonId"])
    assert pairs.loc[(1, 11), "sharedFloorSeconds"] == pytest.approx(360.0)
    assert pairs.loc[(11, 1), "sharedFloorSeconds"] == pytest.approx(360.0)  # symmetric
    assert pairs.loc[(6, 11), "pts"] == 1
    assert pairs.loc[(11, 6), "pts"] == 0  # stats belong to the row's own player
    assert len(pairs) == 2 * len(HOME_PLAYERS) * len(AWAY_PLAYERS)


def test_reconstruct_repairs_missing_sub_in() -> None:
    """Player 6 acts without a recorded sub-in after 1 was subbed out: he is back-filled."""
    raw = legacy_game()
    raw.loc[1, "description"] = "SUB:  FOR Home1"  # entering player's name missing
    rec = reconstruct_game(normalize_events(raw), roster())
    players = player_rows(rec).set_index("personId")
    assert rec.repairs["unresolved_sub_in"] == 1
    assert rec.repairs["backfilled"] == 1
    assert players.loc[6, "reconSeconds"] == pytest.approx(360.0)  # entered when the slot opened


def test_later_period_starters_inferred() -> None:
    """Period 2 has no subs: the five home players who act, and the silent carry-overs, start it."""
    raw = pd.concat([
        legacy_game(),
        pd.DataFrame([raw_row(period=2, actionNumber=10, clock="PT10M00.00S", actionType="Rebound",
                              personId="6", playerName="Home6", teamId=str(HOME))]),
    ], ignore_index=True)
    rec = reconstruct_game(normalize_events(raw), roster())
    q2 = [s for s in rec.segments if s.period == 2]
    assert q2[0].lineups[HOME] == {2, 3, 4, 5, 6}
    assert q2[0].lineups[AWAY] == set(AWAY_PLAYERS)
