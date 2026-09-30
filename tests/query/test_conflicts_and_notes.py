"""Conflict detection with one-click fixes, proxy-defender wording, and the pre-dataset career note."""

from __future__ import annotations

import pandas as pd
import pytest

from nbalab.query import schema as s
from nbalab.query.conflicts import current_team, find_conflicts
from nbalab.query.data import QueryData
from nbalab.query.engine import career_started_before_data, relabel_named_defenders, run_split

SONICS = 1610612760
DURANT, COLLISON = 201142, 2555


@pytest.fixture(scope="module")
def data(sample_tables: dict[str, pd.DataFrame]) -> QueryData:
    t = sample_tables
    return QueryData(t["player_games"], t["team_games"], t["players"], t["teams"])


def _opponent_player(data: QueryData) -> tuple[int, int, int]:
    """(person, his team, another opponent team he never played for)."""
    pg = data.player_games
    opp = pg[pg["teamId"].ne(SONICS)]
    one_team = opp.groupby("personId")["teamId"].nunique().eq(1)
    pid = int(one_team[one_team].index[0])
    team = int(opp.loc[opp["personId"].eq(pid), "teamId"].iloc[0])
    other = int(next(t for t in opp["teamId"].unique() if t != team))
    return pid, team, other


def test_projection_defender_on_other_team_offers_both_fixes(data: QueryData) -> None:
    pid, team, other = _opponent_player(data)
    q = s.StatQuery(subject_ids=[DURANT], stats=["points"], mode="projection",
                    projection={"context": {"opponent_team_id": other, "defender_person_id": pid}})
    [c] = find_conflicts(q, data)
    assert "can't defend" in c.message
    labels = [f.label for f in c.fixes]
    assert labels[0].startswith("Use the") and labels[1] == "Drop the defender"
    assert c.fixes[0].query.projection.context.opponent_team_id == team == current_team(data, pid)
    assert c.fixes[1].query.projection.context.defender_person_id is None
    assert find_conflicts(c.fixes[0].query, data) == [] and find_conflicts(c.fixes[1].query, data) == []


def test_teammate_as_defender(data: QueryData) -> None:
    q = s.StatQuery(subject_ids=[DURANT], stats=["points"], mode="projection",
                    projection={"context": {"defender_person_id": COLLISON}})
    [c] = find_conflicts(q, data)
    assert "teammate" in c.message and [f.label for f in c.fixes] == ["Drop the defender"]


def test_split_defender_never_on_opponent(data: QueryData) -> None:
    pid, team, other = _opponent_player(data)
    q = s.StatQuery(subject_ids=[DURANT], stats=["points"],
                    filters=[{"type": "opponent_team", "team_ids": [other]}, {"type": "defender_player", "person_id": pid}])
    [c] = find_conflicts(q, data)
    assert [f.label for f in c.fixes][1:] == ["Drop the defender", "Drop the opponent"]
    fixed = c.fixes[0].query
    assert [f.team_ids for f in fixed.filters if f.type == "opponent_team"] == [[team]]
    ok = s.StatQuery(subject_ids=[DURANT], stats=["points"],
                     filters=[{"type": "opponent_team", "team_ids": [team]}, {"type": "defender_player", "person_id": pid}])
    assert find_conflicts(ok, data) == []


def test_team_cannot_play_itself(data: QueryData) -> None:
    q = s.StatQuery(subject_type="team", subject_ids=[SONICS], stats=["team_score"],
                    filters=[{"type": "opponent_team", "team_ids": [SONICS]}])
    [c] = find_conflicts(q, data)
    assert c.fixes[0].label == "Drop the opponent" and not c.fixes[0].query.filters


def test_proxy_defender_is_described_as_shared_floor(data: QueryData) -> None:
    pid, team, _ = _opponent_player(data)
    q = s.StatQuery(subject_ids=[DURANT], stats=["points"], filters=[{"type": "defender_player", "person_id": pid}])
    res = run_split(q, data)[0]
    res.defender_sources = {"shared_floor_time": 3}
    res.filters_described = ["defended by X"]
    relabel_named_defenders(res, q, data)
    assert res.filters_described == [f"games where Kevin Durant and {data.player_name(pid)} shared the floor"]
    res.defender_sources = {"official_matchups": 5, "shared_floor_time": 1}
    res.filters_described = ["defended by X"]
    relabel_named_defenders(res, q, data)
    assert res.filters_described == ["defended by X"]  # real matchup data: keep "defended by"


def test_career_before_data_note(data: QueryData) -> None:
    # the sample starts in 2007-08: Collison (drafted 2003) began earlier, Durant (2007 rookie) did not
    assert career_started_before_data(COLLISON, data) == "drafted in 2003"
    assert career_started_before_data(DURANT, data) is None
    res = run_split(s.StatQuery(subject_ids=[COLLISON], stats=["points"]), data)[0]
    assert res.career_note and "not included" in res.career_note
