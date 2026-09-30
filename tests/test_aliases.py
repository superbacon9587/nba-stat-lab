"""Tests for alias generation."""

from __future__ import annotations

import pandas as pd

from nbalab.data.aliases import attach_nicknames, normalize_alias, player_aliases, team_aliases


def test_normalize_alias() -> None:
    assert normalize_alias("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert normalize_alias("D'Angelo  Russell") == "dangelo russell"
    assert normalize_alias("P.J. Tucker") == "pj tucker"


def test_player_aliases_diminutive_initial_and_last_name() -> None:
    a = player_aliases("Stephen", "Curry")
    assert {"stephen curry", "steph curry", "s curry", "curry", "stephen"} <= set(a)


def test_player_aliases_strip_suffix() -> None:
    a = player_aliases("Jaren", "Jackson Jr.")
    assert {"jaren jackson jr", "jaren jackson", "jackson"} <= set(a)


def test_player_aliases_initials_for_three_part_names() -> None:
    assert "sga" in player_aliases("Shai", "Gilgeous-Alexander")
    assert "sc" not in player_aliases("Stephen", "Curry")  # two-part initials are too ambiguous


def test_team_aliases_city_shorthand() -> None:
    a = team_aliases("Los Angeles", "Lakers", "LAL ")
    assert {"lakers", "los angeles lakers", "la lakers", "lal"} <= set(a)


def test_attach_nicknames_merges_by_id() -> None:
    base = pd.Series([["a"], ["b"]])
    nick = pd.DataFrame({"personId": [2, 2], "alias": ["x", "b"]})
    out = attach_nicknames(base, nick, pd.Series([1, 2]), "personId")
    assert out.tolist() == [["a"], ["b", "x"]]
