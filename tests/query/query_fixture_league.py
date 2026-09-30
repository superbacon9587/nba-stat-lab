"""A tiny hand-built league with known answers for the query engine tests.

Subject: player 10 (a guard, team 1). Twelve games, 30 minutes each:

    games 1-4   vs team 2 ("Bees")    points 10, 20, 30, 20   -> mean 20
    games 5-10  vs team 3 ("Cats")    points 25, 25, 30, 20, 30, 20 -> mean 25
    games 11-12 vs team 4 ("Dogs")    points 35, 35           -> mean 35
    baseline mean = 300 / 12 = 25, sample variance = 600 / 11

Games 1-6 are season 2023, games 7-12 season 2024. Home games are the odd ones.
The subject starts at G in games 1-10 and comes off the bench in 11-12.

Opponents:
    20  Bees guard, 78 in, starts at G, 30 min; plays games 1-3 (misses game 4)
    22  Bees guard, 76 in, starts at G, 20 min (fewer minutes than 20, so never the proxy)
    21  Bees forward, 81 in, starts at F
    30  Cats guard, 72 in, starts at G
    40  Dogs center, 80 in, 30 min;  41 Dogs guard, 70 in, 10 min
        -> Dogs minutes-weighted height = (80*30 + 70*10) / 40 = 77.5
Teammate 11 (team 1) plays games 1-8 only.

Optional tables:
    on_court:  games 1 and 2 have shared floor time between 10 and 20 (20 and 5 minutes)
    matchups:  game 3, subject guarded by 20 for 2 partial possessions and by 21 for 8
"""

from __future__ import annotations

import pandas as pd

from nbalab.query.data import QueryData, normalize_matchups, normalize_on_court

SUBJECT, TEAMMATE = 10, 11
BEES, CATS, DOGS, HOME_TEAM = 2, 3, 4, 1
SUBJECT_POINTS = [10, 20, 30, 20, 25, 25, 30, 20, 30, 20, 35, 35]
OPPONENT_BY_GAME = [BEES] * 4 + [CATS] * 6 + [DOGS] * 2
CITY = {HOME_TEAM: "Hometown", BEES: "Beeville", CATS: "Catcity", DOGS: "Dogtown"}

BIO = {  # personId: (team, height, weight, position group)
    SUBJECT: (HOME_TEAM, 75, 190, "G"), TEAMMATE: (HOME_TEAM, 80, 220, "F"),
    20: (BEES, 78, 210, "G"), 22: (BEES, 76, 200, "G"), 21: (BEES, 81, 240, "F"),
    30: (CATS, 72, 180, "G"), 40: (DOGS, 80, 250, "C"), 41: (DOGS, 70, 170, "G"),
}


def game_context(game: int) -> dict:
    """Shared context columns for game number 1..12."""
    date = pd.Timestamp("2023-11-01") + pd.Timedelta(days=2 * (game - 1))
    if game > 6:
        date = pd.Timestamp("2024-11-01") + pd.Timedelta(days=2 * (game - 7))
    opp = OPPONENT_BY_GAME[game - 1]
    home = game % 2 == 1
    return {
        "gameId": 1000 + game,
        "gameDateTimeEst": date,
        "game_date": date,
        "season": 2023 if game <= 6 else 2024,
        "season_label": "2023-24" if game <= 6 else "2024-25",
        "game_type": "regular",
        "is_playoff": False,
        "day_of_week": date.day_name(),
        "day_of_week_num": date.dayofweek,
        "month": date.month,
        "week_of_season": (game - 1) // 3 + 1,
        "venue_team_id": HOME_TEAM if home else opp,
        "venue_city": CITY[HOME_TEAM if home else opp],
        "arena_city": None,
        "team_rest_days": 0.0 if game == 2 else 1.0,
        "team_is_back_to_back": game == 2,
        "opp_pre_def_rating": 110.0,
        "opp_pre_pace": 99.0,
    }


def box(person: int, team: int, opp: int, minutes: float, points: float, start: str | None, ctx: dict) -> dict:
    h, w, pos = BIO[person][1:]
    home = ctx["venue_team_id"] == team
    return {
        **ctx, "personId": person, "teamId": team, "opponentTeamId": opp, "home": int(home),
        "minutes": minutes, "points": points, "assists": 5.0, "reboundsTotal": 4.0,
        "reboundsOffensive": 1.0, "reboundsDefensive": 3.0, "steals": 1.0, "blocks": 0.0,
        "turnovers": 2.0, "foulsPersonal": 2.0, "fieldGoalsMade": points / 2.5,
        "fieldGoalsAttempted": 20.0, "threePointersMade": 2.0, "threePointersAttempted": 5.0,
        "freeThrowsMade": 2.0, "freeThrowsAttempted": 2.0, "plusMinusPoints": 0.0,
        "starter": start is not None, "startingPosition": start,
        "heightInches": h, "bodyWeightLbs": w, "position_group": pos, "age": 27.0,
    }


def build_player_games() -> pd.DataFrame:
    rows = []
    for g in range(1, 13):
        ctx = game_context(g)
        opp = OPPONENT_BY_GAME[g - 1]
        rows.append(box(SUBJECT, HOME_TEAM, opp, 30, SUBJECT_POINTS[g - 1], "G" if g <= 10 else None, ctx))
        if g <= 8:
            rows.append(box(TEAMMATE, HOME_TEAM, opp, 25, 12, "F", ctx))
        if opp == BEES:
            if g != 4:
                rows.append(box(20, BEES, HOME_TEAM, 30, 15, "G", ctx))
            rows.append(box(22, BEES, HOME_TEAM, 20, 8, "G", ctx))
            rows.append(box(21, BEES, HOME_TEAM, 32, 18, "F", ctx))
        elif opp == CATS:
            rows.append(box(30, CATS, HOME_TEAM, 34, 22, "G", ctx))
        else:
            rows.append(box(40, DOGS, HOME_TEAM, 30, 14, "C", ctx))
            rows.append(box(41, DOGS, HOME_TEAM, 10, 4, None, ctx))
    df = pd.DataFrame(rows)
    df["gameId"] = df["gameId"].astype("int64")
    df["startingPosition"] = df["startingPosition"].astype("category")
    return df


def build_team_games(pg: pd.DataFrame) -> pd.DataFrame:
    """Team 1's games: team score = sum of its players' points + 70; opponent 100."""
    ours = pg[pg["teamId"] == HOME_TEAM]
    ctx_cols = [c for c in game_context(1) if c != "gameId"]
    per_game = ours.groupby("gameId").agg(teamScore=("points", "sum")).reset_index()
    ctx = ours.drop_duplicates("gameId").set_index("gameId")[ctx_cols + ["opponentTeamId", "home"]]
    tg = per_game.join(ctx, on="gameId")
    tg["teamId"] = HOME_TEAM
    tg["teamScore"] = tg["teamScore"] + 70
    tg["opponentScore"] = 100.0
    for c in ("fieldGoalsMade", "fieldGoalsAttempted", "freeThrowsAttempted", "threePointersMade",
              "threePointersAttempted", "freeThrowsMade", "assists", "reboundsTotal"):
        tg[c] = 10.0
    return tg


def build_data() -> QueryData:
    """The fixture league with on_court and matchups tables."""
    pg = build_player_games()
    players = pd.DataFrame(
        [{"personId": p, "full_name": f"Player {p}", "firstName": "Player", "lastName": str(p),
          "aliases": [f"player {p}", str(p)], "heightInches": b[1], "bodyWeightLbs": b[2],
          "position_group": b[3], "games": 10, "first_season": 2023, "last_season": 2024}
         for p, b in BIO.items()]
    )
    teams = pd.DataFrame(
        [{"teamId": t, "full_name": f"{c} {n}", "abbrev": n[:3].upper(), "aliases": [n.lower(), c.lower()],
          "name_history": [{"city": c, "name": n, "abbrev": n[:3].upper(), "first_season": 2000,
                            "last_season": None}]}
         for t, c, n in [(1, "Hometown", "Humans"), (2, "Beeville", "Bees"), (3, "Catcity", "Cats"),
                         (4, "Dogtown", "Dogs")]]
    )
    on_court = normalize_on_court(pd.DataFrame({
        "gameId": [1001, 1002], "personId": [SUBJECT, SUBJECT], "opponentPersonId": [20, 20],
        "shared_floor_minutes": [20.0, 5.0], "reliable": [True, True],
        "pts": [8, 2], "fgm": [3, 1], "fga": [6, 3], "fg3m": [1, 0], "fg3a": [2, 1], "ftm": [1, 0],
        "fta": [2, 0], "ast": [2, 1],
    }))
    matchups = normalize_matchups(pd.DataFrame({
        "gameId": [1003, 1003], "offPersonId": [SUBJECT, SUBJECT], "defPersonId": [20, 21],
        "partialPossessions": [2.0, 8.0],
    }))
    return QueryData(pg, build_team_games(pg), players, teams, on_court, matchups)
