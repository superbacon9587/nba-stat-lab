"""A small in-memory name index for parser tests (real ids, no parquet needed).

Rows are copied from ``players.parquet`` / ``teams.parquet`` / ``player_games``
(September 2026 build). Aliases are generated with the same functions and
curated nickname files the pipeline uses.
"""

from __future__ import annotations

import io

import pandas as pd

from nbalab.data.aliases import attach_nicknames, load_nicknames, player_aliases, team_aliases
from nbalab.data.config import REFERENCE_DIR
from nbalab.nlp.entities import EntityIndex

PLAYERS_CSV = """personId,firstName,lastName,position_group,heightInches,first_season,last_season,games,last_team_id
103,Todd,Day,G,78,1996,2000,182,1610612750
209,Dell,Curry,G,77,1996,2001,389,1610612761
688,Michael,Curry,F,77,1996,2004,640,1610612754
779,Glen,Rice,F,80,1996,2003,490,1610612746
2201,Eddy,Curry,C,84,2001,2012,527,1610612742
2229,Mike,James,G,74,2001,2013,621,1610612741
2544,LeBron,James,F,81,2003,2025,1927,1610612747
201142,Kevin,Durant,F,83,2007,2025,1373,1610612745
201935,James,Harden,G,77,2009,2025,1410,1610612739
201939,Stephen,Curry,G,74,2009,2025,1229,1610612744
202691,Klay,Thompson,G,77,2011,2025,1095,1610612742
203076,Anthony,Davis,F,82,2012,2025,873,1610612742
203110,Draymond,Green,F,78,2012,2025,1122,1610612744
203318,Glen,Rice,F,78,2013,2014,14,1610612764
203507,Giannis,Antetokounmpo,F,83,2013,2025,979,1610612749
203552,Seth,Curry,G,73,2013,2025,599,1610612744
203954,Joel,Embiid,C,84,2016,2025,557,1610612755
203999,Nikola,Jokic,C,83,2015,2025,909,1610612743
1627759,Jaylen,Brown,F,78,2016,2025,815,1610612738
1628369,Jayson,Tatum,F,80,2017,2025,729,1610612738
1628455,Mike,James,G,73,2017,2020,56,1610612751
1628973,Jalen,Brunson,G,74,2018,2025,646,1610612752
1629029,Luka,Doncic,G,80,2018,2025,567,1610612747
1630178,Tyrese,Maxey,G,74,2020,2025,440,1610612755
1641705,Victor,Wembanyama,C,88,2023,2025,204,1610612759
"""

TEAMS_CSV = """teamId,city,name,abbrev
1610612737,Atlanta,Hawks,ATL
1610612738,Boston,Celtics,BOS
1610612739,Cleveland,Cavaliers,CLE
1610612740,New Orleans,Pelicans,NOP
1610612741,Chicago,Bulls,CHI
1610612742,Dallas,Mavericks,DAL
1610612743,Denver,Nuggets,DEN
1610612744,Golden State,Warriors,GSW
1610612745,Houston,Rockets,HOU
1610612746,Los Angeles,Clippers,LAC
1610612747,Los Angeles,Lakers,LAL
1610612748,Miami,Heat,MIA
1610612749,Milwaukee,Bucks,MIL
1610612750,Minnesota,Timberwolves,MIN
1610612751,Brooklyn,Nets,BKN
1610612752,New York,Knicks,NYK
1610612753,Orlando,Magic,ORL
1610612754,Indiana,Pacers,IND
1610612755,Philadelphia,76ers,PHI
1610612756,Phoenix,Suns,PHX
1610612757,Portland,Trail Blazers,POR
1610612758,Sacramento,Kings,SAC
1610612759,San Antonio,Spurs,SAN
1610612760,Oklahoma City,Thunder,OKC
1610612761,Toronto,Raptors,TOR
1610612762,Utah,Jazz,UTA
1610612763,Memphis,Grizzlies,MEM
1610612764,Washington,Wizards,WAS
1610612765,Detroit,Pistons,DET
1610612766,Charlotte,Hornets,CHA
"""

# Earlier names for the franchises the tests exercise (starting years, None = current).
EXTRA_ERAS: dict[int, list[tuple[str, str, str, int, int | None]]] = {
    1610612740: [("New Orleans", "Hornets", "NOH", 2002, 2012), ("New Orleans", "Pelicans", "NOP", 2013, None)],
    1610612766: [("Charlotte", "Hornets", "CHH", 1988, 2001), ("Charlotte", "Bobcats", "CHA", 2004, 2013),
                 ("Charlotte", "Hornets", "CHA", 2014, None)],
    1610612760: [("Seattle", "SuperSonics", "SEA", 1967, 2007), ("Oklahoma City", "Thunder", "OKC", 2008, None)],
}

STINTS_CSV = """personId,teamId,first_season,last_season,games
103,1610612738,1996,1996,81
103,1610612756,1999,1999,67
103,1610612750,2000,2000,29
209,1610612766,1996,1997,132
209,1610612761,1999,2001,212
688,1610612765,1996,2002,418
688,1610612754,2004,2004,18
779,1610612766,1996,1997,173
779,1610612747,1998,1999,137
779,1610612746,2003,2003,18
2201,1610612741,2001,2004,289
2201,1610612752,2005,2009,222
2229,1610612748,2001,2002,92
2229,1610612741,2011,2013,20
2544,1610612739,2003,2009,550
2544,1610612748,2010,2013,381
2544,1610612739,2014,2017,451
2544,1610612747,2018,2025,545
201142,1610612760,2007,2015,732
201142,1610612744,2016,2018,256
201142,1610612751,2020,2022,146
201142,1610612756,2022,2024,160
201142,1610612745,2025,2025,79
201935,1610612760,2009,2011,263
201935,1610612745,2012,2020,706
201935,1610612751,2020,2021,88
201935,1610612755,2021,2022,102
201935,1610612746,2023,2025,207
201935,1610612739,2025,2025,44
201939,1610612744,2009,2025,1229
202691,1610612744,2011,2023,952
202691,1610612742,2024,2025,143
203076,1610612740,2012,2018,479
203076,1610612747,2019,2024,363
203076,1610612742,2024,2025,31
203110,1610612744,2012,2025,1122
203318,1610612764,2013,2014,14
203507,1610612749,2013,2025,979
203552,1610612755,2020,2021,114
203552,1610612766,2023,2024,74
203552,1610612744,2025,2025,11
203954,1610612755,2016,2025,557
203999,1610612743,2015,2025,909
1627759,1610612738,2016,2025,815
1628369,1610612738,2017,2025,729
1628455,1610612751,2020,2020,20
1628973,1610612742,2018,2021,301
1628973,1610612752,2022,2025,345
1629029,1610612742,2018,2024,472
1629029,1610612747,2024,2025,95
1630178,1610612755,2020,2025,440
1641705,1610612759,2023,2025,204
"""


def fixture_players() -> pd.DataFrame:
    p = pd.read_csv(io.StringIO(PLAYERS_CSV))
    p["full_name"] = p["firstName"] + " " + p["lastName"]
    generated = pd.Series([player_aliases(f, l) for f, l in zip(p["firstName"], p["lastName"])])
    nick = load_nicknames(REFERENCE_DIR / "player_nicknames.csv", "personId")
    p["aliases"] = attach_nicknames(generated, nick, p["personId"], "personId")
    return p


def _eras(team_id: int, city: str, name: str, abbrev: str) -> list[dict]:
    rows = EXTRA_ERAS.get(team_id, [(city, name, abbrev, 1996, None)])
    return [{"city": c, "name": n, "abbrev": a, "first_season": f, "last_season": l} for c, n, a, f, l in rows]


def fixture_teams() -> pd.DataFrame:
    t = pd.read_csv(io.StringIO(TEAMS_CSV))
    t["full_name"] = t["city"] + " " + t["name"]
    t["name_history"] = [_eras(*r) for r in zip(t["teamId"], t["city"], t["name"], t["abbrev"])]
    generated = pd.Series([
        sorted({a for e in hist for a in team_aliases(e["city"], e["name"], e["abbrev"])})
        for hist in t["name_history"]
    ])
    nick = load_nicknames(REFERENCE_DIR / "team_nicknames.csv", "teamId")
    t["aliases"] = attach_nicknames(generated, nick, t["teamId"], "teamId")
    return t


def fixture_index() -> EntityIndex:
    stints = pd.read_csv(io.StringIO(STINTS_CSV))
    return EntityIndex(fixture_players(), fixture_teams(), stints)
