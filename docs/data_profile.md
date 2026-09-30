# Data profile: `data/raw/`

Profiled 2026-09-29 with duckdb (full scans, no sampling). Null rates are over **all** rows in the file, back to 1946 where applicable. The pipeline defaults to 1996-97 onward.

## Summary

| File | Rows | Distinct games | Date range (EST) | Notes |
|---|---:|---:|---|---|
| `PlayerStatistics.csv` | 1,669,922 | 73,471 | 1946-11-26 → 2026-06-13 | One row per player per game, including DNPs. No duplicate `(personId, gameId)`. |
| `PlayerStatisticsExtended.csv` | 838,803 | 40,024 | 1996-11-01 → 2026-06-13 | Same grain plus ~70 advanced columns. No duplicates. |
| `TeamStatistics.csv` | 146,560 | 73,279 | 1946-11-26 → 2026-06-13 | 2 rows per game. No duplicate `(gameId, teamId)`. |
| `TeamStatisticsExtended.csv` | 79,724 | 39,862 | 1996-11-01 → 2026-06-13 | 2 rows per game. Ratings, pace, possessions, four factors. |
| `Games.csv` | 73,279 | 73,279 | 1946-11-26 → 2026-06-13 | One row per game. |
| `Players.csv` | 6,692 | | birthDates 1900-01-01 → 2006-12-21 | One row per person. `1900-01-01` is a placeholder, not a real birthday. |
| `TeamHistories.csv` | 140 | | seasons 1946 → "2100" (= still active) | 30 NBA franchises plus ~55 exhibition/international/All-Star ids. |
| `LeagueSchedule24_25.csv` | 1,408 | 1,408 | 2024-10-04 → 2025-10-17 | Includes preseason, and its tail runs into the 2025-26 preseason. `weekNumber` 0–35. Timestamps carry a `+00:00` suffix but are EST wall-clock. |
| `LeagueSchedule25_26.csv` | 1,400 | 1,400 | 2025-10-02 → 2026-06-13 | `weekNumber` 0–34. Column names differ from 24-25 (`homeTeamId` vs `hometeamId`). |
| `PlayByPlay.parquet` | 18,727,295 | 39,164 | 1996-11-01 → 2026-04-19 | Ends before the 2026 playoffs. 199 games have null dates. |

## Findings that drive pipeline decisions

### `gameType` is messy and sometimes missing
Raw values seen across files: `Regular Season`, `Playoffs`, `Preseason`, `Pre Season`, `Play-in Tournament`, `All-Star Game`, `NBA Emirates Cup`, `Emirates NBA Cup`, `NBA Cup`, `in-season-knockout`, and null (7,590 rows in `TeamStatistics`, 7,510 in `TeamStatisticsExtended`, 50 in `PlayerStatistics`).

The **first digit of `gameId`** is reliable and matches the official NBA encoding: `1` preseason, `2` regular season (NBA Cup group and knockout games live here), `3` All-Star, `4` playoffs, `5` play-in, `6` NBA Cup final (which does not count in regular-season stats). The pipeline normalizes the text label and falls back to the gameId prefix when the label is null.

Normalized values are `regular`, `nba_cup`, `play_in`, `playoffs`, `preseason`, and `all_star`. `preseason` and `all_star` are dropped.

### `gameId` also encodes the season
Every `gameId` is 8 digits: `T YY NNNNN`, where `YY` is the season's starting year (`22400123` is regular season 2024-25). A date-only rule (Oct–Dec = this year's season) breaks on the 2020 bubble, where the Finals were played in October 2020 but belong to 2019-20. The pipeline uses the gameId season and cross-checks it against the date rule. The date rule is kept as a documented fallback.

### Team ids are missing on ~6.5% of player rows
`playerteamId` / `opponentteamId` are null on 108k rows. The null rate is 0% before 2000, 8–9% in the 2000s and 2010s, and **21% in the 2020s**. That includes 45,873 regular-season and 2,374 playoff rows since 1996. The team *name* is present on all of them. Matching `playerteamName` to the home or away team name in `Games.csv` for the same `gameId` recovers **100%** (105,603 / 105,603 checked). The two teams in a game never share a name.

### `numMinutes` is mixed-type
Values are mostly decimal strings (`"24.0"`, `"19.983333333333334"`). A few hundred are `"MM:SS"` (`"22:29"`). 10.1% are null, which marks a DNP (`comment` holds the reason, e.g. "DNP - Coach's Decision"). 103k rows are exactly `0.0`. Since 1996 there are 1,003,892 rows: 168,168 have null minutes and 8,429 have zero minutes. Both groups are dropped as "did not play".

### Venue has to be derived
`Games.arenaName/arenaCity` is populated only for 2025-10 → 2026-06 (98.1% null overall). `arenaId` is always filled but has no lookup table. The venue is therefore the home team's city for that season, taken from `TeamHistories`. Relocations are handled because city is looked up by season. Arena name comes from `Games.csv` and the two schedule files where available. Those sources also catch neutral-site games (e.g. Abu Dhabi, Mexico City).

### Franchise relocations
Since 1996 only the 30 standard NBA ids (`1610612737`–`1610612766`) appear in non-All-Star games. Relocations and renames that share one id: SuperSonics → Thunder (`…760`), NJ → Brooklyn Nets (`…751`), Vancouver → Memphis (`…763`), Bullets → Wizards (`…764`), Charlotte Hornets → Bobcats → Hornets (`…766`), and NO/OKC Hornets → Pelicans (`…740`). **"Hornets" refers to two different franchises** depending on season: `…766` for 1988-2001 and 2014+, `…740` for 2002-2012.

`TeamHistories` uses the season's starting year (`seasonFounded=1997` for the Wizards means 1997-98). Its abbreviation column is space-padded, and San Antonio is `SAN` (not the common `SAS`).

### Players bio coverage
Height and weight are ~76% populated overall. Of the 6,692 people, 1,540 have no G/F/C flag and 508 have two flags (G-F or F-C). Bio fields are a single current snapshot, not per season. 149 personIds in `PlayerStatistics` are missing from `Players.csv`. Two player rows have commas inside `firstName` (1940s nicknames). All names are plain ASCII (no diacritics: "Jokic", "Doncic"). Some non-NBA exhibition players duplicate NBA names (three "Luka Doncic" ids). They disappear once preseason is dropped.

### Placeholder rows for postponed games
`TeamStatistics.csv` has extra rows for postponed games (e.g. gameIds `22500651`, `22500652`): `teamId` 0, the original date (2026-01-25), and no stats. They sit next to the real rows of the rescheduled game under the same `gameId`. The pipeline keeps only team rows whose `teamId` is one of the game's two teams in `Games.csv`.

### Other
- `PlayerStatistics.startingPosition` (`G`/`F`/`C`) is set only for starters (34% of rows), so it doubles as a starter flag.
- `TeamStatistics.coachId` is 100% null.
- `Games.officials` is populated only for 2025-26.
- `TeamStatisticsExtended` starts at 1996-11-01, which is why the opponent defensive-rating and pace features exist only from 1996-97. That matches the default start season.

## On-court matchups and lineups

The box scores cannot say who guarded whom. There is no `data/raw/matchups` directory. The columns below are what we do have.

### `PlayByPlay.parquet`: can reconstruct lineups (who was on the floor together), not defensive assignments

**Two schemas are mixed in one file:**

- **Legacy format** (1996 → early 2019): `actionType` is capitalized (`Substitution`, `Made Shot`, …). Substitution rows have `personId` = the player **leaving**. The player entering appears only as a last name in `description` (`"SUB: Odom FOR Jones"`), and `subsInPersonId` is essentially never filled. The incoming player has to be resolved by last name against that game's box-score roster.
- **Modern format** (2019-20 partial, 100% from 2020-21): `actionType` is lowercase (`substitution`, `2pt`, `3pt`, …). There are separate `subType` `in` / `out` rows, each with its own `personId`. `personIdsFilter` lists every player involved in an event. `possession` gives the team with the ball. `x`/`y` and `area`/`areaDetail` give shot locations.

**Columns useful for matchups and lineups:**

| Column | Why it helps |
|---|---|
| `actionType`, `subType`, `description` | Substitution events. Starters come from box-score `startingPosition`. Apply subs to get the 5-man units on court at every event. |
| `personId`, `playerName`, `teamId`, `teamTricode` | Who acted, and for which team. |
| `subsInPersonId`, `subsInFullName` | Incoming sub (sparse, ~1% of modern sub rows). |
| `period`, `clock`, `actionNumber`, `orderNumber`, `timeActual` | Ordering, and stint durations for on-court minutes. |
| `personIdsFilter` | All players tied to an event (modern only). |
| `possession` | Offense/defense attribution per possession (modern only). |
| `scoreHome`, `scoreAway` | Points scored during each stint. |
| `blockPersonId`, `stealPersonId`, `foulDrawnPersonId`, `assistPersonId` | **The closest thing to a direct defender link:** who blocked a shot, stole from, or fouled whom. These are sparse event-level pairs, not full-possession guarding. |
| `x`, `y`, `xLegacy`, `yLegacy`, `area`, `areaDetail`, `shotDistance`, `shotResult` | Shot quality context for a player while a given opponent is on court. |
| `jumpBall*PersonId` | Minor: who jumped center. |

**Feasible proxy:** "Player X *while defender Y was on the court*" (on/off splits by opposing lineup), plus direct block, steal, and foul events. Where the play-by-play is too thin, a positional proxy can be used: Y's team's defense against X's position. Both must be labeled as proxies in the UI. True "guarded by" data would need NBA tracking matchup data, which is not in this dataset.

**Coverage gaps:** 39,164 games vs 39,862 in the extended box scores. The 2026 playoffs are missing. 1996, 1998, and 2011 are thin because those seasons started late (lockouts) or were cut short. 199 games have null dates. They can still be joined on `gameId`, and the `gameDateTimeEst_g` / `gameType_g` columns are near-empty. Stray `actionType` values have trailing whitespace (`"Rebound   …"`).

### Built: lineups and official matchups (2026-09-29)

- `scripts/build_on_court.py` -> `on_court.parquet` (**shared floor time**, not "guarded by"), 1996-97 to 2025-26, 38,016 games, ~75 s. Validation over 779k player-games: 99.5% within 2 min of box-score minutes, points match exactly on 99.96%, assists on 98.95%. 2019-20 early-season modern games have only `out` sub rows; the entering player is in `subsInPersonId`. Rows carry a `reliable` flag (95.3% true).
- `scripts/fetch_matchups.py` + `scripts/build_matchups.py` -> `matchups.parquet` (official BoxScoreMatchupsV3, 2017-18 on). Summed matchup FGA is ~108% of box FGA (a shot can be credited to more than one defender) and summed points ~96.5%, so use `partialPossessions` as the denominator.

### `PlayerStatisticsExtended.csv`: no matchup data, some lineup-adjacent signals

Every column is per player per game. None identify an opponent player.

| Column(s) | Relevance |
|---|---|
| `startingPosition` | Starter plus nominal position (G/F/C). Starting lineups per game come from this (also in `PlayerStatistics`). |
| `numMinutes`, `possessions` | On-court exposure. Needed to weight any lineup-based proxy. |
| `plusMinusPoints`, `offensiveRating`, `defensiveRating`, `netRating` (+ `estimated*`, `spWork*` variants) | On-court team efficiency while the player played. This is the input to on/off-style comparisons, but it can't isolate a single opponent. |
| `usagePercentage`, `percentTeam*` | Role and share of team production. Useful for "when Y's team plays, X's usage drops" style proxies. |
| `blocksAgainst`, `foulsAgainst` | How often the player was blocked or fouled. This is aggregate, and the defender is not identified. |
| `opponentPoints*` (`OffTurnovers`, `SecondChance`, `FastBreak`, `InPaint`) | Opponent scoring while the player was on court. Defensive context. |

Everything else (shooting splits, `percentAssisted*`, `pace*`, `playerImpactEstimate`, `doubleDouble`, `tripleDouble`) describes the player's own game and has no matchup value.

**Bottom line:** real "X vs defender Y" data is not in this dataset. Lineup reconstruction from play-by-play (with legacy-format name resolution) is the best honest proxy, and it should be a separate follow-up pipeline step.

## Full column listings

<details><summary><code>PlayerStatistics.csv</code>: all 40 columns</summary>

| column | type | null rate |
|---|---|---|
| `firstName` | VARCHAR | 0.0% |
| `lastName` | VARCHAR | 0.0% |
| `personId` | BIGINT | 0.0% |
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP | 0.2% |
| `playerteamCity` | VARCHAR | 0.0% |
| `playerteamName` | VARCHAR | 0.2% |
| `opponentteamCity` | VARCHAR | 0.2% |
| `opponentteamName` | VARCHAR | 0.2% |
| `gameType` | VARCHAR | 0.0% |
| `gameLabel` | VARCHAR | 93.0% |
| `gameSubLabel` | VARCHAR | 93.7% |
| `seriesGameNumber` | VARCHAR | 91.8% |
| `win` | BIGINT | 0.2% |
| `home` | BIGINT | 0.2% |
| `numMinutes` | VARCHAR | 10.1% |
| `points` | DOUBLE | 0.1% |
| `assists` | DOUBLE | 0.1% |
| `blocks` | DOUBLE | 0.1% |
| `steals` | DOUBLE | 0.1% |
| `fieldGoalsAttempted` | DOUBLE | 0.1% |
| `fieldGoalsMade` | DOUBLE | 0.1% |
| `fieldGoalsPercentage` | DOUBLE | 0.1% |
| `threePointersAttempted` | DOUBLE | 0.1% |
| `threePointersMade` | DOUBLE | 0.1% |
| `threePointersPercentage` | DOUBLE | 0.1% |
| `freeThrowsAttempted` | DOUBLE | 0.1% |
| `freeThrowsMade` | DOUBLE | 0.1% |
| `freeThrowsPercentage` | DOUBLE | 0.1% |
| `reboundsDefensive` | DOUBLE | 0.1% |
| `reboundsOffensive` | DOUBLE | 0.1% |
| `reboundsTotal` | DOUBLE | 0.1% |
| `foulsPersonal` | DOUBLE | 0.1% |
| `turnovers` | DOUBLE | 0.1% |
| `plusMinusPoints` | DOUBLE | 0.2% |
| `playerteamId` | BIGINT | 6.5% |
| `opponentteamId` | BIGINT | 6.5% |
| `comment` | VARCHAR | 91.0% |
| `startingPosition` | VARCHAR | 65.7% |
| `gameDate` | TIMESTAMP | 0.2% |

</details>

<details><summary><code>PlayerStatisticsExtended.csv</code>: all 110 columns</summary>

| column | type | null rate |
|---|---|---|
| `firstName` | VARCHAR | 0.0% |
| `lastName` | VARCHAR | 0.0% |
| `personId` | BIGINT | 0.0% |
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP | 0.4% |
| `gameType` | VARCHAR | 0.0% |
| `gameLabel` | VARCHAR | 92.6% |
| `gameSubLabel` | VARCHAR | 93.6% |
| `seriesGameNumber` | VARCHAR | 90.6% |
| `win` | BIGINT | 0.4% |
| `home` | BIGINT | 0.4% |
| `playerteamId` | BIGINT | 10.8% |
| `playerteamCity` | VARCHAR | 0.0% |
| `playerteamName` | VARCHAR | 0.4% |
| `opponentteamId` | BIGINT | 10.8% |
| `opponentteamCity` | VARCHAR | 0.4% |
| `opponentteamName` | VARCHAR | 0.4% |
| `comment` | VARCHAR | 100.0% |
| `startingPosition` | VARCHAR | 31.7% |
| `numMinutes` | VARCHAR | 0.1% |
| `points` | DOUBLE | 0.1% |
| `assists` | DOUBLE | 0.1% |
| `reboundsTotal` | DOUBLE | 0.1% |
| `reboundsOffensive` | DOUBLE | 0.1% |
| `reboundsDefensive` | DOUBLE | 0.1% |
| `fieldGoalsMade` | DOUBLE | 0.1% |
| `fieldGoalsAttempted` | DOUBLE | 0.1% |
| `fieldGoalsPercentage` | DOUBLE | 0.1% |
| `threePointersMade` | DOUBLE | 0.1% |
| `threePointersAttempted` | DOUBLE | 0.1% |
| `threePointersPercentage` | DOUBLE | 0.1% |
| `freeThrowsMade` | DOUBLE | 0.1% |
| `freeThrowsAttempted` | DOUBLE | 0.1% |
| `freeThrowsPercentage` | DOUBLE | 0.1% |
| `steals` | DOUBLE | 0.1% |
| `blocks` | DOUBLE | 0.1% |
| `blocksAgainst` | BIGINT | 0.0% |
| `turnovers` | DOUBLE | 0.1% |
| `foulsPersonal` | DOUBLE | 0.1% |
| `foulsAgainst` | BIGINT | 0.0% |
| `plusMinusPoints` | DOUBLE | 0.2% |
| `doubleDouble` | BIGINT | 0.0% |
| `tripleDouble` | BIGINT | 0.0% |
| `estimatedOffensiveRating` | DOUBLE | 0.0% |
| `offensiveRating` | DOUBLE | 0.0% |
| `spWorkOffensiveRating` | DOUBLE | 0.0% |
| `estimatedDefensiveRating` | DOUBLE | 0.0% |
| `defensiveRating` | DOUBLE | 0.0% |
| `spWorkDefensiveRating` | DOUBLE | 0.0% |
| `estimatedNetRating` | DOUBLE | 0.0% |
| `netRating` | DOUBLE | 0.0% |
| `spWorkNetRating` | DOUBLE | 0.0% |
| `assistPercentage` | DOUBLE | 0.0% |
| `assistToTurnoverRatio` | DOUBLE | 0.0% |
| `assistRatio` | DOUBLE | 0.0% |
| `offensiveReboundPercentage` | DOUBLE | 0.0% |
| `defensiveReboundPercentage` | DOUBLE | 0.0% |
| `reboundPercentage` | DOUBLE | 0.0% |
| `teamTurnoverPercentage` | DOUBLE | 0.0% |
| `estimatedTurnoverPercentage` | DOUBLE | 0.0% |
| `effectiveFieldGoalPercentage` | DOUBLE | 0.0% |
| `trueShootingPercentage` | DOUBLE | 0.0% |
| `usagePercentage` | DOUBLE | 0.0% |
| `estimatedUsagePercentage` | DOUBLE | 0.0% |
| `estimatedPace` | DOUBLE | 0.0% |
| `pace` | DOUBLE | 0.0% |
| `pacePer40` | DOUBLE | 0.0% |
| `spWorkPace` | DOUBLE | 0.0% |
| `playerImpactEstimate` | DOUBLE | 0.0% |
| `possessions` | BIGINT | 0.0% |
| `pointsOffTurnovers` | BIGINT | 0.0% |
| `pointsSecondChance` | BIGINT | 0.0% |
| `pointsFastBreak` | BIGINT | 0.0% |
| `pointsInPaint` | BIGINT | 0.0% |
| `opponentPointsOffTurnovers` | DOUBLE | 0.0% |
| `opponentPointsSecondChance` | DOUBLE | 0.0% |
| `opponentPointsFastBreak` | DOUBLE | 0.0% |
| `opponentPointsInPaint` | DOUBLE | 0.0% |
| `percentFieldGoalAttempts2Point` | DOUBLE | 0.0% |
| `percentFieldGoalAttempts3Point` | DOUBLE | 0.0% |
| `percentPoints2Point` | DOUBLE | 0.0% |
| `percentPoints2PointMidRange` | DOUBLE | 0.0% |
| `percentPoints3Point` | DOUBLE | 0.0% |
| `percentPointsFastBreak` | DOUBLE | 0.0% |
| `percentPointsFreeThrow` | DOUBLE | 0.0% |
| `percentPointsOffTurnovers` | DOUBLE | 0.0% |
| `percentPointsInPaint` | DOUBLE | 0.0% |
| `percentAssisted2PointMade` | DOUBLE | 0.0% |
| `percentUnassisted2PointMade` | DOUBLE | 0.0% |
| `percentAssisted3PointMade` | DOUBLE | 0.0% |
| `percentUnassisted3PointMade` | DOUBLE | 0.0% |
| `percentAssistedFieldGoalsMade` | DOUBLE | 0.0% |
| `percentUnassistedFieldGoalsMade` | DOUBLE | 0.0% |
| `percentTeamFieldGoalsMade` | DOUBLE | 0.0% |
| `percentTeamFieldGoalsAttempted` | DOUBLE | 0.0% |
| `percentTeamThreePointersMade` | DOUBLE | 0.0% |
| `percentTeamThreePointersAttempted` | DOUBLE | 0.0% |
| `percentTeamFreeThrowsMade` | DOUBLE | 0.0% |
| `percentTeamFreeThrowsAttempted` | DOUBLE | 0.0% |
| `percentTeamOffensiveRebounds` | DOUBLE | 0.0% |
| `percentTeamDefensiveRebounds` | DOUBLE | 0.0% |
| `percentTeamRebounds` | DOUBLE | 0.0% |
| `percentTeamAssists` | DOUBLE | 0.0% |
| `percentTeamTurnovers` | DOUBLE | 0.0% |
| `percentTeamSteals` | DOUBLE | 0.0% |
| `percentTeamBlocks` | DOUBLE | 0.0% |
| `percentTeamBlocksAgainst` | DOUBLE | 0.0% |
| `percentTeamFoulsPersonal` | DOUBLE | 0.0% |
| `percentTeamFoulsDrawn` | DOUBLE | 0.0% |
| `percentTeamPoints` | DOUBLE | 0.0% |

</details>

<details><summary><code>PlayByPlay.parquet</code>: all 89 columns</summary>

| column | type | null rate |
|---|---|---|
| `clock` | VARCHAR | 0.0% |
| `actionType` | VARCHAR | 0.0% |
| `description` | VARCHAR | 0.2% |
| `playerFullName` | VARCHAR | 8.9% |
| `teamTricode` | VARCHAR | 0.9% |
| `gameId` | VARCHAR | 0.0% |
| `gameDateTimeEst` | VARCHAR | 0.6% |
| `periodType` | VARCHAR | 72.8% |
| `period` | INTEGER | 0.0% |
| `shortFormattedClock` | VARCHAR | 99.8% |
| `timeActual` | VARCHAR | 72.8% |
| `actionNumber` | INTEGER | 0.0% |
| `orderNumber` | BIGINT | 72.8% |
| `actionId` | BIGINT | 27.2% |
| `teamId` | VARCHAR | 0.9% |
| `side` | VARCHAR | 91.4% |
| `location` | VARCHAR | 27.2% |
| `playerteamCity` | VARCHAR | 7.7% |
| `playerteamName` | VARCHAR | 7.7% |
| `opponentteamCity` | VARCHAR | 7.7% |
| `opponentteamName` | VARCHAR | 7.7% |
| `scoreHome` | INTEGER | 15.9% |
| `scoreAway` | INTEGER | 15.9% |
| `possession` | VARCHAR | 72.8% |
| `isTargetScoreLastPeriod` | BOOLEAN | 84.0% |
| `subType` | VARCHAR | 0.4% |
| `descriptor` | VARCHAR | 92.7% |
| `qualifiers` | VARCHAR[] | 72.8% |
| `edited` | VARCHAR | 72.8% |
| `value` | VARCHAR | 100.0% |
| `personId` | VARCHAR | 0.0% |
| `playerName` | VARCHAR | 2.1% |
| `playerNameI` | VARCHAR | 2.1% |
| `jerseyNumber` | VARCHAR | 99.8% |
| `x` | DOUBLE | 91.4% |
| `y` | DOUBLE | 91.4% |
| `xLegacy` | INTEGER | 18.6% |
| `yLegacy` | INTEGER | 18.6% |
| `area` | VARCHAR | 89.1% |
| `areaDetail` | VARCHAR | 89.1% |
| `isFieldGoal` | BOOLEAN | 0.9% |
| `shotDistance` | DOUBLE | 18.6% |
| `shotValue` | INTEGER | 27.2% |
| `shotResult` | VARCHAR | 16.4% |
| `pointsTotal` | INTEGER | 21.5% |
| `shotActionNumber` | INTEGER | 94.9% |
| `assistPersonId` | VARCHAR | 97.5% |
| `assistFullName` | VARCHAR | 97.6% |
| `assistPlayerName` | VARCHAR | 99.9% |
| `assistPlayerNameInitial` | VARCHAR | 97.6% |
| `assistTotal` | INTEGER | 97.5% |
| `reboundTotal` | INTEGER | 95.7% |
| `reboundDefensiveTotal` | INTEGER | 95.7% |
| `reboundOffensiveTotal` | INTEGER | 95.7% |
| `turnoverTotal` | INTEGER | 98.7% |
| `stealPersonId` | VARCHAR | 99.2% |
| `stealFullName` | VARCHAR | 99.3% |
| `stealPlayerName` | VARCHAR | 99.2% |
| `stealTotal` | INTEGER | 100.0% |
| `foulPersonalTotal` | INTEGER | 98.0% |
| `foulTechnicalTotal` | INTEGER | 98.0% |
| `foulDrawnPersonId` | VARCHAR | 98.1% |
| `foulDrawnFullName` | VARCHAR | 98.1% |
| `foulDrawnPlayerName` | VARCHAR | 98.1% |
| `blockPersonId` | VARCHAR | 99.5% |
| `blockFullName` | VARCHAR | 99.5% |
| `blockPlayerName` | VARCHAR | 99.5% |
| `blockTotal` | INTEGER | 100.0% |
| `jumpBallRecoveredName` | VARCHAR | 99.9% |
| `jumpBallRecoveredPersonId` | VARCHAR | 100.0% |
| `jumpBallRecoveredFullName` | VARCHAR | 99.9% |
| `jumpBallWonPersonId` | VARCHAR | 99.9% |
| `jumpBallWonFullName` | VARCHAR | 99.9% |
| `jumpBallWonPlayerName` | VARCHAR | 99.9% |
| `jumpBallLostPersonId` | VARCHAR | 99.9% |
| `jumpBallLostFullName` | VARCHAR | 99.9% |
| `jumpBallLostPlayerName` | VARCHAR | 99.9% |
| `subsInPersonId` | VARCHAR | 99.9% |
| `subsInFullName` | VARCHAR | 99.9% |
| `subsInPlayerName` | VARCHAR | 99.9% |
| `officialId` | VARCHAR | 97.3% |
| `videoAvailable` | BOOLEAN | 27.2% |
| `personIdsFilter` | VARCHAR[] | 72.8% |
| `gameType` | INTEGER | 100.0% |
| `playerteamId` | VARCHAR | 99.7% |
| `opponentteamId` | VARCHAR | 99.7% |
| `jumpBallRecoverdPersonId` | VARCHAR | 100.0% |
| `gameDateTimeEst_g` | VARCHAR | 99.7% |
| `gameType_g` | VARCHAR | 99.7% |

</details>

<details><summary><code>TeamStatistics.csv</code>: all 59 columns</summary>

| column | type | null rate |
|---|---|---|
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP | 0.0% |
| `teamCity` | VARCHAR | 0.0% |
| `teamName` | VARCHAR | 0.0% |
| `teamId` | BIGINT | 0.0% |
| `opponentTeamCity` | VARCHAR | 0.0% |
| `opponentTeamName` | VARCHAR | 0.0% |
| `opponentTeamId` | BIGINT | 0.0% |
| `home` | BIGINT | 0.0% |
| `win` | BIGINT | 0.0% |
| `teamScore` | BIGINT | 0.0% |
| `opponentScore` | BIGINT | 0.0% |
| `assists` | BIGINT | 0.0% |
| `blocks` | BIGINT | 0.0% |
| `steals` | BIGINT | 0.0% |
| `fieldGoalsAttempted` | BIGINT | 0.0% |
| `fieldGoalsMade` | BIGINT | 0.0% |
| `fieldGoalsPercentage` | DOUBLE | 0.0% |
| `threePointersAttempted` | BIGINT | 0.0% |
| `threePointersMade` | BIGINT | 0.0% |
| `threePointersPercentage` | DOUBLE | 0.0% |
| `freeThrowsAttempted` | BIGINT | 0.0% |
| `freeThrowsMade` | BIGINT | 0.0% |
| `freeThrowsPercentage` | DOUBLE | 0.0% |
| `reboundsDefensive` | BIGINT | 0.0% |
| `reboundsOffensive` | BIGINT | 0.0% |
| `reboundsTotal` | BIGINT | 0.0% |
| `foulsPersonal` | BIGINT | 0.0% |
| `turnovers` | BIGINT | 0.0% |
| `plusMinusPoints` | BIGINT | 0.0% |
| `numMinutes` | DOUBLE | 0.0% |
| `q1Points` | BIGINT | 5.4% |
| `q2Points` | BIGINT | 5.4% |
| `q3Points` | BIGINT | 5.4% |
| `q4Points` | BIGINT | 5.4% |
| `benchPoints` | BIGINT | 5.0% |
| `biggestLead` | BIGINT | 5.0% |
| `biggestScoringRun` | BIGINT | 5.0% |
| `leadChanges` | BIGINT | 5.0% |
| `pointsFastBreak` | BIGINT | 5.0% |
| `pointsFromTurnovers` | BIGINT | 5.0% |
| `pointsInThePaint` | BIGINT | 5.0% |
| `pointsSecondChance` | BIGINT | 5.0% |
| `timesTied` | BIGINT | 5.0% |
| `timeoutsRemaining` | BIGINT | 5.0% |
| `seasonWins` | BIGINT | 5.0% |
| `seasonLosses` | BIGINT | 5.0% |
| `coachId` | VARCHAR | 100.0% |
| `gameType` | VARCHAR | 5.2% |
| `gameLabel` | VARCHAR | 93.6% |
| `gameSubLabel` | VARCHAR | 93.7% |
| `seriesGameNumber` | BIGINT | 93.9% |
| `seed` | BIGINT | 94.6% |
| `reboundsTeam` | BIGINT | 5.2% |
| `turnoversTeam` | BIGINT | 5.2% |
| `ot1Points` | BIGINT | 94.7% |
| `ot2Points` | BIGINT | 99.2% |
| `otAllPoints` | BIGINT | 94.7% |
| `gameDate` | TIMESTAMP | 0.0% |

</details>

<details><summary><code>TeamStatisticsExtended.csv</code>: all 104 columns</summary>

| column | type | null rate |
|---|---|---|
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP | 0.0% |
| `gameType` | VARCHAR | 9.4% |
| `gameLabel` | VARCHAR | 93.4% |
| `gameSubLabel` | VARCHAR | 93.6% |
| `seriesGameNumber` | BIGINT | 94.1% |
| `teamId` | BIGINT | 0.0% |
| `teamCity` | VARCHAR | 0.0% |
| `teamName` | VARCHAR | 0.0% |
| `opponentTeamId` | BIGINT | 0.0% |
| `opponentTeamCity` | VARCHAR | 0.0% |
| `opponentTeamName` | VARCHAR | 0.0% |
| `home` | BIGINT | 0.0% |
| `win` | BIGINT | 0.0% |
| `teamScore` | BIGINT | 0.0% |
| `opponentScore` | BIGINT | 0.0% |
| `seed` | BIGINT | 94.0% |
| `numMinutes` | DOUBLE | 0.0% |
| `assists` | BIGINT | 0.0% |
| `steals` | BIGINT | 0.0% |
| `blocks` | BIGINT | 0.0% |
| `blocksAgainst` | BIGINT | 0.0% |
| `fieldGoalsMade` | BIGINT | 0.0% |
| `fieldGoalsAttempted` | BIGINT | 0.0% |
| `fieldGoalsPercentage` | DOUBLE | 0.0% |
| `threePointersMade` | BIGINT | 0.0% |
| `threePointersAttempted` | BIGINT | 0.0% |
| `threePointersPercentage` | DOUBLE | 0.0% |
| `freeThrowsMade` | BIGINT | 0.0% |
| `freeThrowsAttempted` | BIGINT | 0.0% |
| `freeThrowsPercentage` | DOUBLE | 0.0% |
| `reboundsOffensive` | BIGINT | 0.0% |
| `reboundsDefensive` | BIGINT | 0.0% |
| `reboundsTotal` | BIGINT | 0.0% |
| `reboundsTeam` | BIGINT | 9.4% |
| `foulsPersonal` | BIGINT | 0.0% |
| `personalFoulsDrawn` | BIGINT | 0.0% |
| `turnovers` | BIGINT | 0.0% |
| `turnoversTeam` | BIGINT | 9.4% |
| `plusMinusPoints` | BIGINT | 0.1% |
| `q1Points` | BIGINT | 9.1% |
| `q2Points` | BIGINT | 9.1% |
| `q3Points` | BIGINT | 9.1% |
| `q4Points` | BIGINT | 9.1% |
| `ot1Points` | BIGINT | 94.6% |
| `ot2Points` | BIGINT | 99.2% |
| `otAllPoints` | BIGINT | 94.6% |
| `benchPoints` | BIGINT | 9.1% |
| `biggestLead` | BIGINT | 9.1% |
| `biggestScoringRun` | BIGINT | 9.1% |
| `leadChanges` | BIGINT | 9.1% |
| `pointsFastBreak` | BIGINT | 9.1% |
| `pointsFromTurnovers` | BIGINT | 9.1% |
| `pointsInThePaint` | BIGINT | 9.1% |
| `pointsSecondChance` | BIGINT | 9.1% |
| `timesTied` | BIGINT | 9.1% |
| `timeoutsRemaining` | BIGINT | 9.1% |
| `seasonWins` | BIGINT | 9.1% |
| `seasonLosses` | BIGINT | 9.1% |
| `estimatedOffensiveRating` | DOUBLE | 0.0% |
| `offensiveRating` | DOUBLE | 0.0% |
| `estimatedDefensiveRating` | DOUBLE | 0.0% |
| `defensiveRating` | DOUBLE | 0.0% |
| `estimatedNetRating` | DOUBLE | 0.0% |
| `netRating` | DOUBLE | 0.0% |
| `assistPercentage` | DOUBLE | 0.0% |
| `assistToTurnoverRatio` | DOUBLE | 0.0% |
| `assistRatio` | DOUBLE | 0.0% |
| `offensiveReboundPercentage` | DOUBLE | 9.8% |
| `defensiveReboundPercentage` | DOUBLE | 9.8% |
| `reboundPercentage` | DOUBLE | 18.1% |
| `teamTurnoverPercentage` | DOUBLE | 0.0% |
| `effectiveFieldGoalPercentage` | DOUBLE | 0.0% |
| `trueShootingPercentage` | DOUBLE | 0.0% |
| `estimatedPace` | DOUBLE | 0.0% |
| `pace` | DOUBLE | 0.0% |
| `pacePer40` | DOUBLE | 0.0% |
| `possessions` | BIGINT | 0.0% |
| `playerImpactEstimate` | DOUBLE | 0.0% |
| `pointsOffTurnovers` | DOUBLE | 0.0% |
| `opponentPointsOffTurnovers` | DOUBLE | 0.0% |
| `opponentPointsSecondChance` | DOUBLE | 0.0% |
| `opponentPointsFastBreak` | DOUBLE | 0.0% |
| `opponentPointsInPaint` | DOUBLE | 0.0% |
| `percentFieldGoalAttempts2Point` | DOUBLE | 0.0% |
| `percentFieldGoalAttempts3Point` | DOUBLE | 0.0% |
| `percentPoints2Point` | DOUBLE | 0.0% |
| `percentPoints2PointMidRange` | DOUBLE | 0.0% |
| `percentPoints3Point` | DOUBLE | 0.0% |
| `percentPointsFastBreak` | DOUBLE | 0.0% |
| `percentPointsFreeThrow` | DOUBLE | 0.0% |
| `percentPointsOffTurnovers` | DOUBLE | 0.0% |
| `percentPointsInPaint` | DOUBLE | 0.0% |
| `percentAssisted2PointMade` | DOUBLE | 0.0% |
| `percentUnassisted2PointMade` | DOUBLE | 0.0% |
| `percentAssisted3PointMade` | DOUBLE | 0.0% |
| `percentUnassisted3PointMade` | DOUBLE | 0.0% |
| `percentAssistedFieldGoalsMade` | DOUBLE | 0.0% |
| `percentUnassistedFieldGoalsMade` | DOUBLE | 0.0% |
| `freeThrowAttemptRate` | DOUBLE | 0.0% |
| `opponentEffectiveFieldGoalPercentage` | DOUBLE | 0.0% |
| `opponentFreeThrowAttemptRate` | DOUBLE | 0.0% |
| `opponentTurnoverPercentage` | DOUBLE | 0.0% |
| `opponentOffensiveReboundPercentage` | DOUBLE | 9.8% |

</details>

<details><summary><code>Games.csv</code>: all 23 columns</summary>

| column | type | null rate |
|---|---|---|
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP | 0.0% |
| `hometeamCity` | VARCHAR | 0.0% |
| `hometeamName` | VARCHAR | 0.0% |
| `hometeamId` | BIGINT | 0.0% |
| `awayteamCity` | VARCHAR | 0.0% |
| `awayteamName` | VARCHAR | 0.0% |
| `awayteamId` | BIGINT | 0.0% |
| `homeScore` | BIGINT | 0.0% |
| `awayScore` | BIGINT | 0.0% |
| `winner` | BIGINT | 0.0% |
| `gameType` | VARCHAR | 0.0% |
| `gameSubtype` | VARCHAR | 99.9% |
| `gameLabel` | VARCHAR | 94.5% |
| `gameSubLabel` | VARCHAR | 99.6% |
| `seriesGameNumber` | VARCHAR | 92.1% |
| `attendance` | BIGINT | 98.1% |
| `arenaId` | BIGINT | 0.0% |
| `arenaName` | VARCHAR | 98.1% |
| `arenaCity` | VARCHAR | 98.1% |
| `arenaState` | VARCHAR | 98.1% |
| `officials` | VARCHAR | 98.1% |
| `gameDate` | TIMESTAMP | 0.0% |

</details>

<details><summary><code>Players.csv</code>: all 20 columns</summary>

| column | type | null rate |
|---|---|---|
| `personId` | BIGINT | 0.0% |
| `firstName` | VARCHAR | 0.0% |
| `lastName` | VARCHAR | 0.0% |
| `birthDate` | DATE | 17.0% |
| `school` | VARCHAR | 21.1% |
| `country` | VARCHAR | 20.9% |
| `heightInches` | BIGINT | 23.7% |
| `bodyWeightLbs` | BIGINT | 23.8% |
| `jersey` | VARCHAR | 42.9% |
| `guard` | BIGINT | 0.0% |
| `forward` | BIGINT | 0.0% |
| `center` | BIGINT | 0.0% |
| `dleagueFlag` | BIGINT | 18.4% |
| `nbaFlag` | BIGINT | 18.4% |
| `gamesPlayedFlag` | BIGINT | 18.4% |
| `draftYear` | BIGINT | 17.0% |
| `draftRound` | BIGINT | 19.4% |
| `draftNumber` | BIGINT | 20.3% |
| `fromYear` | BIGINT | 20.0% |
| `toYear` | BIGINT | 28.7% |

</details>

<details><summary><code>TeamHistories.csv</code>: all 7 columns</summary>

| column | type | null rate |
|---|---|---|
| `teamId` | BIGINT | 0.0% |
| `teamCity` | VARCHAR | 0.0% |
| `teamName` | VARCHAR | 0.0% |
| `teamAbbrev` | VARCHAR | 0.0% |
| `seasonFounded` | BIGINT | 0.0% |
| `seasonActiveTill` | BIGINT | 0.0% |
| `league` | VARCHAR | 0.0% |

</details>

<details><summary><code>LeagueSchedule24_25.csv</code>: all 15 columns</summary>

| column | type | null rate |
|---|---|---|
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP WITH TIME ZONE | 0.0% |
| `gameDay` | VARCHAR | 0.0% |
| `arenaCity` | VARCHAR | 0.0% |
| `arenaState` | VARCHAR | 0.6% |
| `arenaName` | VARCHAR | 0.0% |
| `gameLabel` | VARCHAR | 82.5% |
| `gameSubLabel` | VARCHAR | 88.2% |
| `gameSubtype` | VARCHAR | 94.5% |
| `gameSequence` | BIGINT | 0.0% |
| `seriesGameNumber` | BIGINT | 94.0% |
| `seriesText` | VARCHAR | 92.7% |
| `weekNumber` | BIGINT | 0.0% |
| `hometeamId` | BIGINT | 0.0% |
| `awayteamId` | BIGINT | 0.0% |

</details>

<details><summary><code>LeagueSchedule25_26.csv</code>: all 17 columns</summary>

| column | type | null rate |
|---|---|---|
| `gameId` | BIGINT | 0.0% |
| `gameDateTimeEst` | TIMESTAMP | 0.0% |
| `gameDay` | VARCHAR | 0.0% |
| `homeTeamId` | BIGINT | 0.0% |
| `awayTeamId` | BIGINT | 0.0% |
| `homeTeamName` | VARCHAR | 0.0% |
| `homeTeamCity` | VARCHAR | 0.5% |
| `awayTeamName` | VARCHAR | 0.0% |
| `awayTeamCity` | VARCHAR | 0.5% |
| `arenaName` | VARCHAR | 0.0% |
| `arenaCity` | VARCHAR | 0.0% |
| `arenaState` | VARCHAR | 0.6% |
| `gameLabel` | VARCHAR | 82.1% |
| `gameSubLabel` | VARCHAR | 87.9% |
| `gameSubtype` | VARCHAR | 94.7% |
| `seriesGameNumber` | VARCHAR | 93.9% |
| `weekNumber` | BIGINT | 0.0% |

</details>

