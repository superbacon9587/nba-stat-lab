# Parser comparison (2026-09-30, claude-haiku-4-5)

22/27 prompts agree. Claude calls: 26 ok, 1 failed.
Tokens: 11652 uncached in, 0 cache-write, 129506 cache-read, 20137 out.
Cost ~$0.1253 (would be ~$0.2418 without caching). Median latency 4.9s.

| Prompt | Agree | Latency | Cache read | Differences (claude vs rules) |
|---|---|---|---|---|
| What is Stephen Curry's ppg against LeBron as a defender | yes | 4.9s | 4981 |  |
| What is Steph Curry's apg against the Spurs | yes | 4.5s | 4981 |  |
| What is the avg ppg for the Boston Celtics in the last 5 years playing against the Lakers | yes | 4.9s | 4981 |  |
| How might Curry perform against a 6'7 defender | yes | 4.8s | 4981 |  |
| Curry is playing against Philly in San Francisco, his defender is LeBron. Expected points and assists? Chances he exceeds 27.5 points and 5.5 assists? | yes | 5.0s | 4981 |  |
| How does day of week affect Jayson Tatum's rebounds | yes | 4.6s | 4981 |  |
| How do back-to-backs affect scoring for all guards | yes | 4.8s | 4981 |  |
| Knicks points at home vs away on Sundays since 2020 | yes | 6.2s | 4981 |  |
| Nikola Jokic points at OKC, line 28.5 | no | 6.2s | 4981 | projection: claude={"lines": {"points": 28.5}, "context": {"opponent_team_id": 1610612760, "venue_team_id": null, "defender_person_id": null, "home_away": null, "rest_days": null, "back_to_back": null, "projected_minutes": null}} \| rules={"lines": {"points": 28.5}, "context": {"opponent_team_id": 1610612760, "venue_team_id": 1610612760, "defender_person_id": null, "home_away": "away", "rest_days": null, "back_to_back": null, "projected_minutes": null}} |
| Victor Wembanyama blocks vs the Warriors, line 3.5 | yes | 4.8s | 4981 |  |
| Kobe Bryant ppg vs the Celtics in the playoffs | yes | 5.0s | 4981 |  |
| Seattle SuperSonics points per game in 2005 | yes | 6.6s | 4981 |  |
| Anthony Edwards threes on back-to-backs, line 3.5 | yes | 4.8s | 4981 |  |
| Lakers team threes at home, line 13 | no | 4.6s | - | LLMUnavailable: Claude's tool call did not match the schema: 1 validation error for QueryDraft
relative_season
  Input should be 'this' or 'last' [type=literal_error, input_value='null', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/literal_error |
| How many rebounds does Giannis average against the Celtics since 2021? | yes | 4.5s | 4981 |  |
| Does Luka Doncic score more on the road? | no | 6.0s | 4981 | filters: claude=[] \| rules=[{"type": "home_away", "value": "away"}]<br>group_by: claude="home_away" \| rules=null |
| Tyrese Haliburton assists in his last 15 games | yes | 6.3s | 4981 |  |
| Will Shai score 30+ points against Denver next game? | yes | 6.3s | 4981 |  |
| Warriors points allowed at home this season | yes | 4.5s | 4981 |  |
| How does Joel Embiid shoot from three when he has 2+ days of rest | no | 4.5s | 4981 | filters: claude=[{"type": "rest_days", "min_days": 2, "max_days": 99}] \| rules=[]<br>stats: claude=["three_pct"] \| rules=["points"] |
| before vs after the all star break | yes | 4.8s | 4981 |  |
| Celtics after the all star break | yes | 4.6s | 4981 |  |
| how do teams change after the all star break | yes | 6.0s | 4981 |  |
| Tatum's scoring leading up to the playoffs | no | 6.3s | 4981 | filters: claude=[{"type": "season_range", "start": 2025, "end": 2025}] \| rules=[] |
| compare team ppg from months 10,11,12,1 to months 2,3,4,5,6 | yes | 6.4s | 4981 |  |
| who improves most after the break | yes | 6.0s | 4981 |  |
| Lakers net rating last 20 games before playoffs | yes | 6.0s | 4981 |  |
