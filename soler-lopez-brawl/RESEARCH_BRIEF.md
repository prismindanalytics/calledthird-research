# Research Brief: Do Umpires Call Differently After Bench-Clearing Incidents?

## Hypothesis
After bench-clearing incidents, umpires call a wider/tighter zone in the subsequent game(s) to assert control, resulting in measurable changes in called-strike rate and borderline pitch outcomes.

## The Hook
Jorge Soler and Reynaldo Lopez were just suspended (7 and 5 games respectively) for their brawl in the Angels-Braves game on April 7, 2026. Baseball Twitter is debating it. CalledThird can ask the question nobody else is asking: does the umpire's zone change after a brawl?

## Approach
For each bench-clearing incident:
1. Identify the **incident game** (game_pk, date, umpire)
2. Identify the **next game** the same umpire worked after the incident
3. Pull called-pitch data for both games
4. Compare the umpire's zone metrics (accuracy, borderline strike rate, called-strike rate) between:
   - The post-incident game vs their season baseline
   - Post-incident games across all incidents vs a matched control sample

## Kill Criteria
- If post-incident called-strike rates are within 1pp of baseline → KILL the "umpires change behavior" angle
- Story becomes: "Umpires Don't Flinch" (myth-bust, equally publishable)

## Historical Incidents (Statcast Era, 2015-2026)

All incidents below resulted in suspensions, confirming they were significant bench-clearing events.

| # | Date | Teams | Key Players | Game Context |
|---|------|-------|-------------|-------------|
| 1 | 2026-04-07 | LAA vs ATL | Soler/Lopez | Soler charged mound after HBP sequence; punches thrown |
| 2 | 2024-05-?? | MIL vs TB | Peralta/Siri | Peralta HBP Siri; brawl; 4 suspensions (Uribe 6G, Peralta 5G, Siri 3G) |
| 3 | 2023-08-05 | CWS vs CLE | Anderson/Ramirez | Anderson/Ramirez exchange; 6 ejections, Anderson 6G, Ramirez 3G |
| 4 | 2022-06-26 | SEA vs LAA | Winker/Rendon/Nevin | Wantz HBP Winker; massive brawl; 12 suspensions, 47 total games |
| 5 | 2022-04-27 | STL vs NYM | Arenado/Cabrera | Lopez threw near Arenado's head; Arenado 2G, Cabrera 1G |
| 6 | 2020-08-?? | OAK vs HOU | Laureano/Castellanos | Laureano charged mound after multiple HBPs; 6G suspension |
| 7 | 2019-07-30 | PIT vs CIN | Kela/Garrett/Puig | Massive brawl; 8 suspended, 40 total games; Kela 10G |

## Data Needed Per Incident

For each incident, we need:
1. **Incident game**: game_pk, date, home plate umpire
2. **Next game same umpire worked**: game_pk, date (usually 4-5 days later in rotation)
3. **Called pitches for both games**: plate_x, plate_z, sz_top, sz_bot, description, zone_dist
4. **Umpire's season baseline**: accuracy, borderline strike rate, called-strike rate (from their other games that season)
5. **Control games**: Random non-incident games by the same umpire for comparison

## Data Sources
- **Statcast** (pybaseball): Pitch-level data with plate location and call outcomes
- **Retrosheet/Baseball Reference**: Umpire game assignments
- **MLB Stats API**: Umpire schedule, game lineup
- **CalledThird D1**: 2025-2026 called_pitches table (already has accuracy data)

## Files in This Folder

- `RESEARCH_BRIEF.md` — This file
- `data/incidents.json` — Compiled incident list with game_pks and umpire IDs
- `data/incident_games/` — Statcast data for each incident game
- `data/next_games/` — Statcast data for the umpire's next game after each incident
- `data/baselines/` — Season baseline data for each umpire involved

## Expected Output
- Is there a statistically significant difference in zone metrics after incidents?
- If yes: which direction (wider? tighter?) and by how much?
- If no: "Umpires Don't Flinch" myth-bust with confidence intervals

## Article Framing
- **If confirmed**: "After the Brawl: How Umpires Quietly Adjust the Zone Following Bench-Clearing Incidents"
- **If rejected**: "Umpires Don't Flinch: The Zone Stays the Same After a Fight" (classic CalledThird myth-bust — equally strong)
