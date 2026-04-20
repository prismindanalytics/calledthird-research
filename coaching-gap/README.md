# The Coaching Gap

Six rounds of dual-agent research on ~2.9M pitches across five MLB seasons (2022–2026), asking a simple question: when an MLB pitcher is predictable, who actually cashes the check?

**Published articles:**
- [The Coaching Gap That Lives Where Hitters Don't Chase](https://calledthird.com/analysis/coaching-gap-patience) — Findings, chase-tertile hero chart, quality-hitter list
- [Which Pitchers Can You Predict?](https://calledthird.com/analysis/which-pitchers-can-you-predict) — The Layer 2 pitch-prediction model that powers this study

**Live data:** [calledthird.com/explore#coaching-gap](https://calledthird.com/explore#coaching-gap) — batter chase trajectories, pitcher predictability ledger, team trends.

## The Question

Pitcher predictability is measurable — we can score every pitch by P(pitch type | count, sequence, handedness, context) and flag the top-quintile as "we know what's coming." Across 401K of those predictable terminal pitches in 2022–2025, wOBA is ~0.092 lower than on unpredictable ones. That's the pooled coaching gap.

The question was never *does the gap exist*. The question was *where does it live* — who extracts it and who doesn't. If the answer is "everyone, roughly equally," the coaching story collapses into mechanics. If it concentrates on a specific hitter archetype, there's a real coaching surface.

## The Approach

Two agents analyzed the same pre-registered substrate independently:

- **Claude** — between-batter Bayesian framework with pooled and batter-fixed-effects estimators
- **Codex** — within-batter matched-pairs framework with season-stratified bootstraps and permutation tests

Each round had a written-down kill criterion. Neither agent could see the other's code until the cross-review at the end of each round. The reviewers flagged each other's methodology errors; the next round's brief resolved them.

Round 6 forced both agents onto the **same substrate** (`layer2_holdout_predictions.parquet`), the **same tertile definitions**, and the **same three estimators** (pooled-between, matched-pairs, batter-fixed-effects) so every remaining disagreement had to decompose to method, not data.

## Key Findings

1. **One of 17 hypotheses survived: chase-tertile spread of ~0.04 wOBA.** Low-chase hitters extract far more wOBA edge from predictable pitches than high-chase hitters do (Claude pooled: −0.040 wOBA; Codex pooled: −0.040; Claude FE: −0.050; Codex FE: −0.050; matched-pairs: −0.006 to −0.017 with lower power). Permutation p < 0.01.
2. **Power, not contact, not pitch type.** Tests on xwOBA-on-contact, whiff rate, zone-contact, TTO, count state, pitch family, stuff tier, team, batter archetype, and arsenal breadth all returned null or diagnostic-confounded spreads.
3. **Seasonal replication survives strict 2026 holdout.** The Layer 2 model was trained on 2022–2024 and scored 2025/2026 out-of-sample. The chase-tertile gap replicates in every season including the 2026 opening month.
4. **Quality-hitter refinement (+0.025 wOBA).** Hitters in the low-chase × high-xwOBA-contact quadrant (36 names in 2025 — Soto, Judge, Schwarber, Olson, Lindor, Wood...) extract an additional +0.025 wOBA beyond the league baseline, confirmed under both pooled and matched-pairs. This is a refinement of the chase finding, not an independent effect.
5. **Power simulation reconciled the two methods.** At observed N, matched-pairs has ~40% power to detect the true spread while pooled-between has 85%+. The matched-pairs attenuation is a known power loss, not a contradiction.

## Files

| File | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | Original hypothesis, substrate design, Layer 1 / Layer 2 framework, kill criteria |
| `METHODOLOGY.md` | The 17-hypothesis scoreboard across six rounds: what we tested, what died, why |
| `COMPARISON_MEMO.md` | Where Claude and Codex converged, where they diverged, and how Round 6 resolved the divergence |
| `analyze_claude.py` | Claude's Round 6 reconciliation analysis (H1–H5) as one self-contained script |
| `analyze_codex.py` | Codex's Round 6 reconciliation analysis (H1–H5) as one self-contained script |
| `requirements.txt` | Python dependencies |

## Dependencies

```bash
pip install -r requirements.txt
```

## Reproducing

1. Pull 2022–2026 Statcast pitches:
   ```python
   from pybaseball import statcast
   for year in range(2022, 2027):
       df = statcast(start_dt=f'{year}-03-15', end_dt=f'{year}-11-15')
       df.to_parquet(f'data/statcast_{year}.parquet')
   ```

2. Build the shared substrate — the Layer 2 pitch-prediction model and the terminal-pitch parquet. This is a ~90-minute pipeline; both agents consumed the same canonical output file `data/agent_substrate/career_pitches.parquet` and `data/layer2_holdout_predictions.parquet`. See the `## Substrate` section of `RESEARCH_BRIEF.md` for the full build.

3. Run either analysis script:
   ```bash
   python analyze_claude.py --data-dir data/
   python analyze_codex.py  --data-dir data/
   ```

   Each prints an H1–H5 summary and writes `findings.json` with point estimates, bootstrap CIs, and permutation p-values.

## Caveats

- **Between-batter, not tonight's-matchup.** The coaching gap is a season-aggregate structural feature. It does *not* predict individual matchup outcomes; a backtest on ~70K 2026 matchups showed Spearman ≈ 0 between predicted and realized wOBA at the matchup level. This study validates the mechanism; it does not sell a tonight's-slate edge.
- **Predictability is Layer 2-defined.** The "predictable" flag is top-quintile of `context_prob` from the Layer 2 model. Different model families would produce modestly different flags; the headline survives on 80/20, 75/25, and within-pitcher vs league-wide cuts (Round 6 H4).
- **The cohort is 371 pre-2026 pitchers** with sufficient career footprint for the Layer 1 archetype work. 2026 replication uses the same 371 + any qualifying call-ups.

## License

MIT. The data is public (MLB Statcast via pybaseball); the analysis is released for anyone to reproduce, extend, or dispute.
