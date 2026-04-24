# ABS Walk Spike — Round 1

Two weeks ago we wrote that the 2026 walk-rate spike was pitchers nibbling, not umpires squeezing the zone. The Associated Press and ESPN ran stories on April 23, 2026 that quoted players blaming the new ABS-defined zone. We rebuilt the analysis with two independent ML pipelines, hit a 96-point sign disagreement on the headline counterfactual, ran a third implementation as a tiebreaker, and shipped an updated answer within 24 hours.

**Published article:** [ABS Took the High Strike — and That's Roughly 40-50% of the Walk Spike. Pitchers Own the Rest.](https://calledthird.com/analysis/abs-walk-spike-zone-correction) (April 23, 2026)

**Prior position (now updated):** [The Walk Rate Spike: Umpires or Pitchers?](https://calledthird.com/analysis/the-walk-rate-spike) (April 9, 2026 — concluded "pitchers, not umpires." This Round 1 piece is the honest update.)

## The question

MLB walk rate hit 9.77% over Mar 27 – Apr 22, 2026 — the highest since 1950. Players blame the new ABS-era zone. Two weeks ago we said it was pitchers, not the zone. Did the zone actually shrink, where, and how much of the +0.82pp YoY walk-rate spike does it explain?

## The approach

Two agents analyzed the same brief in parallel using deliberately divergent methodologies:

- **Agent A (Claude)** — interpretability-first: 2D grid binning with bootstrap CIs, spline-smoothed difference surface, logistic GAM with a season-by-zone interaction term, time-series Z-score, stratified count tests
- **Agent B (Codex)** — ML-engineering: LightGBM year-classifier with SHAP attribution, regularized polynomial-2 logistic zone classifiers per season, learned-counterfactual walk-rate replay

Each agent then peer-reviewed the other under the rule "we publish if and only if both agents bless." Claude's review of Codex surfaced a Statcast schema artifact in 2026 (per-batter `sz_top`/`sz_bot` switched from per-pitch posture estimates to deterministic ABS-rule values) that broke Codex's normalized-coordinate analysis. Codex's review of Claude surfaced a multiple-testing concern on the bin-grid scan and a counter-finding that the 2026 zone is more strike-friendly at 0-0 — pushing back on Claude's first-pitch causal chain.

The disagreement landed at a fundamental level: Codex's normalized-coord counterfactual said the zone change accounts for **−56.17%** of the walk spike (myth-bust). Claude's bin-level absolute-coord heatmap said the zone shrank at the top edge by 22pp. We ran a focused **adjudication round** in absolute plate coordinates to resolve the dispute, then triangulated with a third clean implementation and a first-principles aggregate diagnostic. Both agents then issued written publish-readiness reviews on the resolved synthesis.

## Key findings

1. **Walk spike is real and not seasonality.** 2026 walk rate (9.77% incl. IBB, Mar 27 – Apr 22) is **+4.4σ** above the 2018–2025 historical mean of 9.02% (SD 0.17pp). It sits +0.60pp above the prior 8-year April maximum.
2. **The 2026 zone moved up.** In absolute plate coordinates: ~7–8pp called-strike rate drop on a large region at the top edge (z ≈ 3.2–3.9 ft), ~5–6pp expansion on a smaller region at the bottom edge (z ≈ 1.0–2.0 ft). Multiple-testing-uncorrected at the bin level, but corroborated by the counterfactual which doesn't depend on the bin grid.
3. **Roughly 40–50% of the +0.82pp walk spike is the zone change.** Two independent counterfactual implementations: Codex's PA-replay produces +40.46% attribution; a third clean implementation produces +49.40%. We report the range. The remaining 50–60% reflects unmodeled pitcher behavior, pitch-mix shifts, and adaptations.
4. **3-2 is NOT where the damage concentrates.** Walk-rate delta at 3-2 is −0.11pp (CI [−2.6, +2.4]). Cochran's Q across all 12 count states: p = 0.67. No per-count concentration.
5. **The Apr 9 piece's "pitchers, not umpires" framing was too strong.** It correctly identified pitcher behavior as the larger share, but dismissed the zone effect at zero. The zone effect is real and roughly half the spike. Honest correction in the published article.

## The methodology story (what makes this project unusual)

Round 1 surfaced a real divergence between the two agents on the headline counterfactual sign — Codex initially returned −56% attribution while Claude's geometry suggested +40%+. Rather than picking a side, we ran an explicit adjudication round:

- Each agent wrote a peer review of the other (`reviews/`)
- Both agents reran the counterfactual in absolute plate coordinates with no `sz_*` features
- A third clean implementation (`scripts/clean_counterfactual.py`) was written from scratch as a triangulation check
- A first-principles aggregate diagnostic (`scripts/debug_counterfactual.py`) confirmed the direction
- The orchestrator's first independent reimplementation (`scripts/adjudicate_absolute_coords.py`) had a sign-flipping bug that we caught — the buggy script is included in the release for transparency
- Both agents then issued publish-readiness reviews on the resolved synthesis (`reviews/claude-publish-readiness.md`, `reviews/codex-publish-readiness.md`) listing the specific framing changes the article had to make before they would bless publication

The published article incorporates every framing concession both reviewers asked for. The full editorial reasoning chain is in `COMPARISON_MEMO.md`. The full technical record is in `ADJUDICATION_SUMMARY.md`.

## Files

| Path | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | The original brief: hypotheses, data, methodology mandates per agent, branch criteria |
| `ADJUDICATION_SUMMARY.md` | Canonical technical record of how the dispute was resolved (v2 reflects publish-readiness corrections) |
| `COMPARISON_MEMO.md` | Editorial-facing synthesis: convergent claims, divergent claims and resolution, what gets published, open questions for next round |
| `PRIOR_ART.md` | Citation map of the 7+ prior CalledThird ABS pieces this article builds on or updates |
| `claude-analysis/` | Agent A's full Round 1 work (analyze.py + module scripts, REPORT.md, READY_FOR_REVIEW.md, findings.json) |
| `codex-analysis/` | Agent B's full Round 1 work + adjudication.py with absolute-coord rerun |
| `reviews/` | Cross-reviews and publish-readiness reviews from both agents, plus orchestrator-side adjudication results |
| `scripts/` | Orchestrator data prep (`build_2026_master.py`, `fetch_april_history.py`) and adjudication scripts (`adjudicate_absolute_coords.py` — buggy, included for transparency; `clean_counterfactual.py` — third independent implementation; `debug_counterfactual.py` — aggregate sanity check) |
| `data/` | Small derived data: substrate_summary.json (pre-validated baseline numbers) and april_walk_history.csv (2018–2025 April aggregates). Statcast parquets are NOT included — pull them via pybaseball using the scripts in `scripts/` |
| `requirements.txt` | Python dependencies |

## Reproducing the analysis

```bash
pip install -r requirements.txt

# Step 1: pull and assemble the data substrate (Statcast pitch-by-pitch)
python scripts/build_2026_master.py     # → data/statcast_2026_mar27_apr22.parquet
python scripts/fetch_april_history.py   # → data/april_walk_history.csv (already included)

# Step 2: run each agent's analysis independently
python claude-analysis/analyze.py
python codex-analysis/analyze.py

# Step 3: run the absolute-coord adjudication (post-cross-review)
python codex-analysis/adjudication.py        # Codex's abs-coord rerun (+40.46%)
python scripts/clean_counterfactual.py       # Third independent implementation (+49.40%)
python scripts/debug_counterfactual.py       # Aggregate first-principles sanity check
```

The 2025 same-window Statcast file (`statcast_2025_mar27_apr14.parquet`) is referenced from the `count-distribution-abs/` project (a prior CalledThird research project). For self-contained reproduction, pull it via `pybaseball.statcast(start_dt='2025-03-27', end_dt='2025-04-14')`.

## Caveats

The article (and `ADJUDICATION_SUMMARY.md`) lists eight specific caveats explicitly. Two are particularly worth flagging here:

1. **The 40–50% attribution is a zone-only counterfactual.** It assumes pitchers and batters wouldn't have changed behavior under the 2025 zone. The estimate is therefore a lower bound on the zone-change effect (and an upper bound on pitcher adaptation as a residual).
2. **First-pitch / 0-0-only counterfactual flips negative** (−20% to −42% across implementations). The 2026 zone is more strike-friendly on first pitches but less strike-friendly in late counts at the top edge. We have not formally decomposed this tension by count, and we acknowledge it as the largest unresolved gap. Round 2 will address it.

## Round 2 (planned, +3 to +7 days)

- Decompose the all-pitches counterfactual by count to mechanistically explain the first-pitch sign flip
- Pitcher response model: are pitchers throwing differently in response to the new zone? (Pitch-type mix, location-distribution shift)
- Per pitch type (fastball vs breaking vs offspeed): which classes are losing the most strike calls?

A Round 3 will then handle per-umpire, per-team, and catcher-framing breakdowns.
