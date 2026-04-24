# ABS Walk Spike — Research Brief

**Status:** Round 1 — agents launching
**Target deliverable:** Multi-article CalledThird package, 24h Round 1, 1-week full sequence
**News hook expiration:** ~5-7 days (story going national right now via AP wire pickup)
**Principal risk:** None catastrophic — both "zone shrunk" and "zone didn't shrink" outcomes are publishable.

---

## 0. Prior CalledThird coverage

See `PRIOR_ART.md` for the full citation map. Key prior piece: **"The Walk Rate Spike: Umpires or Pitchers?" (April 9, 2026)** argued — based on 10 days of data — that the walk spike is **pitchers nibbling, not umpires squeezing**. Shadow-zone called-strike rate had actually ROSEN +4pp (65.7% → 69.7%). This new investigation is, in part, a 17-days-more-data update to that prior position. Agents do not need to act on this — the editorial layer (citation, comparison, position-update) is added at synthesis time. But agents may read `PRIOR_ART.md` if useful.

## 1. The Question

**Did the strike zone actually shrink under MLB's new ABS-defined zone in 2026, where on the plate did it shrink, and how much of the historic walk-rate spike does that shrinkage explain?**

Triggered by AP/ESPN coverage on April 23, 2026 reporting MLB's walk rate at 9.8% — highest since 1950 — with players (Sewald, McCann) blaming a "smaller strike zone" and Hoerner specifically pointing to hitters laying off pitches at the **top** of the zone. League BA is roughly flat at .240 vs .242 prior year same window.

**Validated YoY baseline (computed Apr 23, 2026, see `data/substrate_summary.json`):**
- 2025 same-window (Mar 27–Apr 14) walk rate, incl IBB: **9.11%**
- 2026 same-window (Mar 27–Apr 14) walk rate, incl IBB: **9.92%**
- 2026 full window (Mar 27–Apr 22) walk rate, incl IBB: **9.77%** (matches ESPN's 9.8%)
- **YoY delta: +0.82 percentage points (+9.0% relative)** — the spike is real

**Validated H2 (seasonality control, 2018-2025 Mar 27-Apr 22, incl IBB):**
- Historical mean: **9.02%**, SD: **0.17pp**, range: 8.69% – 9.17%
- 2026 Z-score: **+4.41σ** (incl IBB) / **+6.38σ** (excl IBB)
- 2026 is **+0.60pp above the prior 8-year April max** (was 9.17% in 2018)
- **H2 PASSES decisively. Walk spike is NOT seasonality.** Branch B3 ruled out.

New ABS zone definition (effective 2026):
- Vertical: 27% to 53.5% of batter standing height
- Horizontal: 17 inches (matching plate width)
- All pitches measured at midpoint of plate
- Replaced legacy "midpoint of torso to hollow beneath kneecap" language

---

## 2. Hypotheses

**H1 (zone shape change):** The 2026 called-strike rate differs from 2025 by ≥3pp in at least one contiguous zone region. Direction unspecified — could be top (Hoerner's claim), bottom (kneecap redefinition), both, or neither. Both outcomes publish.

**H2 (seasonality controlled):** 2026 April walk rate is at least 1.5 standard deviations above the mean April walk rate from 2018-2025, after restricting to the same calendar window.

**H3 (count concentration):** 2026 minus 2025 walk-rate delta at the 3-2 count is at least 1.5x the all-counts walk-rate delta.

---

## 3. Editorial branches (replaces "kill criteria")

Both branches publish. The data picks the article framing.

| Branch | Trigger | Article framing |
|--------|---------|-----------------|
| **B1: Zone confirmed** | H1 holds (≥3pp shrinkage with CI excluding zero in some contiguous region) | "The Strike Zone Really Did Shrink — at the [top/bottom/both]. Here's the Map." |
| **B2: Zone myth-bust** | H1 fails (no contiguous region ≥3pp, or all CIs cross zero) | "Players Blame ABS for the Walks. The Zone Hasn't Actually Shrunk. Here's What Did." |
| **B3: Seasonality dominates** | H2 fails (Z<1.5 vs 2018-2025 April distribution) | ~~RULED OUT~~ — orchestrator pre-validation: 2026 is +4.4σ above 2018-2025 April mean. Branch closed. |
| **B4: 3-2 leverage adds** | H3 holds AND H1 holds | Add a leverage-concentration section to B1 article: "And It Hits Hardest at 3-2." |

These branches stack (B1+B4 is the strongest article) and don't substitute. The ONLY data outcome that doesn't publish is "no signal in any direction." With B3 ruled out, the article is **either B1+B4 (zone confirmed) or B2 (zone myth-bust)** — both publishable.

---

## 4. Round structure (multi-round)

This is now a multi-round project to capture a multi-article news cycle.

### Round 1 — News-anchored quick ship (TODAY → +24h)
**Sub-claims:** H1, H2, H3 (the brief above)
**Article:** "Did the Strike Zone Shrink? Here's What the Data Says"
**Goal:** Be the first analytical answer to the question every reader has after the AP wire piece.

### Round 2 — Attribution & mechanism (+3 to +7 days)
**Sub-claims:**
- Counterfactual: re-run 2026 PAs through a 2025-zone classifier — what fraction of the +0.82pp walk delta is attributable to zone change alone?
- Pitcher response: are pitchers throwing differently (more out-of-zone, slower, different mix)?
- By pitch type: which pitches lost the most strike calls (fastball vs breaking vs offspeed)?
**Article:** "How Much of the Walk Spike Is the Zone? Here's the Decomposition."
**Goal:** Move from descriptive ("zone shrunk") to causal-attribution ("zone explains X% of walks").

### Round 3 — Per-actor deep dives (+1 week, conditional on Round 1+2 finding signal)
**Sub-claims:**
- Per-umpire: which umpires' zones changed most? (now that ABS feedback is in the workflow)
- Per-team: which teams are most/least helped or hurt by the new zone?
- Catcher framing: have framers lost their advantage in 2026?
**Article(s):** "ABS's Winners and Losers: A Team and Catcher Breakdown"
**Goal:** Move from league-level to actionable team/player-level insights. Connects to fantasy and team-strategy audiences.

Each round runs as a separate dual-agent launch. Round N reads Round N-1's `reviews/COMPARISON_MEMO.md` as input.

---

## 5. Data

### 5.1 Primary corpus (all available)
- `data/statcast_2026_mar27_apr22.parquet` — **106,770 rows**, Mar 27–Apr 22 2026, 118 cols [BUILT]
- `research/count-distribution-abs/data/statcast_2025_mar27_apr14.parquet` — 70,876 rows 2025 same window
- `research/count-distribution-abs/data/statcast_2025_full.parquet` — full 2025 season (~108MB)
- `data/april_walk_history.csv` — April PA/walk aggregates 2018-2025 [BUILDING]

### 5.2 Apples-to-apples comparison rules
- For zone delta (H1): use 2026 Mar 27–Apr 22 vs 2025 same calendar window. Note 2025 parquet ends Apr 14, so primary comparison is **Mar 27–Apr 14 both years** (~70K vs ~74K rows). Extend with full Mar 27–Apr 22 2026 only as supplementary.
- For walk rate (H2, H3): use Mar 27–Apr 22 both years where 2025 data is available. For 2018-2024, use full April (Apr 1–30) since pre-2025 doesn't have a uniform start date in our pulls.
- Always include intentional walks (`events in {'walk', 'intent_walk'}`) for headline numbers — that's the convention ESPN/MLB uses.
- Always exclude `'automatic_ball'` and `'automatic_strike'` from "called pitches" subset — those are ABS challenge artifacts, not human zone calls.

### 5.3 Known data-quality issues
- `sz_top` and `sz_bot` are per-batter Statcast estimates of the human strike zone, not the ABS zone. Use only as references; primary analysis is in absolute (`plate_x`, `plate_z`) and height-normalized coordinates.
- Statcast's PA-terminating rule: `events.notna() & events != ''`. This includes hit-by-pitches, sacrifices, etc. — denominator is total PA, walk = `events == 'walk' or 'intent_walk'`.
- pybaseball schema may differ slightly between 2024 and 2025+ (newer columns). Use `set(2025.columns) & set(2026.columns)` intersection for any cross-season pooled analysis.
- Statcast may have publishing lag for very recent dates (Apr 22 2026 may have partial data); if PA counts look low for the most recent day vs a Tue/Wed, exclude the most recent day.

---

## 6. Features / Variables

| # | Name | Definition | Source |
|---|------|------------|--------|
| 1 | `plate_x` | Horizontal position at plate, ft | Statcast |
| 2 | `plate_z` | Vertical position at plate, ft | Statcast |
| 3 | `plate_z_norm` | Vertical position / (sz_top - sz_bot proxy) — height-normalized | Derived |
| 4 | `is_called_strike` | 1 if `description == 'called_strike'`, 0 if `'ball'` | Derived |
| 5 | `count_state` | (balls, strikes), 12 distinct counts | Derived |
| 6 | `season` | Year (2025, 2026) | Derived from `game_date` |
| 7 | `walk_event` | 1 if PA ended in `walk` or `intent_walk` | Derived |

---

## 7. Methodology

### 7.1 Agent A (Claude) — interpretability-first, statistical
**Mandate:** Transparent statistical methods that any analyst can replicate from the report alone.

Required methods:
- **2D grid binning** of `plate_x` × `plate_z_norm` into ~0.1 ft (or 1% height) bins; per-bin called-strike rate per season; delta map with bootstrap 95% CIs (≥1000 bootstrap samples).
- **Spline-smoothed difference surface** (2D LOWESS or thin-plate spline) over the binned deltas to visualize contiguous shrinkage regions.
- **Logistic GAM** (e.g., `pygam`) with `s(plate_x) + s(plate_z_norm) + season + season:s(plate_z_norm)` to test whether year-by-zone-shape interaction is significant.
- **Time-series Z-score** of 2026 April walk rate against the 2018-2025 April distribution from `april_walk_history.csv`.
- **Stratified walk-rate test** by count state with binomial CIs and a heterogeneity test (Cochran's Q or pairwise) for the 3-2 vs all-counts difference.

Forbidden: black-box ML methods. Everything reduces to a chart + a number a reader can replicate.

### 7.2 Agent B (Codex) — ML-engineering, model-driven
**Mandate:** Train predictive models that identify zone-shape change as a learnable signal, with attribution via standard ML interpretability tooling.

Required methods:
- **Year-classifier model** — gradient-boosted classifier (LightGBM or XGBoost) predicting `season` from `(plate_x, plate_z_norm, sz_top, sz_bot, pitch_type)` restricted to taken pitches. Report AUC; use **SHAP** or partial dependence to localize the year signal.
- **Two-zone-classifier comparison** — fit one logistic/GBM zone classifier per season predicting `is_called_strike` from `(plate_x, plate_z_norm)`; compute pointwise probability deltas across a fine grid; identify zone regions with highest |Δ| with bootstrap stability.
- **Counterfactual walk-rate simulation** — apply 2025 zone classifier to 2026 taken pitches; recompute would-be PA outcomes via a simple count-progression rule (each "ball" -> ball+1, each "strike" -> strike+1, terminate at 4-balls = walk or 3-strikes = K, otherwise PA continues to next pitch); compare counterfactual 2026 walk rate to actual. Report fraction of YoY walk delta attributable to zone change.
- **Distribution-shift detection** (KS test by zone region or energy distance) as supporting validation.

Forbidden: hand-coded grid binning or spline smoothing as primary method (those are Claude's lane). Use models, not bin counts.

**Why the divergence matters:** Claude produces a publishable map of where the zone moved. Codex produces an attribution number — *X% of the walk spike is the zone*. When both report the same locus and direction, finding is locked. When they disagree on attribution magnitude, the comparison memo IS the story.

---

## 8. Round 1 Deliverables per agent

Each agent must produce in their respective `claude-analysis/` or `codex-analysis/` directory:

1. `analyze.py` — one-command reproduction entry point
2. Module scripts per methodology component
3. `REPORT.md` — 1500-2500 word structured narrative covering H1, H2, H3
4. `charts/` — at minimum:
   - **Heat map of called-strike rate delta** (2026 minus 2025) over the plate, with colorbar centered at 0 and CIs visualized
   - **April walk rate by season**, 2018-2026, with 2026 highlighted and 2018-2025 mean ± SD band
   - **Walk rate by count state**, 2025 vs 2026, with 3-2 highlighted and CI bars
5. `findings.json` — machine-readable summary including:
   - `largest_delta_region`: {x_range, z_range, delta_pp, ci_low, ci_high}
   - `april_walk_z_score`: number, with ranking 1-9 of where 2026 falls in the 2018-2026 distribution
   - `count_3_2_delta_vs_all`: {three_two_delta_pp, all_counts_delta_pp, ratio, p_value}
   - (Codex only) `counterfactual_walk_rate_attribution_pct`: number in [0, 100]
6. `READY_FOR_REVIEW.md` — ≤500-word handoff with explicit answer to each of H1/H2/H3 and the recommended editorial branch (B1/B2/B3/B4)

---

## 9. Timeline (Round 1)

| Hour | Focus | Gate |
|------|-------|------|
| 0 (now) | Brief approved, data validated, agents launched | Both `data/` files present |
| 0–6 | Both agents complete initial analysis | Both `READY_FOR_REVIEW.md` exist |
| 6–9 | Cross-review (each agent reviews the other) | Both review files in `reviews/` |
| 9–12 | Comparison memo (orchestrator-side synthesis) | `reviews/COMPARISON_MEMO.md` written |
| 12–24 | Article drafting | Draft ready for editor review |

Round 2 launches +3 days from Round 1 article ship.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Sample too thin in 4 weeks for stable bin estimates | Pool taken pitches across plate halves where appropriate; bootstrap CIs not raw rates; require ≥30 pitches per bin or merge bins |
| Statcast `sz_top`/`sz_bot` definitions changed between 2025 and 2026 to reflect ABS | Use **absolute** `plate_z` for primary heat map; report height-normalized as supplementary; flag the schema check in both REPORTs |
| Counterfactual ignores pitcher adaptation | Caveat explicitly in Round 1; explore the question itself in Round 2 |
| Two agents converge by accident on same wrong methodology | Mandates are deliberately divergent — convergence is meaningful only because methods differ |
| Article moves slow because we wait for agents | Both agents launch in background; comparison memo is short-form |

---

## 11. How to Run

```bash
cd abs-walk-spike

# Step 0: backfill data (Statcast pull via pybaseball)
python scripts/build_2026_master.py     # -> data/statcast_2026_mar27_apr22.parquet
python scripts/fetch_april_history.py   # -> data/april_walk_history.csv (already included)

# Step 1: each agent's analysis (run independently)
python claude-analysis/analyze.py
python codex-analysis/analyze.py

# Step 2: post-cross-review adjudication in absolute coords
python codex-analysis/adjudication.py        # Codex's absolute-coord rerun
python scripts/clean_counterfactual.py       # Third independent implementation
python scripts/debug_counterfactual.py       # Aggregate first-principles sanity check

# Step 3: review the synthesis docs
# - reviews/claude-review-of-codex.md
# - reviews/codex-review-of-claude.md
# - reviews/claude-publish-readiness.md
# - reviews/codex-publish-readiness.md
# - ADJUDICATION_SUMMARY.md (canonical resolved state)
# - COMPARISON_MEMO.md (editorial synthesis)
```
