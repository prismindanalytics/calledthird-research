# Adjudication Summary — Round 1 Resolution (v2, post publish-readiness reviews)

**Status:** Resolved + reviewed. Three independent implementations + first-principles diagnostic converge. Both agents have issued publish-readiness blessings (`reviews/claude-publish-readiness.md`, `reviews/codex-publish-readiness.md`) conditional on the framing refinements applied below.
**Question:** Did the 2026 ABS zone change cause the +0.82pp walk-rate spike, and if so, by how much?
**Answer:** Roughly **40-50%** of the spike is attributable to zone change (range, not point estimate). The remaining 50-60% reflects unmodeled pitcher behavior, pitch-mix shifts, and adaptations not captured by a zone-only counterfactual.

---

## Convergent findings (all 3 implementations agree)

### 1. Bin-level zone surface (absolute coordinates)
- **Top edge (z ≈ 3.2-3.9 ft)**: ~-7-8pp called-strike rate drop, large CI-significant region (787-1800 cells across implementations)
- **Bottom edge (z ≈ 1.0-2.0 ft)**: ~+5-6pp expansion, smaller region (395-851 cells)
- The negative region is bigger in area than the positive region

### 2. Walk-rate Z-score
- 2026 (Mar 27-Apr 22) walk rate: **9.77%** (matches ESPN's 9.8% incl IBB)
- Historical 2018-2025 April mean: 9.02%, SD 0.17pp
- **Z = +4.41σ** (or +6.36σ excl IBB) — walk spike is unambiguously NOT seasonality
- Apr 9 piece's seasonality dismissal is fully validated

### 3. 3-2 leverage hypothesis
- **REJECTED**. 3-2 walk-rate delta = -0.11pp vs +0.82pp all-counts. Cochran's Q p = 0.67 (no per-count heterogeneity)
- The walk damage is NOT concentrated at high-leverage counts

### 4. Year-classifier sanity
- With sz_top/sz_bot/batter_height_proxy: AUC = 0.999 (artifactual — Statcast schema changed in 2026)
- Location-only (no sz_*): AUC = 0.524 (essentially chance)
- Confirms the original 0.999 was Statcast metadata, not real plate signal

---

## The dispute and its resolution

### The dispute
- **Codex (Round 1, normalized coords)**: -56.17% attribution → "zone change moved walks the wrong way" → B2 (myth-bust)
- **Claude (Round 1)**: zone shrunk -22pp at top edge, expanded +22pp at bottom (absolute coords) → B1 (confirmed)

### The cause of the dispute
Statcast changed `sz_top`/`sz_bot` storage in 2026 to deterministic per-batter ABS values (within-batter SD = 0.000), versus 2025's per-pitch posture estimates (SD = 0.072 ft). Codex's `plate_z_norm = plate_z / batter_height_proxy` was therefore non-comparable across seasons. The "+48.38pp middle expansion" in normalized coords mapped to ≈ 2.50-4.24 ft absolute — the upper-edge region where Claude found shrinkage.

### The resolution
Three independent counterfactual implementations in **absolute coords** (no normalization, no sz_* features):

| Implementation | All-pitches attribution | Method note |
|---|---|---|
| Codex's `adjudication.py` (his existing replay code, abs coords) | **+40.46%** | Average across draws, continuation probs for unresolved tails |
| Clean third implementation (`scripts/clean_counterfactual.py`) | **+49.40%** | Same design as Codex; built from scratch; matches direction within ~10pp magnitude |
| Aggregate first-principles diagnostic (`scripts/debug_counterfactual.py`) | **DIRECTION: POSITIVE** | 2025 model predicts +0.64pp more strikes than actual 2026 — direction unambiguous |
| ~~Original orchestrator (`scripts/adjudicate_absolute_coords.py`)~~ | ~~-46.09%~~ | **BUG** — sign flipped due to single-draw RNG variance, dropna shedding intent_walks, and replay logic differences. Disregard. |

**Three converging implementations + first-principles diagnostic establish: zone change accounts for ~40-50% of the walk spike (positive attribution), not -56% as initially claimed.**

### The first-principles test (sanity check, NOT standalone sign-lock — see Codex review)
On 2026 called pitches:
- Actual 2026 called-strike rate: **32.34%**
- 2025 zone model's predicted rate: **32.99%** (+0.65pp HIGHER)
- 2026 zone model's predicted rate: **32.35%** (matches actual — sanity check)

The +0.64pp aggregate strike surplus under the 2025 zone is consistent with positive attribution (more strikes → fewer balls → fewer walks). However, this aggregate test ignores count-state leverage and PA sequencing, so it should not stand alone as a sign-lock proof — it is a corroborating sanity check. The dispositive evidence is the two PA-replay implementations (+40.46% and +49.40%), which incorporate sequencing.

By region:
- **Top (3.2-3.5 ft)**: 2025 over-calls strikes by +10.21pp (this is where 2026 shrunk)
- **Bottom (1.5-2.0 ft)**: 2025 under-calls strikes by -9.86pp (this is where 2026 expanded)
- Net across all pitches: +0.64pp more strikes under 2025

---

## The synthesis (publish-ready, post publish-readiness reviews)

The 2026 ABS zone moved up — modestly. The top-edge called-strike rate dropped ~7-8pp on a large region (z ≈ 3.2-3.9 ft); the bottom edge expanded ~5-6pp on a smaller region. Roughly **40-50% of the +0.82pp YoY walk-rate spike is attributable to this zone change** (a range, not a point estimate). The remaining 50-60% reflects unmodeled pitcher behavior, pitch-mix shifts, and adaptations not captured by a zone-only counterfactual.

We have NOT decomposed the counterfactual by count, so we cannot claim the zone effect is concentrated in any specific count. (H3 — concentration at 3-2 — was actively rejected.)

**Player and prior-art map:**
- ✅ **Hoerner ("hitters laying off the top of the zone")** — directionally consistent. Hoerner described batter behavior; our finding is mechanical (umpires calling the top differently). Same direction, different mechanism.
- 🟡 **Sewald/McCann ("smaller zone")** — partially right on net effect (zone change accounts for ~40-50% of walks), but the broader claim that "zone is smaller everywhere" overstates: the zone shrunk at the top and EXPANDED at the bottom.
- ❌→🟡 **Our Apr 9 piece ("it's pitchers, not umpires")** — **we under-weighted the zone effect**. The Apr 9 piece concluded the zone wasn't squeezing and the walk spike was purely pitcher nibbling. With 17 more days of data and proper counterfactual modeling, ~40-50% of the walk spike IS the zone change. Pitchers are still the larger share (~50-60%), but the zone is not zero. Article must explicitly state this correction — not "partly right."

---

## Caveats the article must include

1. **The "40-50%" attribution is ZONE-ONLY counterfactual.** It assumes pitchers and batters wouldn't have changed behavior if the 2025 zone were still in effect. This is unrealistic in the real causal sense — likely a lower bound on the ZONE-ONLY effect and an upper bound on PITCHER ADAPTATION as a residual.
2. **Sample window is short** (Mar 27-Apr 22, ~27 days). The counterfactual specifically uses Mar 27-Apr 14 (apples-to-apples with the 2025 same-window parquet); the longer Mar 27-Apr 22 frame is for the H2 seasonality Z-score, not the counterfactual. CalledThird commits to a re-run by mid-May before declaring victory.
3. **Counterfactual attribution range is ±10pp** between two careful implementations (Codex 40.46%, clean third 49.40%). Report as a range, not a point estimate. The residual variance source (RNG, unresolved-tail handling) is not formally decomposed.
4. **First-pitch / 0-0 counterfactual is definition-sensitive AND sign-flipped:**
   - Restricted to 0-0-count called pitches only: -20.41% (clean) to -41.95% (Codex)
   - Restricted to literally the first called pitch in each PA: +11.93%
   The integrated all-pitches counterfactual is positive (+40-50%), but we have NOT mechanistically explained why the 0-0-only restriction flips negative. This is the most honest skeptic gap; the article must disclose it explicitly.
5. **2026 Statcast schema changed sz_top/sz_bot** to deterministic per-batter ABS values (within-batter SD = 0.000 in 2026 vs 0.072 ft in 2025). This caused the original -56.17% artifact; methodology must use absolute plate coords or season-invariant batter heights.
6. **Counterfactual cannot distinguish "ABS rule change" from "umpire adaptation to ABS feedback".** What we measure is the change in calling behavior, not the cause of that change. The article should say "the zone changed" / "ABS-era zone," not attribute the change to a specific causal mechanism.
7. **5.40% of all-pitch counterfactual replays** ended in unresolved-tail count states where we used empirical 2026 walk rates as continuation probabilities. That is not fatal but is real and should be disclosed.
8. **Heatmap is descriptive geometry, NOT FDR-controlled inference.** Per Codex's original cross-review concern about multiple testing on bin-grid scans, the bin-level deltas are corroborated by the counterfactual (which doesn't depend on the bin grid) but should not be presented as formal hypothesis tests.

---

## Files of record

**Round 1 work:**
- `claude-analysis/REPORT.md`, `findings.json`, `READY_FOR_REVIEW.md`
- `codex-analysis/REPORT.md`, `findings.json`, `READY_FOR_REVIEW.md`

**Cross-reviews:**
- `reviews/claude-review-of-codex.md` (794 words, surfaced the schema artifact)
- `reviews/codex-review-of-claude.md` (5195 chars, surfaced the multiple-testing concern + first-pitch causal critique)

**Adjudication run:**
- `codex-analysis/adjudication.py` + `adjudication_results.json` (Codex's abs-coord rerun: +40.46%)
- `scripts/adjudicate_absolute_coords.py` + `reviews/adjudication_results_orchestrator.json` (BUGGY — disregard)
- `scripts/clean_counterfactual.py` + `reviews/adjudication_results_clean.json` (third impl: +49.40%)
- `scripts/debug_counterfactual.py` + log (aggregate diagnostic — direction-locking)

**This document** is the canonical synthesis. The article should cite these findings and caveats.
