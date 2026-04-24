# Comparison Memo — ABS Walk Spike Round 1

**Status:** Final. Editorial-facing synthesis derived from `ADJUDICATION_SUMMARY.md` + both agents' publish-readiness reviews.
**Audience:** The article author, and the editor for sign-off.
**Cross-references:** `PRIOR_ART.md` (CalledThird's prior ABS coverage), `ADJUDICATION_SUMMARY.md` (full technical record), `claude-publish-readiness.md` + `codex-publish-readiness.md` (publish blessings).

---

## 1. Headline finding (one sentence)

The 2026 ABS-era zone shrank at the top edge and modestly expanded at the bottom; **roughly 40-50% of the +0.82pp YoY walk-rate spike is attributable to this zone change, and the remaining 50-60% reflects pitcher behavior and other adaptations** that a zone-only counterfactual cannot capture.

## 2. Convergent claims (high publication priority)

Both agents independently reached the same conclusion on each:

- **Walk spike is real and not seasonality.** 9.77% over Mar 27-Apr 22 vs 9.02% historical mean (2018-2025), Z = +4.41σ. Above the prior 8-year max by +0.60pp.
- **Zone changed shape in absolute coordinates.** Top edge (z ≈ 3.2-3.9 ft): ~-7-8pp called-strike rate drop, large region. Bottom edge (z ≈ 1.0-2.0 ft): ~+5-6pp expansion, smaller region. Three implementations agree on this geometry.
- **Walk spike is NOT concentrated at 3-2.** H3 actively rejected: 3-2 walk-rate delta = -0.11pp vs +0.82pp all-counts (Cochran's Q p = 0.67).
- **40-50% counterfactual attribution to zone change.** Codex's adjudication: +40.46%. Clean third implementation: +49.40%. Aggregate sanity check: 2025 zone is +0.65pp more strike-friendly on 2026 pitches → POSITIVE direction.
- **The original -56% normalized-coord attribution was a Statcast schema artifact.** 2026 sz_top/sz_bot are deterministic per-batter (within-batter SD = 0.000) vs 2025's per-pitch posture estimates (SD = 0.072 ft). The schema change broke `plate_z_norm` cross-season comparability. Now resolved by using absolute plate coordinates.
- **Year-classifier AUC = 0.999 was metadata-driven, not location-driven.** Location-only year-classifier (no sz_*) AUC = 0.524 — essentially chance. The real plate-location year signal is much weaker than the headline AUC suggested.

## 3. Divergent claims (resolution)

Originally divergent, now resolved:

| Original divergence | Resolution |
|---|---|
| Codex (Round 1, normalized coords): -56.17% attribution → B2 | Codex CONCEDED: schema artifact was real. Absolute-coord rerun: +40.46%. |
| Claude (Round 1): top -22pp / bottom +22pp on edge bands → B1 | Magnitudes were specific-band overstatements. Smoother absolute-coord deltas: -7-8pp top, +5-6pp bottom on broader regions. Direction confirmed. |
| Claude's "first-pitch causal chain" via count-tree | Codex's location-conditional check showed 2026 zone is MORE strike-friendly at 0-0 by +2.48pp, suggesting the first-pitch CS rate drop is mostly pitch-LOCATION mix change. Causal claim downgraded; observational pattern still real. |

**Unresolved tension that ships in the article as honest disclosure:**

- The all-pitches counterfactual is +40-50% (positive, zone change ADDS walks).
- The 0-0-only counterfactual is -20% to -42% (negative, zone change REMOVES walks at first pitches).
- The literally-first-called-pitch counterfactual is +12% (small positive).
- We have not formally explained why the 0-0 restriction flips negative while the integrated effect is positive. Best hypothesis: 2026 zone is more strike-friendly on first pitches (reducing 1-0 traffic) but less strike-friendly in late counts at the top edge (allowing more terminal balls). The article must disclose this tension explicitly — it is the biggest skeptic gap.

## 4. Methodology deltas — what we learned

- **Coordinate system matters.** Anything depending on per-season normalization needs to handle the Statcast schema change explicitly (use absolute plate_z, or season-invariant batter heights via Lahman roster).
- **Single-cell-max headlines mislead.** Codex's "+48.38pp" headline was a single grid cell; the region mean was +17.66pp. CalledThird should report region means with CIs, not maxima.
- **Counterfactuals need averaging across draws.** Single-replay counterfactuals have high variance and can flip signs spuriously. The orchestrator's first attempt (-46%) demonstrated this; averaging across 32+ draws (Codex and clean third) resolves it.
- **Aggregate sanity checks are necessary but not sufficient.** The +0.64pp aggregate strike-rate test confirmed direction but doesn't substitute for PA-replay (which incorporates sequencing and count-state leverage).
- **Multi-implementation triangulation works.** The 10pp magnitude gap (40 vs 49) between two thoughtful implementations is real and unresolved — but both directions agreed. Reporting a range is appropriate.

## 5. What gets published

### Article #1 (this round, ship within 24 hours of brief)

**Working title:** *"ABS Took Away the High Strike — and That's Roughly 40-50% of the Walk Spike. Pitchers Own the Rest."*

**Article structure:**
1. **The hook:** AP/ESPN report (Apr 23, 2026) on walks at a 75-year high; players blame ABS.
2. **Our prior position:** [Apr 9 piece](/analysis/the-walk-rate-spike) said it was pitchers, not umpires (shadow-zone CS rate was UP).
3. **What changed in 17 more days of data + better methodology:** Heat map of zone delta in absolute coords. Zone moved up.
4. **The counterfactual:** Two independent ML pipelines say zone change accounts for 40-50% of the walk spike.
5. **The honest correction:** Apr 9 was right that pitchers are the larger share. We were wrong to dismiss the zone effect at zero. **40-50% is real.**
6. **Player vindication map:** Hoerner directionally consistent (mechanical, not behavioral); Sewald/McCann partly right; the zone DID change (just not "everywhere").
7. **The methodology box:** Dual-agent + adjudication round. Schema artifact diagnosed. Three implementations + first-principles diagnostic. 27-day window; mid-May re-run committed.

**Mandatory framing rules** (from publish-readiness reviews):
- Range ("40-50%"), never point estimate
- Drop "late-count" qualifier (we didn't decompose CF by count; H3 rejects concentration)
- "Hoerner consistent with" (NOT "vindicated"; he claimed behavioral, we showed mechanical)
- "Unmodeled behavior, mix shift, and adaptation" (NOT cleanly "pitcher behavior")
- "ABS-era zone changed" / "ABS rule change AND/OR umpire adaptation" (we cannot cleanly separate ABS rule from umpire learning curve)
- Apr 9 correction: "we under-weighted the zone effect" — explicit, not "partly right"
- Counterfactual window: Mar 27 - Apr 14 (apples-to-apples). Apr 22 frame is for H2 Z-score only.

**Mandatory disclosures (caveats section):**
1. First-pitch / 0-0 counterfactual sign-flip is the biggest skeptic gap — disclose mechanistic uncertainty
2. Implementation gap: 40% vs 49% across two careful implementations; report as range, not decomposed variance
3. 27-day window; mid-May re-run committed
4. Apr 9 prior position requires explicit correction language
5. Counterfactual is zone-only — pitcher response not modeled (that's Round 2)
6. Counterfactual cannot distinguish ABS rule change from umpire adaptation
7. 5.40% of all-pitch replays use continuation probabilities (not fatal but real)
8. Heatmap is descriptive geometry; not FDR-controlled inference

**Required charts (in priority order):**
1. **Heat map of zone delta (absolute coords)** — top down, bottom up, both edges visible
2. **April walk rate 2018-2026** — 2026 highlighted, historical mean ± SD band
3. **The counterfactual bar** — actual 2025, actual 2026, counterfactual 2026 under 2025 zone (range from 9.59% to 9.81%, plus actual 9.92%)
4. **Vertical strike-rate profile** — 2025 vs 2026 called-strike rate by absolute z (shows the "top moved down" pattern viscerally)
5. (Optional) Walk rate by count, 2025 vs 2026 — to support the "NOT concentrated at 3-2" claim

**Citations to other CalledThird pieces** (per `PRIOR_ART.md`):
- The Apr 9 piece — central correction
- "The Count That Matters" (Apr 16) — for context that 3-2 is leverage-rich, even though the walk spike isn't concentrated there
- "The Best ABS Challengers Are Catchers" (Apr 16) — to address Reddit altfillischryan's point that catcher challenges net-add strikes (so the spike isn't from challenges)
- "Anatomy of a Missed Call" (Apr 5) — foundational 7.2% miss rate, half-inch cliff

### Deferred to Round 2 (next 3-7 days)
- Decompose the all-pitches counterfactual BY COUNT to mechanistically explain the first-pitch sign-flip
- By pitch type (fastball vs breaking vs offspeed) — which pitches are losing the most strike calls?
- Pitcher response model: are pitchers throwing differently in response to the new zone?
- Per-pitcher and per-team breakdowns

### Deferred to Round 3 (next 1-2 weeks)
- Per-umpire breakdown
- Catcher framing's role in the new zone
- Mid-May re-run with ~6 weeks of additional data

## 6. Open questions for next round

1. **Mechanistic explanation for first-pitch sign-flip.** Why does 0-0-restricted CF flip negative while integrated CF is positive? Likely answer: 2026 zone is more strike-friendly on first pitches but less so in late counts. Decompose CF by count to confirm.
2. **Implementation magnitude gap.** Why do Codex and clean-third implementations differ by ~10pp on the headline number? Continuation-prob handling? RNG variance? Decomposition isn't blocking publication but is worth understanding.
3. **Pitcher response.** Counterfactual residual (~50-60%) is loosely labeled "pitcher behavior" — but is that pitch-mix shift, location-distribution shift, or count-sequencing shift? Round 2 should decompose.
4. **ABS rule change vs umpire adaptation.** What we measure is calling behavior; we cannot distinguish whether umpires are following the new rule precisely or adapting in response to ABS feedback. Could potentially be probed via pre/post-challenge calling patterns.
5. **Window stability.** The 40-50% range is from a 27-day window. Mid-May re-run will reveal whether it tightens or shifts.

## 7. Sign-off

- ✅ Claude (Agent A) blessed publication conditional on the framing refinements applied above (`reviews/claude-publish-readiness.md`)
- ✅ Codex (Agent B) blessed publication conditional on the framing refinements applied above (`reviews/codex-publish-readiness.md`)
- ✅ ADJUDICATION_SUMMARY.md updated to v2 with both agents' corrections incorporated
- ⏸ Editor final sign-off pending
- ⏸ Article drafting — next step
