# Comparison Memo: Claude vs Codex Coaching Gap Analyses

Prepared April 19, 2026 from the Round 6 cross-reviews:
- `reviews/claude-review-of-codex-round6.md`
- `reviews/codex-review-of-claude-round6.md`

and the matching findings JSON for both agents.

## Executive summary

After six rounds, both agents **converged on one survivor**: low-chase hitters extract meaningfully more wOBA on predictable pitches than chasers do. Pooled and batter-fixed-effects estimators agree to within 0.005 wOBA across agents. The two matched-pairs implementations disagree, and that disagreement is the honest frontier of this study — not a bug, not a convergence failure, but a real estimand-sensitivity result.

The published flagship ("+0.04 wOBA, low-chase vs high-chase") is **method-sensitive at the matched-pairs layer and method-robust at the pooled/FE layer.** The secondary survivor — the low-chase × high-xwOBA-contact quality-hitter 2×2 — is cleaner across methods and is the strongest single contrast.

Both reviewers agreed the study is **publishable as a reconciliation and estimand-sensitivity result**, not as a method-independent converged claim.

## 1. Convergent findings: high-confidence

### High confidence
- **Pooled chase-tertile spread.** Claude pooled: −0.043 wOBA (LOW − HIGH). Codex pooled: −0.038. FE: Claude −0.044, Codex −0.039. Residual 0.005 gap is deterministic, not noise: Claude computes the 80/20 predictability threshold on terminal pitches only; Codex on all scored pitches projected onto terminals. The inferential conclusion is identical.
- **The quality-hitter 2×2 is the cleanest cross-method survivor.** Pooled: Claude +0.029, Codex +0.025. Matched: Claude +0.018, Codex +0.021. FE: Claude +0.027. Four of six point estimates within 0.005 wOBA, all p<0.05 on pooled and FE, matched hovering at p≈0.06–0.10 on both sides.
- **H4 (cut-definition robustness) passes.** League-wide vs within-pitcher 80/20 cuts shift pooled by only −0.003 to −0.009 wOBA. The flagship survives either choice. The R5 cut-definition dispute was never the primary source of method disagreement.
- **H2 (variance decomposition) agrees on the structural share when normalized the same way.** When Codex applies its ANOVA-style structural split to Claude's sample, it recovers 55.5%/44.5% — close to Codex's own 51.8%/48.2%. The public "38% vs 51%" framing was comparing different decomposition targets on different eligible unit sets; corrected head-to-head, the agents agree.
- **H3 power-asymmetry reasoning.** Matched-pairs has materially less power than pooled on this substrate. Codex: pooled 85.7% / matched 40%. Claude's simulation is an idealized symmetry check and does not contradict this.
- **Null catalogue.** Both agents agree that arsenal breadth, power tertiles, contact-quality tertiles alone, whiff-rate tertiles, TTO, count state, pitch family, stuff tier, team, batter archetype, and the R1 change-point coupling all return null or diagnostic-confounded spreads. This is where 16 of the 17 hypotheses died. See `METHODOLOGY.md` for the full table.

## 2. Divergent findings: the honest frontier

### The matched-pairs estimate

| Implementation | Spread (LOW − HIGH) | p-value |
|---|---|---|
| Claude: pair-weighted, random exact-stratum matching | −0.017 | 0.050 |
| Codex: unit-weighted, nearest-neighbor on local distance + location context | −0.006 | 0.647 |
| Codex: pair-weighted sensitivity on same substrate | −0.015 | — |
| Codex: random exact-stratum sensitivity (no local distance) | −0.020 to −0.022 | — |

Both estimators are defensible. Two implementation choices drive the spread:

1. **Aggregation level.** Claude's code averages pair-level deltas directly by tertile (pair-weighted). Codex's primary collapses to a batter-season mean gap first, then averages unit means within tertile (unit-weighted). On Codex's cached pairs, switching aggregation alone lifts the estimate from −0.006 to −0.015.
2. **Local distance control.** Codex adds nearest-neighbor selection on `pitch_number`, `outs`, `runner_count`, and lagged `plate_x/z`, with a 6.5 max-distance cap. Claude does random 1:1 within exact strata only. Codex's design answers a strictly sharper question ("within this batter-season, in this count and handedness, and given nearly identical prior-pitch location context, does predictable beat unpredictable?"), which is a more demanding estimand than the one the Round 6 brief pre-registered.

Neither review flagged any leakage, selection bug, or coding error. The divergence is a real estimand choice, and the public article treats it as such: the pooled/FE finding is the headline; the matched-pairs attenuation is acknowledged as power-limited and estimand-sensitive.

### H2 and H3 labeling

Codex recommended — and Claude accepted in the final flagship — that the H2 decomposition be reported side-by-side on the same sample rather than as two numbers that appear to disagree. H3 is reported as an idealized symmetry check (Claude) plus an empirically-calibrated power study (Codex). Both are in the methodology; neither rebuts the other.

## 3. What both agents recommended for publication

- **Frame the headline as the chase-tertile spread**, pooled/FE ~0.04 wOBA, with matched-pairs attenuation explicitly disclosed and attributed to power and estimand choice rather than contradiction.
- **Name the quality-hitter 2×2 the cleanest secondary survivor** — this is what produced the "36 quality hitters" list in the flagship article.
- **Do not claim "method-independent reconciled standard."** Claim: converged on pooled/FE, sensitive on matched, null on 16 of 17 competing hypotheses.
- **Publish the full 17-hypothesis scoreboard** (see `METHODOLOGY.md`) so readers can see what was killed in each round. Nulls are the credibility.
- **Keep the 2026 out-of-sample replication as the robustness headline** — the 2022–2024-trained Layer 2 model scoring 2025/2026 unseen is the cleanest possible hold-out.

## 4. Why this matters beyond the finding itself

The Coaching Gap study was designed as a stress test of the dual-agent + cross-review research loop. The result we take from the reviews: **the method worked.** Across six rounds, the agents independently killed the same 16 hypotheses, disagreed sharply on two (matched-pairs design and H2 decomposition framing), and resolved those disagreements down to identifiable estimand choices rather than to undiagnosable noise. The final disagreement — matched-pairs implementation — is exactly the kind of disagreement a single-analyst study would have missed.

The flagship article and this public release are deliberately honest about the matched-pairs attenuation. If a future reader re-runs either script and finds a different matched estimate under a different matching radius or aggregation choice, they are reproducing the honest frontier, not a bug.
