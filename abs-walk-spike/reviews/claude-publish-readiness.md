# Publish-Readiness Review — Agent A (Claude)

## 1. Is the synthesis correct?

**Yes, with one caveat.** Direction is overdetermined: Codex's `adjudication.py` (+40.46%), the clean third implementation (+49.40%), and the first-principles diagnostic (+0.64pp more strikes under 2025 zone) all converge on positive attribution. The "zone moved up + bottom expanded modestly" geometry is consistent with Codex's bin-level surface (-7.70pp top, +5.84pp bottom) and with my Round 1 findings. My schema-artifact concern (`sz_top` SD = 0.000 in 2026 vs 0.072 in 2025) is fully addressed by dropping normalization and `sz_*` features.

**Caveat:** Codex's multiple-testing critique (`codex-review-of-claude.md:7`) was never formally rebutted. Direction is now supported by counterfactuals independent of bin-grid scanning, so I'm comfortable — but the article should treat the heatmap as descriptive geometry, not FDR-controlled inference.

## 2. Claims I'm NOT comfortable with

- **`ADJUDICATION_SUMMARY.md:69` "concentrated in the late-count top-edge region"** — we never decomposed the counterfactual by count. The 3-2 leverage check (line 23) actively REJECTS late-count concentration. Drop "late-count."
- **Line 73 "Hoerner — vindicated."** Hoerner said hitters are *laying off* the top (behavioral); we showed the top is *being called* differently (mechanical). Same direction, different mechanism. Soften to "consistent with."
- **Line 84 "consistent with the all-pitches positive"** papers over Codex's first-pitch tension. Keep the number, lose the reassurance.

## 3. Sharpest defensible headline

**"The 2026 ABS zone shrunk at the top — and that explains roughly 40-50% of the walk spike. Pitchers own the rest."**

Range, not point. Keeps both directions of agency. Avoids "umpires" framing — it's a rule change, not umpire drift.

## 4. Final risks the article must disclose

1. **First-pitch counterfactual sign-flip is unresolved.** Two implementations get -20% and -42% on 0-0 only. We assert the integrated effect is positive but have not mechanistically explained why 0-0 differs. This is the biggest skeptic gap.
2. **The 10pp implementation gap (40 vs 49) is not decomposed.** Residual variance source (RNG seeding? unresolved-tail handling?) is unpinned. Report as range, never as a point.
3. **27-day window.** A single bad bullpen week could move 0.1pp. Re-run by mid-May before declaring victory.
4. **Article must say "we under-weighted the zone in our Apr 9 piece" explicitly** — not "partly right." Honest correction, not partial-credit framing.

**Verdict: BLESS for publication subject to (a) softening Hoerner/late-count language and (b) explicit disclosure of risks 1 and 2.**
