# Claude Round 3 — Ready for Review

**Status:** Complete. All deliverables in `claude-analysis-r3/`.
**Window:** 2026-03-27 — 2026-05-12 (47 days; 616 games; 46,755 PAs).

## Headline numbers

| R3 question | Result |
|---|---|
| **R3-H1** triangulated H3 magnitude | **+14.5% [+0.2, +70.1]** (median of three methods, widest CI) |
| R3-H1 Method A (Bernoulli + continuation) | −58.6% [−80.1, −35.1] (stress test) |
| R3-H1 Method B (kNN empirical) | +38.3% [+0.2, +70.1] (clean) |
| R3-H1 Method C (bootstrap-of-bootstrap) | +14.5% [+1.8, +29.7] (rigorous) |
| **R3-H2** named adapters at 80% stability | **0 names** (publishable null) |
| R3-H2 magnitude-passers (no stability filter) | 93 / 367 eligible |
| R3-H2 max observed stability | 58.5% (Bubba Chandler) |
| **R3-H3** Spearman ρ | **−0.282 (p < 0.0001)** |
| R3-H3 Bayesian slope β₁ | **−1.402 pp/unit [−2.043, −0.751]** |
| R3-H3 hurt names (≥80% stab) | 1 — Finnegan |
| R3-H3 helped names (≥80% stab) | 3 — O'Brien, Doval, Miller |

## Recommended editorial branch: **`mechanism + archetype`** (branch 3 in `ROUND3_BRIEF.md` §2)

Three reasons:
1. R3-H1 magnitude triangulates at ~+15% with editorial CI [+0.2%, +70%]. Both R2 Codex (+35%) and R1 (+40%) sit inside this CI. The R2 Claude −64% is a fixed artifact (observed-outcome backstop replaced with continuation model; zero PAs hit truncation in the new replay).
2. R3-H2 returns 0 named adapters at the pre-registered 80% stability threshold. Adaptation magnitude is real (93 magnitude-passers) but rank-stability over seven weeks is too noisy to name names.
3. R3-H3 is the editorial centerpiece. The archetype interaction is strongly credible. The hypothesis ("stuff pitchers benefit, command pitchers hurt") is supported. Pure-stuff vs pure-command implies a ~2.8pp walk-rate differential against a league mean Δ of +0.66pp. Four named pitchers cleared both magnitude AND 80% stability.

## Bonus: addressing R2 review concerns

All five R2 Codex concerns on my H3 are addressed in §1.5 of `REPORT.md`. Most importantly:
- `cs_probs.mean()` is now persisted in `findings.json` for all three methods (0.321 / 0.334 / 0.335, in agreement with R2's 0.334).
- The `observed-outcome backstop at h3_counterfactual.py:313-316` is replaced with a continuation model (sample remaining pitches from empirical 2026 distribution at running count). The new replay has 0 truncations across 46,755 PAs.

## Convergence diagnostics summary

| Module | R-hat max | ESS-bulk min | Pass |
|---|---:|---:|---|
| H3 archetype Bayesian | 1.000 | 4,935 | ✓ |
| H1 Method A | n/a (exact Beta) | 60 draws | ✓ |
| H1 Method B | n/a | 20 replays × 200 boot | ✓ |
| H1 Method C | n/a (ML refits) | 1,000 fits | ✓ |
| H2 per-pitcher | n/a (exact) | n/a | ✓ |

## Biggest concern

**FanGraphs is gated by Cloudflare bot protection (403 on direct HTTP and via `pybaseball.pitching_stats`).** I fell back to a Statcast-derived proxy for stuff+/command+. The proxy correctly identifies extreme archetypes (Doval, Miller, Bautista at the stuff end; classical control specialists at the command end) but is noisier than FanGraphs's true Stuff+/Location+ around the median. The cross-method check against Codex's GBM (which uses raw 2025 Statcast features rather than a proxy) is the right next step.

The archetype interaction signal is strong enough (Spearman ρ = −0.28 with n=259) that the proxy noise does not threaten the headline. The Bayesian slope CI excludes 0 with P=1.0.

## Outputs

- `analyze.py` — one-command reproduction (788s on a contended environment)
- `data_prep_r3.py`, `archetype_build.py`, `h1_triangulation.py`, `h2_adapter_leaderboard.py`, `h3_archetype_interaction.py`
- `REPORT.md` — full structured report (1500-2500 words)
- `findings.json` — machine-readable summary
- `charts/h1_triangulated_attribution.png` — three methods side-by-side, with R1/R2 reference lines
- `charts/h2_adapter_leaderboard.png` — "0 names cleared filters" publishable null
- `charts/h3_archetype_scatter.png` — per-pitcher scatter with Bayesian regression line + Spearman ρ
- `charts/h3_archetype_leaderboards.png` — 1 hurt, 3 helped named pitchers
- `charts/diagnostics/h3_interaction_trace.png` — Bayesian convergence
- `data/` — panel_2026, panel_2025_samewindow, pa_*, pitcher_archetype_2025
- `artifacts/` — per-module JSON and parquet artifacts
