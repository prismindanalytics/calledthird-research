# READY FOR REVIEW

Codex Round 3 deliverables are complete in `codex-analysis-r3/`.

**R3-H1:** The ML triangulated zone-attribution headline is **27.0%** with editorial CI **[5.7%, 56.9%]**. Method A is the R2 expectation-propagation replay fixed with **200 game-level refit bootstraps**; Method C uses **100 x 10** bootstrap-of-bootstrap refits. This replaces the invalid R2 seed-only interval.

**R3-H2:** ML-side stable adapters: **9**. The candidate table, stability scores, and per-pitcher SHAP feature-group attribution are in `artifacts/`; final named claims still require cross-method intersection with the Bayesian pipeline.

**R3-H3:** Stuff-command interaction: Spearman rho **-0.258** (p=0.000); interaction permutation importance **0.00243** vs permuted-label baseline **0.00333**. Stable leaderboard counts: hurt-command **5**, helped-stuff **2**.

Calibration artifacts are in `charts/model_diagnostics/` and per-refit audits are in `artifacts/*calibration_audit.csv`. Biggest concern: At least one H1 bootstrap OOB calibration bin exceeded 5pp; see h1_gbm_calibration_audit.csv. 343 of 1000 H2 feature-importance GBMs had an OOB calibration bin over 5pp.
