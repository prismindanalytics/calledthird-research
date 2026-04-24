# READY FOR REVIEW

- **H1:** FAIL as a shrink test. The largest stable zone-classifier delta is **+48.38 pp** in `x_range=[-1.0505, 1.1152]`, `z_range=[0.3915, 0.6574]`, and that dominant region is not shrinkage. A secondary shrink region does exist at `x_range=[-1.2121212121212122, 1.212121212121212]`, `z_range=[0.13696969696969696, 0.39717171717171723]`, but it is not the dominant zone move. The full year-classifier AUC is **0.999**, but SHAP says that signal is dominated by `sz_top/sz_bot` and height-proxy metadata; the location-only auxiliary year model still reaches **0.930**.
- **H2:** PASS. Recomputed 2026 Mar 27-Apr 22 walk rate is **9.77%** and the April-history Z-score is **4.41**, matching the substrate’s +4.41σ result within rounding.
- **H3:** FAIL. The 3-2 walk-rate delta is **-0.11 pp** versus **+0.82 pp** overall, a **-0.13x** multiplier. The strongest location-matched called-strike loss by count is **3-2** at **-0.48 pp**, but that did not translate into a realized H3 pass.
- **Counterfactual:** Applying the 2025 zone to 2026 primary-window called pitches moves the 2026 walk rate to **10.38%**. The attribution estimate is **-56.2%**, which is negative in current runs and therefore points the wrong way for a smaller-zone explanation. This remains a lower-bound estimate because pitcher behavior is held fixed.
- **Recommended branch:** **B2**.
- **Suggested one-line framing:** The early ABS zone does not look smaller overall: the dominant called-zone change is a middle full-width expansion, and the zone-only counterfactual moves walks the wrong way.
