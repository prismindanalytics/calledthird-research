# Ready For Review

H1: fails. Spot 7's raw overturn rate is 51.2% (n=213, bootstrap 95% CI 43.7% to 57.7%).

H2: fails by the challenge-model counterfactual. Holding pitch, count, pitcher tier, catcher tier, umpire, and location fixed, spot 7 vs spot 3 changes predicted overturn probability by +0.15 pp [+0.08, +0.23].

H3: fails on borderline called pitches. The called-pitch model estimates spot 7 vs spot 3 at -0.35 pp called-strike probability [-0.39, -0.31], n=2767.

Recommended branch: **B4**.

Biggest methodological concern: The challenge model has only a few hundred rows per lineup region, so the H2 counterfactual is less stable than the called-pitch H3 model; treat it as a controlled diagnostic, not a standalone causal estimate.
