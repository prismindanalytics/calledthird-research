# READY FOR REVIEW R3 - Codex

## Fix-status checklist

- Fix 1 QRF calibration framing: done, Path B. R3 reports raw QRF coverage and drops calibrated-language claims.
- Fix 2 zero-prior sleeper: done. `preseason_prior_woba > 0`; Tristan Peters is removed.
- Fix 3 sleeper ranking rule: done by filter path; ranking remains predicted delta to preserve the sleeper/upside estimand.
- Fix 4 fake-hot rule: done. Strict `pred_ros_woba < prior - 1 prior SD`; count = 1.
- Fix 5 xwOBA-gap hedge: done. Only `xwoba_minus_prior_woba_22g` is reported in importance.

## Named-starter R3 verdicts

andy_pages: NOISE (high), ben_rice: NOISE (medium), mason_miller: AMBIGUOUS (medium), mike_trout: NOISE (medium), munetaka_murakami: AMBIGUOUS (low)

## Top sleeper hitters

Everson Pereira, Jorge Barrosa, Samuel Basallo, Jac Caglianone, Dillon Dingler, Coby Mayo, Kyle Karros, Leody Taveras, Brady House, Ildemaro Vargas

## Top sleeper relievers

Antonio Senzatela, Blade Tidwell, Daniel Lynch, John King, Caleb Kilian

## What changed from R2

Killed picks: Tristan Peters (zero-prior hitter sleeper removed by preseason_prior_woba > 0 filter), Cole Wilcox (low-prior reliever dropped by R3 prior K% floor), Louis Varland (excluded from R3 sleeper-reliever list; remains fake-dominant). Named verdict changes: none. QRF intervals are now raw-QRF only, Peters is removed, the fake-hot screen is stricter, and the feature-importance table reports one xwOBA-gap variant.
