# Ready for Review R2


Sleeper hitter picks: Tristan Peters, Everson Pereira, Jorge Barrosa, Jac Caglianone, Owen Caissie, Samuel Basallo, Coby Mayo, Brady House, Dillon Dingler, Angel Martínez.

Fake-hot hitter picks: Carter Jensen, Aaron Judge, Corbin Carroll, Xavier Edwards, Max Muncy, Sal Stewart, Drake Baldwin, Ryan O'Hearn, Matt Olson, Mike Trout.

Sleeper reliever picks: Cole Wilcox, Daniel Lynch, Antonio Senzatela, Tyler Phillips, John King.

Methodology fixes: Murakami resolver done (MLBAM 808959); QRF coverage done with 2025 80% coverage 85.4%; era counterfactual dropped; SHAP done by dropping it and using permutation only; xwOBA gap done.

Universe: 289 hitters and 218 relievers, actual Statcast max game date 2026-04-24.

Kill gates: {'universe_coverage': 'pass', 'sleeper_signal_yield_h1': 'pass', 'fake_hot_yield_h2': 'pass', 'reliever_sleepers_h4': 'pass', 'qrf_coverage_gate': 'pass'}. Cross-agent comparison should treat any interval verdicts according to the QRF coverage warning in `findings_r2.json`.