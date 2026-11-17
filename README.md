# GTA VI Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/faridhasyim99/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/faridhasyim99/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)


Forecasts the Twitch concurrent viewers of the "Grand Theft Auto VI"
category for each hour 2..24 of its release day (2026-11-19 UTC), as five
quantiles per hour, from the case's hour-1 print and the release-day
shapes of the past big launches in the case's `comparables`.

## Model

1. **Shape.** Each comparable contributes its release-day log-ratio curve
   `L[h] = ln(v_h / v_1)`.
2. **Size conditioning.** Per hour, `L[h]` is regressed on `ln(v_1)` across
   the comparables (OLS). The rise is strongly size-dependent
   (corr −0.94 at hour 20): a big hour 1 means most of the audience is
   already watching. The OLS slope is shrunk 20% toward the unconditioned
   median, because the full slope over-trusts the fit at the top end of
   the size range, where the final's hour-1 print will sit.
3. **Spread.** Normal band in log space, per-hour sigma = OLS residual SD
   × 0.4. The raw residual SD is inflated by structurally different
   launches, so the full width over-covers; 0.4 is the interior optimum
   with honest coverage (0.88 vs the 0.90 nominal).
4. **Floor.** Comparables with hour-1 print below 1,000 viewers are left
   out (sparse sampling = noise).
5. **Fallback.** An hour with fewer than 5 usable curves falls back to
   the house persistence baseline.

Python 3.13 standard library only, no network, deterministic. Runs in
~0.05 s per case, far under the 8 s cap.

## Practice score (local scorer, `dev/score.py`)

Loss **0.170967** vs house baseline 0.387097 (−56%). Per case: p00
Wolverine 0.1658 (baseline 0.7437), p01 RDR2 0.1833 (0.2868), p02 God of
War 0.1578 (0.2146), p03 Spider-Man 2 0.1769 (0.3033). Wins 4/4 cases.
Coverage_90 0.88, typical miss ~19%. Leave-one-case-out means 0.172–0.175
in every fold, so the gain is not driven by any single case.

See `TESTS.md` for the variant log and rejected levers.
