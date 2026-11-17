# gta6-twitch-ccu-forecast-01 entry: comparable-shape forecaster.
#
# Forecasts the Twitch concurrent viewers of the "Grand Theft Auto VI"
# category for each hour 2..24 of its release day (2026-11-19 UTC), as five
# quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per hour, from the case's hour-1
# print and the release-day shapes of the past big launches shipped in the
# case's `comparables`.
#
# Model (Python 3.13 standard library only, no network, deterministic).
#
#   1. SHAPE. For each comparable, take its UTC release day and form the
#      log-ratio curve L[h] = ln(v_h / v_1), h = 2..24, where v_1 is that
#      launch's hour-1 print. A launch's audience arrives over the day
#      (New Zealand first, the US last), so the curve rises from an
#      overnight dip to an evening peak; the level is set by hour 1.
#
#   2. SIZE CONDITIONING. The rise is strongly size-dependent: across the
#      comparables, corr(ln v_1, L[h]) reaches -0.94 at hour 20 -- a
#      launch whose hour 1 is already big has most of its audience
#      watching, while a small hour 1 means most of the audience is yet
#      to arrive. So per hour, regress L[h] on ln(v_1) over the
#      comparables (OLS) and predict the median at the case's own hour-1
#      print. The OLS slope is shrunk 20% toward the unconditioned
#      median (SHRINK = 0.8): the full slope over-trusts the fit at the
#      top end of the size range, exactly where the final's hour-1 print
#      will sit, and the shrunk version wins the final-like practice
#      case (God of War, the biggest hour 1) by 0.019.
#
#   3. SPREAD. The band is normal in log space around the median, with
#      per-hour sigma = the OLS residual SD, scaled by WIDTH = 0.4.
#      The raw residual SD is inflated by structurally different
#      launches (PC-simultaneous unlocks, tiny categories), so the full
#      width over-covers (coverage 1.00 at W = 0.7); W = 0.4 is the
#      interior optimum with honest coverage (0.88 vs the 0.90 nominal),
#      and W = 0 (median-only) is worse (0.199) than W = 0.4 (0.171).
#
#   4. FLOOR. Comparables with hour-1 print below 1,000 viewers are
#      left out of the fit: small categories are sampled sparsely by the
#      source and their hourly curves are noise. (The floor barely
#      matters: floor 1 vs 1000 changes the mean by 0.001.)
#
#   5. FALLBACK. If a case has too few usable comparables (< 5 for an
#      hour), that hour falls back to the house persistence baseline
#      (median = hour-1 print, spread 0.15*sqrt(h-1) in log space).
#
# Practice score (LOCAL scorer, dev/score.py, on SullyGnome's public
# hourly averages -- the same data the practice cases were cut from;
# see dev/build_answers.py for the validation: every hour-1 print and
# 926 history points match the case files exactly):
#   loss 0.170967 (house baseline 0.387097; -56%).
#   Per case: p00 Wolverine 0.1658 (baseline 0.7437), p01 RDR2 0.1833
#   (0.2868), p02 God of War 0.1578 (0.2146), p03 Spider-Man 2 0.1769
#   (0.3033). Wins 4/4 cases. Coverage_90 0.88, typical miss ~19%.
#   Leave-one-case-out means for this variant: 0.172-0.175 in every
#   fold (baseline LOCO 0.268-0.445), so the gain is not driven by any
#   single case.
#
# Rejected levers (all tested on practice, see TESTS.md):
#   - Unconditioned median curve (v1): 0.310 -- the size conditioning
#     is the whole game.
#   - Full OLS, no shrinkage (S = 1.0, W = 1.0): 0.2426 and LOSES p02
#     (0.2704 vs baseline 0.2146) -- over-trusts the slope at the top.
#   - Console-only comparables (9 curves): 0.214-0.231, p00 blows up
#     to 0.43-0.48 -- too few curves, unstable median.
#   - Empirical residual quantiles instead of normal spread: 0.207 at
#     best -- no better than normal at the same width.
#   - Width 0.7-1.2: 0.191-0.217 -- over-wide bands pay at every hour.
#
# The final's comparables include the four practice launches (the
# program reads whatever comparables the case ships, so they are used
# automatically) and the GTA VI pre-release curve, which has no
# release-day hour 1 and is skipped by the fit, as designed.
#
# Usage (mirrors the kit starter):
#   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# The grader runs one case at a time with only that case's file in /input;
# the program reads every *.in file it finds, in name order, and writes one
# /output/<id>.out per case. Runs in ~0.05 s per case locally, far under
# the 8 s cap; two runs on the same case write byte-identical output.

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# OLS slope shrinkage toward the unconditioned median (see header, 2).
SHRINK = 0.8
# Residual-SD multiplier for the log-space band (see header, 3).
WIDTH = 0.4
# Comparables with an hour-1 print below this are left out (see header, 4).
V1_FLOOR = 1000
# Minimum band sigma, in log units (a band must never collapse to zero).
SIGMA_FLOOR = 0.15
# Fallback spread (house persistence baseline) when a fit has too few curves.
FALLBACK_SIGMA = 0.15
# Minimum usable comparable curves for a per-hour fit.
MIN_CURVES = 5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit y ~ a + b*x; returns (a, b, residual SD)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    if n > 2:
        sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd


def release_day_hours(comp: dict) -> dict[int, float]:
    """{hour: viewers} for the comparable's UTC release day (hours 1..24)."""
    day = comp["day_start"][:10]
    hours: dict[int, float] = {}
    for pt in comp.get("hourly", []):
        if pt["from"][:10] == day and pt["viewers"] is not None:
            hours[int(pt["from"][11:13]) + 1] = float(pt["viewers"])
    return hours


def comparable_curves(case: dict) -> list[tuple[float, dict[int, float]]]:
    """(ln hour-1 print, {hour: log-ratio}) per usable comparable."""
    curves = []
    for comp in case.get("comparables", []):
        hours = release_day_hours(comp)
        v1 = hours.get(1, 0)
        if v1 < V1_FLOOR:
            continue
        ratios = {h: math.log(v / v1) for h, v in hours.items() if h >= 2 and v > 0}
        if ratios:
            curves.append((math.log(v1), ratios))
    return curves


def hour_quantiles(p1: float, h: int, curves: list[tuple[float, dict[int, float]]]) -> list[float | int]:
    """The five quantiles for hour h: OLS-conditioned median + normal band."""
    pts = [(x, r[h]) for x, r in curves if h in r]
    if len(pts) < MIN_CURVES:
        # Fallback: house persistence baseline for this hour.
        return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in Z]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    med_all = median(ys)
    a, b, sd = ols(xs, ys)
    med = med_all + SHRINK * ((a + b * math.log(p1)) - med_all)
    sig = max(sd * WIDTH, SIGMA_FLOOR)
    qs = [p1 * math.exp(med + z * sig) for z in Z]
    # Enforce non-decreasing (exp is monotone, so this is a formality)
    # and the output cap of 1e9 viewers.
    out: list[float | int] = []
    prev = 0.0
    for q in qs:
        q = min(max(q, prev), 1e9)
        prev = q
        out.append(round2(q))
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = comparable_curves(case)
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], curves)}
                    for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
