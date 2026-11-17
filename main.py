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
#      comparables (OLS) and predict at the case's own hour-1 print.
#      The OLS prediction is shrunk toward the unconditioned MEAN
#      (v5 change: SIZE-DEPENDENT shrinkage -- see shrink_at() below):
#      the full slope over-trusts the fit at the top end of the size
#      range, exactly where the final's hour-1 print will sit, and
#      under-trusts it at the bottom end, because the size effect
#      saturates: the biggest launches rise less than the linear fit
#      predicts. So the shrink factor falls smoothly with ln(hour-1):
#      S = 0.75 - 0.10 * (ln(v1) - 12.2), clipped to [0.40, 1.00]
#      (at ln v1 = 12.2 this is exactly the v4 constant 0.75; a
#      1M-viewer hour 1 gets S = 0.59, a 23k one gets S = 0.96).
#      The anchor is the MEAN, not the median (v2 change):
#      L[h] is right-skewed across launches (small launches explode
#      from a tiny hour 1), so the median sits below where realized
#      launches land; anchoring on the mean wins 3 of 4 practice cases,
#      including the biggest and most final-like one (God of War,
#      0.158 -> 0.105), and the leave-one-comparable-out mean confirms
#      it (0.151 vs 0.172 for the median-anchored version).
#      (v6 change: the per-hour OLS SLOPES are smoothed first -- see
#      smooth_slopes() below. The slope curve b(h) falls smoothly from
#      ~0 at hour 2 to ~-0.86 at hour 20, but each hour's OLS slope
#      carries a standard error of ~0.09, larger than the typical
#      hour-to-hour change (0.045), so the raw slopes are mostly noise
#      around a smooth curve. Smoothing them toward a Gaussian-kernel
#      average (bandwidth 3.5 h) with empirical-Bayes weighting cuts
#      that noise. It barely moves interpolation (practice/LOCO are
#      flat) but it matters exactly where the final lives: far outside
#      the training size range, the slope error is multiplied by the
#      extrapolation distance, so the smoothed slopes forecast big
#      launches much better -- see the big-launch holdout below.)
#
#   3. SPREAD. The band is normal in log space around the median, with
#      sigma = a single POOLED residual SD (v2 change), scaled by
#      WIDTH = 0.32 (v5 change: re-tuned on the new center; the median
#      moved toward the realized launches so slightly less spread is
#      needed). The pooled SD is computed once per case from the
#      residuals of all 23 per-hour OLS fits (about 800 residuals), so
#      it is far more stable than 23 separate per-hour SDs (which range
#      0.29..1.05). The raw residual SD is inflated by structurally
#      different launches (PC-simultaneous unlocks, tiny categories),
#      so the full width over-covers; W = 0.35 is the interior optimum
#      at the pooled scale, with honest coverage (0.88 vs the 0.90
#      nominal). The band is ASYMMETRIC (v3 change): the lower quantiles
#      use 0.70x the width and the upper quantiles 1.35x falling to
#      1.00x over hours 2..24 (v4 change: the right-skew of the
#      residuals is strongest in the early hours, when the audience is
#      still arriving, and fades as the day fills in -- so the upper
#      tail needs the most room early and less late), because the
#      log-ratio residuals are right-skewed -- realized launches
#      overshoot the median far more often than they undershoot it, so
#      the upper tail needs room the lower tail does not. On top of
#      that, the per-hour residual skewness nudges the top wider still
#      (SKEW_K = 0.05, v4 change), and the outer quantiles (|z| = 1.645)
#      are narrowed 5% (WIDTH_OUTER = 0.95, v4 change): the far tails
#      of the pooled normal are slightly too fat for the realized
#      launches. The whole asymmetry family is a broad plateau
#      (lo 0.70-0.75, hi2 1.25-1.40, hi24 0.95-1.10 all beat v3 on both
#      measures), not a tuned point.
#      (v6 change: ONE-SIDED size scaling of the band -- the residual
#      scale falls with launch size (corr(ln v1, log|resid|) = -0.29),
#      but only the big end is trustworthy: mid-size targets must not
#      be widened. So the band is narrowed only above ln(v1) = 12.2,
#      by exp(-0.10 * max(0, ln(v1) - 12.2)). A symmetric scaling was
#      tested and rejected: it widens mid-size bands and loses on
#      practice and LOCO. The one-sided form is neutral on
#      practice/LOCO and slightly better on the big-launch holdout.)
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
#   loss 0.138915 (house baseline 0.387097; -64%; banked v5: 0.139014;
#   submitted v2: 0.150081).
#   Per case: p00 Wolverine 0.1059 (baseline 0.7437), p01 RDR2 0.1441
#   (0.2868), p02 God of War 0.1009 (0.2146), p03 Spider-Man 2 0.2048
#   (0.3033). Wins 4/4 cases vs the baseline. Coverage_90 0.90.
#   Leave-one-comparable-out mean for this variant: 0.139896 over 34
#   folds (wins 21/34 folds vs v5; worst fold +0.0006), so the gain is
#   not driven by any single comparable. Big-launch holdout (each of
#   the 6 biggest comparables held out in turn and forecast as a
#   pseudo-target): 0.120345 vs v5's 0.130241 (-7.6%), which matters
#   because the final's hour-1 print will likely exceed every
#   comparable -- the regime practice/LOCO never test. Per-launch:
#   ELDEN RING 0.072 -> 0.056, Cyberpunk 0.115 -> 0.109, Diablo IV
#   0.130 -> 0.126, Hogwarts 0.156 -> 0.135, Last of Us II 0.115 ->
#   0.099 (only Nightreign worsens, 0.193 -> 0.198).
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
#
# Revision (2026-11-17): removed the dead sys.path setup and the now-unused
# `import sys` — the grader runs with -P, but this program imports only the
# standard library and ships no local module, so the insert had no effect.
# Re-scored after the removal on the four practice cases: loss still
# 0.170967 (p00 0.165828, p01 0.183308, p02 0.157833, p03 0.176899).
#
# v2 (2026-11-17, entry-v2): mean anchor instead of median, pooled
# residual SD instead of per-hour SD, SHRINK 0.8 -> 0.7, WIDTH 0.4 ->
# 0.35. Verified on the four practice cases plus leave-one-comparable-
# out (34 folds): practice 0.150081, LOCO 0.150810, vs the submitted
# v1's 0.170967 / 0.171595. Grid-searched over shrink/width/floor and
# structural variants (winsorized OLS, Theil-Sen, top-half slope,
# trimmed-mean/q60-q70 anchors, sigma floor, min-hours); the mean
# anchor is the real gain, the pooled sigma a small stable extra.
# Rejected: top-half slope (0.225-0.260, p00 blows up), winsorized OLS
# (0.189-0.202), Theil-Sen (0.195-0.202), floor 5000 (p00 0.39-0.42).
#
# v3 (2026-11-17, entry-v3): asymmetric band (lower quantiles 0.80x,
# upper 1.18x the pooled width) and SHRINK 0.7 -> 0.75. Verified on the
# four practice cases plus leave-one-comparable-out (34 folds):
# practice 0.146698, LOCO 0.147507, vs the submitted v2's 0.150081 /
# 0.150810. Round-3 search also tested and rejected: size-similarity
# kernel weighting (0.18-0.28 -- collapses the effective sample),
# console-only upweighting and a console second regressor (0.15-0.19),
# simultaneous-unlock downweighting/regressor (0.16), hour-dependent
# shrink (0.153-0.156), median-blend and q55-q65 anchors (0.15-0.17),
# per-hour width scaling (0.1506), 3-point median smoothing (0.1515),
# sigma-floor variations (no change -- floor never binds).
#
# v4 (2026-11-17, entry-v4): hour-varying asymmetry (lower quantiles
# 0.70x; upper quantiles 1.35x at hour 2 falling linearly to 1.00x at
# hour 24), per-hour residual-skewness nudge on the top (SKEW_K 0.05),
# and 5% narrower outer quantiles (WIDTH_OUTER 0.95). Verified on the
# four practice cases plus leave-one-comparable-out (34 folds):
# practice 0.145225, LOCO 0.146137, vs the banked v3's 0.146698 /
# 0.147507 (and the submitted v2's 0.150081 / 0.150810). Round-4 search
# also tested and rejected: slope smoothing across hours (0.1470,
# corrected pivot -- the size-dependence is NOT smooth over the day),
# dropping the noisiest comparables (0.19-0.29 -- destroys the fit),
# trimmed-mean anchors (0.149-0.161), ensembles of nearby band
# settings (0.1467 -- no gain over the single best point), and
# sigma_alpha 0.1 on top of the final combination (no gain).
#
# v5 (2026-11-17, entry-v5): SIZE-DEPENDENT slope shrinkage
# (S = 0.75 - 0.10 * (ln(v1) - 12.2), clipped to [0.40, 1.00]) and
# WIDTH 0.35 -> 0.32. Verified on the four practice cases plus
# leave-one-comparable-out (34 folds) plus the big-launch holdout
# (6 biggest comparables as pseudo-targets): practice 0.139014,
# LOCO 0.140156, big-holdout 0.130241, vs the banked v4's 0.145225 /
# 0.146137 / 0.157460. The gain is a broad plateau (shrink-slope
# k = 0.06-0.15 and center c = 12.0-12.4 all beat v4 on all three
# measures), and it wins 5/6 big-launch holdouts -- ELDEN RING
# 0.149 -> 0.077, Cyberpunk 0.163 -> 0.117, Diablo IV 0.194 -> 0.131
# (only Nightreign worsens, 0.160 -> 0.189). Round-5 search also
# tested and rejected: direct per-hour quantile regression (0.16-0.23
# -- far too noisy on ~34 points per hour), empirical residual
# quantiles instead of the normal band (0.153-0.154), Student-t bands
# (0.146-0.147 -- tails are lighter than normal, not heavier),
# pre-release peak/last-hour ratios, viewers-per-channel and release
# year as second regressors (0.148-0.167), pre-release timezone-share
# features (0.147-0.163), quadratic size conditioning (big-holdout
# 0.190 -- unstable extrapolation), and a hard clamp of the
# conditioning point at the max comparable size (no effect).
#
# v6 (2026-11-17, entry-v6): KERNEL-SMOOTHED slopes + ONE-SIDED band
# narrowing for big targets. (1) Each hour's OLS slope is shrunk
# toward a Gaussian-kernel-smoothed slope curve (bandwidth 3.5 h)
# with empirical-Bayes weighting (tau^2 from the slope residuals
# around the smoothed curve, scaled 0.25); the line is re-pivoted
# through the per-hour mean point and the pooled SD is recomputed
# from the smoothed-slope residuals. (2) The band is narrowed only
# above ln(v1) = 12.2, by exp(-0.10 * max(0, ln(v1) - 12.2)).
# Verified on the four practice cases plus leave-one-comparable-out
# (34 folds) plus the big-launch holdout: practice 0.138915, LOCO
# 0.139896, big-holdout 0.120345, vs the banked v5's 0.139014 /
# 0.140156 / 0.130241. The kernel family is a broad plateau (bw
# 3.0-4.0 all beat v5 on BIG by 0.008-0.010 with practice/LOCO
# within 0.0002 of v5), and the candidate wins 21/34 LOCO folds
# and 5/6 big-launch holdouts. Round-6 search also tested and
# rejected: symmetric sigma-by-size scaling (0.140-0.151 -- widens
# mid-size bands), recency weighting (0.140-0.149), global
# polynomial slope priors (practice 0.141-0.144 -- the quadratic
# cannot follow the slope curve's S-shape; the cubic is
# practice-neutral but gains less on BIG), target-aware prior
# blending by extrapolation distance (practice 0.139-0.141, BIG
# 0.118-0.123 -- more BIG gain but a real practice cost), a
# two-component mixture quantile band (0.142-0.143), a
# precision-weighted anchor (0.150 -- the noisy launches carry real
# signal about the mean shape), a slope-uncertainty band term
# (neutral), and a two-anchor model conditioning on hour-2/hour-3
# early growth (impossible: only hour 1 is known at the checkpoint).
#
# v6 verification note (2026-11-17): the numbers above are the
# real entry file's end-to-end scores, not harness-only -- the
# scorer (dev/score.py) gives 0.138915, an independent 34-fold
# LOCO of this exact file (verify_loco.py, real subprocess runs)
# gives 0.139896, and an independent big-launch holdout of this
# exact file (verify_big.py, 6 subprocess runs) gives 0.120345.
# Two runs write byte-identical output; ~0.09 s/case.

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Size-dependent OLS slope shrinkage toward the unconditioned mean
# (see header, 2): S falls smoothly with ln(hour-1) because the size
# effect saturates at the top end. At ln(v1) = 12.2 this is exactly
# the v4 constant 0.75; a 1M-viewer hour 1 gets S = 0.59.
SHRINK_K = 0.10
SHRINK_C = 12.2
SHRINK_LO = 0.40
SHRINK_HI = 1.00


def shrink_at(ln_p1: float) -> float:
    return max(SHRINK_LO, min(SHRINK_HI, 0.75 - SHRINK_K * (ln_p1 - SHRINK_C)))


# Residual-SD multiplier for the log-space band (see header, 3).
WIDTH = 0.32
# Asymmetric band multipliers (see header, 3): lower quantiles use
# WIDTH_LO; upper quantiles fall linearly from WIDTH_HI_H2 at hour 2
# to WIDTH_HI_H24 at hour 24. Per-hour residual skewness nudges the
# top wider (SKEW_K), and the outer quantiles are narrowed
# (WIDTH_OUTER).
WIDTH_LO = 0.70
WIDTH_HI_H2 = 1.35
WIDTH_HI_H24 = 1.00
SKEW_K = 0.05
WIDTH_OUTER = 0.95
# One-sided band narrowing for big targets (see header, 3): the band
# is scaled by exp(-SIGMA_GAMMA * max(0, ln(v1) - SIGMA_CENTER)).
SIGMA_GAMMA = 0.10
SIGMA_CENTER = 12.2
# Comparables with an hour-1 print below this are left out (see header, 4).
V1_FLOOR = 1000
# Minimum band sigma, in log units (a band must never collapse to zero).
SIGMA_FLOOR = 0.15
# Fallback spread (house persistence baseline) when a fit has too few curves.
FALLBACK_SIGMA = 0.15
# Minimum usable comparable curves for a per-hour fit.
MIN_CURVES = 5
# Kernel-smoothed slope prior (see header, 2): Gaussian kernel
# bandwidth in hours, and the empirical-Bayes tau^2 multiplier.
KERNEL_BW = 3.5
KERNEL_MULT = 0.25


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


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Least-squares fit y ~ a + b*x; returns (a, b, residual SD, slope SE)."""
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
    se_b = (sd / math.sqrt(sxx)) if sxx and n > 2 else 0.0
    return a, b, sd, se_b


def compute_fits(curves: list[tuple[float, dict[int, float]]]) -> dict[int, dict]:
    """Per-hour OLS fits: {h: dict(a, b, sd, se, mx, my, pts)}."""
    fits: dict[int, dict] = {}
    for h in range(2, 25):
        pts = [(x, r[h]) for x, r in curves if h in r]
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, b, sd, se_b = ols(xs, ys)
        fits[h] = dict(a=a, b=b, sd=sd, se=se_b,
                       mx=sum(xs) / len(xs), my=sum(ys) / len(ys), pts=pts)
    return fits


def smooth_slopes(fits: dict[int, dict]) -> None:
    """Empirical-Bayes shrinkage of each hour's OLS slope toward a
    Gaussian-kernel-smoothed slope curve (see header, 2).

    The slope curve b(h) is smooth (it falls from ~0 at hour 2 to
    ~-0.86 at hour 20) but each hour's OLS slope carries a standard
    error of ~0.09, larger than the typical hour-to-hour change, so
    the raw slopes are mostly noise around the smooth curve. The
    kernel prior averages neighbouring hours' slopes; tau^2 is the
    method-of-moments excess variance of the slopes around that
    prior, scaled by KERNEL_MULT. Each line is re-pivoted through
    its per-hour mean point, so the fit at the training mean is
    unchanged -- only the slope (hence the extrapolation) moves.
    """
    hs = sorted(fits.keys())
    if len(hs) < 3:
        return
    bs = [fits[h]["b"] for h in hs]
    ses = [fits[h]["se"] for h in hs]
    prior: dict[int, float] = {}
    for h in hs:
        num = den = 0.0
        for j, bj in zip(hs, bs):
            k = math.exp(-0.5 * ((h - j) / KERNEL_BW) ** 2)
            num += k * bj
            den += k
        prior[h] = num / den if den else fits[h]["b"]
    tau2 = sum(max(0.0, (bs[i] - prior[hs[i]]) ** 2 - ses[i] ** 2)
               for i in range(len(hs))) / len(hs)
    tau2 *= KERNEL_MULT
    for h in hs:
        se2 = fits[h]["se"] ** 2
        if tau2 > 0 and se2 > 0:
            b_new = (fits[h]["b"] / se2 + prior[h] / tau2) / (1.0 / se2 + 1.0 / tau2)
        else:
            b_new = prior[h]
        fits[h]["b"] = b_new
        fits[h]["a"] = fits[h]["my"] - b_new * fits[h]["mx"]


def pooled_residual_sd(fits: dict[int, dict]) -> float:
    """One residual SD pooled over all per-hour fits (after slope smoothing)."""
    ss = 0.0
    nn = 0
    for f in fits.values():
        ss += sum((y - (f["a"] + f["b"] * x)) ** 2 for x, y in f["pts"])
        nn += len(f["pts"]) - 2
    return math.sqrt(ss / nn) if nn > 0 else 0.0


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


def hour_quantiles(p1: float, h: int, fit: dict | None,
                   pooled_sd: float, sig_scale: float) -> list[float | int]:
    """The five quantiles for hour h: OLS-conditioned median + normal band."""
    if fit is None or len(fit["pts"]) < MIN_CURVES:
        # Fallback: house persistence baseline for this hour.
        return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in Z]
    xs = [p[0] for p in fit["pts"]]
    ys = [p[1] for p in fit["pts"]]
    mean_all = mean(ys)
    a, b, sd = fit["a"], fit["b"], fit["sd"]
    med = mean_all + shrink_at(math.log(p1)) * ((a + b * math.log(p1)) - mean_all)
    # Pooled SD when available; per-hour SD only in the degenerate case.
    # Hour-varying asymmetric band (v4): the lower tail gets WIDTH_LO,
    # the upper tail falls linearly from WIDTH_HI_H2 at hour 2 to
    # WIDTH_HI_H24 at hour 24, because the residuals are right-skewed
    # most strongly in the early hours. The per-hour residual skewness
    # nudges the top wider still (SKEW_K), and the outer quantiles are
    # narrowed 5% (WIDTH_OUTER) because the pooled normal's far tails
    # are slightly too fat.
    resid = [y - (a + b * x) for x, y in fit["pts"]]
    sd_r = math.sqrt(sum(e * e for e in resid) / max(len(resid) - 2, 1)) or 1.0
    skew = mean([(e / sd_r) ** 3 for e in resid]) if len(resid) > 3 else 0.0
    t = (h - 2) / 22.0
    lo_m = WIDTH_LO
    hi_m = WIDTH_HI_H2 + (WIDTH_HI_H24 - WIDTH_HI_H2) * t
    s = max(0.0, min(2.0, skew))
    lo_m = max(0.4, lo_m * (1.0 - 0.3 * SKEW_K * s))
    hi_m = hi_m * (1.0 + SKEW_K * s)
    sig_base = (pooled_sd if pooled_sd > 0 else sd) * WIDTH * sig_scale
    qs = []
    for z in Z:
        mult = lo_m if z < 0 else hi_m
        if abs(z) >= 1.645:
            mult *= WIDTH_OUTER
        sig = max(sig_base * mult, SIGMA_FLOOR)
        qs.append(p1 * math.exp(med + z * sig))
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
    fits = compute_fits(curves)
    smooth_slopes(fits)
    pooled_sd = pooled_residual_sd(fits)
    # One-sided band narrowing for big targets (see header, 3).
    sig_scale = math.exp(-SIGMA_GAMMA * max(0.0, math.log(p1) - SIGMA_CENTER))
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, row["hour"], fits.get(row["hour"]),
                                           pooled_sd, sig_scale)}
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
