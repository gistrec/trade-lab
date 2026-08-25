# DSR reporting convention — two-layer statement (owner decision)

**Date:** 2026-08-25
**Trigger:** deep review 2026-08-24, finding #9 (issue #13).
**Decision (owner):** the honest two-layer statement below becomes
the primary citation everywhere alive (RESULTS.md, CLAUDE.md, code
comments); DSR 0.770 stays as a clearly labeled secondary figure
under the 1/sqrt(T) convention. No computation changes — both
numbers remain reproducible.

## The problem

The project's headline "DSR 0.770" depends on one input the data
does not pin down: `sharpe_std_dev`, the dispersion of the trial
pool under the null. `walk_forward_v2.py` uses the *minimal*
deflator 1/sqrt(T) — the null sampling std of a single per-period
Sharpe estimate. That corrects for estimation noise only; it does
not model how far the best of 500 dispersed trials drifts. The
project's own pinned conservative estimate (RESULTS.md pinned
constants, both annualized: empirical `sd_trial_sharpes ≈ 0.7`,
`E[max Sharpe over 500 trials] ≈ 2.14`) gives a very different
answer on the same returns. Citing only 0.770 was quoting the
flattering convention without the label.

## Both conventions, same code, same returns

Dimensional convention first: `deflated_sharpe_ratio(...)` works
per-period — it compares the non-annualized daily Sharpe against
`sharpe_std_dev`. The pinned trial sd ≈ 0.7 is annualized
(annualization factor 365, daily bars), so it must be de-annualized
before use: `0.7 / sqrt(365) ≈ 0.0366` per-period. Passing raw 0.7
would set the expected-max bar at ≈ 2.14 *per-period* (the pinned
annualized value in the wrong unit) and return DSR = 0.0 exactly —
a dimensional artifact, not a measurement. Sanity check: E[max] at
sd 0.0366 is 0.112 per-period = 2.14 annualized, matching the
pinned constant.

Computed with `deflated_sharpe_ratio(...)` at
`PROJECT_NUM_TRIALS = 500` (pinned) on the stitched OOS series of
the deployed config (T = 2339 daily bars, 2020-01-01 → 2026-05-27,
concat-OOS Sharpe +1.478; verified-window slice
2022-01-21 → 2026-05-27, T = 1588, Sharpe +0.722):

| Convention | `sharpe_std_dev` (per-period) | DSR (computed) |
|---|---|---|
| Minimal (1/sqrt(T)) | `1/sqrt(2339) ≈ 0.0207` | **0.770** at the peak config; **0.736** neighborhood median, 7/7 neighbors > 0.5 (both figures exist only under this convention) |
| Conservative (empirical) | `0.7 / sqrt(365) ≈ 0.0366` | **0.037** at concat-OOS Sharpe +1.48; **0.002** on the venue-verified window (+0.72) |

Reference Sharpe layers (unchanged, from
`findings/han_28d_tsmom.md` and the validation phase): concat-OOS
Sharpe **+1.48**, venue-verified window **+0.72**.

## The primary statement (verbatim)

> Concat-OOS Sharpe +1.48 (verified window 0.72); DSR under the
> project's own conservative deflator (empirical
> `sd_trial_sharpes ≈ 0.7` annualized, `0.7/sqrt(365)` per-period)
> = 0.037 ≈ 0. The deploy case rests on parameter stability stated
> convention-free — all seven neighbor configs land in the raw
> concat-OOS Sharpe band +1.37…+1.49, no lone peak — plus the
> forward test.

DSR 0.770 may still be cited, always labeled "under the 1/sqrt(T)
convention"; the same label applies to the cluster's DSR-threshold
form (median 0.736, 7/7 > 0.5).

## Why the deploy case does not rest on the DSR headline

The headline flips from 0.770 to ≈ 0 on a single convention choice,
so it cannot carry a deploy decision by itself. What carries it:

* **Parameter stability, stated convention-free.** All seven
  neighbor configs of (28, 60) post raw concat-OOS Sharpe in the
  tight band +1.37…+1.49 (`findings/cluster_stability.md`) — the
  config is a plateau, not a lone peak. Honest limits: the
  neighbors are not independent evidence — they hold identical
  positions on 86–97% of days and their daily returns correlate at
  0.97–0.99 with the deployed config (one bet measured seven times);
  stability rules out single-point cherry-picking, nothing more.
  The familiar "median DSR 0.736, 7/7 > 0.5" is this same fact
  expressed under the secondary 1/sqrt(T) convention — under the
  conservative deflator all seven sit at DSR ≈ 0, so the threshold
  form cannot back the primary statement (that would be circular).
* **The forward test.** Live observation on the target venue with
  the real order pipeline measures the thing DSR only bounds.

Those two are the deploy case. The DSR figures are reproducibility
artifacts of the chosen deflator and are reported as layers, per the
"layered honesty" principle (CLAUDE.md).

## Non-changes

* `walk_forward_v2.py` computation untouched (1/sqrt(T) as before);
  only the comment was corrected — the earlier "that's what López
  de Prado uses" justification was unsupported.
* `PROJECT_NUM_TRIALS = 500` unchanged. This is a reporting
  convention, not a new parametric search; zero new trials.
