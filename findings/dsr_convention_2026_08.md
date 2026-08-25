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
constants: empirical `sd_trial_sharpes ≈ 0.7`,
`E[max Sharpe over 500 trials] ≈ 2.14`) gives a very different
answer on the same returns. Citing only 0.770 was quoting the
flattering convention without the label.

## Both conventions, same code, same returns

`deflated_sharpe_ratio(...)` at `PROJECT_NUM_TRIALS = 500` (pinned):

| Convention | `sharpe_std_dev` | DSR |
|---|---|---|
| Minimal (1/sqrt(T)) | null sampling std of one Sharpe estimate | **0.770** at the peak config; **0.736** neighborhood median (7/7 neighbors > 0.5) |
| Conservative (empirical) | pooled trial sd ≈ 0.7 | **≈ 0.05** at concat-OOS Sharpe +1.48; **≈ 0.000** on the venue-verified window (+0.72) |

Reference Sharpe layers (unchanged, from
`findings/han_28d_tsmom.md` and the validation phase): concat-OOS
Sharpe **+1.48**, venue-verified window **+0.72**.

## The primary statement (verbatim)

> Concat-OOS Sharpe +1.48 (verified window 0.72); DSR under the
> project's own conservative deflator (empirical
> `sd_trial_sharpes ≈ 0.7`) ≈ 0; the deploy case rests on cluster
> stability (neighborhood median DSR 0.736, 7/7 neighbors > 0.5)
> plus the forward test.

DSR 0.770 may still be cited, always labeled "under the 1/sqrt(T)
convention".

## Why the deploy case does not rest on the DSR headline

The headline flips from 0.770 to ≈ 0 on a single convention choice,
so it cannot carry a deploy decision by itself. What is not
convention-sensitive:

* **Cluster stability.** Under any one fixed convention, the
  (28, 60) neighborhood posts a high median (0.736 under 1/sqrt(T))
  with 7/7 neighbors above threshold — the config is a plateau, not
  a lone peak cherry-picked from noise.
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
