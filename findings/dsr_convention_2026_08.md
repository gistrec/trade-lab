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
project's own pinned conservative *assumption* (RESULTS.md pinned
constants, both annualized: `sd_trial_sharpes ≈ 0.7`,
`E[max Sharpe over 500 trials] ≈ 2.14`) gives a very different
answer on the same returns. Citing only 0.770 was quoting the
flattering convention without the label.

## Where 0.7 comes from — assumption, not measurement

`sd_trial_sharpes ≈ 0.7` is a **pinned conservative assumption**,
not an empirical estimate. Nothing in the repo derives it from a
trial-Sharpe panel:

* `findings/ctrend_proxy_price_only.md` §DSR — "a conservative
  pooled `sd_trial_sharpes = 0.7` representing the cross-project
  dispersion of trial Sharpes"; its checklist says "honest about
  conservative `sd_trial_sharpes` estimate".
* `findings/strategy_test_session_2026_05_29.md` §6 and
  `findings/mvrv_overlay.md` / `findings/hmm_regime_overlay.md`
  reuse the same pooled 0.7 as a verdict floor.
* No committed panel of trial Sharpes exists, and
  `deflated_sharpe_from_trial_sharpes` (the code path that would
  estimate the sd from an actual sweep, `np.std(..., ddof=1)`) is
  never called on the deployed config. `expected_max_sharpe`'s own
  docstring lists the alternative used here: "an a-priori estimate
  of how variable Sharpes are across the search space".

So the label everywhere in this branch is "pinned conservative
assumption". Calling it "empirical" (as the first draft of this
finding did) would lend the conservative DSR figures evidentiary
weight the project has not earned. The direction of the assumption
is what makes it usable: a *larger* assumed dispersion raises the
expected-max bar, so 0.7 is the pessimistic end — a smaller true
dispersion would raise DSR, not lower it. Pinning down the real
number would require a committed trial panel, which does not exist.

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
`PROJECT_NUM_TRIALS = 500` (pinned) on two distinct return series —
they are different statistical objects and are kept apart on
purpose:

* **Concat-OOS (walk-forward stitched)** — the object behind the
  0.770 headline: T = 2339 daily bars, 2020-01-01 → 2026-05-27,
  concat-OOS Sharpe +1.478.
* **Venue-verified window (direct frozen-config backtest)** — the
  object `findings/validation_multiexchange.md` §"RISK FLAG" calls
  authoritative: the frozen config (hash `ac8919…`) replayed on the
  Binance basket, sliced to 2022-01-21 → end of sample. That
  finding reports 1589 bars / Sharpe **+0.721** through 2026-05-28;
  recomputed today on the same parquets it is 1588 bars /
  **+0.7217** through 2026-05-27, because the local BTC snapshot
  now ends one bar earlier than the 2026-05-29 run. That one bar
  moves the Sharpe by 0.0003 and the DSR by < 0.001.

| Convention | `sharpe_std_dev` (per-period) | DSR (computed) |
|---|---|---|
| Minimal (1/sqrt(T)) | `1/sqrt(2339) ≈ 0.0207` | **0.770** at the peak config; **0.736** neighborhood median, 7/7 neighbors > 0.5 (both figures exist only under this convention) |
| Conservative (pinned assumption) | `0.7 / sqrt(365) ≈ 0.0366` | **0.037** on the concat-OOS series (Sharpe +1.48); **0.002** (0.0016) on the venue-verified frozen-config backtest (+0.72) |

The venue-verified 0.002 was **recomputed from the validation
run's own object**, not from a slice of the stitched series: the
first draft of this finding deflated the 2022-01-21 → 2026-05-27
slice of the walk-forward concat series (T = 1588, Sharpe +0.722)
and labelled the result "venue-verified", which mixed two objects.
Rebuilding the basket + frozen config directly (the
`validation_multiexchange.md` path) and deflating those 1588 daily
returns gives DSR = **0.0016**; the reported 0.002 is unchanged in
substance, only its provenance is now the authoritative run. Both
objects land at DSR ≈ 0 under this deflator, so the verdict never
depended on which one was used.

Reference Sharpe layers (unchanged, from
`findings/han_28d_tsmom.md` and the validation phase): concat-OOS
Sharpe **+1.48**, venue-verified window **+0.72**.

## The primary statement (verbatim)

> Concat-OOS Sharpe +1.48 (verified window 0.72); DSR under the
> project's own conservative deflator (pinned conservative
> assumption `sd_trial_sharpes ≈ 0.7` annualized, `0.7/sqrt(365)`
> per-period — an a-priori pool dispersion, no trial panel exists)
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

## Reproducing the venue-verified 0.0016

No new script is committed (this is a reporting convention, not a
new experiment). The steps are the `validation_multiexchange.md`
path plus one deflator call, on the seven `data/binance_*_1d.parquet`
files:

1. `build_crypto_market_index(candles, ...)` and
   `TimeSeriesMomentumStrategy(...)` from `PRODUCTION_CONFIG`
   (frozen hash `ac8919…`), then `run_backtest(...)` — identical to
   `scripts/validation_test1_multiexchange.py::run_pipeline`. Trim
   the panel to the common last bar 2026-05-27 first: the local BTC
   parquet ends there while the other six end 2026-05-28, and the
   index builder refuses a hole rather than shrinking the basket
   silently.
2. Slice `backtest.returns` to `2022-01-21 … 2026-05-27`
   (1588 bars; annualized Sharpe +0.7217; full sample 2018-01-01 →
   2026-05-27 gives +1.377, matching the validation finding).
3. `deflated_sharpe_ratio(returns, num_trials=500,
   sharpe_std_dev=0.7/sqrt(365))` → **0.0016**.

For reference the same slice under the minimal deflator
(`1/sqrt(1588)`) gives 0.061 — the convention gap is the whole
point of this finding.

Reproducibility caveat: `data/` is gitignored, so the parquets are a
local snapshot, not committed history (the "committed to the repo"
line in `validation_multiexchange.md` § Reproducing is inaccurate on
that point). Anyone re-fetching from Binance will get a longer
sample and slightly different bar counts; the DSR verdict (≈ 0
under the conservative deflator) is not close to any threshold.

## Non-changes

* `walk_forward_v2.py` computation untouched (1/sqrt(T) as before);
  only the comment was corrected — the earlier "that's what López
  de Prado uses" justification was unsupported, and "empirical trial
  sd" was relabelled to the pinned conservative assumption it is.
* `PROJECT_NUM_TRIALS = 500` unchanged. This is a reporting
  convention, not a new parametric search; zero new trials.
* The 0.7 assumption itself is unchanged — only its label. No
  number moved except the venue-verified DSR's provenance
  (0.002 → 0.0016 from the authoritative object).
