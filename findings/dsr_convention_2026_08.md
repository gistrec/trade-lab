# DSR reporting convention: two-layer statement (owner decision)

**Date:** 2026-08-25
**Trigger:** deep review 2026-08-24, finding #9 (issue #13).
**Decision (owner):** the two-layer statement below becomes the
primary citation everywhere alive (RESULTS.md, CLAUDE.md, code
comments). DSR 0.770 stays as a clearly labeled secondary figure
under the 1/sqrt(T) convention. No computation changes.

**Reproducibility (revised 2026-08-26).** Both numbers in this finding
are computed on two derived return series, now committed as immutable
artifacts:

* `docs/results/dsr_convention_2026_08_concat_oos_returns.csv`:
  2339 daily bars, the walk-forward concatenated OOS series.
* `docs/results/dsr_convention_2026_08_venue_replay_returns.csv`:
  1588 daily bars, the venue-verified replay window.

Each file has a `#` provenance header with the frozen-config hash
`ac8919…`, the walk-forward parameters, the per-asset data vintage
(local `data/binance_*_1d.parquet` snapshot as of 2026-08-26, panel
trimmed to the common last bar 2026-05-27), the generating commit,
and the Sharpe and DSR values the file reproduces. Read them with
`pandas.read_csv(path, comment="#", index_col=0, parse_dates=True)`
and pass them to `deflated_sharpe_ratio(..., num_trials=500, ...)`.

The original wording claimed both numbers "remain reproducible", which
was not true from committed history: `data/` is gitignored, no parquet
snapshot is committed, and a refetch yields a longer sample with
different bar counts. About 96 KB of committed floats makes the claim
true from this repository alone and also fixes the vintage, so a later
refetch cannot move the published numbers. The parquets stay
uncommitted; every DSR figure here is computed on the derived series.

## The problem

The headline "DSR 0.770" depends on one input the data does not pin
down: `sharpe_std_dev`, the dispersion of the trial pool under the
null. `walk_forward_v2.py` uses the minimal deflator 1/sqrt(T), the
null sampling std of a single per-period Sharpe estimate.

Both conventions apply the N=500 extreme-value correction. In
`src/trade_lab/backtest/dsr.py`, `deflated_sharpe_ratio` passes
`sharpe_std_dev` to `expected_max_sharpe(num_trials, sharpe_std_dev)`,
which multiplies it by the Bailey-LdP factor
`(1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))` and uses the product as
the bar `SR_0`. At N = 500 the factor is ≈ 3.053, so the two
conventions give two bars from the same correction:

* minimal: 1/sqrt(2339) ≈ 0.0207 × 3.053 → SR_0 ≈ 0.0631 per-period
  (≈ 1.21 annualized);
* conservative: 0.0366 × 3.053 → SR_0 ≈ 0.1118 per-period (≈ 2.14
  annualized, matching the pinned constant).

The difference is in what 1/sqrt(T) assumes about the trials. It
treats the cross-trial dispersion as pure sampling noise, as if the
500 trials were 500 re-estimates of one zero-skill strategy on the
same length of data. The project's trials were different strategies,
lookbacks and overlays, and their Sharpes spread far wider than
re-estimation noise. The narrower dispersion sets a lower bar, which
is what produces 0.770. The pinned conservative assumption (RESULTS.md
pinned constants, both annualized: `sd_trial_sharpes ≈ 0.7`,
`E[max Sharpe over 500 trials] ≈ 2.14`) puts a pool-sized dispersion
into the same formula and gives a very different answer on the same
returns. Citing 0.770 alone quoted the lenient convention without
saying so.

## Where 0.7 comes from: assumption, not measurement

`sd_trial_sharpes ≈ 0.7` is a pinned conservative assumption. Nothing
in the repo derives it from a trial-Sharpe panel:

* `findings/ctrend_proxy_price_only.md` §DSR describes "a conservative
  pooled `sd_trial_sharpes = 0.7` representing the cross-project
  dispersion of trial Sharpes"; its checklist says "honest about
  conservative `sd_trial_sharpes` estimate".
* `findings/strategy_test_session_2026_05_29.md` §6,
  `findings/mvrv_overlay.md` and `findings/hmm_regime_overlay.md`
  reuse the same pooled 0.7 as a verdict floor.
* No panel of trial Sharpes is committed.
  `deflated_sharpe_from_trial_sharpes`, the code path that would
  estimate the sd from an actual sweep (`np.std(..., ddof=1)`), is
  never called on the deployed config. The docstring of
  `expected_max_sharpe` names the alternative used here: "an a-priori
  estimate of how variable Sharpes are across the search space".

The label everywhere in this branch is therefore "pinned conservative
assumption". Calling it empirical would give the conservative DSR
figures evidentiary weight the project has not earned. The direction
of the assumption is what makes it usable: a larger assumed dispersion
raises the expected-max bar, so 0.7 sits at the pessimistic end, and a
smaller true dispersion would raise DSR. Measuring the real number
requires a committed trial panel, and none exists.

## Both conventions, same code, same returns

`deflated_sharpe_ratio(...)` works per-period: it compares the daily,
non-annualized Sharpe against `sharpe_std_dev`. The pinned trial
sd ≈ 0.7 is annualized (factor 365, daily bars) and must be converted
first: `0.7 / sqrt(365) ≈ 0.0366` per-period. Passing raw 0.7 sets the
expected-max bar at ≈ 2.14 per-period, the annualized value in the
wrong unit, and returns DSR = 0.0 for both series measured here. That
zero is a unit error. A series with a per-period Sharpe above 2.14
would still score positive, but ours are nowhere near it. Check:
E[max] at sd 0.0366 is 0.112 per-period, or 2.14 annualized, which
matches the pinned constant.

DSR is computed with `deflated_sharpe_ratio(...)` at
`PROJECT_NUM_TRIALS = 500` (pinned) on two return series. They are
different statistical objects and are kept apart on purpose.

* **Concat-OOS (walk-forward stitched).** The series behind the 0.770
  headline and the only selection-OOS series here: T = 2339 daily
  bars, 2020-01-01 → 2026-05-27, concat-OOS Sharpe +1.478. Committed
  as `docs/results/dsr_convention_2026_08_concat_oos_returns.csv`.
* **Venue-verified replay window (direct frozen-config backtest).**
  The series `findings/validation_multiexchange.md` §"RISK FLAG" calls
  authoritative: the frozen config (hash `ac8919…`) replayed on the
  Binance basket from 2022-01-21 to the end of the sample. Committed
  as `docs/results/dsr_convention_2026_08_venue_replay_returns.csv`.

Cite the replay window as a fixed-config historical replay, never as
a second out-of-sample result. The config was selected before this
window was measured, the window overlaps the walk-forward OOS sample
instead of extending past it, and none of it is forward data. It adds
no protection against selection bias beyond the concat-OOS number.
What it does test is venue agreement (whether Binance prices and
independently sourced Bybit prices produce the same strategy) and how
much of the historical edge falls in the venue-verifiable era: much
less than the full-sample Sharpe suggests, +0.72 against +1.38.

The validation finding reports 1589 bars and Sharpe +0.721 through
2026-05-28. Recomputed on the same parquets it is 1588 bars and
+0.7217 through 2026-05-27, because the local BTC snapshot now ends
one bar before the 2026-05-29 run. The missing bar moves Sharpe by
0.0003 and DSR by less than 0.001.

| Convention | `sharpe_std_dev` (per-period) | DSR (computed) |
|---|---|---|
| Minimal (1/sqrt(T)), also the project's gate convention | `1/sqrt(2339) ≈ 0.0207` | **0.770** for the deployed config; **0.736** cluster median over its 7 members (the deployed config + 6 neighbours), 6/6 neighbours > 0.5 (both figures exist only under this convention) |
| Conservative (pinned assumption), the primary *reported* convention | `0.7 / sqrt(365) ≈ 0.0366` | **0.037** on the concat-OOS series (Sharpe +1.48); **0.002** (0.0016) on the venue-verified *replay* window (+0.72, fixed-config historical replay, not selection-OOS) |

The venue-verified 0.002 is computed on the validation run's own
series: the basket and frozen config rebuilt directly (the
`validation_multiexchange.md` path), giving 1588 daily returns and
DSR = 0.0016. It is not a slice of the stitched walk-forward series.
Deflating that slice (T = 1588, Sharpe +0.722) also lands at DSR ≈ 0,
so the verdict does not depend on which series is used.

Reference Sharpe layers (unchanged, from `findings/han_28d_tsmom.md`
and the validation phase): concat-OOS Sharpe **+1.48**
(selection-OOS), venue-verified **replay** window **+0.72**
(fixed-config historical replay, not OOS).

## The primary statement (verbatim)

> Concat-OOS Sharpe +1.48 (venue-verified *replay* window 0.72 —
> a fixed-config historical replay of the frozen config over
> 2022-01-21 → 2026-05-27 checking venue agreement, not
> selection-OOS and not forward data); DSR under the
> project's own conservative deflator (pinned conservative
> assumption `sd_trial_sharpes ≈ 0.7` annualized, `0.7/sqrt(365)`
> per-period — an a-priori pool dispersion, no trial panel exists)
> = 0.037 ≈ 0. The deploy case rests on parameter stability stated
> convention-free — the deployed config plus its six neighbour
> configs (seven cluster members, i.e. six confirmations, not seven)
> land in the raw concat-OOS Sharpe band +1.37…+1.49, no lone peak —
> plus the forward test.

DSR 0.770 may still be cited, always labeled "under the 1/sqrt(T)
convention". The same label applies to the cluster's DSR-threshold
form (median 0.736 over the 7 cluster members, 6/6 neighbours > 0.5).

**Scope of that labeling rule.** It binds every statement about the
deployed config and every place a gate or threshold is defined, since
those are what the project acts on. It does not require rewriting the
historical per-strategy rows in RESULTS.md (#2–#14 and the family
sections). Those rows carry one blanket footnote declaring the whole
table minimal-convention, which is accurate: none of them has been
recomputed under the conservative deflator. One scoping sentence and
a column footnote cover them, and no unlabeled DSR is left anywhere.

**The gate keeps the minimal convention.** The deploy gate ("DSR > 0.5
at N=500, cluster-stable") was and remains defined on 1/sqrt(T). This
decision changed reporting only: nothing was re-run and no threshold
was reset. RESULTS.md says, verbatim, that the deployed config
**passed the gate as the gate is defined and would not pass a gate
restated on the conservative deflator**. No config in this project
would pass a gate restated that way, which is why the bar was left as
it is. For the same reason the deploy case is argued from the Sharpe
plateau, which needs no convention, and from the forward test, not
from any DSR number.

## Why the deploy case does not rest on the DSR headline

The headline moves from 0.770 to ≈ 0 on one convention choice, so it
cannot carry a deploy decision. Two things carry it:

* **Parameter stability, independent of the convention.** The
  deployed (28, 60) config and its six neighbours (seven cluster
  members) post raw concat-OOS Sharpe in the band +1.37…+1.49
  (`findings/cluster_stability.md`). The six neighbours alone span
  the same band, so the plateau does not depend on the deployed point.
  The neighbours are not independent evidence: they hold identical
  positions on 86–97% of days, and their daily returns correlate at
  0.97–0.99 with the deployed config, so this is one bet measured six
  more times. Stability rules out picking a single lucky point and
  nothing more. The deployed config is not one of its own
  confirmations: "7 of 7 pass" is one deployed point plus six
  confirmations. "Median DSR 0.736, 7/7 > 0.5" is the same fact under
  the secondary 1/sqrt(T) convention. Under the conservative deflator
  all seven sit at DSR ≈ 0, so the threshold form cannot support the
  primary statement without circularity.
* **The forward test.** Live trading on the target venue through the
  real order pipeline measures what DSR can only bound.

The DSR figures are reproducibility artifacts of the chosen deflator
and are reported as separate layers, following the "layered honesty"
principle in CLAUDE.md.

## Reproducing the venue-verified 0.0016

Shortest path, no private data needed: read the committed series and
call the deflator once.

```python
import math, pandas as pd
from trade_lab.backtest.dsr import deflated_sharpe_ratio

r = pd.read_csv(
    "docs/results/dsr_convention_2026_08_venue_replay_returns.csv",
    comment="#", index_col=0, parse_dates=True,
)["net_return"]
deflated_sharpe_ratio(r, num_trials=500, sharpe_std_dev=0.7/math.sqrt(365))
# -> 0.0016184134334346623
```

The same two lines on
`dsr_convention_2026_08_concat_oos_returns.csv` give 0.0368 at
`0.7/sqrt(365)` and 0.7705 at `1/sqrt(2339)`.

Long path, from the parquet snapshot that produced those files. No
script is committed for it, because this is a reporting convention
and not a new experiment. The steps are the
`validation_multiexchange.md` path plus one deflator call, on the
seven `data/binance_*_1d.parquet` files:

1. `build_crypto_market_index(candles, ...)` and
   `TimeSeriesMomentumStrategy(...)` from `PRODUCTION_CONFIG`
   (frozen hash `ac8919…`), then `run_backtest(...)`, identical to
   `scripts/validation_test1_multiexchange.py::run_pipeline`. First
   trim the panel to the common last bar 2026-05-27: the local BTC
   parquet ends there while the other six end 2026-05-28, and the
   index builder refuses a hole instead of shrinking the basket.
2. Slice `backtest.returns` to `2022-01-21 … 2026-05-27`
   (1588 bars; annualized Sharpe +0.7217; the full sample 2018-01-01 →
   2026-05-27 gives +1.377, matching the validation finding).
3. `deflated_sharpe_ratio(returns, num_trials=500,
   sharpe_std_dev=0.7/sqrt(365))` → **0.0016**.

The same slice under the minimal deflator (`1/sqrt(1588)`) gives
0.061. That gap between conventions is what this finding is about.

What a reader can and cannot reproduce:

* **From this repository, exactly:** every DSR and Sharpe figure in
  this finding. The two committed CSVs are the series the numbers
  were computed on, so the deflator calls reproduce bit for bit.
* **Not from this repository:** the step that produces those series.
  `data/` is gitignored, so the seven Binance parquets are a local
  snapshot. The "committed to the repo" line in
  `validation_multiexchange.md` § Reproducing is wrong on this point
  (see the addendum there). A fresh Binance fetch gives a longer
  sample and slightly different bar counts than the 2026-08-26 vintage
  pinned in the CSV headers, so a rebuilt series will not match row
  for row.
* **Effect on the verdict:** none. DSR ≈ 0 under the conservative
  deflator is far from any threshold, and the one-bar vintage
  difference against the 2026-05-29 run moved Sharpe by 0.0003 and DSR
  by less than 0.001.

## Non-changes

* `walk_forward_v2.py` computation untouched (1/sqrt(T) as before).
  Only the comment changed: the claim "that's what López de Prado
  uses" had no support, and "empirical trial sd" is relabeled as the
  pinned conservative assumption.
* `PROJECT_NUM_TRIALS = 500` unchanged. This is a reporting
  convention, not a parametric search, so it adds no trials.
* The 0.7 assumption is unchanged; only its label moved. The one
  number that changed is the provenance of the venue-verified DSR
  (0.002 → 0.0016, from the authoritative series).
* The two CSVs under `docs/results/` are derived artifacts, not
  experiments or code. They pin series this finding already reported.
  No trials, no behavior change.

## Revision 2026-08-26 (third Codex review round)

Documentation-only pass over this finding and the docs that cite it:

1. **Corrected a factual error.** The first draft said 1/sqrt(T) "does
   not model how far the best of 500 dispersed trials drifts". That
   was wrong: `deflated_sharpe_ratio` feeds `sharpe_std_dev` into
   `expected_max_sharpe`, so both conventions apply the N=500
   extreme-value correction. The actual limitation, stated in
   § "The problem", is that 1/sqrt(T) treats cross-trial dispersion as
   pure sampling noise instead of the wider dispersion of the strategy
   pool searched. The same wording was corrected in RESULTS.md and in
   the `walk_forward_v2.py` comment.
2. **Made "remain reproducible" true** by committing the two derived
   return series (see the Reproducibility note at the top).
3. **Labeled the venue-verified figure as a fixed-config historical
   replay** wherever it is cited (here, RESULTS.md, CLAUDE.md), so it
   no longer reads as an independent OOS result under the project's
   "OOS unless marked" convention.
4. **Separated the gate convention from the reporting convention** in
   RESULTS.md, which had claimed both that the config passed
   DSR > 0.5 and that its primary DSR is ≈ 0.
5. **Scoped the labeling rule**, so the historical per-strategy rows
   in RESULTS.md are covered by one blanket footnote.
6. **Recomputed the venue-verified 0.002 on the validation run's own
   series.** The first draft had deflated a slice of the walk-forward
   concat series and called it "venue-verified", mixing two objects.
7. **Relabeled 0.7 as an assumption.** The first draft called it
   "empirical".

No number changed. No computation changed.

## Revision 2026-08-26 (fourth review round): counting and labels

Documentation-only again: no computation, no re-run, no new trials.

1. **The cluster is 1 + 6, not 7 confirmations.** Every live doc that
   said "7 neighbor configs" or "7 of 7 pass" counted the deployed
   `(28, 60)` as one of its own confirmations
   (`findings/cluster_stability.md` § 2 lists it inside the
   seven-member TSMOM short-ensemble cluster). Here, in RESULTS.md and
   in CLAUDE.md it now reads **the deployed config plus six
   neighbours**. The cluster still has 7 members and the published
   median 0.736 is still over all 7, but the evidence for the deployed
   point is **6 of 6 neighbours**. The neighbour-only median under the
   same 1/sqrt(T) convention is 0.726 (0.671, 0.693, 0.716, 0.736,
   0.750, 0.782; arithmetic on the finding's per-variant table,
   nothing re-run). The raw-Sharpe band stays +1.37…+1.49 with or
   without the deployed +1.48, so the plateau survives dropping the
   deployed point.
2. **"0.770 at the peak config" was wrong** and now reads "0.770 for
   the deployed config". The cluster peak is `(30, 60)` at DSR 0.782 /
   Sharpe +1.49 (same table). The RESULTS.md line that called
   `(28, 60)` the family's best individual got the same correction.
3. **One scoping note per live doc instead of per-figure edits.**
   RESULTS.md gained a file-wide DSR-convention paragraph above the
   table (footnote ¹ stays as the per-number repeat). README.md, which
   quotes DSR figures from `docs/results/` without naming a
   convention, gained the same note covering itself and the
   commit-pinned writeups it links. Those writeups and `findings/` are
   dated artifacts and are not rewritten.
4. **The deploy record's DSR units are corrected by addendum.**
   `findings/production_config_v1.md` reported the ratified figures as
   "DSR @ N=500, sd=0.7 ≈ 0.000" with "`E[max SR]` ≈ 2.137
   per-period". Passing raw 0.7 to `deflated_sharpe_ratio` is the unit
   error described in § "Both conventions": it returns 0.0 for any
   series, and 2.137 is the annualized bar (per-period 0.1118). Under
   findings immutability the historical lines stay, and the dated
   addendum there gives the conservative-convention figures (0.0016
   replay / 0.037 concat-OOS) and points here.
