# DSR reporting convention — two-layer statement (owner decision)

**Date:** 2026-08-25
**Trigger:** deep review 2026-08-24, finding #9 (issue #13).
**Decision (owner):** the honest two-layer statement below becomes
the primary citation everywhere alive (RESULTS.md, CLAUDE.md, code
comments); DSR 0.770 stays as a clearly labeled secondary figure
under the 1/sqrt(T) convention. No computation changes.

**Reproducibility (revised 2026-08-26).** The original wording here
claimed both numbers "remain reproducible" while `data/` is
gitignored, no parquet snapshot is committed, and this finding itself
concedes that a refetch yields a longer sample and different bar
counts. As written the claim was not true from committed history
alone. It is now made true by committing the two derived return
series as immutable artifacts:

* `docs/results/dsr_convention_2026_08_concat_oos_returns.csv` —
  2339 daily bars, the walk-forward concatenated OOS series.
* `docs/results/dsr_convention_2026_08_venue_replay_returns.csv` —
  1588 daily bars, the venue-verified replay window.

Each file carries a `#` provenance header naming the frozen-config
hash `ac8919…`, the walk-forward parameters, the per-asset data
vintage (local `data/binance_*_1d.parquet` snapshot as of
2026-08-26, panel trimmed to the common last bar 2026-05-27), the
generating commit, and the Sharpe / DSR values the file reproduces.
Read them with
`pandas.read_csv(path, comment="#", index_col=0, parse_dates=True)`
and feed them to `deflated_sharpe_ratio(..., num_trials=500, ...)`.
Route chosen deliberately: ~96 KB of committed floats is a cheap
price for turning "reproducible if you happen to hold the same
private parquet snapshot" into "reproducible from this repository",
and it also fixes the vintage so a later refetch cannot silently
move the published numbers. The parquets themselves stay
uncommitted — the artifacts are the *derived* series, which is the
object every DSR figure in this finding is actually computed on.

## The problem

The project's headline "DSR 0.770" depends on one input the data
does not pin down: `sharpe_std_dev`, the dispersion of the trial
pool under the null. `walk_forward_v2.py` uses the *minimal*
deflator 1/sqrt(T) — the null sampling std of a single per-period
Sharpe estimate.

Be precise about what that does and does not skip, because an
earlier draft of this finding got it wrong. It said 1/sqrt(T)
"does not model how far the best of 500 dispersed trials drifts."
**That is false.** Read `src/trade_lab/backtest/dsr.py`:
`deflated_sharpe_ratio` passes `sharpe_std_dev` straight into
`expected_max_sharpe(num_trials, sharpe_std_dev)`, which multiplies
it by the Bailey-LdP extreme-value factor
`(1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))` at N = 500 and uses the
product as the bar `SR_0`. The N=500 extreme-value correction is
applied in **both** conventions; the conventions differ only in the
number it is applied to. At N = 500 that factor is ≈ 3.053, so:
1/sqrt(2339) ≈ 0.0207 × 3.053 → SR_0 ≈ 0.0631 per-period
(≈ 1.21 annualized), versus 0.0366 × 3.053 → SR_0 ≈ 0.1118
per-period (≈ 2.14 annualized, matching the pinned constant). Same
correction, two different bars.

The real limitation is narrower. 1/sqrt(T) assumes the cross-trial
dispersion comes **only from sampling noise** — as if the 500 trials
were 500 re-estimates of one and the same zero-skill strategy on the
same length of data. The trials the project actually ran are not
that: they are different strategies, lookbacks and overlays, whose
Sharpes spread out far more than re-estimation noise would. Using
the narrower dispersion sets a lower bar, and a lower bar is what
produces the flattering 0.770. The project's own pinned conservative
*assumption* (RESULTS.md pinned constants, both annualized:
`sd_trial_sharpes ≈ 0.7`, `E[max Sharpe over 500 trials] ≈ 2.14`)
substitutes a pool-sized dispersion into the same formula and gives
a very different answer on the same returns. Citing only 0.770 was
quoting the flattering convention without the label.

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
  0.770 headline, and the only genuinely selection-OOS series here:
  T = 2339 daily bars, 2020-01-01 → 2026-05-27, concat-OOS Sharpe
  +1.478. Committed as
  `docs/results/dsr_convention_2026_08_concat_oos_returns.csv`.
* **Venue-verified replay window (direct frozen-config backtest)** —
  the object `findings/validation_multiexchange.md` §"RISK FLAG"
  calls authoritative: the frozen config (hash `ac8919…`) replayed
  on the Binance basket, sliced to 2022-01-21 → end of sample.
  **Label this correctly wherever it is cited: it is a fixed-config
  HISTORICAL REPLAY, not a second out-of-sample result.** The config
  was already selected before this window was measured; the window
  overlaps the walk-forward OOS sample rather than extending past
  it; and none of it is forward data. What the replay tests is
  *venue agreement* — do Binance prices and independently sourced
  Bybit prices produce the same strategy — plus how much of the
  historical edge sits in the venue-verifiable era (a lot less than
  the full-sample Sharpe suggests: +0.72 vs +1.38). It adds no
  selection-bias protection on top of the concat-OOS number. That
  finding reports 1589 bars / Sharpe **+0.721** through 2026-05-28;
  recomputed today on the same parquets it is 1588 bars /
  **+0.7217** through 2026-05-27, because the local BTC snapshot
  now ends one bar earlier than the 2026-05-29 run. That one bar
  moves the Sharpe by 0.0003 and the DSR by < 0.001. Committed as
  `docs/results/dsr_convention_2026_08_venue_replay_returns.csv`.

| Convention | `sharpe_std_dev` (per-period) | DSR (computed) |
|---|---|---|
| Minimal (1/sqrt(T)) — also the project's gate convention | `1/sqrt(2339) ≈ 0.0207` | **0.770** for the deployed config; **0.736** cluster median over its 7 members (the deployed config + 6 neighbours), 6/6 neighbours > 0.5 (both figures exist only under this convention) |
| Conservative (pinned assumption) — the primary *reported* convention | `0.7 / sqrt(365) ≈ 0.0366` | **0.037** on the concat-OOS series (Sharpe +1.48); **0.002** (0.0016) on the venue-verified *replay* window (+0.72 — fixed-config historical replay, not selection-OOS) |

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
Sharpe **+1.48** (selection-OOS), venue-verified **replay** window
**+0.72** (fixed-config historical replay, not OOS).

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
convention"; the same label applies to the cluster's DSR-threshold
form (median 0.736 over the 7 cluster members, 6/6 neighbours > 0.5).

**Scope of that labeling rule, so it does not become
self-contradicting.** It binds every statement about the **deployed
config** and every place a **gate or threshold** is defined — those
are what the project acts on. It does not oblige a rewrite of the
historical per-strategy rows in RESULTS.md (#2–#14 and the family
sections): those carry one blanket footnote declaring the whole
table minimal-convention, which is accurate — none of those figures
has ever been recomputed under the conservative deflator. One
scoping sentence plus a column footnote, not dozens of per-row
edits; but no bare, unscoped DSR is left standing anywhere.

**And the gate is not the reporting convention.** The project's
deploy gate ("DSR > 0.5 at N=500, cluster-stable") was and remains
defined on the *minimal* convention. This decision moved reporting,
not gating: nothing was re-run and no threshold was re-set. The
honest phrasing, used verbatim in RESULTS.md, is that the deployed
config **passed the gate as the gate is defined and would not pass
a gate restated on the conservative deflator** — under which no
config in this project would pass, which is why the bar was not
restated. That is also why the deploy case is argued from the
convention-free Sharpe plateau plus the forward test rather than
from any DSR number.

## Why the deploy case does not rest on the DSR headline

The headline flips from 0.770 to ≈ 0 on a single convention choice,
so it cannot carry a deploy decision by itself. What carries it:

* **Parameter stability, stated convention-free.** The deployed
  (28, 60) config **plus its six neighbours** — seven cluster members
  in total — post raw concat-OOS Sharpe in the tight band
  +1.37…+1.49 (`findings/cluster_stability.md`); the six neighbours
  alone span that same band, so the plateau does not depend on the
  deployed point being in it. Honest limits: the neighbours are not
  independent evidence — they hold identical positions on 86–97% of
  days and their daily returns correlate at 0.97–0.99 with the
  deployed config (one bet measured six more times); stability rules
  out single-point cherry-picking, nothing more. And the deployed
  config must not be counted among its own confirmations: the
  familiar "7 of 7 pass" is one deployed point plus **six**
  confirmations. The familiar "median DSR 0.736, 7/7 > 0.5" is this
  same fact expressed under the secondary 1/sqrt(T) convention —
  under the conservative deflator all seven sit at DSR ≈ 0, so the
  threshold form cannot back the primary statement (that would be
  circular).
* **The forward test.** Live observation on the target venue with
  the real order pipeline measures the thing DSR only bounds.

Those two are the deploy case. The DSR figures are reproducibility
artifacts of the chosen deflator and are reported as layers, per the
"layered honesty" principle (CLAUDE.md).

## Reproducing the venue-verified 0.0016

Shortest path (no private data needed): read the committed series
and make one deflator call.

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

The same two lines against
`dsr_convention_2026_08_concat_oos_returns.csv` give 0.0368 at
`0.7/sqrt(365)` and 0.7705 at `1/sqrt(2339)`.

Long path, from the parquet snapshot (what produced those files).
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

Reproducibility caveat, stated precisely — what a reader can and
cannot reproduce:

* **Can, exactly, from this repository:** every DSR and Sharpe
  figure in this finding. The two committed CSVs are the exact
  derived series the numbers were computed on, so the deflator calls
  reproduce bit-for-bit.
* **Cannot, from this repository:** the step *upstream* of those
  series. `data/` is gitignored, so the seven Binance parquets are a
  local snapshot, not committed history (the "committed to the repo"
  line in `validation_multiexchange.md` § Reproducing is inaccurate
  on that point — see the addendum there). Rebuilding the basket
  from a fresh Binance fetch will give a longer sample and slightly
  different bar counts than the 2026-08-26 vintage pinned in the
  CSV headers, so the re-derived series will not match row-for-row.
* **Does it matter for the verdict:** no. The DSR verdict (≈ 0 under
  the conservative deflator) is not close to any threshold, and the
  one-bar vintage difference already observed against the 2026-05-29
  run moved Sharpe by 0.0003 and DSR by < 0.001.

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
* The two CSVs added under `docs/results/` are **derived artifacts**,
  not new experiments and not code: they pin series this finding
  already reported. Zero trials, zero behavior change.

## Revision 2026-08-26 (third Codex review round)

Documentation-only pass over this finding and the docs that cite it:

1. **Corrected a factual error in this file.** "1/sqrt(T) does not
   model how far the best of 500 dispersed trials drifts" was wrong —
   `deflated_sharpe_ratio` does feed `sharpe_std_dev` into
   `expected_max_sharpe`, so the N=500 extreme-value correction is
   applied under both conventions. The real limitation, now stated in
   § "The problem", is that 1/sqrt(T) assumes the cross-trial
   dispersion is pure sampling noise instead of the wider dispersion
   of the strategy pool actually searched. Same wording corrected in
   RESULTS.md and in the `walk_forward_v2.py` comment.
2. **Made "remain reproducible" true** by committing the two derived
   return series (see § Decision).
3. **Labeled the venue-verified figure as a fixed-config historical
   replay** everywhere it is cited (here, RESULTS.md, CLAUDE.md) so
   it stops reading as an independent OOS result under the
   "OOS unless marked" project convention.
4. **Separated the gate convention from the reporting convention**
   in RESULTS.md, which previously asserted both that the config
   passed DSR > 0.5 and that its primary DSR is ≈ 0.
5. **Scoped the labeling rule** so the historical per-strategy rows
   in RESULTS.md are covered by one blanket footnote rather than
   left as bare unscoped figures.

No number changed. No computation changed.

## Revision 2026-08-26 (fourth review round) — counting and labels

Documentation-only again; no computation, no re-run, no new trials.

1. **The cluster is 1 + 6, not 7 confirmations.** Every live doc that
   said "7 neighbor configs" / "7 of 7 pass" was counting the
   deployed `(28, 60)` as one of its own confirmations —
   `findings/cluster_stability.md` § 2 lists it inside the seven-member
   TSMOM short-ensemble cluster. Restated here, in RESULTS.md and in
   CLAUDE.md as **the deployed config plus six neighbours**: the
   cluster still has 7 members, the published median 0.736 is still
   the median over all 7, but the evidence it supplies for the
   deployed point is **6 of 6 neighbours**, not 7 of 7. Neighbour-only
   median under the same 1/sqrt(T) convention is 0.726 (0.671, 0.693,
   0.716, 0.736, 0.750, 0.782 — plain arithmetic on the finding's own
   per-variant table, nothing re-run). The raw-Sharpe band is
   unchanged at +1.37…+1.49 whether or not the deployed +1.48 is
   counted, which is the point: the plateau survives dropping the
   deployed point from it.
2. **"0.770 at the peak config" was wrong** and is now "0.770 for the
   deployed config". The cluster peak is `(30, 60)` at DSR 0.782 /
   Sharpe +1.49 (same table). Same correction applied to the
   RESULTS.md line that called `(28, 60)` the family's best
   individual.
3. **One unmissable scoping note per live doc instead of per-figure
   edits.** RESULTS.md gained a file-wide DSR-convention paragraph
   above the table (footnote ¹ stays as the per-number repeat), and
   README.md — which quotes DSR figures from `docs/results/` with no
   convention anywhere — gained the same note covering itself and the
   commit-pinned writeups it links. Those writeups and `findings/`
   are dated artifacts and are not rewritten.
4. **The deploy record's DSR units are corrected by addendum.**
   `findings/production_config_v1.md` reported the ratified figures as
   "DSR @ N=500, sd=0.7 ≈ 0.000" with "`E[max SR]` ≈ 2.137
   per-period". Raw 0.7 into `deflated_sharpe_ratio` is the
   dimensional artifact described in § "Both conventions" — it returns
   exactly 0.0 for any series, and 2.137 is the *annualized* bar
   (per-period 0.1118). Per findings immutability the historical lines
   stay; the dated addendum there states the conservative-convention
   figures (0.0016 replay / 0.037 concat-OOS) and points here.
