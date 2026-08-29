# `production_config_v1` — closes the research phase

**Date:** 2026-05-29
**Status:** Research phase closed. Forward paper trading is the next
gate. The operator turns the daily clock from here; this document is
the immutable record of what was validated, what was NOT, and how
to read the bands going forward.
**Frozen-config hash (the contract):**
`ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753`

## The frozen configuration

`PRODUCTION_CONFIG` (defined in `src/trade_lab/config/production_config.py`):

* **Basket:** equal-weight 7-major
  `[BTC, ETH, BNB, SOL, ADA, XRP, DOGE]`
* **Signal:** `TimeSeriesMomentumStrategy`
  * `lookbacks = (28, 60)` — both are TSMOM lookbacks; ladder
    `{0, 0.5, 1.0}` = mean of binary sign-of-return states.
  * `sma_filter_periods = (200,)` on the basket close (market-level
    regime gate, NOT per-asset).
  * `use_vol_target = False`.
* **Basket construction:** `build_crypto_market_index` with
  monthly rebalance (`freq="MS"`) + on N_active change.
* **Cost model (Binance-realistic):** `fee_rate = 0.0010`,
  `slippage_rate = 0.0005`.
* **Other knobs (inactive but in the hash):** `vol_lookback = 30`,
  `annual_vol_target = 0.25`, `max_position_size = 1.0`,
  `rebalance_threshold = 0.05`, `annualization_factor = 365`,
  `signal_shift_bars = 1`, `warmup_days = 200`.

Hash pin in `tests/test_production_config.py`. Bumping it is a new
research cycle by project rule — not a refactor.

## Validation results — what was tested

| Step | Test | Result |
|---|---|---|
| Step 0 | Freeze config + SHA-256 hash | DONE — `ac8919...` |
| Test 1 | Multi-exchange replay (Kraken + Bybit) | **PASS** — signal-level venue-robust |
| Test 1 Check A | Binance parquet date hygiene | **CLEAN** — no pre-listing rows |
| Test 1 Check B | Full Binance↔Bybit Sharpe on 1589 bars | Binance 0.721, Bybit 0.719 |
| Test 2 | Realistic execution (fees + spread) | **PER-VENUE:** Binance PASS, Kraken fee-fragile |
| Test 5 | Universe / availability bias | **PASS** — listing CLEAN + liquidity moot |
| Step 2 | Forward-test harness (frozen-hash gate + immutable vintage) | DONE |
| Step 3 | Reference behavioral fingerprint (frozen artifact) | DONE — hash `f8dd5b...` |
| Test 4 Part A | Retrospective truncation audit (signal-level) | **CLEAN** — 0 mismatches across 1589 bars × 4 metrics |
| Test 4 Part B | Live look-ahead detector | Infrastructure DONE; verdict accumulates with forward data |

Total project trials added by validation: **0** (all tests are
confirmatory diagnostics against the FROZEN pre-registered config).
`PROJECT_NUM_TRIALS = 500` unchanged.

## Deployable Sharpe expectation — be specific

The number to anchor on for forward expectations is **NOT** the
headline DSR 0.770 from `findings/han_28d_tsmom.md` and **NOT** the
full-sample Sharpe 1.377. Both are mathematically real on Binance
data, but neither survives independent verification at the public
REST tier (pre-2022 era is venue-unverifiable; Test 1 RISK FLAG).

The honest deployable band is the **venue-verified post-2022 sample**
(Binance ≈ Bybit to 0.005 Sharpe on 1589 bars):

| Sub-period | Bars | Years | Net SR | Total return |
|---|---:|---:|---:|---:|
| Pre-ETF (2022-01-21 → 2024-01-10) | 720 | ~2.0 | **+0.459** | +20.4% |
| Post-ETF (2024-01-11 → 2026-05-28) | 869 | ~2.4 | **+0.902** | +91.9% |
| **Verified full** | **1589** | **~4.4** | **+0.721** | **+131.0%** |

**Honest expectation band: ~0.46 (bear-leaning) … ~0.90 (bull-leaning),
centre ≈ 0.72.** Do not anchor forward on 1.38 (venue-unverifiable
era contributes most of it) or on 0.770 (different DSR convention,
walk-forward concatenated OOS folds, not the apples-to-apples
verified-window number).

## DSR — diagnostic on the venue-verified sample

Re-running the project's existing `deflated_sharpe_ratio` (in
`src/trade_lab/backtest/dsr.py`) at the pinned project settings:

| Sample | n bars | Annualized SR | DSR @ N=500, sd=0.7 |
|---|---:|---:|---:|
| Verified (2022-01-21 → 2026-05-28) | 1589 | +0.721 | **≈ 0.000** |
| Full Binance (2018-01-01 → 2026-05-28) | 3070 | +1.377 | ≈ 0.000 |

`E[max SR over 500 trials | sd=0.7]` ≈ 2.137 per-period — a bar that
neither the verified nor the full-sample annualized Sharpes clear
under the project's conservative multiple-testing penalty.

How to read this:

* The original "DSR 0.770" in `han_28d_tsmom.md` was computed on
  **walk-forward concatenated OOS returns**, on a different
  effective sample than the direct backtest used here. The two
  numbers measure different statistical objects and are not
  interchangeable.
* On the apples-to-apples diagnostic above, the verified-window
  Sharpe does **not** clear the project's pool-dispersion-corrected
  expected-max-Sharpe bar at N=500. This is the most conservative
  honest read of the strategy's statistical edge.
* This does NOT void the strategy. It frames the forward
  expectation: there is a measurable raw edge (+0.72 Sharpe over
  4.4 years), but under the project's own multiple-testing
  discipline at N=500 the deploy-confidence is modest, not high.
* This is exactly the result the validation prompt anticipated:
  "DSR на verified-окне ... сядет заметно ниже full-sample 0.770,
  возможно ниже 0.5".

No new trials were spent. The DSR is a diagnostic on a pre-
registered config; reproducible from
`src/trade_lab/backtest/dsr.py`.

## Validation failures and limits — explicit

These are NOT footnotes. They are first-class outputs of the
validation phase and must be carried forward to operations.

### 1. Kraken cost regime — `fee-fragile, not advisable @ 0.40% taker`

Test 2 marginal cost-tax decomposition (`findings/validation_execution.md`):

| Scenario | Verified Net SR | Δ vs Binance | Pre-ETF SR |
|---|---:|---:|---:|
| Binance baseline (0.10% + 5 bps) | +0.721 | — | +0.459 |
| Kraken mid (0.40% + 10 bps) | **+0.581** | **−0.141** | **+0.321** |
| Kraken wide (0.40% + 15 bps) | +0.560 | −0.161 | +0.301 |

The Kraken-vs-Binance tax is structurally constant at ≈ −0.14
Sharpe across regimes (taker-fee dominated, NOT regime-absorbed).
Kraken pre-ETF raw Sharpe at +0.30–0.33 is comfortably below the
project's confidence floor on a 2-year sub-sample. **Real-money
Kraken deployment under the current 0.40% taker is a venue-fragility
veto.** Conditional re-entry path: Kraken maker-priced or ≤ 0.20%
taker tier closes most of the marginal-tax gap and warrants a
separate evaluation if and when it materializes.

### 2. Pre-2022 era is venue-unverifiable

Bybit spot did not exist before mid-2021. Kraken's public REST is
hard-capped at the trailing 720 daily candles. The pre-2022 sample
that contributes the bulk of the headline Sharpe is therefore
verifiable ONLY against Binance itself. Test 1 Check B decomposition:

| Window | Years | Net SR |
|---|---:|---:|
| 2018-01-01 → 2022-01-20 (Bybit absent) | ~4.0 | **+1.857** |
| 2022-01-21 → 2026-05-28 (venue-verified) | ~4.4 | +0.721 |
| Full | ~8.4 | +1.377 |

Full / verified = 1.91× (the headline is dominated by the
unverifiable era). Early-block / verified = 2.58×. **Forward
anchoring on the full-sample Sharpe extrapolates from a sample that
no independent venue can confirm at the project's data-access tier.**

### 3. Current drawdown sits at the band edge — starting condition matters

Step 3 reference fingerprint
(`findings/validation_behavioral_fingerprint.md`) on the verified
window:

* `max_historical_dd` = **−32.17%** (2022 bear, FTX collapse era).
  This is the breach threshold for Monitor M4.
* Current drawdown at window end (2026-05-28) = **−26.86%**, sitting
  at the p25 percentile of historical drawdowns.
* Live headroom to breach = **5.31 pp**.

**The forward operator inherits this state as the starting condition.**
The strategy is not in neutral; it is at the lower edge of the
historical DD band, with limited headroom before the
Monitor's load-bearing flag fires. Within band → NOT an anomaly,
but NOT a soft drawdown either. A further ~5 pp slide takes the
strategy into the breach zone where Step-4 detector + operator
review become load-bearing.

### 4. `build_pit_universe` NaN-mcap (CTREND-class blocker)

CoinMetrics community-tier cache returns `market_cap = NaN` for BNB
on every bar. `build_pit_universe` ranks NaN to bottom
(`na_option="bottom"`), silently dropping BNB from any cross-
sectional universe derived from this code path. The deployable
frozen basket is hand-picked and bypasses this; it is unaffected.
**Any future cross-sectional rotation strategy (including the
CTREND-class candidate from the compass report) MUST NOT use
`build_pit_universe` until either paid-tier data closes the
mcap gap OR the function is patched to fail-loud on NaN inputs.**
Recorded as a fixed pre-condition for that research cycle, not
patched here. See `findings/validation_universe_bias.md`.

### 5. Verified window is regime-fragile in the recent slice

The 2024-12-25 → 2026-05-28 sub-window is net-negative on all three
venues independently (Test 1: Binance −13.33%, Bybit −13.58%,
Kraken −20.58%). This is **venue-AGREEMENT on a poor period**, not
strategy failure. The Step 3 fingerprint already absorbs this
sub-period into the reference distribution, so monitor M1-M3 will
not flag it as anomalous when it reappears live.

## Forward operating contract

* **Run the harness daily** via
  `python -m trade_lab.paper_trading.cli`. Idempotent; refuses to
  run if `CANONICAL_HASH` drifts.
* **Run the monitor** via
  `python -m trade_lab.paper_trading.fingerprint_cli` to compare
  live behavior against frozen reference bands. Descriptive only.
* **Run the live look-ahead detector** via
  `python -m trade_lab.paper_trading.lookahead_cli` whenever you
  want to cross-check live signals against backtest replay on the
  exact vintage bytes. The detector accumulates a verdict as
  forward rows arrive; it does NOT fabricate one on an empty
  journal.
* **Re-run Part A truncation audit**
  (`scripts/validation_lookahead_truncation_audit.py`) after any
  edit to basket / strategy / config modules. The bar to maintain
  is: zero mismatches on the verified window. The current verdict
  is CLEAN as of this commit.

## Change-management rule (the contract)

**Any change to `PRODUCTION_CONFIG` is a new research cycle.** That
includes:

* Asset list or order (basket composition).
* TSMOM lookbacks or SMA period.
* Vol-targeting toggle.
* Cost-model rates.
* Basket rebalance frequency.
* Any inactive-but-recorded knob (e.g. `rebalance_threshold`).

Procedure (mirrored in `tests/test_production_config.py`):

1. Open `findings/<descriptive_name>.md` documenting the new config
   as a new research cycle, counting against `PROJECT_NUM_TRIALS`.
2. Re-run walk-forward + DSR on the new config; record results.
3. Update CLAUDE.md "Deployable strategy" section if appropriate.
4. ONLY THEN update the pinned hash in
   `tests/test_production_config.py`.

The forward paper trade horizon does NOT count as a change-permitted
window; it is precisely the time during which the config must
remain stable to give the harness + monitor + detector a clean
forward signal.

## What "PASS" means here, explicitly

The project entered validation to falsify the candidate. We tried
to break it on:

* Data hygiene (Tests 1 Check A, 5) — survived.
* Venue robustness (Test 1 Bybit replay) — survived.
* Realistic execution (Test 2) — survived on Binance, fee-fragile
  on Kraken (recorded, not deployed).
* Universe bias (Test 5) — survived for the deployable basket;
  CTREND-class follow-up blocked behind a documented data fix.
* Temporal look-ahead in the signal pipeline (Test 4 Part A) —
  survived; 0 mismatches across 1589 bars × 4 metrics.

What we did NOT falsify, and cannot at the project's data-access
tier:

* Pre-2022 single-venue veracity — open by structural data limit.

What we EXPLICITLY chose not to act on:

* DSR ≈ 0 at N=500 sd=0.7 on the verified window — recorded as the
  honest confidence-of-edge number; does not invalidate the
  observed Sharpe but tempers its weight.

Forward paper trading on Binance testnet is the next honest gate.
Real-money deployment requires that gate to clear over multiple
months including at least one regime transition, plus the Test 4
Part B detector running clean against accumulated live rows.

## Reproducing the whole chain

```bash
# Step 0: hash pin
.venv/bin/python -m pytest tests/test_production_config.py

# Test 1: multi-exchange (network)
.venv/bin/python scripts/validation_test1_multiexchange.py

# Test 2: cost-tax (no network)
.venv/bin/python scripts/validation_test2_execution.py

# Test 5: universe-bias (no network — uses CoinMetrics cache)
# inline check; see findings/validation_universe_bias.md

# Step 2: harness (write a row; no network if you pass fetch_callable)
.venv/bin/python -m trade_lab.paper_trading.cli

# Step 3: reference fingerprint (no network)
.venv/bin/python scripts/build_reference_fingerprint.py

# Test 4 Part A: truncation audit (no network)
.venv/bin/python scripts/validation_lookahead_truncation_audit.py
```

All scripts read from `data/binance_*_USDT_1d.parquet`. Re-fetching
the parquets is via `trade_lab.data.fetch_ohlcv` against Binance
public REST.

Last reviewed: 2026-05-29.

## Addendum 2026-08-25 — universe bias did NOT survive as a whole

Two lines above must be read narrower than they are written. Both
stay as written per findings immutability; this addendum is the
correction.

* The validation table's **"Test 5 | Universe / availability bias |
  PASS — listing CLEAN + liquidity moot"**, and
* the "What PASS means here" bullet **"Universe bias (Test 5) —
  survived for the deployable basket"**.

Test 5 asked about two axes and closed both of them: **listing** (no
pre-listing rows in the panel) and **liquidity** (deploy notional
moot at $1.4k/asset). It did not ask about, and could not close, the
third: **composition** — *why these seven coins*. BTC, ETH, BNB, SOL,
ADA, XRP and DOGE were hand-picked in 2026 knowing ex post which
majors survived the sample. That is a live survivorship channel, and
it is untested pending the PIT-universe diagnostic run (deep review
2026-08-24). Size gauge from this project's own numbers: the
cross-sectional-momentum measurement dropped Sharpe 1.40 → 0.93 when
moved to a PIT universe.

So: read "Universe bias — survived" as **"the two tested axes of
universe bias survived; the composition axis is open."** The frozen
config, the hash, and every other test verdict in this document are
unaffected — they were all conditional on the chosen basket to begin
with. Details: the 2026-08-25 addendum in
`findings/validation_universe_bias.md`, and the matching addendum in
`findings/validation_lookahead_audit.md`.

## Addendum 2026-08-26 — the "DSR ≈ 0" bullet, in current terms

The "What we EXPLICITLY chose not to act on" bullet above — *"DSR ≈ 0
at N=500 sd=0.7 on the verified window"* — is still correct and is
now the project's **primary reported** DSR convention, not a footnote
(owner decision 2026-08-25, `findings/dsr_convention_2026_08.md`).
Two clarifications for anyone reading this document as the deploy
record:

* The deploy **gate** ("DSR > 0.5 at N=500, cluster-stable") is
  defined on the *minimal* 1/sqrt(T) convention and was not re-run;
  the config passed the gate as defined and would not pass a gate
  restated on the conservative deflator. RESULTS.md § "What PAPER
  status actually means" states this split explicitly.
* "The verified window" is a **fixed-config historical replay**
  (frozen config re-run over 2022-01-21 → 2026-05-27 to check venue
  agreement), not an independent out-of-sample result.

The derived return series behind both DSR figures are now committed
under `docs/results/dsr_convention_2026_08_*.csv`, so the numbers in
this document are reproducible without the gitignored parquets.

## Addendum 2026-08-26 (second) — units of the ratified DSR figures

The ratified DSR numbers in § "DSR — diagnostic on the venue-verified
sample" are quoted in the wrong unit. The historical lines stay as
written (findings are immutable); this addendum is the correction.
Two lines are affected:

* the table header **"DSR @ N=500, sd=0.7"** with both rows at
  **"≈ 0.000"**, and
* **"`E[max SR over 500 trials | sd=0.7]` ≈ 2.137 per-period"**.

`deflated_sharpe_ratio` works **per-period**: it compares the
non-annualized daily Sharpe against `sharpe_std_dev`. The project's
pinned conservative assumption `sd_trial_sharpes ≈ 0.7` is
**annualized** (factor 365, daily bars) and must be de-annualized
before it is passed: `0.7 / sqrt(365) ≈ 0.0366` per-period. So:

* **2.137 is the annualized bar, not a per-period one.** The
  per-period expected-max bar is
  `expected_max_sharpe(500, 0.7/sqrt(365)) = 0.1118`, which
  annualizes to 2.137 — the pinned constant. The document's own comparison
  ("neither the verified nor the full-sample *annualized* Sharpes
  clear 2.137") is therefore right in substance; only the label
  "per-period" is wrong.
* **The "≈ 0.000" cells are a dimensional artifact.** Passing raw
  0.7 puts the bar at 2.137 *per-period* (≈ 40.8 annualized), which
  no daily Sharpe can clear, so the call returns exactly `0.0` for
  any series whatsoever. That is not a measurement of this strategy.

Under the convention as it is actually defined
(`sharpe_std_dev = 0.7/sqrt(365)`, `num_trials = 500`), recomputed on
the return series committed with `findings/dsr_convention_2026_08.md`:

| Object | Bars | Annualized SR | DSR, conservative (`0.7/sqrt(365)`) | DSR, minimal (`1/sqrt(T)`) |
|---|---:|---:|---:|---:|
| Venue-verified replay window (2022-01-21 → 2026-05-27) | 1588 | +0.7217 | **0.0016** | 0.061 |
| Walk-forward concat-OOS (2020-01-01 → 2026-05-27) | 2339 | +1.4784 | **0.037** | 0.770 |

Notes on reading that table against the historical one above:

* The replay row is the same object as this document's "Verified"
  row, one bar shorter: the table above says 1589 bars / +0.721
  through 2026-05-28, the committed artifact is 1588 bars / +0.7217
  through 2026-05-27 because the local BTC parquet vintage ends a bar
  earlier. Sharpe moves by < 0.001 at the precision the two
  sources quote (+0.721 vs +0.7217), DSR by < 0.001.
* The **full-sample row (3070 bars, +1.377) has not been recomputed**
  — no derived series for that window is committed. Its literal
  "≈ 0.000" came from the same raw-0.7 call, so treat it as
  unquantified rather than measured. Only the *direction* follows from
  what is known: +1.377 annualized sits below the 2.137 annualized
  expected-max bar, so `SR_hat - SR_0 < 0` and its DSR is **< 0.5**.
  No magnitude is claimed. `deflated_sharpe_ratio` scales that gap and
  then divides it by a term in the series' own skew and kurtosis before
  the normal CDF; that series is exactly what is unavailable here, so
  *how far* below 0.5 the value lands does not follow from the two
  numbers in the table. "< 0.5", not "≈ 0".
* **"0.770 (different DSR convention)"** in § "Deployable Sharpe
  expectation" is, precisely: the **minimal 1/sqrt(T)** convention on
  the walk-forward concat-OOS series (T = 2339). Same code, same
  N=500 extreme-value correction — only the assumed trial-pool
  dispersion differs.
* The verdict of this document is unchanged. Under the conservative
  deflator — now the project's primary *reported* convention — the
  deployed config sits at DSR ≈ 0 (0.0016 on the replay window,
  0.037 concat-OOS), which is exactly what "What we EXPLICITLY chose
  not to act on" recorded. Only the units and the labels are
  corrected here; no number moved, no computation changed, no trials
  added.

Convention decision, both committed series, and the reproduction
snippet: `findings/dsr_convention_2026_08.md`.
