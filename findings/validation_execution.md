# Validation Test 2: Kraken cost tax on the frozen config, per-venue verdict

**Date:** 2026-05-29
**Status:** PASS (Binance) / fee-fragile (Kraken). The verdict is per
venue, not a single PASS/FAIL.
**Frozen-config hash:** `ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753`
**Script:** `scripts/validation_test2_execution.py`
**Output:** `outputs/validation_test2_execution.json`

## TL;DR

* **Binance stays the deployable venue** and continues in testnet
  paper trading.
* **Kraken is not advisable at its current 0.40% taker fee.** On the
  venue-verified window (2022-01-21 → 2026-05-28, 1589 bars, the
  Binance ↔ Bybit-confirmed window from Test 1) net Sharpe falls from
  **0.721 on Binance** to **0.56–0.59 on Kraken** across the
  cost-sensitivity band, a marginal tax of **Δ −0.13 to −0.16 Sharpe**.
* The tax barely depends on the regime: pre-ETF Δ ≈ −0.13, post-ETF
  Δ ≈ −0.13 to −0.16. An extra 30 bps of taker fee on a strategy with
  24 flips a year drags bull and bear periods alike.
* The verdict comes from the pre-ETF sub-period, where Kraken's raw
  Sharpe is ~0.30–0.33 over two years, well below the project's
  confidence floor (see the note on raw Sharpe vs DSR in Methodology).
  The full verified-window Kraken Sharpe is +0.58, above 0.5, so this
  is a risk-posture call that gives the adverse regime more weight for
  go/no-go, not a claim that Kraken's edge is zero.
* A maker-priced or ≤ 0.20% taker tier would close most of the gap
  and deserves a separate evaluation if one becomes available.

## Methodology: what is being measured

The test measures a **marginal Binance → Kraken cost tax** on the
frozen config, isolated as follows:

* **Frozen Binance index.** The basket close series is built once from
  Binance parquets at `PRODUCTION_CONFIG.fee_rate=0.0010` +
  `PRODUCTION_CONFIG.slippage_rate=0.0005`. The synthetic ``close``
  series passed to the strategy is identical in every cost scenario.
* **Frozen signals.** ``TimeSeriesMomentumStrategy(lookbacks=(28,60),
  sma_filter_periods=(200,))`` produces identical signal series in
  every run; costs change only at the engine layer.
* **Costs swapped on the engine only.** ``run_backtest(...,
  fee_rate=X, slippage_rate=Y)``: same signals, different costs.
* **Composition frozen.** BNB's thin Kraken liquidity (Test 1 RISK
  FLAG, corr 0.9661) is modeled as a wider basket-average half-spread
  band. Dropping BNB from the basket would be a different, untested
  strategy, and whether to exclude an asset at deploy time is a
  Test 3 (paper-trading harness) question.

Baseline numbers here (Binance SR 0.721 verified, SR 1.377 full) are
already net of Binance fees per `PRODUCTION_CONFIG`. The deltas are a
**marginal** tax, not gross → net.

**Raw Sharpe vs DSR.** Every Sharpe in this report is raw. The
project's DSR > 0.5 confidence floor is a DSR threshold, and a DSR on
a 2-year sub-sample with the `PROJECT_NUM_TRIALS = 500` penalty would
come out **lower** than the raw Sharpe. A raw pre-ETF Sharpe of
0.30–0.33 is therefore far below confidence of edge. DSR is deferred
to the final synthesis (see "What this test does NOT settle").

## Cost scenarios

| Scenario | Taker fee | Half-spread | Rationale |
|---|---:|---:|---|
| `binance_baseline` | 0.10% | 5 bps | `PRODUCTION_CONFIG` defaults; reproduces the headline 0.72 verified SR. |
| `kraken_tight`     | 0.40% | 7 bps | Best case on Kraken: BTC/ETH/XRP at ~3 bps, SOL/ADA/DOGE at ~5–8 bps, BNB at ~15 bps. Basket-weighted ~7 bps. |
| `kraken_mid`       | 0.40% | 10 bps | Defensible mid: BTC/ETH ~3 bps, alts 5–10 bps, BNB ~25 bps (Test 1 thin-liquidity outlier). |
| `kraken_wide`      | 0.40% | 15 bps | Conservative: occasional spread widening + BNB persistently noisy. Captures the worst-case basket-weighted half-spread. |

Half-spreads are basket-weighted averages over the equal-weight
7-asset basket. The model uses one flat per-transition
`slippage_rate`, so per-asset spreads are folded into that average.

## Headline results per scenario and window

| Scenario | Window | Bars | Total return | Net Sharpe | Cost-drag (bps/y) |
|---|---|---:|---:|---:|---:|
| **binance_baseline** | full (2018–2026) | 3070 | +13,221.6% | **+1.377** | 332 |
|                      | verified (2022–2026) | 1589 | +131.00% | **+0.721** | 261 |
|                      | pre-ETF (2022 → Jan 2024) | 720 | +20.39% | +0.459 | — |
|                      | post-ETF (Jan 2024 → 2026) | 869 | +91.88% | +0.902 | — |
| **kraken_tight**     | full | 3070 | +9,477.1% | +1.300 | 1020 |
|                      | verified | 1589 | +89.46% | +0.593 | 801 |
|                      | pre-ETF | 720 | +11.48% | +0.332 | — |
|                      | post-ETF | 869 | +69.96% | +0.771 | — |
| **kraken_mid**       | full | 3070 | +9,185.0% | +1.293 | 1083 |
|                      | verified | 1589 | +85.97% | **+0.581** | 850 |
|                      | pre-ETF | 720 | +10.68% | **+0.321** | — |
|                      | post-ETF | 869 | +68.03% | +0.758 | — |
| **kraken_wide**      | full | 3070 | +8,717.7% | +1.281 | 1188 |
|                      | verified | 1589 | +80.29% | +0.560 | 932 |
|                      | pre-ETF | 720 | +9.35% | +0.301 | — |
|                      | post-ETF | 869 | +64.87% | +0.738 | — |

## Marginal Binance → Kraken cost tax (verified window, the go/no-go cell)

| Scenario | Δ Sharpe | Δ cost-drag bps/y | Net Sharpe |
|---|---:|---:|---:|
| kraken_tight | −0.128 | +540 | +0.593 |
| kraken_mid   | −0.141 | +589 | +0.581 |
| kraken_wide  | −0.161 | +671 | +0.560 |

The Kraken scenarios differ from each other by only Δ 0.03 Sharpe
(tight → wide) but sit 0.13–0.16 below Binance. The taker fee
dominates the tax: the 0.30% fee delta (Binance 0.10% → Kraken 0.40%)
is roughly 4–5× the width of the slippage band.

Back-of-envelope check:
* Per-round-trip fee delta: (0.40% − 0.10%) × 2 = **60 bps/RT**
* Per-round-trip slippage delta (mid): (10 − 5) × 2 = **10 bps/RT**
* Total per-round-trip delta: ≈ 70 bps/RT
* Annual exposure flips: ~24/year ≈ ~12 round trips/year
* Expected annual marginal tax: 12 × 70 = **~840 bps/y**
* Measured marginal cost-drag (kraken_mid verified): **+589 bps/y**

The measured tax is below the estimate because the ladder moves in
partial steps (0 → 0.5 → 1.0), so not every flip is a full 0 ↔ 1
round trip.

## Regime decomposition

| Block | Binance SR | Kraken_mid SR | Δ |
|---|---:|---:|---:|
| Pre-ETF (bear-tail era) | +0.459 | +0.321 | **−0.138** |
| Post-ETF (bull era) | +0.902 | +0.758 | **−0.144** |
| Verified full | +0.721 | +0.581 | **−0.141** |

Δ Sharpe stays near −0.14 in every regime, as expected from a fee
delta that hits every round trip the same way. No favorable regime
absorbs the Kraken tax.

The cell that decides deployability is **Kraken pre-ETF Sharpe
≈ 0.30–0.33**, below the project's DSR > 0.5 floor for "edge more
likely real than not". A Kraken deployment running through several
market cycles would spend roughly half its time in this regime.

## Exposure flips drive the tax

* Annual exposure flips: **21.9/year on the full sample, 24.3/year on
  the verified window.**
* A flip is any change in the realized position ``{0, 0.5, 1.0}``:
  ladder transitions and SMA-gate flips.
* More flips mean more cost sensitivity. This strategy flips about
  monthly on average, which is why a 4× taker fee costs ~0.14 Sharpe
  instead of wiping out the edge.

At twice the flip rate (say 50/year from a faster signal), the same
fee delta would cost roughly +1180 bps/y and put the verified Sharpe
near 0.42, a clear fail.

## Per-venue verdict

### Binance: PASS (paper-deployable on testnet at current fees)

* Verified net Sharpe **+0.721** on 4.4 years of venue-verified data
  (independently confirmed in Test 1).
* The full-sample +1.377 / DSR 0.770 that justified entering
  validation is dominated by the pre-2022 era, which no public-tier
  venue can confirm. Expect the venue-verified band in deployment,
  not the headline.
* Regime split: bear 0.46 / bull 0.90.
* Already in paper trading on Binance testnet per the project phase.

### Kraken: fee-fragile (not advisable at the current 0.40% taker)

* Verified net Sharpe **+0.56 to +0.59** across the cost band, still
  above 0.5 raw, but:
  - pre-ETF raw Sharpe is **+0.30 to +0.33** over the 2-year
    sub-sample, and a DSR would be lower still (see Methodology);
  - the marginal cost drag is **+540 to +671 bps/y** against Binance
    in every regime;
  - the 24 flips a year drive the tax, and a faster signal would push
    Kraken negative.
* The signal itself holds up. Binance and Bybit agree on 100% of
  signals (Test 1), and that agreement carries over here; the only
  difference is the cost multiplier.
* The "not advisable" call weights the bear sub-period more than the
  bull for go/no-go on paper-trading deployment. Verified Kraken
  Sharpe of 0.58 is above 0.5, but a deployment running through
  several cycles would spend roughly half its time at pre-ETF-like
  Sharpe.
* **Conditional re-entry:** a maker-priced or ≤ 0.20% taker tier on
  Kraken would close most of the gap and deserves a separate
  evaluation when available.

## What this test does NOT settle

* **Basket composition.** BNB's thin Kraken liquidity (Test 1 RISK
  FLAG) is modeled as a wider half-spread. Excluding BNB on a
  Kraken-only deployment would be a different, untested variant and
  is a Test 3 harness/deployment decision.
* **Market impact at sub-$10k notional.** Per the validation plan,
  MIN_NOTIONAL / LOT_SIZE / partial fills are irrelevant at the
  project's $10k size budget and are not modeled.
* **Full-sample Kraken Sharpe** comes from the same run (each
  scenario is one full backtest, sliced into windows) and is reported
  for completeness, not as a go/no-go input. The verified window
  decides; the Kraken tax is regime-independent (Δ −0.13 to −0.14 in
  every block tested), so the full sample adds nothing to the verdict.
* **DSR.** DSR with `PROJECT_NUM_TRIALS = 500` is left to the final
  synthesis (`production_config_v1`) as a diagnostic on the
  venue-verified sample rather than the full-sample 0.770. The
  diagnostic consumes no trials.

## Project N_TRIALS bookkeeping

Zero new trials. This is a confirmatory cost-sensitivity sweep on the
frozen pre-registered config: no parameter optimization, no signal
change, no composition change. Under the validation rules,
confirmatory tests do not consume `PROJECT_NUM_TRIALS`.

## Reproducing

```
.venv/bin/python scripts/validation_test2_execution.py
```

Reads ``data/binance_*_USDT_1d.parquet``. The data is not committed;
it is local and can be refetched from Binance with ``fetch_ohlcv``.
Writes ``outputs/validation_test2_execution.json`` with the full
bundle.

Last reviewed: 2026-05-29.
