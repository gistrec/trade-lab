# Validation Test 2 — Kraken cost-tax against the FROZEN config — per-venue verdict

**Date:** 2026-05-29
**Status:** PASS (Binance) / fee-fragile (Kraken). Per-venue verdict;
not a single PASS/FAIL.
**Frozen-config hash:** `ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753`
**Script:** `scripts/validation_test2_execution.py`
**Output:** `outputs/validation_test2_execution.json`

## TL;DR

* On the **venue-verified window** (2022-01-21 → 2026-05-28, 1589 bars
  — the Binance ↔ Bybit-confirmed window from Test 1), the strategy
  drops from **net Sharpe 0.721 on Binance** to **net Sharpe 0.56–0.59
  on Kraken** across the cost-sensitivity band — a marginal tax of
  **Δ −0.13 to −0.16 Sharpe**.
* The tax is roughly **regime-independent**: Pre-ETF Δ ≈ −0.13,
  Post-ETF Δ ≈ −0.13 to −0.16. Adding +30 bps of marginal taker fee
  on a 24-flip/year strategy is a structural drag that does not
  selectively affect bull vs bear regimes.
* **Kraken pre-ETF Sharpe is ~0.30-0.33** — below the project's own
  DSR > 0.5 confidence floor. A multi-cycle Kraken deployment would
  spend half its time in this regime.
* **Binance remains the deployable venue** at its existing testnet
  paper trading. **Kraken is fee-fragile** at this strategy's flip
  rate — not a refutation of the strategy, but a venue-deployability
  veto under the current fee schedule.

## Methodology — what is being measured

Per validation prompt: this is a **marginal Binance→Kraken cost-tax**
on the FROZEN config, isolated by:

* **Frozen Binance index.** The basket close-series is built once
  from Binance parquets at `PRODUCTION_CONFIG.fee_rate=0.0010` +
  `PRODUCTION_CONFIG.slippage_rate=0.0005`. The synthetic
  ``close`` series passed to the strategy is identical across cost
  scenarios.
* **Frozen signals.** ``TimeSeriesMomentumStrategy(lookbacks=(28,60),
  sma_filter_periods=(200,))`` produces identical signal series in
  every run — the cost swap is purely at the engine layer.
* **Cost params swapped on the engine only.** ``run_backtest(...,
  fee_rate=X, slippage_rate=Y)`` — same signals, different costs.
* **Composition frozen.** BNB-on-Kraken thin-liquidity (Test 1 RISK
  FLAG, corr 0.9661) is modeled as a **wider basket-average
  half-spread sensitivity band**, NOT as basket-shrinkage. Removing
  BNB from the basket would be a different untested strategy; the
  deploy-side question of asset exclusion belongs to Test 3
  (paper-trading harness), not to a cost-tax measurement.

Baseline numbers in this report (Binance SR 0.721 verified, SR 1.377
full) are already net of Binance fees per `PRODUCTION_CONFIG`. The
deltas reported are **marginal** tax, not gross→net.

## Cost scenarios

| Scenario | Taker fee | Half-spread | Rationale |
|---|---:|---:|---|
| `binance_baseline` | 0.10% | 5 bps | `PRODUCTION_CONFIG` defaults; reproduces the headline 0.72 verified SR. |
| `kraken_tight`     | 0.40% | 7 bps | Best case on Kraken: BTC/ETH/XRP at ~3 bps, SOL/ADA/DOGE at ~5–8 bps, BNB at ~15 bps. Basket-weighted ~7 bps. |
| `kraken_mid`       | 0.40% | 10 bps | Defensible mid: BTC/ETH ~3 bps, alts 5–10 bps, BNB ~25 bps (Test 1 thin-liquidity outlier). |
| `kraken_wide`      | 0.40% | 15 bps | Conservative: occasional spread widening + BNB persistently noisy. Captures the worst-case basket-weighted half-spread. |

Half-spreads are basket-weighted averages over the 7-major
equal-weight composition. The model is a flat per-transition
slippage_rate; per-asset breakouts are absorbed into the
basket-weighted average per validation-prompt framing.

## Headline results — per scenario, per window

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

## Marginal Binance→Kraken cost-tax (verified window — the GO/NO-GO cell)

| Scenario | Δ Sharpe | Δ cost-drag bps/y | Net Sharpe |
|---|---:|---:|---:|
| kraken_tight | −0.128 | +540 | +0.593 |
| kraken_mid   | −0.141 | +589 | +0.581 |
| kraken_wide  | −0.161 | +671 | +0.560 |

The Sharpe gap **between Kraken cost scenarios is small** (Δ 0.03
across tight→wide), and the gap **vs Binance is large** (Δ 0.13–0.16).
This is the structural signature of a **taker-fee-dominated tax**:
the 0.30% fee-delta (Binance 0.10% → Kraken 0.40%) is roughly 4–5×
the slippage-band width.

Back-of-envelope verification:
* Per-round-trip fee delta: (0.40% − 0.10%) × 2 = **60 bps/RT**
* Per-round-trip slippage delta (mid): (10 − 5) × 2 = **10 bps/RT**
* Total per-round-trip delta: ≈ 70 bps/RT
* Annual exposure flips: ~24/year ≈ ~12 round trips/year
* Expected annual marginal tax: 12 × 70 = **~840 bps/y**
* Measured marginal cost-drag (kraken_mid verified): **+589 bps/y**

The measured tax is smaller than the back-of-envelope because the
ladder produces partial transitions (0 → 0.5 → 1.0) rather than full
0 ↔ 1 round trips, and not every "flip" is a full unit of turnover.

## Regime decomposition — the Kraken tax is regime-independent

| Block | Binance SR | Kraken_mid SR | Δ |
|---|---:|---:|---:|
| Pre-ETF (bear-tail era) | +0.459 | +0.321 | **−0.138** |
| Post-ETF (bull era) | +0.902 | +0.758 | **−0.144** |
| Verified full | +0.721 | +0.581 | **−0.141** |

The Δ Sharpe is **structurally constant** at roughly −0.14 across
regimes — what you'd expect from a fee delta that hits every round
trip the same way. This is informative: the Kraken tax is not
absorbed by any "good regime"; it persists.

**Kraken Pre-ETF Sharpe ≈ 0.30–0.33** is the cell that matters most
for honest deployability: it sits below the project's own DSR > 0.5
confidence floor for "edge more likely real than not". A
multi-cycle Kraken deployment would spend roughly half its time in
this regime.

## Exposure-flip count — the tax DRIVER

* Annual exposure-flip count: **21.9/year on the full sample,
  24.3/year on the verified window.**
* A "flip" here means any change in the realized position
  ``{0, 0.5, 1.0}`` (ladder transitions + SMA-gate flips).
* Higher flip rates → higher cost sensitivity. The strategy's flip
  rate is moderate (~monthly cadence on average), which is why the
  4× taker fee jump translates to only ~0.14 Sharpe, not full
  destruction.

If the flip rate were 2× higher (say, 50/year from a faster signal),
the same fee delta would translate to roughly +1180 bps/y
marginal tax — and the verified Sharpe would land around 0.42, in
firm fail territory.

## Per-venue verdict

### Binance — **PASS** (paper-deployable on testnet at current fees)

* Verified net Sharpe **+0.721** on 4.4 years of independent (Test 1)
  venue-verified data.
* The full-sample +1.377 / DSR-0.770 number that justified entering
  validation is dominated by the pre-2022 era no public-tier venue
  can confirm — that number is **not** the deployment expectation.
* Regime split: bear 0.46 / bull 0.90. Honest deployable range is
  the venue-verified band, not the headline.
* Already in paper trading on Binance testnet per project phase.

### Kraken — **fee-fragile but positive**

* Verified net Sharpe **+0.56 to +0.59** across the cost-sensitivity
  band — survives positive, but:
  - Pre-ETF Sharpe **+0.30 to +0.33** — below DSR > 0.5 confidence
    floor.
  - Marginal cost-drag **+540 to +671 bps/y** vs Binance, persistent
    across regimes.
  - The strategy's 24-flip/year cadence is the driver; a faster
    signal would push Kraken into negative territory.
* **Not a refutation of the strategy.** It is venue-fragile, not
  signal-fragile. The Binance↔Bybit 100% signal agreement (Test 1)
  is preserved here — the difference is solely in the
  cost-multiplier.
* **Implication for Kraken deployment:** at the current 0.40% taker
  schedule and 24-flip/year cadence, a real-money Kraken deployment
  is not advisable. A Kraken maker-priced or lower-tier-fee
  deployment (taker ≤ 0.20%) would close most of the gap and
  warrants a separate evaluation if and when it materializes.

## What this test does NOT settle

* **Does not retest the basket composition.** BNB-on-Kraken is
  thin-liquidity (Test 1 RISK FLAG); modeled here as wider
  half-spread sensitivity, not as basket exclusion. Excluding BNB on
  a Kraken-only deployment would be a different (untested) strategy
  variant and is a Test 3 harness/deployment decision.
* **Does not address market-impact at sub-$10k notional.** Per
  validation prompt: MIN_NOTIONAL / LOT_SIZE / partial fills are
  confirmed-irrelevant at the project's $10k size budget; not
  modeled here.
* **Does not retest pre-2022 cost-tax.** The full-sample numbers are
  informative for sanity (the tax ratio compresses to Δ -0.08 on the
  bull-dominated 2018-2026 window), but the deployment-relevant
  decision is on the venue-verified post-2022 sample.
* **Does not re-run DSR.** DSR with `PROJECT_NUM_TRIALS = 500` is
  flagged for the final synthesis (`production_config_v1`) as a
  diagnostic on the **venue-verified** sample, not on the
  full-sample 0.770. The diagnostic count does not consume new
  trials. This test is confirmatory, not selective.

## Project N_TRIALS bookkeeping

Zero new trials added. This test is a confirmatory cost-sensitivity
sweep against the frozen pre-registered config. No parameter
optimization, no signal change, no composition change. Per
validation rules, confirmatory tests do not consume
`PROJECT_NUM_TRIALS`.

## Reproducing

```
.venv/bin/python scripts/validation_test2_execution.py
```

Reads ``data/binance_*_USDT_1d.parquet`` (committed via the
``.gitignore`` exemption is NOT in place — the data is local and
re-fetchable from Binance via ``fetch_ohlcv``). Writes
``outputs/validation_test2_execution.json`` with the full bundle.

Last reviewed: 2026-05-29.
