# MVRV-overlay (BTC weekly tilt) — INCONCLUSIVE (empirically tilting REJECT)

**Date:** 2026-05-29
**Status:** INCONCLUSIVE
**Sources:** Palazzi, Raimundo Júnior, Klotzle (SSRN 6199098, 2026) — Granger / transfer entropy, not P&L; Liu & Zhang (arXiv:2201.12893, 2023) — MVRV-like ratios don't predict short-horizon returns, only long-horizon; compass-artifact for the high/low threshold mapping.
**Important asymmetry — read first:** This is a deliberately reduced proxy. Three deviations from the canonical pseudocode are documented below. INCONCLUSIVE here means "empirical evidence does not support KEEP, and statistical power on a 2–3 cycle sample is too low for a hard REJECT either."

## TL;DR

Across all subperiods (full / pre-ETF / post-ETF) and both cost scenarios (Binance 0.10% + 5bps; Kraken 0.40% + 10bps), the MVRV-ratio overlay **underperforms BH-BTC** on every risk-adjusted metric tested (Sharpe, Calmar, CAGR). It does reduce Max Drawdown substantially (−64% vs BTC's −84% pre-ETF; −37% vs −49% post-ETF) but not by enough to flip Sharpe or Calmar. Cost-drag is small (low turnover). DSR ≈ 0 at `N=500`. The compass artifact explicitly predicted INCONCLUSIVE due to only ~2–3 BTC market cycles in the sample — this run confirms there is no statistical room to demonstrate edge with the data available.

## What this proxy is *not*

Three deliberate deviations:

1. **MVRV ratio, not MVRV-Z score.** The canonical pseudocode uses Z-score with thresholds Z>6 / Z<0. Computing Z requires the realized cap (`CapRealUSD`) AND its rolling stddev — both paid-tier on Coin Metrics. Community tier exposes only `CapMVRVCur` (the raw ratio `MarketCap / RealizedCap`). We use the ratio with thresholds (`low=1.0, high=3.5`) approximately mapped from the Z>6 / Z<0 regions on historical BTC. The mapping is not exact; the Z is more precise; the ratio is what's free.
2. **Linear interpolation between thresholds**, not the binary "cash above / full long below" the compass pseudocode. Linear absorbs small threshold-tuning sensitivity (Q8-class concern from the user's review) — sensitivity table below confirms results are robust to ±20% threshold shifts.
3. **One-day publication lag** on the MVRV input. Coin Metrics community data is published t+1; the signal at day t uses MVRV from day t−1 to honour real-world latency. The look-ahead test in `tests/test_mvrv_overlay.py` confirms the lag is respected.

**Interpretation rule:** The INCONCLUSIVE verdict applies to *this* proxy. A canonical Z-score implementation might yield different thresholds and different results, but the underlying ~2–3 cycle data scarcity remains the binding constraint.

## Configuration (frozen *before* the first run)

| Parameter | Value | Source |
|---|---|---|
| Signal | `CapMVRVCur` (Coin Metrics community) for BTC | free, fetched 2026-05-29 |
| Publication lag | 1 day | community API publishes t+1 |
| Low threshold | 1.0 | maps to Z≈0 region per compass artifact |
| High threshold | 3.5 | maps to Z>6 region per compass artifact |
| Interpolation | Linear between thresholds, clamped to [0, 1] | smoother than binary |
| Rebalance cadence | Weekly | low-turnover overlay by design |
| Universe | BTC only (single-asset overlay) | per Palazzi / Liu-Zhang |
| Initial capital | $10,000 | project convention |

No threshold tuning. Sensitivity table below is diagnostic only.

## Look-ahead checks

11 unit tests in `tests/test_mvrv_overlay.py`:

* Target = 1.0 at/below low_threshold; 0.0 at/above high_threshold; linear midpoint.
* Always-low MVRV → realized position 1.0 → equity tracks BTC bar by bar.
* Always-high MVRV → realized position 0.0 → equity flat at initial.
* Publication lag respected — changing today's MVRV does not change today's position.
* Future-MVRV corruption test: corrupt MVRV strictly after a rebalance day; equity up to that rebalance is byte-for-byte identical to the clean run.
* Cost charged exactly once per `|Δposition|` event.

## Data summary

- BTC: 3434 daily bars from Coin Metrics, 2017-01-01 to 2026-05-27.
- MVRV ratio: 3434 matching bars.
- Observed MVRV range: 0.69 (deep bear bottom) to 4.72 (2017 cycle peak). Median 1.78, mean 1.85.

## Results

### Full sample (9.4 years)

| Strategy | Sharpe | CAGR | Max DD | Calmar | Cost |
|---|---:|---:|---:|---:|---|
| **MVRV-overlay** | **+0.85** | **+32.8%** | **−64.0%** | **+0.51** | Binance |
| **MVRV-overlay** | **+0.84** | **+31.6%** | **−64.3%** | **+0.49** | Kraken |
| BH-BTC | +1.01 | +58.2% | −83.8% | +0.69 | (cost-free baseline) |

### Pre-ETF subperiod (≤ 2024-01-10, 7.0 years)

| Strategy | Sharpe | CAGR | Max DD | Calmar | Cost |
|---|---:|---:|---:|---:|---|
| MVRV-overlay | +0.93 | +39.7% | −64.0% | +0.62 | Binance |
| MVRV-overlay | +0.91 | +38.4% | −64.3% | +0.60 | Kraken |
| BH-BTC | +1.11 | +73.0% | −83.8% | +0.87 | — |

### Post-ETF subperiod (≥ 2024-01-11, 2.4 years)

| Strategy | Sharpe | CAGR | Max DD | Calmar | Cost |
|---|---:|---:|---:|---:|---|
| MVRV-overlay | +0.60 | +14.4% | −36.5% | +0.39 | Binance |
| MVRV-overlay | +0.58 | +13.7% | −36.7% | +0.37 | Kraken |
| BH-BTC | +0.65 | +21.9% | −49.1% | +0.45 | — |

### Cost-drag

| Cost setting | Total fees+slippage over 9.4 years | % of initial capital |
|---|---:|---:|
| Binance 0.10% + 5bps | $2,256 | 22.6% |
| Kraken 0.40% + 10bps | $7,107 | 71.1% |
| (450 rebalances; mean realized position 0.66) | | |

Compared to CTREND-proxy (357 reb, $10k-$24k costs), the MVRV overlay is genuinely low turnover. Kraken cost-drag is meaningful (~70% of capital) but does NOT flip the strategy's sign — Sharpe gap to Binance is only 0.01.

## Deflated Sharpe at `PROJECT_NUM_TRIALS = 500`

Conservative pooled `sd_trial_sharpes = 0.7`:
- `E[max Sharpe over 500 trials] = 2.14`
- MVRV-overlay strategy Sharpes (0.58 to 0.93) fall well below the expected null max.
- DSR p ≈ 0 across all subperiods and both cost scenarios.

DSR is structurally unable to anoint this strategy at `N=500` — even if the strategy genuinely had a Sharpe of 1.5 (it doesn't), the 2–3 cycle effective independent sample is too small for the DSR to find statistical significance after multiple-testing correction.

## Threshold sensitivity (diagnostic, NOT used for verdict)

A 3×3 grid of (low, high) thresholds, Binance and Kraken Sharpe, full sample:

| (low, high) | Binance Sharpe | Kraken Sharpe | CAGR (Binance) |
|---|---:|---:|---:|
| (0.8, 3.0) | +0.77 | +0.75 | +24.4% |
| (0.8, 3.5) | +0.86 | +0.84 | +31.1% |
| (0.8, 4.0) | +0.93 | +0.91 | +37.1% |
| (1.0, 3.0) | +0.77 | +0.75 | +26.1% |
| **(1.0, 3.5)** ★ | **+0.85** | **+0.84** | **+32.8%** |
| (1.0, 4.0) | +0.93 | +0.91 | +38.9% |
| (1.2, 3.0) | +0.76 | +0.74 | +27.0% |
| (1.2, 3.5) | +0.85 | +0.83 | +34.2% |
| (1.2, 4.0) | +0.92 | +0.91 | +40.4% |

★ = main config used for verdict.

Sharpe range across the 9-cell sensitivity: **0.74 to 0.93** (Kraken). None of these beat BH-BTC's full-sample Sharpe of 1.01. The result is stable across reasonable threshold choices — the failure is NOT due to a bad threshold pick. The "best" sensitivity cell (1.0, 4.0) at Sharpe 0.93 still loses by 0.08 Sharpe to BH-BTC.

Doing post-hoc threshold optimisation to chase that 0.93 cell would be a new trial in `N=500` and would not change the conclusion (still loses to BH-BTC).

## Anti-overfit checklist

- [x] OOS truly held-out: strategy uses MVRV with 1-day publication lag, no parameters fit on data.
- [x] DSR with `num_trials=500`; honest about conservative `sd_trial_sharpes` estimate.
- [x] No look-ahead: 11 unit tests pin invariants (future-MVRV corruption test included).
- [x] On-chain data point-in-time with realistic publication lag.
- [x] Pre/post-ETF subperiods shown side by side; institutional-era amplitude shrinkage visible in post-ETF DD halving.
- [x] Bench vs BH-BTC at both cost levels — overlay loses on Sharpe and Calmar on every cut.
- [x] Sensitivity to thresholds shown across 9 cells; result is stable, NOT cherry-picked.
- [x] Turnover and cost-drag reported.
- [x] Result does not depend on one aberrant cycle — pre-ETF and post-ETF agree directionally (overlay underperforms BH-BTC by 0.05-0.18 Sharpe consistently).

## Failure modes observed

1. **Misses the boom phase.** When MVRV ≥ 3.5, the overlay sits in cash. This deactivates during the 2017 H2 / 2021 H1 / 2024-2025 attempted rally — exactly the periods when BH-BTC made most of its alpha. Reducing DD by skipping the top has a symmetric cost: skipping the climb to the top.
2. **Linear interpolation is conservative through normal regime.** Mean realized position = 0.66 — the overlay spends most of its time partly invested even when neither threshold is active. This causes a structural ~33% under-investment vs BH-BTC.
3. **Threshold thresholds were derived from historical Z>6 / Z<0 boundaries on 2013/2017/2021 cycles.** Post-ETF cycle structure (Palazzi et al. 2026; "Institutionalization of Bitcoin" 2025–2026) shows shrinking amplitude — the 2024-2025 attempted top barely reaches MVRV 2.7, far below the legacy 3.5 trigger. In the post-ETF regime the high-threshold gate doesn't fire — the strategy is effectively just BH-BTC at the mean-position level.
4. **DD reduction is real but disproportionate to the return cost.** Pre-ETF DD: −64% vs −84% (20pp reduction). Pre-ETF CAGR: 39.7% vs 73.0% (33pp reduction). The trade is unfavourable for a long-only DD-tolerant investor.

## Verdict

**INCONCLUSIVE**, empirically tilting REJECT. The strategy underperforms BH-BTC on every risk-adjusted metric tested across every cut. DSR ≈ 0 at `N=500`. The data scarcity (~2–3 cycles) prevents a confident REJECT, but it equally prevents any KEEP claim. The compass-artifact prior of "INCONCLUSIVE due to few cycles" is confirmed.

For an operator decision: **do not use this overlay**. It does not add risk-adjusted value vs BH-BTC at any threshold setting tested, the structural under-investment is a permanent cost, and post-ETF amplitude shrinkage may neuter the gating mechanism entirely in the next cycle.

## Important caveats

* **REJECT/INCONCLUSIVE of this proxy does not refute the underlying MVRV literature.** Palazzi et al. (2026) report Granger / transfer entropy *predictability*, not P&L. Liu & Zhang (2023) explicitly state MVRV-like ratios "have little predictive power for short-term bitcoin returns" — only long-horizon. A faithful test would need a multi-cycle independent dataset, which BTC does not provide.
* **A Z-score implementation might find a different threshold structure**, but the binding constraint is the 2-3 cycle sample size. No threshold choice can manufacture cycle history.
* **The TLDR caution about NEXT cycle**: post-ETF the high-threshold trigger barely activates. If the next cycle's peak again falls short of the legacy MVRV 3.5 boundary, the overlay degenerates to BH-BTC minus turnover costs.

## Reproducibility

- Module: `src/trade_lab/backtest/mvrv_overlay.py`
- Tests: `tests/test_mvrv_overlay.py` (11 invariants, all passing)
- Data: `data/coinmetrics/btc_mvrv.parquet` (fetched 2026-05-29 from Coin Metrics community API).
- Frozen config; sensitivity grid is documented; no post-hoc threshold optimisation occurred.
