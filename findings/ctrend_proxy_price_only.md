# CTREND-proxy (price-only) — REJECT

**Date:** 2026-05-29
**Status:** REJECT
**Source paper:** Fieberg, Liedtke, Poddig, Walker, Zaremba, *"A Trend Factor for the Cross-Section of Cryptocurrency Returns,"* JFQA 60(7), 2024 (SSRN 4601972).
**Important asymmetry — read first:** This is a deliberately reduced proxy, not a faithful reproduction. A failure of THIS proxy does NOT refute the paper. See the **What this proxy is not** section before drawing conclusions about CTREND itself.

## TL;DR

CTREND-proxy (price-only, pooled Ridge on 2-yr rolling panel, top-quintile weekly) **fails** to beat both buy-and-hold BTC and the project's existing `run_cross_sectional_momentum` (CSM) on every subperiod (full / pre-ETF / post-ETF) and at both cost levels (Binance taker 0.10% + 5bps slip; Kraken taker 0.40% + 10bps slip). On Kraken, the proxy is net-negative (−3.3% CAGR full sample). Deflated Sharpe at the project's `PROJECT_NUM_TRIALS = 500` is ≈ 0 across the board. Verdict: REJECT.

## What this proxy is *not*

Two deliberate deviations from the published estimator:

1. **No volume features.** Fieberg's CTREND aggregates ~28 indicators including both reported and trusted spot volume. The Coin Metrics community-tier API exposes **only** `volume_reported_spot_usd_1d` (reported volume — contaminated by wash trading and methodology drift across venues). `volume_trusted_spot_usd_1d` is paid-tier only. Building the volume half on reported volume would test data-quality artifacts, not the strategy, so the volume half was omitted entirely.
2. **Different estimator.** Fieberg builds on Han, Zhou, Zhu (2016) "A Trend Factor" — a cross-sectional Fama-MacBeth-style regression with rolling per-date coefficients. This proxy is a pooled Ridge regression on a 2-year rolling panel. Different estimator, different statistical properties.

**Interpretation rule:** If THIS proxy fails (it does), the failure does **not** refute Fieberg. The volume half and the FMB estimator are untested here. A KEEP verdict on CTREND would require V2 with trusted volume + FMB — out of scope for the current data.

## Configuration (frozen *before* the first run)

| Parameter | Value |
|---|---|
| Features | `close / SMA(close, w)` for `w ∈ {5, 10, 20, 50, 100, 200}` (6 features) |
| Standardization | Per-train-window z-score |
| Estimator | `sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True)` |
| Target | 7-day forward return `close[t+7] / close[t] − 1` |
| Train window | Rolling 730 days (≈ 2 years) |
| Purge | 7 days between train end and rebalance day (no target overlap) |
| Rebalance | Weekly |
| Universe | PIT eligibility mask from `data/universe.build_pit_universe` (`top_n=20`, `volume_lookback_days=90`, `exclude_stablecoins=True`), restricted to the 24 coins with Coin Metrics parquet AND a non-NaN price column |
| Top-K | 4 (top-quintile of ~15–18 median eligible) |
| Weighting | Equal weight, long-only |
| Capital | $10,000 initial |
| Min history | Implicit via `min_periods=w` on rolling SMA — no coin enters the rank until it has 200 days of real history |

The walk-forward is implicit: Ridge is refit at every rebalance using the trailing 2-year panel. Per the user's review, the config was frozen before the first run and no post-hoc adjustments were made.

## Look-ahead checks

All four anti-leak invariants pinned by tests in `tests/test_ctrend_proxy.py` (10 tests, all passing):

1. Features at time `t` use only data up to `t` (test corrupts bars strictly after a chosen rebalance; equity up to that rebalance matches the clean run byte-for-byte).
2. Purge enforced: every train date in every fit is at least `purge_days` before the rebalance day (test intercepts `_collect_panel` and asserts the gap).
3. Eligibility honoured: a coin marked ineligible is never selected, even with positive drift.
4. Minimum history: a freshly-listed coin without `max(windows)` days of data has NaN features and is excluded.

## Results

### Full sample (2017–2026, ~9.4 years)

| Strategy | Sharpe | CAGR | Max DD | Cost setting |
|---|---:|---:|---:|---|
| **CTREND-proxy** | **+0.49** | **+9.3%** | **−94.5%** | Binance 0.10% + 5bps |
| **CTREND-proxy** | **+0.32** | **−3.3%** | **−96.3%** | Kraken 0.40% + 10bps |
| CSM (existing) | +0.81 | +38.0% | −82.8% | Binance |
| CSM (existing) | +0.70 | +27.8% | −86.9% | Kraken |
| BH-BTC | +1.01 | +58.2% | −83.8% | (cost-free baseline) |

### Pre-ETF subperiod (≤ 2024-01-10, ~7.0 years)

| Strategy | Sharpe | CAGR | Max DD | Cost |
|---|---:|---:|---:|---|
| **CTREND-proxy** | +0.50 | +9.5% | −94.5% | Binance |
| **CTREND-proxy** | +0.34 | −2.3% | −96.3% | Kraken |
| CSM | +0.77 | +33.7% | −82.8% | Binance |
| CSM | +0.67 | +24.9% | −84.8% | Kraken |
| BH-BTC | +1.11 | +73.0% | −83.8% | — |

### Post-ETF subperiod (≥ 2024-01-11, ~2.4 years)

| Strategy | Sharpe | CAGR | Max DD | Cost |
|---|---:|---:|---:|---|
| **CTREND-proxy** | +0.47 | +8.2% | −57.3% | Binance |
| **CTREND-proxy** | +0.27 | −6.6% | −61.0% | Kraken |
| CSM | +0.95 | +51.1% | −58.5% | Binance |
| CSM | +0.79 | +36.5% | −60.9% | Kraken |
| BH-BTC | +0.65 | +21.9% | −49.1% | — |

The proxy improves slightly post-ETF on Sharpe-relative-to-BH-BTC (BTC's Sharpe dropped after ETF too) but still trails CSM by a wide margin in both cost scenarios. Post-ETF DD halves vs pre-ETF — the universe is just less volatile, not the strategy being smarter.

### Cost drag

| Cost | CTREND-proxy fees + slippage over 9.4 years |
|---|---:|
| Binance 0.10% + 5bps | $10,267 (= 103% of initial capital) |
| Kraken 0.40% + 10bps | $23,852 (= 239% of initial capital) |

Both fee scenarios eat **more than the initial capital** in lifetime trading costs, on 357 rebalances (≈ weekly). The Kraken delta (+$13,585) is what flips the strategy from marginally-positive to net-negative. This was the user's predicted decisive cost: confirmed.

## Deflated Sharpe at `PROJECT_NUM_TRIALS = 500`

Using a conservative pooled `sd_trial_sharpes = 0.7` representing the cross-project dispersion of trial Sharpes:

- `E[max Sharpe over 500 trials] = 2.14`
- CTREND-proxy strategy Sharpes (0.27 to 0.50) fall **far** below this expected null maximum.
- DSR probability ≈ 0 across all subperiods and both cost scenarios.

DSR does not save this proxy. Even if `sd_trial_sharpes` were 0.3 (very tight pool), the Sharpe gap to `E[max]` would still be a multi-sigma rejection.

## Anti-overfit checklist

- [x] OOS truly held-out: Ridge is refit at every rebalance from trailing 730d only; no peeking via purge=7d. Walk-forward implicit at the rebalance grid.
- [x] DSR computed with `num_trials=500` per CLAUDE.md; honest about conservative `sd_trial_sharpes` estimate.
- [x] No look-ahead: tested via future-bar corruption (10 unit tests cover the invariants).
- [x] Survivorship: PIT eligibility mask + 200-day min history requirement on top.
- [x] Pre/post-ETF prepared and shown side by side.
- [x] Bench vs BH-BTC and existing CSM at both cost levels — proxy loses to both on every cut.
- [x] Edge sensitivity not tested at `top_k` / `ridge_alpha` neighbours: by design — config was frozen pre-run to avoid creating new trials.
- [x] Turnover and cost-drag reported in dollars.
- [ ] Robustness to single aberrant coin: not tested in this iteration. The −94.5% DD over the full sample suggests the strategy was concentrated in one or two altcoins during their 2018 collapse. Adding crash-aware vol-targeting would be a v2 hypothesis, but it's a different proxy.

## Failure modes observed

1. **Catastrophic 2018–2019 drawdown.** Top-K=4 equal-weight on a universe dominated by 2017-era ICO altcoins → fully invested in the worst performers of the 2018 bear. The early-period max DD of −94.5% effectively destroys the strategy: even if it generated 9.3% CAGR going forward, the path was unrecoverable in any real trading sense.
2. **Ridge α=1.0 is moderate.** Coefficient inspection showed non-degenerate predictions (not the "zeroed-out everything" failure mode the user warned about). The strategy IS picking, it's just picking poorly.
3. **Worse than naive CSM.** CSM ranks by past 30-day return; CTREND-proxy ranks by Ridge on 6 MA-ratios at 6 horizons. The added complexity of the ML estimator does not pay for itself relative to a single momentum window — on this universe, this period, this estimator.
4. **Post-ETF: less volatile but no edge recovery.** Whatever decoupling the literature reports post-ETF, this proxy did not capture it.

## Verdict

**REJECT** — CTREND-proxy (price-only) does not survive the verdict gate. Underperforms BH-BTC and existing CSM at both cost levels on every subperiod, with DSR ≈ 0 at `N=500`. Cost-drag on Kraken alone flips it net-negative.

**Important caveat on Fieberg:** This rejection is of the *proxy*, not the paper. The two reasons (no volume + pooled Ridge instead of FMB) plausibly explain the gap. A KEEP verdict on CTREND requires V2 — trusted-volume features (Coin Metrics paid tier or equivalent) + faithful Fama-MacBeth estimator — which is out of scope until data access changes. If the user later upgrades data access, this finding is the baseline V2 must beat.

## Reproducibility

- Module: `src/trade_lab/backtest/ctrend_proxy.py`
- Tests: `tests/test_ctrend_proxy.py` (10 invariants, all passing)
- Run convention: `run_ctrend_proxy(asset_candles, eligibility, ...)` on Coin Metrics `price` field as a close proxy; PIT mask from `build_pit_universe`. The full grid in this finding can be re-run by feeding the script's config above.
- Frozen config; no hyperparameter sweep was conducted. Adding one now would consume new trials in the project's `N=500` budget and invalidate this finding's DSR.
