# HMM 2-state regime overlay — REJECT (duplicate of existing vol-target, worse on most cuts)

**Date:** 2026-05-29
**Status:** REJECT
**Sources:** Caporale & Zekokh (Markov-switching GARCH for crypto vol); compass-artifact #3.
**Decision rule pinned by user review:** "Бьёт ли net-of-cost мой существующий vol-targeting overlay. Если нет — REJECT как дубликат." This is the operative criterion for this candidate.

## TL;DR

2-state Gaussian HMM on BTC daily log-returns, refit weekly on a 730-day trailing window, long when the *filtered* (not smoothed) posterior P(bull) > 0.5 else cash. Full-sample and pre-ETF runs **lose to BH-BTC and to a naive VolTarget(0.30) baseline** on all risk-adjusted metrics (Sharpe, CAGR, Calmar). Post-ETF Binance shows a flicker where HMM-overlay beats both BH-BTC and VolTarget on Sharpe (0.77 vs 0.65 vs 0.49) — but the win is 2.4 years of data, vanishes under Kraken costs (HMM ties VolTarget at 0.46), and DSR ≈ 0 at `N=500`. Per the user's pinned decision rule (duplicate test against existing vol-target), the strategy fails on the dominant time windows. REJECT.

## What this proxy is *not*

One intentional implementation choice critical to honesty:

* **Filtered, not smoothed, posteriors.** `hmmlearn.GaussianHMM.predict_proba(X)` returns the *smoothed* P(state[t] | data 1..T) — uses every observation including the future. That is the classical HMM-backtest look-ahead the user's review flagged as make-or-break. This module bypasses `predict_proba` and computes the filtered posterior P(state[t] | data 1..t) by exponentiating the last row of `hmmlearn._hmmc.forward_log(...)` — forward pass only, no backward. Unit-tested directly: corrupting closes after a chosen rebalance does not alter the equity at or before that rebalance.

Everything else mirrors the compass pseudocode:

* 2 components, Gaussian emissions, fit by EM (Baum-Welch).
* State identification by mean: `bull` = component with the higher fitted `means_`.
* Walk-forward refit every rebalance: no parameter leak across folds.
* 1-day publication buffer on the return input — `r[t]` is computed at the close of t but treated as available at t+1 for cleanness.

## Configuration (frozen *before* the first run)

| Parameter | Value |
|---|---|
| Signal | 2-state Gaussian HMM on daily log-returns |
| Filtering | Forward pass only (`_hmmc.forward_log`); NEVER `predict_proba` |
| Train window | Rolling 730 days (≈ 2 years) |
| Buffer | 1-day publication lag |
| Rebalance | Weekly |
| Probability threshold | 0.5 (binary in/out) |
| EM iterations | 30 |
| Random seed | 42 (deterministic replay) |
| Universe | BTC only |

No threshold tuning. No state-count tuning. No mean/vol prior tuning. Single config.

## Look-ahead invariants (10 unit tests, all passing)

* Filtered probability uses forward-only pass — verified by recomputing on a prefix of the input series.
* Future-bar corruption test: closes after a rebalance date corrupted with garbage; equity up to that rebalance is byte-for-byte identical to the clean run.
* Train window strictly trailing with `buffer_days=1` lag.
* Position clamped to {0, 1}.
* Cost charged proportional to `|Δposition|` only.

## Results

### Full sample (9.4 years)

| Strategy | Sharpe | CAGR | Max DD | Calmar | Cost |
|---|---:|---:|---:|---:|---|
| **HMM-overlay** | **+0.67** | **+20.9%** | **−60.0%** | **+0.35** | Binance |
| **HMM-overlay** | **+0.47** | **+11.7%** | **−63.6%** | **+0.18** | Kraken |
| VolTarget(0.30) | +0.98 | +32.0% | −54.8% | +0.58 | Binance |
| VolTarget(0.30) | +0.94 | +30.5% | −55.2% | +0.55 | Kraken |
| BH-BTC | +1.01 | +58.2% | −83.8% | +0.69 | (cost-free baseline) |

### Pre-ETF subperiod (7.0 years)

| Strategy | Sharpe | CAGR | Max DD | Calmar | Cost |
|---|---:|---:|---:|---:|---|
| HMM-overlay | +0.66 | +21.0% | −60.0% | +0.35 | Binance |
| HMM-overlay | +0.48 | +12.2% | −63.6% | +0.19 | Kraken |
| VolTarget(0.30) | +1.13 | +40.0% | −54.8% | +0.73 | Binance |
| VolTarget(0.30) | +1.10 | +38.3% | −55.2% | +0.69 | Kraken |
| BH-BTC | +1.11 | +73.0% | −83.8% | +0.87 | — |

### Post-ETF subperiod (2.4 years) — the only interesting window

| Strategy | Sharpe | CAGR | Max DD | Calmar | Cost |
|---|---:|---:|---:|---:|---|
| **HMM-overlay** ★ | **+0.77** | **+20.6%** | **−24.3%** | **+0.85** | Binance |
| HMM-overlay | +0.46 | +10.0% | −28.0% | +0.36 | Kraken |
| VolTarget(0.30) | +0.49 | +11.4% | −40.7% | +0.28 | Binance |
| VolTarget(0.30) | +0.46 | +10.0% | −41.0% | +0.24 | Kraken |
| BH-BTC | +0.65 | +21.9% | −49.1% | +0.45 | — |

★ The **only** cell where HMM-overlay wins by a meaningful margin. Post-ETF Binance: HMM Sharpe 0.77 beats BH-BTC 0.65 by 0.12 and beats VolTarget 0.49 by 0.28. But:
* 2.4 years of data is too short for DSR-grade confidence (post-ETF DSR p ≈ 0 even at this Sharpe).
* The win vanishes under Kraken costs: HMM ties VolTarget at 0.46.
* The Sharpe gain disappears once `fee_rate` clears the Binance-Kraken delta (~30 bps round trip).

### Cost-drag

| Cost setting | Fees + slippage over 9.4 years | % of initial capital |
|---|---:|---:|
| Binance 0.10% + 5bps | $11,042 | 110.4% |
| Kraken 0.40% + 10bps | $23,597 | 235.9% |
| (213 rebalances; mean realized position 0.41) | | |

The HMM overlay's cost-drag is roughly comparable to CTREND-proxy (similar turnover) and roughly 5× the MVRV overlay's. Kraken cost-drag is what flips the post-ETF Binance flicker into a tie. Even if the post-ETF Binance Sharpe of 0.77 were real, the Kraken-equivalent 0.46 would not be a deploy candidate.

## Deflated Sharpe at `PROJECT_NUM_TRIALS = 500`

`E[max Sharpe over 500 trials, sd_trial_sharpes=0.7] = 2.14`

All six (cost × subperiod) HMM-overlay configurations land at Sharpe 0.46 to 0.77 — far below the expected null max. DSR p ≈ 0 across the board, including the post-ETF Binance flicker. Statistical multiple-testing correction does not save this strategy.

## Comparison to existing `VolatilityTargetWrapper` (the dispositive test)

The user's review pinned: *"бьёт ли net-of-cost мой существующий vol-targeting overlay. Если нет — REJECT как дубликат."* Recall the project's `VolatilityTargetWrapper` exposes the same kind of risk: dial down in high-vol regimes, dial up in low-vol — which is structurally what the HMM bull-state detection is meant to do.

| Cut | HMM Sharpe | VolTarget Sharpe | Delta | Winner |
|---|---:|---:|---:|---|
| Binance / full | +0.67 | +0.98 | −0.31 | VolTarget |
| Binance / pre-ETF | +0.66 | +1.13 | −0.47 | VolTarget |
| Binance / post-ETF | +0.77 | +0.49 | +0.28 | **HMM (only cell)** |
| Kraken / full | +0.47 | +0.94 | −0.47 | VolTarget |
| Kraken / pre-ETF | +0.48 | +1.10 | −0.62 | VolTarget |
| Kraken / post-ETF | +0.46 | +0.46 | 0.00 | Tie |

5 of 6 cuts: HMM-overlay loses or ties. The one HMM win is Binance-only, post-ETF-only, 2.4 years of data. Per the operative decision rule: REJECT.

## Anti-overfit checklist

- [x] OOS truly held-out (no parameter tuning; weekly walk-forward refit; train strictly trailing).
- [x] DSR at `num_trials=500` with conservative `sd_trial=0.7`.
- [x] **Filtered, not smoothed** posteriors (forward pass only, 2 unit tests pin the invariant).
- [x] Buffer-day lag on the return input.
- [x] Pre/post-ETF subperiods isolated.
- [x] Benchmark vs BH-BTC AND vs existing VolTarget overlay at both cost levels.
- [x] No threshold / α / state-count sensitivity sweep — frozen config.
- [x] Turnover and cost-drag reported.
- [ ] HMM convergence: `hmmlearn` emits `"Model is not converging"` warnings on most fits — the EM deltas are very small (~0.001) but the warnings indicate the runs are operationally fragile. Increasing `n_iter` or `tol` is a parameter change and would be a new trial.

## Failure modes observed

1. **Misses the bull phases.** Mean realized position 0.41 → the overlay sits in cash 60% of the time on average. Even when correct about state, the binary 0/1 threshold throws away most of the BH-BTC return path.
2. **Convergence fragility.** Repeated `"Model is not converging"` warnings from `hmmlearn` indicate the EM optimization is finding very-shallow local optima with tiny deltas. The model would change qualitatively under different `random_state` or `n_iter` — a real ops liability.
3. **Pre-ETF: dominated by VolTarget.** Sharpe gap of 0.47 (Binance) and 0.62 (Kraken). The HMM bull-state signal is not adding anything VolTarget doesn't already get from realized-vol scaling.
4. **Post-ETF flicker is cost-regime-dependent.** The 0.77 Binance Sharpe drops to 0.46 on Kraken. A real edge would survive a 30-bp round-trip cost increment; this doesn't.
5. **Lags at regime turns.** Typical Markov-switching: regime estimates lag actual transitions because the EM is averaging over recent history. The 1-day buffer is honest but adds further lag.

## Verdict

**REJECT** — per the user's pinned decision rule, the strategy fails to beat the existing `VolatilityTargetWrapper` overlay on 5 of 6 (cost × subperiod) cuts. The one cell where it wins (Binance post-ETF, 2.4y) is below DSR threshold at `N=500` and vanishes on Kraken. Combined with the convergence fragility, this strategy adds nothing operationally usable over the existing vol-target stack.

## Caveats

* **REJECT of this overlay is NOT a refutation of HMM regime detection as a tool.** The compass artifact's own framing is that HMM is "база сильна для прогноза волатильности, слаба для прогноза доходности." The result here is consistent with that — the overlay struggles precisely on the return-prediction half.
* **Post-ETF Binance flicker is worth watching** if the user accumulates more post-ETF cycles. With another full cycle of data (likely ~2027), the 2.4-year flicker becomes a 4-5-year window and DSR power improves. This is not a deploy decision; it's a "re-test in 2027" note.
* **Convergence warnings** could be silenced by tightening `tol` or raising `n_iter`, but both would be new trials in `N=500`. The frozen-config discipline prevents that here.

## Reproducibility

- Module: `src/trade_lab/backtest/hmm_regime_overlay.py`
- Tests: `tests/test_hmm_regime_overlay.py` (10 invariants, all passing)
- Dependency: `hmmlearn>=0.3,<0.4` in the `[research]` extras (pinned for EM solver reproducibility).
- Frozen config; no sensitivity sweep was conducted.
