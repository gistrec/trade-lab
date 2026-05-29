# Strategy results — master index

Single navigable summary of every strategy, overlay, and cross-section
construction the project has tested. Each row points at the full
finding for details. Project-wide convention: every Sharpe / DSR
quoted is **net of cost** and **OOS** unless explicitly marked
otherwise. `PROJECT_NUM_TRIALS = 500` (pinned, see CLAUDE.md).

## What "PAPER" status actually means

A strategy that reaches PAPER status has cleared **one** gate, not all of them. Specifically:

* **DSR > 0.5 at N=500, cluster-stable** means: after correcting for the project's effective multiple-testing budget (500 trials), the strategy's Sharpe is unlikely to be pure selection noise. **It does NOT mean the strategy is profitable going forward, only that the historical edge is unlikely to be a statistical artifact.**
* The **next honest gate is paper trading itself** — ≥ 4–8 clean weeks on the target venue with the actual order-placement pipeline. Sources of failure that DSR cannot rule out and that only paper trading can catch: signal stability under live data feed jitter, slippage divergence from the modelled rate, partial fills, exchange-side rejections, network reliability, regime shifts the historical sample never saw.
* Real-money deployment requires the paper-trading gate to be passed first, AND (per CLAUDE.md hard rule "Live orders only on testnet") a manual mainnet-migration code-path change. Neither has happened.

PAPER (testnet) status in the table below = "DSR-passed, currently running through `paper-place-orders` on Binance testnet as a deliberate validation step before any real money."

## All strategies at a glance

| # | Strategy / variant | Class | Status | Key metric | Finding |
|---|---|---|---|---|---|
| 1 | **TSMOM (28, 60) + SMA(200) gate on market-basket** | Single-signal trend, 7-asset basket | **PAPER (Binance testnet)** | DSR 0.770 — first config in the project to clear DSR > 0.5 at N=500 with margin | `findings/han_28d_tsmom.md` |
| 2 | TSMOM short-ensemble (lookbacks 28/60/120, etc) | Strategy family | Cluster-stable | DSR median 0.736 (7/7 pass) | `findings/cluster_stability.md` |
| 3 | TSMOM Han single lookbacks | Strategy family | Cluster-stable | DSR median 0.702 (6/6 pass) | `findings/cluster_stability.md` |
| 4 | PMA ratio ladder | Strategy family | Cluster-stable | DSR median 0.716 (6/6 pass) | `findings/cluster_stability.md` |
| 5 | SMA crossover ensemble | Strategy family | Cluster-FAILS | DSR median 0.431 (6/19 pass) | `findings/cluster_stability.md` |
| 6 | Market-basket index construction (pre-Han) | Aggregation primitive | Building block of #1 | DSR 0.658 (now superseded by #1) | `findings/market_basket_tsmom.md` |
| 7 | VolatilityTargetWrapper | Overlay (any strategy) | Asset-conditional | Helps ETH/SOL on Sharpe; hurts BTC on Calmar | `findings/vol_targeting_regime_gate.md` |
| 8 | Breadth filter (`GatedStrategy`) | Overlay (sequence count) | Does NOT improve on SMA200 | Basket DSR identical to SMA200 | `findings/breadth_filter.md` |
| 9 | 21-sleeve ensemble portfolio (3 strats × 7 assets) | Portfolio aggregation | Below threshold | DSR 0.425 (below 0.5; DD halved but DSR lowered) | `findings/ensemble_portfolio.md` |
| 10 | Symmetric B&H cost model (engine fix) | Engine / accounting (not a strategy) | Bug fixed | B&H was getting a free ~0.15% entry round vs strategies; symmetric cost now applied. **0 existing strategy-vs-B&H verdicts flipped** (Δ Sharpe ≤ ~0.05 across all pairs — within noise) | `findings/buy_and_hold_cost_symmetry.md` |
| 11 | Cross-sectional one-day reversal | Cross-section rotation | REJECT | Sharpe +0.01, DSR 0.001 | `findings/cross_sectional_reversal.md` |
| 12 | Cross-sectional momentum (rotation top-K) | Cross-section rotation | Available, used as benchmark | Sharpe 0.70–0.95 on 24-coin universe (benchmark in CTREND test) | (no standalone finding; lives in `backtest/cross_sectional.py`) |
| 13 | CTREND-proxy (price-only) | Cross-section ML ranker | **REJECT** | Sharpe 0.32–0.50; underperforms BH-BTC and CSM | `findings/ctrend_proxy_price_only.md` |
| 14 | MVRV-ratio overlay | BTC weekly tilt (on-chain) | **INCONCLUSIVE** (REJECT-leaning) | Sharpe 0.58–0.93 vs BH 0.65–1.11 | `findings/mvrv_overlay.md` |
| 15 | HMM 2-state regime overlay | BTC regime gate (Markov-switching) | **REJECT** | Loses 5/6 cuts to existing VolTarget; Sharpe 0.46–0.77 | `findings/hmm_regime_overlay.md` |

Status legend:
* **PAPER (Binance testnet)** — passes DSR > 0.5 at N=500, cluster-stable, currently being validated through `paper-place-orders` on Binance testnet. NOT cleared for real money. See "What PAPER status actually means" above.
* Cluster-stable / Cluster-FAILS — see `findings/cluster_stability.md` for the rule.
* Available / benchmark — implemented in code; not a standalone deploy candidate, used to verify other tests.
* REJECT — net of cost and OOS, does not beat the relevant in-stack benchmark.
* INCONCLUSIVE — empirical data does not support KEEP; sample size too small for hard REJECT.

---

## Currently paper-trading (Binance testnet)

### 1. TSMOM (28, 60) on market-basket index with SMA(200) gate
* **Universe:** equal-weight market-basket of 7 majors (BTC, ETH, BNB, SOL, ADA, XRP, DOGE). Monthly rebalance + on-`N_active`-change rebalance.
* **Signal:** TSMOM ladder `{0, 0.5, 1.0}` = mean of binary `sign(28d return), sign(60d return)`. SMA(200) gate zeroes the ladder when basket close < SMA.
* **Concatenated OOS Sharpe = +1.81** on the market-basket; **DSR = 0.770** at N=500.
* First config in the project to clearly survive DSR > 0.5 at N=500 with margin.

**What DSR 0.770 actually says.** After correcting for the 500-trial selection budget, the strategy's historical Sharpe is unlikely to be statistical noise. It does NOT say the strategy will be profitable going forward — only that the backtest's edge is unlikely to be a multiple-testing artifact. Backtest survival is the *previous* gate, not the *final* one.

**Current state and what comes next.**
* Now: running through `paper-place-orders` daily on Binance testnet (see `src/trade_lab/execution/README.md`).
* Next gate: at least 4–8 clean weeks of testnet paper trading, during which signal stability, slippage divergence, partial-fill behaviour, and network-failure handling are observed against the actual order pipeline.
* If testnet validation passes: a deliberate code-path migration to Kraken (CLAUDE.md hard rule "Live orders only on testnet" — mainnet is NOT a flag-flip, it's a manual engineering decision plus KYC plus exchange-specific market-constraint validation).
* Until paper trading is done: this strategy is **not cleared for real money**, even though it has cleared every backtest-side gate.

* Finding: `findings/han_28d_tsmom.md`.

---

## Tested at the strategy family level (cluster-stability discipline)

### 2. TSMOM short-ensemble (multiple lookback pairs)
Median DSR 0.736 across 7 cluster neighbours; **7 of 7 pass** DSR > 0.5. Best individual is `(28, 60)` — the deployed strategy.

### 3. TSMOM Han single lookbacks
Median DSR 0.702 across 6 cluster neighbours; **6 of 6 pass** DSR > 0.5. Adding the short-ensemble (combining two lookbacks) gives a small Sharpe lift over the best single lookback.

### 4. PMA ratio ladder (Detzel et al. 2021)
Median DSR 0.716 across 6 cluster neighbours; **6 of 6 pass**. PMA is a legitimate trend signal and cluster-stable, but the deployed TSMOM (28, 60) on the basket clears the bar with more margin and is simpler. PMA is implemented in `strategies/pma_ratio.py` for research use; not deployed.

### 5. SMA crossover ensemble
**FAILS cluster test**: median DSR 0.431, only 6/19 individual configs pass. Single best is cherry-picked. Lesson: SMA crossover survives single-config DSR but not cluster-stability.

All four families documented in `findings/cluster_stability.md` — the core methodology document that demotes SMA crossover and validates the TSMOM family.

---

## Overlays / wrappers (apply to a base strategy)

### 7. VolatilityTargetWrapper (Moreira-Muir style)
* Wraps any base strategy with `signal × (annual_vol_target / realized_vol)` capped at 1.0.
* **Asset-conditional**: helps ETH / SOL / most alts on Sharpe and Calmar; **hurts BTC on Calmar specifically** (BTC has fat right tail vol-targeting cuts off).
* Used as an in-stack benchmark in the HMM regime overlay test (#15) — HMM is essentially a duplicate of vol-target, and loses to it.
* Finding: `findings/vol_targeting_regime_gate.md`.

### 8. Breadth filter (`GatedStrategy` wrapper)
Adds a "breadth ≥ K%" filter on top of the SMA200 regime gate. **Does NOT improve** the market-basket TSMOM (basket DSR identical at 0.658). **Asset-conditional improvements only**: helps ETH (DSR 0.28 → 0.45), hurts BTC (0.31 → 0.13). Stacking breadth on top of SMA200 strictly hurts: the filters are substitutes, not complements. Finding: `findings/breadth_filter.md`.

### 10. Symmetric B&H cost model (engine fix, not a strategy)
**The bug.** The legacy `buy_and_hold` benchmark didn't pay an entry round of fee+slippage while strategies did — a quiet ~0.15% head-start to B&H on every comparison.

**The fix.** `engine.buy_and_hold_with_costs` now charges B&H the same entry round any strategy pays.

**The "result" — what we were checking after the fix.** After applying symmetric costs, every existing (strategy × asset) comparison from prior findings was re-evaluated to see whether any strategy that had been WINNING vs B&H now loses, or vice versa.
* **0 verdicts flipped.** Every previous KEEP-vs-B&H or REJECT-vs-B&H stayed the same.
* Δ Sharpe in the worst-affected pair was ≤ ~0.05 — inside the noise band of a single run.
* Interpretation: the bug was real, but its magnitude (~15 bps/year for typical turnover) is smaller than every margin existing verdicts cleared. The fix matters going forward (no future comparison will be biased the same way), but it does not retroactively change anything we believed.

Full Δ table per strategy × asset in `findings/buy_and_hold_cost_symmetry.md`.

---

## Portfolio-level aggregation

### 9. 21-sleeve ensemble portfolio (3 strategies × 7 assets)
* Equal-weight, dynamic 1/N_active, rebalance-on-universe-change costing.
* Concatenated OOS Sharpe **+1.13**, DSR **0.425** at N=500.
* Single best sleeve (`pma_medium × BNB` with vol30) achieves Sharpe +1.27, DSR 0.564 — **better than the portfolio**. Diversification halved max DD (good) but lowered DSR (the "many unconfirmed bets has higher DSR than one" intuition is false here).
* Useful as a sanity benchmark; not a deploy candidate. Finding: `findings/ensemble_portfolio.md`.

---

## Cross-sectional rotation strategies

### 11. Cross-sectional one-day reversal
**REJECT.** Annualized Sharpe +0.01, DSR @ N=500 = 0.001. The literature exists (Zaremba 2021, Bianchi 2022) but does not survive on Binance majors with realistic costs and our universe. Critical failure mode: "buy losers" → strategy keeps buying LUNA / FTT / UST as they fall to zero. Documented as a verdict that prevents future re-attempts of the same shape. Finding: `findings/cross_sectional_reversal.md`.

### 12. Cross-sectional momentum (top-K rotation by past return)
Implemented in `backtest/cross_sectional.py` (`run_cross_sectional_momentum`). Used as a benchmark in CTREND-proxy test on the 24-coin coinmetrics universe: Sharpe 0.70–0.95 across (cost × subperiod) cuts. **Not deployed standalone** — the basket-level TSMOM (#1) outperforms with simpler operations.

### 13. CTREND-proxy (price-only) — Fieberg et al. JFQA 2024
**REJECT.** Pooled Ridge on 6 price MA-ratio features at 6 windows, weekly top-quintile, 730-day train. Underperforms BH-BTC and existing CSM on every subperiod × cost cut. Kraken net-negative (CAGR −3.3% full). DSR ≈ 0 at N=500. **Important asymmetry caveat:** rejection is of the proxy, not the paper — volume half omitted (Coin Metrics community has only reported volume; trusted volume paid-tier), and pooled Ridge is not Fieberg's Fama-MacBeth estimator. Faithful V2 would need paid data + FMB. Finding: `findings/ctrend_proxy_price_only.md`.

---

## On-chain overlays

### 14. MVRV-ratio overlay (weekly BTC tilt)
**INCONCLUSIVE** (empirically tilting REJECT). Underperforms BH-BTC on Sharpe / Calmar / CAGR on every subperiod × cost. DD reduction is real (−64% vs BTC's −84% pre-ETF) but disproportionately costs return. DSR ≈ 0 at N=500; ~2–3 BTC market cycles in available data is too small for a confident hard REJECT. Compass-artifact prior of INCONCLUSIVE confirmed. **Important caveats:** ratio thresholds approximate the canonical Z-score (paid-tier `CapMVRVZ` / `CapRealUSD` not available on community tier); a faithful Z-score implementation might find different thresholds, but the binding constraint is sample size. Finding: `findings/mvrv_overlay.md`.

---

## Regime-switching overlays

### 15. HMM 2-state regime overlay
**REJECT.** Gaussian HMM on daily log-returns, refit weekly on trailing 730d, long when **filtered** (not smoothed) P(bull) > 0.5 else cash. Per the user's operative decision rule for this candidate ("must beat existing VolTarget — duplicate test"), HMM loses 5 of 6 (cost × subperiod) cuts. The one HMM win is Binance-only / post-ETF-only (2.4y, Sharpe 0.77 vs VolTarget 0.49) but vanishes on Kraken (HMM ties VolTarget at 0.46) and falls below DSR threshold at N=500. **Critical implementation invariant**: uses forward-only `_hmmc.forward_log(...)` for filtered probabilities; never `predict_proba` (which is smoothed and uses future data). Finding: `findings/hmm_regime_overlay.md`.

---

## Methodology & session writeups (not strategies)

* `findings/strategy_test_session_2026_05_29.md` — sweep of three candidates (CTREND, MVRV, HMM) in one sitting; reusable methodology lessons.
* `findings/literature_review_v1.md`, `_v2.md`, `_v3.md` — three external survey indexes; navigation maps, not deploy candidates.
* `findings/cluster_stability.md` — the core cluster-stability discipline that demotes single-config DSR survivors.

## Pinned constants

* `PROJECT_NUM_TRIALS = 500` (CLAUDE.md hard rule)
* Conservative pooled `sd_trial_sharpes ≈ 0.7`; `E[max Sharpe over 500 trials] ≈ 2.14`
* Single-config DSR threshold for deployment: 0.5 cluster-median.

Last updated: 2026-05-29.
