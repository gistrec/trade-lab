# Strategy test session — 2026-05-29 — three REJECTs

External prompt: `strategy_test_prompt.md` (extracted in this session).
External literature survey: `findings/literature_review_v3.md` (extracted).

Three candidates from the prompt's priority list tested in order. None
survived. Details for each in its own finding:

* `findings/ctrend_proxy_price_only.md`
* `findings/mvrv_overlay.md`
* `findings/hmm_regime_overlay.md`

## Verdicts at a glance

| # | Candidate | Verdict | Main reason | Sharpe vs strongest benchmark |
|---|---|---|---|---|
| 1 | CTREND-proxy (price-only) | REJECT | Underperforms BH-BTC and existing CSM on every (cost × subperiod) cut. Kraken cost-drag flips it net-negative (CAGR −3.3% full). | 0.32–0.50 vs CSM 0.70–0.81; BH-BTC 1.01 |
| 2 | MVRV ratio overlay | INCONCLUSIVE (REJECT-leaning) | Underperforms BH-BTC on Sharpe / Calmar / CAGR on every cut. DD reduction is real but disproportionately costs return. ~2–3 cycles of data — DSR no power for hard REJECT or for KEEP. | 0.58–0.93 vs BH-BTC 0.65–1.11 |
| 3 | HMM 2-state regime | REJECT | Per the user's operative rule for this candidate: must beat existing `VolatilityTargetWrapper`. Loses 5 of 6 (cost × subperiod) cuts to VolTarget. The one HMM win is Binance-only, post-ETF only (2.4y), cost-fragile (vanishes on Kraken). | 0.46–0.77 vs VolTarget 0.46–1.13 |

## Methodological lessons (reusable)

1. **Freeze the config before the first run.** Q1-style dispatch: any post-hoc parameter change to chase a result is a new trial that consumes the project's `PROJECT_NUM_TRIALS = 500` budget. Documented sensitivity sweeps are diagnostics, not verdict inputs.
2. **Filtered vs smoothed posteriors** is the single make-or-break invariant for HMM-class strategies. `hmmlearn.GaussianHMM.predict_proba(X)` returns smoothed (uses future data); the trading signal must use the forward-pass-only filtered posterior. Pin with a unit test that corrupts data after a chosen date and confirms equity at that date is unchanged.
3. **Purge/embargo whenever target horizon > 1 day.** CTREND's 7-day forward target requires at least a 7-day gap between train end and rebalance date — otherwise train target windows leak into the test period and produce a phantom edge.
4. **Publication lag on on-chain inputs.** On-chain data publishes t+1; using day-t value at day t is a small but real look-ahead. Lag by 1 bar.
5. **Asymmetry of interpretation for proxies.** Rejection of a deliberately reduced proxy does NOT refute the underlying paper. Document the deviations from the canonical estimator and what a faithful V2 would require.
6. **DSR with `PROJECT_NUM_TRIALS = 500` + conservative pooled `sd_trial_sharpes = 0.7`.** `E[max Sharpe over 500 trials] ≈ 2.14` — any single-config Sharpe below ~1.5 will fail DSR even at this conservative pool dispersion. Use this as the verdict floor.
7. **Test both cost scenarios** (Binance 0.10%+5bps, Kraken 0.40%+10bps). The Kraken-equivalent is the deciding metric — winning only on Binance is fragile.
8. **Test against existing in-stack benchmarks**, not just BH-BTC. A new strategy that loses to `cross_sectional_momentum` or `VolatilityTargetWrapper` is a duplicate worth rejecting independent of paper provenance.

## Data blockers (block future faithful tests)

These are *not* problems of the proxies — they are problems of the **available data** that prevent faithful V2 implementations on community-tier accounts:

* **Coin Metrics community** exposes only `volume_reported_spot_usd_1d` — reported volume, contaminated by wash trading and methodology drift. `volume_trusted_spot_usd_1d` is paid tier only. CTREND's volume half cannot be honestly tested without paid access.
* **Coin Metrics community** exposes only `CapMVRVCur` (raw MVRV ratio). `CapRealUSD` (realized cap) and `CapMVRVZ` (Z-score) are paid tier. MVRV-Z faithful reproduction is impossible without paid data; ratio thresholds are an approximation.
* **2–3 BTC market cycles** in 2017–2026 daily history. Even at perfect signal quality, DSR at `N=500` cannot statistically anoint a BTC-only single-asset strategy on this sample size. This is a structural floor on power, not a strategy property.

## What would change the verdicts

| Verdict | Condition for re-test |
|---|---|
| CTREND-proxy REJECT | Paid Coin Metrics for trusted volume + Fama-MacBeth-style cross-sectional regression (the original estimator, not pooled Ridge). |
| MVRV INCONCLUSIVE | Paid Coin Metrics for `CapRealUSD` to compute proper Z-score AND multi-cycle independent dataset (effectively impossible on BTC; would need an asset with longer cycle history). |
| HMM REJECT | Re-run after 2027 if the Binance post-ETF flicker survives another full cycle of data accumulation. Promoting the strategy on the 2.4-year sample alone is below project DSR discipline. |

## Project N_TRIALS bookkeeping

3 frozen-config strategy runs in this session. No grid sweeps. Sensitivity tables for MVRV are diagnostic only and were not used for verdicts. Implicit trial count added to the project pool: **3 of 500**.

Last reviewed: 2026-05-29.
