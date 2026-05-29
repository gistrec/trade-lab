# Literature review v3 (external research report — post-ETF survey)

Source: `compass_artifact_wf-f4cbab2e-6009-4115-95c1-8eeabfd39559_text_markdown.md` (external research run, extracted 2026-05-29).
This is a **navigation map, not a to-do list.** Nothing here is deployable without full validation (walk-forward + DSR at fixed `PROJECT_NUM_TRIALS`). Do not modify the current TSMOM strategy on the basis of any item below.

## Confirmations of current direction (no action needed)

- **TSMOM/MA-trend as the strongest evidence class on price-only spot** — Detzel-Liu-Strauss-Zhou-Zhu (2021 *Financial Management*, Sharpe 2.48 vs 1.82 BH) flagged as the *single* rigorous OOS survivor in the survey. Project already on TSMOM (28, 60) + SMA gate.
- **Reversal on large-cap is dead net of cost** — Cakici-Zaremba (2021 IRFA), Bianchi et al. (Riksbank) reconfirmed. Independently consistent with `findings/cross_sectional_reversal.md`.
- **Post-ETF (≥ Jan 2024) is a structural break, not a regime fluctuation** — *Institutionalization of Bitcoin* (2025–2026), *Bitcoin ETFs and structural decoupling* (Taylor & Francis 2026). BTC-altcoin correlations collapsed; momentum/trend degraded but did not disappear. Material context for any cross-sectional rotation strategy.
- **On-chain net-flows / USDT inflows do not predict daily returns** — Chi-Chu-Hao (arXiv:2411.06327). Effect is intraday at best (1-hour for USDT, 4-hour for BTC). Reasserts the project's choice not to chase intraday on-chain signals.
- **HMM/Markov-switching evidence is on volatility, not returns** — Caporale & Zekokh. Compass framing matches the empirical finding in `findings/hmm_regime_overlay.md` (HMM regime overlay REJECTED as duplicate of existing VolatilityTargetWrapper).
- **Volume management / volatility-targeting helps momentum crash risk** — *Cryptocurrency momentum has (not) its moments* (Springer FMPM 2025). Project already encodes this via `VolatilityTargetWrapper`.

## Potential TIER-2 strategies (future research, NOT part of current TSMOM)

- **CTREND v2 (faithful)** — Fieberg et al. JFQA 2024. CTREND-proxy already tested as REJECT in `findings/ctrend_proxy_price_only.md`, but two intentional deviations from the original (no volume features, pooled Ridge instead of Fama-MacBeth) mean the proxy result does NOT refute the paper. Faithful V2 requires: paid Coin Metrics for trusted volume + cross-sectional FMB with rolling coefficients. Separate research budget when paid data is available.
- **MVRV-Z faithful (paid data only)** — Palazzi 2026, Liu-Zhang 2023. Ratio-only proxy tested as INCONCLUSIVE in `findings/mvrv_overlay.md`. Paid `CapRealUSD` + `CapMVRVZ` would enable proper Z-score, but the binding constraint is sample size (~2–3 BTC cycles), not the ratio approximation. Effectively impossible to convert to KEEP on BTC alone.
- **Block-count / network-activity fundamentals** — *A Comprehensive Look…* (2024). Episodic; marked "Maybe" in compass.
- **Cross-asset macro filter (DXY / VIX / risk-on-off)** — slow, unstable, deteriorated post-ETF per Palazzi. Marked "Maybe (overlay only)" in compass.
- **Turn-of-month / EOM** — cheap to check; expect data-mining failure under strict DSR. Not yet tested.

## Explicitly NOT to add to current TSMOM strategy

Per CLAUDE.md hard rules, execution must replicate the validated backtest exactly. No CTREND overlay, no MVRV gate, no HMM regime switch, no macro filter, no seasonality calendar on top of the deployable TSMOM (28, 60) on the 7-asset basket. Any such change is a different untested strategy and must go through full walk-forward + DSR at fixed `PROJECT_NUM_TRIALS = 500` before any consideration.

The survey also confirms the project's existing refusals:

* **BTC exchange net-flows** as a return signal — refuted (Chi-Chu-Hao).
* **USDT inflows** as a return signal — intraday only, out of mandate.
* **Day-of-week / Monday effect** — does not survive 2015 (Qadan et al. 2022).
* **ML on-chain with extreme Sharpe (>5)** — classic overfit red flag; realistic OOS-blended benchmark is Sharpe ≈ 1.1.
* **Intraday seasonality** — out of mandate.
* **Funding/basis/perp signals** — out of mandate.

## Tested this session (2026-05-29)

All three candidates from this survey's "Top-3" have now been backtested with frozen configs in `findings/strategy_test_session_2026_05_29.md`. Two are REJECT, one is INCONCLUSIVE. No candidate from this survey is currently a deploy candidate. Survey-implied research priority going forward: data access (paid Coin Metrics) to enable CTREND v2 and faithful MVRV-Z, otherwise this survey is functionally exhausted at the community-tier data level.

Last reviewed: 2026-05-29.
