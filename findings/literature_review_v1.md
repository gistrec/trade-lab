# Literature review v1 (external research report — strategy survey)

Source: `compass_artifact_wf-ee29bed8-d8df-4752-b300-60fbd2d40660_text_markdown.md` (external research run, extracted 2026-05-28).
This is a **navigation map, not a to-do list.** Nothing here is deployable without full validation (walk-forward + DSR at fixed `PROJECT_NUM_TRIALS`). Do not modify the current TSMOM strategy on the basis of any item below.

## Confirmations of current direction (no action needed)
- **TSMOM as foundational evidence class** — Moskowitz-Ooi-Pedersen (2012 JFE, 58 instruments), Hurst-Ooi-Pedersen (2017 JPM, 137 years × 67 markets), Liu-Tsyvinski (2021 RFS, strong TSMOM on BTC/ETH/XRP at 1–4 week horizons). Project already on TSMOM (28, 60).
- **MA / P-vs-MA filters as legitimate trend signal** — Detzel-Liu-Strauss-Zhou-Zhu (2021 Financial Management, BTC OOS), Corbet-Eraslan-Lucey-Sensoy (2019 FRL). Project already implements `pma_ratio` and `sma_cross`.
- **Vol-targeting as risk overlay, not standalone alpha** — Moreira-Muir (2017 JoF), Hurst-Ooi-Pedersen (2017). Project already has `VolatilityTargetWrapper`.
- **DSR + walk-forward as anti-overfitting discipline** — Bailey-Lopez de Prado (2014 JPM), PBO / Combinatorial Purged CV (Lopez de Prado 2018). Project already uses both with `PROJECT_NUM_TRIALS = 500` pinned.
- **Reversal is an illiquidity / small-cap effect** — Cakici-Zaremba (2021 IRFA, >3600 coins; largest coins show daily momentum, not reversal). Independently confirms `findings/cross_sectional_reversal.md`.
- **Survivorship-bias-free universe is mandatory for cross-section** — Liu-Tsyvinski-Wu (2022 JoF three-factor model). Project uses Coin Metrics + curated PIT registry.
- **Trend-following underperforming HODL in pure bull market is expected** — Hurst-Ooi-Pedersen (2017): trend value shows up via drawdown reduction, not per-year outperformance. This is the rationale for the SMA(200) gate, not a reason to remove it.

## Potential TIER-2 strategies (future research, NOT part of current TSMOM)
- **Cross-sectional momentum on liquid top-N with weekly rotation** — Liu-Tsyvinski-Wu (2022 JoF), Tzouvanas et al (2019), Starkiller Capital implementation. Project has `cross_sectional_momentum` in research codebase; whether it survives DSR at fixed N is the gating question, not the literature.
- **Donchian breakout on liquid top-N** — Clenow "Following the Trend"; v2 review flags the same family. Project has `donchian_trend`; this is a literature anchor, not a new recipe.
- **Funding / perp carry** — Schmeling-Schrimpf-Todorov (BIS WP 1087), Christin et al (CMU 2022, in-sample on Binance). Requires data the project does NOT have: funding rate history, mark/index price, per-symbol funding intervals, basis. Spot-only project cannot run this; separate module after exchange/account expansion.
- **Basis / cash-and-carry on dated futures** — Schmeling-Schrimpf-Todorov (BIS WP 1087). Same data requirement as carry plus futures chain. BIS paper documents post-spot-ETF basis compression — alpha may already be largely gone.

## Explicitly NOT to add to current TSMOM strategy
Per CLAUDE.md hard rules, execution must replicate the validated backtest exactly. No cross-sectional rotation overlay, no Donchian blending, no carry sleeve, no MA-based regime switch beyond the existing SMA(200) gate. Any such change is a different untested strategy and must go through full walk-forward + DSR at fixed `PROJECT_NUM_TRIALS = 500` before any consideration.

Last reviewed: 2026-05-28.

## How to use this file
Re-read at the start of any new research cycle (i.e., when current deployment has matured and capacity exists for TIER-2 work). Do NOT consult during execution-phase work — it will only distract.
