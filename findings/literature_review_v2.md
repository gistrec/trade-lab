# Literature review v2 (external research report)

Source: `deep-research-report v2.md` (external research run, extracted 2026-05-28).
This is a **navigation map, not a to-do list.** Nothing here is deployable without full validation (walk-forward + DSR at fixed `PROJECT_NUM_TRIALS`). Do not modify the current TSMOM strategy on the basis of any item below.

## Confirmations of current direction (no action needed)
- **TSMOM as strongest-evidence class** — Hurst-Ooi-Pedersen (1880–2016, 67 markets), Rozario et al. (BTC walk-forward), Deprez-Frömmel (BTC with FDR + OOS). Project already on TSMOM (28, 60).
- **Vol-targeting as overlay, not standalone alpha** — Moreira-Muir (JoF). Project already has `VolatilityTargetWrapper`; per-asset decision logged in `findings/vol_targeting_regime_gate.md`.
- **Reversal in crypto is an illiquidity premium that vanishes on liquid value-weighted baskets** — Zaremba et al., Bianchi et al. (Riksbank, 20/30 bps costs), Kakushadze-Yu. Independently confirms `findings/cross_sectional_reversal.md`.
- **Survivorship-bias-free universe is non-negotiable for cross-sectional work** — Zarattini, Liu-Tsyvinski-Wu. Project uses Coin Metrics + curated PIT registry.
- **Simple-and-honest > complex-and-fragile** — recurring conclusion across the report. Already encoded as principle in CLAUDE.md.

## Potential TIER-2 strategies (future research, NOT part of current TSMOM)
- **Funding / perp carry** — Christin-Routledge-Soska-Zetlin-Jones (Carnegie Mellon, in-sample Sharpe on Binance — number not validated out-of-sample), Schmeling-Schrimpf-Todorov (BIS). Requires data the project does NOT have: funding rate history, mark/index price, per-symbol funding intervals, basis. Would be a separate module, not an overlay on TSMOM.
- **Donchian breakout on liquid top-N** — Zarattini-Pagani-Barbon (SSRN, survivorship-bias-free, net-of-fees). Project already has `donchian_trend`; this is a literature anchor for that family, not a new recipe.

## Explicitly NOT to add to current TSMOM strategy
Per CLAUDE.md hard rules, execution must replicate the validated backtest exactly. No carry overlay, no Donchian blending, no regime-ML filter, no reversal sleeve on top of the deployable TSMOM (28, 60) on the 7-asset basket. Any such change is a different untested strategy and must go through full walk-forward + DSR at fixed `PROJECT_NUM_TRIALS = 500` before any consideration.

Last reviewed: 2026-05-28.

## How to use this file
Re-read at the start of any new research cycle (i.e., when current deployment has matured and capacity exists for TIER-2 work). Do NOT consult during execution-phase work — it will only distract.
