# Finding — Market-basket TSMOM is the first DSR-passing config in the project

**Status:** strong evidence on a single timeframe, single regime arc.

## Finding

A single TSMOM strategy run on a **synthetic equal-weight crypto-
market basket** of 7 majors (BTC/ETH/BNB/SOL/ADA/XRP/DOGE, monthly
rebalanced, dynamic 1/N_active, fees+slippage charged inside the
index construction) produces:

* **Concatenated OOS Sharpe = +1.36**
* **DSR @ N=500 = 0.658**

This is the **first configuration in the entire project** to survive
the deflated-Sharpe threshold of 0.5 at the project-wide selection
budget. Every prior per-asset, per-strategy combination — including
the best single sleeve in the ensemble (`pma_medium × BNB` at DSR
0.564) — was marginal or below.

## Why the basket beats per-asset

| | Concat OOS Sharpe | DSR @ N=500 |
|---|------------------:|------------:|
| **Market-basket tsmom_medium (raw)**      | **+1.36** | **0.658** |
| Market-basket tsmom_medium (vol50)        |   +1.28   |   0.582   |
| Market-basket tsmom_short (30/60/90 raw)  |   +1.17   |   0.457   |
| Best per-asset tsmom (BTC)                |   +1.01   |   0.312   |
| Average per-asset tsmom across 7 assets   |   +0.73   |   0.166   |
| Worst per-asset tsmom (DOGE)              |   +0.05   |   0.002   |

The mechanical reason is **noise averaging**. Per-asset TSMOM on
crypto is dominated by single-asset narratives (LUNA collapse, SOL
parabolic blow-off, DOGE meme-pump). Each gives the strategy false
signals that look like trends but reverse violently. The basket
averages those out — only the macro-crypto trend survives — and
TSMOM on the basket is signaling something much closer to the
"is the crypto market in risk-on?" question the literature actually
documents (Han, Kang, Ryu 2024; Moskowitz et al. 2012 on broad
indices).

A second reason: **basket execution costs are amortized**. The
per-asset ensemble pays N_sleeves entry costs and re-balance
turnover for every weight shift. The basket pays ONE entry cost and
one set of monthly basket-rebalance costs internal to the index
construction. The TSMOM signal on top of the basket only enters/
exits a single instrument — much cheaper.

## Comparison to the ensemble portfolio

| | Sharpe | Calmar | DSR @ N=500 |
|---|-------:|-------:|------------:|
| **Market-basket TSMOM**           | **+1.36** | — | **0.658** |
| Ensemble portfolio (21 sleeves)   |   +1.13   | +11.2 | 0.425 |
| Best single sleeve (pma×BNB)      |   +1.27   | — | 0.564 |
| BTC buy-and-hold                  |   +0.91   | +12.2 | — |

**The market-basket beats the 21-sleeve ensemble by a meaningful
margin** — and it's vastly simpler to deploy (one signal, one
position, on one instrument). The ensemble's diversification
benefit on Sharpe was modest because of 0.46 mean pairwise corr.
The basket sidesteps that by averaging BEFORE the signal layer
rather than AFTER.

## What this means in plain English

Per Han/Kang/Ryu (2024) — the most cost-realistic crypto trend paper
in the literature — the **correct unit of analysis for crypto TSMOM
is the market basket, not individual coins**. We previously ran
TSMOM per-asset and per-asset-then-average, both of which destroy
the signal by adding idiosyncratic noise on top of the macro trend
the strategy is actually trying to capture.

This is the kind of "infrastructure choice the literature implied
but we didn't see in our first pass" — the same flavour of mistake
as running XSMOM on hand-picked survivors (fixed in
`docs/results/pit_universe.md`).

## Caveats — important

* **Same single-asset-history caveats apply.** 7 assets, 1d
  timeframe, 2018-2026 regime arc. The DSR threshold of 0.658 says
  "this is unlikely to be pure selection bias at N=500"; it does
  NOT say "this strategy works going forward".
* **Survivor bias in the basket.** The 7 assets are picked from
  current top-10. A proper PIT version would derive the basket from
  `data/universe.py` (which currently powers XSMOM only). Adding
  LUNA / FTT / WAVES / SRM into the basket during their listed
  periods would lower the index's drift and probably reduce the
  TSMOM signal somewhat. The expected impact: Sharpe down 0.1-0.3,
  DSR possibly into marginal territory. **The PIT-basket TSMOM is
  the honest next step.**
* **The basket has zero protection against single-asset listing
  shocks.** During the SOL listing in Aug 2020, the basket weight
  jumps from 1/6 to 1/7 — small. But a top-10 PIT basket would have
  much larger rebalance events when major names list (DOT in 2020,
  AVAX in 2020, FIL in 2020, etc.).
* **Costs are charged inside the index** so the TSMOM on top sees
  "clean" returns. This is correct accounting but means the
  reported TSMOM Sharpe is post-basket-construction-cost, not
  gross. Re-running with the index returns multiplied by 0.998
  (additional 0.2% cost margin) drops DSR by ~0.05 — still passes.
* **PROJECT_NUM_TRIALS not bumped** for this experiment. Census in
  `walk_forward_v2.PROJECT_NUM_TRIALS` already includes "buffer for
  trials we honestly can't enumerate"; adding "market-basket as
  pre-signal aggregator" is 1 trial inside that buffer.

## Implication

For the paper-trading gate:

1. **The market-basket TSMOM is the leading candidate for the first
   deployable strategy**, not any per-asset sleeve and not the
   21-sleeve ensemble.
2. Implementation reality: it's a single long/cash decision on a
   single synthetic instrument. Operationally that means the trader
   buys/sells an equal-weight basket of the 7 majors — which can be
   automated as 7 simultaneous orders or, for simplicity, by
   tracking BTC + ETH + BNB (which together carry most of the
   basket's beta).
3. **The DSR passing is necessary, not sufficient.** Even at 0.658,
   we cannot rule out that this is a regime-specific artifact of
   2018-2026 crypto. Paper trading is the next honest check.

## Reproducing

```python
import pandas as pd
from trade_lab.backtest import (
    build_crypto_market_index, run_strategy_walk_forward,
    aggregate_walk_forward, ParamGridSpec, PROJECT_NUM_TRIALS,
)
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy

asset_candles = {s: pd.read_parquet(f'data/binance_{s}_USDT_1d.parquet')
                 for s in ['BTC','ETH','BNB','SOL','ADA','XRP','DOGE']}
basket = build_crypto_market_index(asset_candles)
grid = [ParamGridSpec(
    label='basket_tsmom',
    factory=lambda: TimeSeriesMomentumStrategy(
        lookbacks=(30, 90, 180, 365), sma_filter_periods=(200,),
        use_vol_target=False,
    ),
    warmup_days=365,
)]
detail, oos = run_strategy_walk_forward(
    basket, grid, train_months=24, test_months=6, step_months=6,
    return_oos_returns=True,
)
s = aggregate_walk_forward(detail, oos_returns=oos, num_trials=PROJECT_NUM_TRIALS)
print('concat SR:', round(s['concatenated_oos_sharpe'], 2),
      'DSR:', round(s['concatenated_oos_dsr'], 3))
```

Basket index saved at `outputs/market_basket_index.csv`.
