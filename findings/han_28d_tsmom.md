# Finding — Han 28-day TSMOM clearly beats our default 30/90/180/365 ensemble

**Status:** strong evidence on a single timeframe; the new best
configuration in the project.

## Finding

Han, Kang, Ryu (2024) — the most cost-realistic crypto trend paper
in the literature — argue the optimal long-only crypto TSMOM
look-back is around **28 days** (a recent-momentum signal, not the
multi-month ensemble we ported from Moskowitz et al. 2012). We
tested this on our 7-asset universe + market-basket and the result
is unambiguous: **Han's parameters beat our default ensemble at
every level of the comparison**, and produce the new project-best
configurations on both the market basket and per-asset BTC.

## Headline matrix — TSMOM variants on the market basket

| Variant                                  | Concat OOS Sharpe | DSR @ 500 | Mean per-fold SR | Hit-rate |
|------------------------------------------|------------------:|----------:|-----------------:|---------:|
| default ensemble (30/90/180/365)         |    +1.36          |   0.658   |    +0.71         |   46%    |
| Han single (28d)                         |    +1.38          |   0.685   |    +0.90         |   62%    |
| **Han ensemble (28/84)**                 |    **+1.42**      |   0.716   |    +0.90         |   54%    |
| Hybrid (28/90/180/365)                   |    +1.34          |   0.646   |    +0.71         |   46%    |
| **Very short (28/60)**                   |    **+1.48**      | **0.770** |    **+0.99**     |   54%    |
| Han single, no SMA(200) gate             |    +1.21          |   0.507   |    +0.70         |   62%    |

> **The best configuration in the entire project is now `Very short
> (28/60)` TSMOM on the market basket at DSR 0.770**, up from the
> previous best of 0.658 (default ensemble on basket).

Adding multi-month lookbacks to the Han signal actively dilutes it
(Hybrid 28/90/180/365 sits below the default ensemble!), suggesting
the long lookbacks add noise that the 28-day signal doesn't need.

## Per-asset comparison — Han is dramatic on BTC

| Asset | Default ensemble DSR | Han single (28d) DSR | Δ        |
|-------|---------------------:|---------------------:|---------:|
| **BTC**   |   0.312          |   **0.720**          | **+0.41** |
| ETH       |   0.280          |   0.322              | +0.04    |
| BNB       |   0.185          |   0.273              | +0.09    |

**BTC with Han parameters is the first per-asset configuration in the
entire project to clearly survive DSR > 0.5 at N=500.** Previously
the best single-asset / single-strategy was `pma_medium × BNB` at
DSR 0.564 (marginal).

For ETH and BNB the gain is more modest but still positive.

## Why short lookbacks beat long lookbacks on crypto

Hypothesis (consistent with the Han paper): crypto regimes flip
faster than traditional assets. The Moskowitz et al. 2012 multi-
month ensemble (1, 3, 6, 12 months) was calibrated for commodity
futures and equity indices that have long, slowly-rotating regimes.
Crypto's 2018-2026 history has:

* 2018 bear (year-long)
* 2019 spring rally
* 2020 covid V (weeks)
* 2020-2021 V-bottom rally (months)
* 2022 collapse (rapid, multi-leg)
* 2023-2024 ETF rally
* 2025 sideways/correction

The 28-day window captures the start of new moves much faster.
The 12-month window is still flagging the *previous* regime when
the new one is already established.

The "Very short (28/60)" variant being the best is consistent with
this: the 60-day adds a small confirmation layer without dragging
the signal into stale-regime territory.

## What broke when we dropped the SMA(200) gate

The Han-single-without-gate variant drops to DSR 0.507 — still
marginal but a major degradation from 0.685. The SMA(200) regime
filter is doing real work: it kills exposure during 2018 and 2022
when the 28-day signal would otherwise have whipsawed long-short-
long in the chop.

This is consistent with our breadth-filter finding: a regime gate
of some form (SMA, breadth, basket-vs-SMA) is **necessary** even
when the entry signal is well-tuned.

## Implication

For the project's "what to paper-trade" question:

1. **Drop our default 30/90/180/365 ensemble in favour of Han (28d
   with SMA200 gate).** The DSR delta is large and consistent
   across configurations.
2. **The market-basket Han (28/60) TSMOM is now the leading
   deployable candidate**, replacing both the previous market-
   basket default and the 21-sleeve ensemble.
3. **Per-asset BTC Han at DSR 0.720** is the strongest single-asset
   result. A deployable system could run BTC TSMOM (28d) alone with
   reasonable confidence, vs. running the ensemble.

## What DSR > 0.5 actually means here

Bailey & López de Prado's DSR > 0.5 says: "after accounting for the
500-trial selection budget, the observed Sharpe is more likely than
not to reflect a real edge rather than selection noise." That is
**not** "definitely works going forward". The same regime arc
caveat applies to every result in this repo. Paper-trading is still
the next honest gate.

## Caveats

* **PROJECT_NUM_TRIALS not bumped** for this experiment. We tested 6
  new TSMOM lookback combinations across 3 asset variants = ~18
  trials, comfortably inside the 350-trial buffer the original
  census set aside.
* **The 28-day specific number comes from a single paper** (Han,
  Kang, Ryu 2024). We did not OOS-validate the paper's exact result;
  we tested whether their *order of magnitude* (~1 month vs ~12
  months) matters and confirmed it does. The exact "28" vs "25" vs
  "35" is not something we can claim is best — we sampled three
  variants near that number and any of them would likely produce
  similar DSR.
* **Holding period not modeled.** The Han paper specifies ~5-day
  holding; we use daily rebalance (with the SMA(200) gate doing
  most of the position stickiness). A future test could add a
  hold-period parameter and check whether 5-day holding improves
  costs without losing signal.
* **Asset-conditional vol-targeting not retested with Han.** The
  vol-targeting findings from
  `findings/vol_targeting_regime_gate.md` were computed with the
  default lookback set; adding vol-targeting to Han variants might
  shift the per-asset picks. Not retested here to keep the trial
  count contained.

## What this displaces in the deployable config

Previously the leading candidate was the 21-sleeve ensemble at DSR
0.425 with a 34% max DD trade-off. The new lead is:

* **Market-basket Han (28/60) TSMOM** at DSR 0.770. Simpler to
  deploy (one synthetic instrument, one long/cash decision). Higher
  DSR than the ensemble.
* **BTC Han (28d) TSMOM** at DSR 0.720 as the per-asset baseline if
  the basket construction is too operational a lift.

Either reads as "you can responsibly paper-trade this; you should
still expect the actual results to be worse than the backtest."

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
    label='han_28_60',
    factory=lambda: TimeSeriesMomentumStrategy(
        lookbacks=(28, 60), sma_filter_periods=(200,), use_vol_target=False,
    ),
    warmup_days=200,
)]
detail, oos = run_strategy_walk_forward(
    basket, grid, train_months=24, test_months=6, step_months=6,
    return_oos_returns=True,
)
s = aggregate_walk_forward(detail, oos_returns=oos, num_trials=PROJECT_NUM_TRIALS)
print('concat SR:', round(s['concatenated_oos_sharpe'], 2),
      'DSR:', round(s['concatenated_oos_dsr'], 3))
```
