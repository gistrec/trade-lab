# Walk-forward validation: priority-5 strategies on BTC daily

Second of the five honesty checks queued after the initial implementation.
Question: **what happens to the in-sample Sharpe of `sma_cross`, `tsmom`
and `pma_ratio` once we evaluate them honestly on rolling 6-month
out-of-sample folds?**

## Setup

* Data: `data/binance_BTC_USDT_1d.parquet`, 2018-01-01 → 2026-05-27,
  3,069 daily bars.
* Window cadence: **24-month train / 6-month test / 6-month step**,
  yielding 13 OOS folds covering 2020-H1 .. 2026-H1.
* Train range: limited by the dataset to **2018-2019** for fold 0
  (the user's spec mentioned 2015 but Binance USDT pairs do not have
  that history).
* Objective: **Sharpe**, computed per-window (mean / std × √365).
* Costs: 0.10% fee + 0.05% slippage per turnover, charged symmetrically.
* Warmup: full pre-window candles are fed to the strategy so signals
  are valid on bar 1 of each fold. The "warmup feed" is *not* a leak —
  it mirrors what a live trader has access to. Metrics are computed
  on the test-window slice only.
* Purge: `purge_days=0` (immediate adjacency between train end and
  test start, matching live semantics). The runner supports
  `purge_days > 0` for stricter Lopez-de-Prado-style validation.

Parameter grids — small on purpose, to control multiple-testing
pressure:

| Strategy   | Grid size | Variants                                              |
|-----------|----------:|-------------------------------------------------------|
| sma_cross  |    19     | (fast, slow) in {10,20,30,50} × {50,100,150,200,300}  |
| tsmom      |     3     | short (30,60,90) / medium (30,90,180,365) / long (90,180,365) |
| pma_ratio  |     3     | short (5,10,20) / medium (5,10,20,50,100) / long (10,20,50,100,200) |

## Headline numbers — IS vs OOS gap

| Strategy   | Train mean Sharpe | OOS mean Sharpe | **Gap**  | Hit-rate | OOS median return |
|-----------|------------------:|-----------------:|---------:|---------:|------------------:|
| sma_cross  |       +1.45       |       +0.63      | **-0.82** |  62%     |       +14.5%       |
| tsmom      |       +1.25       |       +0.70      | **-0.55** |  46%     |        +0.0%       |
| pma_ratio  |       +1.19       |       +0.51      | **-0.68** |  62%     |        +3.5%       |

> **All three strategies lose ~0.55-0.82 Sharpe going from train to
> OOS.** That's roughly the magnitude Bailey & López de Prado (2014)
> document as typical IS→OOS shrinkage for trend-following on small
> samples. None of the OOS averages are above 1.0 once the look-ahead
> is honest.

The single high-Sharpe full-window numbers we reported in
`strategy_comparison.md` (Donchian 1.10, PMA 1.06, TSMOM 0.98) are
**in-sample** results. The OOS truth is ~0.5-0.7.

## Why DSR doesn't catch this

```
sma_cross:  observed SR +1.00 (full window), DSR = 0.992  →  passes  >0.95
tsmom:      observed SR +1.20 (full window), DSR = 0.999  →  passes
pma_ratio:  observed SR +1.06 (full window), DSR = 0.998  →  passes
```

All three pass the Deflated Sharpe threshold. **DSR catches selection
bias from multiple-testing on the same sample; it does NOT catch
parameter overfitting whose damage shows up only out-of-sample.** With
only 3-19 trials per strategy, even a large-sigma random walk would
struggle to produce SR > 0.5; the fact that all three observe SR > 1.0
in-sample says "this is not pure noise". The fact that OOS Sharpe is
half says "it's also not a stable edge".

Skeptical reading: DSR is necessary but not sufficient. Walk-forward
is sufficient where DSR is not.

## Selection dynamics — does the "best" variant move?

### sma_cross

|  Fold | Test window      | Selected      | Train SR | OOS SR | OOS ret |
|------|------------------|---------------|----------|--------|---------|
| 0    | 2020-01..2020-06 | sma_30_100    |  +1.19   |  -0.63 |  -33%   |
| 1    | 2020-07..2020-12 | sma_30_50     |  +1.16   |  +3.66 | +137%   |
| 2    | 2021-01..2021-06 | sma_30_50     |  +2.11   |  +2.16 |  +93%   |
| 3    | 2021-07..2021-12 | sma_30_50     |  +1.71   |  +0.69 |  +11%   |
| 4    | 2022-01..2022-06 | sma_20_50     |  +1.99   |  -1.39 |  -22%   |
| 5    | 2022-07..2022-12 | sma_10_100    |  +1.87   |  -1.64 |  -11%   |
| 6    | 2023-01..2023-06 | sma_10_100    |  +0.74   |  +1.21 |  +25%   |
| 7    | 2023-07..2023-12 | sma_20_300    |  +0.69   |  +1.89 |  +39%   |
| 8    | 2024-01..2024-06 | sma_30_300    |  +1.32   |  +1.72 |  +49%   |
| 9    | 2024-07..2024-12 | sma_30_300    |  +1.52   |  +1.79 |  +49%   |
| 10   | 2025-01..2025-06 | sma_30_300    |  +1.77   |  +0.82 |  +15%   |
| 11   | 2025-07..2025-12 | sma_20_300    |  +1.54   |  -0.82 |  -14%   |
| 12   | 2026-01..2026-05 | sma_50_100    |  +1.16   |  -1.23 |   -6%   |

**Five different (fast, slow) pairs across 13 folds.** The selection
ladder slow_period in particular slides from 50 → 100 → 300 over the
window — that's not a plateau, that's the bull regime increasingly
preferring slower trend-following. Pattern is plausible, but it means
"the best SMA today" is not "the best SMA in two years".

### tsmom

|  Fold | Selected                       | Train SR | OOS SR | OOS ret |
|------|--------------------------------|----------|--------|---------|
| 0    | tsmom_short_30_60_90           |  +1.39   |  -0.86 |   -7%   |
| 1    | tsmom_short_30_60_90           |  +0.98   |  +4.23 |  +87%   |
| 2    | tsmom_short_30_60_90           |  +2.18   |  +0.93 |   +8%   |
| 3    | tsmom_short_30_60_90           |  +1.66   |  -1.38 |  -11%   |
| 4    | tsmom_short_30_60_90           |  +1.44   |   0.00 |    0%   |
| 5    | tsmom_short_30_60_90           |  +1.71   |   0.00 |    0%   |
| 6    | tsmom_medium_30_90_180_365     |  -0.06   |  +1.62 |  +14%   |
| 7    | tsmom_short_30_60_90           |  +0.23   |  +1.40 |  +15%   |
| 8    | tsmom_short_30_60_90           |  +1.01   |  +2.59 |  +30%   |
| 9    | tsmom_short_30_60_90           |  +1.49   |  +1.37 |  +13%   |
| 10   | tsmom_medium_30_90_180_365     |  +1.57   |  -0.10 |   -2%   |
| 11   | tsmom_short_30_60_90           |  +1.44   |  -0.65 |   -6%   |
| 12   | tsmom_short_30_60_90           |  +1.20   |   0.00 |    0%   |

**TSMOM picks `short` 11/13 folds.** That's a stable selection — the
parameter sensitivity is low, which is what the literature wants. But
the **OOS return is 0.0% on 4/13 folds** because the SMA(200) regime
gate keeps the strategy in cash during bear / chop. That's the design
working as intended, but it also means hit-rate is only 46%.

The 2023-H1 fold is striking: train Sharpe -0.06 (i.e. losing on
train) but the picked-anyway medium variant produced +1.62 OOS. This
is randomness, not a reliable pattern.

### pma_ratio

|  Fold | Selected                       | Train SR | OOS SR | OOS ret |
|------|--------------------------------|----------|--------|---------|
| 0    | pma_long_10_20_50_100_200      |  +1.04   |  +0.49 |   +4%   |
| 1    | pma_long_10_20_50_100_200      |  +1.10   |  +4.20 |  +81%   |
| 2    | pma_long_10_20_50_100_200      |  +2.38   |  +1.12 |  +11%   |
| 3    | pma_medium_5_10_20_50_100      |  +1.72   |  +0.49 |   +4%   |
| 4    | pma_medium_5_10_20_50_100      |  +1.90   |  -3.55 |  -15%   |
| 5    | pma_long_10_20_50_100_200      |  +1.58   |  -2.08 |  -10%   |
| 6    | pma_long_10_20_50_100_200      |  -0.28   |  +2.18 |  +25%   |
| 7    | pma_long_10_20_50_100_200      |  +0.14   |  +1.08 |  +11%   |
| 8    | pma_long_10_20_50_100_200      |  +0.43   |  +2.02 |  +24%   |
| 9    | pma_long_10_20_50_100_200      |  +1.23   |  +1.80 |  +18%   |
| 10   | pma_medium_5_10_20_50_100      |  +1.84   |  +0.06 |   -0%   |
| 11   | pma_medium_5_10_20_50_100      |  +1.38   |  -0.58 |   -5%   |
| 12   | pma_medium_5_10_20_50_100      |  +0.97   |  -0.56 |   -4%   |

**Long / medium ladder alternates.** Short ladder never wins on
train — its short MAs are too noisy on BTC. Medium dominates the 2025-
2026 stretch, long dominates 2020-2024. There IS a regime dependency:
in fast bull regimes the short MAs adapt faster; in slower bull /
sideways the long ladder benefits.

## What worked across all three

* **2020-H2 (post-COVID rally)**: every strategy returned 60-140%.
  Anyone in the market made money.
* **2024 H1-H2 (ETF rally)**: 25-50% per fold.
* **2023-H1 picked-from-negative-train cases**: an instructive
  reminder that "selecting on positive train Sharpe" is not a
  sufficient filter. The runner just picks the *best* variant on
  train, even if it's negative.

## What broke for everyone

* **2022 H1 (terra collapse + 3AC)**: SMA -22%, TSMOM 0%, PMA -15%.
  Fast crashes whip-saw trend-following: the strategy enters on the
  late-stage upmove, then the position is held into a sharp reversal.
* **2022 H2 (FTX collapse)**: SMA -11%, TSMOM 0%, PMA -10%. Same
  pattern; the regime gate eventually puts everything in cash.
* **2026 H1 (current fold)**: all three are negative. The 2025-H2
  selection extrapolated to the 2026 regime; new train data would
  pick differently but we won't know until the fold ends.

## Recommendations

1. **Quote OOS Sharpe, not full-window Sharpe**, in any summary that
   doesn't explicitly say "in-sample only". The 0.6-0.7 number is the
   honest one.
2. **Hit-rate of 46-62%** matters at least as much as average return.
   A strategy that wins 6 of 13 folds and loses 7 is not a reliable
   moneymaker; it's a coin flip with positive expected value.
3. **Don't extrapolate parameter selection.** SMA's selection ladder
   slid from `slow=50` to `slow=300` over six years. The "right" slow
   period today is probably different from the right one in 2027.
4. **The 2022 H1-H2 cells are a warning.** Even with regime gates the
   strategies lost 10-22%. A real allocation has to absorb that.

## Reproducing

```bash
# Walk-forward all three on BTC, default 24/6/6 month windows, Sharpe objective.
python -c "
import pandas as pd
from trade_lab.backtest import (
    run_strategy_walk_forward, aggregate_walk_forward,
    build_sma_grid, build_tsmom_grid, build_pma_grid,
)
candles = pd.read_parquet('data/binance_BTC_USDT_1d.parquet')
for name, grid in [('sma_cross', build_sma_grid()),
                   ('tsmom',     build_tsmom_grid()),
                   ('pma_ratio', build_pma_grid())]:
    detail = run_strategy_walk_forward(candles, grid)
    detail.to_csv(f'outputs/wf_{name}_BTC.csv', index=False)
    summary = aggregate_walk_forward(detail)
    print(name, summary)
"
```

CSV outputs live in `outputs/wf_{sma_cross,tsmom,pma_ratio}_BTC.csv`
(13 rows each: train_*, test_*, selected_label, sharpe, return, dd).

## Sanity checks for the runner

If anyone tweaks the WF runner, these should still hold:

1. **No look-ahead**: appending garbage candles past the last fold
   must leave every prior fold's selection and metrics unchanged.
   (Tested: `test_appending_future_garbage_does_not_change_any_fold`.)
2. **Train metrics independent of purge gap**: setting `purge_days=14`
   shifts the test window but not the train window — train Sharpe per
   fold must be identical to `purge_days=0`. (Tested.)
3. **Warmup feed actually enables short-window tests**: a 50-day SMA
   on a 3-month test must trade at least sometimes. Without warmup
   feed the strategy would be permanently in the rolling-NaN region.
   (Tested: `test_warmup_feed_lets_strategy_produce_signals_on_short_test_window`.)
4. **Determinism**: same inputs → identical output frame. (Tested.)

## Next on the queue

Per the user's priority list — item #3: **volatility targeting as an
optional layer**. The goal is to compare strategy vs strategy +
vol_target=30% vs strategy + vol_target=50% across all three priority-5
strategies, to test whether vol-targeting closes the OOS gap or just
shifts it.
