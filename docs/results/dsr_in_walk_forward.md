# DSR in walk-forward — the hardest honesty check yet

Fourth honesty check. Question: **after correcting for the number of
parameter combinations the user has explored during the project's
lifetime, do any of the priority-5 strategies remain statistically
significant on Binance daily spot data 2020-2026?**

## Method

Two parallel computations, per Bailey & López de Prado (2014).

### Variant 1 — per-fold train DSR (diagnostic)

For each walk-forward fold:
1. Train every variant in the grid on the train slice (warmup from
   before train_start).
2. Compute per-period Sharpe of each variant.
3. Take the best Sharpe and compute Bailey-LdP DSR with
   - `num_trials = len(grid)` (variants tried IN THIS FOLD only)
   - `sharpe_std_dev = std(train_sharpes_in_grid, ddof=1)`

The result tells us: *given just this fold's grid, is the best train
Sharpe a robust pick or a noisy one?* High train DSR (>0.95) means the
best variant clearly dominates its sibling grid. Low train DSR means
the variants are similar and "best" is mostly luck.

This is a **diagnostic** — it doesn't predict OOS performance.

### Variant 2 — concatenated OOS DSR (honest verdict)

1. Walk-forward as normal; record per-bar test returns for every
   fold.
2. Stitch into one continuous series (with `step_months == test_months`
   the folds are adjacent and non-overlapping; duplicates dropped
   keep-first if any).
3. Compute Sharpe of the concatenated series, annualize by √365.
4. Apply Bailey-LdP DSR with
   - `num_trials = PROJECT_NUM_TRIALS = 500`
   - `sharpe_std_dev = 1/√T` (the null-distribution estimator — each
     trial's per-period SR estimate has std ≈ 1/√T under no-skill
     sampling; this is the textbook fallback when no panel of trial
     Sharpes is available).

### Project-wide `num_trials = 500` — fixed in code

```python
PROJECT_NUM_TRIALS = 500  # src/trade_lab/backtest/walk_forward_v2.py
```

Census of what we've actually evaluated in this repo:

* SMA crossover grid: 19 variants
* TSMOM ensemble configs: 3 baseline + 6 sensitivity
* PMA ladder configs: 3 baseline + 6 sensitivity
* Donchian rebalance-threshold sweep: 4
* Vol-target wrapper sweep: 15 (5 targets × 3 strategies)
* XSMOM knobs: 12 (top_k × weighting × BTC-gate)
* 7-asset × 3-strategy × 3-vol-variant: 63
* Walk-forward window variations: ~10
* Honest buffer for "trials I tried but threw away": ~350

Total ≈ 130 traced + 350 buffer = **480**. Rounded up to **500**.

The number is **fixed in the codebase and in commit messages** — not
amended after seeing results. That's the whole point: the deflation
budget has to be set ex ante, otherwise we re-define "honest" to flatter
whatever we found.

## Headline — 7 assets × 3 strategies on Binance 1d

| Strategy   | Asset | N_grid | Concat OOS Sharpe | DSR @ 500 | Train DSR avg | Verdict       |
|-----------|-------|-------:|------------------:|----------:|--------------:|---------------|
| sma_cross  | BTC   |   19   |        +0.94      |   0.255   |     0.867     | UNCONFIRMED   |
| tsmom      | BTC   |    3   |        +1.04      |   0.332   |     0.886     | UNCONFIRMED   |
| pma_ratio  | BTC   |    3   |        +1.03      |   0.321   |     0.853     | UNCONFIRMED   |
| sma_cross  | ETH   |   19   |        +0.84      |   0.180   |     0.753     | UNCONFIRMED   |
| tsmom      | ETH   |    3   |        +0.91      |   0.226   |     0.863     | UNCONFIRMED   |
| pma_ratio  | ETH   |    3   |        +1.15      |   0.438   |     0.863     | UNCONFIRMED   |
| sma_cross  | BNB   |   19   |        +1.09      |   0.382   |     0.845     | UNCONFIRMED   |
| tsmom      | BNB   |    3   |        +0.75      |   0.120   |     0.831     | UNCONFIRMED   |
| pma_ratio  | BNB   |    3   |      **+1.16**    | **0.449** |     0.903     | UNCONFIRMED   |
| sma_cross  | SOL   |   19   |        +0.55      |   0.024   |     0.880     | UNCONFIRMED   |
| tsmom      | SOL   |    3   |        +0.89      |   0.088   |     0.927     | UNCONFIRMED   |
| pma_ratio  | SOL   |    3   |        +0.76      |   0.053   |     0.955     | UNCONFIRMED   |
| sma_cross  | ADA   |   19   |        +1.10      |   0.366   |     0.694     | UNCONFIRMED   |
| tsmom      | ADA   |    3   |        +1.00      |   0.278   |     0.773     | UNCONFIRMED   |
| pma_ratio  | ADA   |    3   |        +1.04      |   0.306   |     0.816     | UNCONFIRMED   |
| sma_cross  | XRP   |   19   |        +0.56      |   0.043   |     0.528     | UNCONFIRMED   |
| tsmom      | XRP   |    3   |        +0.38      |   0.014   |     0.623     | UNCONFIRMED   |
| pma_ratio  | XRP   |    3   |        +0.98      |   0.242   |     0.747     | UNCONFIRMED   |
| sma_cross  | DOGE  |   19   |        +0.10      |   0.002   |     0.742     | UNCONFIRMED   |
| tsmom      | DOGE  |    3   |        +0.41      |   0.014   |     0.812     | UNCONFIRMED   |
| pma_ratio  | DOGE  |    3   |        +0.49      |   0.020   |     0.915     | UNCONFIRMED   |

> **0 of 21 (strategy, asset) combinations survive DSR > 0.5 at the
> project-wide N=500.** Zero of 21 survive at the conventional 0.95
> threshold. The strongest single result — `pma_ratio` on BNB with
> concat OOS Sharpe +1.16 — sits at DSR 0.449, just below the
> "marginal" line.

This is the honest answer. Nothing in this repo is statistically
distinguishable from "lucky parameter pick" once you account for the
breadth of the search.

## Sensitivity to `num_trials`

Same observed Sharpe, varying selection-bias correction. `tsmom` on
BTC, concat OOS Sharpe = +1.04:

| `num_trials` | DSR     | Visualization                          |
|-------------:|---------|----------------------------------------|
|        1     | 0.997   | `#######################################`|
|        3     | 0.966   | `######################################` |
|       10     | 0.861   | `##################################`     |
|       30     | 0.717   | `############################`           |
|      100     | 0.541   | `#####################`                  |
|      300     | 0.392   | `###############`                        |
|      500 ←   | 0.332   | `#############`                          |
|     1000     | 0.260   | `##########`                             |
|     3000     | 0.170   | `######`                                 |
|    10000     | 0.102   | `####`                                   |

Read this as **"how confident should I be in this Sharpe given the
size of my mental search budget?"** A single strategy I ran once has
DSR 0.997. The same strategy if my decisions touched 500 prior trials
has DSR 0.332. The Sharpe didn't change — my prior touched data did.

## Per-fold train DSR — what the diagnostic reveals

`sma_cross` on BTC, 19-variant grid per fold:

| Fold | Test window         | Selected     | Train SR | Train DSR | Test SR  |
|------|---------------------|--------------|----------|-----------|----------|
| 0    | 2020-01..2020-06   | sma_30_100   |  +1.19   |   0.815   |  -0.63   |
| 1    | 2020-07..2020-12   | sma_30_50    |  +1.16   |   0.732   |  +3.66   |
| 2    | 2021-01..2021-06   | sma_30_50    |  +2.11   | **0.978** |  +2.16   |
| 3    | 2021-07..2021-12   | sma_30_50    |  +1.71   |   0.943   |  +0.69   |
| 4    | 2022-01..2022-06   | sma_20_50    |  +1.99   | **0.965** |  -1.39   |
| 5    | 2022-07..2022-12   | sma_10_100   |  +1.87   | **0.987** |  -1.64   |
| 6    | 2023-01..2023-06   | sma_10_100   |  +0.74   |   0.762   |  +1.21   |
| 7    | 2023-07..2023-12   | sma_20_300   |  +0.69   | **0.587** |  +1.89   |
| 8    | 2024-01..2024-06   | sma_30_300   |  +1.32   |   0.775   |  +1.72   |
| 9    | 2024-07..2024-12   | sma_30_300   |  +1.52   |   0.924   |  +1.79   |
| 10   | 2025-01..2025-06   | sma_30_300   |  +1.77   | **0.981** |  +0.82   |
| 11   | 2025-07..2025-12   | sma_20_300   |  +1.54   |   0.937   |  -0.82   |
| 12   | 2026-01..2026-05   | sma_50_100   |  +1.16   |   0.887   |  -1.23   |

Observations:

* **Train DSR does not predict OOS.** Folds 4 and 5 have train DSR
  0.965 and 0.987 — the selected variant clearly dominated its fold's
  grid. Their OOS Sharpe: −1.39 and −1.64. Crash.
* **Low train DSR doesn't predict failure either.** Fold 7 has the
  lowest train DSR (0.587, marginal). The selected variant gave
  +1.89 on test.
* The diagnostic value is *within-fold selection robustness*, not OOS
  prediction. A fold with train DSR < 0.5 is one where the picked
  variant was effectively a coin flip against its peers; OOS noise
  dominates.

So variant 1 catches *within-fold* selection bias, variant 2 catches
*project-wide* selection bias. They answer different questions; both
are useful.

## What this means for the project

1. **The IS / OOS shrinkage we found in `walk_forward_priority5.md`
   was not a fluke.** Variant 2 DSR adds a third layer of compression:
   in-sample Sharpe 1.0+ → mean fold OOS Sharpe ~0.5-0.7 → DSR-deflated
   confidence < 0.5. The "honest" Sharpe in the project's voice is
   not 1.0 or even 0.7. It's "we can't rule out random walk".
2. **`pma_ratio` is the closest to surviving** (DSR 0.24-0.45 across
   assets), consistent with the previous step's finding that PMA had
   the highest mean OOS Sharpe.
3. **Single best (strategy, asset) — `pma_ratio` on BNB** at DSR
   0.449 — is still UNCONFIRMED at the marginal threshold. If we
   really wanted to push it, we'd need either (a) more data (a second
   regime arc post-2026, longer history) or (b) a smaller project-
   wide N (i.e., less experimentation budget consumed before this
   one). Both are out of our control today.

## What we should NOT do

* **Lower `PROJECT_NUM_TRIALS` to make DSR look better.** That's
  retroactive p-hacking on the deflation metric itself.
* **Treat any single Sharpe number as "the result"** without
  surrounding it with the IS / OOS / DSR triple.
* **Quote full-window Sharpes** without the "in-sample only" qualifier.
  The full-window `tsmom_short` on BTC at 1.20 looked great in
  `strategy_comparison.md`; the OOS DSR is 0.33. The pre-deflation
  number is technically true and practically misleading.

## What we SHOULD do

* **Always carry forward the DSR alongside the Sharpe** in any
  consumer-facing output (compare report, walk-forward CSVs, etc.).
* **If we eventually claim a strategy works in live trading**, the
  bar should be DSR > 0.5 on paper-trading data plus DSR > 0.5 on
  a hold-out validation period not used during development. We have
  neither in this repo yet.
* **Keep `PROJECT_NUM_TRIALS = 500` fixed** until a major refactor.
  When the project's experimentation budget grows substantially, the
  number can be bumped, but only with explicit acknowledgment in the
  commit history.

## Sanity checks

If anyone tweaks the DSR plumbing, these must still hold:

1. `train_dsr` column exists and every value is in `[0, 1]`.
   (Tested.)
2. Passing `return_oos_returns=True` yields a `(detail, oos_list)`
   tuple; default still returns just `detail`. (Tested.)
3. `aggregate_walk_forward(detail, oos_returns=oos, num_trials=N)`
   produces `concatenated_oos_dsr` monotonically decreasing in `N`.
   (Tested with N=10 vs N=10000.)
4. `PROJECT_NUM_TRIALS == 500` — pinned. (Tested.)

## Reproducing

```bash
python -c "
import pandas as pd
from trade_lab.backtest import (
    run_strategy_walk_forward, aggregate_walk_forward,
    build_tsmom_grid, PROJECT_NUM_TRIALS,
)
candles = pd.read_parquet('data/binance_BTC_USDT_1d.parquet')
detail, oos = run_strategy_walk_forward(
    candles, build_tsmom_grid(), return_oos_returns=True,
)
s = aggregate_walk_forward(detail, oos_returns=oos, num_trials=PROJECT_NUM_TRIALS)
print('concat OOS SR:', round(s['concatenated_oos_sharpe'], 2))
print('DSR @ 500:    ', round(s['concatenated_oos_dsr'], 3))
print('mean train DSR:', round(s['mean_per_fold_train_dsr'], 3))
"
```

Full 7-asset matrix: `outputs/wf_dsr_7assets.csv` (21 rows).

## Next on the queue

Per the user's priority list — item #5: **verify the B&H benchmark is
charged the same fees and slippage as the strategies.** This is a
small but real bug-check; current benchmarks zero out fees in some
paths (`compare.py`'s buy-and-hold). The reader needs to know the
~3% return advantage strategies have over B&H wasn't paid for in
B&H costs that the strategies were paying.
