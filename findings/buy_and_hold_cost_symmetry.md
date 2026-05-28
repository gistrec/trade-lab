# Finding — Asymmetric B&H cost model gave it a quiet ~0.15% head-start

**Status:** bug found, fixed, impact quantified, no verdicts flipped
on existing data — but the principle now holds and future comparisons
won't be biased the same way.

## Finding

Four code paths in this repo (`engine.run_backtest`,
`compare._buy_and_hold_metrics`, `yearly._yearly_rows_for_buy_and_hold`,
`yearly._yearly_row_for_strategy`, `walk_forward_v2._buy_and_hold_on_window`)
computed buy-and-hold equity as

```python
equity = initial_capital * (close / close.iloc[0])
total_return = close.iloc[-1] / close.iloc[0] - 1
```

while strategies paid one entry's worth of `(fee_rate + slippage_rate)`
on bar 1 for an equivalent always-long position. **The B&H benchmark
silently received a free ~0.15% head-start over every strategy entering
on the first bar.**

The asymmetry is small per-bar but compounds: on the full BTC window
2018-2026 the bare close ratio is 5.56x, so a +0.15% reduction at
entry propagates to a -0.83pp reduction in total return. On BNB
(74x over the window) the propagation is -11.5pp.

## What the fix does

Single helper in `engine.py`:

```python
def buy_and_hold_with_costs(close, initial_capital, fee_rate, slippage_rate):
    """B&H equity and return after one entry round of fee + slippage."""
    if close.empty or len(close) < 2:
        return close.copy() * 0.0, 0.0
    entry_cost = fee_rate + slippage_rate
    effective_capital = initial_capital * (1 - entry_cost)
    equity = effective_capital * (close / close.iloc[0])
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    return equity, total_return
```

Applied uniformly in all four paths. The convention:

* B&H pays the same `fee + slippage` on bar 1 as a strategy entering an
  equal-sized long.
* B&H does NOT pay an exit fee at the end of the window — same
  convention the engine uses for strategies finishing with an open
  position (mark-to-market, no closing turnover charge).
* Tests that previously asserted `total_fees == 0` for B&H now assert
  `total_fees == initial_capital * fee_rate` and similar for slippage.

## Quantified impact — B&H return shift across our datasets

### Per-year on BTC

| Year | Close ratio | Pre-cost return | Post-cost return | Shift |
|------|------------:|----------------:|-----------------:|------:|
| 2018 |    0.277    |     -72.33%     |     -72.37%      | -0.04 |
| 2019 |    1.895    |     +89.49%     |     +89.21%      | -0.28 |
| 2020 |    4.017    |    +301.67%     |    +301.07%      | -0.60 |
| 2021 |    1.576    |     +57.57%     |     +57.33%      | -0.24 |
| 2022 |    0.347    |     -65.34%     |     -65.39%      | -0.05 |
| 2023 |    2.545    |    +154.46%     |    +154.08%      | -0.38 |
| 2024 |    2.118    |    +111.81%     |    +111.49%      | -0.32 |
| 2025 |    0.927    |      -7.34%     |      -7.48%      | -0.14 |
| 2026 |    0.838    |     -16.20%     |     -16.32%      | -0.13 |

Magnitude scales with the close ratio (bull years move more than bear
years). At 0.15% entry cost the propagated shift is `(close_ratio - 1) *
-0.0015`. For 2020 with 4x close ratio, that's 3 * -0.0015 = -0.45pp,
in the ballpark of the observed -0.60pp (additional pennies come from
compounding).

### Full-window per asset

| Asset | Pre-cost return | Post-cost return | Shift |
|-------|----------------:|-----------------:|------:|
| BTC   |     +456.42%    |     +455.59%     | -0.83 |
| ETH   |     +168.27%    |     +167.87%     | -0.40 |
| BNB   |    +7569.82%   |    +7558.32%     |**-11.50**|
| SOL   |    +2402.05%    |    +2398.29%     | -3.75 |

The BNB shift is the largest because BNB grew the most. **In absolute
PnL terms the asymmetry on BNB was 0.15% × initial capital, but
quoted as a percentage return it scales with whatever you grew to.**
This is why the bug stayed invisible: the relative comparison
("strategy vs B&H by X percentage points") was within 0.15% only on
unusually-flat years.

## Verdict-flip count on current strategies

I scanned every (strategy, period, asset) cell in
`run_comparison_report` and every (strategy, year) cell in
`run_yearly_validation` for cases where the strategy beat B&H by less
than ~0.5pp (a margin small enough to flip from OUTPERFORMS_BH to
UNDERPERFORMS_BH given the new ~0.4-1pp B&H reduction).

**Result: 0 cells in the narrow-margin band.** Every strategy verdict
in the current `outputs/compare_research_claude.*` and
`outputs/yearly_*` results survives the symmetric-cost fix.

That's not because the bug didn't exist — it's because **the
strategies' gap to B&H was always either much larger or much smaller
than the 0.15% cost slice**. Either we were comfortably above or
comfortably below the line.

## Why no flips ≠ "bug didn't matter"

Two reasons to fix it anyway:

1. **Future strategies on flatter assets WILL straddle the line.**
   If someone evaluates a strategy on a stable-rate-of-return ETF
   or a low-vol crypto pair, the 0.15% asymmetry can dominate
   genuine differences.
2. **The "OUTPERFORMS_BH" badge means something specific.** Awarding
   it when one side paid no fees is a category error — the same
   error you accuse anyone of when they show a Sharpe with zero
   transaction costs.

The fix takes us from "lucky no flips" to "the test is well-defined".

## Implication for future work

* **Don't restore the pre-cost B&H curve in a follow-up "feature".**
  If someone needs the academic pre-cost view, they can compute it
  with `fee_rate=0, slippage_rate=0` explicitly. The default is the
  honest symmetric one.
* **Add the cost params to any new B&H computation** added to the
  repo. The helper `buy_and_hold_with_costs` is the canonical entry
  point.

## Caveat

* **Only entry cost is charged.** A strategy that voluntarily exits
  before the window's end pays an exit cost too; B&H by definition
  doesn't. This is consistent with the engine's open-position
  convention but means B&H still gets a small (one-round) cost
  advantage relative to a strategy that round-tripped 10 times.
  That's a *correct* asymmetry — the strategy chose to trade more,
  so it pays more.
* **No bid/ask model.** Our slippage is a flat rate; real-world
  slippage is non-linear in size and time-varying. The 0.0005 here
  is a Binance-major rough mid; thin pairs at retail size would
  pay much more.
* **No funding-rate effect.** Spot only.

## Sanity check

```python
from trade_lab.backtest.engine import buy_and_hold_with_costs
import pandas as pd, numpy as np

# Synthetic up-and-down series.
close = pd.Series([100, 110, 121, 110, 100], dtype=float)

equity, total_return = buy_and_hold_with_costs(
    close, initial_capital=10_000.0,
    fee_rate=0.001, slippage_rate=0.0005,
)

# Expected: pay 1.5% × 0.0015 = 0.15% on $10k → start with $9985.
assert np.isclose(equity.iloc[0], 9985.0)
# End at close[-1] / close[0] × $9985 = 1.0 × $9985 = $9985.
assert np.isclose(equity.iloc[-1], 9985.0)
# Total return: $9985 / $10000 - 1 = -0.15%.
assert np.isclose(total_return, -0.0015)
```

If anyone reverts the helper without updating callers, that test
catches the regression.
