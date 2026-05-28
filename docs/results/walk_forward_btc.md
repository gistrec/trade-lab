# Walk-forward multi-strategy validation — BTC/USDT 1d

Output of `trade-lab walk-forward` on BTC/USDT 1d, with both
`sma_cross` and `regime_sma_cross` candidates considered each window.

Train window: 2 years; test window: 1 year; step: 1 year. On each
train slice every candidate `(fast, slow [, regime])` combination is
swept and the winner by **train total return** is evaluated on the
following test slice.

## Per-window result (objective = `total_return`)

| train period           | test period           | selected_strategy   | params           | train_ret | test_ret | test_BH  | verdict               |
|------------------------|-----------------------|---------------------|------------------|-----------|----------|----------|-----------------------|
| 2018-01-01 - 2019-12-31 | 2020-01-01 - 2020-12-31 | sma_cross           | f=30/s=100        | +125.17%  | +237.34% | +301.67% | LOWER_RETURN_LOWER_DD |
| 2019-01-01 - 2020-12-31 | 2021-01-01 - 2021-12-31 | sma_cross           | f=30/s=50         | +644.85%  | +10.87%  | +57.57%  | LOWER_RETURN_LOWER_DD |
| 2020-01-01 - 2021-12-31 | 2022-01-01 - 2022-12-31 | sma_cross           | f=5/s=100         | +562.11%  | -16.18%  | -65.34%  | OUTPERFORMS_BH        |
| 2021-01-01 - 2022-12-31 | 2023-01-01 - 2023-12-31 | sma_cross           | f=5/s=150         | +10.96%   | +31.18%  | +154.46% | LOWER_RETURN_LOWER_DD |
| 2022-01-01 - 2023-12-31 | 2024-01-01 - 2024-12-31 | regime_sma_cross    | f=5/s=50/r=100    | +85.42%   | +15.44%  | +111.81% | UNDERPERFORMS_BH      |
| 2023-01-01 - 2024-12-31 | 2025-01-01 - 2025-12-31 | regime_sma_cross    | f=30/s=100/r=300  | +175.57%  | +0.00%   | -7.34%   | OUTPERFORMS_BH        |
| 2024-01-01 - 2025-12-31 | 2026-01-01 - 2026-05-27 | sma_cross           | f=30/s=50         | +54.77%   | +5.36%   | -16.20%  | OUTPERFORMS_BH        |

## Two takeaways

1. **Parameter instability.** The "best" pair changes every window:
   `30/100` → `30/50` → `5/100` → `5/150` → `5/50/r=100` → `30/100/r=300` → `30/50`.
   If the SMA crossover had a stable edge on this market, the optimum
   on train would generalize. It doesn't — train selections shift
   with the regime, which is the classic signature of overfitting on
   a sweep.
2. **The full-history sweep is misleading.** A one-pass
   `trade-lab sweep` shows `30/100` with +1419% over the whole 2018-26
   window. Walk-forward shows the strategy actually delivers far less
   than that out-of-sample: median window test return is in the
   single-to-low-double digits.

The strategy still has a real edge in bear regimes — the 2022 column
is the only `OUTPERFORMS_BH` with a positive *relative* return when
B&H lost 65%. That's the signal worth keeping; the "huge total
return" number from the full-history sweep is mostly a sweep artefact.

Switching `--objective return_div_drawdown` swaps two of the
selections to the regime variant (lower DD on train wins the ratio
even when raw return is similar), and confirms the same shape: the
edge lives in bear / sideways regimes, not in bulls.

## Reproduce

```bash
trade-lab walk-forward --symbol BTC/USDT --timeframe 1d \
    --strategies sma_cross,regime_sma_cross \
    --fast-periods 5,10,20,30 \
    --slow-periods 50,100,150,200 \
    --regime-periods 100,200,300 \
    --objective total_return \
    --train-years 2 --test-years 1 --step-years 1 \
    --output-csv outputs/walk_forward.csv
```
