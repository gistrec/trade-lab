# `sma_cross` — Simple Moving Average crossover

Long when `SMA(fast) > SMA(slow)`; flat otherwise. Long-only. The
classic trend-following baseline.

## Rules

- `fast = close.rolling(fast_period).mean()` (default 20)
- `slow = close.rolling(slow_period).mean()` (default 100)
- Signal = `1` when `fast > slow`, else `0`.
- Constructor requires `fast_period < slow_period`. Warm-up bars (any
  SMA still NaN) → signal `0`.

Signals are produced from data available at bar `N` and the engine
shifts them by one bar, so a flip at the close of `N` only affects
bar `N+1` onward.

## How to run

```bash
trade-lab backtest --strategy sma_cross --symbol BTC/USDT --timeframe 1d \
    --param fast_period=20 --param slow_period=100
```

## Useful variants

### Faster, on 4h

The hourly default is noisy; 4h smooths out a lot of intrabar wiggle
and usually trades a lot less.

```bash
trade-lab fetch --symbol BTC/USDT --timeframe 4h --since 2023-01-01
trade-lab backtest --strategy sma_cross --symbol BTC/USDT --timeframe 4h \
    --param fast_period=20 --param slow_period=100
```

### Golden cross (50 / 200, daily)

The textbook trend filter. On crypto it triggers rarely — on daily
bars it might trigger once a year.

```bash
trade-lab backtest --strategy sma_cross --symbol BTC/USDT --timeframe 1d \
    --param fast_period=50 --param slow_period=200
```

Things to look at:
- `Avg holding period` (should be hundreds of bars on 1d 50/200),
- `Number of trades` (single digits per year),
- `Verdict:` vs the faster 20/100 pair on the same window.

## Where it shows up in cross-strategy reports

- 9-year BTC/USDT 1d yearly: see [docs/results/yearly_btc.md](../results/yearly_btc.md)
- 4-asset (BTC/ETH/BNB/SOL) summary: see [docs/results/multi_asset.md](../results/multi_asset.md)
- Walk-forward selection across years: see
  [docs/results/walk_forward_btc.md](../results/walk_forward_btc.md)

## Limitations

- Long-only. The strategy never goes short.
- Pure crossover — no regime filter, no risk overlay. Use
  `regime_sma_cross` or `donchian_trend` for those.
- Vulnerable to whipsaw in sideways markets; the bigger / slower the
  SMAs, the fewer false signals but the more lag at turning points.
