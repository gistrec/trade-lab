# `regime_sma_cross` — SMA crossover gated by a regime filter

Same `fast > slow` crossover as `sma_cross`, but the additional
condition `close > SMA(regime_period)` must also hold for a long
signal. The aim is to skip the false signals during bear regimes that
drag the plain crossover below buy-and-hold.

## Rules

- `fast = close.rolling(fast_period).mean()` (default 20)
- `slow = close.rolling(slow_period).mean()` (default 100)
- `regime = close.rolling(regime_period).mean()` (default 200)
- Signal = `1` iff `fast > slow` **AND** `close > regime`, else `0`.
- Constructor requires `fast < slow < regime`.

Signals at bar `N` use only data available at close of `N`. Engine
shifts by one bar.

## How to run

```bash
trade-lab backtest --strategy regime_sma_cross --symbol BTC/USDT --timeframe 1d \
    --param fast_period=20 --param slow_period=100 --param regime_period=200
```

Expect fewer trades, much smaller drawdown during downtrends, and
often a `LOWER_RETURN_LOWER_DD` verdict on bull-only windows — you
give up some upside for the regime guard. Compare against the plain
`sma_cross` with the same fast / slow on the same window.

## Where it shows up in cross-strategy reports

The default `(20, 100, 200)` pair is included in
- the yearly fixed-strategy validation:
  [docs/results/yearly_btc.md](../results/yearly_btc.md)
- the multi-asset comparison:
  [docs/results/multi_asset.md](../results/multi_asset.md)
- and is a candidate in the walk-forward selector:
  [docs/results/walk_forward_btc.md](../results/walk_forward_btc.md)

On the 4-asset (BTC / ETH / BNB / SOL) panel, `regime_sma_cross` has
the best *average worst year* across assets (-14% vs B&H's -76%) —
the regime filter sits the strategy out during the biggest bear
windows on each asset.

## Limitations

- Same long-only + warm-up flatness as `sma_cross`.
- In strong bull markets the regime filter is permissive (price is
  always above the 200-day SMA), so the strategy is essentially
  `sma_cross` with the same drag. The filter pays back during bears.
- 50 / 200 / regime variants aren't bundled; if you want them, pass
  the corresponding parameters to the constructor.
