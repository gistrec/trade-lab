# `regime_only` — pure regime filter, no crossover

Long iff `close > SMA(regime_period)`. Flat otherwise. No crossover,
no entry timing besides the regime check. The simplest possible
"trend-following but not in downtrends" rule.

## Rules

- `regime = close.rolling(regime_period).mean()` (default 200)
- Signal = `1` iff `close > regime`, else `0`.
- Warm-up bars (regime still NaN) → signal `0`.

Causal at every bar: changing any future close does not change a
signal at or before that bar (see
`tests/test_regime_only.py::test_signal_at_bar_n_does_not_use_close_after_n`).
The engine then shifts the signal by one bar.

## How to run

```bash
trade-lab backtest --strategy regime_only --symbol BTC/USDT --timeframe 1d \
    --param regime_period=200
```

## Where it shows up in cross-strategy reports

The default `200` and a `300` variant are bundled in the yearly
fixed-strategy validation. On 9 years of BTC/USDT 1d, `regime_only_300`
is one of the strongest by B&H-comparison count — see
[docs/results/yearly_btc.md](../results/yearly_btc.md).

On the 4-asset panel, both `regime_only_200` and `regime_only_300`
reduce drawdown vs buy-and-hold in 26 out of 34 asset-years, but at
a meaningful cost to raw return (especially on SOL). See
[docs/results/multi_asset.md](../results/multi_asset.md).

## Limitations

- No timing — exposure is binary (`0` or `1` × position size). Don't
  expect partial sizing.
- The regime SMA alone is a slow signal: missed turns at the start of
  bull runs, late exits from bear runs.
- Choice of `regime_period` matters a lot. `200` reacts faster but
  is whipsaw-prone; `300` is steadier but slower to re-enter.
