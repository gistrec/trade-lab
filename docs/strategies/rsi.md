# `rsi` — RSI mean-reversion

Mean-reversion strategy: long when RSI dips below `buy_threshold`,
exit when RSI rises above `sell_threshold`. Long-only.

## Rules

- Compute RSI on `close` using `period` (default 14).
- Enter long when `rsi < buy_threshold` (default 30).
- Exit when `rsi > sell_threshold` (default 70).
- Between thresholds, hold the previous state.

Mean-reversion is fundamentally a different bet from the trend
strategies in this project. It's bundled for contrast, not because it
has a strong out-of-sample track record on crypto.

## How to run

```bash
trade-lab backtest --strategy rsi --symbol BTC/USDT --timeframe 1d \
    --param period=14 --param buy_threshold=30 --param sell_threshold=70
```

## Limitations

- Mean-reversion on a strongly-trending asset (crypto) is the wrong
  bet most of the time. Use with skepticism.
- No regime filter — the strategy will keep buying dips during a real
  bear market.
- Parameters here are textbook defaults, not tuned to anything in this
  repo's history.
