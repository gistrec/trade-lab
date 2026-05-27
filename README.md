# trade-lab

A small research framework for fetching crypto OHLCV data, defining trading
strategies, and backtesting them with realistic fees and slippage.

This is an MVP focused on:

- Pulling historical candles from any [ccxt](https://github.com/ccxt/ccxt)-supported
  exchange.
- Storing candles locally as Parquet for fast iteration.
- Long-only, single-asset, vectorized backtests with no leverage.
- Pluggable strategies (SMA crossover and RSI mean reversion bundled).
- Performance metrics and an equity-curve plot saved under `outputs/`.

Paper / live trading is intentionally out of scope; the strategy and engine
interfaces are designed so an execution layer can be added later without
touching strategy code.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and tweak defaults if desired.

## Quick start

```bash
# 1. Fetch a year of BTC/USDT 1h candles from Binance.
trade-lab fetch --since 2024-01-01

# 2. Backtest SMA crossover (fast=20, slow=100) against those candles.
trade-lab backtest --strategy sma_cross
```

The first command writes `data/binance_BTC_USDT_1h.parquet`, the second loads
it, prints metrics, and saves the equity curve to
`outputs/sma_cross_BTC_USDT_1h.png`.

## Fetching candles

```bash
trade-lab fetch \
    --exchange binance \
    --symbol BTC/USDT \
    --timeframe 1h \
    --since 2024-01-01 \
    --until 2025-01-01
```

Defaults: `--exchange binance --symbol BTC/USDT --timeframe 1h`. Output goes to
`<TRADE_LAB_DATA_DIR>/<exchange>_<symbol>_<timeframe>.parquet`. The fetcher
paginates automatically, deduplicates timestamps, and validates that the
resulting frame has columns `open, high, low, close, volume` indexed by a UTC
`timestamp` (the schema the rest of the toolkit expects). Pass `--output PATH`
to write somewhere else.

## Running a backtest

```bash
trade-lab backtest \
    --strategy sma_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --initial-cash 10000 \
    --fee-rate 0.001 \
    --slippage 0.0005 \
    --param fast_period=20 \
    --param slow_period=100
```

You can also point at a Parquet file directly:

```bash
trade-lab backtest --strategy sma_cross --input data/binance_BTC_USDT_1h.parquet
```

Sample output:

```
Strategy:             sma_cross
Symbol/timeframe:     BTC/USDT 1h
Bars:                 400
Initial cash:         $10,000.00
Final equity:         $13,533.00
Total return:         35.33%
Buy & hold return:    50.26%
Max drawdown:         3.70%
Number of trades:     1
Win rate:             100.00%
Average trade:        35.33%
Fees paid:            $10.00
Plot saved to outputs/sma_cross_BTC_USDT_1h.png
```

The equity curve (with a drawdown panel underneath) is saved under `outputs/`
by default; pass `--save-plot PATH` to override, `--show-plot` to also display
it interactively, or `--no-plot` to skip plotting entirely.

## Reading the output metrics

| Metric                | What it means                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| **Initial cash**      | Starting equity. Fed by `--initial-cash` (env: `TRADE_LAB_INITIAL_CAPITAL`).                        |
| **Final equity**      | Equity at the last bar after fees and slippage.                                                     |
| **Total return**      | `(final_equity / initial_cash) - 1`, expressed as a percentage. Net of fees and slippage.           |
| **Buy & hold return** | Gross asset return over the period (`close[-1] / close[0] - 1`). Useful baseline.                   |
| **Max drawdown**      | Worst peak-to-trough decline of the equity curve, reported as a positive percentage.                |
| **Number of trades**  | Round-trip long trades. An open trade at the end of the window is closed at the last bar.           |
| **Win rate**          | Share of trades with a positive net P/L.                                                            |
| **Average trade**     | Mean net return per trade (compounded fees/slippage included).                                      |
| **Fees paid**         | Cumulative dollar fees paid across all rebalances. Slippage is *not* included here.                 |

The CLI sometimes saves an equity curve plot you can inspect alongside the
numbers — the underwater panel makes drawdown intuitive at a glance.

## Backtest assumptions

- Long-only, single asset, no leverage.
- Signals generated on bar `N` execute against bar `N+1` — signals are shifted
  by one bar before computing returns, so a strategy cannot peek at the close
  it just observed.
- Fees and slippage are charged on every change in exposure, proportional to
  the size of the change.
- `position_size` (default `1.0`) scales exposure: `0.5` means we deploy half
  of equity per long trade.

## Programmatic usage

```python
from pathlib import Path

from trade_lab.backtest.engine import run_backtest
from trade_lab.backtest.metrics import compute_metrics
from trade_lab.data.storage import load_candles
from trade_lab.strategies.sma_cross import SMACrossStrategy

candles = load_candles(Path("data"), "binance", "BTC/USDT", "1h")
result = run_backtest(
    candles=candles,
    strategy=SMACrossStrategy(fast_period=20, slow_period=100),
    initial_capital=10_000,
    fee_rate=0.001,
    slippage_rate=0.0005,
)
print(compute_metrics(result))
```

## Adding a new strategy

1. Create `src/trade_lab/strategies/my_strategy.py`.
2. Subclass `Strategy` and implement `generate_signals(self, candles)`. It
   must return a `0`/`1` target-position `pd.Series` aligned with `candles`.
3. Register the class in `STRATEGIES` in `src/trade_lab/cli.py` to make it
   selectable via `--strategy my_strategy`.

Minimal example:

```python
import pandas as pd

from trade_lab.strategies.base import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=candles.index, dtype=int)
```

The engine handles look-ahead protection, fees, slippage, and trade
extraction — your strategy only needs to express the target position.

## Project layout

```
src/trade_lab/
  config.py             Environment-driven configuration
  data/                 Candle fetching + Parquet storage
  strategies/           Strategy base class + bundled strategies
  backtest/             Engine, metrics, plotting
  risk/                 Position sizing helpers
  cli.py                argparse entry point
```

## Tests

```bash
pytest
```

Covers SMA signal generation, the backtest engine (look-ahead prevention,
fee/slippage handling, trade extraction, position sizing, total-fee
calculation, buy & hold), and metrics (returns, max drawdown across multiple
peaks).

## Roadmap

- Paper / live execution behind a `Broker` interface.
- Multi-asset portfolios and rebalancing.
- Short selling and leverage.
- Walk-forward and parameter optimization helpers.
