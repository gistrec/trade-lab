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

The optional Streamlit dashboard has its own extras group:

```bash
pip install -e ".[dashboard]"
```

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

Restrict the backtest to a sub-window with `--start-date` and `--end-date`
(both inclusive at the day level — `--end-date 2024-06-30` keeps every bar
through `2024-06-30 23:59`). This is how you carve train vs test splits
without refetching:

```bash
# Train window
trade-lab backtest --strategy sma_cross \
    --start-date 2023-01-01 --end-date 2023-12-31 \
    --param fast_period=20 --param slow_period=100

# Test window
trade-lab backtest --strategy sma_cross \
    --start-date 2024-01-01 --end-date 2024-06-30 \
    --param fast_period=20 --param slow_period=100
```

The actual tested window is echoed in the report's `Period:` line along with
the bar count, so you can confirm the slice landed where you expected.

### Yearly validation (fixed parameters)

`trade-lab yearly` evaluates a *fixed* set of strategies on each
calendar year and lays out a comparison alongside buy-and-hold. No
parameter sweep, no tuning per year. Bundled strategies:
`buy_and_hold`, `regime_only_200`, `regime_only_300`,
`sma_cross_20_100`, `regime_sma_cross_20_100_200`, `golden_cross_50_200`.

```bash
trade-lab yearly --symbol BTC/USDT --timeframe 1d --output-csv outputs/yearly.csv
```

Real 9-year aggregate on BTC/USDT 1d — full table and commentary in
[docs/results/yearly_btc.md](docs/results/yearly_btc.md).

### Multi-asset validation

`trade-lab multi-asset` runs the same fixed-parameter yearly validation
across several symbols. The point is to test whether behaviour you saw
on BTC generalizes — or whether you got lucky on one ticker.

```bash
trade-lab multi-asset --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT \
    --timeframe 1d --output-csv outputs/multi_asset.csv
```

Three CSVs are produced: per-(asset, year, strategy) detail,
per-(asset, strategy) aggregate, and an across-asset summary. Real
4-asset / 34-year results in
[docs/results/multi_asset.md](docs/results/multi_asset.md).

### Walk-forward validation

`trade-lab walk-forward` cuts the dataset into rolling train / test
windows (default: 2-year train, 1-year test, step 1 year), runs a
sweep on each train slice, picks the best `(strategy, params)`
candidate from train only, and evaluates that pair on the immediately
following test slice.

```bash
trade-lab walk-forward --symbol BTC/USDT --timeframe 1d \
    --strategies sma_cross,regime_sma_cross \
    --fast-periods 5,10,20,30 --slow-periods 50,100,150,200 \
    --regime-periods 100,200,300 \
    --objective total_return \
    --train-years 2 --test-years 1 --step-years 1 \
    --output-csv outputs/walk_forward.csv
```

`--strategies` picks which SMA-family candidates enter the train sweep.
`--objective` is `total_return` (default) or `return_div_drawdown` for
risk-adjusted picks. Real 7-window BTC/USDT 1d run, plus what the
parameter instability tells us, in
[docs/results/walk_forward_btc.md](docs/results/walk_forward_btc.md).

### Parameter sweeps

`trade-lab sweep` grid-searches SMA crossover parameters across a date range
and writes one row per combination to CSV. Invalid combos
(`fast_period >= slow_period`) are skipped automatically.

```bash
trade-lab sweep \
    --strategy sma_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --fast-periods 5,10,20,30 \
    --slow-periods 50,100,150,200 \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --output-csv outputs/sweep.csv
```

The console prints a sorted table (highest total return on top) and saves
the full result with raw numeric values to the CSV. This is **research
only** — picking "the best" parameters from a sweep is a notorious way to
overfit. Use it to compare neighborhoods and to look for stable regions,
not to choose live settings.

## Streamlit dashboard

A local-only dashboard for poking at a single backtest visually:

```bash
pip install -e ".[dashboard]"
streamlit run src/trade_lab/dashboard/app.py
```

Sidebar controls: candle file path, strategy + parameters
(`sma_cross` or `rsi`), initial cash / fee / slippage / position size,
and an optional date range.

The main panel has metric cards at the top (with a "vs B&H" delta on
total return), context-aware warnings when the strategy underperforms
buy & hold, has a worse drawdown than buy & hold, or trades too often,
followed by five tabs:

1. **Overview** — run summary + metrics CSV download.
2. **Price & Trades** — Plotly price chart with buy / sell markers on
   execution candles.
3. **Equity** — strategy equity vs buy-and-hold + equity CSV download.
4. **Drawdown** — underwater chart.
5. **Trades** — trade list with a CSV download.

The dashboard reuses the regular backtest engine, strategies, and metrics
— there is no duplicate logic. Candles are cached by file path; the
backtest itself runs fresh on every interaction since parameters change
too often to memoize safely. The dashboard is local-only by design: no
auth, no execution, no live trading.

Typical workflow:

```bash
trade-lab fetch --since 2024-01-01            # pull candles once
streamlit run src/trade_lab/dashboard/app.py  # explore parameters
```

Sample output:

```
Strategy:             sma_cross
Symbol/timeframe:     BTC/USDT 1h
Period:               2024-01-01 00:00 to 2024-01-15 23:00
Bars:                 400
Initial cash:         $10,000.00

Strategy
  Final equity:       $13,533.00
  Total return:       35.33%
  Max drawdown:       3.70%
  Number of trades:   1
  Win rate:           100.00%
  Average trade:      35.33%
  Fees paid:          $10.00

Buy & hold
  Final equity:       $15,026.00
  Total return:       50.26%
  Max drawdown:       4.81%
Plot saved to outputs/sma_cross_BTC_USDT_1h.png
```

The plot shows the strategy equity curve and the buy-and-hold equity curve on
the same axes, with a strategy-only drawdown panel underneath. It is saved
under `outputs/` by default; pass `--save-plot PATH` to override,
`--show-plot` to also display it interactively, or `--no-plot` to skip
plotting entirely.

### Visually verifying entries / exits

Pass `--show-trades` to add a price panel above the equity panel with buy and
sell markers:

```bash
trade-lab backtest --strategy sma_cross --show-trades
```

Markers are placed on the **execution candle** (the bar where the engine
actually holds the position), not the signal candle. This matters for
look-ahead checks: signal at bar `N` is shifted into a position at bar `N+1`,
so the green ▲ should sit on the bar after the cross — never on the bar that
produced the cross itself. If you ever see a marker on a candle whose close
*caused* the signal, that's the smell of look-ahead bias.

Note: `Trade.entry_time` in the engine's result records the signal bar (the
close where the decision was made); the marker derived from
`execution_bars(positions)` records the bar after — they're intentionally
one bar apart.

### Exporting trades to CSV

Pass `--trades-csv PATH` to dump the trade list:

```bash
trade-lab backtest --strategy sma_cross --trades-csv outputs/trades.csv
```

The CSV has one row per completed trade with these columns:

| Column             | Meaning                                                                                              |
|--------------------|------------------------------------------------------------------------------------------------------|
| `entry_time`       | Timestamp of the execution candle (one bar after the signal candle).                                 |
| `entry_price`      | `close[entry_bar] * (1 + slippage_rate)` — the slippage-adjusted price the strategy effectively paid.|
| `exit_time`        | Timestamp of the execution candle for the exit.                                                      |
| `exit_price`       | `close[exit_bar] * (1 - slippage_rate)` — what the strategy effectively received on the sell.        |
| `gross_return_pct` | Raw close-to-close return between execution bars (no fees, no slippage).                             |
| `net_return_pct`   | Net return from the strategy equity curve. Includes fees AND slippage.                               |
| `fees_paid`        | Dollar exchange fees for the round trip. Excludes slippage (it's already in entry/exit prices).      |
| `holding_period`   | Number of bars the position was actually held.                                                       |
| `pnl`              | Dollar P&L (`equity_at_exit - equity_just_before_entry`).                                            |

Sample row from a real run:

```
entry_time,entry_price,exit_time,exit_price,gross_return_pct,net_return_pct,fees_paid,holding_period,pnl
2024-01-02 06:00:00+00:00,97.744,2024-01-03 04:00:00+00:00,100.864,0.03295,0.01211,20.14,22,121.12
```

Open positions at the end of the window are **excluded** from the CSV (the
CLI prints "excluded N open position(s)" so you know). For programmatic use
that needs them, call `trades_to_dataframe(result, candles, include_open=True)`
— that adds an `is_open` column.

## Reading the output metrics

| Metric                | What it means                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| **Initial cash**      | Starting equity. Fed by `--initial-cash` (env: `TRADE_LAB_INITIAL_CAPITAL`).                        |
| **Final equity**      | Equity at the last bar after fees and slippage.                                                     |
| **Total return**      | `(final_equity / initial_cash) - 1`, expressed as a percentage. Net of fees and slippage.           |
| **Max drawdown**      | Worst peak-to-trough decline of the equity curve, reported as a positive percentage.                |
| **Number of trades**  | Round-trip long trades. An open trade at the end of the window is closed at the last bar.           |
| **Win rate**          | Share of trades with a positive net P/L.                                                            |
| **Average trade**     | Mean net return per trade (compounded fees/slippage included).                                      |
| **Fees paid**         | Cumulative dollar fees paid across all rebalances. Slippage is *not* included here.                 |

The **Buy & hold** block reports the same starting cash parked in the asset at
the first bar and held to the last bar (no fees). It produces three figures —
*final equity*, *total return*, and *max drawdown* — so you can see whether
the strategy actually beat just holding the asset, and how much of the peak
equity it gave back relative to a passive position.

The CLI also saves an equity curve plot under `outputs/` showing strategy and
buy-and-hold equity on the same axes, with a strategy-only underwater panel
underneath — useful for eyeballing whether outperformance is real or whether
the strategy is just along for the ride.

## Backtest assumptions

- Long-only, single asset, no leverage.
- Signals generated on bar `N` execute against bar `N+1` — signals are shifted
  by one bar before computing returns, so a strategy cannot peek at the close
  it just observed.
- Fees and slippage are charged on every change in exposure, proportional to
  the size of the change.
- `position_size` (default `1.0`) scales exposure: `0.5` means we deploy half
  of equity per long trade.

## Trading costs

Every backtest carries two explicit per-side costs:

- **Fee** (`--fee-rate`, default `0.001` = 0.1%) — what the exchange charges.
  Charged on every buy and every sell.
- **Slippage** (`--slippage`, default `0.0005` = 0.05%) — the gap between the
  close price and the price you actually trade at. Buys fill at
  `close * (1 + slippage_rate)` (you pay more), sells at
  `close * (1 - slippage_rate)` (you receive less).

The report makes the round trip explicit:

```
Cost model
  Buy cost (fee + slip):  0.15%
  Sell cost (fee + slip): 0.15%
  Round-trip cost:        0.30%
```

A trade has to clear the round-trip cost just to break even, so 0.30% per
trade is the floor for "real" profitability with the defaults.

### Maker vs taker fees

Exchanges typically charge two different fee tiers:

- **Maker** fees apply when you *add* liquidity to the order book (limit
  orders that rest before being filled). They're lower — sometimes zero or
  even negative as a rebate.
- **Taker** fees apply when you *remove* liquidity (market orders, or limit
  orders that match immediately). They're higher.

trade-lab models *market* orders at the close of the execution bar (the
simplest realistic model for a vectorized backtest). Market orders pay the
**taker** fee, so set `--fee-rate` to the taker tier of your venue. Binance
spot's standard taker fee is 0.10% at the time of writing — hence the default.

### Why costs hit both entry and exit

Even a single round-trip trade is two transactions. The exchange charges on
each side; slippage hurts you on each side too. If you only charged costs on
the entry (or only on the exit), the equity curve would look about half a
percent more optimistic than reality per round-trip — enough to make a
marginal strategy look profitable when it isn't.

```bash
trade-lab backtest --strategy sma_cross --fee-rate 0.001 --slippage 0.0005
```

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

## Strategies

Each bundled strategy has a dedicated doc with exact rules, defaults,
example CLI invocations, and any per-strategy results worth pinning:

- [`sma_cross`](docs/strategies/sma_cross.md) — simple fast-vs-slow
  SMA crossover; the trend-following baseline.
- [`regime_sma_cross`](docs/strategies/regime_sma_cross.md) — crossover
  gated by a long-term regime SMA.
- [`regime_only`](docs/strategies/regime_only.md) — pure long-or-flat
  regime filter, no crossover.
- [`donchian_trend`](docs/strategies/donchian_trend.md) — Donchian
  breakout ensemble + SMA filter + volatility targeting.
- [`rsi`](docs/strategies/rsi.md) — RSI mean reversion (contrast
  baseline, not a recommendation).

Adding a new strategy is a five-minute job — subclass `Strategy`,
implement `generate_signals(candles)`, register the class in
`STRATEGIES` in `src/trade_lab/cli.py`. See
[Adding a new strategy](#adding-a-new-strategy) below.

## Roadmap

- Paper / live execution behind a `Broker` interface.
- Multi-asset portfolios and rebalancing.
- Short selling and leverage.
- Walk-forward and parameter optimization helpers.
