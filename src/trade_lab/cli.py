"""Command-line interface for trade-lab."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.metrics import compute_metrics
from .backtest.plotting import plot_equity_curve
from .backtest.reports import trades_to_dataframe, write_trades_csv
from .backtest.sweep import run_sma_sweep
from .config import load_config
from .data.fetch_ohlcv import fetch_ohlcv, validate_ohlcv
from .data.storage import (
    candles_path,
    filter_candles_by_date,
    load_candles,
    save_candles,
)
from .strategies.base import Strategy
from .strategies.rsi import RSIMeanReversionStrategy
from .strategies.sma_cross import SMACrossStrategy


STRATEGIES: dict[str, type[Strategy]] = {
    "sma_cross": SMACrossStrategy,
    "rsi": RSIMeanReversionStrategy,
}


def _coerce(value: str) -> Any:
    """Best-effort cast of a CLI value to int/float, falling back to str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    if not items:
        return {}
    out: dict[str, Any] = {}
    for kv in items:
        key, sep, value = kv.partition("=")
        if not sep:
            raise SystemExit(f"--param expects key=value, got: {kv!r}")
        out[key] = _coerce(value)
    return out


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of ints, ignoring blanks/whitespace."""
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Expected comma-separated integers, got {value!r}: {exc}")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange
    since = datetime.fromisoformat(args.since) if args.since else None
    until = datetime.fromisoformat(args.until) if args.until else None
    print(f"Fetching {args.symbol} {args.timeframe} from {exchange}...")
    df = fetch_ohlcv(
        exchange_id=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=since,
        until=until,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    else:
        path = save_candles(
            df,
            data_dir=cfg.data_dir,
            exchange=exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
    print(f"Saved {len(df)} candles to {path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange

    if args.input:
        candles = pd.read_parquet(args.input)
        validate_ohlcv(candles)
    else:
        candles = load_candles(
            data_dir=cfg.data_dir,
            exchange=exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

    candles = filter_candles_by_date(
        candles, start_date=args.start_date, end_date=args.end_date
    )
    if candles.empty:
        raise SystemExit(
            f"No candles in range [{args.start_date or '...'}, "
            f"{args.end_date or '...'}]"
        )

    strategy_cls = STRATEGIES[args.strategy]
    params = _parse_params(args.param)
    strategy = strategy_cls(**params)

    result = run_backtest(
        candles=candles,
        strategy=strategy,
        initial_capital=(
            args.initial_cash if args.initial_cash is not None else cfg.initial_capital
        ),
        fee_rate=args.fee_rate if args.fee_rate is not None else cfg.fee_rate,
        slippage_rate=args.slippage if args.slippage is not None else cfg.slippage_rate,
        position_size=args.position_size,
    )
    metrics = compute_metrics(result)

    fmt = "%Y-%m-%d %H:%M"
    period_start = candles.index[0].strftime(fmt)
    period_end = candles.index[-1].strftime(fmt)

    print()
    print(f"Strategy:             {strategy.name}")
    print(f"Symbol/timeframe:     {args.symbol} {args.timeframe}")
    print(f"Period:               {period_start} to {period_end}")
    print(f"Bars:                 {len(candles)}")
    print(f"Initial cash:         ${metrics.initial_capital:,.2f}")
    print()
    print("Strategy")
    print(f"  Final equity:       ${metrics.final_equity:,.2f}")
    print(f"  Total return:       {metrics.total_return:.2%}")
    print(f"  Max drawdown:       {metrics.max_drawdown:.2%}")
    print(f"  Number of trades:   {metrics.num_trades}")
    print(f"  Win rate:           {metrics.win_rate:.2%}")
    print(f"  Average trade:      {metrics.avg_trade_return:.2%}")
    print(f"  Fees paid:          ${metrics.total_fees:,.2f}")
    print()
    print("Buy & hold")
    print(f"  Final equity:       ${metrics.buy_and_hold_final_equity:,.2f}")
    print(f"  Total return:       {metrics.buy_and_hold_return:.2%}")
    print(f"  Max drawdown:       {metrics.buy_and_hold_max_drawdown:.2%}")

    if args.trades_csv:
        csv_path = write_trades_csv(result, candles, args.trades_csv)
        n_completed = len(trades_to_dataframe(result, candles))
        n_total = len(trades_to_dataframe(result, candles, include_open=True))
        n_open = n_total - n_completed
        print(f"Trades CSV:           {csv_path} ({n_completed} completed)")
        if n_open > 0:
            print(f"  excluded {n_open} open position(s)")

    if args.no_plot:
        return

    if args.save_plot:
        save_path: Path | None = Path(args.save_plot)
    else:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        save_path = (
            outputs_dir
            / f"{strategy.name}_{_safe_symbol(args.symbol)}_{args.timeframe}.png"
        )

    plot_equity_curve(
        result,
        candles=candles,
        title=f"{strategy.name} on {args.symbol} {args.timeframe}",
        save_path=save_path,
        show=args.show_plot,
        show_trades=args.show_trades,
    )
    print(f"Plot saved to {save_path}")


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange

    candles = load_candles(
        data_dir=cfg.data_dir,
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    candles = filter_candles_by_date(
        candles, start_date=args.start_date, end_date=args.end_date
    )
    if candles.empty:
        raise SystemExit(
            f"No candles in range [{args.start_date or '...'}, "
            f"{args.end_date or '...'}]"
        )

    fast_periods = _parse_int_list(args.fast_periods)
    slow_periods = _parse_int_list(args.slow_periods)
    n_total = len(fast_periods) * len(slow_periods)
    n_valid = sum(1 for f in fast_periods for s in slow_periods if f < s)
    n_skipped = n_total - n_valid

    fmt = "%Y-%m-%d %H:%M"
    print(f"Sweep:                {args.strategy}")
    print(f"Symbol/timeframe:     {args.symbol} {args.timeframe}")
    print(
        f"Period:               {candles.index[0].strftime(fmt)} "
        f"to {candles.index[-1].strftime(fmt)}"
    )
    print(f"Bars:                 {len(candles)}")
    print(
        f"Combinations:         {n_valid} valid "
        f"({n_skipped} skipped where fast >= slow)"
    )

    if n_valid == 0:
        raise SystemExit("No valid combinations to test.")

    df = run_sma_sweep(
        candles,
        fast_periods=fast_periods,
        slow_periods=slow_periods,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    formatters = {
        "final_equity": "${:,.2f}".format,
        "total_return_pct": "{:+.2%}".format,
        "buy_and_hold_return_pct": "{:+.2%}".format,
        "max_drawdown_pct": "{:.2%}".format,
        "win_rate": "{:.2%}".format,
        "fees_paid": "${:,.2f}".format,
    }
    print()
    print(df.to_string(index=False, formatters=formatters))
    print(f"\nResults saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade-lab",
        description="trade-lab: research framework for backtesting trading strategies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch and store historical OHLCV candles.")
    p_fetch.add_argument("--symbol", default="BTC/USDT", help="Symbol (default BTC/USDT)")
    p_fetch.add_argument("--timeframe", default="1h", help="Candle timeframe (default 1h)")
    p_fetch.add_argument("--exchange", default=None, help="ccxt exchange id (default binance)")
    p_fetch.add_argument("--since", default=None, help="ISO timestamp to start from")
    p_fetch.add_argument("--until", default=None, help="ISO timestamp to stop at")
    p_fetch.add_argument(
        "--output",
        default=None,
        help="Output Parquet path (default: <data_dir>/<exchange>_<symbol>_<timeframe>.parquet)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    p_bt = sub.add_parser("backtest", help="Run a backtest on stored candles.")
    p_bt.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    p_bt.add_argument(
        "--input",
        default=None,
        help="Path to an OHLCV Parquet file (overrides --symbol/--timeframe/--exchange lookup)",
    )
    p_bt.add_argument("--symbol", default="BTC/USDT")
    p_bt.add_argument("--timeframe", default="1h")
    p_bt.add_argument("--exchange", default=None)
    p_bt.add_argument(
        "--start-date",
        default=None,
        help="Filter candles from this date inclusive (YYYY-MM-DD)",
    )
    p_bt.add_argument(
        "--end-date",
        default=None,
        help="Filter candles through this date inclusive (YYYY-MM-DD)",
    )
    p_bt.add_argument("--initial-cash", type=float, default=None, help="Starting capital")
    p_bt.add_argument("--fee-rate", type=float, default=None, help="Per-side fee rate")
    p_bt.add_argument("--slippage", type=float, default=None, help="Per-side slippage rate")
    p_bt.add_argument("--position-size", type=float, default=1.0, help="Fraction of equity per trade (0, 1]")
    p_bt.add_argument(
        "--param",
        action="append",
        help="Strategy parameter as key=value (repeatable)",
    )
    p_bt.add_argument(
        "--save-plot",
        default=None,
        help="Override path for the equity curve PNG (default outputs/<strategy>_<symbol>_<timeframe>.png)",
    )
    p_bt.add_argument("--show-plot", action="store_true", help="Also display the plot interactively")
    p_bt.add_argument("--no-plot", action="store_true", help="Skip plotting entirely")
    p_bt.add_argument(
        "--show-trades",
        action="store_true",
        help="Add a price panel with buy/sell markers on execution candles",
    )
    p_bt.add_argument(
        "--trades-csv",
        default=None,
        help="Export completed trades to this CSV path (e.g. outputs/trades.csv)",
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_sw = sub.add_parser("sweep", help="Grid-search a strategy's parameters.")
    p_sw.add_argument("--strategy", default="sma_cross", choices=["sma_cross"])
    p_sw.add_argument("--symbol", default="BTC/USDT")
    p_sw.add_argument("--timeframe", default="1h")
    p_sw.add_argument("--exchange", default=None)
    p_sw.add_argument(
        "--fast-periods",
        default="5,10,20,30",
        help="Comma-separated list of fast SMA periods",
    )
    p_sw.add_argument(
        "--slow-periods",
        default="50,100,150,200",
        help="Comma-separated list of slow SMA periods",
    )
    p_sw.add_argument(
        "--start-date",
        default=None,
        help="Filter candles from this date inclusive (YYYY-MM-DD)",
    )
    p_sw.add_argument(
        "--end-date",
        default=None,
        help="Filter candles through this date inclusive (YYYY-MM-DD)",
    )
    p_sw.add_argument(
        "--output-csv",
        default="outputs/sweep.csv",
        help="Path to write the full results table",
    )
    p_sw.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
