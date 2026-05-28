"""Command-line interface for trade-lab."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.metrics import benchmark_verdict, compute_metrics
from .backtest.plotting import plot_equity_curve
from .backtest.reports import (
    trades_to_dataframe,
    write_debug_trades_csv,
    write_trades_csv,
)
from .backtest.sweep import run_sma_sweep
from .backtest.walk_forward import (
    OBJECTIVE_RETURN_DIV_DRAWDOWN,
    OBJECTIVE_TOTAL_RETURN,
    run_multi_walk_forward,
)
from .backtest.compare import (
    render_comparison_markdown,
    run_comparison_report,
)
from .backtest.cross_sectional import run_cross_sectional_momentum
from .backtest.multi_asset import (
    aggregate_multi_asset,
    run_multi_asset_yearly_validation,
    summarize_across_assets,
)
from .backtest.yearly import (
    aggregate_yearly_results,
    run_yearly_validation,
)
from .config import load_config
from .data.fetch_ohlcv import fetch_ohlcv, validate_ohlcv
from .data.storage import (
    candles_path,
    filter_candles_by_date,
    load_candles,
    save_candles,
)
from .strategies.base import Strategy
from .strategies.donchian_trend import DonchianTrendEnsembleStrategy
from .strategies.regime_only import RegimeOnlyStrategy
from .strategies.regime_sma_cross import RegimeSMACrossStrategy
from .strategies.pma_ratio import PriceMaRatioStrategy
from .strategies.rsi import RSIMeanReversionStrategy
from .strategies.sma_cross import SMACrossStrategy
from .strategies.tsmom import TimeSeriesMomentumStrategy


STRATEGIES: dict[str, type[Strategy]] = {
    "sma_cross": SMACrossStrategy,
    "regime_sma_cross": RegimeSMACrossStrategy,
    "regime_only": RegimeOnlyStrategy,
    "donchian_trend": DonchianTrendEnsembleStrategy,
    "tsmom": TimeSeriesMomentumStrategy,
    "pma_ratio": PriceMaRatioStrategy,
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
    print("Cost model")
    print(f"  Buy cost (fee + slip):  {metrics.buy_cost_pct:.2%}")
    print(f"  Sell cost (fee + slip): {metrics.sell_cost_pct:.2%}")
    print(f"  Round-trip cost:        {metrics.round_trip_cost_pct:.2%}")
    print()
    print("Strategy")
    print(f"  Final equity:       ${metrics.final_equity:,.2f}")
    print(f"  Gross return:       {metrics.gross_return:+.2%}")
    print(f"  Net return:         {metrics.total_return:+.2%}")
    print(f"  Max drawdown:       {metrics.max_drawdown:.2%}")
    print(f"  Win rate:           {metrics.win_rate:.2%}")
    print(f"  Avg gross trade:    {metrics.avg_gross_trade_return:+.2%}")
    print(f"  Avg net trade:      {metrics.avg_net_trade_return:+.2%}")
    print(f"  Avg cost / trade:   {metrics.avg_cost_per_trade:.2%}")
    print()
    print("Activity")
    print(f"  Completed trades:   {metrics.num_trades}")
    if metrics.num_open_trades:
        print(f"  Open at end:        {metrics.num_open_trades}")
    print(f"  Avg holding period: {metrics.avg_holding_period:.1f} bars")
    print(f"  Median holding:     {metrics.median_holding_period:.1f} bars")
    print(f"  Exposure time:      {metrics.exposure_pct:.2%}")
    print(f"  Avg trade return:   {metrics.avg_net_trade_return:+.2%}")
    print(f"  Best trade:         {metrics.best_trade_return:+.2%}")
    print(f"  Worst trade:        {metrics.worst_trade_return:+.2%}")
    print(f"  Total fees paid:    ${metrics.total_fees:,.2f}")
    print(f"  Fees / initial:     {metrics.fees_pct_of_initial_cash:.2%}")
    print(f"  Slippage cost est:  ${metrics.total_slippage:,.2f}")
    print()
    print("Buy & hold")
    print(f"  Final equity:       ${metrics.buy_and_hold_final_equity:,.2f}")
    print(f"  Total return:       {metrics.buy_and_hold_return:+.2%}")
    print(f"  Max drawdown:       {metrics.buy_and_hold_max_drawdown:.2%}")
    print()
    print(f"Verdict:              {benchmark_verdict(metrics)}")

    if args.trades_csv:
        csv_path = write_trades_csv(result, candles, args.trades_csv)
        n_completed = len(trades_to_dataframe(result, candles))
        n_total = len(trades_to_dataframe(result, candles, include_open=True))
        n_open = n_total - n_completed
        print(f"Trades CSV:           {csv_path} ({n_completed} completed)")
        if n_open > 0:
            print(f"  excluded {n_open} open position(s)")

    if args.debug_trades_csv:
        debug_path = write_debug_trades_csv(
            result,
            candles,
            args.debug_trades_csv,
            strategy=strategy,
            limit=args.debug_trades_limit,
        )
        print(
            f"Debug trades CSV:     {debug_path} "
            f"(first {args.debug_trades_limit} completed trades)"
        )

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


def cmd_walk_forward(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange

    candles = load_candles(
        data_dir=cfg.data_dir,
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if candles.empty:
        raise SystemExit("No candles to validate on.")

    fast_periods = _parse_int_list(args.fast_periods)
    slow_periods = _parse_int_list(args.slow_periods)
    regime_periods = _parse_int_list(args.regime_periods) if args.regime_periods else []
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())

    fmt = "%Y-%m-%d"
    print(f"Walk-forward:         {', '.join(strategies)}")
    print(f"Symbol/timeframe:     {args.symbol} {args.timeframe}")
    print(
        f"Period:               {candles.index[0].strftime(fmt)} "
        f"to {candles.index[-1].strftime(fmt)}"
    )
    print(
        f"Windows:              train={args.train_years}y "
        f"test={args.test_years}y step={args.step_years}y"
    )
    print(f"Objective:            {args.objective}")
    print(f"Grid:                 fast={fast_periods} slow={slow_periods}")
    if "regime_sma_cross" in strategies and regime_periods:
        print(f"                      regime={regime_periods}")

    df = run_multi_walk_forward(
        candles,
        fast_periods=fast_periods,
        slow_periods=slow_periods,
        regime_periods=regime_periods,
        strategies=strategies,
        objective=args.objective,
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )

    if df.empty:
        raise SystemExit(
            "No walk-forward windows fit the dataset (need at least "
            f"{args.train_years + args.test_years} years of data)."
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Pretty-print the columns the human cares about most.
    display = df.copy()
    display["train_start"] = display["train_start"].dt.strftime(fmt)
    display["train_end"] = display["train_end"].dt.strftime(fmt)
    display["test_start"] = display["test_start"].dt.strftime(fmt)
    display["test_end"] = display["test_end"].dt.strftime(fmt)
    # Combine fast/slow/regime into a compact parameters column for display.
    display["params"] = display.apply(_format_params, axis=1)
    cols_to_show = [
        "train_start", "train_end", "test_start", "test_end",
        "selected_strategy", "params",
        "train_return_pct", "train_max_drawdown_pct",
        "test_return_pct", "test_max_drawdown_pct",
        "test_buy_and_hold_return_pct", "test_buy_and_hold_max_drawdown_pct",
        "test_verdict",
    ]
    formatters = {
        "train_return_pct": "{:+.2%}".format,
        "train_max_drawdown_pct": "{:.2%}".format,
        "test_return_pct": "{:+.2%}".format,
        "test_max_drawdown_pct": "{:.2%}".format,
        "test_buy_and_hold_return_pct": "{:+.2%}".format,
        "test_buy_and_hold_max_drawdown_pct": "{:.2%}".format,
    }
    print()
    print(display[cols_to_show].to_string(index=False, formatters=formatters))
    print(f"\nResults saved to {out_path}")


def _format_params(row: pd.Series) -> str:
    if row["selected_strategy"] == "regime_sma_cross":
        return (
            f"f={int(row['fast_period'])}/s={int(row['slow_period'])}"
            f"/r={int(row['regime_period'])}"
        )
    return f"f={int(row['fast_period'])}/s={int(row['slow_period'])}"


def cmd_yearly(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange

    candles = load_candles(
        data_dir=cfg.data_dir,
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if candles.empty:
        raise SystemExit("No candles to validate on.")

    detail = run_yearly_validation(
        candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    if detail.empty:
        raise SystemExit("No yearly rows produced.")

    aggregate = aggregate_yearly_results(detail)

    fmt = "%Y-%m-%d"
    print(f"Yearly validation:    {args.symbol} {args.timeframe}")
    print(
        f"Period:               {candles.index[0].strftime(fmt)} "
        f"to {candles.index[-1].strftime(fmt)}"
    )
    print(f"Years:                {sorted(detail['year'].unique())}")
    print(f"Strategies:           {sorted(detail['strategy'].unique())}")

    detail_path = Path(args.output_csv)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)

    aggregate_path = (
        Path(args.aggregate_csv)
        if args.aggregate_csv
        else detail_path.with_name(detail_path.stem + "_aggregate.csv")
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)

    detail_formatters = {
        "return_pct": "{:+.2%}".format,
        "buy_and_hold_return_pct": "{:+.2%}".format,
        "max_drawdown_pct": "{:.2%}".format,
        "buy_and_hold_max_drawdown_pct": "{:.2%}".format,
        "exposure_pct": "{:.2%}".format,
        "fees_paid": "${:,.2f}".format,
    }
    aggregate_formatters = {
        "avg_annual_return": "{:+.2%}".format,
        "median_annual_return": "{:+.2%}".format,
        "best_year_return": "{:+.2%}".format,
        "worst_year_return": "{:+.2%}".format,
        "avg_exposure": "{:.2%}".format,
    }

    print()
    print("Per-year detail")
    print(detail.to_string(index=False, formatters=detail_formatters))
    print()
    print("Aggregate across years (per strategy)")
    print(aggregate.to_string(index=False, formatters=aggregate_formatters))
    print()
    print(f"Detail CSV:     {detail_path}")
    print(f"Aggregate CSV:  {aggregate_path}")


def cmd_multi_asset(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols must list at least one symbol")

    # Load each symbol's candles. Missing files are reported and skipped so
    # one missing asset doesn't tank the rest of the report.
    asset_candles: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            asset_candles[symbol] = load_candles(
                data_dir=cfg.data_dir,
                exchange=exchange,
                symbol=symbol,
                timeframe=args.timeframe,
            )
        except FileNotFoundError as exc:
            print(f"warn: skipping {symbol} ({exc})")

    if not asset_candles:
        raise SystemExit("No candle files found for any of the requested symbols.")

    detail = run_multi_asset_yearly_validation(
        asset_candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    if detail.empty:
        raise SystemExit("No rows produced.")
    aggregate = aggregate_multi_asset(detail)
    summary = summarize_across_assets(aggregate)

    detail_path = Path(args.output_csv)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)

    aggregate_path = (
        Path(args.aggregate_csv)
        if args.aggregate_csv
        else detail_path.with_name(detail_path.stem + "_aggregate.csv")
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)

    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else detail_path.with_name(detail_path.stem + "_summary.csv")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    pct = "{:+.2%}".format
    pct_abs = "{:.2%}".format
    aggregate_formatters = {
        "avg_annual_return": pct,
        "median_annual_return": pct,
        "best_year_return": pct,
        "worst_year_return": pct,
        "avg_exposure": pct_abs,
    }
    summary_formatters = {
        "avg_return_across_assets": pct,
        "avg_worst_year": pct,
        "avg_exposure_across_assets": pct_abs,
    }

    print(f"Multi-asset yearly:   {', '.join(asset_candles.keys())}")
    print(f"Timeframe:            {args.timeframe}")
    print()
    print("Per-(asset, strategy) aggregate")
    print(aggregate.to_string(index=False, formatters=aggregate_formatters))
    print()
    print("Across-asset summary (per strategy)")
    print(summary.to_string(index=False, formatters=summary_formatters))
    print()
    print(f"Detail CSV:     {detail_path}")
    print(f"Aggregate CSV:  {aggregate_path}")
    print(f"Summary CSV:    {summary_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols must list at least one symbol")

    asset_candles: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            asset_candles[symbol] = load_candles(
                data_dir=cfg.data_dir,
                exchange=exchange,
                symbol=symbol,
                timeframe=args.timeframe,
            )
        except FileNotFoundError as exc:
            print(f"warn: skipping {symbol} ({exc})")

    if not asset_candles:
        raise SystemExit("No candle files found for any of the requested symbols.")

    detail = run_comparison_report(
        asset_candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    if detail.empty:
        raise SystemExit("No rows produced.")

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(csv_path, index=False)

    md_path = (
        Path(args.output_md)
        if args.output_md
        else csv_path.with_suffix(".md")
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_comparison_markdown(detail))

    fmt_pct = "{:+.2%}".format
    fmt_pct_abs = "{:.2%}".format
    formatters = {
        "total_return_pct": fmt_pct,
        "cagr_pct": fmt_pct,
        "max_drawdown_pct": fmt_pct_abs,
        "sharpe": "{:+.2f}".format,
        "exposure_pct": fmt_pct_abs,
        "total_fees": "${:,.2f}".format,
        "total_slippage": "${:,.2f}".format,
        "turnover": "{:.2f}".format,
    }
    display_cols = [
        "asset", "strategy", "period",
        "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe",
        "exposure_pct", "num_trades", "total_fees", "turnover",
    ]
    print(f"Comparison report:    {', '.join(asset_candles.keys())}")
    print(f"Timeframe:            {args.timeframe}")
    print()
    print(detail[display_cols].to_string(index=False, formatters=formatters))
    print()
    print(f"Detail CSV:     {csv_path}")
    print(f"Markdown table: {md_path}")


def cmd_xsmom(args: argparse.Namespace) -> None:
    cfg = load_config()
    exchange = args.exchange or cfg.default_exchange
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols must list at least one symbol")

    asset_candles: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            asset_candles[symbol] = load_candles(
                data_dir=cfg.data_dir,
                exchange=exchange,
                symbol=symbol,
                timeframe=args.timeframe,
            )
        except FileNotFoundError as exc:
            print(f"warn: skipping {symbol} ({exc})")
    if not asset_candles:
        raise SystemExit("No candle files found for any of the requested symbols.")

    btc_candles = None
    if args.btc_gate_symbol:
        try:
            btc_candles = load_candles(
                data_dir=cfg.data_dir,
                exchange=exchange,
                symbol=args.btc_gate_symbol,
                timeframe=args.timeframe,
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                f"--btc-gate-symbol {args.btc_gate_symbol}: candles not found ({exc})"
            )
        # Conventional XSMOM only rotates altcoins — drop the gate symbol
        # from the universe if it happens to be there.
        asset_candles.pop(args.btc_gate_symbol, None)
        if not asset_candles:
            raise SystemExit(
                "Universe is empty after removing the BTC gate symbol; "
                "pass at least one altcoin symbol via --symbols."
            )

    result = run_cross_sectional_momentum(
        asset_candles,
        lookback_days=args.lookback_days,
        rebalance_days=args.rebalance_days,
        top_k=args.top_k,
        weighting=args.weighting,
        vol_lookback=args.vol_lookback,
        btc_candles=btc_candles,
        btc_gate_sma_period=args.btc_gate_sma_period,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "xsmom_weights.csv"
    equity_path = out_dir / "xsmom_equity.csv"
    result.weights.to_csv(weights_path)
    result.equity.to_frame(name="equity").to_csv(equity_path)

    print("Cross-sectional momentum:")
    print(f"  Universe:          {', '.join(asset_candles.keys())}")
    print(f"  BTC gate:          "
          f"{args.btc_gate_symbol or '(none)'} > SMA({args.btc_gate_sma_period})")
    print(f"  Lookback (days):   {args.lookback_days}")
    print(f"  Rebalance (days):  {args.rebalance_days}")
    print(f"  Top-K:             {args.top_k}")
    print(f"  Weighting:         {args.weighting}")
    print()
    print(f"  Total return:      {result.total_return:+.2%}")
    print(f"  Max drawdown:      {result.max_drawdown:.2%}")
    print(f"  Sharpe (annual):   {result.sharpe:+.2f}")
    print(f"  Num rebalances:    {result.num_rebalances}")
    print(f"  Avg basket size:   {result.average_basket_size:.2f}")
    print(f"  Avg cash fraction: {result.average_cash_fraction:.2%}")
    print(f"  Total fees:        ${result.total_fees:,.2f}")
    print(f"  Total slippage:    ${result.total_slippage:,.2f}")
    print()
    print(f"  Weights CSV:       {weights_path}")
    print(f"  Equity CSV:        {equity_path}")


def cmd_paper_status(args: argparse.Namespace) -> None:
    """Connect to the configured exchange and print balance + equity.

    Refuses to point at mainnet unless both TRADE_LAB_PAPER_SANDBOX=false
    and TRADE_LAB_PAPER_ALLOW_MAINNET=true are set — same two-flag
    requirement enforced inside the config loader.
    """
    from .execution import Broker, BrokerError, load_paper_config, PaperConfigError

    try:
        config = load_paper_config()
    except PaperConfigError as exc:
        raise SystemExit(f"Config error: {exc}")

    print(f"Exchange:    {config.exchange_id}")
    print(f"Sandbox:     {config.sandbox} "
          f"{'(testnet)' if config.sandbox else '(mainnet — allow_mainnet must be true)'}")
    print(f"Quote:       {config.quote_currency}")
    print(f"Basket:      {', '.join(config.basket)}")
    print()

    try:
        broker = Broker.connect(config)
    except BrokerError as exc:
        raise SystemExit(f"Broker connection failed: {exc}")
    print(f"Connected. Pulling balance...")
    snap = broker.fetch_balance_snapshot()
    print()
    print(f"  {config.quote_currency:6s}: free {snap.quote_free:>12,.2f}  "
          f"used {snap.quote_used:>12,.2f}  total {snap.quote_total:>12,.2f}")
    for sym in config.basket:
        total = snap.asset_totals.get(sym, 0.0)
        if total > 0.0:
            try:
                price = broker.fetch_ticker_price(f"{sym}/{config.quote_currency}")
                print(f"  {sym:6s}: total {total:>12.6f} × {price:>12,.2f} "
                      f"= {total * price:>14,.2f} {config.quote_currency}")
            except BrokerError as exc:
                print(f"  {sym:6s}: total {total:>12.6f}  (ticker error: {exc})")
        else:
            print(f"  {sym:6s}: total {total:>12.6f}")
    print()
    equity = broker.estimate_total_equity_usd(snapshot=snap)
    print(f"Estimated total equity: {equity:,.2f} {config.quote_currency}")


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
    p_bt.add_argument(
        "--debug-trades-csv",
        default=None,
        help="Audit CSV with signal vs execution timing for the first N trades",
    )
    p_bt.add_argument(
        "--debug-trades-limit",
        type=int,
        default=10,
        help="Max trades to include in --debug-trades-csv (default 10)",
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

    p_wf = sub.add_parser(
        "walk-forward",
        help="Rolling-window walk-forward validation across SMA-family strategies.",
    )
    p_wf.add_argument(
        "--strategies",
        default="sma_cross,regime_sma_cross",
        help="Comma-separated subset of {sma_cross, regime_sma_cross}",
    )
    p_wf.add_argument("--symbol", default="BTC/USDT")
    p_wf.add_argument("--timeframe", default="1d")
    p_wf.add_argument("--exchange", default=None)
    p_wf.add_argument(
        "--fast-periods",
        default="5,10,20,30",
        help="Comma-separated list of fast SMA periods (shared by both strategies)",
    )
    p_wf.add_argument(
        "--slow-periods",
        default="50,100,150,200",
        help="Comma-separated list of slow SMA periods (shared by both strategies)",
    )
    p_wf.add_argument(
        "--regime-periods",
        default="100,150,200,300",
        help="Comma-separated list of regime SMA periods (regime_sma_cross only)",
    )
    p_wf.add_argument(
        "--objective",
        default=OBJECTIVE_TOTAL_RETURN,
        choices=[OBJECTIVE_TOTAL_RETURN, OBJECTIVE_RETURN_DIV_DRAWDOWN],
        help="Criterion used to pick the best params on each train window",
    )
    p_wf.add_argument(
        "--train-years", type=int, default=2,
        help="Length of each training window in years",
    )
    p_wf.add_argument(
        "--test-years", type=int, default=1,
        help="Length of each test window in years",
    )
    p_wf.add_argument(
        "--step-years", type=int, default=1,
        help="How many years to roll forward between windows",
    )
    p_wf.add_argument(
        "--output-csv",
        default="outputs/walk_forward.csv",
        help="Path to write the per-window results",
    )
    p_wf.set_defaults(func=cmd_walk_forward)

    p_yr = sub.add_parser(
        "yearly",
        help="Fixed-parameter yearly validation across a set of strategies.",
    )
    p_yr.add_argument("--symbol", default="BTC/USDT")
    p_yr.add_argument("--timeframe", default="1d")
    p_yr.add_argument("--exchange", default=None)
    p_yr.add_argument(
        "--output-csv",
        default="outputs/yearly.csv",
        help="Per-(year, strategy) detail CSV",
    )
    p_yr.add_argument(
        "--aggregate-csv",
        default=None,
        help="Aggregate per-strategy CSV (default: alongside --output-csv)",
    )
    p_yr.set_defaults(func=cmd_yearly)

    p_ma = sub.add_parser(
        "multi-asset",
        help="Fixed-strategy yearly validation across multiple assets.",
    )
    p_ma.add_argument(
        "--symbols",
        default="BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT",
        help="Comma-separated list of symbols to evaluate",
    )
    p_ma.add_argument("--timeframe", default="1d")
    p_ma.add_argument("--exchange", default=None)
    p_ma.add_argument(
        "--output-csv",
        default="outputs/multi_asset.csv",
        help="Per-(asset, year, strategy) detail CSV",
    )
    p_ma.add_argument(
        "--aggregate-csv",
        default=None,
        help="Aggregate per-(asset, strategy) CSV (default: alongside --output-csv)",
    )
    p_ma.add_argument(
        "--summary-csv",
        default=None,
        help="Across-asset summary CSV (default: alongside --output-csv)",
    )
    p_ma.set_defaults(func=cmd_multi_asset)

    p_cm = sub.add_parser(
        "compare",
        help="Subperiod comparison across strategies and assets.",
    )
    p_cm.add_argument(
        "--symbols",
        default="BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT",
        help="Comma-separated list of symbols",
    )
    p_cm.add_argument("--timeframe", default="1d")
    p_cm.add_argument("--exchange", default=None)
    p_cm.add_argument(
        "--output-csv",
        default="outputs/compare.csv",
        help="Per-(asset, strategy, subperiod) detail CSV",
    )
    p_cm.add_argument(
        "--output-md",
        default=None,
        help="Markdown summary (default: alongside --output-csv)",
    )
    p_cm.set_defaults(func=cmd_compare)

    p_xs = sub.add_parser(
        "xsmom",
        help="Cross-sectional momentum (top-N rotation) across an asset universe.",
    )
    p_xs.add_argument(
        "--symbols",
        default="ETH/USDT,BNB/USDT,SOL/USDT",
        help="Comma-separated alt universe (BTC excluded by convention if it is "
             "the BTC-gate symbol).",
    )
    p_xs.add_argument("--timeframe", default="1d")
    p_xs.add_argument("--exchange", default=None)
    p_xs.add_argument("--lookback-days", type=int, default=30)
    p_xs.add_argument("--rebalance-days", type=int, default=7)
    p_xs.add_argument("--top-k", type=int, default=2)
    p_xs.add_argument(
        "--weighting",
        choices=("equal", "inverse_vol"),
        default="equal",
    )
    p_xs.add_argument("--vol-lookback", type=int, default=30)
    p_xs.add_argument(
        "--btc-gate-symbol",
        default="BTC/USDT",
        help="Pass empty string to disable the BTC > SMA(N) gate.",
    )
    p_xs.add_argument("--btc-gate-sma-period", type=int, default=200)
    p_xs.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the weights/equity CSV outputs.",
    )
    p_xs.set_defaults(func=cmd_xsmom)

    p_ps = sub.add_parser(
        "paper-status",
        help="Connect to the configured exchange (CCXT) and print live balance.",
    )
    p_ps.set_defaults(func=cmd_paper_status)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
