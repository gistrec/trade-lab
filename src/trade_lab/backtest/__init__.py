"""Backtest engine, metrics, plotting, and exports."""
from .engine import BacktestResult, Trade, execution_bars, run_backtest
from .metrics import Metrics, compute_metrics
from .plotting import plot_equity_curve
from .reports import TRADE_COLUMNS, trades_to_dataframe, write_trades_csv

__all__ = [
    "BacktestResult",
    "Metrics",
    "TRADE_COLUMNS",
    "Trade",
    "compute_metrics",
    "execution_bars",
    "plot_equity_curve",
    "run_backtest",
    "trades_to_dataframe",
    "write_trades_csv",
]
