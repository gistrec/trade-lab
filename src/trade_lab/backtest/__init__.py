"""Backtest engine, metrics, plotting, exports, and parameter sweeps."""
from .engine import BacktestResult, Trade, execution_bars, run_backtest
from .metrics import Metrics, compute_metrics
from .plotting import plot_equity_curve
from .reports import TRADE_COLUMNS, trades_to_dataframe, write_trades_csv
from .sweep import SWEEP_COLUMNS, run_sma_sweep

__all__ = [
    "BacktestResult",
    "Metrics",
    "SWEEP_COLUMNS",
    "TRADE_COLUMNS",
    "Trade",
    "compute_metrics",
    "execution_bars",
    "plot_equity_curve",
    "run_backtest",
    "run_sma_sweep",
    "trades_to_dataframe",
    "write_trades_csv",
]
