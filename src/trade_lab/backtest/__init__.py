"""Backtest engine, metrics, and plotting."""
from .engine import BacktestResult, Trade, run_backtest
from .metrics import Metrics, compute_metrics
from .plotting import plot_equity_curve

__all__ = [
    "BacktestResult",
    "Metrics",
    "Trade",
    "compute_metrics",
    "plot_equity_curve",
    "run_backtest",
]
