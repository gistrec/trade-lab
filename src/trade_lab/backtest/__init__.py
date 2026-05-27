"""Backtest engine, metrics, plotting, exports, and parameter sweeps."""
from .engine import BacktestResult, Trade, execution_bars, run_backtest
from .metrics import (
    VERDICT_LOWER_RETURN_LOWER_DD,
    VERDICT_OUTPERFORMS_BH,
    VERDICT_UNDERPERFORMS_BH,
    Metrics,
    benchmark_verdict,
    compute_metrics,
)
from .plotting import plot_equity_curve
from .reports import (
    DEBUG_TRADE_COLUMNS,
    TRADE_COLUMNS,
    debug_trades_dataframe,
    trades_to_dataframe,
    write_debug_trades_csv,
    write_trades_csv,
)
from .sweep import SWEEP_COLUMNS, run_sma_sweep

__all__ = [
    "BacktestResult",
    "DEBUG_TRADE_COLUMNS",
    "Metrics",
    "SWEEP_COLUMNS",
    "TRADE_COLUMNS",
    "Trade",
    "VERDICT_LOWER_RETURN_LOWER_DD",
    "VERDICT_OUTPERFORMS_BH",
    "VERDICT_UNDERPERFORMS_BH",
    "benchmark_verdict",
    "compute_metrics",
    "debug_trades_dataframe",
    "execution_bars",
    "plot_equity_curve",
    "run_backtest",
    "run_sma_sweep",
    "trades_to_dataframe",
    "write_debug_trades_csv",
    "write_trades_csv",
]
