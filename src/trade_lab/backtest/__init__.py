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
from .sweep import (
    REGIME_SWEEP_COLUMNS,
    SWEEP_COLUMNS,
    run_regime_sma_sweep,
    run_sma_sweep,
)
from .walk_forward import (
    MULTI_WALK_FORWARD_COLUMNS,
    OBJECTIVE_RETURN_DIV_DRAWDOWN,
    OBJECTIVE_TOTAL_RETURN,
    WALK_FORWARD_COLUMNS,
    WalkForwardWindow,
    generate_windows,
    run_multi_walk_forward,
    run_sma_walk_forward,
)

__all__ = [
    "BacktestResult",
    "DEBUG_TRADE_COLUMNS",
    "Metrics",
    "MULTI_WALK_FORWARD_COLUMNS",
    "OBJECTIVE_RETURN_DIV_DRAWDOWN",
    "OBJECTIVE_TOTAL_RETURN",
    "REGIME_SWEEP_COLUMNS",
    "SWEEP_COLUMNS",
    "TRADE_COLUMNS",
    "Trade",
    "VERDICT_LOWER_RETURN_LOWER_DD",
    "VERDICT_OUTPERFORMS_BH",
    "VERDICT_UNDERPERFORMS_BH",
    "WALK_FORWARD_COLUMNS",
    "WalkForwardWindow",
    "benchmark_verdict",
    "compute_metrics",
    "debug_trades_dataframe",
    "execution_bars",
    "generate_windows",
    "plot_equity_curve",
    "run_backtest",
    "run_multi_walk_forward",
    "run_regime_sma_sweep",
    "run_sma_sweep",
    "run_sma_walk_forward",
    "trades_to_dataframe",
    "write_debug_trades_csv",
    "write_trades_csv",
]
