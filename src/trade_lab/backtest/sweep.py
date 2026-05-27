"""Parameter sweeps over a strategy's hyperparameters.

Grid-searches a strategy across a small parameter space and returns one
row per backtest. Designed for research only — no automatic selection
of "live" parameters happens here.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..strategies.sma_cross import SMACrossStrategy
from .engine import run_backtest
from .metrics import compute_metrics


SWEEP_COLUMNS = [
    "fast_period",
    "slow_period",
    "final_equity",
    "total_return_pct",
    "buy_and_hold_return_pct",
    "max_drawdown_pct",
    "num_trades",
    "win_rate",
    "fees_paid",
]


def run_sma_sweep(
    candles: pd.DataFrame,
    fast_periods: Iterable[int],
    slow_periods: Iterable[int],
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Grid-search the SMA crossover strategy.

    Combinations where ``fast_period >= slow_period`` are skipped — those
    don't form a meaningful crossover (and the strategy constructor would
    reject them). Results are returned sorted by ``total_return_pct``
    descending; ties keep insertion order.
    """
    rows: list[dict] = []
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            strategy = SMACrossStrategy(fast_period=int(fast), slow_period=int(slow))
            result = run_backtest(
                candles,
                strategy,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                position_size=position_size,
            )
            m = compute_metrics(result)
            rows.append(
                {
                    "fast_period": int(fast),
                    "slow_period": int(slow),
                    "final_equity": m.final_equity,
                    "total_return_pct": m.total_return,
                    "buy_and_hold_return_pct": m.buy_and_hold_return,
                    "max_drawdown_pct": m.max_drawdown,
                    "num_trades": m.num_trades,
                    "win_rate": m.win_rate,
                    "fees_paid": m.total_fees,
                }
            )

    df = pd.DataFrame(rows, columns=SWEEP_COLUMNS)
    if not df.empty:
        df = (
            df.sort_values("total_return_pct", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
    return df
