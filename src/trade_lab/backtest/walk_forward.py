"""Walk-forward validation for SMA strategies.

Splits a candles DataFrame into rolling train / test windows, runs a
parameter sweep on each train slice, picks the best parameters from
*train only*, and evaluates that exact pair on the immediately following
test slice. This gives a much more honest picture of generalization than
optimizing on the full window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd
from pandas.tseries.offsets import DateOffset

from ..data.storage import filter_candles_by_date
from ..strategies.regime_sma_cross import RegimeSMACrossStrategy
from ..strategies.sma_cross import SMACrossStrategy
from .engine import run_backtest
from .metrics import benchmark_verdict, compute_metrics
from .sweep import run_regime_sma_sweep, run_sma_sweep


OBJECTIVE_TOTAL_RETURN = "total_return"
OBJECTIVE_RETURN_DIV_DRAWDOWN = "return_div_drawdown"
_VALID_OBJECTIVES = (OBJECTIVE_TOTAL_RETURN, OBJECTIVE_RETURN_DIV_DRAWDOWN)


MULTI_WALK_FORWARD_COLUMNS = [
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "selected_strategy",
    "fast_period",
    "slow_period",
    "regime_period",
    "train_return_pct",
    "train_max_drawdown_pct",
    "test_return_pct",
    "test_max_drawdown_pct",
    "test_buy_and_hold_return_pct",
    "test_buy_and_hold_max_drawdown_pct",
    "test_verdict",
]


WALK_FORWARD_COLUMNS = [
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "fast_period",
    "slow_period",
    "train_return_pct",
    "test_return_pct",
    "test_buy_and_hold_return_pct",
    "test_max_drawdown_pct",
    "test_verdict",
]


@dataclass(frozen=True)
class WalkForwardWindow:
    """One (train, test) split of the candles index."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_windows(
    candles: pd.DataFrame,
    train_years: int = 2,
    test_years: int = 1,
    step_years: int = 1,
) -> List[WalkForwardWindow]:
    """Build rolling train / test windows that fit inside ``candles``.

    Bounds are calendar-based (``DateOffset(years=N)``) and inclusive at
    the day level: a 2-year train starting 2018-01-01 ends 2019-12-31
    and the matching test starts 2020-01-01. The last window's test
    range is truncated if it would extend past the last candle.
    """
    if candles.empty:
        return []
    first = pd.Timestamp(candles.index[0])
    last = pd.Timestamp(candles.index[-1])

    windows: List[WalkForwardWindow] = []
    cursor = first
    while True:
        train_start = cursor
        train_end = train_start + DateOffset(years=train_years) - DateOffset(days=1)
        test_start = train_end + DateOffset(days=1)
        test_end = test_start + DateOffset(years=test_years) - DateOffset(days=1)

        # The train window has to fit. We allow the test to be truncated
        # at the end (so the last partial year is still examined).
        if train_end > last:
            break
        if test_start > last:
            break
        if test_end > last:
            test_end = last

        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor = cursor + DateOffset(years=step_years)

    return windows


def run_sma_walk_forward(
    candles: pd.DataFrame,
    fast_periods: Iterable[int],
    slow_periods: Iterable[int],
    train_years: int = 2,
    test_years: int = 1,
    step_years: int = 1,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Run walk-forward validation of the SMA crossover.

    For each rolling window: sweep ``(fast, slow)`` on the train slice
    only, pick the pair with the highest total return on train, then
    evaluate that pair on the *following* test slice. Returns one row
    per window.

    The function never sees train and test mixed: parameter selection
    only touches train candles, and the test backtest only touches test
    candles. The full-history sweep is therefore *not* run.
    """
    fast_periods = list(fast_periods)
    slow_periods = list(slow_periods)
    windows = generate_windows(
        candles, train_years=train_years, test_years=test_years, step_years=step_years
    )

    rows: list[dict] = []
    for window in windows:
        train_candles = filter_candles_by_date(
            candles,
            start_date=_fmt_date(window.train_start),
            end_date=_fmt_date(window.train_end),
        )
        test_candles = filter_candles_by_date(
            candles,
            start_date=_fmt_date(window.test_start),
            end_date=_fmt_date(window.test_end),
        )
        if train_candles.empty or test_candles.empty:
            continue

        # 1) Sweep on TRAIN only and pick the best total-return pair.
        train_sweep = run_sma_sweep(
            train_candles,
            fast_periods=fast_periods,
            slow_periods=slow_periods,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
        )
        if train_sweep.empty:
            continue
        best = train_sweep.iloc[0]
        fast = int(best["fast_period"])
        slow = int(best["slow_period"])

        # 2) Evaluate the chosen pair on TEST only.
        test_result = run_backtest(
            test_candles,
            SMACrossStrategy(fast_period=fast, slow_period=slow),
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
        )
        test_metrics = compute_metrics(test_result)

        rows.append(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "fast_period": fast,
                "slow_period": slow,
                "train_return_pct": float(best["total_return_pct"]),
                "test_return_pct": test_metrics.total_return,
                "test_buy_and_hold_return_pct": test_metrics.buy_and_hold_return,
                "test_max_drawdown_pct": test_metrics.max_drawdown,
                "test_verdict": benchmark_verdict(test_metrics),
            }
        )

    return pd.DataFrame(rows, columns=WALK_FORWARD_COLUMNS)


def _fmt_date(ts: pd.Timestamp) -> str:
    """Format a timestamp as YYYY-MM-DD for :func:`filter_candles_by_date`."""
    return ts.strftime("%Y-%m-%d")


def _score(total_return_pct: float, max_drawdown_pct: float, objective: str) -> float:
    """Translate (return, drawdown) into a scalar to maximize."""
    if objective == OBJECTIVE_TOTAL_RETURN:
        return total_return_pct
    if objective == OBJECTIVE_RETURN_DIV_DRAWDOWN:
        if max_drawdown_pct <= 0:
            # A truly zero-DD path is "infinitely good" if it made money,
            # and just a flat result otherwise. Use a large multiplier to
            # outrank any finite-DD candidate without producing NaN/inf.
            return total_return_pct * 1e6 if total_return_pct >= 0 else total_return_pct
        return total_return_pct / max_drawdown_pct
    raise ValueError(
        f"Unknown objective {objective!r}. "
        f"Expected one of {_VALID_OBJECTIVES}."
    )


def run_multi_walk_forward(
    candles: pd.DataFrame,
    fast_periods: Iterable[int],
    slow_periods: Iterable[int],
    regime_periods: Iterable[int] | None = None,
    strategies: tuple[str, ...] = ("sma_cross", "regime_sma_cross"),
    objective: str = OBJECTIVE_TOTAL_RETURN,
    train_years: int = 2,
    test_years: int = 1,
    step_years: int = 1,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Walk-forward across multiple SMA-family strategies.

    For each rolling (train, test) window:

    1. Sweep every requested strategy's parameter grid on the train slice.
    2. Rank candidates by ``objective`` (``total_return`` by default; the
       alternative ``return_div_drawdown`` favours risk-adjusted picks).
    3. Take the single best ``(strategy, params)`` across all candidates.
    4. Run a fresh backtest with that exact selection on the test slice.

    Returns one row per window. The chosen strategy is recorded so a
    later analysis can see which family generalized best in each regime.
    """
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
        )

    fast_periods = list(fast_periods)
    slow_periods = list(slow_periods)
    regime_periods = list(regime_periods) if regime_periods is not None else []
    strategies = tuple(strategies)

    windows = generate_windows(
        candles, train_years=train_years, test_years=test_years, step_years=step_years
    )

    rows: list[dict] = []
    for window in windows:
        train_candles = filter_candles_by_date(
            candles,
            start_date=_fmt_date(window.train_start),
            end_date=_fmt_date(window.train_end),
        )
        test_candles = filter_candles_by_date(
            candles,
            start_date=_fmt_date(window.test_start),
            end_date=_fmt_date(window.test_end),
        )
        if train_candles.empty or test_candles.empty:
            continue

        candidates: list[dict] = []

        if "sma_cross" in strategies:
            sma_df = run_sma_sweep(
                train_candles,
                fast_periods=fast_periods,
                slow_periods=slow_periods,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                position_size=position_size,
            )
            for _, row in sma_df.iterrows():
                candidates.append(
                    {
                        "strategy": "sma_cross",
                        "fast_period": int(row["fast_period"]),
                        "slow_period": int(row["slow_period"]),
                        "regime_period": None,
                        "train_return_pct": float(row["total_return_pct"]),
                        "train_max_drawdown_pct": float(row["max_drawdown_pct"]),
                    }
                )

        if "regime_sma_cross" in strategies and regime_periods:
            regime_df = run_regime_sma_sweep(
                train_candles,
                fast_periods=fast_periods,
                slow_periods=slow_periods,
                regime_periods=regime_periods,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                position_size=position_size,
            )
            for _, row in regime_df.iterrows():
                candidates.append(
                    {
                        "strategy": "regime_sma_cross",
                        "fast_period": int(row["fast_period"]),
                        "slow_period": int(row["slow_period"]),
                        "regime_period": int(row["regime_period"]),
                        "train_return_pct": float(row["total_return_pct"]),
                        "train_max_drawdown_pct": float(row["max_drawdown_pct"]),
                    }
                )

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda c: _score(
                c["train_return_pct"], c["train_max_drawdown_pct"], objective
            ),
        )

        strategy = _build_strategy(best)
        test_result = run_backtest(
            test_candles,
            strategy,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
        )
        test_metrics = compute_metrics(test_result)

        rows.append(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_strategy": best["strategy"],
                "fast_period": best["fast_period"],
                "slow_period": best["slow_period"],
                "regime_period": best["regime_period"],
                "train_return_pct": best["train_return_pct"],
                "train_max_drawdown_pct": best["train_max_drawdown_pct"],
                "test_return_pct": test_metrics.total_return,
                "test_max_drawdown_pct": test_metrics.max_drawdown,
                "test_buy_and_hold_return_pct": test_metrics.buy_and_hold_return,
                "test_buy_and_hold_max_drawdown_pct": test_metrics.buy_and_hold_max_drawdown,
                "test_verdict": benchmark_verdict(test_metrics),
            }
        )

    return pd.DataFrame(rows, columns=MULTI_WALK_FORWARD_COLUMNS)


def _build_strategy(spec: dict):
    """Reconstruct a strategy instance from a candidate dict."""
    if spec["strategy"] == "sma_cross":
        return SMACrossStrategy(
            fast_period=spec["fast_period"],
            slow_period=spec["slow_period"],
        )
    if spec["strategy"] == "regime_sma_cross":
        return RegimeSMACrossStrategy(
            fast_period=spec["fast_period"],
            slow_period=spec["slow_period"],
            regime_period=spec["regime_period"],
        )
    raise ValueError(f"Unknown strategy {spec['strategy']!r}")
