"""Strategy-agnostic walk-forward validation with warmup feeding.

This module complements :mod:`.walk_forward` (which is SMA-specific) by
exposing a generic runner that takes an arbitrary list of "variants"
of a strategy and walk-forwards across them. The OOS evaluation
deliberately *does* let the strategy see candles from before the test
window — that's not look-ahead, that's the warmup an indicator needs
to compute its first signal. The metric, however, is computed only on
the test-window slice.

Two skeptical choices baked in:

1. **No look-ahead** in either direction. The train-window evaluation
   also gets warmup from before its start, so the train and test
   scores are apples-to-apples; a parameter that depends heavily on
   warmup is not artificially penalized on train.
2. **Optional purge gap** between train end and test start, in the
   spirit of López de Prado's purged CV. Default ``purge_days=0``
   matches live-trading semantics (where there *is* no gap).

The literature standard "1 year train / 6 months test, step 6 months"
gives 12 test folds across 2020-2026, which is enough samples for the
aggregate OOS Sharpe to be more than a coincidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from ..strategies.base import Strategy
from .engine import run_backtest


OBJECTIVE_SHARPE = "sharpe"
OBJECTIVE_TOTAL_RETURN = "total_return"
OBJECTIVE_RETURN_DIV_DRAWDOWN = "return_div_drawdown"
_VALID_OBJECTIVES = (
    OBJECTIVE_SHARPE,
    OBJECTIVE_TOTAL_RETURN,
    OBJECTIVE_RETURN_DIV_DRAWDOWN,
)

# Project-wide upper bound on the number of distinct (strategy, params)
# combinations a human in this repo has ever directly evaluated.
#
# Bailey & López de Prado (2014) recommend using *every* trial the
# decision-maker was exposed to — including ones thrown away — when
# computing the Deflated Sharpe Ratio. In practice this is impossible
# to count exactly; the standard advice is to fix a conservative
# number and never change it retroactively.
#
# This number is FIXED. Census as of commit (do not amend after-the-fact):
#   * SMA crossover sweep: 19 variants
#   * TSMOM ensemble configurations: 3 baseline + ~6 sensitivity (lookbacks, vol_target)
#   * PMA ladder configurations: 3 baseline + ~6 sensitivity
#   * Donchian rebalance-threshold sweep: 4
#   * VolatilityTargetWrapper target sweep: 5 (15/20/30/50/70 %) × 3 strategies = 15
#   * Cross-sectional momentum knobs: ~12 (top_k × weighting × BTC gate variants)
#   * Per-strategy × asset combinations in `wf_voltarget_7assets`: 63
#   * Walk-forward window variations explored informally: ~10
#   * Buffer for trials we honestly can't enumerate but that
#     happened during development: ~350
#
# Total ≈ 130 + 350 buffer = 480. Rounded up to 500.
PROJECT_NUM_TRIALS = 500


STRATEGY_WALK_FORWARD_COLUMNS = [
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "selected_label",
    "train_sharpe",
    "train_return_pct",
    "train_max_drawdown_pct",
    "train_calmar",
    "train_dsr",                # selection-bias-corrected DSR on train (per-fold N=|grid|)
    "test_sharpe",
    "test_return_pct",
    "test_max_drawdown_pct",
    "test_calmar",
    "test_buy_and_hold_return_pct",
    "test_buy_and_hold_max_drawdown_pct",
    "test_bars",
]


def _safe_calmar(return_pct: float, max_drawdown_pct: float) -> float:
    """Calmar = return / max DD with safe handling of zero drawdown.

    A zero-DD fold (strategy in cash the whole window) returns 0.0 — we
    explicitly do *not* return inf or NaN since that would poison
    downstream aggregates. A negative-return / zero-DD fold also
    returns 0.0; that asymmetry is intentional, since a strategy that
    stayed flat and returned 0% is being correctly summarized as
    "neither risk nor reward".
    """
    if max_drawdown_pct <= 1e-9:
        return 0.0
    return return_pct / max_drawdown_pct


@dataclass(frozen=True)
class ParamGridSpec:
    """One variant in the parameter grid.

    ``factory`` is called with no arguments and must return a fresh
    :class:`Strategy` instance — important when the runner evaluates
    the same variant on train and then test.

    ``warmup_days`` tells the runner how much pre-window candle data
    the strategy needs before its signal becomes valid. Set it >= the
    longest rolling lookback inside the strategy.
    """

    label: str
    factory: Callable[[], Strategy]
    warmup_days: int


@dataclass(frozen=True)
class WindowSpec:
    """One walk-forward (train, test) split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# ---------------------------------------------------------------------------
# Window generator
# ---------------------------------------------------------------------------


def generate_month_windows(
    candles: pd.DataFrame,
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    purge_days: int = 0,
) -> List[WindowSpec]:
    """Build calendar-aligned rolling (train, test) windows.

    Bounds are inclusive at the day level. The final window is dropped
    if either the train block or the test block falls past the last
    candle (we do *not* truncate train, only the last test).
    """
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_months, test_months, step_months must be positive")
    if purge_days < 0:
        raise ValueError("purge_days must be >= 0")
    if candles.empty:
        return []
    first = pd.Timestamp(candles.index[0])
    last = pd.Timestamp(candles.index[-1])

    windows: List[WindowSpec] = []
    cursor = first
    while True:
        train_start = cursor
        train_end = (
            train_start + DateOffset(months=train_months) - DateOffset(days=1)
        )
        test_start = train_end + DateOffset(days=1 + purge_days)
        test_end = (
            test_start + DateOffset(months=test_months) - DateOffset(days=1)
        )
        if train_end > last or test_start > last:
            break
        if test_end > last:
            test_end = last
        windows.append(
            WindowSpec(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor = cursor + DateOffset(months=step_months)
    return windows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_strategy_walk_forward(
    candles: pd.DataFrame,
    grid: Sequence[ParamGridSpec],
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    purge_days: int = 0,
    objective: str = OBJECTIVE_SHARPE,
    annualization_factor: int = 365,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
    return_oos_returns: bool = False,
):
    """Walk-forward an arbitrary set of strategy variants.

    For each rolling window: evaluate every variant on the train slice
    (with warmup from before ``train_start``), pick the one maximizing
    ``objective``, then re-evaluate it on the test slice (with warmup
    from before ``test_start``, possibly preceded by ``purge_days`` of
    gap).

    When ``return_oos_returns=True`` the function returns a tuple
    ``(detail_df, oos_returns_list)`` where each list entry is the
    per-bar net-return series for one fold's test window — useful for
    computing the concatenated-OOS DSR. The default keeps the
    original DataFrame-only signature so existing callers are
    untouched.
    """
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
        )
    if not grid:
        if return_oos_returns:
            return pd.DataFrame(columns=STRATEGY_WALK_FORWARD_COLUMNS), []
        return pd.DataFrame(columns=STRATEGY_WALK_FORWARD_COLUMNS)

    windows = generate_month_windows(
        candles,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        purge_days=purge_days,
    )

    rows: list[dict] = []
    oos_returns_list: list[pd.Series] = []
    for window in windows:
        # 1) Score every variant on train.
        scored = []
        for spec in grid:
            metrics = _evaluate_strategy_on_window(
                candles=candles,
                strategy=spec.factory(),
                window_start=window.train_start,
                window_end=window.train_end,
                warmup_days=spec.warmup_days,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                position_size=position_size,
                annualization_factor=annualization_factor,
            )
            scored.append((spec, metrics))

        # 2) Pick the variant with the best in-sample objective.
        scored.sort(key=lambda x: -_objective_score(x[1], objective))
        best_spec, best_train = scored[0]

        # 2b) Per-fold train DSR — Bailey-LdP DSR with selection-bias
        # correction over THIS fold's grid only. ``num_trials`` is the
        # number of variants tried *in this fold*; the cross-trial
        # sigma is the std of the variants' per-bar Sharpes (NOT
        # annualized — DSR's formula wants per-period Sharpes).
        train_dsr = _per_fold_train_dsr(
            [m for _, m in scored],
            annualization_factor=annualization_factor,
        )

        # 3) Evaluate the chosen variant on the held-out test slice.
        test_metrics = _evaluate_strategy_on_window(
            candles=candles,
            strategy=best_spec.factory(),
            window_start=window.test_start,
            window_end=window.test_end,
            warmup_days=best_spec.warmup_days,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
            annualization_factor=annualization_factor,
        )

        # Stash the OOS returns for the concatenated-OOS DSR computed
        # at the aggregation step.
        oos_returns_list.append(test_metrics["returns"])

        # Buy-and-hold benchmark on the test slice — same window only.
        # Charge the same one-round entry cost the strategy paid so the
        # comparison is apples-to-apples.
        bh_return, bh_dd = _buy_and_hold_on_window(
            candles, window.test_start, window.test_end,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        rows.append(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_label": best_spec.label,
                "train_sharpe": best_train["sharpe"],
                "train_return_pct": best_train["total_return"],
                "train_max_drawdown_pct": best_train["max_drawdown"],
                "train_calmar": _safe_calmar(
                    best_train["total_return"], best_train["max_drawdown"]
                ),
                "train_dsr": train_dsr,
                "test_sharpe": test_metrics["sharpe"],
                "test_return_pct": test_metrics["total_return"],
                "test_max_drawdown_pct": test_metrics["max_drawdown"],
                "test_calmar": _safe_calmar(
                    test_metrics["total_return"], test_metrics["max_drawdown"]
                ),
                "test_buy_and_hold_return_pct": bh_return,
                "test_buy_and_hold_max_drawdown_pct": bh_dd,
                "test_bars": test_metrics["bars"],
            }
        )

    detail_df = pd.DataFrame(rows, columns=STRATEGY_WALK_FORWARD_COLUMNS)
    if return_oos_returns:
        return detail_df, oos_returns_list
    return detail_df


def _per_fold_train_dsr(
    metrics_list: Sequence[dict],
    *,
    annualization_factor: int = 365,
) -> float:
    """DSR on the train slice with selection over the same-fold grid.

    Pass the list of metric-dicts (one per variant evaluated on train
    in this fold). The function uses each variant's per-period returns
    to compute a non-annualized Sharpe, picks the best, and applies
    Bailey-LdP DSR with ``num_trials = len(metrics_list)`` and the
    cross-trial standard deviation of those Sharpes as
    ``sharpe_std_dev``.
    """
    # Lazy import to avoid circular module loading at package import time.
    from .dsr import deflated_sharpe_ratio, sharpe_ratio_per_period

    if not metrics_list:
        return 0.0
    # Per-period (non-annualized) Sharpes — the DSR formula expects them.
    per_period_sharpes = []
    return_series_for_best: pd.Series = pd.Series(dtype=float)
    best_idx = -1
    for i, m in enumerate(metrics_list):
        r = m.get("returns")
        if r is None or len(r) < 4:
            sr = 0.0
        else:
            sr = sharpe_ratio_per_period(r)
        per_period_sharpes.append(sr)
        if i == 0 or sr > per_period_sharpes[best_idx]:
            best_idx = i
            return_series_for_best = m.get("returns", pd.Series(dtype=float))

    sharpe_std_dev = float(np.std(per_period_sharpes, ddof=1)) if len(per_period_sharpes) > 1 else 0.0
    return deflated_sharpe_ratio(
        returns=return_series_for_best,
        num_trials=len(metrics_list),
        sharpe_std_dev=sharpe_std_dev,
    )


def aggregate_walk_forward(
    detail_df: pd.DataFrame,
    *,
    annualization_factor: int = 365,
    oos_returns: Optional[Sequence[pd.Series]] = None,
    num_trials: int = PROJECT_NUM_TRIALS,
) -> dict:
    """Summary statistics across all OOS folds.

    When ``oos_returns`` (the per-fold return series list returned by
    ``run_strategy_walk_forward(..., return_oos_returns=True)``) is
    provided, the function also computes:

    * ``concatenated_oos_sharpe`` — annualized Sharpe on the stitched
      per-bar return series (assumes folds are non-overlapping and
      adjacent — true when ``step_months == test_months``).
    * ``concatenated_oos_dsr`` — Bailey-LdP DSR on the same stitched
      series with ``num_trials`` defaulting to
      :data:`PROJECT_NUM_TRIALS`. Pass ``num_trials`` explicitly if
      you want a different correction strength.
    * ``mean_per_fold_train_dsr`` — diagnostic for selection-bias
      strength fold-by-fold (high = train Sharpe is robust within
      the fold's grid; low = train Sharpe might be selection noise).
    """
    # Lazy import to avoid circular module loading at package import time.
    from .dsr import deflated_sharpe_ratio

    if detail_df.empty:
        return {
            "n_folds": 0,
            "mean_test_sharpe": 0.0,
            "median_test_sharpe": 0.0,
            "mean_test_calmar": 0.0,
            "median_test_calmar": 0.0,
            "hit_rate": 0.0,
            "mean_test_return": 0.0,
            "median_test_return": 0.0,
            "mean_test_max_dd": 0.0,
            "worst_test_return": 0.0,
            "best_test_return": 0.0,
            "mean_per_fold_train_dsr": 0.0,
            "concatenated_oos_sharpe": 0.0,
            "concatenated_oos_dsr": 0.0,
            "num_trials": num_trials,
        }
    summary = {
        "n_folds": len(detail_df),
        "mean_test_sharpe": float(detail_df["test_sharpe"].mean()),
        "median_test_sharpe": float(detail_df["test_sharpe"].median()),
        "mean_test_calmar": float(detail_df["test_calmar"].mean()),
        "median_test_calmar": float(detail_df["test_calmar"].median()),
        "hit_rate": float((detail_df["test_return_pct"] > 0).mean()),
        "mean_test_return": float(detail_df["test_return_pct"].mean()),
        "median_test_return": float(detail_df["test_return_pct"].median()),
        "mean_test_max_dd": float(detail_df["test_max_drawdown_pct"].mean()),
        "worst_test_return": float(detail_df["test_return_pct"].min()),
        "best_test_return": float(detail_df["test_return_pct"].max()),
        "mean_per_fold_train_dsr": float(detail_df["train_dsr"].mean())
            if "train_dsr" in detail_df.columns else 0.0,
        "num_trials": num_trials,
    }

    if oos_returns is None:
        summary["concatenated_oos_sharpe"] = 0.0
        summary["concatenated_oos_dsr"] = 0.0
        return summary

    # Concatenate per-fold OOS returns into one continuous series.
    # We accept that folds may have overlapping date stamps if the
    # caller used step_months < test_months; in that case we drop
    # duplicates by keeping the FIRST occurrence (i.e. the earlier
    # fold's view of that bar).
    valid_series = [s for s in oos_returns if s is not None and len(s) > 0]
    if not valid_series:
        summary["concatenated_oos_sharpe"] = 0.0
        summary["concatenated_oos_dsr"] = 0.0
        return summary
    concatenated = pd.concat(valid_series).sort_index()
    concatenated = concatenated[~concatenated.index.duplicated(keep="first")]

    std = float(concatenated.std())
    if std > 0 and not np.isnan(std):
        ann_sharpe = float(
            concatenated.mean() / std * np.sqrt(annualization_factor)
        )
    else:
        ann_sharpe = 0.0
    summary["concatenated_oos_sharpe"] = ann_sharpe

    # ``sharpe_std_dev`` should be the standard deviation of *trial*
    # per-period Sharpes under the null of zero true skill. When we
    # have no panel of trial-OOS-Sharpes to estimate it empirically,
    # the textbook fallback is the null estimator: per-period Sharpe
    # estimates have std ≈ 1/sqrt(T) under N(0, 1/T) sampling. That's
    # also what López de Prado uses when the trial cross-section is
    # unavailable.
    #
    # Earlier implementation accidentally used the std of *out-of-sample*
    # fold Sharpes, which is sample-driven noise, not null-driven, and
    # several times larger — making SR_0 huge and DSR collapse to 0.
    T = len(concatenated)
    sharpe_std_dev = 1.0 / np.sqrt(T) if T > 0 else 0.0

    summary["concatenated_oos_dsr"] = deflated_sharpe_ratio(
        returns=concatenated,
        num_trials=num_trials,
        sharpe_std_dev=sharpe_std_dev,
    )
    return summary


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _evaluate_strategy_on_window(
    candles: pd.DataFrame,
    strategy: Strategy,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    warmup_days: int,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    position_size: float,
    annualization_factor: int,
) -> dict:
    """Run ``strategy`` with warmup-from-before-window, then return
    metrics computed *only* on the in-window slice.

    The 2x safety factor on ``warmup_days`` is intentional: some
    strategies have rebalance bands or state machines that need a few
    bars of "settling" beyond the strict rolling window before signals
    become representative of steady state.
    """
    warmup_bars = max(int(warmup_days * 2), 1)
    warmup_start = window_start - pd.Timedelta(days=warmup_bars)
    extended = candles[
        (candles.index >= warmup_start) & (candles.index <= window_end)
    ]
    if extended.empty:
        return _empty_metrics()

    result = run_backtest(
        extended,
        strategy,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        position_size=position_size,
    )

    in_window = (result.returns.index >= window_start) & (
        result.returns.index <= window_end
    )
    window_returns = result.returns[in_window]
    if window_returns.empty:
        return _empty_metrics()

    window_equity = (1.0 + window_returns).cumprod()
    total_return = float(window_equity.iloc[-1] - 1.0)
    max_dd = float(abs(((window_equity / window_equity.cummax()) - 1.0).min()))
    std = float(window_returns.std())
    if std > 0 and not np.isnan(std):
        sharpe = float(
            window_returns.mean() / std * np.sqrt(annualization_factor)
        )
    else:
        sharpe = 0.0
    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "bars": int(len(window_returns)),
        "returns": window_returns,
    }


def _empty_metrics() -> dict:
    return {
        "sharpe": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "bars": 0,
        "returns": pd.Series(dtype=float),
    }


def _objective_score(metrics: dict, objective: str) -> float:
    if objective == OBJECTIVE_SHARPE:
        return metrics["sharpe"]
    if objective == OBJECTIVE_TOTAL_RETURN:
        return metrics["total_return"]
    if objective == OBJECTIVE_RETURN_DIV_DRAWDOWN:
        dd = metrics["max_drawdown"]
        if dd <= 1e-6:
            return metrics["total_return"] * 1e6 if metrics["total_return"] >= 0 else metrics["total_return"]
        return metrics["total_return"] / dd
    raise ValueError(f"Unknown objective {objective!r}")


def _buy_and_hold_on_window(
    candles: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> tuple[float, float]:
    """B&H return and max drawdown on a window, one entry round of costs.

    Same fee+slippage convention as the engine: one entry cost charged
    on the window's first bar, no exit cost. Default ``fee_rate=0``
    and ``slippage_rate=0`` preserve the academic pre-cost curve when
    the caller doesn't supply the strategy's cost params (e.g. tests
    that construct B&H benchmarks in isolation).
    """
    from .engine import buy_and_hold_with_costs

    sliced = candles[(candles.index >= window_start) & (candles.index <= window_end)]
    close = sliced["close"]
    if close.empty or len(close) < 2:
        return 0.0, 0.0
    equity, total_return = buy_and_hold_with_costs(
        close, initial_capital=initial_capital,
        fee_rate=fee_rate, slippage_rate=slippage_rate,
    )
    max_dd = float(abs(((equity / equity.cummax()) - 1.0).min()))
    return total_return, max_dd
