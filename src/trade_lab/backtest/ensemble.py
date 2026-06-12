"""Multi-asset ensemble portfolio of single-sleeve walk-forward runs.

A "sleeve" is one (strategy, asset, vol-target variant) combination.
This module wires per-sleeve walk-forward results together into an
equal-weight portfolio with explicit re-balancing semantics:

* Dynamic equal-weight: each bar's target weight is ``1 / N_active``
  where ``N_active`` is the number of sleeves whose OOS return is
  defined at that bar. Sleeves whose underlying asset has not yet
  listed are simply absent from the panel — they do NOT count
  toward the denominator and do NOT drag the portfolio toward cash.
* Rebalance-on-universe-change: when ``N_active`` changes (e.g. SOL
  comes online in 2020-08), every existing sleeve's weight shifts
  from ``1/N_old`` to ``1/N_new``. The aggregate ``|weight_change|``
  is treated exactly like a strategy's turnover at the engine level —
  charged ``(fee_rate + slippage_rate)`` per unit. This is the
  conservative side of the choice; the alternative ("new asset gets
  weight only from future cash inflows") would be cheaper but
  doesn't match how a fully-invested equal-weight CTA actually
  operates.

The sleeve-level returns coming out of :func:`run_strategy_walk_forward`
are already net of each sleeve's own intra-strategy costs (the engine
charges fees + slippage on every turnover inside the strategy). The
costs charged inside this module are only the **additional** transaction
costs the portfolio-level allocator pays when re-balancing across
sleeves.
"""
from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..strategies.base import Strategy
from .walk_forward_v2 import (
    ParamGridSpec,
    PROJECT_NUM_TRIALS,
    aggregate_walk_forward,
    run_strategy_walk_forward,
)

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class SleeveSpec:
    """One sleeve of the ensemble.

    ``factory`` returns a fresh :class:`Strategy` instance (potentially
    wrapped in :class:`VolatilityTargetWrapper`). ``warmup_days`` is
    forwarded to the per-sleeve walk-forward runner.
    """

    label: str           # human-readable, e.g. "tsmom_medium__BTC__raw"
    asset: str           # registry key, e.g. "BTC"
    factory: Callable[[], Strategy]
    warmup_days: int


@dataclass(frozen=True)
class EnsembleResult:
    """Output bundle of :func:`run_ensemble_walk_forward`."""

    per_sleeve_detail: dict[str, pd.DataFrame]    # sleeve_label -> WF detail
    per_sleeve_oos_returns: dict[str, pd.Series]  # sleeve_label -> stitched OOS returns
    sleeve_returns_panel: pd.DataFrame            # date x sleeve
    sleeve_active_panel: pd.DataFrame             # date x sleeve, bool
    target_weights: pd.DataFrame                  # date x sleeve, dynamic 1/N_active
    portfolio_returns_gross: pd.Series            # before rebalance cost
    portfolio_returns_net: pd.Series              # after rebalance cost
    portfolio_equity: pd.Series                   # net cumulative
    rebalance_turnover: pd.Series                 # sum |weight diff| per bar
    rebalance_cost: pd.Series                     # turnover * (fee + slippage)
    correlation_matrix: pd.DataFrame              # pairwise corr of sleeve OOS returns
    portfolio_metrics: dict                       # CAGR/Sharpe/Sortino/Calmar/...
    portfolio_dsr: float                          # DSR @ num_trials on portfolio OOS


def run_ensemble_walk_forward(
    sleeves: Sequence[SleeveSpec],
    asset_candles: Mapping[str, pd.DataFrame],
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    objective: str = "sharpe",
    annualization_factor: int = 365,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    num_trials_for_dsr: int = PROJECT_NUM_TRIALS,
) -> EnsembleResult:
    """Walk-forward each sleeve, then aggregate into one portfolio."""
    if not sleeves:
        raise ValueError("sleeves must be non-empty")

    per_sleeve_detail: dict[str, pd.DataFrame] = {}
    per_sleeve_oos: dict[str, pd.Series] = {}
    for sleeve in sleeves:
        candles = asset_candles.get(sleeve.asset)
        if candles is None or candles.empty:
            logger.warning(
                "ensemble: sleeve %r skipped — no candles for asset %r; "
                "the portfolio runs on fewer sleeves than configured.",
                sleeve.label, sleeve.asset,
            )
            continue
        grid = [ParamGridSpec(
            label=sleeve.label,
            factory=sleeve.factory,
            warmup_days=sleeve.warmup_days,
        )]
        detail, oos_list = run_strategy_walk_forward(
            candles, grid,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            objective=objective,
            annualization_factor=annualization_factor,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            return_oos_returns=True,
        )
        per_sleeve_detail[sleeve.label] = detail
        # Concatenate per-fold OOS series into one continuous return
        # series for the sleeve. Folds are adjacent when step_months
        # == test_months; if there is overlap (step < test) we keep
        # the earlier fold's value.
        non_empty = [s for s in oos_list if s is not None and len(s) > 0]
        if not non_empty:
            per_sleeve_oos[sleeve.label] = pd.Series(dtype=float)
            continue
        stitched = pd.concat(non_empty).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        per_sleeve_oos[sleeve.label] = stitched

    sleeve_returns_panel = pd.concat(per_sleeve_oos, axis=1).sort_index()
    sleeve_active_panel = sleeve_returns_panel.notna()

    # Dynamic equal-weight target weights. NaN return at a bar means
    # the sleeve was not in the universe yet (asset unlisted on the
    # date that fold's WF covers).
    n_active = sleeve_active_panel.sum(axis=1)
    target_weights = sleeve_active_panel.div(
        n_active.where(n_active > 0, 1.0), axis=0
    )
    # Cells where no sleeve is active → keep target weight at 0.
    target_weights = target_weights.where(sleeve_active_panel, 0.0)

    # Rebalance-on-universe-change turnover. Naively ``diff().abs().sum(axis=1)``
    # picks up both new-sleeve entries (weight 0 → 1/N) and existing-
    # sleeve trims (1/N_old → 1/N_new) at the same bar. But the
    # per-sleeve OOS returns are ALREADY net of one entry's worth of
    # cost (the sleeve's internal engine charges it). So if we also
    # bill the entry to the portfolio allocator, we double-count.
    #
    # The fix: subtract each sleeve's first-positive-weight contribution
    # from the portfolio-level turnover at that bar. What remains is
    # the trim cost on EXISTING sleeves (which the sleeves' internals
    # didn't bill — those sleeves assumed they were running on 100%
    # capital, not getting silently trimmed by an allocator) plus exit
    # costs when a sleeve leaves the universe.
    naive_diff = target_weights.diff().abs()
    naive_diff.iloc[0] = target_weights.iloc[0].abs()
    # Mask first-positive-weight cell per sleeve and zero it out.
    first_active = target_weights.gt(0).cumsum().eq(1) & target_weights.gt(0)
    entry_credit = target_weights.where(first_active, 0.0)
    portfolio_turnover_panel = (naive_diff - entry_credit).clip(lower=0.0)
    rebalance_turnover = portfolio_turnover_panel.sum(axis=1)
    rebalance_cost = rebalance_turnover * (fee_rate + slippage_rate)

    # Portfolio return at bar t uses weights set at the close of bar
    # t-1, applied to sleeve returns over [t-1, t]. The shift mirrors
    # the engine's one-bar execution lag and prevents look-ahead.
    shifted_weights = target_weights.shift(1).fillna(0.0)
    interior_nan = int(
        (sleeve_returns_panel.isna() & sleeve_returns_panel.notna().cummax())
        .to_numpy().sum()
    )
    if interior_nan:
        logger.warning(
            "ensemble: zero-filling %d sleeve-return value(s) missing "
            "after sleeve start — gaps are treated as flat days, which "
            "understates that sleeve's variance.",
            interior_nan,
        )
    sleeve_returns_filled = sleeve_returns_panel.fillna(0.0)
    portfolio_returns_gross = (shifted_weights * sleeve_returns_filled).sum(axis=1)
    portfolio_returns_net = portfolio_returns_gross - rebalance_cost

    portfolio_equity = initial_capital * (1.0 + portfolio_returns_net).cumprod()

    corr_matrix = sleeve_returns_panel.corr(min_periods=30)

    metrics = compute_portfolio_metrics(
        portfolio_returns_net=portfolio_returns_net,
        portfolio_returns_gross=portfolio_returns_gross,
        portfolio_equity=portfolio_equity,
        rebalance_cost=rebalance_cost,
        annualization_factor=annualization_factor,
        initial_capital=initial_capital,
        target_weights=target_weights,
        sleeve_active_panel=sleeve_active_panel,
    )

    # DSR on the portfolio's concatenated OOS return curve.
    from .dsr import deflated_sharpe_ratio

    cleaned_returns = portfolio_returns_net.dropna()
    T = len(cleaned_returns)
    sharpe_std_dev = 1.0 / np.sqrt(T) if T > 0 else 0.0
    portfolio_dsr = deflated_sharpe_ratio(
        returns=cleaned_returns,
        num_trials=num_trials_for_dsr,
        sharpe_std_dev=sharpe_std_dev,
    )

    return EnsembleResult(
        per_sleeve_detail=per_sleeve_detail,
        per_sleeve_oos_returns=per_sleeve_oos,
        sleeve_returns_panel=sleeve_returns_panel,
        sleeve_active_panel=sleeve_active_panel,
        target_weights=target_weights,
        portfolio_returns_gross=portfolio_returns_gross,
        portfolio_returns_net=portfolio_returns_net,
        portfolio_equity=portfolio_equity,
        rebalance_turnover=rebalance_turnover,
        rebalance_cost=rebalance_cost,
        correlation_matrix=corr_matrix,
        portfolio_metrics=metrics,
        portfolio_dsr=portfolio_dsr,
    )


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------


def sortino_ratio(returns: pd.Series, *, annualization_factor: int = 365) -> float:
    """Sortino = mean / std_of_negative_returns × sqrt(N).

    Uses the standard "downside deviation" denominator (RMS of returns
    below zero), not the full-distribution std. With no negative
    returns at all the function returns ``0.0`` rather than infinity,
    matching the conservative convention we use for Sharpe/Calmar
    elsewhere.
    """
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    neg = cleaned[cleaned < 0]
    if neg.empty:
        return 0.0
    downside = float(np.sqrt((neg ** 2).mean()))
    if downside == 0.0:
        return 0.0
    return float(cleaned.mean() / downside * np.sqrt(annualization_factor))


def _max_drawdown(equity: pd.Series) -> float:
    cleaned = equity.dropna()
    if cleaned.empty:
        return 0.0
    running_max = cleaned.cummax()
    dd = cleaned / running_max - 1.0
    return float(abs(dd.min()))


def compute_portfolio_metrics(
    *,
    portfolio_returns_net: pd.Series,
    portfolio_returns_gross: pd.Series,
    portfolio_equity: pd.Series,
    rebalance_cost: pd.Series,
    annualization_factor: int,
    initial_capital: float,
    target_weights: pd.DataFrame,
    sleeve_active_panel: pd.DataFrame,
) -> dict:
    """All the headline numbers in one dict."""
    cleaned = portfolio_returns_net.dropna()
    if cleaned.empty:
        return {k: 0.0 for k in (
            "total_return", "cagr", "sharpe", "sortino", "calmar",
            "max_drawdown", "time_in_market", "portfolio_turnover",
            "rebalance_cost_total", "cost_share_of_gross_pnl",
        )}

    total_return = float(portfolio_equity.iloc[-1] / initial_capital - 1.0)
    years = len(cleaned) / annualization_factor if annualization_factor > 0 else 0.0
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    std = float(cleaned.std())
    sharpe = float(cleaned.mean() / std * np.sqrt(annualization_factor)) if std > 0 else 0.0
    sortino_val = sortino_ratio(cleaned, annualization_factor=annualization_factor)

    max_dd = _max_drawdown(portfolio_equity)
    calmar = float(total_return / max_dd) if max_dd > 1e-9 else 0.0

    # Time in market: bar is "in market" when at least one sleeve is active.
    bars_in_market = sleeve_active_panel.any(axis=1).sum()
    total_bars = len(sleeve_active_panel)
    time_in_market = float(bars_in_market / total_bars) if total_bars > 0 else 0.0

    # Aggregate portfolio-level turnover and cost (in capital-fraction units).
    portfolio_turnover = float(target_weights.diff().abs().sum(axis=1).fillna(0.0).sum())
    rebalance_cost_total_frac = float(rebalance_cost.fillna(0.0).sum())

    # Costs as % of gross PnL. ``gross PnL`` = sum of absolute gross
    # returns × initial_capital — a rough scale; the ratio is what
    # matters, not the absolute level.
    gross_pnl = float(portfolio_returns_gross.fillna(0.0).abs().sum())
    cost_share = (
        rebalance_cost_total_frac / gross_pnl if gross_pnl > 1e-9 else 0.0
    )

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino_val,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "time_in_market": time_in_market,
        "portfolio_turnover": portfolio_turnover,
        "rebalance_cost_total": rebalance_cost_total_frac,
        "cost_share_of_gross_pnl": cost_share,
        "n_bars": len(cleaned),
    }


# ---------------------------------------------------------------------------
# Correlation summary
# ---------------------------------------------------------------------------


def correlation_summary(corr_matrix: pd.DataFrame) -> dict:
    """Boil the full corr matrix down to the headline numbers."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return {
            "n_sleeves": int(corr_matrix.shape[0]),
            "mean_pairwise_corr": 0.0,
            "median_pairwise_corr": 0.0,
            "max_pairwise_corr": 0.0,
            "min_pairwise_corr": 0.0,
            "high_corr_pair_count": 0,
        }
    # Take strict upper triangle to avoid duplicates and the diag.
    n = corr_matrix.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairs = corr_matrix.to_numpy()[upper_mask]
    pairs = pairs[np.isfinite(pairs)]
    if pairs.size == 0:
        return {
            "n_sleeves": n,
            "mean_pairwise_corr": 0.0,
            "median_pairwise_corr": 0.0,
            "max_pairwise_corr": 0.0,
            "min_pairwise_corr": 0.0,
            "high_corr_pair_count": 0,
        }
    return {
        "n_sleeves": n,
        "mean_pairwise_corr": float(np.mean(pairs)),
        "median_pairwise_corr": float(np.median(pairs)),
        "max_pairwise_corr": float(np.max(pairs)),
        "min_pairwise_corr": float(np.min(pairs)),
        "high_corr_pair_count": int(np.sum(pairs > 0.6)),
    }
