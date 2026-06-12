"""Subperiod comparison report across strategies and assets.

Produces a long-format table of (asset, strategy, subperiod) -> metrics
for robustness inspection. Not a tuning tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.storage import filter_candles_by_date
from ..strategies.base import Strategy
from ..strategies.donchian_trend import DonchianTrendEnsembleStrategy
from ..strategies.pma_ratio import PriceMaRatioStrategy
from ..strategies.sma_cross import SMACrossStrategy
from ..strategies.tsmom import TimeSeriesMomentumStrategy
from .engine import buy_and_hold_with_costs, run_backtest
from .metrics import _max_drawdown, compute_metrics


@dataclass(frozen=True)
class StrategySpec:
    """A labelled fixed-parameter strategy factory.

    ``factory=None`` means the buy-and-hold baseline: a synthesized
    no-cost long-only path computed directly from close prices.
    """
    label: str
    factory: Optional[Callable[[], Strategy]]


@dataclass(frozen=True)
class Subperiod:
    label: str
    start_date: Optional[str]
    end_date: Optional[str]


DEFAULT_COMPARISON_STRATEGIES: Sequence[StrategySpec] = (
    StrategySpec("buy_and_hold", None),
    StrategySpec("sma_cross_20_100", lambda: SMACrossStrategy(20, 100)),
    StrategySpec(
        "donchian_trend_rb0",
        lambda: DonchianTrendEnsembleStrategy(rebalance_threshold=0.0),
    ),
    StrategySpec(
        "donchian_trend_rb005",
        lambda: DonchianTrendEnsembleStrategy(rebalance_threshold=0.05),
    ),
    # TSMOM: sign-of-trailing-return ensemble over 1, 3, 6, 12 months.
    # Moskowitz et al. 2012 + Liu & Tsyvinski 2021. Uses the same SMA(200)
    # regime filter + vol-targeting layer so it stays comparable to the
    # other trend-following entries.
    StrategySpec(
        "tsmom_1_3_6_12m",
        lambda: TimeSeriesMomentumStrategy(),
    ),
    # P/MA ratio ensemble. Detzel et al. 2021 (Financial Management).
    # ``close > SMA(k)`` votes over k in {5, 10, 20, 50, 100}.
    StrategySpec(
        "pma_ratio_ensemble",
        lambda: PriceMaRatioStrategy(),
    ),
)


DEFAULT_SUBPERIODS: Sequence[Subperiod] = (
    Subperiod("2018", "2018-01-01", "2018-12-31"),
    Subperiod("2019", "2019-01-01", "2019-12-31"),
    Subperiod("2020-2021", "2020-01-01", "2021-12-31"),
    Subperiod("2022", "2022-01-01", "2022-12-31"),
    Subperiod("2023-2025", "2023-01-01", "2025-12-31"),
    Subperiod("full", None, None),
)


COMPARISON_COLUMNS = [
    "asset",
    "strategy",
    "period",
    "period_start",
    "period_end",
    "bars",
    "total_return_pct",
    "cagr_pct",
    "max_drawdown_pct",
    "sharpe",
    "exposure_pct",
    "num_trades",
    "total_fees",
    "total_slippage",
    "turnover",
]


def run_comparison_report(
    asset_candles: Mapping[str, pd.DataFrame],
    strategies: Sequence[StrategySpec] = DEFAULT_COMPARISON_STRATEGIES,
    subperiods: Sequence[Subperiod] = DEFAULT_SUBPERIODS,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """Run every (asset, strategy, subperiod) combination and return one
    row per cell."""
    rows: list[dict] = []
    for asset, candles in asset_candles.items():
        if candles.empty:
            continue
        for period in subperiods:
            sliced = filter_candles_by_date(
                candles, start_date=period.start_date, end_date=period.end_date
            )
            if sliced.empty:
                continue
            for spec in strategies:
                rows.append(
                    _evaluate(
                        asset=asset,
                        period=period,
                        candles=sliced,
                        spec=spec,
                        initial_capital=initial_capital,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        annualization_factor=annualization_factor,
                    )
                )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def render_comparison_markdown(detail: pd.DataFrame) -> str:
    """Render a compact ``return / DD`` pivot, one table per asset."""
    if detail.empty:
        return "# Comparison report\n\n_No data._\n"

    period_order = list(dict.fromkeys(detail["period"].tolist()))
    strategy_order = list(dict.fromkeys(detail["strategy"].tolist()))

    parts = ["# Strategy comparison\n",
             "Cells are formatted as `return / max DD` (both as percentages). "
             "Negative returns are shown explicitly. Full detail in the CSV.\n"]
    for asset in dict.fromkeys(detail["asset"].tolist()):
        parts.append(f"\n## {asset}\n")
        header = "| strategy | " + " | ".join(period_order) + " |"
        sep = "|---" * (1 + len(period_order)) + "|"
        parts.append(header)
        parts.append(sep)
        for strategy in strategy_order:
            cells = [strategy]
            for period in period_order:
                row = detail[
                    (detail["asset"] == asset)
                    & (detail["strategy"] == strategy)
                    & (detail["period"] == period)
                ]
                if row.empty:
                    cells.append("—")
                    continue
                ret = row.iloc[0]["total_return_pct"]
                dd = row.iloc[0]["max_drawdown_pct"]
                cells.append(f"{ret:+.1%} / {dd:.1%}")
            parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _evaluate(
    asset: str,
    period: Subperiod,
    candles: pd.DataFrame,
    spec: StrategySpec,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    annualization_factor: int,
) -> dict:
    bars = len(candles)
    period_start = candles.index[0]
    period_end = candles.index[-1]

    base = {
        "asset": asset,
        "strategy": spec.label,
        "period": period.label,
        "period_start": period_start,
        "period_end": period_end,
        "bars": bars,
    }

    if spec.factory is None:
        return {**base, **_buy_and_hold_metrics(
            candles, initial_capital, annualization_factor,
            fee_rate=fee_rate, slippage_rate=slippage_rate,
        )}

    strategy = spec.factory()
    result = run_backtest(
        candles,
        strategy,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    m = compute_metrics(result)
    turnover = float(result.positions.diff().abs().sum())
    return {
        **base,
        "total_return_pct": m.total_return,
        "cagr_pct": _cagr(result.equity, bars, annualization_factor),
        "max_drawdown_pct": m.max_drawdown,
        "sharpe": _sharpe(result.equity, annualization_factor),
        "exposure_pct": m.exposure_pct,
        "num_trades": m.num_trades,
        "total_fees": m.total_fees,
        "total_slippage": m.total_slippage,
        "turnover": turnover,
    }


def _buy_and_hold_metrics(
    candles: pd.DataFrame, initial_capital: float, annualization_factor: int,
    *, fee_rate: float = 0.0, slippage_rate: float = 0.0,
) -> dict:
    """Buy-and-hold cell with **one entry round of costs** applied.

    Same semantics as :func:`engine.buy_and_hold_with_costs`: B&H pays
    the same fee+slippage on bar 1 as any strategy entering an
    equal-sized long; it does NOT pay an exit fee at window end (open
    position is mark-to-market). Setting both rates to 0 reproduces
    the academic pre-cost curve.
    """
    close = candles["close"].astype(float)
    equity, total_return = buy_and_hold_with_costs(
        close, initial_capital=initial_capital,
        fee_rate=fee_rate, slippage_rate=slippage_rate,
    )
    # CAGR derived from the gross-of-entry-cost initial_capital — the
    # equity series after entry costs would understate CAGR slightly
    # because its starting value is already reduced. We want the
    # "growth of my $10k" not "growth of my $9985".
    years = (len(candles) / annualization_factor) if annualization_factor > 0 else 0.0
    if years > 0 and (1.0 + total_return) > 0:
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0
    # Charge the entry cost as a dollar amount on initial_capital so
    # the `total_fees` / `total_slippage` columns are non-zero and
    # comparable to the strategies' columns. Turnover for B&H = 1
    # round-trip side (the entry), exactly matching the engine's
    # ``turnover.iloc[0] = abs(positions.iloc[0])`` convention.
    entry_fee = float(initial_capital * fee_rate)
    entry_slippage = float(initial_capital * slippage_rate)
    return {
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "max_drawdown_pct": _max_drawdown(equity),
        "sharpe": _sharpe(equity, annualization_factor),
        "exposure_pct": 1.0,
        "num_trades": 0,
        "total_fees": entry_fee,
        "total_slippage": entry_slippage,
        "turnover": 1.0,
    }


def _cagr(equity: pd.Series, bars: int, annualization_factor: int) -> float:
    if equity.empty or bars <= 0:
        return 0.0
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    if initial <= 0 or final <= 0:
        return 0.0
    years = bars / annualization_factor
    if years <= 0:
        return 0.0
    return (final / initial) ** (1.0 / years) - 1.0


def _sharpe(equity: pd.Series, annualization_factor: int) -> float:
    if equity.empty:
        return 0.0
    returns = equity.pct_change(fill_method=None).dropna()
    if returns.empty:
        return 0.0
    std = float(returns.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualization_factor))
