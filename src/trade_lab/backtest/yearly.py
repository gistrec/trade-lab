"""Per-calendar-year validation of a fixed set of strategies.

No parameter sweep, no optimization. Each strategy is run once on the
full candle history (so indicators have proper warmup) and the equity /
position / trade streams are sliced by calendar year for per-year
metrics. Aggregates are then computed across years per strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import pandas as pd

from ..strategies.base import Strategy
from ..strategies.regime_only import RegimeOnlyStrategy
from ..strategies.regime_sma_cross import RegimeSMACrossStrategy
from ..strategies.sma_cross import SMACrossStrategy
from .engine import run_backtest
from .metrics import (
    VERDICT_LOWER_RETURN_LOWER_DD,
    VERDICT_OUTPERFORMS_BH,
    VERDICT_UNDERPERFORMS_BH,
    _max_drawdown,
)


VERDICT_BUY_AND_HOLD = "BUY_AND_HOLD"

_MEANINGFUL_DD_DIFFERENCE = 0.02

YEARLY_COLUMNS = [
    "year",
    "strategy",
    "return_pct",
    "buy_and_hold_return_pct",
    "max_drawdown_pct",
    "buy_and_hold_max_drawdown_pct",
    "exposure_pct",
    "num_trades",
    "fees_paid",
    "verdict",
]

YEARLY_AGGREGATE_COLUMNS = [
    "strategy",
    "total_years",
    "avg_annual_return",
    "median_annual_return",
    "best_year_return",
    "worst_year_return",
    "years_outperforming_bh",
    "years_lower_dd_than_bh",
    "avg_exposure",
]


@dataclass(frozen=True)
class FixedStrategySpec:
    """A fixed-parameter strategy plus a display label.

    ``factory=None`` is the special ``buy_and_hold`` case: the row is
    synthesized from close prices alone (no fees, full exposure).
    """

    label: str
    factory: Optional[Callable[[], Strategy]]


DEFAULT_FIXED_STRATEGIES: Sequence[FixedStrategySpec] = (
    FixedStrategySpec("buy_and_hold", None),
    FixedStrategySpec("regime_only_200", lambda: RegimeOnlyStrategy(200)),
    FixedStrategySpec("regime_only_300", lambda: RegimeOnlyStrategy(300)),
    FixedStrategySpec("sma_cross_20_100", lambda: SMACrossStrategy(20, 100)),
    FixedStrategySpec(
        "regime_sma_cross_20_100_200",
        lambda: RegimeSMACrossStrategy(20, 100, 200),
    ),
    FixedStrategySpec("golden_cross_50_200", lambda: SMACrossStrategy(50, 200)),
)


def run_yearly_validation(
    candles: pd.DataFrame,
    strategies: Sequence[FixedStrategySpec] = DEFAULT_FIXED_STRATEGIES,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Evaluate each strategy on the full history, slice metrics by year.

    Returns a long-format DataFrame with one row per ``(year, strategy)``.
    The non-``buy_and_hold`` strategies share the same indicator-warmup
    benefit because each is run once over the whole window before metrics
    are bucketed.
    """
    if candles.empty:
        return pd.DataFrame(columns=YEARLY_COLUMNS)

    years = sorted({ts.year for ts in candles.index})
    rows: list[dict] = []

    for spec in strategies:
        if spec.factory is None:
            rows.extend(_yearly_rows_for_buy_and_hold(
                candles, years, initial_capital,
                fee_rate=fee_rate, slippage_rate=slippage_rate,
            ))
            continue

        result = run_backtest(
            candles,
            spec.factory(),
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
        )

        for year in years:
            row = _yearly_row_for_strategy(
                spec.label, result, candles, year, initial_capital
            )
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows, columns=YEARLY_COLUMNS)


def aggregate_yearly_results(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-(year, strategy) results to one summary row per strategy."""
    if yearly_df.empty:
        return pd.DataFrame(columns=YEARLY_AGGREGATE_COLUMNS)

    aggregates: list[dict] = []
    # `sort=False` preserves the original strategy order from yearly_df.
    for strategy, group in yearly_df.groupby("strategy", sort=False):
        bh_outperform = int(
            (group["return_pct"] > group["buy_and_hold_return_pct"]).sum()
        )
        lower_dd = int(
            (group["max_drawdown_pct"] < group["buy_and_hold_max_drawdown_pct"]).sum()
        )
        aggregates.append(
            {
                "strategy": strategy,
                "total_years": len(group),
                "avg_annual_return": float(group["return_pct"].mean()),
                "median_annual_return": float(group["return_pct"].median()),
                "best_year_return": float(group["return_pct"].max()),
                "worst_year_return": float(group["return_pct"].min()),
                "years_outperforming_bh": bh_outperform,
                "years_lower_dd_than_bh": lower_dd,
                "avg_exposure": float(group["exposure_pct"].mean()),
            }
        )

    return pd.DataFrame(aggregates, columns=YEARLY_AGGREGATE_COLUMNS)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _yearly_rows_for_buy_and_hold(
    candles: pd.DataFrame,
    years: List[int],
    initial_capital: float,
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> list[dict]:
    """Build the buy-and-hold rows directly from close prices.

    Each year's row treats B&H as "enter on the first bar of the year,
    hold through year-end". Entry cost (one round of fee + slippage)
    is applied to ``initial_capital`` so the row is comparable to a
    strategy that also entered on the same bar.
    """
    from .engine import buy_and_hold_with_costs

    close = candles["close"]
    rows = []
    for year in years:
        year_close = close[close.index.year == year]
        if year_close.empty:
            continue
        year_equity, year_return = buy_and_hold_with_costs(
            year_close, initial_capital=initial_capital,
            fee_rate=fee_rate, slippage_rate=slippage_rate,
        )
        year_dd = _max_drawdown(year_equity)
        # fees_paid is fees-only, matching the strategy rows' semantic
        # (sum of Trade.fees_paid, which excludes slippage). Folding
        # slippage in here made one column carry two meanings. Slippage
        # is still reflected in return_pct via buy_and_hold_with_costs.
        entry_cost_paid = float(initial_capital * fee_rate)
        rows.append(
            {
                "year": year,
                "strategy": "buy_and_hold",
                "return_pct": year_return,
                "buy_and_hold_return_pct": year_return,
                "max_drawdown_pct": year_dd,
                "buy_and_hold_max_drawdown_pct": year_dd,
                "exposure_pct": 1.0,
                "num_trades": 0,
                "fees_paid": entry_cost_paid,
                "verdict": VERDICT_BUY_AND_HOLD,
            }
        )
    return rows


def _yearly_row_for_strategy(
    label: str,
    result,
    candles: pd.DataFrame,
    year: int,
    initial_capital: float,
) -> Optional[dict]:
    """Slice a full-history result by year and build the row."""
    year_mask = candles.index.year == year
    if not year_mask.any():
        return None

    year_index = candles.index[year_mask]
    year_equity = result.equity.loc[year_index]
    year_positions = result.positions.loc[year_index]
    year_close = candles["close"].loc[year_index]

    if year_equity.empty:
        return None

    # Return for the year over the WITHIN-year window: from equity at the
    # first bar of the year to the last. This mirrors the B&H reference
    # below (buy_and_hold_with_costs enters at the year's first-bar close),
    # so both sides of the verdict span the same bars. Basing it on the
    # prior year's last-bar equity pulled the first-bar-of-year move into
    # the strategy return while B&H excluded it — a one-bar window mismatch
    # at the year boundary that could flip a borderline verdict.
    start_equity = float(year_equity.iloc[0])
    end_equity = float(year_equity.iloc[-1])
    year_return = (end_equity / start_equity) - 1 if start_equity > 0 else 0.0

    # Drawdown computed within the year only — each year stands alone.
    year_max_dd = _max_drawdown(year_equity)

    # Buy & hold reference for the year, with symmetric entry cost so
    # the per-year comparison mirrors the per-year strategy run.
    from .engine import buy_and_hold_with_costs

    year_bh_equity, year_bh_return = buy_and_hold_with_costs(
        year_close, initial_capital=initial_capital,
        fee_rate=result.fee_rate, slippage_rate=result.slippage_rate,
    )
    year_bh_max_dd = _max_drawdown(year_bh_equity)

    # Exposure: bars where positions are non-zero.
    year_exposure = float((year_positions > 0).mean())

    # Trades that completed within this year.
    year_trades = [
        t for t in result.trades
        if t.exit_signal_time is not None
        and pd.Timestamp(t.exit_time).year == year
    ]
    year_fees = float(sum(t.fees_paid for t in year_trades))

    verdict = _verdict_for_year(
        year_return, year_max_dd, year_bh_return, year_bh_max_dd
    )

    return {
        "year": year,
        "strategy": label,
        "return_pct": year_return,
        "buy_and_hold_return_pct": year_bh_return,
        "max_drawdown_pct": year_max_dd,
        "buy_and_hold_max_drawdown_pct": year_bh_max_dd,
        "exposure_pct": year_exposure,
        "num_trades": len(year_trades),
        "fees_paid": year_fees,
        "verdict": verdict,
    }


def _verdict_for_year(
    strat_return: float, strat_dd: float,
    bh_return: float, bh_dd: float,
) -> str:
    """Same logic as :func:`benchmark_verdict` but inlined for per-year use."""
    if strat_return > bh_return and strat_dd <= bh_dd:
        return VERDICT_OUTPERFORMS_BH
    if (
        strat_return < bh_return
        and strat_dd < bh_dd - _MEANINGFUL_DD_DIFFERENCE
    ):
        return VERDICT_LOWER_RETURN_LOWER_DD
    return VERDICT_UNDERPERFORMS_BH
