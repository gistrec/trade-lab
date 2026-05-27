"""Equity curve plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .engine import BacktestResult, execution_bars


def plot_equity_curve(
    result: BacktestResult,
    candles: pd.DataFrame | None = None,
    title: str = "Equity Curve",
    save_path: Path | str | None = None,
    show: bool = False,
    show_trades: bool = False,
) -> None:
    """Render the equity curve, drawdown, and (optionally) a price panel
    with buy/sell markers placed on the actual execution candles.

    When ``show_trades`` is enabled, ``candles`` must be supplied — the
    close prices come from there. Buy / sell markers are derived from
    position transitions (see :func:`execution_bars`), not from the
    engine's :class:`Trade` objects, so they fall on the *execution*
    candle (one bar after the signal candle, by construction of the
    one-bar shift).
    """
    equity = result.equity
    if equity.empty:
        raise ValueError("Nothing to plot: equity curve is empty")
    if show_trades and candles is None:
        raise ValueError("candles must be provided when show_trades=True")

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    if show_trades:
        fig, (ax_price, ax_eq, ax_dd) = plt.subplots(
            3, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1]},
        )
        _draw_price_panel(ax_price, candles, result, title)
        eq_title: str | None = None
    else:
        fig, (ax_eq, ax_dd) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        eq_title = title

    _draw_equity_panel(ax_eq, result, title=eq_title)
    _draw_drawdown_panel(ax_dd, drawdown)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=100)
    if show:
        plt.show()
    plt.close(fig)


def _draw_price_panel(
    ax,
    candles: pd.DataFrame,
    result: BacktestResult,
    title: str,
) -> None:
    close = candles["close"]
    ax.plot(close.index, close.values, color="tab:gray", linewidth=1.0, label="Close")

    entries, exits = execution_bars(result.positions)
    if entries:
        ax.scatter(
            [close.index[i] for i in entries],
            [close.iloc[i] for i in entries],
            marker="^", color="tab:green", s=70, zorder=3, label="Buy",
        )
    if exits:
        ax.scatter(
            [close.index[i] for i in exits],
            [close.iloc[i] for i in exits],
            marker="v", color="tab:red", s=70, zorder=3, label="Sell",
        )

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _draw_equity_panel(ax, result: BacktestResult, title: str | None) -> None:
    equity = result.equity
    ax.plot(
        equity.index, equity.values,
        color="tab:blue", linewidth=1.5, label="Strategy",
    )
    bh_equity = result.buy_and_hold_equity
    if not bh_equity.empty:
        ax.plot(
            bh_equity.index, bh_equity.values,
            color="tab:orange", linestyle="--", linewidth=1.2, label="Buy & hold",
        )
    ax.axhline(
        result.initial_capital,
        color="gray", linestyle=":", linewidth=0.8, label="Initial",
    )
    if title:
        ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _draw_drawdown_panel(ax, drawdown: pd.Series) -> None:
    ax.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.4)
    ax.set_ylabel("Strategy DD")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
