"""Equity curve plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .engine import BacktestResult


def plot_equity_curve(
    result: BacktestResult,
    title: str = "Equity Curve",
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Render the equity curve and a drawdown panel for ``result``."""
    equity = result.equity
    if equity.empty:
        raise ValueError("Nothing to plot: equity curve is empty")

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_eq.plot(equity.index, equity.values, color="tab:blue", label="Equity")
    ax_eq.axhline(
        result.initial_capital,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Initial",
    )
    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(loc="best")

    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.4)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Time")
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=100)
    if show:
        plt.show()
    plt.close(fig)
