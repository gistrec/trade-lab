"""Simple position-sizing helpers.

The backtest engine accepts a ``position_size`` fraction in ``(0, 1]`` that
scales exposure when long. The helpers here are placeholders for future,
richer sizing rules (Kelly, volatility-targeted, equal-risk, etc.).
"""
from __future__ import annotations


def fixed_fraction(fraction: float = 1.0) -> float:
    """Return a fixed fraction of equity to deploy per trade."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return fraction
