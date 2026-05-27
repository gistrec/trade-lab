"""Simple moving-average crossover strategy."""
from __future__ import annotations

import pandas as pd

from .base import Strategy


class SMACrossStrategy(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise."""

    name = "sma_cross"

    def __init__(self, fast_period: int = 20, slow_period: int = 100) -> None:
        if fast_period < 1 or slow_period < 1:
            raise ValueError("SMA periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be strictly less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"]
        fast = close.rolling(self.fast_period).mean()
        slow = close.rolling(self.slow_period).mean()
        signal = (fast > slow).astype(int)
        # Force flat while either SMA is still warming up.
        signal[fast.isna() | slow.isna()] = 0
        return signal
