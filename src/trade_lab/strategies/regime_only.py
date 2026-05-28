"""Pure regime-filter strategy: long whenever close is above a long SMA."""
from __future__ import annotations

import pandas as pd

from .base import Strategy


class RegimeOnlyStrategy(Strategy):
    """Long-only when ``close > SMA(regime_period)``, flat otherwise.

    No crossover signal. No entry timing besides "is the price above its
    own long moving average". Useful as a simple bull/bear regime filter
    to compare against more involved strategies.

    Signals are produced from data available at bar ``N`` only; the engine
    shifts them by one bar before applying, so a flip at the close of N
    only affects bar ``N+1`` onward.
    """

    name = "regime_only"

    def __init__(self, regime_period: int = 200) -> None:
        if regime_period < 1:
            raise ValueError("regime_period must be a positive integer")
        self.regime_period = regime_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"]
        regime = close.rolling(self.regime_period).mean()
        signal = (close > regime).astype(int)
        # Force flat while the SMA is still warming up.
        signal[regime.isna()] = 0
        return signal
