"""SMA crossover gated by a long-term regime filter."""
from __future__ import annotations

import pandas as pd

from .base import Strategy


class RegimeSMACrossStrategy(Strategy):
    """SMA crossover that only takes longs while price is above a long-term SMA.

    The crossover (``fast > slow``) gives the entry timing; the regime filter
    (``close > regime``) forces the strategy flat during bear periods, so it
    sits in cash instead of buying every false bottom. Useful as a baseline
    for "trend-following but not in downtrends".
    """

    name = "regime_sma_cross"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 100,
        regime_period: int = 200,
    ) -> None:
        if fast_period < 1 or slow_period < 1 or regime_period < 1:
            raise ValueError("All SMA periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be strictly less than slow_period")
        if slow_period >= regime_period:
            raise ValueError(
                "slow_period must be strictly less than regime_period "
                "(the regime SMA is the slowest one)"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.regime_period = regime_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"]
        fast = close.rolling(self.fast_period).mean()
        slow = close.rolling(self.slow_period).mean()
        regime = close.rolling(self.regime_period).mean()
        crossover_long = fast > slow
        in_bull_regime = close > regime
        signal = (crossover_long & in_bull_regime).astype(int)
        # Force flat while any indicator is still warming up.
        signal[fast.isna() | slow.isna() | regime.isna()] = 0
        return signal
