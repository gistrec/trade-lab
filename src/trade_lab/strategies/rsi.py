"""RSI mean-reversion strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI on a close-price series."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class RSIMeanReversionStrategy(Strategy):
    """Long when RSI dips below ``lower``; exit when RSI rises above ``upper``."""

    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        lower: float = 30.0,
        upper: float = 70.0,
    ) -> None:
        if period < 2:
            raise ValueError("RSI period must be >= 2")
        if not 0 < lower < upper < 100:
            raise ValueError("Require 0 < lower < upper < 100")
        self.period = period
        self.lower = lower
        self.upper = upper

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        rsi = compute_rsi(candles["close"], self.period)
        # Mark explicit entries / exits, then forward-fill in between.
        raw = pd.Series(np.nan, index=rsi.index, dtype=float)
        raw[rsi < self.lower] = 1.0
        raw[rsi > self.upper] = 0.0
        return raw.ffill().fillna(0).astype(int)
