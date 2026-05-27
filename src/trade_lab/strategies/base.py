"""Strategy base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for trading strategies.

    Subclasses produce a signal series aligned with the input candles. Each
    signal value represents the *target* position for that bar:

    - ``1`` -> hold a long position
    - ``0`` -> flat

    The backtest engine shifts signals by one bar before applying them, so an
    entry decided at bar ``N`` is executed against bar ``N+1``. This prevents
    look-ahead bias even if a strategy accidentally references the current
    close when computing its signal.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Return a 0/1 target-position series indexed like ``candles``."""
