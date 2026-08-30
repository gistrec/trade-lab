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

    The backtest engine shifts signals by one bar before applying them, so a
    position decided at bar ``N`` is first *held* through bar ``N+1``, which
    earns the move ``close[N] -> close[N+1]``. The implied fill is therefore
    ``close[N]`` — the close that produced the signal. Either way the
    strategy cannot trade on information it did not have: shifting prevents
    look-ahead bias even if a strategy accidentally references the current
    close when computing its signal.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Return a 0/1 target-position series indexed like ``candles``."""

    @property
    def required_warmup(self) -> int:
        """Longest rolling window the strategy needs before its signal is valid.

        The walk-forward runner sizes its pre-window candle slice from
        this. Any strategy with rolling state MUST override it: an
        under-sized warmup leaves indicators NaN over the head of every
        train and test window, and a regime filter that reads NaN as
        'closed' then flattens the variant for reasons that have nothing
        to do with its edge.
        """
        return 0
