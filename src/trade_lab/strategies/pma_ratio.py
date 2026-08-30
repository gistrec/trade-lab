"""Price-to-moving-average ratio ensemble (Detzel et al. 2021).

Detzel, Liu, Strauss, Zhou, Zhu (2021). *Learning and predictability via
technical analysis: Evidence from Bitcoin and stocks with hard-to-value
fundamentals*. **Financial Management**.

The paper models the price-to-MA ratio as a rational-learning signal in
assets with hard-to-value fundamentals (a category that includes
Bitcoin) and shows that an ensemble of ``close / SMA(k)`` ratios over
``k in {5, 10, 20, 50, 100}`` predicts daily Bitcoin returns both
in- and out-of-sample, with positive alpha vs HODL.

This long-only implementation operationalizes the ratios as discrete
"is the ratio above 1?" votes and averages them, producing a smooth
ladder on ``{0, 1/n, 2/n, ..., 1}``. We deliberately *do not* read
quantitative magnitudes off the ratios — that would invite an
overfitted scaling and the literature warns that the paper's evidence
is for direction, not for size.

The strategy shares two optional layers with the rest of this
repository's trend stack (regime filter + vol targeting) so its
contribution to a strategy comparison is the *signal*, not a different
sizing scheme.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ._trend_layers import (
    apply_rebalance_band,
    sma_filter,
    validate_trend_params,
    vol_weight,
)
from .base import Strategy
from .donchian_trend import _coerce_bool, _coerce_int_sequence


class PriceMaRatioStrategy(Strategy):
    """Ensemble of ``close > SMA(k)`` votes over a panel of windows."""

    name = "pma_ratio"

    def __init__(
        self,
        ma_periods: Iterable[int] | str = (5, 10, 20, 50, 100),
        sma_filter_periods: Iterable[int] | str = (),
        vol_lookback: int = 30,
        annual_vol_target: float = 0.25,
        annualization_factor: int = 365,
        max_position_size: float = 1.0,
        rebalance_threshold: float = 0.05,
        use_vol_target: bool = True,
    ) -> None:
        self.ma_periods = _coerce_int_sequence(ma_periods, "ma_periods")
        self.sma_filter_periods = (
            _coerce_int_sequence(sma_filter_periods, "sma_filter_periods")
            if sma_filter_periods
            else ()
        )
        validate_trend_params(
            vol_lookback,
            annual_vol_target,
            annualization_factor,
            max_position_size,
            rebalance_threshold,
        )

        self.vol_lookback = int(vol_lookback)
        self.annual_vol_target = float(annual_vol_target)
        self.annualization_factor = int(annualization_factor)
        self.max_position_size = float(max_position_size)
        self.rebalance_threshold = float(rebalance_threshold)
        self.use_vol_target = _coerce_bool(use_vol_target, "use_vol_target")

    @property
    def required_warmup(self) -> int:
        return max([*self.ma_periods, *self.sma_filter_periods,
                    self.vol_lookback])

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"].astype(float)

        raw_signal = self._pma_ensemble(close)
        if self.sma_filter_periods:
            raw_signal = raw_signal.where(
                sma_filter(close, self.sma_filter_periods), 0.0
            )

        if not self.use_vol_target:
            # Pass the {0, 1/n, ..., 1} P/MA-vote ladder straight to
            # the engine; the rebalance band has nothing to suppress
            # on a discrete ladder either.
            return raw_signal.clip(lower=0.0, upper=self.max_position_size).fillna(0.0)

        weight = vol_weight(
            close,
            self.vol_lookback,
            self.annual_vol_target,
            self.annualization_factor,
        )
        target_position = (raw_signal * weight).clip(
            lower=0.0, upper=self.max_position_size
        ).fillna(0.0)
        return apply_rebalance_band(target_position, self.rebalance_threshold)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _pma_ensemble(self, close: pd.Series) -> pd.Series:
        components: list[pd.Series] = []
        for period in self.ma_periods:
            sma = close.rolling(period).mean()
            state = (close > sma).astype(float)
            # Treat warm-up (SMA NaN) as "flat", never as "long".
            state[sma.isna()] = 0.0
            components.append(state)
        stacked = pd.concat(components, axis=1)
        return stacked.mean(axis=1)
