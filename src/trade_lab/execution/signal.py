"""Live signal computation for the deployable strategy.

The deployable configuration (per `findings/cluster_stability.md` +
`findings/han_28d_tsmom.md`):

* **Universe**: 7 majors (`BTC`, `ETH`, `BNB`, `SOL`, `ADA`, `XRP`,
  `DOGE`). Configurable via ``PaperConfig.basket``.
* **Aggregation**: equal-weight monthly-rebalanced market-basket
  index, built by :func:`build_crypto_market_index` — exactly the
  same code path the backtest used.
* **Strategy**: :class:`TimeSeriesMomentumStrategy` with
  ``lookbacks=(28, 60)``, ``sma_filter_periods=(200,)``, and
  ``use_vol_target=False``.

The signal value is a ladder ``{0, 0.5, 1.0}`` — *not* binary.
Half-agreement (one of the two lookbacks positive, one negative)
returns 0.5 as a real signal. The backtest's DSR 0.77 was computed
against this exact ladder, so the live executor must replicate it
pro-rata; rounding to binary would deploy a different strategy that
was never validated.

This module is **pure computation**. It fetches candles via a broker
function passed in (defaulting to CCXT through the broker), builds
the basket, runs TSMOM, and returns a :class:`SignalSnapshot`. No
orders are placed here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Sequence

import pandas as pd

from ..backtest.market_index import build_crypto_market_index
from ..strategies.tsmom import TimeSeriesMomentumStrategy
from .broker import Broker


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalSnapshot:
    """One live signal evaluation against the broker's latest data."""

    asof: pd.Timestamp                # the candle close timestamp used
    signal: float                     # ladder value in {0, 0.5, 1.0}
    basket_close: float               # synthetic-index close at asof
    asset_closes: dict[str, float]    # base symbol -> close at asof
    sma_gate_open: bool               # True if basket close > SMA(200)
    n_assets_in_basket: int           # active count at asof (excludes NaN-only)
    per_lookback_states: dict[int, int] = field(default_factory=dict)
    # Pre-gate {0,1} state per lookback. Averaged → ladder before the
    # SMA(200) gate is applied. Monitoring uses this to explain *why*
    # the ladder landed where it did on a given day.
    basket_close_tail: Optional[pd.Series] = None
    # Last N basket closes for the monitoring chart (basket vs SMA(200)).
    # None means "not computed" — older callers stay backward-compatible.


def compute_live_signal(
    broker: Broker,
    *,
    lookbacks: Sequence[int] = (28, 60),
    sma_filter_period: int = 200,
    candles_per_asset: int = 400,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    fetch_candles: Optional[
        Callable[[Broker, str, int], pd.DataFrame]
    ] = None,
) -> SignalSnapshot:
    """Run the deployable strategy on freshly-fetched candles.

    ``candles_per_asset`` must be large enough that every rolling
    window inside the strategy is fully warmed. With ``sma_filter_period
    = 200`` and ``lookbacks max = 60``, we need at least 200 days; 400
    gives a comfortable buffer for the monthly basket rebalance dates
    to line up cleanly.

    ``fetch_candles`` defaults to ``_fetch_recent_candles`` via the
    broker's underlying CCXT exchange. Tests pass a stub returning
    canned OHLCV frames.
    """
    fetch = fetch_candles or _fetch_recent_candles
    asset_candles: dict[str, pd.DataFrame] = {}
    quote = broker.config.quote_currency
    for sym in broker.config.basket:
        pair = f"{sym}/{quote}"
        try:
            df = fetch(broker, pair, candles_per_asset)
        except Exception as exc:
            # A single asset failing to return candles is a real
            # operational event. Don't pretend the basket is whole;
            # surface the missing asset to the caller.
            raise SignalComputationError(
                f"Could not fetch candles for {pair}: {exc}"
            ) from exc
        if df.empty:
            raise SignalComputationError(
                f"Empty candles returned for {pair}."
            )
        asset_candles[sym] = df

    basket = build_crypto_market_index(
        asset_candles,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    if basket.empty:
        raise SignalComputationError("Basket construction returned empty frame.")

    strategy = TimeSeriesMomentumStrategy(
        lookbacks=tuple(lookbacks),
        sma_filter_periods=(sma_filter_period,),
        use_vol_target=False,
    )
    signal_series = strategy.generate_signals(basket)
    if signal_series.empty:
        raise SignalComputationError("Strategy returned an empty signal series.")
    asof = signal_series.index[-1]
    signal_value = float(signal_series.iloc[-1])

    # Diagnostics: report whether the SMA(200) regime gate was open at
    # asof. This is useful in the dry-run log to explain why the
    # signal is zero on a particular day.
    basket_close = float(basket["close"].iloc[-1])
    sma = basket["close"].rolling(sma_filter_period).mean().iloc[-1]
    sma_gate_open = bool(pd.notna(sma) and basket_close > sma)

    # Pre-gate per-lookback states. Mirrors the strategy's internal
    # _tsmom_ensemble logic exactly: sign of pct_change(L) at the last
    # bar. The strategy then zeroes them out via the SMA(200) gate; we
    # expose the pre-gate values for diagnostic visibility.
    per_lookback_states: dict[int, int] = {}
    close_series = basket["close"]
    for L in lookbacks:
        past = close_series.pct_change(int(L)).iloc[-1]
        per_lookback_states[int(L)] = (
            1 if (pd.notna(past) and past > 0) else 0
        )

    return SignalSnapshot(
        asof=asof,
        signal=signal_value,
        basket_close=basket_close,
        asset_closes={
            sym: float(df["close"].iloc[-1]) for sym, df in asset_candles.items()
        },
        sma_gate_open=sma_gate_open,
        n_assets_in_basket=len(asset_candles),
        per_lookback_states=per_lookback_states,
        basket_close_tail=basket["close"].tail(100),
    )


def _fetch_recent_candles(
    broker: Broker, symbol: str, limit: int,
) -> pd.DataFrame:
    """Default candle fetcher: thin wrapper around ccxt.fetch_ohlcv.

    Returns a UTC-indexed OHLCV frame in the same shape the backtest
    used (columns ``open, high, low, close, volume``).
    """
    raw = broker.exchange.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
    if not raw:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).rename_axis("timestamp")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


class SignalComputationError(RuntimeError):
    """Raised when the live signal cannot be computed (missing candles,
    empty basket, strategy returned empty series). Distinct from
    ``BrokerError`` so callers can branch on data vs connectivity."""
