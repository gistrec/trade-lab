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
from typing import Callable, Optional, Sequence

import pandas as pd

from ..backtest.market_index import build_crypto_market_index_with_weights
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
    sma_value: Optional[float] = None
    # Actual SMA(sma_filter_period) value at asof. None when warm-up
    # data is insufficient. Used to show "basket is X% below the gate"
    # in monitoring, not just "gate CLOSED".
    per_lookback_returns: dict[int, float] = field(default_factory=dict)
    # Actual pct_change(L) at asof per lookback. Magnitude distinguishes
    # "barely positive" from "screaming uptrend" — the binary states
    # alone hide that distance to a flip.
    basket_weights: dict[str, float] = field(default_factory=dict)
    # Per-asset drifted weight at asof, taken from the basket index's
    # weight matrix (flat 1/N_active right after a monthly rebalance,
    # drifting between). The allocator sizes each asset to
    # ``signal × w_i × equity`` so live execution tracks the backtest's
    # monthly-rebalanced, between-rebalance-drifting weights instead of
    # resetting to flat 1/N every daily cycle (C3 / Option B). Empty for
    # callers/tests that construct the snapshot by hand.


def required_basket_bars(
    lookbacks: Sequence[int] = (28, 60),
    sma_filter_period: int = 200,
) -> int:
    """Minimum completed basket bars for a fully-warmed signal at asof.

    Exact pandas warm-up semantics (empirically pinned by tests, not
    guessed):

    * ``close.rolling(P).mean()`` (``min_periods`` defaults to the
      window) is first non-NaN once ``P`` bars exist — the SMA at the
      last bar needs ``P`` bars.
    * ``close.pct_change(L)`` at the last bar reaches back to
      ``close[-1 - L]`` — it needs ``L + 1`` bars.

    Below this depth the strategy does not error on its own:
    ``_sma_filter`` treats a NaN SMA as "regime gate closed", the
    ladder silently collapses to 0.0 even when every lookback state is
    1, and the allocator reads that as "liquidate the book".
    :func:`compute_live_signal` therefore refuses to compute a signal
    on a shallower basket (hard rule: missing candles raise).
    """
    if not lookbacks:
        raise ValueError("lookbacks must be non-empty")
    return max(int(sma_filter_period), max(int(L) for L in lookbacks) + 1)


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
    = 200`` and ``lookbacks max = 60``, we need at least 200 completed
    days plus the in-progress candle that gets dropped; 400 gives a
    comfortable buffer for the monthly basket rebalance dates to line
    up cleanly. Both the requested window and the *actual* basket depth
    are enforced — an unwarmed SMA(200) must raise
    :class:`SignalComputationError`, never silently read as "gate
    closed" (that would plan a full liquidation of the book).

    ``fetch_candles`` defaults to ``_fetch_recent_candles`` via the
    broker's underlying CCXT exchange. Tests pass a stub returning
    canned OHLCV frames.
    """
    required = required_basket_bars(lookbacks, sma_filter_period)
    # +1: a live fetch's last row is the in-progress daily candle, which
    # is dropped below — a request of exactly `required` bars could
    # never produce a warmed basket against a real exchange.
    if candles_per_asset < required + 1:
        raise SignalComputationError(
            f"candles_per_asset={candles_per_asset} cannot warm the signal: "
            f"need >= {required + 1} bars per asset "
            f"({required} completed bars for SMA({sma_filter_period}) / "
            f"lookbacks {tuple(int(L) for L in lookbacks)} warm-up, plus "
            f"the in-progress candle that is dropped)."
        )
    fetch = fetch_candles or _fetch_recent_candles
    asset_candles: dict[str, pd.DataFrame] = {}
    quote = broker.config.quote_currency
    # Exchanges return the currently-forming daily candle as the last
    # row (open timestamp = today's UTC midnight). The backtest decides
    # on the *completed* close of day t and trades on t+1
    # (signals.shift(1) in the engine); including the partial bar
    # shifts every lookback window by one bar and lets intraday noise
    # into the SMA gate. Drop it.
    cutoff = pd.Timestamp.now(tz="UTC").normalize()
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
        df = df[df.index < cutoff]
        if df.empty:
            raise SignalComputationError(
                f"Empty candles returned for {pair}."
            )
        asset_candles[sym] = df

    try:
        market = build_crypto_market_index_with_weights(
            asset_candles,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
    except ValueError as exc:
        # Data gaps (missing candles after listing) — same category as
        # a failed fetch: the basket is not whole, refuse loudly.
        raise SignalComputationError(str(exc)) from exc
    basket = market.index
    if basket.empty:
        raise SignalComputationError("Basket construction returned empty frame.")
    if len(basket) < required:
        # Truncated kline history (exchange wipe, API glitch, too-small
        # fetch window). The strategy would NOT error here on its own:
        # a NaN SMA reads as "regime gate closed", the ladder collapses
        # to 0.0 even with every lookback state at 1, and the allocator
        # turns that into a full liquidation plan. Fail loud instead.
        max_lookback = max(int(L) for L in lookbacks)
        raise SignalComputationError(
            f"Basket history too short to warm the signal: {len(basket)} "
            f"completed bars, need >= {required} "
            f"(SMA({sma_filter_period}) needs {sma_filter_period} bars; "
            f"max lookback {max_lookback} needs {max_lookback + 1}). "
            f"An unwarmed SMA would silently read as 'gate closed' "
            f"(signal=0) and plan a full liquidation — refusing."
        )

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

    # Per-asset drifted weight at asof (same code path as the backtest,
    # so the weights the executor sizes to are the ones the backtest
    # actually held). ``.loc[asof]`` fails loud if the weight matrix and
    # the signal series ever disagree on the last bar.
    weights_row = market.weights.loc[asof]
    basket_weights = {sym: float(weights_row[sym]) for sym in asset_candles}

    # Diagnostics: report whether the SMA(200) regime gate was open at
    # asof. This is useful in the dry-run log to explain why the
    # signal is zero on a particular day.
    basket_close = float(basket["close"].iloc[-1])
    sma = basket["close"].rolling(sma_filter_period).mean().iloc[-1]
    sma_gate_open = bool(pd.notna(sma) and basket_close > sma)

    # Pre-gate per-lookback states + actual returns. Mirrors the
    # strategy's internal _tsmom_ensemble logic exactly: sign of
    # pct_change(L) at the last bar. The strategy then zeroes them
    # out via the SMA(200) gate; we expose the pre-gate values for
    # diagnostic visibility.
    per_lookback_states: dict[int, int] = {}
    per_lookback_returns: dict[int, float] = {}
    close_series = basket["close"]
    for L in lookbacks:
        past = close_series.pct_change(int(L), fill_method=None).iloc[-1]
        ok = pd.notna(past)
        per_lookback_states[int(L)] = 1 if (ok and past > 0) else 0
        per_lookback_returns[int(L)] = float(past) if ok else 0.0

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
        sma_value=float(sma) if pd.notna(sma) else None,
        per_lookback_returns=per_lookback_returns,
        basket_weights=basket_weights,
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
