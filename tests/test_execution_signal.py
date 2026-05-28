"""Tests for the live-signal computation pipeline.

No network, no real CCXT. Signal computation is deterministic given
candles; we feed canned OHLCV frames through the broker stub and
assert the ladder value (0 / 0.5 / 1.0) and gate diagnostics line up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.signal import (
    SignalComputationError, SignalSnapshot, compute_live_signal,
)


def _config(basket=("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")):
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


class _ExchangeStub:
    """Minimal CCXT stand-in that only needs to satisfy Broker plumbing."""
    id = "stub"
    def set_sandbox_mode(self, enabled): pass
    def fetch_balance(self): return {"USDT": {"free": 0, "used": 0, "total": 0}}
    def fetch_ticker(self, symbol): return {"last": 1.0, "close": 1.0}
    def fetch_status(self): return {"status": "ok"}
    def load_markets(self, reload=False): return {}


def _ohlcv(closes, start="2023-01-01"):
    """Build an OHLCV frame indexed by daily UTC timestamps."""
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": 1.0},
        index=idx,
    )


def _candles_factory(symbol_to_closes):
    """Return a fetch_candles function that emits canned frames."""
    def _fetch(broker, pair, limit):
        sym = pair.split("/")[0]
        closes = symbol_to_closes.get(sym)
        if closes is None:
            raise RuntimeError(f"no canned candles for {sym}")
        return _ohlcv(closes).iloc[-limit:]
    return _fetch


def test_signal_full_long_when_both_lookbacks_positive():
    """7 assets all monotonically up → basket up → both 28d and 60d
    look-back returns positive → signal = 1.0."""
    n = 500
    closes = (100 + np.linspace(0, 200, n)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch)
    assert isinstance(snap, SignalSnapshot)
    assert snap.signal == 1.0
    assert snap.sma_gate_open is True


def test_signal_zero_when_basket_below_sma200():
    """All assets in a clean downtrend → basket falls under its own
    SMA(200) → regime gate closes → signal = 0."""
    n = 500
    closes = np.linspace(200, 100, n).tolist()  # clean downtrend
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch)
    assert snap.signal == 0.0
    assert snap.sma_gate_open is False


def test_signal_returns_ladder_only():
    """Across a random walk window, the signal must NEVER take a
    continuous value — only ``{0, 0.5, 1.0}``."""
    rng = np.random.default_rng(0)
    closes = (100 + rng.normal(0.5, 1.5, 500).cumsum()).clip(min=10).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch)
    assert snap.signal in (0.0, 0.5, 1.0)


def test_signal_records_diagnostics():
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch)
    assert snap.n_assets_in_basket == 7
    assert set(snap.asset_closes.keys()) == {
        "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE",
    }
    assert snap.basket_close > 0
    assert pd.Timestamp(snap.asof).tzinfo is not None  # tz-aware UTC


def test_signal_records_sma_value_and_returns():
    """sma_value is the SMA(200) at asof; per_lookback_returns gives the
    actual pct_change magnitude per lookback (not just the binary state)."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch)
    assert snap.sma_value is not None
    assert snap.sma_value > 0
    assert set(snap.per_lookback_returns.keys()) == {28, 60}
    # Clean uptrend → both lookback returns positive AND >> 0.
    assert snap.per_lookback_returns[28] > 0
    assert snap.per_lookback_returns[60] > 0


def test_signal_returns_match_states():
    """The sign of per_lookback_returns must match per_lookback_states
    bit-for-bit; otherwise the journal would surface contradictory
    diagnostics to the monitor."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch)
    for L, ret in snap.per_lookback_returns.items():
        expected_state = 1 if ret > 0 else 0
        assert snap.per_lookback_states[L] == expected_state


def test_signal_missing_asset_raises():
    """If a single basket asset returns no candles, the signal computation
    refuses to proceed — we never want the basket to silently shrink."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    # BTC missing on purpose.
    fetch = _candles_factory({s: closes for s in
        ("ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match="BTC"):
        compute_live_signal(broker, fetch_candles=fetch)


def test_signal_empty_candles_raises():
    def empty_fetch(broker, pair, limit):
        return _ohlcv([])
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match="Empty candles"):
        compute_live_signal(broker, fetch_candles=empty_fetch)
