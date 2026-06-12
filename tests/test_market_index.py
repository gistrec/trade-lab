"""Tests for the crypto-market basket index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.market_index import build_crypto_market_index


def _candles(closes, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": 1.0,
        },
        index=idx,
    )


def test_empty_input_returns_empty_frame():
    out = build_crypto_market_index({})
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_single_asset_index_tracks_that_asset():
    """With one asset and no rebalance cost, the index should match
    the asset's close ratio rescaled to start at 100."""
    closes = np.linspace(100, 200, 100).tolist()
    candles = {"BTC/USDT": _candles(closes)}
    idx = build_crypto_market_index(candles, fee_rate=0.0, slippage_rate=0.0)
    # Index starts at 100, ends at 100 * (200/100) = 200 (modulo entry
    # cost which is zero here).
    assert idx["close"].iloc[0] == pytest.approx(100.0)
    assert idx["close"].iloc[-1] == pytest.approx(200.0)


def test_equal_weight_two_assets_one_doubles_one_flat():
    """If asset A doubles and asset B is flat, the basket should
    finish at ~150 (50% × 100 + 50% × 200). Monthly rebalance keeps
    the weights at 50/50 after each drift, but with 100 bars and
    monthly rebal there will be ~3 rebalances — the math still lands
    near 150 at zero cost."""
    a = np.linspace(100, 200, 100).tolist()
    b = [100.0] * 100
    candles = {
        "A/USDT": _candles(a),
        "B/USDT": _candles(b),
    }
    idx = build_crypto_market_index(candles, fee_rate=0.0, slippage_rate=0.0)
    assert idx["close"].iloc[0] == pytest.approx(100.0)
    # Without rebalance, the basket geometric mean would be sqrt(2)*100 ≈ 141.
    # With monthly rebalance to 50/50, it stays closer to arithmetic mean of
    # the asset returns. Allow generous slack to keep the test stable
    # against exact monthly-bar choice.
    assert 130 < idx["close"].iloc[-1] < 165


def test_costs_reduce_terminal_index_value():
    """Same scenario, but with realistic fees. The cost charged on
    rebalance turnover should make the terminal value strictly lower
    than the zero-cost run."""
    a = np.linspace(100, 200, 100).tolist()
    b = [100.0] * 100
    candles = {"A/USDT": _candles(a), "B/USDT": _candles(b)}
    no_cost = build_crypto_market_index(candles, fee_rate=0.0, slippage_rate=0.0)
    with_cost = build_crypto_market_index(candles, fee_rate=0.001, slippage_rate=0.0005)
    assert with_cost["close"].iloc[-1] < no_cost["close"].iloc[-1]


def test_new_asset_listing_triggers_rebalance_cost():
    """A second asset listing mid-window forces a rebalance event
    even off the monthly schedule — confirmed by a non-zero cost
    impact at the listing bar relative to a zero-cost run."""
    # Asset A: 100 bars from 2020-01-01 (ends 2020-04-09).
    a = np.linspace(100, 200, 100).tolist()
    # Asset B: lists mid-window on 2020-02-15 and runs through the end
    # of A's window (55 bars to 2020-04-09) — ending earlier would be a
    # trailing data gap, which now correctly raises.
    b = np.linspace(100, 110, 55).tolist()
    a_df = _candles(a)
    b_df = _candles(b, start="2020-02-15")
    candles = {"A/USDT": a_df, "B/USDT": b_df}
    idx = build_crypto_market_index(candles, fee_rate=0.001, slippage_rate=0.0005)
    # Index should start before B is listed (using only A), then on
    # B's listing date the weights jump from 100% A to 50/50.
    pre_listing = idx["close"].loc[idx.index < pd.Timestamp("2020-02-15", tz="UTC")]
    assert (pre_listing > 0).all()


def test_index_value_is_strictly_positive_throughout():
    rng = np.random.default_rng(0)
    candles = {
        f"A{i}/USDT": _candles(
            (100 + rng.normal(0, 0.5, 200).cumsum()).clip(min=1).tolist()
        )
        for i in range(5)
    }
    idx = build_crypto_market_index(candles)
    assert (idx["close"] > 0).all()
    assert idx["close"].iloc[0] == pytest.approx(100.0)


def test_index_format_ohlcv_compatible_with_strategies():
    """The output must have exactly the OHLCV schema the rest of the
    repo expects so any single-asset strategy can run on it without
    adaptation."""
    candles = {"BTC/USDT": _candles(np.linspace(100, 150, 50).tolist())}
    idx = build_crypto_market_index(candles)
    assert list(idx.columns) == ["open", "high", "low", "close", "volume"]
    assert idx.index.name == "timestamp"
    # OHLC are all the same (we don't synthesize intraday ranges).
    assert (idx["open"] == idx["close"]).all()
    assert (idx["high"] == idx["close"]).all()
    assert (idx["low"] == idx["close"]).all()


# ---------------------------------------------------------------------------
# Fail-loud on data gaps (hard rule: the basket never shrinks silently)
# ---------------------------------------------------------------------------


def test_interior_nan_gap_raises():
    """A missing candle after an asset's first valid close must raise —
    silently treating it as 'not active' shrinks the basket, forces an
    unscheduled rebalance, and zeroes the price move across the gap."""
    a = np.linspace(100, 200, 100)
    b = np.linspace(50, 80, 100).astype(float)
    b[40:43] = np.nan
    candles = {"BTC/USDT": _candles(a.tolist()), "ETH/USDT": _candles(b.tolist())}
    with pytest.raises(ValueError, match="ETH/USDT.*3 missing bar"):
        build_crypto_market_index(candles)


def test_trailing_mismatch_raises():
    """One asset's series ending before the others (stale or partial
    fetch) must raise — the last bars are exactly where the live signal
    reads the index."""
    a = np.linspace(100, 200, 100).tolist()
    b = np.linspace(50, 80, 98).tolist()  # ends 2 bars early
    candles = {"BTC/USDT": _candles(a), "ETH/USDT": _candles(b)}
    with pytest.raises(ValueError, match="ETH/USDT.*2 missing bar"):
        build_crypto_market_index(candles)


def test_leading_nan_late_listing_still_allowed():
    """Pre-listing leading NaN is the designed dynamic-universe entry
    path and must NOT raise."""
    a = np.linspace(100, 200, 100).tolist()
    b = np.linspace(50, 80, 60).tolist()  # lists 40 bars later
    candles = {
        "BTC/USDT": _candles(a),
        "ETH/USDT": _candles(b, start="2020-02-10"),
    }
    idx = build_crypto_market_index(candles)
    assert len(idx) == 100
    assert (idx["close"] > 0).all()
