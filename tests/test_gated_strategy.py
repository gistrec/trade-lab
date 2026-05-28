"""Tests for GatedStrategy + compute_breadth_gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.gated_strategy import (
    GatedStrategy, compute_breadth_gate,
)
from trade_lab.strategies.sma_cross import SMACrossStrategy
from trade_lab.strategies.regime_only import RegimeOnlyStrategy


def _candles(closes, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Breadth gate
# ---------------------------------------------------------------------------


def test_breadth_gate_all_uptrending_all_above():
    """Every asset in a clean uptrend → 100% of universe above SMA(50)
    after warmup → gate is True for the post-warmup region."""
    n = 200
    closes = np.linspace(100, 200, n).tolist()
    universe = {f"A{i}": _candles(closes) for i in range(5)}
    gate = compute_breadth_gate(universe, sma_period=50, threshold=0.5)
    # After warmup (bar 49) every asset is above its SMA in the uptrend.
    assert gate.iloc[60:].all()
    # Within warmup the gate is False (no SMA defined).
    assert not gate.iloc[:48].any()


def test_breadth_gate_split_universe_threshold():
    """Half the universe up, half flat. Breadth = 50%. With
    threshold=0.5 (inclusive >=), the gate is True; with threshold
    above 0.5 it should be False."""
    n = 200
    up = np.linspace(100, 200, n).tolist()
    flat = [100.0] * n
    universe = {"up1": _candles(up), "up2": _candles(up),
                "flat1": _candles(flat), "flat2": _candles(flat)}
    gate_50 = compute_breadth_gate(universe, sma_period=50, threshold=0.5)
    gate_60 = compute_breadth_gate(universe, sma_period=50, threshold=0.6)
    # In the uptrend region, exactly 50% are above SMA (the two "up"
    # assets); the two flat assets sit at or near their SMA so they
    # don't pass. Floating-point makes "==SMA" unreliable so we
    # check the post-warmup portion of the result.
    assert gate_50.iloc[100:].all()
    assert not gate_60.iloc[100:].any()


def test_breadth_gate_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        compute_breadth_gate({}, threshold=-0.1)
    with pytest.raises(ValueError):
        compute_breadth_gate({}, threshold=1.1)


def test_breadth_gate_empty_universe():
    assert compute_breadth_gate({}).empty


def test_breadth_gate_no_lookahead():
    """Appending future garbage candles must not change any earlier
    gate value."""
    rng = np.random.default_rng(0)
    n = 200
    base = {f"A{i}": _candles((100 + rng.normal(0, 1, n).cumsum()).clip(min=1).tolist())
            for i in range(3)}
    gate_base = compute_breadth_gate(base, sma_period=50, threshold=0.5)

    # Append 50 garbage bars to each asset.
    ext = {}
    for k, df in base.items():
        next_day = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        garbage = _candles([1e6, 1e-6, 1e6, 1e-6, 1e6] * 10, start=next_day)
        ext[k] = pd.concat([df, garbage])
    gate_ext = compute_breadth_gate(ext, sma_period=50, threshold=0.5)
    # Prefix of gate_ext must equal gate_base.
    np.testing.assert_array_equal(
        gate_base.values,
        gate_ext.iloc[: len(gate_base)].values,
    )


# ---------------------------------------------------------------------------
# Gated strategy wrapper
# ---------------------------------------------------------------------------


def test_gated_strategy_forces_flat_when_gate_false():
    n = 200
    closes = np.linspace(100, 200, n).tolist()  # clean uptrend
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)
    gate = pd.Series(False, index=candles.index)
    gated = GatedStrategy(inner, gate)
    sig = gated.generate_signals(candles)
    assert (sig == 0.0).all()


def test_gated_strategy_passes_inner_when_gate_true():
    n = 200
    closes = np.linspace(100, 200, n).tolist()
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)
    inner_sig = inner.generate_signals(candles)
    gate = pd.Series(True, index=candles.index)
    gated = GatedStrategy(inner, gate)
    sig = gated.generate_signals(candles)
    pd.testing.assert_series_equal(
        sig.astype(float), inner_sig.astype(float), check_names=False,
    )


def test_gated_strategy_with_partial_gate_zeroes_only_false_bars():
    n = 200
    closes = np.linspace(100, 200, n).tolist()
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)
    gate = pd.Series(True, index=candles.index)
    gate.iloc[100:150] = False
    gated = GatedStrategy(inner, gate)
    sig = gated.generate_signals(candles)
    # Inside the gate-false window the signal must be 0.
    assert (sig.iloc[100:150] == 0.0).all()


def test_gated_strategy_reindex_ffill_on_index_mismatch():
    """Gate computed on a wider date range than the input candles must
    be reindexed onto the candles' index without error."""
    candles = _candles(np.linspace(100, 200, 100).tolist())
    wider_idx = pd.date_range("2019-12-15", periods=200, freq="1D", tz="UTC")
    gate = pd.Series(True, index=wider_idx)
    gated = GatedStrategy(SMACrossStrategy(10, 30), gate)
    sig = gated.generate_signals(candles)
    assert len(sig) == len(candles)


def test_gated_strategy_engine_integration():
    """End-to-end: gated strategy runs through the engine with the
    one-bar shift applied as usual."""
    n = 200
    rng = np.random.default_rng(0)
    closes = (100 + rng.normal(0.1, 0.5, n).cumsum()).clip(min=1).tolist()
    candles = _candles(closes)
    inner = RegimeOnlyStrategy(regime_period=20)
    gate = pd.Series(True, index=candles.index)
    gate.iloc[50:100] = False
    gated = GatedStrategy(inner, gate)
    result = run_backtest(candles, gated)
    raw = gated.generate_signals(candles)
    expected_positions = raw.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        result.positions.astype(float), expected_positions.astype(float),
        check_names=False,
    )


def test_gated_strategy_rejects_none_gate():
    with pytest.raises(ValueError):
        GatedStrategy(SMACrossStrategy(10, 30), gate=None)


def test_gated_strategy_name_includes_inner_label():
    inner = SMACrossStrategy(10, 30)
    gate = pd.Series([True, False])
    g = GatedStrategy(inner, gate, gate_name="breadth50")
    assert "sma_cross" in g.name
    assert "breadth50" in g.name
