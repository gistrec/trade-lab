"""Tests for the allocator (signal → target qty per asset).

The allocator sizes each asset to ``signal × w_i × equity`` where ``w_i``
is the basket index's drifted weight (C3 / Option B). Passing equal
weights reproduces the old flat-``1/N`` behaviour; passing drifted
weights is what makes live execution track the monthly-rebalanced
backtest.
"""
from __future__ import annotations

import math

import pytest

from trade_lab.execution.allocator import (
    TargetAllocation, compute_target_allocation,
)


_PRICES = {
    "BTC": 50_000.0,
    "ETH": 3_000.0,
    "BNB": 600.0,
    "SOL": 150.0,
    "ADA": 0.40,
    "XRP": 0.50,
    "DOGE": 0.10,
}
_BASKET = ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")
_EQUAL = {sym: 1.0 / len(_BASKET) for sym in _BASKET}   # flat 1/N weights


def test_signal_one_equal_weights_each_asset():
    """signal=1 with equal weights → each asset gets exactly 1/N of total
    equity in quote terms (the old flat behaviour, now expressed via
    weights)."""
    alloc = compute_target_allocation(
        signal=1.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET, weights=_EQUAL,
    )
    expected_per_asset = 70_000.0 / 7   # = 10_000 quote
    for sym in _BASKET:
        assert alloc.target_quote_per_asset[sym] == pytest.approx(expected_per_asset)
        assert alloc.target_qty_per_asset[sym] == pytest.approx(
            expected_per_asset / _PRICES[sym]
        )


def test_signal_half_halves_each_asset():
    """signal=0.5 → each asset gets half the per-asset target."""
    alloc = compute_target_allocation(
        signal=0.5, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET, weights=_EQUAL,
    )
    expected_per_asset = 0.5 * 70_000.0 / 7   # = 5_000 quote
    for sym in _BASKET:
        assert alloc.target_quote_per_asset[sym] == pytest.approx(expected_per_asset)


def test_signal_zero_yields_all_cash():
    alloc = compute_target_allocation(
        signal=0.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET, weights=_EQUAL,
    )
    for sym in _BASKET:
        assert alloc.target_quote_per_asset[sym] == 0.0
        assert alloc.target_qty_per_asset[sym] == 0.0


def test_drifted_weights_size_each_asset_pro_rata():
    """The core C3 behaviour: target_quote[i] = signal × w_i × equity,
    with w_i the drifted index weight — NOT flat 1/N. An overweight asset
    gets a proportionally larger dollar target."""
    # BTC drifted to 40% of the basket, the rest split the remaining 60%.
    weights = {"BTC": 0.40}
    rest = 0.60 / 6
    for sym in _BASKET[1:]:
        weights[sym] = rest
    alloc = compute_target_allocation(
        signal=1.0, total_equity=100_000.0,
        prices=_PRICES, basket=_BASKET, weights=weights,
    )
    assert alloc.target_quote_per_asset["BTC"] == pytest.approx(40_000.0)
    for sym in _BASKET[1:]:
        assert alloc.target_quote_per_asset[sym] == pytest.approx(0.60 / 6 * 100_000.0)
    # Total invested still equals signal × equity (weights sum to 1).
    assert sum(alloc.target_quote_per_asset.values()) == pytest.approx(100_000.0)


def test_weights_used_recorded_on_allocation():
    """The drifted weights the math used are echoed back for the journal /
    reconciliation audit."""
    alloc = compute_target_allocation(
        signal=1.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET, weights=_EQUAL,
    )
    assert alloc.weights_used == pytest.approx(_EQUAL)


def test_partial_weights_with_signal_raise():
    """Weights summing to < 1 while signal > 0 mean the active basket is
    silently under-invested / shrunken — the fail-loud hard rule forbids
    sizing to a shrunken book, so raise instead of quietly parking the
    remainder in cash. (Not reachable via the deployed pipeline, which
    renormalises weights to 1 over active assets, but the allocator must
    not delegate its whole fail-loud guarantee to the index.)"""
    weights = {sym: 0.10 for sym in _BASKET}   # sums to 0.70
    with pytest.raises(ValueError, match="sum"):
        compute_target_allocation(
            signal=1.0, total_equity=100_000.0,
            prices=_PRICES, basket=_BASKET, weights=weights,
        )


def test_partial_weights_all_cash_ok():
    """signal=0 is all-cash regardless of weights, so a sub-1 weights row
    is harmless there — no under-investment to flag."""
    weights = {sym: 0.10 for sym in _BASKET}
    alloc = compute_target_allocation(
        signal=0.0, total_equity=100_000.0,
        prices=_PRICES, basket=_BASKET, weights=weights,
    )
    assert all(v == 0.0 for v in alloc.target_quote_per_asset.values())


def test_invalid_signal_raises():
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=1.1, total_equity=1.0,
            prices=_PRICES, basket=_BASKET, weights=_EQUAL,
        )
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=-0.1, total_equity=1.0,
            prices=_PRICES, basket=_BASKET, weights=_EQUAL,
        )


def test_negative_equity_raises():
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=0.5, total_equity=-1.0,
            prices=_PRICES, basket=_BASKET, weights=_EQUAL,
        )


def test_empty_basket_raises():
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=1.0, total_equity=1.0,
            prices=_PRICES, basket=(), weights={},
        )


def test_missing_price_raises_loudly():
    """Allocator never silently shrinks the basket — a missing price is
    an operational event, surface it."""
    prices_minus_btc = {k: v for k, v in _PRICES.items() if k != "BTC"}
    with pytest.raises(KeyError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=prices_minus_btc, basket=_BASKET, weights=_EQUAL,
        )


def test_missing_weight_raises_loudly():
    """A basket asset with no weight is symmetric to a missing price:
    guessing a weight would silently reshape the basket, so raise."""
    weights_minus_btc = {k: v for k, v in _EQUAL.items() if k != "BTC"}
    with pytest.raises(KeyError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=_PRICES, basket=_BASKET, weights=weights_minus_btc,
        )


def test_negative_weight_raises():
    bad = {**_EQUAL, "BTC": -0.1}
    with pytest.raises(ValueError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=_PRICES, basket=_BASKET, weights=bad,
        )


def test_nan_weight_raises():
    bad = {**_EQUAL, "BTC": math.nan}
    with pytest.raises(ValueError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=_PRICES, basket=_BASKET, weights=bad,
        )


def test_weights_summing_over_one_raise():
    """Weights that sum past a fully-invested book are garbage (e.g. a
    normalisation bug upstream) — refuse to over-invest."""
    bad = {sym: 0.5 for sym in _BASKET}   # sums to 3.5
    with pytest.raises(ValueError, match="sum"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=_PRICES, basket=_BASKET, weights=bad,
        )


def test_weights_just_over_one_within_tolerance_pass():
    """A renormalised drifted-weight snapshot can overshoot 1.0 by float
    noise; the 1e-6 tolerance must let it through, not raise."""
    weights = {sym: 1.0 / len(_BASKET) for sym in _BASKET}
    weights["BTC"] += 5e-7   # total ~1.0000005, inside 1.0 + 1e-6
    assert sum(weights.values()) > 1.0
    alloc = compute_target_allocation(
        signal=1.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET, weights=weights,
    )
    assert alloc.total_equity == 70_000.0


def test_weights_over_one_beyond_tolerance_raise():
    weights = {sym: 1.0 / len(_BASKET) for sym in _BASKET}
    weights["BTC"] += 1e-3   # total well past 1.0 + 1e-6
    with pytest.raises(ValueError, match="sum"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=_PRICES, basket=_BASKET, weights=weights,
        )


def test_weights_just_under_one_within_tolerance_pass():
    """Symmetric lower-bound tolerance: a hair under 1.0 is float noise, not
    an under-invested basket — it must pass, not trip the sum<1 guard."""
    weights = {sym: 1.0 / len(_BASKET) for sym in _BASKET}
    weights["BTC"] -= 5e-7   # total ~0.9999995, inside 1.0 - 1e-6
    assert sum(weights.values()) < 1.0
    alloc = compute_target_allocation(
        signal=1.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET, weights=weights,
    )
    assert alloc.total_equity == 70_000.0


def test_zero_price_raises():
    bad_prices = {**_PRICES, "BTC": 0.0}
    with pytest.raises(ValueError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=bad_prices, basket=_BASKET, weights=_EQUAL,
        )


def test_subset_basket_with_smaller_n():
    """A 3-asset configurable subset (Kraken without BNB etc.) with equal
    weights splits equity into 1/3 per asset."""
    subset = ("BTC", "ETH", "SOL")
    weights = {sym: 1.0 / 3 for sym in subset}
    alloc = compute_target_allocation(
        signal=1.0, total_equity=30_000.0,
        prices=_PRICES, basket=subset, weights=weights,
    )
    for sym in subset:
        assert alloc.target_quote_per_asset[sym] == pytest.approx(10_000.0)
