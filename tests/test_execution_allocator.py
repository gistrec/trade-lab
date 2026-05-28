"""Tests for the allocator (signal → target qty per asset)."""
from __future__ import annotations

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


def test_signal_one_equal_weights_each_asset():
    """signal=1 → each asset gets exactly 1/N of total equity in quote terms."""
    alloc = compute_target_allocation(
        signal=1.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET,
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
        prices=_PRICES, basket=_BASKET,
    )
    expected_per_asset = 0.5 * 70_000.0 / 7   # = 5_000 quote
    for sym in _BASKET:
        assert alloc.target_quote_per_asset[sym] == pytest.approx(expected_per_asset)


def test_signal_zero_yields_all_cash():
    alloc = compute_target_allocation(
        signal=0.0, total_equity=70_000.0,
        prices=_PRICES, basket=_BASKET,
    )
    for sym in _BASKET:
        assert alloc.target_quote_per_asset[sym] == 0.0
        assert alloc.target_qty_per_asset[sym] == 0.0


def test_invalid_signal_raises():
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=1.1, total_equity=1.0,
            prices=_PRICES, basket=_BASKET,
        )
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=-0.1, total_equity=1.0,
            prices=_PRICES, basket=_BASKET,
        )


def test_negative_equity_raises():
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=0.5, total_equity=-1.0,
            prices=_PRICES, basket=_BASKET,
        )


def test_empty_basket_raises():
    with pytest.raises(ValueError):
        compute_target_allocation(
            signal=1.0, total_equity=1.0,
            prices=_PRICES, basket=(),
        )


def test_missing_price_raises_loudly():
    """Allocator never silently shrinks the basket — a missing price is
    an operational event, surface it."""
    prices_minus_btc = {k: v for k, v in _PRICES.items() if k != "BTC"}
    with pytest.raises(KeyError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=prices_minus_btc, basket=_BASKET,
        )


def test_zero_price_raises():
    bad_prices = {**_PRICES, "BTC": 0.0}
    with pytest.raises(ValueError, match="BTC"):
        compute_target_allocation(
            signal=1.0, total_equity=70_000.0,
            prices=bad_prices, basket=_BASKET,
        )


def test_subset_basket_with_smaller_n():
    """A 3-asset configurable subset (Kraken without BNB etc.) splits
    equity into 1/3 per asset, not 1/7."""
    subset = ("BTC", "ETH", "SOL")
    alloc = compute_target_allocation(
        signal=1.0, total_equity=30_000.0,
        prices=_PRICES, basket=subset,
    )
    for sym in subset:
        assert alloc.target_quote_per_asset[sym] == pytest.approx(10_000.0)
