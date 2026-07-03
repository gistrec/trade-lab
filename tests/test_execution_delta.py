"""Tests for the delta planner (target vs current → orders + skipped)."""
from __future__ import annotations

import pytest

from trade_lab.execution.allocator import compute_target_allocation
from trade_lab.execution.broker import MarketConstraints
from trade_lab.execution.delta import (
    compute_delta_plan, total_skipped_quote_drift,
)


_PRICES = {"BTC": 50_000.0, "ETH": 3_000.0}
_BASKET = ("BTC", "ETH")
_EQUAL = {"BTC": 0.5, "ETH": 0.5}   # flat 1/N weights for a 2-asset basket


def _allocation(signal=1.0, equity=70_000.0, weights=_EQUAL):
    return compute_target_allocation(
        signal=signal, total_equity=equity, prices=_PRICES,
        basket=_BASKET, weights=weights,
    )


def _binance_like_constraints():
    """Approximate Binance constraints — both min_amount and min_cost
    are populated."""
    return {
        "BTC/USDT": MarketConstraints(
            symbol="BTC/USDT",
            min_amount=0.0001, min_cost=10.0,
            amount_precision=8, raw={},
        ),
        "ETH/USDT": MarketConstraints(
            symbol="ETH/USDT",
            min_amount=0.001, min_cost=10.0,
            amount_precision=8, raw={},
        ),
    }


def test_full_buy_when_no_current_holdings():
    """Empty current_holdings → every target qty becomes a buy order."""
    alloc = _allocation()
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints={}, quote_currency="USDT",
    )
    sides = {o.symbol: o.side for o in plan.orders}
    assert sides == {"BTC/USDT": "buy", "ETH/USDT": "buy"}
    assert plan.skipped == []


def test_partial_buy_to_close_delta():
    """If current holdings are half the target, the buy amount is the
    other half."""
    alloc = _allocation()                          # 35k each, qty target
    btc_target = alloc.target_qty_per_asset["BTC"]
    current = {"BTC": btc_target * 0.5}            # halfway there
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints={}, quote_currency="USDT",
    )
    btc_order = next(o for o in plan.orders if o.symbol == "BTC/USDT")
    assert btc_order.side == "buy"
    assert btc_order.base_amount == pytest.approx(btc_target * 0.5)


def test_sell_when_currently_overweight():
    """current > target → sell the excess."""
    alloc = _allocation(signal=0.5)
    btc_target_half = alloc.target_qty_per_asset["BTC"]
    current = {"BTC": btc_target_half * 2.0, "ETH": 0.0}
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints={}, quote_currency="USDT",
    )
    btc_order = next(o for o in plan.orders if o.symbol == "BTC/USDT")
    assert btc_order.side == "sell"
    assert btc_order.base_amount == pytest.approx(btc_target_half)


def test_zero_delta_produces_no_order():
    alloc = _allocation()
    current = dict(alloc.target_qty_per_asset)
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints={}, quote_currency="USDT",
    )
    assert plan.orders == []
    assert plan.skipped == []


def test_min_cost_filter_skips_small_orders():
    """A tiny order (e.g. 0.0001 BTC × $50k = $5) below Binance's
    min_cost of $10 must be SKIPPED and logged in plan.skipped."""
    alloc = _allocation(signal=0.5, equity=70_000.0)
    # Force a tiny delta on BTC: current is almost exactly target.
    target = alloc.target_qty_per_asset["BTC"]
    current = {"BTC": target - (5.0 / _PRICES["BTC"])}   # $5 short of target
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(),
        quote_currency="USDT",
    )
    # No BTC order should be sendable (sub-$10 notional).
    btc_orders = [o for o in plan.orders if o.symbol == "BTC/USDT"]
    assert btc_orders == []
    btc_skipped = [s for s in plan.skipped if s.symbol == "BTC/USDT"]
    assert len(btc_skipped) == 1
    assert "min_cost" in btc_skipped[0].reason


def test_min_amount_filter_skips_below_amount_min():
    """If desired amount < min_amount the order is skipped even if
    notional is acceptable."""
    constraints = _binance_like_constraints()
    # Bump BTC min_amount above any realistic order so every BTC delta
    # gets blocked. ETH stays normal.
    constraints["BTC/USDT"] = MarketConstraints(
        symbol="BTC/USDT", min_amount=10.0,   # absurd 10 BTC minimum
        min_cost=10.0, amount_precision=8, raw={},
    )
    alloc = _allocation()
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints=constraints, quote_currency="USDT",
    )
    btc_skipped = [s for s in plan.skipped if s.symbol == "BTC/USDT"]
    assert len(btc_skipped) == 1
    assert "min_amount" in btc_skipped[0].reason


def test_total_skipped_quote_drift_sums_notional():
    alloc = _allocation(signal=0.5, equity=70_000.0)
    target = alloc.target_qty_per_asset["BTC"]
    current = {"BTC": target - (5.0 / _PRICES["BTC"])}  # $5 sub-min on BTC
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(),
        quote_currency="USDT",
    )
    drift = total_skipped_quote_drift(plan)
    assert drift == pytest.approx(5.0)


_DRIFTED = {"BTC": 0.6, "ETH": 0.4}   # BTC outperformed → overweight in the index


def _holdings_tracking(weights, equity=70_000.0):
    """Base-unit holdings whose quote value equals signal=1 × weights ×
    equity — i.e. holdings that have drifted along with the index."""
    return {sym: (weights[sym] * equity) / _PRICES[sym] for sym in _BASKET}


def test_drifted_weight_target_produces_no_orders_when_holdings_track_index():
    """Option B self-gating (C3): between monthly rebalances the
    drifted-weight target equals the drifted holdings, so NO orders fire —
    exactly the low daily turnover the backtest measured. Sizing to flat
    1/N instead would churn every day (see the contrast test)."""
    alloc = _allocation(signal=1.0, weights=_DRIFTED)
    current = _holdings_tracking(_DRIFTED)
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(), quote_currency="USDT",
    )
    assert plan.orders == []
    assert plan.skipped == []


def test_drifted_weight_target_gated_by_min_cost_between_rebalances():
    """Production reality: holdings never bit-match the target (ticker !=
    bar close, amount precision, fees), so on the in-between days self-
    gating rests on the min_cost GATE, not the exact-zero fast path. A ~$3
    residual (< $10 min_cost) must be suppressed and recorded as sub-min
    drift, not sent — this is the path the deployed executor actually
    walks between monthly rebalances."""
    alloc = _allocation(signal=1.0, weights=_DRIFTED)
    current = dict(_holdings_tracking(_DRIFTED))
    current["BTC"] += 3.0 / _PRICES["BTC"]   # $3 residual, below min_cost $10
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(), quote_currency="USDT",
    )
    assert plan.orders == []                               # gated, not sent
    assert [s.symbol for s in plan.skipped] == ["BTC/USDT"]
    assert "min_cost" in plan.skipped[0].reason
    assert total_skipped_quote_drift(plan) == pytest.approx(3.0)


def test_kraken_like_no_min_cost_fires_tiny_residual_order():
    """Documents the known gap: an exchange reporting no min_cost/min_amount
    (Kraken via CCXT) has nothing to suppress the tiny residual, so the
    SAME $3 delta becomes a live order. Pinned so a change here is noticed,
    not silently assumed safe."""
    cons = {
        "BTC/USDT": MarketConstraints(symbol="BTC/USDT", min_amount=None,
                                      min_cost=None, amount_precision=8, raw={}),
        "ETH/USDT": MarketConstraints(symbol="ETH/USDT", min_amount=None,
                                      min_cost=None, amount_precision=8, raw={}),
    }
    alloc = _allocation(signal=1.0, weights=_DRIFTED)
    current = dict(_holdings_tracking(_DRIFTED))
    current["BTC"] += 3.0 / _PRICES["BTC"]
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=cons, quote_currency="USDT",
    )
    assert any(o.symbol == "BTC/USDT" for o in plan.orders)


def test_flat_weight_target_churns_the_same_drifted_holdings():
    """Contrast + regression witness: the SAME drifted holdings sized to
    FLAT 1/N force a full rebalance (BTC sell, ETH buy). This is both the
    pre-C3 daily-churn bug AND, correctly, what the month-start weight
    reset does — the difference is cadence, and only the drifted-weight
    path suppresses it on the in-between days."""
    alloc = _allocation(signal=1.0, weights=_EQUAL)   # flat 1/N reset weights
    current = _holdings_tracking(_DRIFTED)            # holdings still drifted
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(), quote_currency="USDT",
    )
    sides = {o.symbol: o.side for o in plan.orders}
    assert sides == {"BTC/USDT": "sell", "ETH/USDT": "buy"}


def test_missing_constraints_does_no_filtering():
    """Empty constraints dict = trust the allocator, send anything
    non-zero. Useful for tests and exchanges where CCXT doesn't
    populate limits."""
    alloc = _allocation()
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints={}, quote_currency="USDT",
    )
    assert len(plan.orders) == 2
    assert plan.skipped == []
