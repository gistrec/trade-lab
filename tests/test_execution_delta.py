"""Tests for the delta planner (target vs current → orders + skipped)."""
from __future__ import annotations

import ccxt
import pytest

from trade_lab.execution.allocator import compute_target_allocation
from trade_lab.execution.broker import MarketConstraints
from trade_lab.execution.delta import (
    RESERVE_BPS, OrderIntent, apply_reserve_cap, compute_delta_plan,
    total_funding_cap_quote, total_skipped_quote_drift,
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


# ---------------------------------------------------------------------------
# RESERVE_BPS buy cap (#28, owner-sanctioned) — a full entry from cash
# sized off ticker last fills at the ask, so planning buys that sum to
# exactly quote_free loses the tail order to InsufficientFunds on any
# uptick between snapshot and send.
# ---------------------------------------------------------------------------


def test_full_cash_entry_reserves_10bp_of_free_quote():
    """Full-cash entry (no holdings, quote_free == equity): total planned
    buy spend stays ≤ quote_free × (1 − 10 bp) and strictly < quote_free;
    relative (drifted-weight) proportions are preserved."""
    assert RESERVE_BPS == 10   # pinned — a silent change is a sizing change
    quote_free = 70_000.0
    alloc = _allocation(equity=quote_free, weights=_DRIFTED)   # 0.6 / 0.4
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints={}, quote_currency="USDT", quote_free=quote_free,
        fee_rate=0.001,
    )
    total = sum(o.notional_quote for o in plan.orders)
    assert total <= quote_free * (1 - RESERVE_BPS / 10_000) + 1e-6
    assert total < quote_free
    assert total == pytest.approx(quote_free * 0.999)
    by_sym = {o.symbol: o.notional_quote for o in plan.orders}
    assert by_sym["BTC/USDT"] / by_sym["ETH/USDT"] == pytest.approx(0.6 / 0.4)
    # The shaved 10 bp is journaled as first-class funding_cap drift —
    # metered APART from sub-min drift, whose contract (monitoring's
    # "unfillable rebalance drift") is work the exchange refused.
    assert [s.reason for s in plan.skipped] == ["funding_cap", "funding_cap"]
    assert total_skipped_quote_drift(plan) == 0.0
    assert total_funding_cap_quote(plan) == pytest.approx(quote_free * 0.001)


def test_partial_rebalance_below_cap_unchanged_by_reserve():
    """Buy spend well under free quote → the reserve must not bind: the
    plan is identical to the no-quote_free planner, with the pre-change
    numbers pinned exactly (no proportional shave leaks in)."""
    alloc = _allocation()                       # $35k target per asset
    current = {"BTC": alloc.target_qty_per_asset["BTC"] * 0.5}
    kwargs = dict(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(), quote_currency="USDT",
    )
    with_reserve = compute_delta_plan(
        **kwargs, quote_free=60_000.0, fee_rate=0.001)
    assert with_reserve == compute_delta_plan(**kwargs)
    btc = next(o for o in with_reserve.orders if o.symbol == "BTC/USDT")
    assert btc.base_amount == 0.35              # exact, not 0.35 × 0.999
    assert btc.notional_quote == 17_500.0


def test_cross_rebalance_cap_counts_sell_proceeds():
    """Month-start weight reset at quote_free ≈ 0: sells place first and
    fund the buys, so the cap must count sendable sell proceeds. Capping
    at bare quote_free would scale the buys to dust and bleed the book
    into cash every rebalance."""
    alloc = _allocation()                       # $35k target per asset
    current = {                                 # BTC $45k, ETH $25k held
        "BTC": 45_000.0 / _PRICES["BTC"],
        "ETH": 25_000.0 / _PRICES["ETH"],
    }
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints={}, quote_currency="USDT", quote_free=0.0,
        fee_rate=0.001,
    )
    sides = {o.symbol: o.side for o in plan.orders}
    assert sides == {"BTC/USDT": "sell", "ETH/USDT": "buy"}
    btc = next(o for o in plan.orders if o.symbol == "BTC/USDT")
    eth = next(o for o in plan.orders if o.symbol == "ETH/USDT")
    assert btc.notional_quote == pytest.approx(10_000.0)   # sells unscaled
    # Buys are funded by the sell NET of the taker fee, then the 10 bp
    # price reserve — sizing off the gross notional would hand the whole
    # reserve to the fee.
    assert eth.notional_quote == pytest.approx(10_000.0 * (1 - 0.001) * 0.999)


def test_pure_sell_plan_unaffected_by_reserve():
    """No buys → nothing to cap, whatever quote_free says."""
    alloc = _allocation(signal=0.0)
    current = {"BTC": 0.5, "ETH": 5.0}
    kwargs = dict(
        allocation=alloc, current_holdings=current,
        constraints=_binance_like_constraints(), quote_currency="USDT",
    )
    assert compute_delta_plan(
        **kwargs, quote_free=3.0, fee_rate=0.001,
    ) == compute_delta_plan(**kwargs)


def test_cap_holds_after_lot_step_requantization():
    """Codex review: scaling the RAW delta and requantizing can bounce
    the total back above the cap (one asset, $100 target, price $1, step
    0.19: quantized 99.94 → cap 99.90 → scaled raw ≈99.96 → truncates
    back to 99.94). The POST-quantization total must honor the cap."""
    alloc = compute_target_allocation(
        signal=1.0, total_equity=100.0, prices={"BTC": 1.0},
        basket=("BTC",), weights={"BTC": 1.0},
    )
    cons = {
        "BTC/USDT": MarketConstraints(
            symbol="BTC/USDT", min_amount=None, min_cost=None,
            amount_precision=None, raw={"precision": {"amount": 0.19}},
            precision_mode=ccxt.TICK_SIZE,
        ),
    }
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={}, constraints=cons,
        quote_currency="USDT", quote_free=100.0, fee_rate=0.001,
    )
    cap = 100.0 * (1 - RESERVE_BPS / 10_000)
    [order] = plan.orders
    assert order.notional_quote <= cap
    assert order.base_amount == pytest.approx(525 * 0.19)   # max lots ≤ cap
    # Still on the lot grid — ccxt re-truncation must be a no-op.
    assert float(ccxt.decimal_to_precision(
        order.base_amount, ccxt.TRUNCATE, 0.19, ccxt.TICK_SIZE,
    )) == order.base_amount
    [skip] = plan.skipped
    assert skip.reason == "funding_cap"
    assert skip.desired_notional == pytest.approx(99.94 - order.notional_quote)


def test_zero_free_quote_records_true_gap_as_funding_cap():
    """quote_free=0 with no sells → cap 0 suppresses every buy. The skip
    records must carry the ORIGINAL unscaled gap under 'funding_cap' —
    scaled-to-zero desired values would report zero drift and blame the
    lot step."""
    alloc = _allocation(equity=70_000.0)
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints=_binance_like_constraints(), quote_currency="USDT",
        quote_free=0.0, fee_rate=0.001,
    )
    assert plan.orders == []
    assert {s.symbol for s in plan.skipped} == {"BTC/USDT", "ETH/USDT"}
    assert all(s.reason == "funding_cap" for s in plan.skipped)
    assert all(s.desired_notional > 0 for s in plan.skipped)
    assert total_skipped_quote_drift(plan) == 0.0   # not exchange-refused
    assert total_funding_cap_quote(plan) == pytest.approx(70_000.0, rel=1e-6)


def test_no_spend_symbol_is_excluded_from_cap_spend_and_scaling():
    """A buy that draws nothing from quote_free (caller's call: the same
    coid is still open on the exchange, or already terminal so nothing
    gets placed) must neither consume the cap nor be scaled. Only the buy
    the free quote actually has to fund is shaved."""
    orders = [
        OrderIntent(symbol="BTC/USDT", side="buy", base_amount=0.1,
                    notional_quote=5_000.0, price_used=50_000.0,
                    reason="delta from target"),
        OrderIntent(symbol="ETH/USDT", side="buy", base_amount=1.0,
                    notional_quote=3_000.0, price_used=3_000.0,
                    reason="delta from target"),
    ]
    capped, skips = apply_reserve_cap(
        orders, quote_free=3_000.0, constraints={}, fee_rate=0.001,
        no_spend_symbols={"BTC/USDT"},
    )
    by_sym = {o.symbol: o for o in capped}
    assert by_sym["BTC/USDT"] == orders[0]
    assert by_sym["ETH/USDT"].notional_quote == pytest.approx(3_000.0 * 0.999)
    assert [(s.symbol, s.reason) for s in skips] == [("ETH/USDT", "funding_cap")]


# ---------------------------------------------------------------------------
# Lot-step quantization (false-partial regression) — ccxt truncates the
# amount to the LOT_SIZE step inside create_order, so an intent carrying
# the raw amount makes intended_amount unreachable and a fully filled
# order journals as a false 'partial' (cycle outcome 'partial', exit 2).
# ---------------------------------------------------------------------------


def _btc_only_allocation(equity: float, price: float):
    return compute_target_allocation(
        signal=1.0, total_equity=equity, prices={"BTC": price},
        basket=("BTC",), weights={"BTC": 1.0},
    )


def _btc_tick_constraints(step=1e-05, min_amount=1e-05, min_cost=5.0):
    return {
        "BTC/USDT": MarketConstraints(
            symbol="BTC/USDT", min_amount=min_amount, min_cost=min_cost,
            amount_precision=None, raw={"precision": {"amount": step}},
            precision_mode=ccxt.TICK_SIZE,
        ),
    }


def test_intent_amount_quantized_to_lot_step():
    """A 25 USDT buy at $98,350 wants 2.5419e-4 BTC; on Binance's 1e-5
    lot step the sendable quantity is exactly 0.00025. The intent must
    carry that — the same value ccxt transmits — so a full fill compares
    equal to intended."""
    alloc = _btc_only_allocation(equity=25.0, price=98_350.0)
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints=_btc_tick_constraints(), quote_currency="USDT",
    )
    assert plan.skipped == []
    [order] = plan.orders
    assert order.base_amount == 0.00025
    assert order.notional_quote == pytest.approx(0.00025 * 98_350.0)
    # Byte-identical to ccxt's own truncation (idempotent re-truncation).
    resent = float(ccxt.decimal_to_precision(
        order.base_amount, ccxt.TRUNCATE, 1e-05, ccxt.TICK_SIZE,
    ))
    assert resent == order.base_amount


def test_delta_below_one_lot_step_is_first_class_skip():
    """A delta smaller than one lot step truncates to zero. It must
    surface as a SkippedDelta with an explicit reason — never a silent
    drop, never a zero-amount order reaching the broker."""
    alloc = _btc_only_allocation(equity=25.0, price=98_350.0)
    target = alloc.target_qty_per_asset["BTC"]
    current = {"BTC": target - 4.2e-06}          # gap under the 1e-5 step
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints=_btc_tick_constraints(min_amount=None, min_cost=None),
        quote_currency="USDT",
    )
    assert plan.orders == []
    [s] = plan.skipped
    assert "truncates to 0" in s.reason
    assert s.desired_amount == pytest.approx(4.2e-06)
    assert s.desired_notional == pytest.approx(4.2e-06 * 98_350.0)


def test_min_cost_gate_evaluates_truncated_notional():
    """Raw notional 26 clears min_cost 25, but after truncation to the
    1e-4 lot step only 20 is sendable. The gate must judge what will
    actually be sent, so this is a skip — not an order the exchange
    would reject (or fill 20/26 as a false partial)."""
    alloc = _btc_only_allocation(equity=26.0, price=100_000.0)
    plan = compute_delta_plan(
        allocation=alloc, current_holdings={},
        constraints=_btc_tick_constraints(
            step=1e-04, min_amount=None, min_cost=25.0,
        ),
        quote_currency="USDT",
    )
    assert plan.orders == []
    [s] = plan.skipped
    assert "min_cost" in s.reason
    # Drift metric keeps the raw desired gap, not the truncated one.
    assert s.desired_notional == pytest.approx(26.0)


def test_sell_proceeds_are_netted_by_the_fee():
    """Codex 3858750080: a sell credits its notional MINUS the taker fee.
    Sizing buys off the gross ticker notional hands the whole 10 bp
    reserve to the fee — at a fee above 10 bp the plan is underfunded
    even at unchanged prices."""
    alloc = _allocation()                       # $35k target per asset
    current = {                                 # BTC $45k, ETH $25k held
        "BTC": 45_000.0 / _PRICES["BTC"],
        "ETH": 25_000.0 / _PRICES["ETH"],
    }
    fee = 0.005                                 # 50 bp: well above the reserve
    plan = compute_delta_plan(
        allocation=alloc, current_holdings=current,
        constraints={}, quote_currency="USDT", quote_free=0.0, fee_rate=fee,
    )
    sell = next(o for o in plan.orders if o.side == "sell")
    buy = next(o for o in plan.orders if o.side == "buy")
    spendable = sell.notional_quote * (1.0 - fee)
    assert buy.notional_quote == pytest.approx(spendable * 0.999)
    # The invariant that matters: what we spend never exceeds what the
    # exchange actually credits us.
    assert buy.notional_quote < spendable


def test_cap_restores_a_buy_it_pushed_below_the_minimum():
    """Codex 3858750071: proportional scaling must not drop an order that
    was sendable before the cap — suppressing it strands its quote and can
    leave a basket asset out of the entry entirely."""
    constraints = {
        "BTC/USDT": MarketConstraints(
            symbol="BTC/USDT", min_amount=0.0, min_cost=10.0,
            amount_precision=8, raw={},
        ),
        "ETH/USDT": MarketConstraints(
            symbol="ETH/USDT", min_amount=0.0, min_cost=10.0,
            amount_precision=8, raw={},
        ),
    }
    # BTC target sits a hair above the $10 minimum; ETH is large. The cap
    # shaves ~0.1% off both, which alone would push BTC under the minimum.
    orders = [
        OrderIntent(
            symbol="BTC/USDT", side="buy",
            base_amount=10.005 / _PRICES["BTC"], notional_quote=10.005,
            price_used=_PRICES["BTC"], reason="delta from target",
        ),
        OrderIntent(
            symbol="ETH/USDT", side="buy",
            base_amount=9_989.995 / _PRICES["ETH"], notional_quote=9_989.995,
            price_used=_PRICES["ETH"], reason="delta from target",
        ),
    ]
    capped, skips = apply_reserve_cap(
        orders, quote_free=10_000.0, constraints=constraints, fee_rate=0.0,
    )
    by_sym = {o.symbol: o for o in capped}
    assert "BTC/USDT" in by_sym, "the small buy must survive the cap"
    assert by_sym["BTC/USDT"].notional_quote >= 10.0
    total = sum(o.notional_quote for o in capped)
    assert total <= 10_000.0 * 0.999 + 1e-6, "the cap still holds"
    assert all(s.reason == "funding_cap" for s in skips)
