"""Tests for the deterministic clientOrderId generator.

Coverage focus:

* Format matches the v2 schema contract exactly — any change breaks
  idempotency with existing exchange-side orders.
* Idempotency: same inputs always produce the same string.
* Distinguishability: different dates / symbols / sides never collide.
* Length cap: all 7 basket pairs fit within Binance's 32-char limit.
* Validation: naive datetime, malformed symbol, bad side, oversized
  output all raise ``ValueError`` with descriptive messages.
* Parse: inverse of the format spec, tolerant of non-string input.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from trade_lab.execution.clientorder import (
    ID_PREFIX,
    MAX_ID_LEN,
    make_client_order_id,
    normalize_symbol,
    parse_client_order_id,
)


# ---------------------------------------------------------------------------
# Format and idempotency
# ---------------------------------------------------------------------------


def test_format_matches_spec():
    coid = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    assert coid == "tsmom_20260530_BTCUSDT_buy"


def test_idempotency_same_inputs_same_id():
    a = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    b = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    assert a == b


def test_different_dates_different_ids():
    a = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    b = make_client_order_id(date(2026, 5, 31), "BTC/USDT", "buy")
    assert a != b


def test_different_symbols_different_ids():
    a = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    b = make_client_order_id(date(2026, 5, 30), "ETH/USDT", "buy")
    assert a != b


def test_different_sides_different_ids():
    a = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    b = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "sell")
    assert a != b


def test_slash_removed_from_symbol():
    coid = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    assert "/" not in coid


def test_id_uses_zero_padded_date():
    coid = make_client_order_id(date(2026, 1, 5), "BTC/USDT", "buy")
    assert "_20260105_" in coid


def test_id_starts_with_prefix():
    coid = make_client_order_id(date(2026, 5, 30), "BTC/USDT", "buy")
    assert coid.startswith(ID_PREFIX + "_")


# ---------------------------------------------------------------------------
# Length cap — must fit Binance's 32-char limit for the full basket
# ---------------------------------------------------------------------------


def test_id_under_max_for_all_basket_symbols():
    basket = ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE"]
    for base in basket:
        for side in ("buy", "sell"):
            coid = make_client_order_id(
                date(2026, 12, 31), f"{base}/USDT", side,
            )
            assert len(coid) <= MAX_ID_LEN, (
                f"{coid!r} = {len(coid)} chars > {MAX_ID_LEN}"
            )


def test_oversized_id_raises():
    """Constructed scenario: too-long combined symbol forces overflow.

    Both BASE and QUOTE here are within the 10-char regex but the
    combined ID exceeds 32 chars — the cap catches it.
    """
    with pytest.raises(ValueError, match="exceeds"):
        make_client_order_id(
            date(2026, 5, 30),
            "WAYTOOLONG/SUPERLONGY",
            "sell",
        )


# ---------------------------------------------------------------------------
# Type and validation errors
# ---------------------------------------------------------------------------


def test_naive_datetime_raises():
    with pytest.raises(ValueError, match="datetime.date, not datetime"):
        make_client_order_id(datetime(2026, 5, 30), "BTC/USDT", "buy")


def test_aware_datetime_also_raises():
    """Aware datetimes are rejected too — caller must call .date() to
    force the TZ-conversion decision."""
    dt = datetime(2026, 5, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="datetime.date, not datetime"):
        make_client_order_id(dt, "BTC/USDT", "buy")


def test_non_date_rebal_raises():
    with pytest.raises(ValueError, match="must be date"):
        make_client_order_id("2026-05-30", "BTC/USDT", "buy")


def test_invalid_side_raises_uppercase():
    with pytest.raises(ValueError, match="side must be"):
        make_client_order_id(date(2026, 5, 30), "BTC/USDT", "BUY")


def test_invalid_side_raises_other():
    with pytest.raises(ValueError, match="side must be"):
        make_client_order_id(date(2026, 5, 30), "BTC/USDT", "long")


def test_invalid_symbol_no_slash_raises():
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        make_client_order_id(date(2026, 5, 30), "BTCUSDT", "buy")


def test_invalid_symbol_lowercase_raises():
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        make_client_order_id(date(2026, 5, 30), "btc/usdt", "buy")


def test_invalid_symbol_empty_quote_raises():
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        make_client_order_id(date(2026, 5, 30), "BTC/", "buy")


# ---------------------------------------------------------------------------
# normalize_symbol
# ---------------------------------------------------------------------------


def test_normalize_symbol_strips_slash():
    assert normalize_symbol("BTC/USDT") == "BTCUSDT"


def test_normalize_symbol_rejects_dash():
    with pytest.raises(ValueError):
        normalize_symbol("BTC-USDT")


def test_normalize_symbol_rejects_missing_base():
    with pytest.raises(ValueError):
        normalize_symbol("/USDT")


def test_normalize_symbol_rejects_non_string():
    with pytest.raises(ValueError):
        normalize_symbol(123)


# ---------------------------------------------------------------------------
# parse_client_order_id
# ---------------------------------------------------------------------------


def test_parse_client_order_id_roundtrip():
    parsed = parse_client_order_id("tsmom_20260530_BTCUSDT_buy")
    assert parsed == {
        "prefix": "tsmom",
        "rebal_date": date(2026, 5, 30),
        "symbol_normalized": "BTCUSDT",
        "side": "buy",
    }


def test_parse_unknown_prefix_returns_none():
    assert parse_client_order_id("foo_20260530_BTCUSDT_buy") is None


def test_parse_malformed_date_returns_none():
    assert parse_client_order_id("tsmom_BADDATEX_BTCUSDT_buy") is None


def test_parse_invalid_side_returns_none():
    assert parse_client_order_id("tsmom_20260530_BTCUSDT_long") is None


def test_parse_empty_string_returns_none():
    assert parse_client_order_id("") is None


def test_parse_non_string_returns_none():
    assert parse_client_order_id(None) is None
    assert parse_client_order_id(12345) is None
