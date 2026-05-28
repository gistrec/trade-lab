"""Deterministic client order ID generation for idempotent placement.

The clientOrderId is a function of the *intent* (rebalance date, symbol,
side), not the placement attempt. Restarting at the same date produces
the same ID; the exchange returns the existing order on duplicate-create
instead of placing a second one.

Format
======
``tsmom_{YYYYMMDD}_{SYMBOL_NORMALIZED}_{side}``

* ``YYYYMMDD`` — UTC date of the rebalance decision. NEVER local time.
* ``SYMBOL_NORMALIZED`` — symbol with slash removed (Binance rejects
  ``/`` in clientOrderId).
* ``side`` — lowercase ``buy`` or ``sell``.

Length ≤32 characters for Binance compatibility.

Hard constraint
===============
The format is part of the v2 journal schema contract. Changing it
retroactively orphans every clientOrderId in the state store from its
corresponding exchange order: the bot will treat existing exchange
orders as unknown, and may place duplicates. DO NOT change without a
schema bump AND a manual state reconciliation step (drain all open
orders → wipe state → bump format).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional


ID_PREFIX = "tsmom"
MAX_ID_LEN = 32

_SYMBOL_RE = re.compile(r"^([A-Z0-9]{2,10})/([A-Z0-9]{2,10})$")
_ID_RE = re.compile(
    rf"^{ID_PREFIX}_(\d{{8}})_([A-Z0-9]{{4,20}})_(buy|sell)$"
)


def make_client_order_id(
    rebal_date: date,
    symbol: str,
    side: str,
) -> str:
    """Generate the deterministic client order ID for a rebalance intent.

    ``rebal_date`` must be a :class:`datetime.date`. Passing a
    :class:`datetime.datetime` is rejected to force callers to be
    explicit about timezone — they must call
    ``datetime.now(timezone.utc).date()`` themselves.
    """
    if isinstance(rebal_date, datetime):
        raise ValueError(
            "rebal_date must be datetime.date, not datetime; "
            "pass datetime.now(timezone.utc).date() explicitly."
        )
    if not isinstance(rebal_date, date):
        raise ValueError(
            f"rebal_date must be date, got {type(rebal_date).__name__}"
        )
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    normalized = normalize_symbol(symbol)
    date_str = rebal_date.strftime("%Y%m%d")
    coid = f"{ID_PREFIX}_{date_str}_{normalized}_{side}"

    if len(coid) > MAX_ID_LEN:
        raise ValueError(
            f"Generated client order ID exceeds {MAX_ID_LEN} chars: "
            f"{coid!r} ({len(coid)} chars). Symbol likely too long; "
            "the cap is a Binance constraint."
        )
    return coid


def normalize_symbol(symbol: str) -> str:
    """Convert ``BTC/USDT`` -> ``BTCUSDT``. Validates BASE/QUOTE shape."""
    if not isinstance(symbol, str):
        raise ValueError(f"symbol must be str, got {type(symbol).__name__}")
    m = _SYMBOL_RE.match(symbol)
    if m is None:
        raise ValueError(
            f"symbol {symbol!r} does not match BASE/QUOTE format "
            "(both 2-10 uppercase alphanumeric)."
        )
    return m.group(1) + m.group(2)


def parse_client_order_id(coid) -> Optional[dict]:
    """Reverse-parse for monitoring / diagnostics.

    Returns ``{"prefix", "rebal_date", "symbol_normalized", "side"}`` or
    None if the ID doesn't match the expected format. Tolerates
    non-string input so monitoring code can call it without prior type
    checks.
    """
    if not isinstance(coid, str):
        return None
    m = _ID_RE.match(coid)
    if m is None:
        return None
    try:
        rebal_date = datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None
    return {
        "prefix": ID_PREFIX,
        "rebal_date": rebal_date,
        "symbol_normalized": m.group(2),
        "side": m.group(3),
    }
