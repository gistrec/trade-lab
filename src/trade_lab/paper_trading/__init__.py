"""Validation forward-test harness.

This is **not** the production execution layer (`trade_lab.execution.*`),
which paper-trades on Binance testnet via real CCXT order placement.
This is the **research harness for Validation Tests 3-4**: it
records what the frozen strategy *would* do every day plus an
immutable, content-hashed snapshot of the data it saw, so the
look-ahead detector (Test 4) can replay backtests against the exact
bytes used in each live decision.

Design principle: the live ledger here exists to be cross-checked by
``findings/validation_behavioral_fingerprint.md`` (Test 4). Any
optimization that breaks reproducibility — non-deterministic
serialization, mutable shared snapshots, in-place file
overwrites — defeats that purpose, no matter how convenient.
"""
from .harness import HarnessError, run_paper_trading_cycle
from .journal import HarnessLogRow, append_row, is_already_logged, read_log
from .vintage_store import (
    canonical_serialize,
    content_hash,
    load_vintage,
    store_vintage,
)

__all__ = [
    "HarnessError",
    "HarnessLogRow",
    "append_row",
    "canonical_serialize",
    "content_hash",
    "is_already_logged",
    "load_vintage",
    "read_log",
    "run_paper_trading_cycle",
    "store_vintage",
]
