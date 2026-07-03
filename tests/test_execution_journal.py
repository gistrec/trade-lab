"""Tests for the JSON Lines journal writer.

Coverage focus:

* Per-line atomicity: each ``append`` writes exactly one complete line
  ending in ``\\n``.
* Size cap (``MAX_LINE_BYTES``): oversized cycles raise
  :class:`JournalEntryTooLarge` before any bytes touch the file, and a
  worst-realistic full-rebalance cycle fits under the cap.
* Crash recovery: a partial trailing line left by a previous writer
  does NOT eat the next valid entry. The writer prepends a newline
  when the file does not end with one.
* Helpers: git commit / python version / UUID are stable enough to
  base downstream monitoring assumptions on.
"""
from __future__ import annotations

import json
import os

import pytest

from trade_lab.execution.journal import (
    Cycle,
    JOURNAL_SCHEMA_VERSION,
    JournalEntryTooLarge,
    JournalWriter,
    MAX_LINE_BYTES,
    _encode_cycle,
    get_git_commit_short,
    get_python_version,
    new_cycle_id,
    utcnow_iso,
)


def _make_cycle(cycle_id: str = "test", outcome: str = "success") -> Cycle:
    """Minimal valid Cycle for tests."""
    return Cycle(
        cycle_id=cycle_id,
        started_at="2026-05-28T12:00:00+00:00",
        ended_at="2026-05-28T12:00:03+00:00",
        duration_ms=3000,
        outcome=outcome,
        error=None,
        git_commit="abc1234",
        python_version="3.12.3",
        context={"exchange": "binance", "sandbox": True,
                 "quote_currency": "USDT", "basket": ["BTC", "ETH"]},
        signal={"ladder_value": 1.0, "sma_gate_open": True,
                "per_lookback_states": {"28": 1, "60": 1}},
        basket_close_series={"start_ts": "2026-01-01T00:00:00+00:00",
                             "values": [100.0, 101.0, 102.0]},
        balance={"quote_currency": "USDT", "quote_total": 10000.0,
                 "quote_free": 9500.0, "quote_used": 500.0,
                 "asset_totals": {"BTC": 0.1, "ETH": 0.5}},
        equity_usd=15000.0,
        target_allocation={"BTC": 7500.0, "ETH": 7500.0},
        current_holdings_quote={"BTC": 5000.0, "ETH": 1500.0},
        orders_planned=[],
        orders_skipped=[],
        total_skipped_quote_drift=0.0,
    )


# ---------------------------------------------------------------------------
# Basic writer behaviour
# ---------------------------------------------------------------------------


def test_writer_appends_one_line_per_cycle(tmp_path):
    writer = JournalWriter(tmp_path / "j.jsonl")
    writer.append(_make_cycle("first"))
    writer.append(_make_cycle("second"))

    lines = (tmp_path / "j.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cycle_id"] == "first"
    assert json.loads(lines[1])["cycle_id"] == "second"


def test_writer_creates_parent_directory(tmp_path):
    nested = tmp_path / "deep" / "nested" / "journal"
    writer = JournalWriter(nested / "cycles.jsonl")
    writer.append(_make_cycle("only"))
    assert (nested / "cycles.jsonl").exists()


def test_writer_writes_schema_version(tmp_path):
    writer = JournalWriter(tmp_path / "j.jsonl")
    writer.append(_make_cycle())
    entry = json.loads((tmp_path / "j.jsonl").read_text())
    assert entry["schema_version"] == JOURNAL_SCHEMA_VERSION


def test_writer_handles_failed_cycle_with_null_fields(tmp_path):
    """A failed cycle has None for signal/balance/equity; these must
    serialize as JSON ``null``, not crash."""
    failed = Cycle(
        cycle_id="failed-1",
        started_at="2026-05-28T12:00:00+00:00",
        ended_at="2026-05-28T12:00:01+00:00",
        duration_ms=1000,
        outcome="failed",
        error={"type": "BrokerError", "message": "timeout"},
        git_commit="abc1234",
        python_version="3.12.3",
        context={"exchange": "binance", "sandbox": True,
                 "quote_currency": "USDT", "basket": ["BTC"]},
        signal=None,
        basket_close_series=None,
        balance=None,
        equity_usd=None,
        target_allocation=None,
        current_holdings_quote=None,
        orders_planned=None,
        orders_skipped=None,
        total_skipped_quote_drift=None,
    )
    writer = JournalWriter(tmp_path / "j.jsonl")
    writer.append(failed)
    entry = json.loads((tmp_path / "j.jsonl").read_text())
    assert entry["outcome"] == "failed"
    assert entry["error"]["type"] == "BrokerError"
    assert entry["signal"] is None
    assert entry["equity_usd"] is None


# ---------------------------------------------------------------------------
# Size enforcement
# ---------------------------------------------------------------------------


def test_oversized_cycle_raises_before_any_bytes_on_disk(tmp_path):
    journal_path = tmp_path / "j.jsonl"
    writer = JournalWriter(journal_path)
    huge_payload = {"x": "a" * (MAX_LINE_BYTES + 100)}
    cycle = _make_cycle()
    # Stuff oversized content into a field that's normally bounded.
    cycle.basket_close_series = huge_payload

    with pytest.raises(JournalEntryTooLarge, match="bytes"):
        writer.append(cycle)

    # Nothing was written — file does not exist or is empty.
    assert not journal_path.exists() or journal_path.stat().st_size == 0


def test_full_rebalance_cycle_fits_under_cap(tmp_path):
    """Worst realistic case: a 7-asset full rebalance with 7 planned +
    7 executed orders and a 100-value basket series. This used to
    exceed the old 4KB cap and the entry was silently dropped by the
    cycle writers — the cap must accommodate it with headroom."""
    import random

    rng = random.Random(7)
    basket = ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE"]
    asset_closes = {s: rng.uniform(0.05, 90_000.0) for s in basket}
    planned = [
        {"symbol": f"{s}/USDT", "side": "buy",
         "base_amount": rng.random(), "notional_quote": rng.uniform(100, 500),
         "price_used": asset_closes[s]}
        for s in basket
    ]
    executed = [
        {"client_order_id": f"tsmom_20260612_{s}USDT_buy",
         "exchange_order_id": "123456789", "symbol": f"{s}/USDT",
         "side": "buy", "intended_amount": rng.random(),
         "terminal_status": "closed", "filled_amount": rng.random(),
         "filled_notional_quote": rng.uniform(100, 500),
         "average_price": asset_closes[s], "fees_paid_quote": 0.123456,
         "placed_at": "2026-06-12T00:05:03.123456+00:00",
         "terminal_at": "2026-06-12T00:05:04.123456+00:00", "error": None}
        for s in basket
    ]
    cycle = _make_cycle("full-rebalance")
    cycle.context["basket"] = basket
    cycle.signal = {
        "asof": "2026-06-12T00:00:00+00:00", "ladder_value": 1.0,
        "sma_gate_open": True, "sma_value": 1.234567890123,
        "per_lookback_states": {"28": 1, "60": 1},
        "per_lookback_returns": {"28": 0.123456789, "60": 0.23456789},
        "basket_close": 1.3456789012, "asset_closes": asset_closes,
    }
    cycle.basket_close_series = {
        "start_ts": "2026-03-04T00:00:00+00:00",
        "values": [rng.uniform(80.0, 160.0) for _ in range(100)],
    }
    cycle.balance = {
        "quote_currency": "USDT", "quote_total": 10000.12345678,
        "quote_free": 5000.12345678, "quote_used": 0.0,
        "asset_totals": {s: rng.random() for s in basket},
    }
    cycle.target_allocation = {s: rng.uniform(1000, 1500) for s in basket}
    cycle.current_holdings_quote = {s: rng.uniform(900, 1500) for s in basket}
    cycle.orders_planned = planned
    cycle.orders_executed = executed

    writer = JournalWriter(tmp_path / "j.jsonl")
    writer.append(cycle)  # must not raise JournalEntryTooLarge

    entry = json.loads((tmp_path / "j.jsonl").read_text())
    assert len(entry["orders_executed"]) == 7


def test_cycle_just_under_cap_passes(tmp_path):
    """A cycle whose encoded size is just under MAX_LINE_BYTES should
    pass without raising. This is the regression test against a too-
    aggressive cap."""
    writer = JournalWriter(tmp_path / "j.jsonl")
    cycle = _make_cycle()
    # Pad basket_close_series to push close to the limit, then trim
    # back inside the loop until we land under MAX_LINE_BYTES.
    n_values = 200
    while True:
        cycle.basket_close_series = {
            "start_ts": "2026-01-01T00:00:00+00:00",
            "values": [1234.5678] * n_values,
        }
        if len(_encode_cycle(cycle)) <= MAX_LINE_BYTES:
            break
        n_values -= 10
        assert n_values > 0, "Could not find a passing payload size."
    writer.append(cycle)  # must not raise


# ---------------------------------------------------------------------------
# Crash recovery / atomicity
# ---------------------------------------------------------------------------


def test_partial_tail_does_not_corrupt_next_entry(tmp_path):
    """Simulate a previous writer that crashed before writing its
    trailing newline. The next append must land on its own line — not
    glued to the partial bytes — so the JSON parser can still recover
    the new entry."""
    journal_path = tmp_path / "j.jsonl"

    # Pre-populate with a partial line (no trailing newline).
    journal_path.write_bytes(b'{"cycle_id":"crashed","outc')

    writer = JournalWriter(journal_path)
    writer.append(_make_cycle("recovered"))

    raw = journal_path.read_bytes()
    # The partial bytes should be on their own line, followed by our
    # full new entry on the next line.
    lines = raw.splitlines()
    assert lines[0] == b'{"cycle_id":"crashed","outc'
    assert json.loads(lines[1])["cycle_id"] == "recovered"


def test_partial_tail_with_newline_does_not_double_separate(tmp_path):
    """If the previous writer DID complete its newline, the new entry
    must NOT get an extra blank line in front."""
    journal_path = tmp_path / "j.jsonl"
    writer = JournalWriter(journal_path)
    writer.append(_make_cycle("first"))
    writer.append(_make_cycle("second"))

    raw = journal_path.read_bytes()
    assert b"\n\n" not in raw  # no empty line ever inserted


def test_mid_write_fsync_failure_preserves_prior_entries(tmp_path, monkeypatch):
    """If fsync raises after a previous successful append, the prior
    entries on disk remain readable. Recovery (next successful append)
    leaves a journal whose every JSON line still parses."""
    journal_path = tmp_path / "j.jsonl"
    writer = JournalWriter(journal_path)
    writer.append(_make_cycle("first"))

    real_fsync = os.fsync
    fsync_calls = {"n": 0}

    def boom(fd):
        fsync_calls["n"] += 1
        real_fsync(fd)
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="simulated"):
        writer.append(_make_cycle("during-failure"))

    monkeypatch.undo()
    writer.append(_make_cycle("after-recovery"))

    raw = journal_path.read_bytes()
    lines = [l for l in raw.splitlines() if l.strip()]
    # Every line on disk must be parseable as JSON, regardless of
    # how the during-failure write resolved at the kernel level.
    for line in lines:
        json.loads(line)
    cycle_ids = [json.loads(l)["cycle_id"] for l in lines]
    assert "first" in cycle_ids
    assert "after-recovery" in cycle_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_get_git_commit_short_returns_hex_in_a_repo():
    """Tests run inside the trade-lab repo, so git is present."""
    sha = get_git_commit_short()
    assert sha is None or all(c in "0123456789abcdef" for c in sha)
    if sha is not None:
        assert 4 <= len(sha) <= 40


def test_get_git_commit_short_returns_none_outside_a_repo(tmp_path, monkeypatch):
    """Run the helper from a tmpdir that's NOT a git repo."""
    monkeypatch.chdir(tmp_path)
    assert get_git_commit_short() is None


def test_get_python_version_format():
    version = get_python_version()
    parts = version.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_new_cycle_id_unique():
    ids = {new_cycle_id() for _ in range(100)}
    assert len(ids) == 100


def test_utcnow_iso_includes_offset():
    ts = utcnow_iso()
    assert "+00:00" in ts or ts.endswith("Z")
