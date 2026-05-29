"""Append-only journal invariants."""
from __future__ import annotations

import json

import pytest

from trade_lab.paper_trading.journal import (
    HarnessLogRow,
    append_row,
    get_row_for_date,
    is_already_logged,
    read_log,
)


def _row(date: str = "2026-05-29", ladder: float = 0.5) -> HarnessLogRow:
    return HarnessLogRow(
        date=date,
        config_hash="abc123",
        vintage_content_hash="def456",
        basket_close=100.0,
        sma_value=99.0,
        sma_gate_open=True,
        ladder_state=ladder,
        prior_ladder_state=0.0,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.05, "60": 0.02},
        target_weights={"BTC": 0.0714, "ETH": 0.0714},
        current_weights={"BTC": 0.0, "ETH": 0.0},
        intended_trades={"BTC": 0.0714, "ETH": 0.0714},
        portfolio_equity=10_000.0,
        daily_return=0.01,
        gross_position_return=0.005,
        net_position_return=0.0042,
    )


def test_append_then_read(tmp_path):
    log = tmp_path / "j.jsonl"
    append_row(_row(), log)
    rows = read_log(log)
    assert len(rows) == 1
    assert rows[0].date == "2026-05-29"
    assert rows[0].ladder_state == 0.5


def test_jsonl_is_valid_one_object_per_line(tmp_path):
    log = tmp_path / "j.jsonl"
    append_row(_row("2026-05-29"), log)
    append_row(_row("2026-05-30"), log)
    text = log.read_text()
    lines = [ln for ln in text.split("\n") if ln]
    assert len(lines) == 2
    for line in lines:
        # Each line must independently parse as JSON.
        json.loads(line)


def test_is_already_logged_detects_existing_date(tmp_path):
    log = tmp_path / "j.jsonl"
    assert not is_already_logged("2026-05-29", log)
    append_row(_row("2026-05-29"), log)
    assert is_already_logged("2026-05-29", log)
    assert not is_already_logged("2026-05-30", log)


def test_truncated_last_line_is_skipped(tmp_path):
    """Simulate a power-loss mid-write: the last line is truncated.
    read_log must skip it, not raise."""
    log = tmp_path / "j.jsonl"
    append_row(_row("2026-05-29"), log)
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"date": "2026-05-30", "partial')  # truncated
    rows = read_log(log)
    assert len(rows) == 1
    assert rows[0].date == "2026-05-29"


def test_get_row_for_date_returns_last_match(tmp_path):
    """If the same date somehow appears twice (operator hand-edit),
    return the latest written — this is read-side defense, not
    a guarantee that duplicates can occur in normal operation."""
    log = tmp_path / "j.jsonl"
    append_row(_row("2026-05-29", ladder=0.5), log)
    # Simulate a hand-appended duplicate (NOT expected in normal flow)
    append_row(_row("2026-05-29", ladder=1.0), log)
    found = get_row_for_date("2026-05-29", log)
    assert found is not None
    assert found.ladder_state == 1.0


def test_missing_log_returns_empty(tmp_path):
    log = tmp_path / "does_not_exist.jsonl"
    assert read_log(log) == []
    assert not is_already_logged("2026-05-29", log)
    assert get_row_for_date("2026-05-29", log) is None
