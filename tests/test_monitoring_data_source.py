"""Tests for the read-only journal data source.

Coverage focus:

* Fail-soft reading: corrupt lines, unknown schema versions, missing
  files, and non-dict JSON values must each be handled without
  raising. A single bad line must not blind the dashboard.
* Staleness bucketing matches the FRESH/STALE/DOWN/NO_DATA contract.
* Cache invalidation: a reused reader sees an updated file on the
  next query (mtime/size change).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trade_lab.monitoring.data_source import (
    JournalReader, KNOWN_SCHEMA_VERSIONS, Staleness, parse_iso,
)


def _cycle_entry(
    cycle_id: str = "c1",
    ended_at: str | None = None,
    schema_version: int = 1,
    signal_value: float = 1.0,
    sma_gate_open: bool = True,
    asof: str | None = None,
    outcome: str = "success",
    skipped_drift: float = 0.0,
) -> dict:
    """Build one journal entry matching schema_version=1."""
    if ended_at is None:
        ended_at = datetime.now(timezone.utc).isoformat()
    if asof is None:
        asof = ended_at
    return {
        "schema_version": schema_version,
        "cycle_id": cycle_id,
        "started_at": ended_at,
        "ended_at": ended_at,
        "duration_ms": 1000,
        "outcome": outcome,
        "error": None,
        "git_commit": "abc1234",
        "python_version": "3.12.3",
        "context": {"exchange": "binance", "sandbox": True,
                    "quote_currency": "USDT", "basket": ["BTC", "ETH"]},
        "signal": None if outcome == "failed" else {
            "asof": asof,
            "ladder_value": signal_value,
            "sma_gate_open": sma_gate_open,
            "per_lookback_states": {"28": 1, "60": 1},
            "basket_close": 12345.67,
            "asset_closes": {"BTC": 50000.0, "ETH": 3000.0},
        },
        "basket_close_series": None,
        "balance": None if outcome == "failed" else {
            "quote_currency": "USDT", "quote_total": 10000.0,
            "quote_free": 9500.0, "quote_used": 500.0,
            "asset_totals": {"BTC": 0.1, "ETH": 0.5},
        },
        "equity_usd": None if outcome == "failed" else 15000.0,
        "target_allocation": None if outcome == "failed" else {"BTC": 7500.0},
        "current_holdings_quote": None if outcome == "failed" else {"BTC": 5000.0},
        "orders_planned": None if outcome == "failed" else [],
        "orders_skipped": None if outcome == "failed" else [],
        "total_skipped_quote_drift": None if outcome == "failed" else skipped_drift,
    }


def _write_journal(path: Path, entries: list[dict | str]) -> None:
    """Write entries to a JSON Lines file. ``str`` entries are written
    verbatim (used for corrupt-line tests)."""
    with open(path, "wb") as f:
        for entry in entries:
            if isinstance(entry, dict):
                f.write(json.dumps(entry).encode("utf-8") + b"\n")
            else:
                f.write(entry.encode("utf-8") + b"\n")


# ---------------------------------------------------------------------------
# Missing / empty file
# ---------------------------------------------------------------------------


def test_missing_journal_returns_no_data(tmp_path):
    reader = JournalReader(tmp_path / "missing.jsonl")
    assert reader.latest_cycle() is None
    assert reader.cycles() == []
    assert reader.staleness(expected_interval_s=3600) is Staleness.NO_DATA
    assert reader.stats().total_lines == 0


def test_empty_journal_returns_no_data(tmp_path):
    journal = tmp_path / "j.jsonl"
    journal.write_text("")
    reader = JournalReader(journal)
    assert reader.latest_cycle() is None
    assert reader.staleness(expected_interval_s=3600) is Staleness.NO_DATA


def test_blank_lines_are_skipped(tmp_path):
    journal = tmp_path / "j.jsonl"
    journal.write_text("\n\n\n")
    reader = JournalReader(journal)
    assert reader.latest_cycle() is None
    assert reader.stats().total_lines == 0


# ---------------------------------------------------------------------------
# Normal reading
# ---------------------------------------------------------------------------


def test_reads_valid_entries(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("first"),
        _cycle_entry("second"),
        _cycle_entry("third"),
    ])
    reader = JournalReader(journal)
    assert reader.latest_cycle()["cycle_id"] == "third"
    assert [c["cycle_id"] for c in reader.cycles()] == ["first", "second", "third"]
    assert reader.stats().valid_cycles == 3


def test_cycles_n_returns_last_n(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry(f"c{i}") for i in range(10)
    ])
    reader = JournalReader(journal)
    last3 = reader.cycles(n=3)
    assert [c["cycle_id"] for c in last3] == ["c7", "c8", "c9"]


def test_cycles_zero_or_negative_returns_empty(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_cycle_entry("c1")])
    reader = JournalReader(journal)
    assert reader.cycles(n=0) == []
    assert reader.cycles(n=-1) == []


# ---------------------------------------------------------------------------
# Fail-soft on corruption
# ---------------------------------------------------------------------------


def test_corrupt_line_skipped_and_counted(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("good1"),
        "this is not json {{{",
        _cycle_entry("good2"),
    ])
    reader = JournalReader(journal)
    cycles = reader.cycles()
    assert [c["cycle_id"] for c in cycles] == ["good1", "good2"]
    stats = reader.stats()
    assert stats.valid_cycles == 2
    assert stats.corrupt_lines == 1
    assert stats.total_lines == 3


def test_non_dict_json_is_treated_as_corrupt(tmp_path):
    """A line like ``[1, 2, 3]`` parses as JSON but is not a cycle.
    The reader must reject it, not crash trying to call ``.get()``."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("good"),
        "[1, 2, 3]",
        "\"just a string\"",
    ])
    reader = JournalReader(journal)
    assert [c["cycle_id"] for c in reader.cycles()] == ["good"]
    assert reader.stats().corrupt_lines == 2


def test_unknown_schema_version_skipped_separately(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("known"),
        _cycle_entry("future", schema_version=99),
    ])
    reader = JournalReader(journal)
    assert [c["cycle_id"] for c in reader.cycles()] == ["known"]
    stats = reader.stats()
    assert stats.valid_cycles == 1
    assert stats.unknown_version_lines == 1
    assert stats.corrupt_lines == 0


def test_partial_tail_line_skipped_no_crash(tmp_path):
    """Mid-write crash scenario: last line is truncated. The reader
    should return the valid earlier lines and count the partial as
    corrupt."""
    journal = tmp_path / "j.jsonl"
    valid = json.dumps(_cycle_entry("good")).encode("utf-8")
    with open(journal, "wb") as f:
        f.write(valid + b"\n")
        f.write(b'{"cycle_id":"crashed","outc')  # no trailing newline
    reader = JournalReader(journal)
    assert [c["cycle_id"] for c in reader.cycles()] == ["good"]
    assert reader.stats().corrupt_lines == 1


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def _entry_at_age(seconds_ago: float, **kwargs) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    return _cycle_entry(ended_at=ts, **kwargs)


def test_staleness_fresh_when_within_window(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_entry_at_age(60)])
    reader = JournalReader(journal)
    assert reader.staleness(expected_interval_s=3600) is Staleness.FRESH


def test_staleness_stale_when_past_window(tmp_path):
    journal = tmp_path / "j.jsonl"
    # 2 hours ago, expected 1 hour → 2.0× → STALE (between 1.5× and 10×)
    _write_journal(journal, [_entry_at_age(7200)])
    reader = JournalReader(journal)
    assert reader.staleness(expected_interval_s=3600) is Staleness.STALE


def test_staleness_down_when_way_past_window(tmp_path):
    journal = tmp_path / "j.jsonl"
    # 20 hours ago, expected 1 hour → 20× → DOWN
    _write_journal(journal, [_entry_at_age(72000)])
    reader = JournalReader(journal)
    assert reader.staleness(expected_interval_s=3600) is Staleness.DOWN


def test_staleness_uses_latest_entry_not_average(tmp_path):
    """Multiple cycles; staleness is based on the most recent end_at."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _entry_at_age(72000, cycle_id="old"),
        _entry_at_age(60, cycle_id="new"),
    ])
    reader = JournalReader(journal)
    assert reader.staleness(expected_interval_s=3600) is Staleness.FRESH


# ---------------------------------------------------------------------------
# Signal history
# ---------------------------------------------------------------------------


def test_signal_history_excludes_failed_cycles(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("ok"),
        _cycle_entry("failed", outcome="failed"),
    ])
    reader = JournalReader(journal)
    hist = reader.signal_history(days=365)
    assert len(hist) == 1
    asof, value, gate_open = hist[0]
    assert value == 1.0


def test_signal_history_excludes_old_entries(tmp_path):
    journal = tmp_path / "j.jsonl"
    old_asof = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    recent_asof = datetime.now(timezone.utc).isoformat()
    _write_journal(journal, [
        _cycle_entry("old", asof=old_asof),
        _cycle_entry("new", asof=recent_asof),
    ])
    reader = JournalReader(journal)
    hist = reader.signal_history(days=30)
    assert len(hist) == 1


# ---------------------------------------------------------------------------
# Cumulative drift
# ---------------------------------------------------------------------------


def test_cumulative_skipped_drift_sums_all_cycles(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("c1", skipped_drift=5.0),
        _cycle_entry("c2", skipped_drift=2.5),
        _cycle_entry("failed", outcome="failed"),
    ])
    reader = JournalReader(journal)
    assert reader.cumulative_skipped_drift() == 7.5


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def test_reader_picks_up_new_entries_after_mtime_change(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_cycle_entry("first")])
    reader = JournalReader(journal)
    assert reader.latest_cycle()["cycle_id"] == "first"

    time.sleep(0.01)  # ensure mtime tick
    with open(journal, "ab") as f:
        f.write(json.dumps(_cycle_entry("second")).encode("utf-8") + b"\n")

    assert reader.latest_cycle()["cycle_id"] == "second"


def test_reader_picks_up_deletion(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_cycle_entry("first")])
    reader = JournalReader(journal)
    assert reader.latest_cycle() is not None
    journal.unlink()
    assert reader.latest_cycle() is None
    assert reader.staleness(expected_interval_s=3600) is Staleness.NO_DATA


# ---------------------------------------------------------------------------
# parse_iso helper
# ---------------------------------------------------------------------------


def test_parse_iso_with_offset():
    dt = parse_iso("2026-05-28T12:00:00+00:00")
    assert dt.tzinfo is not None


def test_parse_iso_with_z_suffix():
    dt = parse_iso("2026-05-28T12:00:00Z")
    assert dt.tzinfo is not None


def test_parse_iso_naive_treated_as_utc():
    dt = parse_iso("2026-05-28T12:00:00")
    assert dt.tzinfo == timezone.utc


def test_known_schema_versions_includes_one():
    assert 1 in KNOWN_SCHEMA_VERSIONS
