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

import pytest

from trade_lab.monitoring.data_source import (
    JournalReader, KNOWN_SCHEMA_VERSIONS, Staleness,
    cycle_orders_executed, is_live_cycle, max_inter_cycle_gap_seconds,
    open_order_incidents, parse_iso, recent_incidents,
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


def test_signal_history_tolerates_null_ladder_value(tmp_path):
    """A cycle whose signal.ladder_value is JSON null must not crash
    signal_history: .get(key, default) does NOT catch a present null, so
    float(None) would raise TypeError and blank the whole dashboard. The
    module contract is to tolerate null/garbage journal fields
    (regression: R4)."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        _cycle_entry("good", signal_value=1.0),
        _cycle_entry("nullish", signal_value=None),
    ])
    reader = JournalReader(journal)
    history = reader.signal_history(days=30)  # must not raise
    ladders = [v for _, v, _ in history]
    assert 1.0 in ladders
    assert 0.0 in ladders          # null coerced, not crashed
    assert len(history) == 2


def test_signal_history_tolerates_garbage_ladder_value(tmp_path):
    """A non-numeric ladder_value (corrupt/hand-edited row) must not
    crash signal_history either (regression: R4)."""
    journal = tmp_path / "j.jsonl"
    good = _cycle_entry("good", signal_value=1.0)
    garbage = _cycle_entry("garbage", signal_value=1.0)
    garbage["signal"]["ladder_value"] = "not-a-number"
    _write_journal(journal, [good, garbage])
    reader = JournalReader(journal)
    history = reader.signal_history(days=30)  # must not raise
    assert any(v == 1.0 for _, v, _ in history)


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


def test_known_schema_versions_includes_one_and_two():
    """v1 is the dry-run-only shape; v2 adds orders_executed."""
    assert 1 in KNOWN_SCHEMA_VERSIONS
    assert 2 in KNOWN_SCHEMA_VERSIONS


def test_v2_entry_with_orders_executed_reads_correctly(tmp_path):
    """A schema_version=2 entry with orders_executed must parse and be
    surfaced by latest_cycle / cycles like any v1 entry."""
    journal = tmp_path / "j.jsonl"
    entry = _cycle_entry("v2-cycle", schema_version=2)
    entry["orders_executed"] = [
        {"client_order_id": "tsmom_20260530_BTCUSDT_buy",
         "exchange_order_id": "exch-1", "symbol": "BTC/USDT",
         "side": "buy", "intended_amount": 0.001,
         "terminal_status": "closed", "filled_amount": 0.001,
         "filled_notional_quote": 50.0, "average_price": 50000.0,
         "fees_paid_quote": 0.05, "placed_at": "...",
         "terminal_at": "...", "error": None},
    ]
    _write_journal(journal, [entry])
    reader = JournalReader(journal)
    latest = reader.latest_cycle()
    assert latest["cycle_id"] == "v2-cycle"
    assert latest["schema_version"] == 2
    assert len(latest["orders_executed"]) == 1
    assert latest["orders_executed"][0]["terminal_status"] == "closed"


def test_v1_entry_without_orders_executed_still_reads(tmp_path):
    """Backward compat — old v1 entries written before the schema bump
    have no orders_executed field at all and must still parse."""
    journal = tmp_path / "j.jsonl"
    entry = _cycle_entry("v1-cycle", schema_version=1)
    # Explicitly omit orders_executed; v1 readers never had it.
    assert "orders_executed" not in entry
    _write_journal(journal, [entry])
    reader = JournalReader(journal)
    latest = reader.latest_cycle()
    assert latest["cycle_id"] == "v1-cycle"


# ---------------------------------------------------------------------------
# cycle_orders_executed helper
# ---------------------------------------------------------------------------


def test_cycle_orders_executed_v1_returns_empty():
    """v1 cycles never had orders_executed — helper returns []."""
    cycle = _cycle_entry("v1", schema_version=1)
    assert "orders_executed" not in cycle
    assert cycle_orders_executed(cycle) == []


def test_cycle_orders_executed_v2_none_returns_empty():
    """A dry-run v2 cycle writes orders_executed=None — same as []."""
    cycle = _cycle_entry("v2-dry", schema_version=2)
    cycle["orders_executed"] = None
    assert cycle_orders_executed(cycle) == []


def test_cycle_orders_executed_v2_empty_list_returns_empty():
    """signal=0 cycle plans no orders → orders_executed=[]."""
    cycle = _cycle_entry("v2-noop", schema_version=2)
    cycle["orders_executed"] = []
    assert cycle_orders_executed(cycle) == []


def test_cycle_orders_executed_v2_populated_returns_list():
    cycle = _cycle_entry("v2-with-orders", schema_version=2)
    cycle["orders_executed"] = [
        {"client_order_id": "tsmom_20260530_BTCUSDT_buy",
         "terminal_status": "closed", "filled_amount": 0.001},
    ]
    out = cycle_orders_executed(cycle)
    assert len(out) == 1
    assert out[0]["terminal_status"] == "closed"


def test_parse_iso_returns_none_on_null_and_garbage():
    """parse_iso is total: journal fields are external input, a JSON
    null or malformed string degrades one value instead of raising
    AttributeError through the dashboard."""
    assert parse_iso(None) is None
    assert parse_iso(123) is None
    assert parse_iso("not a timestamp") is None
    assert parse_iso("") is None


def test_staleness_no_data_on_null_ended_at(tmp_path):
    """A cycle with ended_at: null must bucket as NO_DATA, not crash."""
    import json as _json

    path = tmp_path / "j.jsonl"
    path.write_text(_json.dumps({
        "schema_version": 2, "cycle_id": "x", "ended_at": None,
    }) + "\n")
    reader = JournalReader(path)
    assert reader.staleness(3600) is Staleness.NO_DATA


# ---------------------------------------------------------------------------
# DRY vs LIVE discrimination (Theme 1: hourly dry-runs must not mask a dead
# daily order cron)
# ---------------------------------------------------------------------------


def _live_cycle(cycle_id="live", ended_at=None, outcome="success",
                orders_executed=None):
    """A live (real-order) cycle sets orders_executed to a list."""
    c = _cycle_entry(cycle_id, ended_at=ended_at, outcome=outcome,
                     schema_version=2)
    c["orders_executed"] = orders_executed if orders_executed is not None else []
    return c


def test_is_live_cycle_true_only_when_orders_executed_is_a_list():
    assert is_live_cycle({"orders_executed": []}) is True
    assert is_live_cycle({"orders_executed": [{"symbol": "BTC/USDT"}]}) is True
    # dry-run: explicit None or absent → not live
    assert is_live_cycle({"orders_executed": None}) is False
    assert is_live_cycle({}) is False


def test_latest_live_cycle_ignores_dry_runs(tmp_path):
    """The core Theme-1 property: with ~24 dry-runs after the last live
    cycle, latest_cycle() is a dry-run but latest_live_cycle() finds the
    real one."""
    journal = tmp_path / "j.jsonl"
    entries = [_live_cycle("real1"), _live_cycle("real2")]
    for i in range(24):  # a day of hourly dry-runs on top
        entries.append(_cycle_entry(f"dry{i}"))
    _write_journal(journal, entries)
    reader = JournalReader(journal)
    assert reader.latest_cycle()["cycle_id"] == "dry23"        # dry-run
    assert reader.latest_live_cycle()["cycle_id"] == "real2"   # real order cron


def test_latest_live_cycle_none_when_only_dry_runs(tmp_path):
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_cycle_entry("dry1"), _cycle_entry("dry2")])
    reader = JournalReader(journal)
    assert reader.latest_live_cycle() is None


def test_recent_incidents_scans_window_not_just_latest():
    """A failed LIVE cycle followed by dry-run successes is invisible to a
    latest-only check but must appear in the incident scan."""
    cycles = [
        _cycle_entry("ok0"),
        _live_cycle("boom", outcome="failed"),
        _cycle_entry("dry_after"),   # later success would hide the failure
    ]
    incidents = recent_incidents(cycles)
    assert [i["cycle_id"] for i in incidents] == ["boom"]
    assert incidents[0]["outcome"] == "failed"
    assert incidents[0]["mode"] == "LIVE"


def test_recent_incidents_excludes_reconstructed_recovery():
    cycles = [
        {"outcome": "reconstructed", "cycle_id": "recov"},
        {"outcome": "success", "cycle_id": "ok"},
    ]
    assert recent_incidents(cycles) == []


def test_open_order_incidents_lists_non_resolved_orders():
    cycles = [
        _live_cycle("c1", orders_executed=[
            {"terminal_status": "closed", "client_order_id": "a",
             "symbol": "BTC/USDT", "side": "buy"},
            {"terminal_status": "lost_track", "client_order_id": "b",
             "symbol": "ETH/USDT", "side": "sell"},
        ]),
    ]
    out = open_order_incidents(cycles)
    assert len(out) == 1
    assert out[0]["status"] == "lost_track"
    assert out[0]["symbol"] == "ETH/USDT"
    assert out[0]["side"] == "SELL"


def test_open_order_incidents_empty_when_all_resolved():
    cycles = [
        _live_cycle("c1", orders_executed=[
            {"terminal_status": "closed", "client_order_id": "a"},
            {"terminal_status": "canceled", "client_order_id": "b"},
        ]),
    ]
    assert open_order_incidents(cycles) == []


def test_max_inter_cycle_gap_detects_mid_window_pause():
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    cycles = [
        _cycle_entry("c1", ended_at=base.isoformat()),
        _cycle_entry("c2", ended_at=(base + timedelta(hours=1)).isoformat()),
        # 3-day pause here
        _cycle_entry("c3", ended_at=(base + timedelta(days=3)).isoformat()),
    ]
    gap = max_inter_cycle_gap_seconds(cycles)
    assert gap == pytest.approx((timedelta(days=3) - timedelta(hours=1)).total_seconds())


def test_max_inter_cycle_gap_none_with_under_two_timestamps():
    assert max_inter_cycle_gap_seconds([]) is None
    assert max_inter_cycle_gap_seconds([_cycle_entry("c1")]) is None


def test_signal_history_skips_null_asof(tmp_path):
    import json as _json

    rows = [
        {"schema_version": 2, "signal": {"asof": None, "ladder_value": 1.0,
                                          "sma_gate_open": True}},
        {"schema_version": 2, "signal": {
            "asof": "2099-01-01T00:00:00+00:00", "ladder_value": 0.5,
            "sma_gate_open": False}},
    ]
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    reader = JournalReader(path)
    hist = reader.signal_history(days=10**6)
    assert len(hist) == 1
    assert hist[0][1] == 0.5
