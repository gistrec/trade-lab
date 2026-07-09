"""Tests for the read-only Netdata health server (``ops/health_server.py``).

The decision logic (``evaluate_heartbeat`` / ``evaluate_daily``) is a pure
function of a :class:`JournalReader` and an injected ``now``, so it is tested
directly against temp journal files with a frozen clock — no wall-clock
flakiness. One end-to-end test starts the real HTTP server on an ephemeral
port and asserts the status codes a Netdata ``httpcheck`` job would see.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from pathlib import Path

import pytest

import health_server as hs
from trade_lab.monitoring.data_source import JournalReader

NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)


def _entry(
    *,
    ended_at: datetime,
    outcome: str = "success",
    live: bool,
    cycle_id: str = "c1",
    orders_executed=None,
    mode=None,
) -> dict:
    """Build a minimal schema-v2 journal entry the reader will accept.

    ``mode`` defaults from ``live`` but can be set explicitly to model a live
    run that failed before placing an order (mode='live', orders_executed=None).
    """
    if live and orders_executed is None:
        orders_executed = []  # a list (even empty) marks a live cycle
    if mode is None:
        mode = "live" if live else "dry_run"
    return {
        "schema_version": 2,
        "cycle_id": cycle_id,
        "started_at": ended_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_ms": 1000,
        "outcome": outcome,
        "context": {"mode": mode, "exchange": "binance", "sandbox": True},
        "orders_executed": orders_executed,  # None => dry-run, list => live
    }


def _journal(tmp_path: Path, entries: list[dict]) -> JournalReader:
    path = tmp_path / "cycles.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return JournalReader(path)


# --------------------------------------------------------------------------
# heartbeat
# --------------------------------------------------------------------------

def test_heartbeat_no_journal_file(tmp_path):
    reader = JournalReader(tmp_path / "does_not_exist.jsonl")
    r = hs.evaluate_heartbeat(reader, NOW, hs.DEFAULT_HEARTBEAT_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "no cycles" in r.reason


def test_heartbeat_fresh_dry_run_ok(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(minutes=30), live=False),
    ])
    r = hs.evaluate_heartbeat(reader, NOW, hs.DEFAULT_HEARTBEAT_MAX_AGE_S)
    assert r.ok and r.status_code == 200


def test_heartbeat_stale_fails(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=13), live=False),
    ])
    r = hs.evaluate_heartbeat(reader, NOW, hs.DEFAULT_HEARTBEAT_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "no cycle in" in r.reason


def test_heartbeat_unreadable_journal_is_a_directory(tmp_path):
    # A directory at the journal path -> ReadStats.read_error, not a crash.
    (tmp_path / "cycles.jsonl").mkdir()
    reader = JournalReader(tmp_path / "cycles.jsonl")
    r = hs.evaluate_heartbeat(reader, NOW, hs.DEFAULT_HEARTBEAT_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "unreadable" in r.reason


# --------------------------------------------------------------------------
# daily live
# --------------------------------------------------------------------------

def test_daily_dry_run_only_has_no_live_cycle(tmp_path):
    # 6-hourly dry-runs keep the journal warm but no live order cron ran.
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(minutes=10), live=False),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "no live" in r.reason


def test_daily_fresh_live_success_ok(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=3), outcome="success", live=True),
        _entry(ended_at=NOW - timedelta(minutes=5), live=False),  # later dry-run
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert r.ok and r.status_code == 200
    assert r.detail["outcome"] == "success"


def test_daily_reconstruction_alone_is_not_healthy(tmp_path):
    # A reconstruction cycle proves recovery of a PRIOR cycle's orders, not
    # that today's placement ran — it must not satisfy /healthz/daily.
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=2), outcome="reconstructed",
               live=True),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "no live" in r.reason


def test_daily_reconstruction_does_not_mask_stale_main(tmp_path):
    # Fresh reconstruction + a stale main success must still 503 (stale),
    # not be rescued to 200 by the fresh reconstruction cycle.
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=30), outcome="success",
               live=True, cycle_id="old_main"),
        _entry(ended_at=NOW - timedelta(hours=1), outcome="reconstructed",
               live=True, cycle_id="fresh_recon"),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "ago" in r.reason  # keyed off the main success, not the reconstruction


def test_daily_reconstruction_plus_fresh_main_success_ok(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=2), outcome="reconstructed",
               live=True, cycle_id="recon"),
        _entry(ended_at=NOW - timedelta(hours=1), outcome="success",
               live=True, cycle_id="main_ok"),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert r.ok and r.status_code == 200


def test_daily_stale_live_fails(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=30), outcome="success", live=True),
        _entry(ended_at=NOW - timedelta(minutes=5), live=False),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "ago" in r.reason


@pytest.mark.parametrize("bad", ["failed", "unknown_orders", "partial"])
def test_daily_bad_outcome_fails(tmp_path, bad):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=1), outcome=bad, live=True),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert bad in r.reason


def test_daily_live_outcome_failure_when_not_latest(tmp_path):
    # A failed LIVE cycle (orders placed then failed) that is NOT the most
    # recent entry — a later dry-run succeeded. The live-outcome check, not
    # the fresh-failure catch, must still flag it.
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=3), outcome="failed", live=True,
               cycle_id="live_fail"),
        _entry(ended_at=NOW - timedelta(minutes=5), outcome="success",
               live=False, cycle_id="later_dry"),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "last live outcome=failed" in r.reason


def test_daily_catches_fresh_failure_before_placement(tmp_path):
    # THE blind spot, now closed by the mode marker: a live run that failed
    # *before* placing an order writes orders_executed=None but context
    # mode=='live', so it IS the latest main-live cycle and its failed outcome
    # trips 503 — even though a fresh success exists 20h back.
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=20), outcome="success",
               live=True, cycle_id="yesterday_ok"),
        _entry(ended_at=NOW - timedelta(minutes=10), outcome="failed",
               live=False, mode="live", orders_executed=None,
               cycle_id="today_failed_early"),
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert not r.ok and r.status_code == 503
    assert "last live outcome=failed" in r.reason


def test_daily_dry_run_failure_does_not_page(tmp_path):
    # The mode marker's payoff: a benign 6-hourly DRY-RUN failure after a healthy
    # live run must NOT page /daily (it is not a live-order failure).
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=3), outcome="success",
               live=True, cycle_id="live_ok"),
        _entry(ended_at=NOW - timedelta(minutes=10), outcome="failed",
               live=False, cycle_id="dry_blip"),  # mode='dry_run'
    ])
    r = hs.evaluate_daily(reader, NOW, hs.DEFAULT_DAILY_MAX_AGE_S)
    assert r.ok and r.status_code == 200


# --------------------------------------------------------------------------
# end-to-end over a real socket (what Netdata httpcheck actually sees)
# --------------------------------------------------------------------------

def _get(port: int, path: str) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        return resp.status, body
    finally:
        conn.close()


def test_end_to_end_status_codes(tmp_path):
    path = tmp_path / "cycles.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in [
        _entry(ended_at=datetime.now(timezone.utc) - timedelta(minutes=20),
               outcome="success", live=True),
    ]))
    cfg = hs.Config(journal_path=str(path), host="127.0.0.1", port=0)
    httpd = hs.build_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        code, body = _get(port, "/healthz")
        assert code == 200 and body["ok"] is True
        code, body = _get(port, "/healthz/daily")
        assert code == 200 and body["check"] == "daily_live"
        code, body = _get(port, "/nope")
        assert code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_daily_disabled_returns_ok_during_observation_phase(tmp_path):
    """A mainnet journal fed only by dry-run crons (observation phase,
    no live cron yet) must not keep /healthz/daily permanently 503 —
    that trains the operator to ignore the endpoint. The disable is an
    explicit config statement and self-describes in the reason."""
    path = tmp_path / "cycles_mainnet.jsonl"
    path.write_text(json.dumps(_entry(
        ended_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        outcome="success", live=False,
    )) + "\n")
    cfg = hs.Config(journal_path=str(path), host="127.0.0.1", port=0,
                    daily_disabled=True)
    httpd = hs.build_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        code, body = _get(port, "/healthz/daily")
        assert code == 200 and body["ok"] is True
        assert "disabled" in body["reason"]
        # The heartbeat is NOT disabled — it still watches the dry-runs.
        code, body = _get(port, "/healthz")
        assert code == 200 and body["ok"] is True
        # The human summary carries the same disabled marker.
        code, body = _get(port, "/")
        assert code == 200 and "disabled" in body["daily_live"]["reason"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_daily_disabled_self_invalidates_on_live_cycles(tmp_path):
    """A stale DAILY_DISABLED flag must not mask a dead live cron: once
    the journal contains main live cycles, the real verdict returns."""
    now = datetime.now(timezone.utc)
    path = tmp_path / "cycles_mainnet.jsonl"
    path.write_text(json.dumps(_entry(
        ended_at=now - timedelta(days=3), outcome="success", live=True,
    )) + "\n")
    r = hs.evaluate_daily_disabled(
        JournalReader(path), now, hs.DEFAULT_DAILY_MAX_AGE_S,
    )
    assert r.ok is False
    assert "stale" in r.reason


def test_daily_disabled_env_parsing(monkeypatch):
    monkeypatch.setenv("TRADE_LAB_HEALTH_DAILY_DISABLED", "true")
    assert hs.Config.from_env().daily_disabled is True
    monkeypatch.setenv("TRADE_LAB_HEALTH_DAILY_DISABLED", "false")
    assert hs.Config.from_env().daily_disabled is False
    monkeypatch.delenv("TRADE_LAB_HEALTH_DAILY_DISABLED")
    assert hs.Config.from_env().daily_disabled is False
