"""Tests for the Prometheus metrics exporter (``ops/metrics.py``) and the
health server's ``/metrics`` route.

``render_metrics`` is a pure function of a JournalReader + injected ``now``,
so it is asserted directly against temp journals. Totality (never crash on an
empty/corrupt journal) is a hard requirement — a raising exporter would break
the Netdata scrape.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from pathlib import Path

import health_server as hs
import metrics
from trade_lab.monitoring.data_source import JournalReader

NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _entry(*, ended_at, outcome="success", live, cycle_id="c1",
           duration_ms=1000, mode=None, equity=None, ladder=None,
           sma_gate=True):
    if mode is None:
        mode = "live" if live else "dry_run"
    e = {
        "schema_version": 2,
        "cycle_id": cycle_id,
        "started_at": ended_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_ms": duration_ms,
        "outcome": outcome,
        "context": {"mode": mode, "exchange": "binance", "sandbox": True},
        "orders_executed": [] if live else None,
    }
    if equity is not None:
        e["equity_usd"] = equity
    if ladder is not None:
        e["signal"] = {"asof": ended_at.isoformat(), "ladder_value": ladder,
                       "sma_gate_open": sma_gate}
    return e


def _journal(tmp_path: Path, entries) -> JournalReader:
    p = tmp_path / "cycles.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return JournalReader(p)


def _parse(text: str) -> dict[str, float]:
    """Parse simple (labelless) `name value` samples into a dict."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.partition(" ")
        if "{" not in name:
            out[name] = float(val)
    return out


def test_render_basic_shape(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=3), outcome="success", live=True,
               duration_ms=1500, equity=9800.0, ladder=1.0),
        # latest cycle: a dry-run heartbeat that also carries the current signal
        _entry(ended_at=NOW - timedelta(minutes=20), outcome="success",
               live=False, duration_ms=900, ladder=0.5, sma_gate=True),
    ])
    text = metrics.render_metrics(reader, NOW)
    m = _parse(text)
    assert m["tradelab_up"] == 1
    assert m["tradelab_journal_read_error"] == 0
    assert m["tradelab_journal_valid_cycles"] == 2
    # freshness keyed off the last cycle (the 20-min-old dry-run)
    assert abs(m["tradelab_last_cycle_age_seconds"] - 1200) < 2
    # live freshness keyed off the 3h-old live cycle
    assert abs(m["tradelab_last_live_cycle_age_seconds"] - 10800) < 2
    # equity comes from the latest SUCCESSFUL cycle with an equity field
    assert m["tradelab_equity_usd"] == 9800.0
    # signal comes from the most recent cycle
    assert m["tradelab_last_signal_ladder_value"] == 0.5
    assert m["tradelab_sma_gate_open"] == 1
    # duration percentiles are emitted with a quantile label
    assert 'tradelab_cycle_duration_ms{quantile="0.5"}' in text
    assert "# TYPE tradelab_cycles_total counter" in text


def test_outcome_counter_labels(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=5), outcome="success", live=True),
        _entry(ended_at=NOW - timedelta(hours=4), outcome="failed", live=True),
        _entry(ended_at=NOW - timedelta(hours=3), outcome="success", live=False),
    ])
    text = metrics.render_metrics(reader, NOW)
    assert 'tradelab_cycles_total{outcome="success"} 2' in text
    assert 'tradelab_cycles_total{outcome="failed"} 1' in text


def test_empty_journal_is_total(tmp_path):
    reader = JournalReader(tmp_path / "missing.jsonl")
    text = metrics.render_metrics(reader, NOW)  # must not raise
    m = _parse(text)
    assert m["tradelab_up"] == 1
    assert m["tradelab_journal_valid_cycles"] == 0
    assert m["tradelab_journal_read_error"] == 0
    # no cycles -> no freshness/equity lines
    assert "tradelab_last_cycle_age_seconds" not in m


def test_unreadable_journal_sets_read_error(tmp_path):
    (tmp_path / "cycles.jsonl").mkdir()  # a directory at the path
    reader = JournalReader(tmp_path / "cycles.jsonl")
    text = metrics.render_metrics(reader, NOW)  # must not raise
    assert _parse(text)["tradelab_journal_read_error"] == 1


def test_no_nan_or_inf_in_output(tmp_path):
    reader = _journal(tmp_path, [
        _entry(ended_at=NOW - timedelta(hours=1), outcome="success", live=True,
               equity=float("nan"), ladder=float("inf")),
    ])
    text = metrics.render_metrics(reader, NOW)
    low = text.lower()
    assert "nan" not in low and "inf" not in low


def test_metrics_endpoint_end_to_end(tmp_path):
    p = tmp_path / "cycles.jsonl"
    p.write_text(json.dumps(_entry(
        ended_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        outcome="success", live=True)) + "\n")
    cfg = hs.Config(journal_path=str(p), host="127.0.0.1", port=0)
    httpd = hs.build_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert resp.getheader("Content-Type", "").startswith("text/plain")
        assert "tradelab_up 1" in body
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
