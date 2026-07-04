#!/usr/bin/env python3
"""Prometheus-format metrics over the cycle journal, for Netdata's go.d
``prometheus`` scraper (or a real Prometheus).

Same contract as the health server: a **read-only consumer of the journal**
(no execution import, no credentials, no exchange access). Served by the
health server at ``GET /metrics``; ``render_metrics`` is a pure function of a
:class:`JournalReader` + ``now`` so it is unit-testable with a frozen clock.

Totality: every value comes through the ``data_source`` helpers (which never
raise on bad input) and the numeric formatter guards NaN/Inf, so a corrupt
journal degrades individual metrics rather than failing the scrape.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

from trade_lab.monitoring.data_source import (
    duration_stats,
    equity_series,
    open_order_incidents,
    parse_iso,
)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# cycles(_FULL) returns the whole cached journal (slice with a huge n). Used
# for monotonic all-time counters. _RECENT bounds the "current" latency window.
_FULL = 10 ** 9
_RECENT = 200


def _fmt(v) -> str:
    """Format a number as a Prometheus sample value (guards bool/NaN/Inf)."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):  # NaN / +-Inf
        return "0"
    s = f"{f:.6f}".rstrip("0").rstrip(".")
    return s or "0"


def _esc(s) -> str:
    """Escape a Prometheus label value (backslash, quote, newline)."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_metrics(reader, now: datetime) -> str:
    """Render the journal's current state as Prometheus text exposition."""
    out: list[str] = []

    def fam(name: str, typ: str, help_text: str, samples: list) -> None:
        # samples: list of (labels_dict_or_None, value). Skip empty families.
        if not samples:
            return
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {typ}")
        for labels, val in samples:
            if labels:
                lbl = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
                out.append(f"{name}{{{lbl}}} {_fmt(val)}")
            else:
                out.append(f"{name} {_fmt(val)}")

    cycles = reader.cycles(_FULL)          # whole cached journal
    stats = reader.stats()

    # --- exporter / journal self-instrumentation -------------------------
    fam("tradelab_up", "gauge",
        "1 when the metrics exporter served this scrape", [(None, 1)])
    fam("tradelab_journal_read_error", "gauge",
        "1 if the journal could not be read this scrape",
        [(None, 1 if stats.read_error else 0)])
    fam("tradelab_journal_valid_cycles", "gauge",
        "valid cycle rows currently parsed from the journal",
        [(None, stats.valid_cycles)])
    fam("tradelab_journal_corrupt_lines", "gauge",
        "journal lines that failed to parse", [(None, stats.corrupt_lines)])
    fam("tradelab_journal_unknown_version_lines", "gauge",
        "journal rows with an unknown schema_version",
        [(None, stats.unknown_version_lines)])

    # --- freshness (the dead-man's-switch signals, as time series) -------
    last = cycles[-1] if cycles else None
    if last is not None:
        t = parse_iso(last.get("ended_at"))
        if t is not None:
            fam("tradelab_last_cycle_timestamp_seconds", "gauge",
                "unix timestamp of the most recent cycle",
                [(None, t.timestamp())])
            fam("tradelab_last_cycle_age_seconds", "gauge",
                "seconds since the most recent cycle",
                [(None, (now - t).total_seconds())])
    live = reader.latest_live_cycle()
    if live is not None:
        tl = parse_iso(live.get("ended_at"))
        if tl is not None:
            fam("tradelab_last_live_cycle_timestamp_seconds", "gauge",
                "unix timestamp of the most recent live order cycle",
                [(None, tl.timestamp())])
            fam("tradelab_last_live_cycle_age_seconds", "gauge",
                "seconds since the most recent live order cycle",
                [(None, (now - tl).total_seconds())])

    # --- outcomes (all-time counters) ------------------------------------
    oc = Counter(str(c.get("outcome") or "unknown") for c in cycles)
    fam("tradelab_cycles_total", "counter",
        "cycles by outcome over the whole journal",
        [({"outcome": k}, v) for k, v in sorted(oc.items())])

    # --- latency / duration (recent window) ------------------------------
    ds = duration_stats(reader.cycles(_RECENT))
    if ds:
        fam("tradelab_cycle_duration_ms", "gauge",
            f"recent cycle duration percentiles (last {_RECENT} cycles)",
            [({"quantile": "0.5"}, ds["p50"]),
             ({"quantile": "0.95"}, ds["p95"])])
        fam("tradelab_cycle_duration_ms_max", "gauge",
            f"max cycle duration over the last {_RECENT} cycles",
            [(None, ds["max"])])

    # --- open incidents / drift / business -------------------------------
    fam("tradelab_open_order_incidents", "gauge",
        "executed orders not in a resolved terminal state (recent window)",
        [(None, len(open_order_incidents(reader.cycles(_RECENT))))])
    fam("tradelab_cumulative_skipped_drift_usd", "gauge",
        "cumulative quote drift skipped across all cycles",
        [(None, reader.cumulative_skipped_drift())])

    eq = equity_series(cycles)
    if eq:
        fam("tradelab_equity_usd", "gauge",
            "latest paper equity (USD) from a successful cycle",
            [(None, eq[-1][1])])

    if last is not None and isinstance(last.get("signal"), dict):
        sig = last["signal"]
        lv = sig.get("ladder_value")
        if lv is not None:
            fam("tradelab_last_signal_ladder_value", "gauge",
                "latest pro-rata ladder signal (0, 0.5, 1.0)", [(None, lv)])
        fam("tradelab_sma_gate_open", "gauge",
            "1 if the SMA regime gate was open on the last cycle",
            [(None, 1 if sig.get("sma_gate_open") else 0)])

    return "\n".join(out) + "\n"
