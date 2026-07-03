#!/usr/bin/env python3
"""Read-only HTTP health endpoints for Netdata ``httpcheck``.

Why this exists
===============
``paper-place-orders`` is a *batch* job, not an always-on server, so the
classic liveness probe ("is the web process up?") answers the wrong
question. What matters is **freshness + outcome**: did the expected cycle
run recently, and did it succeed? This server encodes that as HTTP status
so the existing Netdata ``httpcheck`` collector + ``botcrit`` notification
path can consume it unchanged — a stale/failed cycle returns 503, exactly
like a refused/timed-out endpoint.

The system has two cadences (see ``execution/README.md``): an **hourly
dry-run** keeps the journal warm, and the **daily live** run places real
orders. One probe cannot separate "the whole bot died" from "today's real
order cron didn't fire", so there are two endpoints:

* ``GET /healthz``        heartbeat  — any cycle within ~2h (hourly dry-run).
                                       Catches a dead cron/process.
* ``GET /healthz/daily``  daily_live — last *live* cycle within ~26h AND a
                                       healthy ``outcome``. Catches "today's
                                       real order placement missing or failed".
* ``GET /``               human summary of both (always 200; not an alarm
                                       target).

Hard-rules compliance
=====================
This is a **read-only consumer of the journal**, like the Streamlit
dashboard: it imports only :mod:`trade_lab.monitoring.data_source` (which
never touches ``trade_lab.execution``), holds no credentials, and never
reaches the exchange. It builds a fresh :class:`JournalReader` per request,
so there is no shared mutable state across threads and every response
reflects the journal on disk right now.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from trade_lab.monitoring.data_source import (
    JournalReader,
    is_live_cycle,
    open_order_incidents,
    parse_iso,
)

logger = logging.getLogger("trade_lab.health")

DEFAULT_JOURNAL_PATH = "data/journal/cycles.jsonl"
# The hourly dry-run keeps the journal warm; 2h covers one missed dry-run
# plus grace before we call the heartbeat dead.
DEFAULT_HEARTBEAT_MAX_AGE_S = 7200
# Live orders are placed once a day; 26h = one day + a 2h grace window.
DEFAULT_DAILY_MAX_AGE_S = 93600

# The only healthy outcome for a *main* live cycle (real order placement).
# 'reconstructed' is deliberately NOT here: a reconstruction cycle proves a
# prior cycle's open orders were reconciled, not that today's placement ran,
# so it is excluded from the daily clock entirely (see evaluate_daily).
HEALTHY_MAIN_LIVE_OUTCOMES = frozenset({"success"})

# How many trailing cycles to scan for the daily check. The hourly dry-run +
# daily live share the journal, so ~200 cycles covers well over a week — more
# than enough to find the last live run and anything that failed after it.
DAILY_WINDOW_CYCLES = 200


@dataclass
class HealthResult:
    """Outcome of one health check: OK flag, human reason, debug detail."""

    ok: bool
    reason: str
    detail: dict = field(default_factory=dict)

    @property
    def status_code(self) -> int:
        return 200 if self.ok else 503


def _age_seconds(cycle: dict, now: datetime) -> Optional[float]:
    """Seconds between ``cycle.ended_at`` and ``now``, or None if unparseable."""
    ended = parse_iso(cycle.get("ended_at"))
    if ended is None:
        return None
    return (now - ended).total_seconds()


def _cycle_mode(cycle: dict) -> Optional[str]:
    """Return the durable ``context.mode`` ('live'/'dry_run'), or None."""
    ctx = cycle.get("context")
    return ctx.get("mode") if isinstance(ctx, dict) else None


def _is_live_attempt(cycle: dict) -> bool:
    """True if this cycle was a live order run — even one that failed before
    placing anything.

    Prefers the durable ``context.mode`` marker (present on every cycle since
    the mode-marker change, including fail-before-placement failures). Falls
    back to ``is_live_cycle`` (orders_executed) for pre-marker journal entries,
    which cannot see a live cycle that died before placing an order — the exact
    gap the marker closes going forward.
    """
    mode = _cycle_mode(cycle)
    if mode is not None:
        return mode == "live"
    return is_live_cycle(cycle)


def evaluate_heartbeat(
    reader: JournalReader, now: datetime, max_age_s: float,
) -> HealthResult:
    """Is the bot writing to the journal at all (hourly dry-run cadence)?"""
    last = reader.latest_cycle()
    read_error = reader.stats().read_error
    if read_error:
        return HealthResult(False, f"journal unreadable: {read_error}",
                            {"read_error": read_error})
    if last is None:
        return HealthResult(False, "no cycles in journal", {})
    age = _age_seconds(last, now)
    if age is None:
        return HealthResult(False, "latest cycle has no parseable ended_at",
                            {"cycle_id": last.get("cycle_id")})
    detail = {
        "cycle_id": last.get("cycle_id"),
        "ended_at": last.get("ended_at"),
        "age_seconds": round(age, 1),
        "max_age_seconds": max_age_s,
        "outcome": last.get("outcome"),
    }
    if age > max_age_s:
        return HealthResult(
            False, f"no cycle in {age:.0f}s (limit {max_age_s:.0f}s)", detail,
        )
    return HealthResult(True, "ok", detail)


def evaluate_daily(
    reader: JournalReader, now: datetime, max_age_s: float,
) -> HealthResult:
    """Did the daily LIVE order cycle run recently and succeed?

    Scans a trailing window rather than trusting the single latest entry, so
    the signal is durable against the hourly dry-run overwriting it. See
    ``ops/README.md`` for the two false-negative paths this closes.
    """
    cycles = reader.cycles(DAILY_WINDOW_CYCLES)  # also forces the refresh
    read_error = reader.stats().read_error
    if read_error:
        return HealthResult(False, f"journal unreadable: {read_error}",
                            {"read_error": read_error})
    # "Main" live cycles are real order runs, identified by the durable
    # context.mode marker so a live run that FAILED before placing an order is
    # still counted (and its failure caught below). Reconstruction cycles
    # (outcome=="reconstructed") are excluded: they only prove a PRIOR cycle's
    # open orders were reconciled, not that today's placement ran. Dry-run
    # cycles (mode=='dry_run') are excluded entirely, so a benign hourly
    # dry-run failure never pages this endpoint.
    main_live = [
        c for c in cycles
        if _is_live_attempt(c) and c.get("outcome") != "reconstructed"
    ]
    if not main_live:
        return HealthResult(False, "no live order cycle in journal window", {})
    last = main_live[-1]
    last_ended = parse_iso(last.get("ended_at"))
    if last_ended is None:
        return HealthResult(False, "latest live cycle has no parseable ended_at",
                            {"cycle_id": last.get("cycle_id")})
    age = (now - last_ended).total_seconds()
    outcome = str(last.get("outcome") or "?")
    open_orders = open_order_incidents(cycles)  # surfaced for humans, not a gate
    detail = {
        "cycle_id": last.get("cycle_id"),
        "ended_at": last.get("ended_at"),
        "age_seconds": round(age, 1),
        "max_age_seconds": max_age_s,
        "outcome": outcome,
        "open_order_incidents": len(open_orders),
    }
    if age > max_age_s:
        return HealthResult(
            False, f"last live cycle {age:.0f}s ago (limit {max_age_s:.0f}s)",
            detail,
        )
    if outcome not in HEALTHY_MAIN_LIVE_OUTCOMES:
        # Covers a live run that placed orders then failed AND one that died
        # before placing (both carry mode=='live' and a non-success outcome).
        return HealthResult(False, f"last live outcome={outcome}", detail)
    return HealthResult(True, "ok", detail)


@dataclass
class Config:
    """Runtime configuration, all overridable by environment."""

    journal_path: str = DEFAULT_JOURNAL_PATH
    host: str = "127.0.0.1"
    port: int = 7001
    heartbeat_max_age_s: float = DEFAULT_HEARTBEAT_MAX_AGE_S
    daily_max_age_s: float = DEFAULT_DAILY_MAX_AGE_S

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            # Reuse the dashboard's journal-path env so both read one place.
            journal_path=os.environ.get(
                "TRADE_LAB_MONITORING_JOURNAL_PATH", DEFAULT_JOURNAL_PATH,
            ),
            host=os.environ.get("TRADE_LAB_HEALTH_HOST", "127.0.0.1"),
            port=int(os.environ.get("TRADE_LAB_HEALTH_PORT", "7001")),
            heartbeat_max_age_s=float(os.environ.get(
                "TRADE_LAB_HEALTH_HEARTBEAT_MAX_AGE_S",
                str(DEFAULT_HEARTBEAT_MAX_AGE_S),
            )),
            daily_max_age_s=float(os.environ.get(
                "TRADE_LAB_HEALTH_DAILY_MAX_AGE_S", str(DEFAULT_DAILY_MAX_AGE_S),
            )),
        )


class _HealthHandler(BaseHTTPRequestHandler):
    """Routes GET requests to the evaluators. Never raises out of a request."""

    server_version = "trade-lab-health/1"
    protocol_version = "HTTP/1.1"
    # Reclaim a stalled connection rather than pin a handler thread forever;
    # the only real client is on-host Netdata with its own 2s client timeout.
    timeout = 10

    def do_GET(self) -> None:  # noqa: N802 (stdlib-mandated name)
        try:
            self._route()
        except Exception as exc:  # a bad request must not kill the server
            logger.exception("health handler error")
            self._respond(500, {"ok": False, "reason": f"internal error: {exc}"})

    def _route(self) -> None:
        cfg: Config = self.server.cfg  # type: ignore[attr-defined]
        reader = JournalReader(cfg.journal_path)
        now = datetime.now(timezone.utc)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/healthz":
            r = evaluate_heartbeat(reader, now, cfg.heartbeat_max_age_s)
            self._respond(r.status_code,
                          {"ok": r.ok, "check": "heartbeat",
                           "reason": r.reason, **r.detail})
        elif path == "/healthz/daily":
            r = evaluate_daily(reader, now, cfg.daily_max_age_s)
            self._respond(r.status_code,
                          {"ok": r.ok, "check": "daily_live",
                           "reason": r.reason, **r.detail})
        elif path == "/":
            hb = evaluate_heartbeat(reader, now, cfg.heartbeat_max_age_s)
            dl = evaluate_daily(reader, now, cfg.daily_max_age_s)
            self._respond(200, {
                "service": "trade-lab-health",
                "journal_path": cfg.journal_path,
                "heartbeat": {"ok": hb.ok, "reason": hb.reason, **hb.detail},
                "daily_live": {"ok": dl.ok, "reason": dl.reason, **dl.detail},
            })
        else:
            self._respond(404, {"ok": False, "reason": f"unknown path {path}"})

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # route to logging
        logger.info("health %s %s", self.address_string(), fmt % args)


def _is_loopback(host: str) -> bool:
    """True if ``host`` is a loopback address (127.0.0.0/8, ::1, localhost)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_server(cfg: Config) -> ThreadingHTTPServer:
    """Construct (but do not start) the health HTTP server."""
    if not _is_loopback(cfg.host):
        # Fail loud (not silent): binding a routable address exposes journal
        # metadata (cycle_id / outcome / filesystem path) to the network. The
        # hard rule is 127.0.0.1; warn rather than hard-fail so an intentional
        # reverse-proxy setup is still possible, but never do it quietly.
        logger.warning(
            "health server binding NON-loopback host %r — this exposes journal "
            "metadata to the network; the hard rule is 127.0.0.1", cfg.host,
        )
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), _HealthHandler)
    httpd.cfg = cfg  # type: ignore[attr-defined]
    return httpd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = Config.from_env()
    httpd = build_server(cfg)
    logger.info(
        "trade-lab-health listening on http://%s:%d (journal=%s)",
        cfg.host, cfg.port, cfg.journal_path,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
