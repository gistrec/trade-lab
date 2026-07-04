"""Streamlit monitoring dashboard for the paper-trading bot.

Read-only by construction. Reads ``JournalReader`` and displays
status, signal, portfolio drift, and recent cycles. There are no
controls — no start/stop buttons, no rebalance triggers, no exchange
calls. Anything that needs to act on the bot must be a separate CLI
on the VPS, not this UI.

Configuration via environment variables:

* ``TRADE_LAB_MONITORING_JOURNAL_PATH`` — path to the journal file
  the bot writes to. Mounted read-only into this process via Unix
  permissions (group-readable to the ``monitoring`` user only).
* ``MONITORING_EXPECTED_CYCLE_INTERVAL_SECONDS`` — used to bucket
  staleness. 3600 (one hour) for daily candles is a generous floor;
  a true daily run misses ≥1 day if STALE triggers.
* ``MONITORING_REFRESH_SECONDS`` — HTML meta-refresh interval. 30s
  default; the underlying data only updates once per bot cycle, so
  smaller values just waste CPU.

Auto-refresh uses a native ``st.fragment(run_every=...)`` around the
dashboard body: Streamlit reruns just that fragment every
``MONITORING_REFRESH_SECONDS`` to pull fresh journal data, with no browser
reload (the active tab and slider state survive). This replaced the earlier
``streamlit-autorefresh`` component, whose iframe flashed a skeleton
placeholder on the page each tick. An HTML ``<meta>`` refresh tag was never
an option — a full page reload throws away the active tab and session state.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_lab.monitoring.data_source import (
    DOWN_MULTIPLIER, JournalReader, ReadStats, STALE_MULTIPLIER, Staleness,
    as_float, cycle_orders_executed, drift_series, duration_series,
    duration_stats, equity_series, is_live_cycle, max_inter_cycle_gap_seconds,
    open_order_incidents, parse_iso, recent_incidents,
)
from trade_lab.uikit import render_tab_safely
from trade_lab.monitoring import research


JOURNAL_PATH = os.environ.get(
    "TRADE_LAB_MONITORING_JOURNAL_PATH",
    "data/journal/cycles.jsonl",
)
EXPECTED_INTERVAL_S = int(
    os.environ.get("MONITORING_EXPECTED_CYCLE_INTERVAL_SECONDS", "3600")
)
REFRESH_SECONDS = int(os.environ.get("MONITORING_REFRESH_SECONDS", "30"))
# The daily `paper-place-orders` cron is the LIVE order path; the hourly
# dry-run shares the same journal. This is the expected spacing of LIVE
# cycles (one calendar day) used to flag a silently-dead order cron that
# the overall (dry-run-dominated) staleness cannot see.
EXPECTED_LIVE_INTERVAL_S = int(
    os.environ.get("MONITORING_EXPECTED_LIVE_INTERVAL_SECONDS", "86400")
)
# Base URL for linking a journalled git_commit to its source on the host.
# Override for a fork / self-hosted git; empty string disables linking (commits
# render as plain code spans).
REPO_URL = os.environ.get(
    "TRADE_LAB_MONITORING_REPO_URL", "https://github.com/gistrec/trade-lab"
).rstrip("/")
# Footer author + contact links. All env-configurable; a contact link renders
# only when its URL is non-empty (LinkedIn is off until a URL is provided).
AUTHOR_NAME = os.environ.get("TRADE_LAB_MONITORING_AUTHOR", "Aleksandr Kovalko")
TELEGRAM_URL = os.environ.get(
    "TRADE_LAB_MONITORING_TELEGRAM_URL", "https://t.me/gistrec"
).rstrip("/")
LINKEDIN_URL = os.environ.get(
    "TRADE_LAB_MONITORING_LINKEDIN_URL", "https://www.linkedin.com/in/gistrec"
).rstrip("/")

# Validation forward-test paths (see paper_trading/README.md). All
# read-only; the validation panel never writes.
VALIDATION_LOG_PATH = Path(
    os.environ.get(
        "TRADE_LAB_VALIDATION_LOG_PATH",
        "paper_trading/logs/journal.jsonl",
    )
)
VALIDATION_VINTAGE_ROOT = Path(
    os.environ.get(
        "TRADE_LAB_VALIDATION_VINTAGE_ROOT",
        "paper_trading/vintages",
    )
)
VALIDATION_REFERENCE_PATH = Path(
    os.environ.get(
        "TRADE_LAB_VALIDATION_REFERENCE_PATH",
        "paper_trading/fingerprint/reference_fingerprint.json",
    )
)


# Single reader instance reused across reruns. JournalReader is
# cache-aware (mtime-based), so this is safe and cheap.
@st.cache_resource
def _get_reader() -> JournalReader:
    return JournalReader(JOURNAL_PATH)


def _cycle_context(cycle: Optional[dict]) -> dict:
    """Return a cycle's ``context`` dict, or ``{}`` for missing/non-dict.

    Journal rows are external input; a truthy non-dict context (schema
    drift, corruption) makes ``.get(...)`` raise AttributeError. Both the
    safety banner and the Portfolio tab read context — route both through
    here so a malformed context degrades instead of crashing.
    """
    ctx = (cycle or {}).get("context")
    return ctx if isinstance(ctx, dict) else {}


def _cycle_mode(cycle: Optional[dict]) -> str:
    """'LIVE' if the cycle placed real orders, else 'DRY'.

    The journal is dominated ~24:1 by hourly dry-runs, so the operator needs
    to know at a glance whether what they are looking at is the real daily
    rebalance or a planning-only heartbeat.
    """
    return "LIVE" if is_live_cycle(cycle or {}) else "DRY"


# ---------------------------------------------------------------------------
# Top banner: testnet vs mainnet
# ---------------------------------------------------------------------------


def _render_top_banner(latest: Optional[dict]) -> None:
    """Render the exchange/sandbox banner. Mainnet is RED and large by
    design — accidental mainnet config must hit the operator visually."""
    if latest is None:
        st.markdown(
            "<div style='background:#37474f;color:white;padding:0.8rem;"
            "border-radius:0.5rem;text-align:center;font-size:1.2rem;'>"
            "NO JOURNAL DATA — bot has not started</div>",
            unsafe_allow_html=True,
        )
        return
    ctx = _cycle_context(latest)
    # Safety banner fails loud: only an explicit True is "safe". A
    # missing or non-bool flag (schema drift, truncated context) must
    # NOT render the reassuring green testnet banner.
    sandbox = ctx.get("sandbox")
    exchange = str(ctx.get("exchange") or "unknown").upper()
    if sandbox is True:
        st.markdown(
            f"<div style='background:#1b5e20;color:white;padding:0.8rem;"
            f"border-radius:0.5rem;text-align:center;font-size:1.4rem;'>"
            f"TESTNET — {exchange}</div>",
            unsafe_allow_html=True,
        )
    elif sandbox is not False:
        st.markdown(
            f"<div style='background:#bf360c;color:white;padding:1.2rem;"
            f"border-radius:0.5rem;text-align:center;font-size:1.6rem;"
            f"font-weight:bold;'>"
            f"SANDBOX FLAG UNKNOWN — {exchange} — verify config before "
            f"trusting this dashboard</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#b71c1c;color:white;padding:1.2rem;"
            f"border-radius:0.5rem;text-align:center;font-size:2rem;"
            f"font-weight:bold;letter-spacing:0.1rem;'>"
            f"MAINNET — {exchange} — REAL MONEY</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# One-line health verdict (always visible, under the banner)
# ---------------------------------------------------------------------------


def _health_verdict(reader: JournalReader) -> tuple[str, str]:
    """Collapse the health signals into (level, why).

    'is the bot healthy right now?' otherwise has no single answer above the
    tabs — the banner only encodes testnet/mainnet, and the real signals live
    inside the Status tab. Level is HEALTHY / DEGRADED / DOWN.
    """
    stats = reader.stats()
    if stats.read_error is not None:
        return ("DOWN", f"journal unreadable ({stats.read_error})")
    staleness = reader.staleness(EXPECTED_INTERVAL_S)
    if staleness is Staleness.NO_DATA:
        return ("DOWN", "no valid cycles in the journal")

    cycles = reader.cycles(n=500)
    open_orders = open_order_incidents(cycles)
    incidents = recent_incidents(cycles)
    live = reader.latest_live_cycle()
    live_overdue = False
    if live is not None:
        dt = parse_iso(live.get("ended_at"))
        if dt is not None:
            age = (datetime.now(tz=timezone.utc) - dt).total_seconds()
            live_overdue = age > EXPECTED_LIVE_INTERVAL_S * STALE_MULTIPLIER

    if staleness is Staleness.DOWN or open_orders or live_overdue:
        why = []
        if staleness is Staleness.DOWN:
            why.append("heartbeat DOWN")
        if live_overdue:
            why.append("live order cron overdue")
        if open_orders:
            why.append(f"{len(open_orders)} unresolved order(s)")
        return ("DOWN", "; ".join(why))

    if staleness is Staleness.STALE or incidents:
        why = []
        if staleness is Staleness.STALE:
            why.append("heartbeat stale")
        if incidents:
            why.append(f"{len(incidents)} incident cycle(s) in window")
        return ("DEGRADED", "; ".join(why))

    return ("HEALTHY", "heartbeat fresh · last live cycle OK · no open incidents")


def _render_health_line(reader: JournalReader) -> None:
    level, why = _health_verdict(reader)
    color = {"HEALTHY": "#1b5e20", "DEGRADED": "#bf360c", "DOWN": "#b71c1c"}[level]
    # A clear margin-top gap separates this from the sandbox/mainnet banner
    # above — without it two same-green plaques (TESTNET + HEALTHY are both
    # #1b5e20) read as one merged block. Neither is sticky: a sticky banner
    # with a non-sticky sibling makes the sibling slide UNDER it on scroll.
    st.markdown(
        f"<div style='background:{color};color:white;padding:0.5rem;"
        f"margin-top:0.5rem;border-radius:0.4rem;text-align:center;"
        f"font-size:1.05rem;'>"
        f"BOT {level} — {why}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Status tab
# ---------------------------------------------------------------------------


def _render_status(reader: JournalReader) -> None:
    latest = reader.latest_cycle()
    stats = reader.stats()
    staleness = reader.staleness(EXPECTED_INTERVAL_S)

    cols = st.columns(4)
    cols[0].metric("Staleness", staleness.value.upper())
    last_ended_iso = latest.get("ended_at") if latest is not None else None
    if latest is not None:
        # Use a *relative* time as the metric value (compact for narrow
        # screens — "5m ago" fits in a column whereas the absolute UTC
        # string truncates) and surface the precise timestamp below
        # the row in a caption.
        cols[1].metric("Last cycle", _humanize_relative(last_ended_iso))
        _dur = latest.get("duration_ms")
        cols[2].metric(
            "Last duration",
            f"{_dur} ms" if isinstance(_dur, (int, float)) else "—",
        )
        cols[3].metric("Outcome", str(latest.get("outcome") or "?").upper())
    else:
        for c in cols[1:]:
            c.metric("—", "—")

    if latest is not None:
        mode = _cycle_mode(latest)
        if mode == "DRY":
            st.caption(
                "Latest journal cycle is a **DRY-RUN** (planning only) — the "
                "hourly heartbeat. Real orders run once daily; see 'Live "
                "order cron' below for the last REAL cycle."
            )
        else:
            st.caption("Latest journal cycle is a **LIVE** (real-order) cycle.")

    if last_ended_iso:
        st.caption(f"Last cycle ended at {_humanize_iso(last_ended_iso)}")

    if staleness is Staleness.DOWN:
        st.error(
            f"Bot appears DOWN: last cycle was over "
            f"{int(EXPECTED_INTERVAL_S * DOWN_MULTIPLIER)}s ago "
            f"(threshold = {DOWN_MULTIPLIER:g}× expected interval of "
            f"{EXPECTED_INTERVAL_S}s)."
        )
    elif staleness is Staleness.STALE:
        st.warning(
            f"Bot is STALE: last cycle elapsed > "
            f"{int(EXPECTED_INTERVAL_S * STALE_MULTIPLIER)}s "
            f"(threshold = {STALE_MULTIPLIER:g}× expected interval of "
            f"{EXPECTED_INTERVAL_S}s)."
        )
    elif staleness is Staleness.NO_DATA:
        st.info(
            "No valid cycles yet. If the bot has been started, "
            f"check that it can write to: `{JOURNAL_PATH}`"
        )

    if latest is not None and latest.get("outcome") == "failed":
        err = latest.get("error") or {}
        st.error(
            f"Most recent cycle FAILED. {err.get('type', '?')}: "
            f"{err.get('message', 'no message')}"
        )

    if latest is not None and latest.get("outcome") == "unknown_orders":
        st.error(
            "Last cycle has orders in UNKNOWN state (timeout or "
            "lost_track). Next cycle will attempt reconstruction. "
            "Manual review recommended — see Cycles tab → cycle detail."
        )

    if latest is not None and latest.get("outcome") == "reconstructed":
        st.info(
            "Latest entry is a reconstruction cycle — orders from a prior "
            "cycle were resolved. The actual rebalance for today is in the "
            "next cycle entry (if it has run yet)."
        )

    _render_live_cron_health(reader)
    _render_incidents(reader)

    commits = _distinct_commits(reader.cycles(n=500))
    if len(commits) > 1:
        links = ", ".join(_commit_link(c) for c in commits)
        st.warning(
            f"Observation window spans {len(commits)} git commits "
            f"({links}). A redeploy mid-window means the signal-stability "
            f"sample mixes code versions — interpret trends across the "
            f"boundary with care."
        )

    drift = reader.cumulative_skipped_drift()
    if drift > 0:
        st.warning(
            f"Cumulative skipped-order drift: ${drift:,.2f} across "
            f"{stats.valid_cycles} cycles. Sub-min divergence is normal "
            f"on tiny balances; investigate if it grows steadily."
        )

    _render_read_stats(stats)


def _render_live_cron_health(reader: JournalReader) -> None:
    """Freshness clock for the LIVE order cron specifically.

    The overall Staleness metric buckets on the last cycle of *any* type, so
    the hourly dry-run keeps it FRESH even if the daily `paper-place-orders`
    cron has been dead for days. This surfaces the last REAL-order cycle on
    its own ~daily clock and fires loud when overdue.
    """
    st.subheader("Live order cron")
    live = reader.latest_live_cycle()
    if live is None:
        st.info(
            "No LIVE (real-order) cycle in the journal yet — only dry-runs. "
            "The daily `paper-place-orders` cron writes the first LIVE cycle "
            "when it next runs."
        )
        return

    ended = live.get("ended_at")
    cols = st.columns(3)
    cols[0].metric("Last LIVE cycle", _humanize_relative(ended))
    cols[1].metric("LIVE outcome", str(live.get("outcome") or "?").upper())
    cols[2].metric("LIVE cycle", (live.get("cycle_id") or "?")[:8])
    st.caption(
        f"Last LIVE cycle ended at {_humanize_iso(ended)}. The Staleness "
        f"metric above tracks the hourly dry-run heartbeat, not this."
    )

    dt = parse_iso(ended)
    if dt is not None:
        age = (datetime.now(tz=timezone.utc) - dt).total_seconds()
        if age > EXPECTED_LIVE_INTERVAL_S * STALE_MULTIPLIER:
            st.error(
                f"LIVE order cron OVERDUE — last real-order cycle was "
                f"{_humanize_relative(ended)} "
                f"(threshold = {STALE_MULTIPLIER:g}× expected "
                f"{EXPECTED_LIVE_INTERVAL_S}s). "
                f"The hourly dry-run may be masking a dead daily cron; "
                f"check the `paper-place-orders` cron on the VPS."
            )


def _render_incidents(reader: JournalReader) -> None:
    """Window-level incident view: non-success cycles, unresolved orders,
    and cadence gaps that the latest-cycle-only alerts above cannot show."""
    cycles = reader.cycles(n=500)
    incidents = recent_incidents(cycles)
    open_orders = open_order_incidents(cycles)
    gap = max_inter_cycle_gap_seconds(cycles)
    gap_overdue = gap is not None and gap > EXPECTED_INTERVAL_S * STALE_MULTIPLIER

    st.subheader("Incidents (last 500 cycles)")
    if not incidents and not open_orders and not gap_overdue:
        st.success(
            "No failed/partial cycles, unresolved orders, or cadence gaps "
            "in the window."
        )
        return

    if incidents:
        st.warning(
            f"{len(incidents)} non-success cycle(s) in the window "
            f"(failed / unknown_orders / partial)."
        )
        rows = [{
            "ended_at": _humanize_iso(i["ended_at"]),
            "mode": i["mode"],
            "outcome": i["outcome"].upper(),
            "cycle": i["cycle_id"],
            "error": i["error_type"] or "",
            "message": (i["error_message"] or "")[:80],
        } for i in incidents]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if open_orders:
        st.error(
            f"{len(open_orders)} executed order(s) NOT in a resolved terminal "
            f"state (partial / rejected / lost_track / timeout). A lost_track "
            f"keeps the CLI exit code non-zero until resolved."
        )
        rows = [{
            "ended_at": _humanize_iso(o["ended_at"]),
            "cycle": o["cycle_id"],
            "side": o["side"],
            "symbol": o["symbol"],
            "status": o["status"],
            "client_order_id": o["client_order_id"],
        } for o in open_orders]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if gap_overdue:
        st.warning(
            f"Largest gap between consecutive cycles: {gap / 3600:.1f}h "
            f"(> {STALE_MULTIPLIER:g}× expected {EXPECTED_INTERVAL_S}s). A "
            f"cycle may have been missed mid-window — the single-latest "
            f"Staleness check cannot see this."
        )


def _render_read_stats(stats: ReadStats) -> None:
    if stats.read_error is not None:
        st.error(
            f"JOURNAL UNREADABLE — {stats.read_error}. The file exists but "
            f"this process cannot read it. Check the path and permissions "
            f"(`{JOURNAL_PATH}`): the monitoring user needs group-read on the "
            f"journal. Until fixed, the dashboard shows NO data."
        )
    # Corrupt lines are a data-integrity event (truncated write, disk glitch,
    # writer regression) → a visible warning, not a grey caption. Unknown-
    # version lines are benign forward-compat and stay a low-key caption.
    if stats.corrupt_lines > 0:
        st.warning(
            f"Journal scan: {stats.corrupt_lines} CORRUPT line(s) skipped "
            f"(of {stats.total_lines} non-empty). {stats.valid_cycles} valid. "
            f"A steady rise means the writer or disk is producing bad rows."
        )
    if stats.unknown_version_lines > 0:
        st.caption(
            f"Journal scan: {stats.unknown_version_lines} unknown-version "
            f"entrie(s) skipped (of {stats.total_lines} non-empty) — likely a "
            f"newer bot schema than this dashboard understands."
        )


# ---------------------------------------------------------------------------
# Signal tab
# ---------------------------------------------------------------------------


def _render_signal(reader: JournalReader) -> None:
    latest = reader.latest_cycle()
    sig = (latest or {}).get("signal") or {}
    if not sig:
        st.info("No signal data in journal yet.")
        return

    basket_close = sig.get("basket_close")
    sma_value = sig.get("sma_value")
    gate_open = sig.get("sma_gate_open")

    # --- Top row: 4 metrics ---
    cols = st.columns(4)
    # as_float, not .get(default): a present JSON-null ladder_value makes
    # .get return None and f"{None:.2f}" raise — the data layer is hardened
    # against this (signal_history), the top metric was the bypass.
    ladder_delta = _ladder_prev_day_delta(reader)
    cols[0].metric(
        "Ladder value",
        f"{as_float(sig.get('ladder_value')):.2f}",
        delta=(f"{ladder_delta:+.2f} vs prior day"
               if ladder_delta is not None else None),
        delta_color="normal" if ladder_delta else "off",
    )
    cols[1].metric(
        "SMA(200) gate",
        "OPEN" if gate_open else ("CLOSED" if gate_open is False else "—"),
    )
    cols[2].metric(
        "Basket close",
        f"{basket_close:,.2f}" if basket_close is not None else "—",
    )
    if basket_close is not None and sma_value:
        dist_pct = (basket_close / sma_value - 1.0) * 100
        cols[3].metric(
            "Basket vs SMA(200)",
            f"{dist_pct:+.2f}%",
            delta=f"SMA = {sma_value:.2f}",
            delta_color="off",
        )
    else:
        cols[3].metric("Basket vs SMA(200)", "—")

    # --- Second row: direction + persistence metrics ---
    bcs_dict = latest.get("basket_close_series") or {}
    values = bcs_dict.get("values") or []
    cols2 = st.columns(3)
    cols2[0].metric("vs 7d ago", _series_return(values, 7))
    cols2[1].metric("vs 30d ago", _series_return(values, 30))
    days_since = _days_since_gate_last_open(reader)
    cols2[2].metric(
        "Days since gate OPEN",
        str(days_since) if days_since is not None else "—",
    )

    # --- Basket close chart with current SMA reference ---
    if len(values) >= 2:
        st.plotly_chart(
            _basket_close_figure(values, bcs_dict.get("start_ts"), sma_value),
            width="stretch",
        )
        st.caption(
            "Basket close, last ~100 days. SMA(200) is shown as the "
            "horizontal reference at its CURRENT value — historical SMA "
            "is not stored in the journal."
        )

    # --- Per-lookback breakdown with returns ---
    st.subheader("Per-lookback breakdown (latest cycle)")
    plb_states = sig.get("per_lookback_states") or {}
    plb_returns = sig.get("per_lookback_returns") or {}
    if plb_states:
        def _lb(key):
            try:
                return int(key)
            except (TypeError, ValueError):
                return None

        rows = []
        # Unparseable keys sort last and display verbatim — journal
        # data is external input, one odd key must not blank the tab.
        for k in sorted(plb_states.keys(),
                        key=lambda x: (_lb(x) is None, _lb(x) or 0, str(x))):
            ret = plb_returns.get(k)
            rows.append({
                "lookback": _lb(k) if _lb(k) is not None else str(k),
                "state": plb_states[k],
                "return %": (
                    f"{ret * 100:+.2f}"
                    if isinstance(ret, (int, float)) else "—"
                ),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "State is the pre-gate {0, 1} sign of pct_change(lookback). "
            "Averaged → ladder; SMA(200) gate then zeroes it if closed. "
            "Return shows the magnitude — the distance to a flip."
        )
    else:
        st.info("Per-lookback states not available in the latest cycle.")

    # --- Ladder history chart ---
    history_days = st.select_slider(
        "Ladder history window",
        options=[7, 30, 90, 180, 365],
        value=30,
    )
    history = reader.signal_history(days=history_days)
    if history:
        st.plotly_chart(_signal_history_figure(history), width="stretch")
    else:
        st.info("No signal history in the selected window.")

    # --- Recent cycles table ---
    st.subheader("Recent cycles")
    recent_n = st.select_slider(
        "Cycles to show", options=[7, 14, 30, 60], value=14,
    )
    recent_cycles = reader.cycles(n=recent_n)
    if recent_cycles:
        rows = []
        for c in reversed(recent_cycles):
            csig = c.get("signal") or {}
            cstates = csig.get("per_lookback_states") or {}
            rows.append({
                "asof": _humanize_iso(csig.get("asof") or c.get("ended_at")),
                "basket": csig.get("basket_close"),
                "ladder": csig.get("ladder_value"),
                "gate": "OPEN" if csig.get("sma_gate_open") else "CLOSED",
                "28d": cstates.get("28"),
                "60d": cstates.get("60"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _series_return(values: list, n_days_ago: int) -> str:
    """Format ``values[-1] / values[-(n+1)] - 1`` as a +/- percent.

    A null / non-numeric element (external journal input) yields ``—``
    instead of raising — the series can carry garbage the same way any other
    journal field can.
    """
    if len(values) < n_days_ago + 1:
        return "—"
    try:
        today = float(values[-1])
        past = float(values[-(n_days_ago + 1)])
    except (TypeError, ValueError):
        return "—"
    if past == 0:
        return "—"
    return f"{(today / past - 1.0) * 100:+.2f}%"


def _latest_ladder_by_day(reader: JournalReader) -> dict:
    """Map signal-date → that day's LAST ladder value.

    The journal has ~24 dry-run cycles per day; dedup to one point per date so
    a day-over-day comparison is meaningful (cache is chronological, so the
    later cycle of a date overwrites).
    """
    by_day: dict = {}
    for c in reader.cycles(n=500):
        sig = c.get("signal") or {}
        dt = parse_iso(sig.get("asof"))
        lv = sig.get("ladder_value")
        if dt is None or lv is None:
            continue
        by_day[dt.date()] = as_float(lv)
    return by_day


def _ladder_prev_day_delta(reader: JournalReader) -> Optional[float]:
    """Change in the (per-day) ladder vs the previous distinct signal day.

    The key signal-stability event is 'did deployed exposure flip today?';
    intraday dry-run repeats must not read as a change, so this compares the
    last-of-day values, not consecutive cycles.
    """
    by_day = _latest_ladder_by_day(reader)
    if len(by_day) < 2:
        return None
    days = sorted(by_day)
    return by_day[days[-1]] - by_day[days[-2]]


def _distinct_commits(cycles: list) -> list:
    """Distinct git_commit values across the window, in first-seen order."""
    seen: list = []
    for c in cycles:
        gc = c.get("git_commit")
        if gc and gc not in seen:
            seen.append(gc)
    return seen


def _commit_link(sha: str) -> str:
    """Markdown for one commit: a link to its source when REPO_URL is set,
    else a plain code span. st.warning renders markdown, so links are
    clickable in the dashboard."""
    if REPO_URL:
        return f"[`{sha}`]({REPO_URL}/commit/{sha})"
    return f"`{sha}`"


def _days_since_gate_last_open(reader: JournalReader) -> Optional[int]:
    """Distinct signal *days* since the most recent OPEN gate.

    Counts dates (signal ``asof``), not cycles: with the hourly
    dry-run sharing the journal, one closed day produces ~24 cycles
    and a per-cycle count overstates by that factor. Cycles without a
    signal (failed, reconstruction) say nothing about the gate and are
    skipped. Walks newest-first across up to 500 cycles; returns None
    if no OPEN gate is visible in that window.
    """
    cycles = reader.cycles(n=500)
    closed_dates: set = set()
    for c in reversed(cycles):  # newest-first
        sig = c.get("signal") or {}
        gate = sig.get("sma_gate_open")
        if gate is None:
            continue
        dt = parse_iso(sig.get("asof"))
        date = dt.date() if dt is not None else None
        if gate is True:
            # An intraday flip means the same date sits on both sides;
            # it has seen an OPEN gate, so don't count it as closed.
            closed_dates.discard(date)
            return len(closed_dates)
        if date is not None:
            closed_dates.add(date)
    return None


def _basket_close_figure(
    values: list,
    start_iso: Optional[str],
    sma_value: Optional[float],
) -> go.Figure:
    """Basket close line with horizontal SMA(200) reference."""
    start = parse_iso(start_iso)
    if start is not None:
        x = [start + timedelta(days=i) for i in range(len(values))]
    else:
        x = list(range(len(values)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=values, mode="lines",
        name="Basket close",
        line=dict(color="#1f77b4", width=2),
    ))
    if sma_value is not None:
        fig.add_hline(
            y=sma_value,
            line_dash="dash", line_color="#d62728",
            annotation_text=f"SMA(200) = {sma_value:.2f}",
            annotation_position="bottom right",
        )
    fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=10),
        hovermode="x unified",
        yaxis_title="basket close",
        xaxis_title="date (UTC)" if start is not None else "day index",
    )
    return fig


def _gate_closed_spans(
    history: list[tuple[datetime, float, bool]],
) -> list[tuple[datetime, datetime]]:
    """Contiguous time ranges where the SMA(200) gate was CLOSED.

    Used to shade the ladder chart so a signal zeroed by the regime gate is
    visibly distinct from a genuine 0 ladder — the gate bool is otherwise
    read from the journal and discarded.
    """
    spans: list[tuple[datetime, datetime]] = []
    start: Optional[datetime] = None
    prev_t: Optional[datetime] = None
    for t, _v, gate_open in history:
        if not gate_open and start is None:
            start = t
        elif gate_open and start is not None:
            spans.append((start, prev_t if prev_t is not None else t))
            start = None
        prev_t = t
    if start is not None and prev_t is not None:
        spans.append((start, prev_t))
    return spans


def _signal_history_figure(
    history: list[tuple[datetime, float, bool]],
) -> go.Figure:
    times = [h[0] for h in history]
    values = [h[1] for h in history]
    fig = go.Figure()
    # Shade gate-CLOSED regions first so the ladder line draws on top.
    for x0, x1 in _gate_closed_spans(history):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor="#d62728", opacity=0.08, line_width=0,
        )
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines+markers",
        name="Ladder",
        # The ladder is a discrete {0, 0.5, 1.0} state that HOLDS until it
        # flips — a step line ("hv") tells the truth; straight segments imply
        # continuous transitions that never happened.
        line=dict(color="#1f77b4", width=2, shape="hv"),
    ))
    fig.update_layout(
        height=360, margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(range=[-0.05, 1.05], tickvals=[0.0, 0.5, 1.0],
                   title="ladder value"),
        xaxis_title="date (UTC)",
        hovermode="x unified",
    )
    return fig


def _timeseries_figure(
    points: list[tuple[datetime, float]],
    *,
    y_title: str,
    color: str = "#1f77b4",
    fill: Optional[str] = None,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
) -> go.Figure:
    """Generic (time, value) line chart with UTC x-axis and titled axes."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=color, width=2), fill=fill,
    ))
    if hline is not None:
        fig.add_hline(
            y=hline, line_dash="dot", line_color="gray",
            annotation_text=hline_label or "",
        )
    fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=10),
        hovermode="x unified",
        yaxis_title=y_title, xaxis_title="date (UTC)",
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio tab
# ---------------------------------------------------------------------------


def _unfilled_order_count(cycle: dict) -> Optional[int]:
    """Planned orders that did not fully close, or ``None`` if no
    execution was attempted.

    A dry-run (planning-only) cycle writes ``orders_executed=None`` with a
    populated ``orders_planned``. ``cycle_orders_executed`` collapses that
    ``None`` to ``[]``, which cannot distinguish "no execution attempted"
    from "execution attempted, nothing closed" — so counting off it fires
    a false "planned orders did not fully close" warning on every dry-run
    cycle (the hourly dry-run shares the monitored journal with the daily
    live run). Gate on the raw field: return ``None`` for planning-only
    cycles so the caller suppresses the partial-fill warning.
    """
    if cycle.get("orders_executed") is None:
        return None
    planned_count = len(cycle.get("orders_planned") or [])
    executed = cycle_orders_executed(cycle)
    fully_closed = sum(
        1 for o in executed if o.get("terminal_status") == "closed"
    )
    return planned_count - fully_closed


def _render_portfolio(reader: JournalReader) -> None:
    latest = reader.latest_cycle()
    if latest is None or latest.get("outcome") != "success":
        st.info("No successful cycle yet to compute portfolio drift from.")
        return
    target = latest.get("target_allocation") or {}
    current = latest.get("current_holdings_quote") or {}
    equity = float(latest.get("equity_usd") or 0.0)
    quote = _cycle_context(latest).get("quote_currency") or "USD"

    rows = []
    total_target = 0.0
    total_current = 0.0
    for asset in sorted(set(list(target.keys()) + list(current.keys()))):
        # as_float, not float(.get(default)): a present JSON-null weight makes
        # .get return None and float(None) raise, blanking the tab.
        t = as_float(target.get(asset))
        c = as_float(current.get(asset))
        rows.append({
            "asset": asset,
            f"target {quote}": t,
            f"current {quote}": c,
            "drift": t - c,
            "drift %": (t - c) / equity * 100 if equity > 0 else 0.0,
        })
        total_target += t
        total_current += c

    df = pd.DataFrame(rows)
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            f"target {quote}": st.column_config.NumberColumn(format="%.2f"),
            f"current {quote}": st.column_config.NumberColumn(format="%.2f"),
            "drift": st.column_config.NumberColumn(format="%.2f"),
            "drift %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    cols = st.columns(3)
    cols[0].metric(f"Equity ({quote})", f"{equity:,.2f}")
    cols[1].metric(f"Total target ({quote})", f"{total_target:,.2f}")
    cols[2].metric(
        f"Total drift ({quote})",
        f"{(total_target - total_current):+,.2f}",
    )

    # Planned vs executed divergence — surface unfilled / partial /
    # rejected counts so the operator sees them without drilling into
    # the Cycles tab. Suppressed on dry-run (planning-only) cycles, where
    # orders_executed is None and no execution was attempted.
    planned_count = len(latest.get("orders_planned") or [])
    unfilled = _unfilled_order_count(latest)
    if unfilled is not None and unfilled > 0:
        st.warning(
            f"{unfilled} of {planned_count} planned orders did not fully "
            f"close this cycle — see the Cycles tab → cycle detail for "
            f"per-order status."
        )

    cumulative = reader.cumulative_skipped_drift()
    st.caption(
        f"Cumulative skipped-order drift across all cycles: "
        f"{cumulative:,.2f} {quote}."
    )

    window = reader.cycles(n=500)

    st.subheader("Paper equity over time")
    eq = equity_series(window)
    if len(eq) >= 2:
        st.plotly_chart(
            _timeseries_figure(eq, y_title=f"equity ({quote})", color="#1f77b4"),
            width="stretch",
        )
    else:
        st.info("Not enough successful cycles yet to chart equity.")

    st.subheader("Index-vs-holdings drift over time")
    dr = drift_series(window)
    if len(dr) >= 2:
        st.plotly_chart(
            _timeseries_figure(
                dr, y_title=f"total drift ({quote})",
                color="#9467bd", fill="tozeroy",
            ),
            width="stretch",
        )
        st.caption(
            "Sum of |target − current| per cycle. By design a sawtooth that "
            "resets on the monthly rebalance (the drifted-weight profile from "
            "C3); a steady monotonic climb would flag a problem."
        )
    else:
        st.info("Not enough successful cycles yet to chart drift.")


# ---------------------------------------------------------------------------
# Cycles tab
# ---------------------------------------------------------------------------


def _render_cycles(reader: JournalReader) -> None:
    n = st.slider("Cycles to show", min_value=5, max_value=100, value=20, step=5)
    cycles = reader.cycles(n=n)
    if not cycles:
        st.info("No cycles in journal.")
        return

    summary_rows = []
    for c in reversed(cycles):  # newest first in the table
        sig = c.get("signal") or {}
        summary_rows.append({
            "ended_at": _humanize_iso(c.get("ended_at")),
            "mode": _cycle_mode(c),
            "outcome": str(c.get("outcome") or "?").upper(),
            "duration_ms": c.get("duration_ms"),
            "signal": sig.get("ladder_value"),
            "gate_open": sig.get("sma_gate_open"),
            "planned": len(c.get("orders_planned") or []),
            "executed": len(cycle_orders_executed(c)),
            "skipped": len(c.get("orders_skipped") or []),
            "skipped_drift": c.get("total_skipped_quote_drift") or 0.0,
            "cycle_id": (c.get("cycle_id") or "")[:8],
        })
    st.dataframe(
        pd.DataFrame(summary_rows), width="stretch", hide_index=True,
    )

    stats = duration_stats(cycles)
    durs = duration_series(cycles)
    if stats is not None:
        dcols = st.columns(3)
        dcols[0].metric("Duration p50", f"{stats['p50']:.0f} ms")
        dcols[1].metric("Duration p95", f"{stats['p95']:.0f} ms")
        dcols[2].metric("Duration max", f"{stats['max']:.0f} ms")
    if len(durs) >= 2:
        st.plotly_chart(
            _timeseries_figure(
                durs, y_title="duration (ms)", color="#2ca02c",
                hline=stats["p95"] if stats else None,
                hline_label=f"p95 = {stats['p95']:.0f} ms" if stats else None,
            ),
            width="stretch",
        )
        st.caption(
            "Per-cycle duration is the retry/latency proxy — the wait-for-ack "
            "backoff records no attempt count, so a rising p95 is the visible "
            "signal of network trouble. LIVE cycles poll for ack and run "
            "longer than dry-runs."
        )

    st.subheader("Cycle detail")
    # str() + or-fallback: a JSON-null cycle_id must not feed None into
    # the selectbox format_func (len(None) → TypeError).
    cycle_ids = [str(c.get("cycle_id") or "?") for c in reversed(cycles)]
    selected = st.selectbox(
        "Pick a cycle to expand",
        options=cycle_ids,
        format_func=lambda x: f"{x[:8]}…" if len(x) > 8 else x,
    )
    chosen = next((c for c in cycles if c.get("cycle_id") == selected), None)
    if chosen is not None:
        _render_cycle_detail(chosen)


def _render_cycle_detail(cycle: dict) -> None:
    cols = st.columns(2)
    cols[0].write({
        "cycle_id": cycle.get("cycle_id"),
        "outcome": cycle.get("outcome"),
        "started_at": cycle.get("started_at"),
        "ended_at": cycle.get("ended_at"),
        "duration_ms": cycle.get("duration_ms"),
        "git_commit": cycle.get("git_commit"),
        "python_version": cycle.get("python_version"),
        "schema_version": cycle.get("schema_version"),
    })
    cols[1].write(cycle.get("context") or {})

    planned = cycle.get("orders_planned") or []
    skipped = cycle.get("orders_skipped") or []
    executed = cycle_orders_executed(cycle)
    if planned:
        st.write("**Orders planned**")
        st.dataframe(pd.DataFrame(planned), width="stretch", hide_index=True)
    if skipped:
        st.write("**Orders skipped (sub-minimum)**")
        st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)
    if executed:
        st.write("**Orders executed**")
        exec_rows = [{
            "side": (o.get("side") or "").upper(),
            "symbol": o.get("symbol"),
            "status": o.get("terminal_status"),
            "intended": o.get("intended_amount"),
            "filled": o.get("filled_amount"),
            "notional": o.get("filled_notional_quote"),
            "avg_price": o.get("average_price"),
            "fees": o.get("fees_paid_quote"),
            "client_order_id": (o.get("client_order_id") or "")[:24],
        } for o in executed]
        st.dataframe(pd.DataFrame(exec_rows), width="stretch", hide_index=True)
        for o in executed:
            if o.get("error"):
                st.error(
                    f"{o.get('client_order_id', '?')}: "
                    f"{o['error'].get('type', '?')}: "
                    f"{o['error'].get('message', '?')}"
                )
    err = cycle.get("error")
    if err:
        st.error(f"{err.get('type', '?')}: {err.get('message', '?')}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation tab — forward-test harness + fingerprint + look-ahead detector
# ---------------------------------------------------------------------------


def _file_sig(path: Path) -> Optional[tuple]:
    """(mtime, size) of a file, or None if absent.

    A cache key that changes *exactly* when the file changes — so caching a
    computation on it recomputes as soon as new data lands and NEVER serves a
    stale result. This is what makes caching the breach checks safe: a new
    journal row bumps mtime → new key → re-evaluation (no masked breach).
    """
    try:
        stt = path.stat()
        return (stt.st_mtime, stt.st_size)
    except OSError:
        return None


def _dir_sig(root: Path, pattern: str = "*.txt") -> tuple:
    """(file_count, max_mtime, total_size) over ``pattern`` under ``root``.

    Cheap signature for the vintage directory: changes when a vintage is
    added (count), rewritten (max mtime), or truncated/grown (total size), so
    the look-ahead cache invalidates on vintage changes even if the live
    journal were unchanged. The default matches the vintage store's on-disk
    format — canonical TEXT under ``h[:2]/<hash>.txt`` (vintage_store.py), NOT
    parquet — and ``rglob`` recurses the two-level layout.
    """
    try:
        stats = [f.stat() for f in root.rglob(pattern)]
    except OSError:
        return (0, 0.0, 0)
    return (
        len(stats),
        max((s.st_mtime for s in stats), default=0.0),
        sum(s.st_size for s in stats),
    )


# The three heavy Validation computations run on EVERY 30s rerun because
# Streamlit executes all tab bodies each run. They are pure functions of their
# input files, so cache them on a file signature: recompute only when the
# journal / reference / vintages actually change. ``sig`` is the cache key;
# the ``_``-prefixed path args are excluded from Streamlit's arg hashing.
@st.cache_data(show_spinner=False)
def _cached_config_hash() -> str:
    from trade_lab.config import PRODUCTION_CONFIG, production_config_hash
    return production_config_hash(PRODUCTION_CONFIG)


@st.cache_data(show_spinner=False)
def _cached_validation_rows(_log_path: Path, sig):
    from trade_lab.paper_trading.journal import read_log
    return read_log(_log_path)


@st.cache_data(show_spinner=False)
def _cached_fingerprint(_log_path: Path, _reference_path: Path, sig):
    from trade_lab.paper_trading.fingerprint_monitor import (
        check_journal_against_reference,
    )
    return check_journal_against_reference(
        log_path=_log_path, reference_path=_reference_path,
    )


@st.cache_data(show_spinner=False)
def _cached_lookahead(_log_path: Path, _vintage_root: Path, sig):
    from trade_lab.paper_trading.lookahead_detector import (
        check_journal_for_lookahead,
    )
    return check_journal_for_lookahead(
        log_path=_log_path, vintage_root=_vintage_root,
    )


def _render_validation() -> None:
    """Read-only view of the validation forward-test infrastructure.

    Shows: frozen-config hash gate, journal stats, latest harness row,
    fingerprint-monitor breach status, and look-ahead-detector status.
    Imports validation modules directly — these are research-side code
    without exchange credentials, so the API-separation rationale that
    prevents the regular tabs from importing ``trade_lab.execution``
    does not apply here.

    Research-side modules and the harness journal drift faster than
    this dashboard; any failure here (ImportError on a renamed module,
    TypeError from a schema-drifted journal row) is contained to this
    tab by ``_render_tab_safely`` in :func:`main`.
    """
    from trade_lab.config import CANONICAL_HASH

    st.markdown("### Frozen-config gate")
    runtime_hash = _cached_config_hash()
    cols = st.columns(2)
    if runtime_hash == CANONICAL_HASH:
        cols[0].success("Hash MATCH — harness will run")
    else:
        cols[0].error("Hash DRIFT — harness will refuse to run")
    cols[1].code(f"{runtime_hash[:16]}…", language="text")

    st.markdown("### Validation journal")
    if not VALIDATION_LOG_PATH.exists():
        st.info(
            f"No validation journal yet at `{VALIDATION_LOG_PATH}`. "
            "Forward paper-clock has not started — run "
            "`python -m trade_lab.paper_trading.cli` daily to begin."
        )
        return

    rows = _cached_validation_rows(
        VALIDATION_LOG_PATH, _file_sig(VALIDATION_LOG_PATH),
    )
    cols = st.columns(4)
    cols[0].metric("Rows", len(rows))
    if rows:
        cols[1].metric("First date", rows[0].date)
        cols[2].metric("Last date", rows[-1].date)
        cols[3].metric("Latest ladder", f"{rows[-1].ladder_state:.2f}")
    else:
        for c in cols[1:]:
            c.metric("—", "—")

    if rows:
        latest = rows[-1]
        st.markdown("**Latest cycle**")
        cols = st.columns(4)
        cols[0].metric("Basket close", f"{latest.basket_close:.2f}")
        cols[1].metric(
            "SMA(200)",
            f"{latest.sma_value:.2f}" if latest.sma_value is not None else "—",
        )
        cols[2].metric("Gate", "OPEN" if latest.sma_gate_open else "CLOSED")
        cols[3].metric("Equity", f"${latest.portfolio_equity:.2f}")
        with st.expander("Per-lookback signals + intended trades"):
            st.write("**Per-lookback states / returns**")
            st.json({
                "states": latest.per_lookback_states,
                "returns": {k: f"{v*100:+.2f}%"
                            for k, v in latest.per_lookback_returns.items()},
            })
            st.write("**Target / intended trades (delta from prior)**")
            df = pd.DataFrame({
                "target_weight": latest.target_weights,
                "current_weight": latest.current_weights,
                "intended_delta": latest.intended_trades,
            })
            st.dataframe(df, width="stretch")

    st.markdown("### Behavioral fingerprint — live vs frozen reference")
    if not VALIDATION_REFERENCE_PATH.exists():
        st.warning(
            f"No reference fingerprint at `{VALIDATION_REFERENCE_PATH}`. "
            "Run `scripts/build_reference_fingerprint.py`."
        )
    else:
        try:
            report = _cached_fingerprint(
                VALIDATION_LOG_PATH,
                VALIDATION_REFERENCE_PATH,
                (_file_sig(VALIDATION_LOG_PATH),
                 _file_sig(VALIDATION_REFERENCE_PATH)),
            )
        except Exception as exc:
            # Broad on purpose (matches the look-ahead detector below):
            # a malformed reference file raises KeyError/TypeError just
            # as easily as ValueError, and any of them is a render-an-
            # error case, not a take-down-the-tab case.
            st.error(f"Fingerprint monitor error: {type(exc).__name__}: {exc}")
        else:
            cols = st.columns(3)
            cols[0].metric(
                "Drawdown headroom",
                f"{report.drawdown.headroom_pp:+.2f} pp",
            )
            cols[1].metric(
                "Multi-metric days",
                report.multi_metric_days,
            )
            cols[2].metric(
                "Sustained breach",
                "YES" if report.overall_sustained_breach else "no",
            )
            if report.drawdown.breached:
                st.error(report.advisory)
            elif report.overall_sustained_breach or report.overall_multi_metric_breach:
                st.warning(report.advisory)
            else:
                st.info(report.advisory)
            with st.expander("Per-metric live status"):
                for metric in (report.exposure_flip, report.regime_gate_flip):
                    st.write(f"**{metric.name}**")
                    st.write({
                        "latest": metric.latest_value,
                        "band [p05, p95]": [metric.p05, metric.p95],
                        "currently_breached": metric.currently_breached,
                        "consecutive_breach_days_now": metric.currently_consecutive_breach,
                        "longest_run_observed": metric.longest_consecutive_breach,
                    })

    st.markdown("### Look-ahead detector — live vs backtest replay")
    if not rows:
        st.info(
            "No live rows to check. Part A (`scripts/validation_lookahead_"
            "truncation_audit.py`) is the dispositive look-ahead test for "
            "the backtest path itself — it has run CLEAN (0 mismatches on "
            "1589 verified-window bars)."
        )
    else:
        try:
            la = _cached_lookahead(
                VALIDATION_LOG_PATH,
                VALIDATION_VINTAGE_ROOT,
                (_file_sig(VALIDATION_LOG_PATH),
                 _dir_sig(VALIDATION_VINTAGE_ROOT)),
            )
        except Exception as exc:
            st.error(f"Look-ahead detector error: {exc}")
        else:
            cols = st.columns(4)
            cols[0].metric("Match", la.n_match)
            cols[1].metric("Offset-1 (labeling)", la.n_offset_1_match)
            cols[2].metric("Random disagreement", la.n_random_disagreement)
            cols[3].metric("Vintage missing", la.n_vintage_missing)
            if la.random_disagreement_present:
                st.error(la.advisory)
            elif la.constant_offset_pattern:
                st.warning(la.advisory)
            elif la.n_match > 0:
                st.success(la.advisory)
            else:
                st.info(la.advisory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _humanize_iso(s: Optional[str]) -> str:
    if not s:
        return "—"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _humanize_relative(s: Optional[str], now: Optional[datetime] = None) -> str:
    """Compact relative-time string: '12s ago' / '5m ago' / '2h 30m ago' / '3d ago'.

    Designed for ``st.metric`` value cells, which truncate on narrow
    screens. Pair with ``_humanize_iso`` in a caption for the precise
    timestamp.
    """
    if not s:
        return "—"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(tz=timezone.utc)
    secs = int((now - dt).total_seconds())
    if secs < 0:
        return "in the future"
    if secs < 60:
        return f"{secs}s ago"
    mins, _ = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m ago"
    hours, rem_min = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {rem_min}m ago" if rem_min else f"{hours}h ago"
    days, rem_h = divmod(hours, 24)
    return f"{days}d {rem_h}h ago" if rem_h and days < 30 else f"{days}d ago"


# ---------------------------------------------------------------------------
# Research corpus (read-only) — Research tab + About modal
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _research_doc(relpath: str) -> str:
    """Cached read of one writeup's markdown (ttl so a deploy shows through)."""
    return research.read_markdown(relpath)


@st.cache_data(ttl=300)
def _research_title(relpath: str) -> str:
    return research.doc_title(relpath)


def _render_research() -> None:
    """Research tab: the master results index + a picker to read any writeup.

    Reads the repo markdown corpus — no journal, no exchange, no credentials.
    """
    st.markdown(
        "The full research corpus behind the deployable strategy — every "
        "writeup, shown net-of-cost and out-of-sample, including the rejected "
        "and inconclusive ideas."
    )
    with st.expander("Master results index — all strategies at a glance",
                     expanded=True):
        st.markdown(_research_doc(research.RESULTS_INDEX))

    st.divider()
    st.markdown("#### Read a full writeup")
    left, right = st.columns(2)
    group = left.selectbox(
        "Section", list(research.GROUPS.keys()), key="research_group")
    labels = {_research_title(p): p for p in research.GROUPS[group]}
    title = right.selectbox("Document", list(labels), key="research_doc")
    st.markdown(_research_doc(labels[title]))


@st.dialog("What trade-lab is", width="large")
def _about_dialog() -> None:
    """Modal overview: what the project is + the master results index."""
    st.markdown(
        "**trade-lab** backtests crypto-spot strategies with a **layered-"
        "honesty** stack — every edge is shown net-of-cost, then out-of-"
        "sample, then as a Deflated Sharpe Ratio at a fixed budget (N=500). "
        "One strategy survived every layer and is **paper-trading live on "
        "Binance testnet** — that is what this dashboard shows.\n\n"
        "Below is the master results index; full writeups are in the "
        "**Research** tab."
    )
    st.markdown(_research_doc(research.RESULTS_INDEX))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_TAB_NOTE = (
    "Other tabs are unaffected. The tab will render again on the next "
    "auto-refresh once the underlying data or module is fixed."
)


def _render_tab_safely(tab_name: str, render_fn) -> None:
    """Contain a tab's failure to that tab (thin wrapper over the shared
    :func:`trade_lab.uikit.render_tab_safely`, carrying this app's
    auto-refresh reassurance caption)."""
    render_tab_safely(tab_name, render_fn, note=_TAB_NOTE)


def _render_footer() -> None:
    """Subtle bottom-of-page footer: project + author links.

    All links are env-configurable; each renders only when its URL is set
    (``LINKEDIN_URL`` is empty by default, so LinkedIn stays hidden until a
    URL is provided). Read-only text — no data, no exchange, no credentials.
    """
    def _a(url: str, label: str) -> str:
        return (
            f"<a href='{url}' target='_blank' rel='noopener' "
            f"style='color:#8ab4f8;text-decoration:none;'>{label} ↗</a>"
        )

    links = []
    if REPO_URL:
        links.append(_a(REPO_URL, "GitHub"))
    if TELEGRAM_URL:
        links.append(_a(TELEGRAM_URL, "Telegram"))
    if LINKEDIN_URL:
        links.append(_a(LINKEDIN_URL, "LinkedIn"))

    line1 = "trade-lab" + ("&nbsp;·&nbsp;" + "&nbsp;·&nbsp;".join(links)
                           if links else "")
    st.markdown(
        f"<hr style='margin-top:2.5rem;margin-bottom:0.5rem;border:none;"
        f"border-top:1px solid #333;'>"
        f"<div style='text-align:center;color:#888;font-size:0.85rem;"
        f"line-height:1.7;'>{line1}<br>"
        f"Read-only monitoring · {AUTHOR_NAME}</div>",
        unsafe_allow_html=True,
    )


@st.fragment(run_every=REFRESH_SECONDS)
def _render_dashboard() -> None:
    """Everything below the title, re-rendered on the auto-refresh tick.

    A native ``st.fragment`` with ``run_every`` reruns THIS block every
    REFRESH_SECONDS to pull fresh journal data — no browser reload (the active
    tab and slider state survive) and, unlike the old streamlit-autorefresh
    component, no iframe and therefore no skeleton placeholder flashing on the
    page. Widget interactions here also rerun only this fragment, not the whole
    script.
    """
    reader = _get_reader()
    # The initial read and the safety banner run BEFORE the per-tab
    # containment, so an unexpected error here would blank the whole page —
    # including the mainnet banner. The reader is hardened to fail into a
    # read_error rather than raise, but keep a belt-and-suspenders guard so
    # nothing can take the banner down.
    try:
        latest = reader.latest_cycle()
    except Exception as exc:  # pragma: no cover - reader is hardened
        latest = None
        st.error(
            f"JOURNAL READ FAILED — {type(exc).__name__}: {exc}. Check the "
            f"journal path and permissions (`{JOURNAL_PATH}`)."
        )
    _render_top_banner(latest)
    _render_tab_safely("Health", lambda: _render_health_line(reader))

    (tab_status, tab_signal, tab_portfolio, tab_cycles,
     tab_validation, tab_research) = st.tabs(
        ["Status", "Signal", "Portfolio", "Cycles", "Validation", "📚 Research"]
    )
    with tab_status:
        _render_tab_safely("Status", lambda: _render_status(reader))
    with tab_signal:
        _render_tab_safely("Signal", lambda: _render_signal(reader))
    with tab_portfolio:
        _render_tab_safely("Portfolio", lambda: _render_portfolio(reader))
    with tab_cycles:
        _render_tab_safely("Cycles", lambda: _render_cycles(reader))
    with tab_validation:
        _render_tab_safely("Validation", _render_validation)
    with tab_research:
        _render_tab_safely("Research", _render_research)

    _render_footer()


def main() -> None:
    st.set_page_config(
        page_title="trade-lab monitoring",
        layout="wide",
    )
    # Static header rendered once; the dynamic body lives in an auto-rerunning
    # fragment (no skeleton-flashing autorefresh iframe).
    title_col, about_col = st.columns([4, 1])
    title_col.title("trade-lab monitoring")
    with about_col:
        st.write("")  # nudge the button down toward the title baseline
        if st.button(
            "📖 What's inside", use_container_width=True,
            help="Project overview + the master results index "
                 "(full writeups live in the Research tab).",
        ):
            _about_dialog()
    st.caption(
        f"Read-only dashboard for the paper-trading bot. Auto-refreshes every "
        f"{REFRESH_SECONDS}s — see the **📚 Research** tab for all findings & "
        f"results. Journal: `{JOURNAL_PATH}`."
    )
    _render_dashboard()


if __name__ == "__main__":
    main()
