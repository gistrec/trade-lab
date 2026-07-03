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

Autorefresh uses ``streamlit-autorefresh`` (a thin JS component that
triggers a Streamlit rerun without reloading the page). An HTML
``<meta>`` tag would also tick but at the cost of a full page reload,
which throws away the active tab and any other session state — for a
multi-tab dashboard that is operationally annoying every refresh.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from trade_lab.monitoring.data_source import (
    JournalReader, ReadStats, Staleness, cycle_orders_executed, parse_iso,
)


JOURNAL_PATH = os.environ.get(
    "TRADE_LAB_MONITORING_JOURNAL_PATH",
    "data/journal/cycles.jsonl",
)
EXPECTED_INTERVAL_S = int(
    os.environ.get("MONITORING_EXPECTED_CYCLE_INTERVAL_SECONDS", "3600")
)
REFRESH_SECONDS = int(os.environ.get("MONITORING_REFRESH_SECONDS", "30"))

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
        cols[2].metric("Last duration", f"{latest.get('duration_ms', 0)} ms")
        cols[3].metric("Outcome", str(latest.get("outcome") or "?").upper())
    else:
        for c in cols[1:]:
            c.metric("—", "—")

    if last_ended_iso:
        st.caption(f"Last cycle ended at {_humanize_iso(last_ended_iso)}")

    if staleness is Staleness.DOWN:
        st.error(
            f"Bot appears DOWN: last cycle was over "
            f"{int(EXPECTED_INTERVAL_S * 10)}s ago "
            f"(threshold = 10× expected interval of {EXPECTED_INTERVAL_S}s)."
        )
    elif staleness is Staleness.STALE:
        st.warning(
            f"Bot is STALE: last cycle elapsed > "
            f"{int(EXPECTED_INTERVAL_S * 1.5)}s "
            f"(threshold = 1.5× expected interval of {EXPECTED_INTERVAL_S}s)."
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

    drift = reader.cumulative_skipped_drift()
    if drift > 0:
        st.warning(
            f"Cumulative skipped-order drift: ${drift:,.2f} across "
            f"{stats.valid_cycles} cycles. Sub-min divergence is normal "
            f"on tiny balances; investigate if it grows steadily."
        )

    _render_read_stats(stats)


def _render_read_stats(stats: ReadStats) -> None:
    if stats.corrupt_lines > 0 or stats.unknown_version_lines > 0:
        st.caption(
            f"Journal scan: {stats.valid_cycles} valid, "
            f"{stats.corrupt_lines} corrupt, "
            f"{stats.unknown_version_lines} unknown-version "
            f"(of {stats.total_lines} non-empty lines)."
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
    cols[0].metric("Ladder value", f"{sig.get('ladder_value', 0.0):.2f}")
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
    """Format ``values[-1] / values[-(n+1)] - 1`` as a +/- percent."""
    if len(values) < n_days_ago + 1:
        return "—"
    today = float(values[-1])
    past = float(values[-(n_days_ago + 1)])
    if past == 0:
        return "—"
    return f"{(today / past - 1.0) * 100:+.2f}%"


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
    )
    return fig


def _signal_history_figure(
    history: list[tuple[datetime, float, bool]],
) -> go.Figure:
    times = [h[0] for h in history]
    values = [h[1] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines+markers",
        name="Ladder", line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(
        height=360, margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(range=[-0.05, 1.05], tickvals=[0.0, 0.5, 1.0]),
        hovermode="x unified",
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
        t = float(target.get(asset, 0.0))
        c = float(current.get(asset, 0.0))
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
    st.dataframe(df, width="stretch", hide_index=True)

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
    from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG, production_config_hash
    from trade_lab.paper_trading.fingerprint_monitor import (
        check_journal_against_reference,
    )
    from trade_lab.paper_trading.journal import read_log
    from trade_lab.paper_trading.lookahead_detector import (
        check_journal_for_lookahead,
    )

    st.markdown("### Frozen-config gate")
    runtime_hash = production_config_hash(PRODUCTION_CONFIG)
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

    rows = read_log(VALIDATION_LOG_PATH)
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
            report = check_journal_against_reference(
                log_path=VALIDATION_LOG_PATH,
                reference_path=VALIDATION_REFERENCE_PATH,
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
            la = check_journal_for_lookahead(
                log_path=VALIDATION_LOG_PATH,
                vintage_root=VALIDATION_VINTAGE_ROOT,
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
# Entry point
# ---------------------------------------------------------------------------


def _render_tab_safely(tab_name: str, render_fn) -> None:
    """Contain a tab's failure to that tab.

    Streamlit aborts the whole script run on an uncaught exception, so
    a single drifted research-side module or journal row would
    otherwise blank every tab at once. The error stays loud — rendered
    red inside the failing tab — without taking down the safety
    banner and the other tabs.
    """
    try:
        render_fn()
    except Exception as exc:
        st.error(f"{tab_name} tab failed: {type(exc).__name__}: {exc}")
        st.caption(
            "Other tabs are unaffected. The tab will render again on "
            "the next auto-refresh once the underlying data or module "
            "is fixed."
        )


def main() -> None:
    st.set_page_config(
        page_title="trade-lab monitoring",
        layout="wide",
    )
    # Streamlit rerun every REFRESH_SECONDS. No browser reload, so the
    # active tab and other session state survive the tick.
    st_autorefresh(
        interval=REFRESH_SECONDS * 1000,
        key="monitoring_autorefresh",
    )

    st.title("trade-lab monitoring")
    st.caption(
        f"Read-only dashboard for the paper-trading bot. "
        f"Auto-refreshes every {REFRESH_SECONDS}s. Journal: `{JOURNAL_PATH}`."
    )

    reader = _get_reader()
    latest = reader.latest_cycle()
    _render_top_banner(latest)

    tab_status, tab_signal, tab_portfolio, tab_cycles, tab_validation = st.tabs(
        ["Status", "Signal", "Portfolio", "Cycles", "Validation"]
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


if __name__ == "__main__":
    main()
