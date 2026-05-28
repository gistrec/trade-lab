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
from datetime import datetime, timezone
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


# Single reader instance reused across reruns. JournalReader is
# cache-aware (mtime-based), so this is safe and cheap.
@st.cache_resource
def _get_reader() -> JournalReader:
    return JournalReader(JOURNAL_PATH)


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
    ctx = latest.get("context") or {}
    sandbox = bool(ctx.get("sandbox", True))
    exchange = str(ctx.get("exchange") or "unknown").upper()
    if sandbox:
        st.markdown(
            f"<div style='background:#1b5e20;color:white;padding:0.8rem;"
            f"border-radius:0.5rem;text-align:center;font-size:1.4rem;'>"
            f"TESTNET — {exchange}</div>",
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
    if latest is not None:
        cols[1].metric("Last cycle", _humanize_iso(latest.get("ended_at")))
        cols[2].metric("Last duration", f"{latest.get('duration_ms', 0)} ms")
        cols[3].metric("Outcome", str(latest.get("outcome") or "?").upper())
    else:
        for c in cols[1:]:
            c.metric("—", "—")

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
        rows = []
        for k in sorted(plb_states.keys(), key=lambda x: int(x)):
            ret = plb_returns.get(k)
            rows.append({
                "lookback": int(k),
                "state": int(plb_states[k]),
                "return %": f"{ret * 100:+.2f}" if ret is not None else "—",
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
    """Count cycles back to the most recent OPEN gate. None if never seen.

    Walks the journal newest-first across up to 500 cycles — enough for
    a year of daily cron at hourly dry-run cadence without scanning the
    whole file every refresh.
    """
    cycles = reader.cycles(n=500)
    days = 0
    for c in reversed(cycles):
        sig = c.get("signal") or {}
        if sig.get("sma_gate_open") is True:
            return days
        days += 1
    return None


def _basket_close_figure(
    values: list,
    start_iso: Optional[str],
    sma_value: Optional[float],
) -> go.Figure:
    """Basket close line with horizontal SMA(200) reference."""
    if start_iso:
        try:
            from datetime import timedelta
            start = parse_iso(start_iso)
            x = [start + timedelta(days=i) for i in range(len(values))]
        except (ValueError, ImportError):
            x = list(range(len(values)))
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


def _render_portfolio(reader: JournalReader) -> None:
    latest = reader.latest_cycle()
    if latest is None or latest.get("outcome") != "success":
        st.info("No successful cycle yet to compute portfolio drift from.")
        return
    target = latest.get("target_allocation") or {}
    current = latest.get("current_holdings_quote") or {}
    equity = float(latest.get("equity_usd") or 0.0)
    quote = (latest.get("context") or {}).get("quote_currency") or "USD"

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
    # the Cycles tab.
    planned_count = len(latest.get("orders_planned") or [])
    executed = cycle_orders_executed(latest)
    fully_closed = sum(1 for o in executed if o.get("terminal_status") == "closed")
    unfilled = planned_count - fully_closed
    if planned_count > 0 and unfilled > 0:
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
    cycle_ids = [c.get("cycle_id", "?") for c in reversed(cycles)]
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


def _humanize_iso(s: Optional[str]) -> str:
    if not s:
        return "—"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    tab_status, tab_signal, tab_portfolio, tab_cycles = st.tabs(
        ["Status", "Signal", "Portfolio", "Cycles"]
    )
    with tab_status:
        _render_status(reader)
    with tab_signal:
        _render_signal(reader)
    with tab_portfolio:
        _render_portfolio(reader)
    with tab_cycles:
        _render_cycles(reader)


if __name__ == "__main__":
    main()
