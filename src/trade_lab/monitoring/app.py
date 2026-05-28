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

Autorefresh uses an HTML ``<meta>`` tag rather than a third-party
package — keeps the dependency surface minimal and survives Streamlit
version changes without breakage. Full page reload, no preserved
scroll position; acceptable for a single-screen monitoring view.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_lab.monitoring.data_source import (
    JournalReader, ReadStats, Staleness,
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

    cols = st.columns(3)
    cols[0].metric(
        "Ladder value",
        f"{sig.get('ladder_value', 0.0):.2f}" if sig else "—",
    )
    gate_open = sig.get("sma_gate_open") if sig else None
    cols[1].metric(
        "SMA(200) gate",
        "OPEN" if gate_open else ("CLOSED" if gate_open is False else "—"),
    )
    cols[2].metric(
        "Basket close",
        f"{sig.get('basket_close', 0.0):,.2f}" if sig else "—",
    )

    history_days = st.select_slider(
        "History window",
        options=[7, 30, 90, 180, 365],
        value=30,
    )
    history = reader.signal_history(days=history_days)
    if history:
        st.plotly_chart(_signal_history_figure(history), width="stretch")
    else:
        st.info("No signal history in the selected window.")

    st.subheader("Per-lookback breakdown (latest cycle)")
    plb = (sig or {}).get("per_lookback_states") or {}
    if plb:
        df = pd.DataFrame(
            [{"lookback": int(k), "state": int(v)} for k, v in plb.items()]
        ).sort_values("lookback")
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(
            "State is the pre-gate {0, 1} sign of pct_change(lookback). "
            "Averaged → ladder; SMA(200) gate then zeroes it if closed."
        )
    else:
        st.info("Per-lookback states not available in the latest cycle.")


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
            "orders_planned": len(c.get("orders_planned") or []),
            "orders_skipped": len(c.get("orders_skipped") or []),
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
    if planned:
        st.write("**Orders planned**")
        st.dataframe(pd.DataFrame(planned), width="stretch", hide_index=True)
    if skipped:
        st.write("**Orders skipped (sub-minimum)**")
        st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)
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
    # Native auto-refresh: full page reload every REFRESH_SECONDS.
    # No third-party dependency; survives Streamlit version changes.
    st.markdown(
        f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">',
        unsafe_allow_html=True,
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
