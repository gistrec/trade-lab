"""Streamlit dashboard for inspecting trade-lab backtests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_lab.backtest.engine import execution_bars, run_backtest
from trade_lab.backtest.metrics import compute_metrics
from trade_lab.backtest.reports import trades_to_dataframe
from trade_lab.data.fetch_ohlcv import validate_ohlcv
from trade_lab.data.storage import filter_candles_by_date
from trade_lab.strategies.rsi import RSIMeanReversionStrategy
from trade_lab.strategies.sma_cross import SMACrossStrategy


@st.cache_data(show_spinner="Loading candles...")
def _load_candles(path: str) -> pd.DataFrame:
    """Cached candle loader keyed on the file path."""
    df = pd.read_parquet(path)
    validate_ohlcv(df)
    return df


def _build_strategy(name: str, params: dict):
    """Instantiate the chosen strategy or surface a friendly error."""
    if name == "sma_cross":
        return SMACrossStrategy(
            fast_period=int(params["fast_period"]),
            slow_period=int(params["slow_period"]),
        )
    if name == "rsi":
        return RSIMeanReversionStrategy(
            period=int(params["rsi_period"]),
            lower=float(params["buy_threshold"]),
            upper=float(params["sell_threshold"]),
        )
    raise ValueError(f"Unknown strategy: {name}")


def _sidebar_controls() -> dict:
    """Render all sidebar controls and return their values."""
    with st.sidebar:
        st.header("Data")
        path = st.text_input(
            "OHLCV Parquet path",
            value="data/binance_BTC_USDT_1h.parquet",
            help="Path to a Parquet file produced by `trade-lab fetch`.",
        )

        st.header("Strategy")
        strategy_name = st.selectbox("Strategy", ["sma_cross", "rsi"])
        params: dict = {}
        if strategy_name == "sma_cross":
            params["fast_period"] = st.number_input(
                "fast_period", min_value=1, max_value=500, value=20, step=1
            )
            params["slow_period"] = st.number_input(
                "slow_period", min_value=2, max_value=1000, value=100, step=1
            )
        else:
            params["rsi_period"] = st.number_input(
                "rsi_period", min_value=2, max_value=200, value=14, step=1
            )
            params["buy_threshold"] = st.number_input(
                "buy_threshold", min_value=1.0, max_value=99.0, value=30.0, step=1.0
            )
            params["sell_threshold"] = st.number_input(
                "sell_threshold", min_value=1.0, max_value=99.0, value=70.0, step=1.0
            )

        st.header("Backtest")
        initial_cash = st.number_input(
            "Initial cash ($)", min_value=100.0, value=10_000.0, step=1_000.0
        )
        fee_rate = st.number_input(
            "Fee rate (per side)", min_value=0.0, value=0.001, step=0.0001, format="%.4f"
        )
        slippage = st.number_input(
            "Slippage rate (per side)",
            min_value=0.0, value=0.0005, step=0.0001, format="%.4f",
        )
        position_size = st.slider(
            "Position size", min_value=0.05, max_value=1.0, value=1.0, step=0.05
        )

        st.header("Date range (optional)")
        start_date = st.text_input("Start date (YYYY-MM-DD)", value="")
        end_date = st.text_input("End date (YYYY-MM-DD)", value="")

    return {
        "path": path,
        "strategy_name": strategy_name,
        "params": params,
        "initial_cash": initial_cash,
        "fee_rate": fee_rate,
        "slippage_rate": slippage,
        "position_size": position_size,
        "start_date": start_date.strip() or None,
        "end_date": end_date.strip() or None,
    }


def _render_metrics(metrics) -> None:
    cols = st.columns(7)
    cols[0].metric("Final equity", f"${metrics.final_equity:,.2f}")
    cols[1].metric("Total return", f"{metrics.total_return:.2%}")
    cols[2].metric("Buy & hold", f"{metrics.buy_and_hold_return:.2%}")
    cols[3].metric("Max drawdown", f"{metrics.max_drawdown:.2%}")
    cols[4].metric("# Trades", f"{metrics.num_trades}")
    cols[5].metric("Win rate", f"{metrics.win_rate:.2%}")
    cols[6].metric("Fees paid", f"${metrics.total_fees:,.2f}")


def _render_price_panel(candles: pd.DataFrame, positions: pd.Series) -> None:
    entries, exits = execution_bars(positions)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=candles.index, y=candles["close"],
            name="Close", line=dict(color="lightgray", width=1.2),
        )
    )
    if entries:
        fig.add_trace(
            go.Scatter(
                x=[candles.index[i] for i in entries],
                y=[candles["close"].iloc[i] for i in entries],
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", color="#2ca02c", size=11),
            )
        )
    if exits:
        fig.add_trace(
            go.Scatter(
                x=[candles.index[i] for i in exits],
                y=[candles["close"].iloc[i] for i in exits],
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", color="#d62728", size=11),
            )
        )
    fig.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_equity_panel(result, initial_cash: float) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index, y=result.equity.values,
            name="Strategy", line=dict(color="#1f77b4", width=2),
        )
    )
    if not result.buy_and_hold_equity.empty:
        bh = result.buy_and_hold_equity
        fig.add_trace(
            go.Scatter(
                x=bh.index, y=bh.values,
                name="Buy & hold",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            )
        )
    fig.add_hline(
        y=initial_cash, line_dash="dot", line_color="gray",
        annotation_text="Initial",
    )
    fig.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_drawdown_panel(equity: pd.Series) -> None:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", name="Drawdown",
            line=dict(color="#d62728", width=1),
        )
    )
    fig.update_layout(
        height=260, margin=dict(t=10, b=10, l=10, r=10),
        yaxis_tickformat=".0%", hovermode="x unified", showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="trade-lab dashboard", layout="wide")
    st.title("trade-lab — backtest dashboard")
    st.caption(
        "Interactive view over a single backtest. Adjust the sidebar; the "
        "page re-runs automatically."
    )

    controls = _sidebar_controls()

    if not Path(controls["path"]).exists():
        st.error(
            f"File not found: `{controls['path']}`. "
            "Run `trade-lab fetch` first, or point at an existing Parquet."
        )
        st.stop()

    try:
        candles = _load_candles(controls["path"])
    except Exception as exc:
        st.error(f"Could not load candles: {exc}")
        st.stop()

    try:
        candles = filter_candles_by_date(
            candles,
            start_date=controls["start_date"],
            end_date=controls["end_date"],
        )
    except Exception as exc:
        st.error(f"Date filter error: {exc}")
        st.stop()

    if candles.empty:
        st.warning("No candles in the selected date range.")
        st.stop()

    try:
        strategy = _build_strategy(controls["strategy_name"], controls["params"])
    except ValueError as exc:
        st.error(f"Strategy error: {exc}")
        st.stop()

    # Run the backtest fresh on every interaction — params change too often
    # to memoize safely, and a single run on 1-2 years of hourly bars is
    # essentially free.
    result = run_backtest(
        candles=candles,
        strategy=strategy,
        initial_capital=controls["initial_cash"],
        fee_rate=controls["fee_rate"],
        slippage_rate=controls["slippage_rate"],
        position_size=controls["position_size"],
    )
    metrics = compute_metrics(result)

    st.caption(
        f"Period **{candles.index[0]:%Y-%m-%d %H:%M}** → "
        f"**{candles.index[-1]:%Y-%m-%d %H:%M}** "
        f"({len(candles)} bars)"
    )

    _render_metrics(metrics)

    st.subheader("Price & trades")
    _render_price_panel(candles, result.positions)

    st.subheader("Strategy equity vs buy & hold")
    _render_equity_panel(result, controls["initial_cash"])

    st.subheader("Drawdown")
    _render_drawdown_panel(result.equity)

    st.subheader("Trades")
    trades_df = trades_to_dataframe(result, candles, include_open=True)
    st.dataframe(trades_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
