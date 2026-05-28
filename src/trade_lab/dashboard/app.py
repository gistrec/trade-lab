"""Streamlit dashboard for inspecting trade-lab backtests."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_lab.backtest.engine import execution_bars, run_backtest
from trade_lab.backtest.metrics import Metrics, compute_metrics
from trade_lab.backtest.reports import trades_to_dataframe
from trade_lab.data.fetch_ohlcv import validate_ohlcv
from trade_lab.data.storage import filter_candles_by_date
from trade_lab.strategies.donchian_trend import DonchianTrendEnsembleStrategy
from trade_lab.strategies.pma_ratio import PriceMaRatioStrategy
from trade_lab.strategies.regime_only import RegimeOnlyStrategy
from trade_lab.strategies.regime_sma_cross import RegimeSMACrossStrategy
from trade_lab.strategies.rsi import RSIMeanReversionStrategy
from trade_lab.strategies.sma_cross import SMACrossStrategy
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy


@st.cache_data(show_spinner="Loading candles...")
def _load_candles(path: str) -> pd.DataFrame:
    """Cached candle loader keyed on the file path."""
    df = pd.read_parquet(path)
    validate_ohlcv(df)
    return df


def _build_strategy(name: str, params: dict):
    if name == "sma_cross":
        return SMACrossStrategy(
            fast_period=int(params["fast_period"]),
            slow_period=int(params["slow_period"]),
        )
    if name == "regime_sma_cross":
        return RegimeSMACrossStrategy(
            fast_period=int(params["fast_period"]),
            slow_period=int(params["slow_period"]),
            regime_period=int(params["regime_period"]),
        )
    if name == "regime_only":
        return RegimeOnlyStrategy(regime_period=int(params["regime_period"]))
    if name == "donchian_trend":
        return DonchianTrendEnsembleStrategy(
            donchian_lookbacks=params["donchian_lookbacks"],
            sma_filter_periods=params["sma_filter_periods"],
            vol_lookback=int(params["vol_lookback"]),
            annual_vol_target=float(params["annual_vol_target"]),
            max_position_size=float(params["max_position_size"]),
            rebalance_threshold=float(params["rebalance_threshold"]),
        )
    if name == "tsmom":
        return TimeSeriesMomentumStrategy(
            lookbacks=params["lookbacks"],
            sma_filter_periods=params["sma_filter_periods"],
            vol_lookback=int(params["vol_lookback"]),
            annual_vol_target=float(params["annual_vol_target"]),
            max_position_size=float(params["max_position_size"]),
            rebalance_threshold=float(params["rebalance_threshold"]),
            use_vol_target=bool(params["use_vol_target"]),
        )
    if name == "pma_ratio":
        return PriceMaRatioStrategy(
            ma_periods=params["ma_periods"],
            sma_filter_periods=params["sma_filter_periods"],
            vol_lookback=int(params["vol_lookback"]),
            annual_vol_target=float(params["annual_vol_target"]),
            max_position_size=float(params["max_position_size"]),
            rebalance_threshold=float(params["rebalance_threshold"]),
            use_vol_target=bool(params["use_vol_target"]),
        )
    if name == "rsi":
        return RSIMeanReversionStrategy(
            period=int(params["rsi_period"]),
            lower=float(params["buy_threshold"]),
            upper=float(params["sell_threshold"]),
        )
    raise ValueError(f"Unknown strategy: {name}")


def _sidebar_controls() -> dict:
    with st.sidebar:
        st.header("Data")
        path = st.text_input(
            "OHLCV Parquet path",
            value="data/binance_BTC_USDT_1h.parquet",
            help="Path to a Parquet file produced by `trade-lab fetch`.",
        )

        st.header("Strategy")
        strategy_name = st.selectbox(
            "Strategy",
            [
                "sma_cross",
                "regime_sma_cross",
                "regime_only",
                "donchian_trend",
                "tsmom",
                "pma_ratio",
                "rsi",
            ],
        )
        params: dict = {}
        if strategy_name == "sma_cross":
            params["fast_period"] = st.number_input(
                "fast_period", min_value=1, max_value=500, value=20, step=1
            )
            params["slow_period"] = st.number_input(
                "slow_period", min_value=2, max_value=1000, value=100, step=1
            )
        elif strategy_name == "regime_sma_cross":
            params["fast_period"] = st.number_input(
                "fast_period", min_value=1, max_value=500, value=20, step=1
            )
            params["slow_period"] = st.number_input(
                "slow_period", min_value=2, max_value=1000, value=100, step=1
            )
            params["regime_period"] = st.number_input(
                "regime_period", min_value=3, max_value=2000, value=200, step=1
            )
        elif strategy_name == "regime_only":
            params["regime_period"] = st.number_input(
                "regime_period", min_value=1, max_value=2000, value=200, step=1
            )
        elif strategy_name == "donchian_trend":
            params["donchian_lookbacks"] = st.text_input(
                "donchian_lookbacks (CSV)", value="20,50,100"
            )
            params["sma_filter_periods"] = st.text_input(
                "sma_filter_periods (CSV)", value="100,200"
            )
            params["vol_lookback"] = st.number_input(
                "vol_lookback", min_value=2, max_value=200, value=30, step=1
            )
            params["annual_vol_target"] = st.number_input(
                "annual_vol_target", min_value=0.01, max_value=2.0,
                value=0.25, step=0.05, format="%.2f",
            )
            params["max_position_size"] = st.number_input(
                "max_position_size", min_value=0.05, max_value=1.0,
                value=1.0, step=0.05, format="%.2f",
            )
            params["rebalance_threshold"] = st.number_input(
                "rebalance_threshold", min_value=0.0, max_value=0.5,
                value=0.05, step=0.01, format="%.3f",
            )
        elif strategy_name == "tsmom":
            params["lookbacks"] = st.text_input(
                "lookbacks (CSV)", value="28,60",
                help="(28, 60) matches the deployable cluster-stable config.",
            )
            params["sma_filter_periods"] = st.text_input(
                "sma_filter_periods (CSV)", value="200",
                help="Empty to disable the regime gate.",
            )
            params["vol_lookback"] = st.number_input(
                "vol_lookback", min_value=2, max_value=200, value=30, step=1
            )
            params["annual_vol_target"] = st.number_input(
                "annual_vol_target", min_value=0.01, max_value=2.0,
                value=0.25, step=0.05, format="%.2f",
            )
            params["max_position_size"] = st.number_input(
                "max_position_size", min_value=0.05, max_value=1.0,
                value=1.0, step=0.05, format="%.2f",
            )
            params["rebalance_threshold"] = st.number_input(
                "rebalance_threshold", min_value=0.0, max_value=0.5,
                value=0.05, step=0.01, format="%.3f",
            )
            params["use_vol_target"] = st.checkbox(
                "use_vol_target", value=False,
                help="Off for the deployable (28, 60) config.",
            )
        elif strategy_name == "pma_ratio":
            params["ma_periods"] = st.text_input(
                "ma_periods (CSV)", value="5,10,20,50,100",
                help="Detzel et al. (2021) use {5, 10, 20, 50, 100}.",
            )
            params["sma_filter_periods"] = st.text_input(
                "sma_filter_periods (CSV)", value="",
                help="Empty follows the paper exactly.",
            )
            params["vol_lookback"] = st.number_input(
                "vol_lookback", min_value=2, max_value=200, value=30, step=1
            )
            params["annual_vol_target"] = st.number_input(
                "annual_vol_target", min_value=0.01, max_value=2.0,
                value=0.25, step=0.05, format="%.2f",
            )
            params["max_position_size"] = st.number_input(
                "max_position_size", min_value=0.05, max_value=1.0,
                value=1.0, step=0.05, format="%.2f",
            )
            params["rebalance_threshold"] = st.number_input(
                "rebalance_threshold", min_value=0.0, max_value=0.5,
                value=0.05, step=0.01, format="%.3f",
            )
            params["use_vol_target"] = st.checkbox(
                "use_vol_target", value=True,
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


def _render_metric_cards(metrics: Metrics) -> None:
    cols = st.columns(7)
    cols[0].metric("Final equity", f"${metrics.final_equity:,.2f}")
    cols[1].metric(
        "Total return",
        f"{metrics.total_return:.2%}",
        delta=f"{(metrics.total_return - metrics.buy_and_hold_return) * 100:.2f}pp vs B&H",
    )
    cols[2].metric("Buy & hold", f"{metrics.buy_and_hold_return:.2%}")
    cols[3].metric("Max drawdown", f"{metrics.max_drawdown:.2%}")
    cols[4].metric("# Trades", f"{metrics.num_trades}")
    cols[5].metric("Win rate", f"{metrics.win_rate:.2%}")
    cols[6].metric("Fees paid", f"${metrics.total_fees:,.2f}")


def _render_warnings(metrics: Metrics, n_bars: int) -> None:
    if metrics.total_return < metrics.buy_and_hold_return:
        diff_pp = (metrics.buy_and_hold_return - metrics.total_return) * 100
        st.warning(
            f"Strategy underperformed buy & hold by {diff_pp:.2f}pp "
            f"(strategy {metrics.total_return:+.2%} vs B&H "
            f"{metrics.buy_and_hold_return:+.2%}). The strategy may not be "
            f"adding alpha on this window."
        )
    if (
        metrics.buy_and_hold_max_drawdown > 0
        and metrics.max_drawdown > metrics.buy_and_hold_max_drawdown
    ):
        st.warning(
            f"Strategy max drawdown ({metrics.max_drawdown:.2%}) is worse than "
            f"buy & hold ({metrics.buy_and_hold_max_drawdown:.2%}). Holding "
            f"the asset passively was less painful here."
        )
    # Scale "high" with the window size: > one trade per 100 bars, with a
    # floor of 20 to avoid spurious warnings on very short backtests.
    high_threshold = max(20, n_bars // 100)
    if metrics.num_trades > high_threshold:
        st.warning(
            f"Trade count is high ({metrics.num_trades} trades over {n_bars} "
            f"bars; fees paid ${metrics.total_fees:,.2f}). Frequent rebalancing "
            f"erodes net return — check whether the signal is too jumpy."
        )


def _price_figure(candles: pd.DataFrame, positions: pd.Series) -> go.Figure:
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
        height=480, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


def _equity_figure(result, initial_cash: float) -> go.Figure:
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
        height=480, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


def _drawdown_figure(equity: pd.Series) -> go.Figure:
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
        height=420, margin=dict(t=10, b=10, l=10, r=10),
        yaxis_tickformat=".0%", hovermode="x unified", showlegend=False,
    )
    return fig


def _metrics_to_csv(metrics: Metrics) -> bytes:
    return pd.DataFrame([asdict(metrics)]).to_csv(index=False).encode("utf-8")


def _equity_to_csv(result) -> bytes:
    running_max = result.equity.cummax()
    drawdown = (result.equity - running_max) / running_max
    df = pd.DataFrame(
        {
            "strategy_equity": result.equity,
            "strategy_drawdown": drawdown,
        }
    )
    if not result.buy_and_hold_equity.empty:
        df["buy_and_hold_equity"] = result.buy_and_hold_equity
    df.index.name = "timestamp"
    return df.to_csv().encode("utf-8")


def _trades_to_csv(result, candles: pd.DataFrame) -> bytes:
    df = trades_to_dataframe(result, candles, include_open=True)
    return df.to_csv(index=False).encode("utf-8")


def _render_overview_tab(
    metrics: Metrics, candles: pd.DataFrame, controls: dict
) -> None:
    st.subheader("Run summary")
    st.write(
        f"**Strategy:** `{controls['strategy_name']}` with params "
        f"`{controls['params']}`."
    )
    st.write(
        f"**Window:** {candles.index[0]:%Y-%m-%d %H:%M} → "
        f"{candles.index[-1]:%Y-%m-%d %H:%M}  ({len(candles)} bars)."
    )
    st.write(
        f"**Initial cash:** ${controls['initial_cash']:,.2f}  ·  "
        f"**Fee:** {controls['fee_rate']:.4%}  ·  "
        f"**Slippage:** {controls['slippage_rate']:.4%}  ·  "
        f"**Position size:** {controls['position_size']:.0%}"
    )
    st.divider()
    st.subheader("Downloads")
    st.download_button(
        "Download metrics (CSV)",
        data=_metrics_to_csv(metrics),
        file_name="metrics.csv",
        mime="text/csv",
    )


def _render_price_tab(candles: pd.DataFrame, positions: pd.Series) -> None:
    st.plotly_chart(_price_figure(candles, positions), use_container_width=True)
    st.caption(
        "Markers are placed on execution candles — one bar after the signal "
        "candle, by construction of the look-ahead-protecting shift."
    )


def _render_equity_tab(result, initial_cash: float) -> None:
    st.plotly_chart(_equity_figure(result, initial_cash), use_container_width=True)
    st.download_button(
        "Download equity curve (CSV)",
        data=_equity_to_csv(result),
        file_name="equity.csv",
        mime="text/csv",
    )


def _render_drawdown_tab(equity: pd.Series) -> None:
    st.plotly_chart(_drawdown_figure(equity), use_container_width=True)


def _render_trades_tab(result, candles: pd.DataFrame) -> None:
    trades_df = trades_to_dataframe(result, candles, include_open=True)
    if trades_df.empty:
        st.info("No trades on this window.")
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download trades (CSV)",
        data=_trades_to_csv(result, candles),
        file_name="trades.csv",
        mime="text/csv",
    )


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

    # Re-run on every interaction — params change too often to memoize.
    result = run_backtest(
        candles=candles,
        strategy=strategy,
        initial_capital=controls["initial_cash"],
        fee_rate=controls["fee_rate"],
        slippage_rate=controls["slippage_rate"],
        position_size=controls["position_size"],
    )
    metrics = compute_metrics(result)

    _render_metric_cards(metrics)
    _render_warnings(metrics, n_bars=len(candles))

    tab_overview, tab_price, tab_equity, tab_dd, tab_trades = st.tabs(
        ["Overview", "Price & Trades", "Equity", "Drawdown", "Trades"]
    )

    with tab_overview:
        _render_overview_tab(metrics, candles, controls)
    with tab_price:
        _render_price_tab(candles, result.positions)
    with tab_equity:
        _render_equity_tab(result, controls["initial_cash"])
    with tab_dd:
        _render_drawdown_tab(result.equity)
    with tab_trades:
        _render_trades_tab(result, candles)


if __name__ == "__main__":
    main()
