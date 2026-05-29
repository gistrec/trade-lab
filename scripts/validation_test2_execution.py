"""Validation Test 2 — Kraken cost-tax against the FROZEN config.

Goal: measure the *marginal* Binance→Kraken cost-tax. Sharpe inputs
are already net of Binance fees in PRODUCTION_CONFIG (0.10% taker +
5 bps slippage). Here we keep the Binance-built basket and Binance
signals frozen and swap **only** the strategy-level fee + slippage
to model what the same equity curve would have done if executed
through Kraken's taker schedule (0.40%) + wider Kraken half-spreads.

Why this isolation
------------------
* Kraken's REST 720-bar cap means a venue-data swap is impossible
  pre-2024. The clean cost-tax measurement is on FROZEN Binance
  signals + swapped cost params, not on swapped data + rebuilt index.
* Composition is frozen (CLAUDE.md hard rule). BNB-on-Kraken is
  thin-liquidity (Test 1 RISK FLAG, corr 0.9661) — modeled here as a
  wider basket-average half-spread sensitivity band, NOT as
  basket-shrinkage. Excluding an asset would break the frozen
  composition; that decision belongs to the harness/deployment
  layer (Test 3), not to a cost-tax measurement.

Reported per scenario
---------------------
* Full-sample net Sharpe (2018-01-01 → 2026-05-28; the headline 0.72
  reference point is the verified-window net Sharpe, not this one).
* Verified-window net Sharpe (2022-01-21 → 2026-05-28; the
  Binance ↔ Bybit-confirmed 1589-bar block from Test 1 — this is the
  Test 2 go/no-go number).
* Annual exposure-flip count (ladder transitions in {0, 0.5, 1.0} +
  SMA-gate flips) — the tax DRIVER.
* Annualized cost-drag in bps (gross CAGR − net CAGR), decomposed
  into fee-drag and slippage-drag.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_lab.backtest.cross_sectional import _sharpe
from trade_lab.backtest.engine import run_backtest
from trade_lab.backtest.market_index import build_crypto_market_index
from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

VERIFIED_START = pd.Timestamp("2022-01-21", tz="UTC")
VERIFIED_END = pd.Timestamp("2026-05-28", tz="UTC")

# Cost scenarios. Binance baseline reproduces the headline numbers;
# the three Kraken scenarios sweep slippage as a sensitivity band.
# Half-spreads are basket-weighted averages over the 7 majors, with
# BTC/ETH tight (~3 bps), SOL/ADA/XRP/DOGE moderate (~5-10 bps), and
# BNB the wide outlier (~25 bps on the Kraken thin-liquidity pair).
COST_SCENARIOS = {
    "binance_baseline":  {"fee_rate": 0.0010, "slippage_rate": 0.0005,
                          "label": "Binance taker 0.10% + 5 bps half-spread"},
    "kraken_tight":      {"fee_rate": 0.0040, "slippage_rate": 0.0007,
                          "label": "Kraken taker 0.40% + 7 bps half-spread (best case)"},
    "kraken_mid":        {"fee_rate": 0.0040, "slippage_rate": 0.0010,
                          "label": "Kraken taker 0.40% + 10 bps half-spread (defensible mid)"},
    "kraken_wide":       {"fee_rate": 0.0040, "slippage_rate": 0.0015,
                          "label": "Kraken taker 0.40% + 15 bps half-spread (conservative)"},
}


def load_binance_basket() -> tuple[pd.DataFrame, dict]:
    """Build the FROZEN basket index from Binance parquets.

    The basket itself is built with PRODUCTION_CONFIG's Binance cost
    rates because that is how the canonical index series was
    constructed historically. The marginal swap below is at the
    strategy-engine level only.
    """
    asset_candles: dict[str, pd.DataFrame] = {}
    for sym in PRODUCTION_CONFIG.assets:
        p = DATA_DIR / f"binance_{sym}_USDT_1d.parquet"
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        asset_candles[sym] = df

    basket = build_crypto_market_index(
        asset_candles,
        initial_capital=PRODUCTION_CONFIG.initial_capital,
        fee_rate=PRODUCTION_CONFIG.fee_rate,
        slippage_rate=PRODUCTION_CONFIG.slippage_rate,
        rebalance_freq=PRODUCTION_CONFIG.basket_rebalance_freq,
    )
    return basket, asset_candles


def make_strategy():
    cfg = PRODUCTION_CONFIG
    return TimeSeriesMomentumStrategy(
        lookbacks=cfg.lookbacks,
        sma_filter_periods=cfg.sma_filter_periods,
        use_vol_target=cfg.use_vol_target,
        vol_lookback=cfg.vol_lookback,
        annual_vol_target=cfg.annual_vol_target,
        max_position_size=cfg.max_position_size,
        rebalance_threshold=cfg.rebalance_threshold,
        annualization_factor=cfg.annualization_factor,
    )


def _annual_flip_count(positions: pd.Series) -> tuple[int, float]:
    """Number of position changes + flips per year.

    A "flip" = any non-zero turnover step (positions[t] != positions[t-1]).
    Annualized by bar count / 365.
    """
    diff = positions.diff().abs()
    flips = int((diff > 1e-12).sum())
    years = len(positions) / 365.0 if len(positions) else 0.0
    per_year = flips / years if years else 0.0
    return flips, per_year


def _cagr(equity: pd.Series) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / 365.0
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def _slice_window(series: pd.Series, start, end) -> pd.Series:
    return series[(series.index >= start) & (series.index <= end)]


def evaluate(basket: pd.DataFrame, scenario_name: str, scenario: dict) -> dict:
    strat = make_strategy()
    bt = run_backtest(
        basket,
        strat,
        initial_capital=PRODUCTION_CONFIG.initial_capital,
        fee_rate=scenario["fee_rate"],
        slippage_rate=scenario["slippage_rate"],
        position_size=PRODUCTION_CONFIG.max_position_size,
    )
    ann = PRODUCTION_CONFIG.annualization_factor

    full_returns = bt.returns
    full_eq = bt.equity
    full_gross = bt.gross_equity
    full_positions = bt.positions

    full_sharpe = _sharpe(full_returns, ann)
    full_total_return = float(full_eq.iloc[-1] / full_eq.iloc[0] - 1.0)
    full_net_cagr = _cagr(full_eq)
    full_gross_cagr = _cagr(full_gross)
    full_cost_drag_bps = (full_gross_cagr - full_net_cagr) * 1e4

    # Verified window (2022-01-21 → 2026-05-28)
    v_rets = _slice_window(full_returns, VERIFIED_START, VERIFIED_END)
    v_eq = (1.0 + v_rets).cumprod() * PRODUCTION_CONFIG.initial_capital
    v_gross_rets = full_eq.pct_change().reindex(v_rets.index).fillna(0)  # placeholder; we compute net here
    # For the verified-window gross we need a clean recompute:
    # gross = position * bar_return; recompute on the slice.
    close = basket["close"]
    bar_returns = close.pct_change().fillna(0.0)
    v_positions = _slice_window(full_positions, VERIFIED_START, VERIFIED_END)
    v_bar_returns = _slice_window(bar_returns, VERIFIED_START, VERIFIED_END)
    v_gross_returns = v_positions * v_bar_returns
    v_gross_eq = (1.0 + v_gross_returns).cumprod() * PRODUCTION_CONFIG.initial_capital

    v_sharpe = _sharpe(v_rets, ann)
    v_total_return = float(v_eq.iloc[-1] / v_eq.iloc[0] - 1.0) if not v_eq.empty else 0.0
    v_net_cagr = _cagr(v_eq)
    v_gross_cagr = _cagr(v_gross_eq)
    v_cost_drag_bps = (v_gross_cagr - v_net_cagr) * 1e4

    # Decompose fees vs slippage on the FULL sample
    fee_drag_bps = (bt.total_fees / PRODUCTION_CONFIG.initial_capital) * (
        365.0 / len(full_eq)
    ) * 1e4 if len(full_eq) else 0.0
    slip_drag_bps = (bt.total_slippage / PRODUCTION_CONFIG.initial_capital) * (
        365.0 / len(full_eq)
    ) * 1e4 if len(full_eq) else 0.0

    n_flips_full, flips_per_year_full = _annual_flip_count(full_positions)
    n_flips_v, flips_per_year_v = _annual_flip_count(v_positions)

    return {
        "scenario": scenario_name,
        "label": scenario["label"],
        "fee_rate": scenario["fee_rate"],
        "slippage_rate": scenario["slippage_rate"],
        "full_sample": {
            "bars": int(len(full_eq)),
            "first": str(full_eq.index.min().date()),
            "last": str(full_eq.index.max().date()),
            "net_sharpe": float(full_sharpe),
            "total_return_pct": float(full_total_return),
            "net_cagr_pct": float(full_net_cagr),
            "gross_cagr_pct": float(full_gross_cagr),
            "cost_drag_bps_per_year": float(full_cost_drag_bps),
            "fee_drag_bps_per_year": float(fee_drag_bps),
            "slip_drag_bps_per_year": float(slip_drag_bps),
            "n_flips": int(n_flips_full),
            "flips_per_year": float(flips_per_year_full),
        },
        "verified_window": {
            "bars": int(len(v_eq)),
            "first": str(v_eq.index.min().date()) if not v_eq.empty else None,
            "last": str(v_eq.index.max().date()) if not v_eq.empty else None,
            "net_sharpe": float(v_sharpe),
            "total_return_pct": float(v_total_return),
            "net_cagr_pct": float(v_net_cagr),
            "gross_cagr_pct": float(v_gross_cagr),
            "cost_drag_bps_per_year": float(v_cost_drag_bps),
            "n_flips": int(n_flips_v),
            "flips_per_year": float(flips_per_year_v),
        },
    }


def main():
    print(f"Frozen config hash: {CANONICAL_HASH}\n")
    print(f"Verified window: {VERIFIED_START.date()} → {VERIFIED_END.date()}\n")

    basket, _ = load_binance_basket()
    print(f"Basket: {len(basket)} bars, "
          f"{basket.index.min().date()} → {basket.index.max().date()}\n")

    results = []
    for name, scn in COST_SCENARIOS.items():
        r = evaluate(basket, name, scn)
        results.append(r)
        print(f"=== {name} ({r['label']}) ===")
        fs = r["full_sample"]
        vw = r["verified_window"]
        print(f"  full   ({fs['bars']:4} bars  {fs['first']}..{fs['last']}):  "
              f"SR {fs['net_sharpe']:+.3f}  TR {fs['total_return_pct']*100:+8.2f}%  "
              f"drag {fs['cost_drag_bps_per_year']:>5.0f} bps/y  "
              f"(fee {fs['fee_drag_bps_per_year']:>4.0f} + slip {fs['slip_drag_bps_per_year']:>4.0f})  "
              f"flips/y {fs['flips_per_year']:>5.1f}")
        print(f"  verif  ({vw['bars']:4} bars  {vw['first']}..{vw['last']}):  "
              f"SR {vw['net_sharpe']:+.3f}  TR {vw['total_return_pct']*100:+8.2f}%  "
              f"drag {vw['cost_drag_bps_per_year']:>5.0f} bps/y  "
              f"flips/y {vw['flips_per_year']:>5.1f}")
        print()

    # Marginal Binance→Kraken cost-tax on the verified window
    baseline = results[0]["verified_window"]["net_sharpe"]
    print("--- Marginal Binance→Kraken cost-tax (verified window) ---")
    for r in results[1:]:
        v = r["verified_window"]
        delta_sr = v["net_sharpe"] - baseline
        delta_drag = (
            v["cost_drag_bps_per_year"]
            - results[0]["verified_window"]["cost_drag_bps_per_year"]
        )
        print(f"  {r['scenario']:18}: Δ Sharpe {delta_sr:+.3f}  "
              f"Δ cost-drag {delta_drag:+5.0f} bps/y  "
              f"net SR {v['net_sharpe']:+.3f}")

    summary = {
        "frozen_config_hash": CANONICAL_HASH,
        "verified_window": [str(VERIFIED_START.date()), str(VERIFIED_END.date())],
        "scenarios": results,
        "verified_window_marginal_tax": [
            {
                "scenario": r["scenario"],
                "delta_sharpe_vs_binance": r["verified_window"]["net_sharpe"] - baseline,
                "delta_cost_drag_bps_per_year": (
                    r["verified_window"]["cost_drag_bps_per_year"]
                    - results[0]["verified_window"]["cost_drag_bps_per_year"]
                ),
            }
            for r in results[1:]
        ],
    }
    out = OUT_DIR / "validation_test2_execution.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
