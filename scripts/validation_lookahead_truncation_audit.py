"""Validation Test 4 — Part A: retrospective truncation-audit.

Per-bar test: for every T in the verified window, rebuild the basket
index AND recompute the strategy signal on data **truncated to ≤ T**;
compare to the full-sample backtest's signal[T] and basket_close[T].

Rationale
---------
A look-ahead bug lives in the signal/index plumbing, NOT in the
realized P&L (P&L is downstream). The audit is therefore at the
signal layer and is offset-FREE (signal[T] vs signal[T] — same
convention on both sides). The 1-bar offset question lives in the
LIVE detector (Part B), not here.

Scope boundary
--------------
* Catches: temporal look-ahead in basket-index construction
  (normalization, rebalance schedule, weight propagation),
  TimeSeriesMomentumStrategy (SMA-200 gate, TSMOM lookbacks), and
  any per-bar fill/alignment that pulls future bytes into the past.
* Does NOT catch: universe-selection bias (the choice of the 7
  majors). That is a survivorship question, closed by Test 1 Check A
  (CLEAN) and Test 5 (PASS).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trade_lab.backtest.market_index import build_crypto_market_index
from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy

VERIFIED_START = pd.Timestamp("2022-01-21", tz="UTC")
VERIFIED_END = pd.Timestamp("2026-05-28", tz="UTC")
SIG_TOL = 1e-9
IDX_TOL_REL = 1e-9   # relative tolerance for index value


def _load_panel() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in PRODUCTION_CONFIG.assets:
        p = Path(f"data/binance_{sym}_USDT_1d.parquet")
        if not p.exists():
            print(f"FATAL: missing {p}", file=sys.stderr)
            sys.exit(2)
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[sym] = df
    return out


def _build_strategy() -> TimeSeriesMomentumStrategy:
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


def _build_basket(ac: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cfg = PRODUCTION_CONFIG
    return build_crypto_market_index(
        ac,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        rebalance_freq=cfg.basket_rebalance_freq,
    )


def main() -> int:
    print(f"Frozen config hash: {CANONICAL_HASH}")
    print(f"Verified audit window: {VERIFIED_START.date()} → {VERIFIED_END.date()}\n")

    # Full-sample reference: one build + one signal pass
    full_panel = _load_panel()
    full_basket = _build_basket(full_panel)
    strat = _build_strategy()
    full_signal = strat.generate_signals(full_basket)
    full_close = full_basket["close"]

    # Audit bars: every bar in the verified window
    audit_bars = full_basket.index[
        (full_basket.index >= VERIFIED_START) & (full_basket.index <= VERIFIED_END)
    ]
    n_bars = len(audit_bars)
    print(f"Audit bars: {n_bars}\n")

    sig_pit_arr = np.zeros(n_bars, dtype=float)
    sig_full_arr = np.zeros(n_bars, dtype=float)
    idx_pit_arr = np.zeros(n_bars, dtype=float)
    idx_full_arr = np.zeros(n_bars, dtype=float)
    sma_pit_arr = np.zeros(n_bars, dtype=float)
    sma_full_arr = np.zeros(n_bars, dtype=float)
    gate_pit_arr = np.zeros(n_bars, dtype=int)
    gate_full_arr = np.zeros(n_bars, dtype=int)

    cfg = PRODUCTION_CONFIG
    sma_period = cfg.sma_filter_periods[0]
    sma_full_series = full_close.rolling(sma_period).mean()
    gate_full_series = (full_close > sma_full_series).fillna(False).astype(int)

    t0 = time.time()
    for i, T in enumerate(audit_bars):
        truncated = {sym: df[df.index <= T] for sym, df in full_panel.items()}
        # All assets that ARE listed at T must have data; pre-listing
        # assets retain empty DataFrames, which the basket builder
        # treats as inactive — same as full.
        pit_basket = _build_basket(truncated)
        pit_signal = strat.generate_signals(pit_basket)
        pit_close = pit_basket["close"]

        sig_pit_arr[i] = float(pit_signal.iloc[-1])
        sig_full_arr[i] = float(full_signal.loc[T])
        idx_pit_arr[i] = float(pit_close.iloc[-1])
        idx_full_arr[i] = float(full_close.loc[T])

        pit_sma = pit_close.rolling(sma_period).mean()
        sma_pit_arr[i] = float(pit_sma.iloc[-1]) if pd.notna(pit_sma.iloc[-1]) else np.nan
        sma_full_arr[i] = float(sma_full_series.loc[T]) if pd.notna(sma_full_series.loc[T]) else np.nan
        gate_pit_arr[i] = int(pit_close.iloc[-1] > pit_sma.iloc[-1]) if pd.notna(pit_sma.iloc[-1]) else 0
        gate_full_arr[i] = int(gate_full_series.loc[T])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  progress {i+1}/{n_bars}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n")

    # Per-bar comparisons
    sig_mismatch = np.abs(sig_pit_arr - sig_full_arr) > SIG_TOL
    idx_abs_diff = np.abs(idx_pit_arr - idx_full_arr)
    idx_rel_diff = idx_abs_diff / np.maximum(np.abs(idx_full_arr), 1e-12)
    idx_mismatch = idx_rel_diff > IDX_TOL_REL
    sma_abs_diff = np.where(
        np.isnan(sma_pit_arr) | np.isnan(sma_full_arr),
        0.0,
        np.abs(sma_pit_arr - sma_full_arr),
    )
    sma_rel_diff = sma_abs_diff / np.maximum(np.abs(sma_full_arr), 1e-12)
    sma_mismatch = sma_rel_diff > IDX_TOL_REL
    gate_mismatch = gate_pit_arr != gate_full_arr

    print("=== Per-bar audit results ===")
    print(f"  Signal mismatches      : {int(sig_mismatch.sum())} / {n_bars}")
    print(f"  Index-close mismatches : {int(idx_mismatch.sum())} / {n_bars}")
    print(f"  SMA(200) mismatches    : {int(sma_mismatch.sum())} / {n_bars}")
    print(f"  Gate flip mismatches   : {int(gate_mismatch.sum())} / {n_bars}\n")

    # Diagnostic detail on the maximum-diff bars
    print("=== Magnitude diagnostics ===")
    print(f"  max |Δsignal|     : {float(np.max(np.abs(sig_pit_arr - sig_full_arr))):.2e}")
    print(f"  max |Δindex|/|idx|: {float(np.max(idx_rel_diff)):.2e}")
    print(f"  max |ΔSMA|/|SMA|  : {float(np.max(sma_rel_diff)):.2e}\n")

    # Final verdict
    total_mismatch = int(sig_mismatch.sum() + idx_mismatch.sum() + sma_mismatch.sum() + gate_mismatch.sum())
    verdict = "CLEAN" if total_mismatch == 0 else "CONTAMINATED"
    print(f"=== Verdict: {verdict} ===\n")

    if total_mismatch:
        print("--- First 10 mismatches ---")
        for i in range(n_bars):
            if sig_mismatch[i] or idx_mismatch[i] or sma_mismatch[i] or gate_mismatch[i]:
                print(
                    f"  {audit_bars[i].date()}  "
                    f"sig pit={sig_pit_arr[i]:.6f} full={sig_full_arr[i]:.6f}  "
                    f"idx pit={idx_pit_arr[i]:.6f} full={idx_full_arr[i]:.6f}  "
                    f"sma pit={sma_pit_arr[i]:.6f} full={sma_full_arr[i]:.6f}  "
                    f"gate pit={gate_pit_arr[i]} full={gate_full_arr[i]}"
                )

    # Persist machine-readable output
    summary = {
        "frozen_config_hash": CANONICAL_HASH,
        "window": [str(VERIFIED_START.date()), str(VERIFIED_END.date())],
        "n_bars_audited": n_bars,
        "tolerances": {"signal_abs": SIG_TOL, "index_rel": IDX_TOL_REL,
                       "sma_rel": IDX_TOL_REL},
        "mismatches": {
            "signal": int(sig_mismatch.sum()),
            "index_close": int(idx_mismatch.sum()),
            "sma200": int(sma_mismatch.sum()),
            "regime_gate_flip": int(gate_mismatch.sum()),
        },
        "max_diff": {
            "signal_abs": float(np.max(np.abs(sig_pit_arr - sig_full_arr))),
            "index_rel": float(np.max(idx_rel_diff)),
            "sma_rel": float(np.max(sma_rel_diff)),
        },
        "verdict": verdict,
        "runtime_seconds": elapsed,
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/validation_test4_truncation_audit.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("\nWrote outputs/validation_test4_truncation_audit.json")
    return 0 if verdict == "CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
