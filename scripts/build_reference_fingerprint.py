"""Build the reference behavioral fingerprint from the Binance backtest.

ONE-TIME script. After this writes
``paper_trading/fingerprint/reference_fingerprint.json``, the file is
the **versioned frozen artifact**. Re-running this script must produce
a byte-identical file (modulo timestamp): the percentile bands are
deterministic given the frozen config + Binance parquets. If the
content-hash changes, an input changed — investigate.

Pipeline mirrors the harness exactly: same basket construction, same
strategy class, same Binance parquets. A diff between the reference
and the harness's signal-generation path would be a calibration bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from trade_lab.backtest.engine import run_backtest
from trade_lab.backtest.market_index import build_crypto_market_index
from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG
from trade_lab.paper_trading.fingerprint import (
    compute_reference_fingerprint,
    fingerprint_content_hash,
    load_reference,
    save_reference,
)
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy

VERIFIED_START = pd.Timestamp("2022-01-21", tz="UTC")
VERIFIED_END = pd.Timestamp("2026-05-28", tz="UTC")
OUT_PATH = Path("paper_trading/fingerprint/reference_fingerprint.json")


def main() -> int:
    cfg = PRODUCTION_CONFIG
    asset_candles: dict[str, pd.DataFrame] = {}
    for sym in cfg.assets:
        p = Path(f"data/binance_{sym}_USDT_1d.parquet")
        if not p.exists():
            print(f"FATAL: missing {p}; re-fetch the Binance parquets first.",
                  file=sys.stderr)
            return 2
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        asset_candles[sym] = df

    basket = build_crypto_market_index(
        asset_candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        rebalance_freq=cfg.basket_rebalance_freq,
    )
    strat = TimeSeriesMomentumStrategy(
        lookbacks=cfg.lookbacks,
        sma_filter_periods=cfg.sma_filter_periods,
        use_vol_target=cfg.use_vol_target,
        annualization_factor=cfg.annualization_factor,
    )
    bt = run_backtest(
        basket, strat,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate,
    )
    sma_series = basket["close"].rolling(cfg.sma_filter_periods[0]).mean()

    fp = compute_reference_fingerprint(
        basket_close=basket["close"],
        positions=bt.positions,
        equity=bt.equity,
        sma_series=sma_series,
        window_start=VERIFIED_START,
        window_end=VERIFIED_END,
        frozen_config_hash=CANONICAL_HASH,
    )

    save_reference(fp, OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"content_hash = {fp.content_hash}")
    print(f"frozen_config_hash = {fp.frozen_config_hash}")
    print(f"window = {fp.window_start} → {fp.window_end}  ({fp.n_bars} bars)\n")

    # Round-trip verify
    loaded = load_reference(OUT_PATH)
    expected = fingerprint_content_hash(loaded)
    if loaded.content_hash != expected:
        print(f"FATAL: round-trip hash mismatch on save→load.", file=sys.stderr)
        return 2
    print("Round-trip hash verification OK.\n")

    # Summary print
    def _summary(name, band):
        print(f"  {name}:")
        for k in sorted(band.percentiles.keys()):
            print(f"    {k} = {band.percentiles[k]:.4f}")
        print(f"    min={band.extremes['min']:.4f}  "
              f"max={band.extremes['max']:.4f}  "
              f"mean={band.extremes['mean']:.4f}")
        print()
    _summary(fp.exposure_flip_freq_rolling.name, fp.exposure_flip_freq_rolling)
    _summary(fp.regime_gate_flip_freq_rolling.name, fp.regime_gate_flip_freq_rolling)
    _summary(fp.rebalance_turnover_per_event.name, fp.rebalance_turnover_per_event)
    dd = fp.drawdown_profile
    print(f"  drawdown_profile:")
    for k in sorted(dd.percentiles.keys()):
        print(f"    {k} = {dd.percentiles[k]*100:.2f}%")
    print(f"    max_historical_dd = {dd.max_historical_dd*100:.2f}%  (breach if live DD goes below this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
