"""Validation Test 1: replay the FROZEN production config on Kraken + Bybit
and compare against the Binance baseline.

Goal of the test (per validation prompt): detect whether the TSMOM(28,60)
+ SMA200 edge is an artifact of one venue's price construction. The
hypothesis to falsify: "the strategy works on Binance but disagrees
materially with the same strategy run on Kraken / Bybit prices."

Outputs
-------
* ``outputs/validation_test1_multiexchange.json`` — machine-readable
  metrics (per-venue equity + Sharpe; pairwise / 3-way signal agreement;
  per-asset price-return correlation).
* ``findings/validation_multiexchange.md`` — the human writeup with
  the verdict.

Key methodological points
-------------------------
* Kraken REST hard-caps public OHLCV at the last 720 daily candles,
  so the 3-way overlap window is fundamentally short (~24 months).
  Bybit + Binance overlap goes back to ~2021-07.
* The strategy needs ~200 days of basket warmup (SMA200). On Kraken
  that means usable signals only after ~2025-01.
* "Signal agreement" is at the basket level (the strategy IS a single
  long/cash decision on the synthetic index). Per-asset return
  correlation is reported as a diagnostic, not as the verdict input.
"""
from __future__ import annotations

import json
from dataclasses import asdict
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


def load_asset_candles(exchange_id: str) -> dict[str, pd.DataFrame]:
    """Load per-asset OHLCV for the canonical 7-major basket from cache.

    Missing pairs are dropped silently (e.g. Bybit BNB starts later)
    and reported separately so the diagnostic can flag them.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in PRODUCTION_CONFIG.assets:
        p = DATA_DIR / f"{exchange_id}_{sym}_USDT_1d.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[sym] = df
    return out


def run_pipeline(asset_candles: dict[str, pd.DataFrame]) -> dict:
    """Build the equal-weight basket and run the FROZEN strategy.

    Returns the full output bundle for the venue (signals, weights,
    equity, sharpe, etc.). Cost rates and lookbacks come from
    ``PRODUCTION_CONFIG`` — no overrides.
    """
    cfg = PRODUCTION_CONFIG
    basket = build_crypto_market_index(
        asset_candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        rebalance_freq=cfg.basket_rebalance_freq,
    )
    strategy = TimeSeriesMomentumStrategy(
        lookbacks=cfg.lookbacks,
        sma_filter_periods=cfg.sma_filter_periods,
        use_vol_target=cfg.use_vol_target,
        vol_lookback=cfg.vol_lookback,
        annual_vol_target=cfg.annual_vol_target,
        max_position_size=cfg.max_position_size,
        rebalance_threshold=cfg.rebalance_threshold,
        annualization_factor=cfg.annualization_factor,
    )

    signal_series = strategy.generate_signals(basket)

    # Diagnostic: per-component states at every bar (mirrors the
    # strategy's internal _tsmom_ensemble and _sma_filter without
    # re-fitting). Used for fine-grained agreement attribution.
    close = basket["close"]
    sma200 = close.rolling(cfg.sma_filter_periods[0]).mean()
    regime_open = (close > sma200).astype(float)
    regime_open[sma200.isna()] = 0.0
    tsmom_states = {}
    for L in cfg.lookbacks:
        past_ret = close.pct_change(L)
        st = (past_ret > 0).astype(float)
        st[past_ret.isna()] = 0.0
        tsmom_states[L] = st

    backtest = run_backtest(
        basket,
        strategy,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        position_size=cfg.max_position_size,
    )

    return {
        "basket_close": basket["close"],
        "signal": signal_series,
        "regime_open": regime_open,
        "tsmom_states": tsmom_states,
        "equity": backtest.equity,
        "returns": backtest.returns,
        "sharpe": backtest.equity,
    }


def _net_sharpe(returns: pd.Series, ann: int) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return _sharpe(returns, annualization_factor=ann)


def per_venue_summary(name: str, bundle: dict) -> dict:
    """Compact per-venue summary for the JSON output."""
    eq = bundle["equity"].dropna()
    ret = bundle["returns"].dropna()
    sr = _net_sharpe(ret, PRODUCTION_CONFIG.annualization_factor)
    if eq.empty:
        return {"venue": name, "rows": 0}
    return {
        "venue": name,
        "first_bar": str(eq.index.min().date()),
        "last_bar": str(eq.index.max().date()),
        "rows": int(len(eq)),
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "net_sharpe": float(sr),
        "mean_signal": float(bundle["signal"].mean()),
        "frac_long_full": float((bundle["signal"] >= 0.99).mean()),
        "frac_flat_full": float((bundle["signal"] <= 0.01).mean()),
    }


def signal_agreement(bundles: dict[str, dict], cfg) -> dict:
    """For each pair (and the 3-way intersection) compute the % of
    overlapping bars where the venue signals exactly agree.
    """
    venues = list(bundles.keys())
    sigs = {v: bundles[v]["signal"] for v in venues}
    regs = {v: bundles[v]["regime_open"] for v in venues}
    ts28 = {v: bundles[v]["tsmom_states"][cfg.lookbacks[0]] for v in venues}
    ts60 = {v: bundles[v]["tsmom_states"][cfg.lookbacks[1]] for v in venues}

    def _pair(a, b, sa, sb):
        common = sa.index.intersection(sb.index)
        if common.empty:
            return {"n_bars": 0}
        # Only measure agreement once both venues are past their SMA200
        # warmup — otherwise we're just comparing forced-zero signals.
        regs_a = bundles[a]["regime_open"].reindex(common)
        regs_b = bundles[b]["regime_open"].reindex(common)
        valid = (~regs_a.isna()) & (~regs_b.isna())
        # SMA200 warmup proxy: need at least 200 prior bars in EACH series.
        warm = pd.Series(False, index=common)
        for v, s in [(a, sa), (b, sb)]:
            ranks = pd.Series(range(len(bundles[v]["basket_close"])),
                              index=bundles[v]["basket_close"].index)
            ranks_at_common = ranks.reindex(common)
            warm = warm | (ranks_at_common < 200)
        usable = valid & (~warm.fillna(True))
        sub_a = sa.reindex(common)[usable]
        sub_b = sb.reindex(common)[usable]
        if sub_a.empty:
            return {"n_bars": 0}
        eq = (sub_a == sub_b).mean()
        return {
            "n_bars": int(usable.sum()),
            "window": (str(common[usable].min().date()),
                       str(common[usable].max().date())),
            "final_signal_agreement": float(eq),
            "regime_gate_agreement": float(
                (regs[a].reindex(common)[usable] == regs[b].reindex(common)[usable]).mean()
            ),
            "tsmom28_agreement": float(
                (ts28[a].reindex(common)[usable] == ts28[b].reindex(common)[usable]).mean()
            ),
            "tsmom60_agreement": float(
                (ts60[a].reindex(common)[usable] == ts60[b].reindex(common)[usable]).mean()
            ),
        }

    pairs = {}
    for i, a in enumerate(venues):
        for b in venues[i + 1:]:
            pairs[f"{a}_vs_{b}"] = _pair(a, b, sigs[a], sigs[b])

    # 3-way: simultaneous agreement
    if len(venues) >= 3:
        common = sigs[venues[0]].index
        for v in venues[1:]:
            common = common.intersection(sigs[v].index)
        # 200-bar warmup proxy in each series
        warmup_mask = pd.Series(True, index=common)
        for v in venues:
            ranks = pd.Series(range(len(bundles[v]["basket_close"])),
                              index=bundles[v]["basket_close"].index)
            warmup_mask = warmup_mask & (ranks.reindex(common) >= 200)
        if warmup_mask.any():
            sub = pd.concat([sigs[v].reindex(common) for v in venues], axis=1)
            sub = sub[warmup_mask.fillna(False)]
            eq3 = (sub.nunique(axis=1) == 1).mean()
            pairs["all3_agreement"] = {
                "n_bars": int(warmup_mask.sum()),
                "window": (str(common[warmup_mask].min().date()),
                           str(common[warmup_mask].max().date())),
                "final_signal_agreement_3way": float(eq3),
            }
        else:
            pairs["all3_agreement"] = {"n_bars": 0}

    return pairs


def per_asset_return_corr(
    bundles: dict[str, dict], asset_candles_by_venue: dict[str, dict]
) -> dict:
    """Pairwise daily-return correlation per asset, on common bars.

    This is a venue-pricing diagnostic, independent of the strategy:
    if BTC pct_change(1) differs between Binance and Kraken on a given
    day, that's the raw data layer disagreeing.
    """
    venues = list(asset_candles_by_venue.keys())
    out: dict[str, dict] = {}
    for sym in PRODUCTION_CONFIG.assets:
        per_pair = {}
        rets = {}
        for v in venues:
            ac = asset_candles_by_venue[v].get(sym)
            if ac is None or ac.empty:
                continue
            rets[v] = ac["close"].pct_change().dropna()
        # All pairwise
        keys = list(rets.keys())
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                common = rets[a].index.intersection(rets[b].index)
                if len(common) < 30:
                    per_pair[f"{a}_vs_{b}"] = None
                    continue
                corr = float(rets[a].reindex(common).corr(rets[b].reindex(common)))
                per_pair[f"{a}_vs_{b}"] = {"n_bars": int(len(common)), "corr": corr}
        out[sym] = per_pair
    return out


def main():
    venues = ("binance", "kraken", "bybit")
    asset_candles_by_venue: dict[str, dict] = {}
    bundles: dict[str, dict] = {}

    print(f"Frozen config hash: {CANONICAL_HASH}\n")

    for v in venues:
        ac = load_asset_candles(v)
        asset_candles_by_venue[v] = ac
        if not ac:
            print(f"=== {v}: no parquets ===")
            continue
        present = sorted(ac.keys())
        print(f"=== {v}: assets present = {present} ===")
        for sym, df in ac.items():
            print(f"  {sym:6}  rows={len(df)}  range={df.index.min().date()}..{df.index.max().date()}")
        bundle = run_pipeline(ac)
        bundles[v] = bundle
        sm = per_venue_summary(v, bundle)
        print(f"  -> {sm}\n")

    # Build agreement metrics on usable bars
    agreement = signal_agreement(bundles, PRODUCTION_CONFIG)

    # Per-asset return correlation (data-layer diagnostic)
    corrs = per_asset_return_corr(bundles, asset_candles_by_venue)

    # Same-window comparison (every venue's SMA200 warm, common dates only):
    # this is the apples-to-apples Sharpe comparison the verdict relies on.
    warmup_start = {}
    for v, b in bundles.items():
        bc = b["basket_close"]
        warmup_start[v] = bc.index[200] if len(bc) > 200 else bc.index[-1]
    common_start = max(warmup_start.values())
    common_end = min(b["equity"].index.max() for b in bundles.values())
    same_window = {}
    for v, b in bundles.items():
        rets = b["returns"]
        rets = rets[(rets.index >= common_start) & (rets.index <= common_end)]
        sigs = b["signal"]
        sigs = sigs[(sigs.index >= common_start) & (sigs.index <= common_end)]
        if rets.empty:
            same_window[v] = {"bars": 0}
            continue
        eq = (1.0 + rets).cumprod() * PRODUCTION_CONFIG.initial_capital
        same_window[v] = {
            "bars": int(len(rets)),
            "first": str(rets.index.min().date()),
            "last": str(rets.index.max().date()),
            "mean_signal": float(sigs.mean()),
            "frac_full_long": float((sigs >= 0.99).mean()),
            "total_return_pct": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
            "net_sharpe": float(_net_sharpe(rets, PRODUCTION_CONFIG.annualization_factor)),
        }

    # Disagreement detail (Binance vs Kraken, post-warmup):
    disagree_detail = []
    if "binance" in bundles and "kraken" in bundles:
        bn = bundles["binance"]["signal"]
        kr = bundles["kraken"]["signal"]
        common = bn.index.intersection(kr.index)
        common = common[(common >= common_start) & (common <= common_end)]
        bn_sub = bn.reindex(common)
        kr_sub = kr.reindex(common)
        for ts in common[bn_sub != kr_sub]:
            disagree_detail.append({
                "date": str(ts.date()),
                "binance": float(bn_sub.loc[ts]),
                "kraken": float(kr_sub.loc[ts]),
            })

    # Materialize JSON output
    summary = {
        "frozen_config_hash": CANONICAL_HASH,
        "per_venue_full_history": {v: per_venue_summary(v, b) for v, b in bundles.items()},
        "same_window_comparison": {
            "window": (str(common_start.date()), str(common_end.date())),
            "venues": same_window,
        },
        "signal_agreement": agreement,
        "per_asset_return_correlation": corrs,
        "venue_basket_membership": {
            v: sorted(asset_candles_by_venue[v].keys()) for v in venues
        },
        "binance_vs_kraken_disagreement_bars": disagree_detail,
    }
    out_path = OUT_DIR / "validation_test1_multiexchange.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Print key numbers to stdout
    print("\n--- Signal agreement (post-warmup) ---")
    for k, v in agreement.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
