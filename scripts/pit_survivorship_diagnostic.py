"""PIT survivorship diagnostic for the deployed 7-major basket (issue #14, prep).

Counterfactual: basket = top-N by PIT market cap + trailing 90d median
volume at each monthly rebalance (``build_pit_universe`` recipe with
``top_n = len(static basket)`` = 7), fed through the SAME frozen
walk-forward as the deploy config (TSMOM(28, 60), SMA(200) gate, ladder,
24/6/6 months — the published reproduce block of
``findings/han_28d_tsmom.md``). A static-7 control is run on the SAME
Coin Metrics price panel so the reported delta isolates basket
*composition* from data-source effects.

Diagnostic re-run of the frozen config — NOT a new parametric search:
``PROJECT_NUM_TRIALS`` stays 500; no new variant is evaluated, the single
grid entry is the deployed configuration.

Invocation (post-merge step; the real-data run does NOT happen in the
prep branch)::

    python scripts/pit_survivorship_diagnostic.py \
        --cache-dir data/coinmetrics --out-dir docs/results \
        [--start 2018-01-01] [--fetch] [--allow-missing SYMBOL ...]

Expected runtime: minutes on a warm Coin Metrics cache; add roughly an
hour+ if ``--fetch`` must pull the ~40-asset panel first (courtesy-paced
community API).

Known first failure modes — deliberate, fail loud, never shrink silently:

* A registry coin with no Coin Metrics coverage at all aborts the run;
  each such gap must be acknowledged explicitly with ``--allow-missing``
  and is recorded in the report as a first-class exclusion.
* A tradable candidate with NaN market cap on any rebalance date aborts
  via ``PITMcapGapError`` (e.g. BNB's community-tier cap gap,
  ``findings/validation_universe_bias.md``). Static-basket members cannot
  be excluded — fix the data instead.

The DEPLOYED strategy's basket does not change based on this
diagnostic's outcome without a new owner decision; the results feed a
NEW finding written after the post-merge run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trade_lab.backtest import (
    PROJECT_NUM_TRIALS,
    ParamGridSpec,
    aggregate_walk_forward,
    run_strategy_walk_forward,
)
from trade_lab.backtest.metrics import max_drawdown
from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG
from trade_lab.data.coin_registry import COIN_REGISTRY, tradable_at
from trade_lab.data.universe import (
    build_pit_universe,
    closes_for_universe,
    load_panel,
)
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy

# Published deploy walk-forward split (findings/han_28d_tsmom.md,
# "Reproducing"). Frozen here on purpose — changing it would make the
# diagnostic incomparable to the shipped DSR 0.77 run.
WF_TRAIN_MONTHS = 24
WF_TEST_MONTHS = 6
WF_STEP_MONTHS = 6

# The published PIT recipe's volume screen (docs/results/pit_universe.md).
VOLUME_LOOKBACK_DAYS = 90

REPORT_STEM = "pit_survivorship_diagnostic"


# ---------------------------------------------------------------------------
# Basket selection
# ---------------------------------------------------------------------------


def monthly_rebalance_bars(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First bar + every month start snapped to the next available bar
    (same snapping as ``build_crypto_market_index``)."""
    if len(idx) == 0:
        raise SystemExit("empty panel index after --start/--end slicing")
    schedule = pd.date_range(
        idx[0], idx[-1], freq=PRODUCTION_CONFIG.basket_rebalance_freq, tz=idx.tz
    )
    bars = {idx[0]}
    for date in schedule:
        pos = idx.searchsorted(date)
        if pos < len(idx):
            bars.add(idx[pos])
    return sorted(bars)


def select_pit_baskets(
    eligibility: pd.DataFrame, bars: list[pd.Timestamp]
) -> dict[pd.Timestamp, tuple[str, ...]]:
    baskets: dict[pd.Timestamp, tuple[str, ...]] = {}
    for ts in bars:
        members = tuple(sorted(eligibility.columns[eligibility.loc[ts].astype(bool)]))
        if not members:
            raise SystemExit(f"empty PIT basket at {ts.date()} — no eligible coin")
        baskets[ts] = members
    return baskets


def select_static_baskets(
    closes: pd.DataFrame,
    bars: list[pd.Timestamp],
    static_basket: tuple[str, ...],
    pool: dict,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Static members gated by Binance listing + an observed close, so the
    control enters assets the way the deployed basket does (dynamic entry)."""
    baskets: dict[pd.Timestamp, tuple[str, ...]] = {}
    for ts in bars:
        members = tuple(
            sorted(
                sym
                for sym in static_basket
                if tradable_at(ts.strftime("%Y-%m-%d"), pool[sym])
                and pd.notna(closes.at[ts, sym])
            )
        )
        if not members:
            raise SystemExit(f"empty static control basket at {ts.date()}")
        baskets[ts] = members
    return baskets


# ---------------------------------------------------------------------------
# Membership-aware basket index
# ---------------------------------------------------------------------------


def build_membership_basket_index(
    closes: pd.DataFrame,
    baskets: dict[pd.Timestamp, tuple[str, ...]],
    *,
    fee_rate: float,
    slippage_rate: float,
    initial_capital: float,
) -> pd.DataFrame:
    """Equal-weight basket index with membership fixed between rebalances.

    Mirrors ``build_crypto_market_index_with_weights`` bar-for-bar
    (drift + renormalise between rebalances, ``fee+slippage`` per unit
    |weight change| on rebalance bars, initial deployment not charged)
    with two deliberate differences that time-varying membership forces:

    * membership changes only on the given rebalance bars, never on
      mid-month listings;
    * later membership *entries* are charged in full — entering the
      basket is a real trade, unlike the deployed builder's
      new-listing credit.

    With constant membership and all assets listed from bar 0 the output
    is identical to ``build_crypto_market_index`` (unit-tested).
    """
    closes = closes.sort_index()
    returns = closes.pct_change(fill_method=None)
    cols = list(closes.columns)
    col_pos = {c: i for i, c in enumerate(cols)}
    bars = list(closes.index)
    returns_arr = returns.to_numpy()

    weights = np.zeros((len(bars), len(cols)))
    cost_per_bar = np.zeros(len(bars))
    current = np.zeros(len(cols))
    first_bar = True
    for i, ts in enumerate(bars):
        if ts in baskets:
            members = baskets[ts]
            missing = [m for m in members if pd.isna(closes.at[ts, m])]
            if missing:
                raise SystemExit(
                    f"missing close for basket member(s) {missing} at "
                    f"{ts.date()} — refusing to size a silently shrunken basket"
                )
            target = np.zeros(len(cols))
            for m in members:
                target[col_pos[m]] = 1.0 / len(members)
            turnover = float(np.abs(target - current).sum())
            charged = 0.0 if first_bar else turnover
            cost_per_bar[i] = charged * (fee_rate + slippage_rate)
            current = target
            first_bar = False
        else:
            r = returns_arr[i]
            held = current > 0
            if np.isnan(r[held]).any():
                gone = [cols[j] for j in np.where(held & np.isnan(r))[0]]
                raise SystemExit(
                    f"missing close for held member(s) {gone} at {ts.date()} "
                    "— refusing to drift a silently shrunken basket"
                )
            grown = current * (1.0 + np.where(np.isnan(r), 0.0, r))
            total = grown.sum()
            current = grown / total if total > 0 else current
        weights[i] = current

    weights_df = pd.DataFrame(weights, index=closes.index, columns=cols)
    shifted = weights_df.shift(1).fillna(0.0)
    portfolio_returns = (shifted * returns.fillna(0.0)).sum(axis=1) - cost_per_bar
    equity = initial_capital * (1.0 + portfolio_returns).cumprod()
    index_close = 100.0 * equity / equity.iloc[0]
    out = pd.DataFrame(
        {
            "open": index_close,
            "high": index_close,
            "low": index_close,
            "close": index_close,
            "volume": 1.0,
        },
        index=closes.index,
    )
    out.index.name = "timestamp"
    return out


# ---------------------------------------------------------------------------
# Walk-forward (frozen deploy config)
# ---------------------------------------------------------------------------


def run_deploy_walk_forward(index_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The deploy config's walk-forward, single grid entry — no search."""
    grid = [
        ParamGridSpec(
            label="tsmom_28_60_sma200",
            factory=lambda: TimeSeriesMomentumStrategy(
                lookbacks=PRODUCTION_CONFIG.lookbacks,
                sma_filter_periods=PRODUCTION_CONFIG.sma_filter_periods,
                use_vol_target=PRODUCTION_CONFIG.use_vol_target,
            ),
            warmup_days=PRODUCTION_CONFIG.warmup_days,
        )
    ]
    detail, oos = run_strategy_walk_forward(
        index_df,
        grid,
        train_months=WF_TRAIN_MONTHS,
        test_months=WF_TEST_MONTHS,
        step_months=WF_STEP_MONTHS,
        annualization_factor=PRODUCTION_CONFIG.annualization_factor,
        initial_capital=PRODUCTION_CONFIG.initial_capital,
        fee_rate=PRODUCTION_CONFIG.fee_rate,
        slippage_rate=PRODUCTION_CONFIG.slippage_rate,
        return_oos_returns=True,
    )
    summary = aggregate_walk_forward(
        detail, oos_returns=oos, num_trials=PROJECT_NUM_TRIALS
    )
    valid = [s for s in oos if s is not None and len(s) > 0]
    if valid:
        concat = pd.concat(valid).sort_index()
        concat = concat[~concat.index.duplicated(keep="first")]
        summary["concatenated_oos_max_drawdown_pct"] = max_drawdown(
            (1.0 + concat).cumprod()
        )
    else:
        summary["concatenated_oos_max_drawdown_pct"] = 0.0
    return detail, summary


# ---------------------------------------------------------------------------
# Panel loading (fail loud on coverage gaps)
# ---------------------------------------------------------------------------


def load_diagnostic_panel(
    cache_dir: Path | str,
    *,
    allow_missing: list[str],
    static_basket: tuple[str, ...],
    fetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, str]]:
    """Returns ``(prices, market_caps, volumes, pool, excluded)``."""
    pool = dict(COIN_REGISTRY)
    excluded: dict[str, str] = {}
    for sym in allow_missing:
        if sym not in pool:
            raise SystemExit(f"--allow-missing {sym}: not in COIN_REGISTRY")
        if sym in static_basket:
            raise SystemExit(
                f"--allow-missing {sym}: static-basket member — excluding it "
                "would distort both the PIT and the control run; fix the data"
            )
        excluded[sym] = "operator-acknowledged missing Coin Metrics coverage"
        del pool[sym]

    prices, market_caps, volumes = load_panel(pool, cache_dir=cache_dir, fetch=fetch)
    missing = sorted(set(pool) - set(market_caps.columns))
    if missing:
        raise SystemExit(
            f"no Coin Metrics coverage for {missing} — acknowledge each gap "
            "explicitly with --allow-missing SYMBOL (recorded in the report) "
            "or fix the cache; the universe is never shrunk silently"
        )
    return prices, market_caps, volumes, pool, excluded


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def run_diagnostic(
    prices: pd.DataFrame,
    market_caps: pd.DataFrame,
    volumes: pd.DataFrame,
    pool: dict,
    *,
    out_dir: Path | str,
    start: str | None = None,
    end: str | None = None,
    excluded: dict[str, str] | None = None,
    # Testability seam only — the CLI never overrides the frozen basket.
    static_basket: tuple[str, ...] = PRODUCTION_CONFIG.assets,
) -> dict:
    excluded = excluded or {}
    idx = market_caps.index
    if start:
        idx = idx[idx >= pd.Timestamp(start, tz=idx.tz)]
    if end:
        idx = idx[idx <= pd.Timestamp(end, tz=idx.tz)]
    bars = monthly_rebalance_bars(idx)

    # Same recipe as the deployed basket size: top-7 = len(static basket).
    eligibility = build_pit_universe(
        market_caps,
        volumes,
        candidates=pool,
        top_n=len(static_basket),
        volume_lookback_days=VOLUME_LOOKBACK_DAYS,
        exclude_stablecoins=True,
        start_date=start,
        end_date=end,
        strict_mcap_dates=bars,  # fail loud on NaN mcap at any rebalance
    )

    # Unmasked closes (with the documented mcap fallback for price-gated
    # coins): membership persists between rebalances even if eligibility
    # flips mid-month, so we must NOT mask closes to the eligibility grid.
    all_true = pd.DataFrame(
        True, index=eligibility.index, columns=eligibility.columns
    )
    closes = closes_for_universe(prices, all_true, fallback=market_caps)

    pit_baskets = select_pit_baskets(eligibility, bars)
    static_baskets = select_static_baskets(closes, bars, static_basket, pool)

    cost_kwargs = dict(
        fee_rate=PRODUCTION_CONFIG.fee_rate,
        slippage_rate=PRODUCTION_CONFIG.slippage_rate,
        initial_capital=PRODUCTION_CONFIG.initial_capital,
    )
    pit_index = build_membership_basket_index(closes, pit_baskets, **cost_kwargs)
    control_index = build_membership_basket_index(
        closes, static_baskets, **cost_kwargs
    )

    pit_detail, pit_summary = run_deploy_walk_forward(pit_index)
    ctl_detail, ctl_summary = run_deploy_walk_forward(control_index)

    payload = _build_payload(
        bars=bars,
        pit_baskets=pit_baskets,
        static_basket=static_basket,
        pit_detail=pit_detail,
        pit_summary=pit_summary,
        ctl_detail=ctl_detail,
        ctl_summary=ctl_summary,
        excluded=excluded,
        start=start,
        end=end,
    )
    _write_report(Path(out_dir), payload)
    return payload


def _fold_rows(detail: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in detail.iterrows():
        rows.append(
            {
                "train_start": str(pd.Timestamp(row["train_start"]).date()),
                "train_end": str(pd.Timestamp(row["train_end"]).date()),
                "test_start": str(pd.Timestamp(row["test_start"]).date()),
                "test_end": str(pd.Timestamp(row["test_end"]).date()),
                "test_sharpe": float(row["test_sharpe"]),
                "test_return_pct": float(row["test_return_pct"]),
                "test_max_drawdown_pct": float(row["test_max_drawdown_pct"]),
                "test_bars": int(row["test_bars"]),
            }
        )
    return rows


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _build_payload(
    *,
    bars: list[pd.Timestamp],
    pit_baskets: dict[pd.Timestamp, tuple[str, ...]],
    static_basket: tuple[str, ...],
    pit_detail: pd.DataFrame,
    pit_summary: dict,
    ctl_detail: pd.DataFrame,
    ctl_summary: dict,
    excluded: dict[str, str],
    start: str | None,
    end: str | None,
) -> dict:
    static_set = set(static_basket)
    rebalances = []
    prev: set[str] = set()
    for ts in bars:
        members = set(pit_baskets[ts])
        rebalances.append(
            {
                "date": str(ts.date()),
                "members": sorted(members),
                "n_members": len(members),
                "added": sorted(members - prev),
                "removed": sorted(prev - members),
                "missing_vs_static": sorted(static_set - members),
                "extra_vs_static": sorted(members - static_set),
                "deviates_from_static": members != static_set,
            }
        )
        prev = members

    delta_keys = (
        "concatenated_oos_sharpe",
        "concatenated_oos_dsr",
        "concatenated_oos_max_drawdown_pct",
        "mean_test_sharpe",
        "hit_rate",
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "production_config_hash": CANONICAL_HASH,
        "num_trials": PROJECT_NUM_TRIALS,
        "walk_forward": {
            "train_months": WF_TRAIN_MONTHS,
            "test_months": WF_TEST_MONTHS,
            "step_months": WF_STEP_MONTHS,
            "lookbacks": list(PRODUCTION_CONFIG.lookbacks),
            "sma_filter_periods": list(PRODUCTION_CONFIG.sma_filter_periods),
            "fee_rate": PRODUCTION_CONFIG.fee_rate,
            "slippage_rate": PRODUCTION_CONFIG.slippage_rate,
        },
        "window": {"start": start, "end": end},
        "static_basket": list(static_basket),
        "excluded_assets": excluded,
        "rebalances": rebalances,
        "runs": {
            "pit_top_n": {"folds": _fold_rows(pit_detail), "summary": pit_summary},
            "static_control": {
                "folds": _fold_rows(ctl_detail),
                "summary": ctl_summary,
            },
        },
        "delta_pit_minus_control": {
            k: float(pit_summary.get(k, 0.0)) - float(ctl_summary.get(k, 0.0))
            for k in delta_keys
        },
    }


def _write_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{REPORT_STEM}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines: list[str] = []
    lines.append("# PIT survivorship diagnostic — top-N-by-PIT-mcap+volume basket")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']} @ commit `{payload['commit']}`")
    lines.append(f"Production config hash: `{payload['production_config_hash']}`")
    wf = payload["walk_forward"]
    lines.append(
        f"Walk-forward: train {wf['train_months']}m / test {wf['test_months']}m / "
        f"step {wf['step_months']}m; TSMOM{tuple(wf['lookbacks'])} + "
        f"SMA{tuple(wf['sma_filter_periods'])} gate; "
        f"fees {wf['fee_rate']} + slippage {wf['slippage_rate']}."
    )
    lines.append(
        f"Diagnostic re-run of the frozen config — no new search; "
        f"`PROJECT_NUM_TRIALS` stays {payload['num_trials']}."
    )
    lines.append("")
    lines.append(
        "**The DEPLOYED strategy's basket does not change based on this "
        "diagnostic regardless of the outcome — that requires a new owner "
        "decision (issue #14).**"
    )
    lines.append("")

    if payload["excluded_assets"]:
        lines.append("## Operator-acknowledged exclusions")
        lines.append("")
        for sym, reason in sorted(payload["excluded_assets"].items()):
            lines.append(f"* `{sym}` — {reason}")
        lines.append("")

    lines.append("## Summary — PIT top-N vs static-7 control (same price panel)")
    lines.append("")
    lines.append("| Metric | PIT top-N | Static control | Δ (PIT − control) |")
    lines.append("|---|---:|---:|---:|")
    pit_s = payload["runs"]["pit_top_n"]["summary"]
    ctl_s = payload["runs"]["static_control"]["summary"]
    for key, label in (
        ("concatenated_oos_sharpe", "Concat OOS Sharpe"),
        ("concatenated_oos_dsr", f"DSR @ N={payload['num_trials']}"),
        ("concatenated_oos_max_drawdown_pct", "Concat OOS max DD"),
        ("mean_test_sharpe", "Mean per-fold OOS Sharpe"),
        ("hit_rate", "Fold hit-rate"),
        ("n_folds", "Folds"),
    ):
        delta = payload["delta_pit_minus_control"].get(key)
        delta_cell = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(
            f"| {label} | {pit_s.get(key, 0.0):.4f} | "
            f"{ctl_s.get(key, 0.0):.4f} | {delta_cell} |"
        )
    lines.append("")

    for run_key, title in (
        ("pit_top_n", "Per-window results — PIT basket"),
        ("static_control", "Per-window results — static control"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        folds = payload["runs"][run_key]["folds"]
        if not folds:
            lines.append(
                "_No complete walk-forward folds in this window "
                "(panel shorter than train+test)._"
            )
            lines.append("")
            continue
        lines.append("| Test window | OOS Sharpe | Return % | Max DD % | Bars |")
        lines.append("|---|---:|---:|---:|---:|")
        for f in folds:
            lines.append(
                f"| {f['test_start']} → {f['test_end']} | {f['test_sharpe']:.2f} "
                f"| {f['test_return_pct'] * 100:.1f} "
                f"| {f['test_max_drawdown_pct'] * 100:.1f} | {f['test_bars']} |"
            )
        lines.append("")

    lines.append("## Basket composition per rebalance")
    lines.append("")
    lines.append(
        f"Static reference basket: {', '.join(payload['static_basket'])}. "
        "Rows marked ≠ deviate from it."
    )
    lines.append("")
    lines.append(
        "| Rebalance | N | Members | Added | Removed | vs static |"
    )
    lines.append("|---|---:|---|---|---|---|")
    for r in payload["rebalances"]:
        dev = ""
        if r["deviates_from_static"]:
            miss = "−" + ",".join(r["missing_vs_static"]) if r["missing_vs_static"] else ""
            extra = "+" + ",".join(r["extra_vs_static"]) if r["extra_vs_static"] else ""
            dev = f"**≠** {extra} {miss}".strip()
        lines.append(
            f"| {r['date']} | {r['n_members']} | {', '.join(r['members'])} "
            f"| {', '.join(r['added']) or '—'} | {', '.join(r['removed']) or '—'} "
            f"| {dev or '='} |"
        )
    lines.append("")
    (out_dir / f"{REPORT_STEM}.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default="data/coinmetrics")
    parser.add_argument("--out-dir", default="docs/results")
    parser.add_argument(
        "--start", default="2018-01-01",
        help="Panel start (default matches the deployed backtest window).",
    )
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch missing Coin Metrics series (otherwise cache-only).",
    )
    parser.add_argument(
        "--allow-missing", action="append", default=[], metavar="SYMBOL",
        help="Explicitly acknowledge a registry coin with no Coin Metrics "
        "coverage; recorded in the report. Repeatable.",
    )
    args = parser.parse_args(argv)

    prices, market_caps, volumes, pool, excluded = load_diagnostic_panel(
        args.cache_dir,
        allow_missing=args.allow_missing,
        static_basket=PRODUCTION_CONFIG.assets,
        fetch=args.fetch,
    )
    payload = run_diagnostic(
        prices,
        market_caps,
        volumes,
        pool,
        out_dir=args.out_dir,
        start=args.start,
        end=args.end,
        excluded=excluded,
    )
    pit = payload["runs"]["pit_top_n"]["summary"]
    ctl = payload["runs"]["static_control"]["summary"]
    print(
        f"PIT concat OOS Sharpe {pit['concatenated_oos_sharpe']:.2f} "
        f"(DSR {pit['concatenated_oos_dsr']:.3f}) vs control "
        f"{ctl['concatenated_oos_sharpe']:.2f} "
        f"(DSR {ctl['concatenated_oos_dsr']:.3f}); report in "
        f"{Path(args.out_dir) / (REPORT_STEM + '.md')}"
    )


if __name__ == "__main__":
    main()
