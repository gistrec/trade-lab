# trade-lab — research & results

A research framework for backtesting crypto-spot trading strategies, built
around a **layered-honesty** validation stack: every edge claim is shown
net-of-cost, then out-of-sample, then as a Deflated Sharpe Ratio against a
fixed multiple-testing budget (`N = 500` trials). The goal is to separate a
real, robust edge from a cherry-picked backtest — and to be honest about which
is which.

The one strategy that survived every layer is now **paper-trading on Binance
testnet**; the live dashboard is at [**/monitoring/**](/monitoring/).

## Start here

- **[Master results index](RESULTS.md)** — all 14 strategies/variants at a
  glance: status (PAPER / cluster-stable / REJECT / inconclusive), key metric,
  and a link to each writeup.

## What's on this site

- **Research findings** — 21 self-contained writeups: the deployable strategy,
  overlays and wrappers, the rejections (with *why* they failed net-of-cost),
  the five independent validation tests, and the literature reviews.
- **Strategy reference** — the signal definitions (TSMOM, SMA-cross, Donchian,
  PMA-ratio, RSI, regime gates, cross-sectional momentum).
- **Results analyses** — walk-forward, DSR-in-walk-forward, vol-targeting,
  multi-asset, point-in-time universe, and yearly breakdowns.
- **Methodology** — the validation discipline and the look-ahead / benchmark
  audit that underpin every number here.

## The deployable strategy (in one line)

TSMOM lookbacks `(28, 60)` on an equal-weight market-basket index of 7 majors
(BTC, ETH, BNB, SOL, ADA, XRP, DOGE), gated by SMA(200) on the basket close,
with a pro-rata ladder `{0, 0.5, 1.0}`. Concatenated OOS Sharpe **+1.81**,
DSR **0.77** at N=500, cluster-stable across its parameter neighbourhood. See
[han_28d_tsmom](findings/han_28d_tsmom.md) and
[production_config_v1](findings/production_config_v1.md) for the full picture.

!!! note "What PAPER status does and does not mean"
    A passing DSR at N=500 only rules out that the result is statistical noise
    given the search budget. It is **not** a promise of forward profitability —
    that is what the paper-trading phase is for.
