# `tsmom` — Time-Series Momentum (multi-lookback)

`tsmom` (long-only) is the textbook implementation of *time-series
momentum*: at each bar, look back over one or more windows and bet
**with** the sign of the trailing return. The Research-Claude survey
(see `compass_artifact_*.md` in the repo root) ranks TSMOM 5/5 for
priority, citing:

- Moskowitz, Ooi, Pedersen (2012). *Time Series Momentum*. JFE 104(2).
  Documents significant TSMOM in 58 liquid futures contracts on 1-12
  month horizons.
- Hurst, Ooi, Pedersen (2017). *A Century of Evidence on Trend-Following
  Investing*. JPM. 137 years × 67 markets, vol-targeting included.
- Liu, Tsyvinski (2021). *Risks and Returns of Cryptocurrency*. RFS
  34(6). Replicates the effect on BTC, ETH, XRP at 1-4 week horizons.

The strategy explicitly drops the short leg from the canonical
long-short formulation (we are spot-only) and replaces the ±1 vote with
a {0, 1} long-vs-flat vote.

## Rules

1. **Multi-lookback sign-of-return ensemble.** For each `L` in
   `lookbacks` (default `30, 90, 180, 365` — roughly 1, 3, 6, 12
   months):
   - `past_return = close.pct_change(L)` — uses `close[N]` and
     `close[N - L]`, both available at the close of bar N.
   - State: `1.0` if `past_return > 0`, else `0.0`. Warm-up bars (NaN
     trailing return) are treated as flat.
   - Raw signal: the mean of the per-lookback states. With four
     lookbacks the levels are `{0, 1/4, 1/2, 3/4, 1}`.

2. **Optional SMA regime filter.** If `sma_filter_periods` is non-empty
   (default `(200,)`), the signal is zeroed wherever `close <= SMA(P)`
   for any configured `P`.

3. **Volatility targeting.** Same as `donchian_trend`:
   - `realized_vol_annual = std(daily_returns over vol_lookback) * sqrt(annualization_factor)`.
   - `vol_weight = annual_vol_target / realized_vol_annual` (default
     target 25% annual).
   - `target_position = clip(raw_signal * vol_weight, 0, max_position_size)`.

4. **Rebalance band.** As in `donchian_trend`: small size adjustments
   (delta below `rebalance_threshold`, default `0.05`) are suppressed,
   but entries (flat → long) and exits (long → flat) always fire.

All inputs use only data available at the close of bar `N`. The engine
shifts the signal by one bar before applying it.

## Constructor parameters

| Parameter            | Default              | Notes |
|----------------------|----------------------|-------|
| `lookbacks`          | `(30, 90, 180, 365)` | Tuple of ints (days). Can also be a CLI-style comma-separated string. |
| `sma_filter_periods` | `(200,)`             | Pass `()` to disable the regime gate. |
| `vol_lookback`       | `30`                 | Rolling window for realized vol. |
| `annual_vol_target`  | `0.25`               | Target annual volatility (25%). |
| `annualization_factor` | `365`              | 24/7 crypto markets. |
| `max_position_size`  | `1.0`                | Spot only — never above 1. |
| `rebalance_threshold`| `0.05`               | Suppresses sub-5pp size adjustments. |

## Where it works

- Strong sustained trends (e.g. 2020-2021 alts: TSMOM holds the move).
- Bear regimes (2018, 2022): regime filter forces cash, drawdown stays
  in single digits.

## Failure modes

- **Choppy markets.** Sign-of-return whipsaws; the ensemble dampens
  this but cannot eliminate it.
- **Late entries / exits.** A 12-month lookback only flips after the
  trend has been visible for months. That is the *cost* of the lower
  drawdown — Moskowitz et al. and Hurst et al. both document that TSMOM
  systematically lags raw buy-and-hold in pure bull regimes and pays
  for the difference with much smaller losses in bear regimes.

See `docs/results/strategy_comparison.md` for the full subperiod-by-asset
breakdown.
