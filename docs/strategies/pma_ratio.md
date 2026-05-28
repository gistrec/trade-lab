# `pma_ratio` — Price-to-Moving-Average Ratio Ensemble

`pma_ratio` (long-only) operationalizes the Detzel et al. (2021) result
on Bitcoin technical analysis. The Research-Claude survey lists it at
priority 5/5 with high evidence quality.

> *"We show that ratios of prices to their moving averages forecast
> daily Bitcoin returns in- and out-of-sample. Trading strategies based
> on these ratios generate an economically significant alpha and Sharpe
> ratio gains relative to a buy-and-hold position."*
> — Detzel, Liu, Strauss, Zhou, Zhu (2021). *Learning and predictability
> via technical analysis: Evidence from Bitcoin and stocks with
> hard-to-value fundamentals.* **Financial Management**.

The paper models the ratio as a rational-learning signal when
fundamentals are hard to value (a category that explicitly includes
crypto). It supplies in- and out-of-sample evidence that the *direction*
of `close / SMA(k)` predicts daily returns for `k ∈ {5, 10, 20, 50, 100}`
— but stops short of letting the magnitude scale position size.

This implementation honors that distinction: each ratio becomes a
`{0, 1}` vote, and we average the votes.

## Rules

1. **Vote ensemble.** For each `k` in `ma_periods` (default
   `5, 10, 20, 50, 100`):
   - `sma_k = close.rolling(k).mean()`
   - State: `1.0` if `close > sma_k`, else `0.0`. Warm-up bars (NaN
     SMA) → `0.0`.
   - Raw signal: the average of the per-`k` states. With five
     `ma_periods` the levels are `{0, 1/5, 2/5, 3/5, 4/5, 1}`.

2. **Optional SMA regime filter.** Same wiring as `donchian_trend` /
   `tsmom`; off by default (the ensemble already encodes regime info
   via the longest window).

3. **Volatility targeting.** Identical to the rest of the trend stack:
   `target_vol / realized_vol`, capped at `max_position_size`.

4. **Rebalance band.** Default `0.05`, same semantics as elsewhere —
   entries and exits always fire; only mid-position size adjustments
   are suppressed.

## Constructor parameters

| Parameter            | Default                | Notes |
|----------------------|------------------------|-------|
| `ma_periods`         | `(5, 10, 20, 50, 100)` | The lookbacks from Detzel et al. table 3. |
| `sma_filter_periods` | `()`                   | Empty = no extra gate. |
| `vol_lookback`       | `30`                   | Same as siblings. |
| `annual_vol_target`  | `0.25`                 | Same as siblings. |
| `annualization_factor` | `365`                | 24/7 crypto. |
| `max_position_size`  | `1.0`                  | Spot only. |
| `rebalance_threshold`| `0.05`                 | Suppresses small size adjustments. |

## Where it works

- High exposure during trends: short MAs flip first, the ensemble
  stair-steps up from `1/5` to `1.0`.
- Conservative in choppy markets: at least one MA usually disagrees,
  so the position lingers at `2/5..3/5` instead of going to full risk.

## Failure modes

- **Highest turnover of the trend stack** in this repo. Short MAs
  flip on noise, costing the most fees per bar. The rebalance band
  helps but does not eliminate the gap.
- **The signal is binary at the per-`k` level.** The paper uses the
  ratio's *magnitude* in a regression; this strategy throws away the
  magnitude. That is a deliberate simplification — the magnitude
  invites parameter overfitting, while the sign is robust per their
  tables 2-3.

See `docs/results/strategy_comparison.md` for the full subperiod-by-asset
breakdown.
