# `donchian_trend` — Donchian trend ensemble with volatility targeting

`donchian_trend` is a research candidate that combines three building
blocks with relatively strong out-of-sample evidence individually, with
no ML, no short-term mean reversion, and no parameter optimization
baked into the strategy itself.

## Rules

1. **Donchian breakout ensemble.** For each lookback `L` in
   `donchian_lookbacks` (default `20, 50, 100`):
   - `prev_high = close.rolling(L).max().shift(1)` — strictly the prior
     `L` bars, never including today's close.
   - `prev_low = close.rolling(L).min().shift(1)`.
   - State machine: long if `close > prev_high`, flat if
     `close < prev_low`, otherwise hold the previous state.
   - The raw signal is the *mean* of the per-lookback long states, so
     partial agreement produces partial exposure (e.g. `1/3`, `2/3`,
     `1`).

2. **SMA trend filter.** Only allow long exposure while
   `close > SMA(P)` for every `P` in `sma_filter_periods` (default
   `100, 200`). Warm-up bars (where any SMA is still NaN) are treated
   as filter-fails-shut.

3. **Volatility targeting.**
   - `realized_vol_annual = std(daily_returns over vol_lookback) * sqrt(annualization_factor)`
     (default 30-bar lookback, `sqrt(365)`).
   - `vol_weight = annual_vol_target / realized_vol_annual` (default
     target 25%).
   - `position = clip(raw_signal * vol_weight, 0, max_position_size)`
     (default cap 1.0 — spot-only, no leverage).
   - Missing or zero realized vol maps to zero exposure (never silently
     levers up).

4. (Optional) **BTC market gate.** If you construct the strategy with
   `btc_candles=...`, exposure is additionally gated on
   `BTC > BTC SMA(btc_gate_sma_period)` (default 200). Useful when
   running the same rules on altcoins; redundant on BTC itself.

## Execution and lookahead

All signals at bar `N` use only data available at close of `N`. The
engine then applies its standard one-bar shift, so a signal at `N`
only affects positions from bar `N+1` onward. Returns are
close-to-close. Fees and slippage are charged on every change in
exposure (so vol-targeting micro-rebalances do incur cost — turnover
is real). The lookahead guarantee is pinned by
`tests/test_donchian_trend.py::test_no_lookahead_in_donchian_thresholds`.

## How to run

```bash
# Default parameters
trade-lab backtest --strategy donchian_trend --symbol BTC/USDT --timeframe 1d

# All parameters explicit
trade-lab backtest --strategy donchian_trend --symbol BTC/USDT --timeframe 1d \
    --param 'donchian_lookbacks=20,50,100' \
    --param 'sma_filter_periods=100,200' \
    --param vol_lookback=30 \
    --param annual_vol_target=0.25 \
    --param max_position_size=1.0
```

The CLI accepts the lookback lists as comma-separated strings via
`--param`; the strategy constructor normalizes them.

## Subperiod result on BTC/USDT 1d (default parameters)

The strategy was *not* tuned to these windows. Default lookbacks /
vol target are the ones documented above; the backtest spans
2018-01-01 through 2026-05-27 and the subperiods slice that data with
`--start-date` / `--end-date`.

| period      | strategy  | buy & hold | strat DD | B&H DD  | exposure | verdict               |
|-------------|-----------|------------|----------|---------|----------|-----------------------|
| 2018        | +0.00%    | −72.33%    | 0.00%    | 81.18%  | 0.0%     | OUTPERFORMS_BH        |
| 2019        | −3.55%    | +89.49%    | 7.02%    | 49.41%  | 13.4%    | LOWER_RETURN_LOWER_DD |
| 2020-2021   | +91.40%   | +541.83%   | 19.20%   | 53.60%  | 53.1%    | LOWER_RETURN_LOWER_DD |
| 2022        | +0.00%    | −65.34%    | 0.00%    | 66.93%  | 0.0%     | OUTPERFORMS_BH        |
| 2023-2025   | +62.43%   | +427.47%   | 16.24%   | 32.02%  | 51.5%    | LOWER_RETURN_LOWER_DD |
| **2018-2026** | **+314.02%** | **+456.42%** | **22.62%** | **81.18%** | **45.5%** | **LOWER_RETURN_LOWER_DD** |

Honest reading: the strategy stays in cash through both 2018 and 2022
bears (0% exposure, 0% return, 0% drawdown), at the cost of leaving
most of the 2020-2021 and 2023-2025 bull-run upside on the table. The
full-window verdict is `LOWER_RETURN_LOWER_DD` — *not* a profitability
claim. 44 round-trip trades across 8 years cost ~16% of initial
capital in cumulative fees + slippage.

## Known limitations

- **Single-asset only.** The engine processes one symbol at a time; the
  multi-asset portfolio cap from the strategy spec is enforced per-asset
  by `max_position_size` rather than across a basket.
- **Daily turnover from vol-targeting.** There is no rebalance band, so
  a daily change in realized vol creates a (small) rebalance every bar
  the strategy is long. Fees scale with that — the 16% lifetime fee
  burn on the BTC backtest above is partly from those micro rebalances.
  A rebalance band parameter is a natural follow-up.
- **BTC gate is per-instance.** It works programmatically by passing
  `btc_candles` to the constructor but isn't yet wired into the CLI
  (which loads only one symbol's candles).
- **No funding / borrowing.** Spot-only, long-or-cash. No carry costs
  modelled for futures or perps.
