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

5. **Rebalance band** (`rebalance_threshold`, default `0.05`). The
   pure target from steps 1-4 changes every bar because realized vol
   drifts daily. Without smoothing, every small adjustment pays fees
   + slippage. The band only updates the held position when
   `abs(target - current) >= rebalance_threshold`; otherwise it sticks
   with the previous size. **Entries (0 → positive target) and exits
   (positive → 0) are never suppressed** — the band only ignores size
   tweaks, not state changes. Setting `rebalance_threshold=0`
   reproduces the unfiltered behaviour exactly.

   The band exists to reduce fee drag, not to create alpha. It does
   not change *when* the strategy is in or out of the market; it just
   reduces unnecessary micro-rebalancing while in a position.

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
capital in cumulative fees + slippage (default `rebalance_threshold=0`
results; the default 0.05 band trims that by ~10%, see below).

## Rebalance-band sensitivity (BTC/USDT 1d, full window)

Same backtest, varying `rebalance_threshold`; everything else default.

| threshold | net return | max DD | completed trades | total fees | exposure |
|-----------|------------|--------|------------------|------------|----------|
| 0.00      | +314.02%   | 22.62% | 44               | $1,587.60  | 45.45%   |
| 0.025     | +314.83%   | 22.09% | 44               | $1,465.61  | 45.45%   |
| 0.05      | +326.16%   | 22.19% | 44               | $1,418.05  | 45.45%   |
| 0.10      | +322.34%   | 21.69% | 44               | $1,360.23  | 45.45%   |

What this shows:

- **Trade count and exposure are unchanged** across every threshold.
  The band only suppresses *size adjustments*, never entries or exits.
- **Fees drop monotonically** as the threshold widens — from $1,588
  (~16% of initial capital) at 0 to $1,360 (~14%) at 0.10.
- **Return / drawdown are essentially unchanged.** The 3-4pp return
  improvement at higher thresholds is within noise and should not be
  treated as alpha. *Do not pick the threshold that maximizes return.*

The default `0.05` is a conservative pick: cuts ~10% off fees while
keeping the return / DD shape recognizable. If you care more about
turnover, `0.10` is fine too. If you want to reproduce the historical
(no-band) behaviour, set `rebalance_threshold=0`.

## Known limitations

- **Single-asset only.** The engine processes one symbol at a time; the
  multi-asset portfolio cap from the strategy spec is enforced per-asset
  by `max_position_size` rather than across a basket.
- **Turnover is still continuous within a trade.** The rebalance band
  (`rebalance_threshold`, default 0.05) reduces *micro*-rebalancing but
  doesn't eliminate it — every time realized vol drifts by more than
  the threshold, the position resizes once and pays a fee. For the
  defaults on BTC/USDT 1d, cumulative fees + slippage are still ~14%
  of initial capital over 8 years.
- **BTC gate is per-instance.** It works programmatically by passing
  `btc_candles` to the constructor but isn't yet wired into the CLI
  (which loads only one symbol's candles).
- **No funding / borrowing.** Spot-only, long-or-cash. No carry costs
  modelled for futures or perps.
