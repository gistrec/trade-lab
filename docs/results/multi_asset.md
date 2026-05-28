# Multi-asset fixed-strategy validation

Output of `trade-lab multi-asset --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT`
on daily candles. BTC / ETH / BNB span 2018-01-01 → 2026-05-27;
SOL starts 2020-08. **34 asset-years** in total.

Same fixed parameters as the BTC-only yearly validation; no tuning.
The point is to test whether behaviour observed on BTC generalizes
to other liquid crypto majors.

## Cross-asset summary (one row per strategy)

| strategy                    | avg_return | avg_worst_year | bh_wins/34 | lower_dd/34 | avg_expo |
|-----------------------------|------------|----------------|------------|-------------|----------|
| buy_and_hold                | +436.14%   | -75.66%        | 0          | 0           | 100%     |
| regime_only_200             | +98.71%    | -23.79%        | 17         | 26          | 46%      |
| regime_only_300             | +88.18%    | -31.54%        | 16         | 26          | 48%      |
| sma_cross_20_100            | +202.05%   | -42.78%        | 20         | 25          | 48%      |
| regime_sma_cross_20_100_200 | +95.36%    | -13.93%        | 18         | 27          | 38%      |
| golden_cross_50_200         | +89.37%    | -37.83%        | 13         | 26          | 46%      |

## Three things to take away

1. **The regime filters generalize.** Every active strategy reduces
   drawdown vs buy-and-hold in 25-27 out of 34 asset-years — the
   effect is not a BTC artefact.
2. **`regime_sma_cross` has the best worst-year on average (-14%).**
   On BTC alone the worst was 0%; ETH/BNB/SOL pull the average down
   only to -14%. Compare to buy-and-hold averaging a -76% worst year
   across the four assets.
3. **SOL skews the average return numbers.** B&H on SOL returned
   +1414% on average (one year was +9128%), which the active
   strategies can't match because they take time to enter. That
   extreme rebound year is the main reason any strategy "loses" on
   raw return across this panel.

## Reproduce

```bash
trade-lab multi-asset --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT \
    --timeframe 1d --output-csv outputs/multi_asset.csv
```

Three CSVs are produced: per-(asset, year, strategy) detail,
per-(asset, strategy) aggregate, and the across-asset summary above.
