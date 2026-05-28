# Yearly fixed-strategy validation — BTC/USDT 1d, 2018-2026

Output of `trade-lab yearly --symbol BTC/USDT --timeframe 1d` on the
9 calendar years between 2018-01-01 and 2026-05-27 (the last year is
partial; metrics are computed on what's available).

Each strategy is run *once on the full history* (so indicators have
proper warmup) and then sliced by calendar year for the per-year
metrics. Buy-and-hold rows are synthesized from close prices alone
(no fees) to give a clean baseline.

## Per-strategy aggregate

| strategy                    | total_years | avg_annual | median | best     | worst   | bh_wins | lower_dd_yrs | avg_exposure |
|-----------------------------|-------------|------------|--------|----------|---------|---------|--------------|--------------|
| buy_and_hold                | 9           | +61.53%    | +57.57% | +301.67% | -72.33% | 0       | 0            | 100%         |
| regime_only_200             | 9           | +40.58%    | +0.00%  | +173.23% | -21.71% | 3       | 8            | 48%          |
| regime_only_300             | 9           | +41.60%    | +0.00%  | +191.17% | -5.61%  | 5       | 7            | 55%          |
| sma_cross_20_100            | 9           | +40.92%    | +52.72% | +169.55% | -33.28% | 5       | 6            | 50%          |
| regime_sma_cross_20_100_200 | 9           | +44.16%    | +10.95% | +140.19% | +0.00%  | 5       | 7            | 43%          |
| golden_cross_50_200         | 9           | +32.00%    | +17.47% | +108.56% | -28.93% | 3       | 7            | 48%          |

## Reading

All active strategies trail buy-and-hold on average annual return
(40-44% vs 61%), but every single one wraps that gap inside a much
better worst year and lower drawdowns in 6-8 of 9 years.
`regime_sma_cross`'s worst year was **0%** (the regime filter kept it
in cash through the entire 2018 and 2022 bears) vs B&H's -72%
worst year.

The defensive strategies earn their keep by sitting in cash during
2018, 2022, and parts of 2025 — years where holding BTC was painful.
That tradeoff — slightly lower upside, dramatically better downside —
is the actual reason to run any of these over passive holding.

## Reproduce

```bash
trade-lab yearly --symbol BTC/USDT --timeframe 1d \
    --output-csv outputs/yearly.csv
```

Detail CSV (one row per (year, strategy)) and aggregate CSV are
written next to each other.
