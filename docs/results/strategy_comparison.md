# Strategy comparison — Research-Claude panel

This file records the side-by-side performance of the strategy families
identified by the Research-Claude survey (`compass_artifact_*.md` in the
repo root). All figures are produced by `trade-lab compare` on Binance
1d candles fetched into `data/`, with `fee_rate=0.001` and
`slippage_rate=0.0005`.

> **Read alongside `compass_artifact_*.md`.** The survey explicitly
> warns: no strategy is guaranteed profitable going forward, every
> historical figure is subject to look-ahead, regime, and survivorship
> biases, and parameters were *not* tuned on this data — they were
> picked from the literature defaults *before* running the comparison.

## Single-asset comparison (`trade-lab compare`)

Strategies (priority from the Research-Claude panel in parentheses):

| Strategy             | Family                            | Research-Claude priority |
|----------------------|-----------------------------------|--------------------------|
| `buy_and_hold`       | Baseline                          | n/a                      |
| `sma_cross_20_100`   | SMA crossover                     | 5/5                      |
| `donchian_trend_rb0` | Donchian + SMA + vol target       | 4/5                      |
| `donchian_trend_rb005` | same with rebalance band        | 4/5                      |
| `tsmom_1_3_6_12m`    | Time-series momentum (Moskowitz)  | **5/5**                  |
| `pma_ratio_ensemble` | P/MA ratios (Detzel et al.)       | **5/5**                  |

Universe: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT (BTC/ETH/BNB from
2018-01-01, SOL from 2020-08-11). Subperiods: 2018, 2019, 2020-2021,
2022, 2023-2025, full.

### Headline averages across all four assets, full window

(Average over the four full-window cells, one per asset.)

| Strategy             | Avg return | Avg CAGR | Avg max DD | Avg Sharpe | Avg exposure | Avg trades |
|----------------------|-----------:|---------:|-----------:|-----------:|-------------:|-----------:|
| `buy_and_hold`       |   +2649%   |  +44%    |   88%      |  +0.82     |  100%        |       0    |
| `sma_cross_20_100`   |   +3279%   |  +52%    |   70%      |  +0.95     |   51%        |      14    |
| `donchian_trend_rb0` |    +258%   |  +17%    |   26%      |  +1.05     |   42%        |      45    |
| `donchian_trend_rb005`|   +268%   |  +18%    |   26%      |  +1.06     |   42%        |      45    |
| `tsmom_1_3_6_12m`    |    +231%   |  +17%    |   **23%**  |  +1.00     |   50%        |      32    |
| `pma_ratio_ensemble` |    +298%   |  +19%    |   25%      |  **+1.07** |   77%        |     166    |

**Direct match with the Research-Claude predictions:**

1. **SMA crossover wins raw return** on the bull-dominated universe
   (BNB/SOL both 10x+ from 2018-2024) — matches Corbet/Eraslan/Lucey/
   Sensoy (2019).
2. **All four trend variants beat `buy_and_hold` on Sharpe**, paid for
   with lower exposure and a fraction of the max DD — the canonical
   Hurst-Ooi-Pedersen 2017 result.
3. **`tsmom_1_3_6_12m` has the lowest average max DD** (23%) — pure
   time-series momentum's trademark insurance characteristic.
4. **`pma_ratio_ensemble` has the highest Sharpe** (1.07) — consistent
   with Detzel et al. (2021) which claims "economically significant
   alpha and Sharpe ratio gains relative to a buy-and-hold position".
   It also has the highest turnover, which the literature warns about.

### What you give up vs `buy_and_hold` in pure bulls

The Research-Claude survey makes a point of saying this **is not a
bug**:

> *"trend systems regularly lose to buy-and-hold in pure bull markets —
> that is the normal price for lower drawdowns"*

Confirmed cell-by-cell:

| Cell                       | B&H return | Best trend return | DD gap |
|----------------------------|-----------:|------------------:|-------:|
| BNB 2020-2021              |   +3628%   |     +144%         | B&H 65% vs trend 11% |
| SOL 2020-2021              |   +5054%   |      +91%         | B&H 75% vs trend 9% |
| BNB full (2018-2026)       |   +7570%   |   +8667% (SMA)    | B&H 80% vs trend 30% |
| SOL full (2020-2026)       |   +2402%   |   +2699% (SMA)    | B&H 96% vs trend 27% |

`sma_cross_20_100` and `buy_and_hold` are roughly tied on the full
window for the alts; the Donchian / TSMOM / PMA variants come in with
about a tenth of the drawdown but a tenth of the return. That's the
trade.

### What you keep in pure bears

| Cell             | B&H return | Best trend return |
|------------------|-----------:|------------------:|
| BTC 2018         |    -73%    |      0%           |
| BTC 2022         |    -64%    |      0%           |
| ETH 2018         |    -82%    |      0%           |
| ETH 2022         |    -67%    |      0%           |
| SOL 2022         |    -94%    |      0%           |

Across every bear year, the Donchian and TSMOM variants either stayed
flat or lost only a few percent. PMA ensemble takes a 6-15% loss
because its short MAs flip during oversold rallies.

## Multi-asset rotation (`run_cross_sectional_momentum`)

Cross-sectional momentum cannot be expressed in the single-asset
`Strategy` interface — it needs the full panel at once. Results below
were generated on the same BTC/ETH/BNB/SOL universe, weekly rebalance,
30-day lookback, top-2 basket, `fee_rate=0.001`, `slippage_rate=0.0005`.

| Variant                                | Total return | Max DD | Sharpe | Rebalances | Avg basket | Cash fraction |
|----------------------------------------|-------------:|-------:|-------:|-----------:|-----------:|--------------:|
| Equal weight, no BTC gate              |   +11077%    |  72%   |  1.16  |    200     |   1.27     |     31%       |
| Equal weight + BTC > SMA(200) (alts)   |   +22737%    |  **48%** |  **1.40** |    101     |   0.80     |     57%       |
| Inverse-vol, no BTC gate               |    +8322%    |  72%   |  1.12  |    328     |   1.27     |     31%       |
| Reference: BTC buy-and-hold (same window) | +456%      |  ~75%  |  ~1.0  |      0     |   1.00     |      0%       |

The BTC-gated, alt-only version is the clear standout — but the
following caveats matter at least as much as the numbers:

1. **Survivorship bias is severe.** The four-asset universe was picked
   knowing all of them survived. A real-time 2020 retail trader would
   have likely held some of LUNA/FTT/SUSHI in their top-N rotation; both
   went to zero (LUNA in May 2022, FTT in Nov 2022). The
   Research-Claude survey calls this out specifically: *"survivorship
   bias critical in crypto, since most coins die."*
2. **Universe is too small.** The literature standard is 10-30 coins.
   With four, "top-2" really means "rotation between the best two of
   four" — much less of a cross-section than the academic setup.
3. **Fees compound.** $94k+ in lifetime fees on a $10k starting account
   means the strategy is exposed to fee model assumptions. Binance
   taker is 0.1% (used here); higher tiers can be lower, retail without
   BNB discount can be higher.
4. **`sharpe` here is annualized using 365** to match daily compounding.

## Reproducing

```bash
# Single-asset comparison
trade-lab compare \
  --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT \
  --timeframe 1d \
  --output-csv outputs/compare_research_claude.csv \
  --output-md outputs/compare_research_claude.md

# Cross-sectional momentum (Python, no CLI yet)
python -c "
import pandas as pd
from trade_lab.backtest.cross_sectional import run_cross_sectional_momentum

assets = {s + '/USDT': pd.read_parquet(f'data/binance_{s}_USDT_1d.parquet')
          for s in ['BTC', 'ETH', 'BNB', 'SOL']}
res = run_cross_sectional_momentum(
    {k: v for k, v in assets.items() if k != 'BTC/USDT'},
    lookback_days=30, rebalance_days=7, top_k=2,
    weighting='equal',
    btc_candles=assets['BTC/USDT'],
    btc_gate_sma_period=200,
)
print(f'return={res.total_return:+.2%}, DD={res.max_drawdown:.2%}, '
      f'Sharpe={res.sharpe:+.2f}')
"
```

## Honest summary

The Research-Claude survey predicted three things that this comparison
confirms:

1. **No single strategy is the "answer".** SMA crossover wins on raw
   return; PMA on Sharpe; TSMOM on drawdown; cross-sectional momentum
   on absolute outperformance with survivorship caveats. The
   diversified portfolio of *several* of them is the honest
   recommendation.
2. **Trend-following pays its premium in bears, not bulls.** Confirmed
   in every 2018/2022 cell.
3. **The differences between trend strategies are small** relative to
   the difference vs `buy_and_hold`. Picking one over another based on
   a 1pp Sharpe gap is exactly the kind of selection bias the
   Deflated Sharpe Ratio (now in `trade_lab.backtest.dsr`) was designed
   to catch.

What this comparison does **not** establish:

- That any of these will work going forward — every figure is
  in-sample on the same historical window;
- That the parameters are optimal — they were picked from literature
  defaults precisely to avoid overfitting;
- That `+22000%` cross-sectional momentum on BTC/ETH/BNB/SOL is
  achievable in practice — see the survivorship caveats above.

The next correct moves, per the survey, are: walk-forward validation
of every strategy on the full universe; cost sensitivity sweep;
running PBO (Probability of Backtest Overfitting) on the parameter
grids; and paper trading on Binance testnet for at least 4-8 weeks
before any real money.
