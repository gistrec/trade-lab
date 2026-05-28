# Finding — Vol-targeting × regime gate interaction is asset-conditional

**Status:** strong evidence, N=7 assets, single timeframe.

## Finding

A Moreira-Muir-style volatility-targeting wrapper layered on top of
a long-only crypto trend-following strategy with an SMA(200) regime
gate does **not** uniformly improve risk-adjusted return. On BTC it
consistently *hurts* both Sharpe and Calmar; on ETH it consistently
*helps*. Across 7 Binance USDT pairs the pattern is strategy- and
asset-dependent rather than universal.

## Hypothesis

The regime gate and the vol-targeting wrapper compete for the same
risk-management role.

* When a strategy has a strong regime gate that already removes
  exposure during crash periods, **most of the asset's "high-vol"
  observations happen outside the strategy's holding window**. The
  remaining in-position vol is closer to the asset's calm regime
  median.
* In that residual regime, vol-targeting still shrinks position
  during *bull-rally* vol spikes (vol cares about magnitude, not
  direction). For BTC 2020-2026, those bull-rally spikes are where
  most of the OOS return came from, so the shrinkage costs more
  return than it saves in drawdown.
* For assets where the gate does *not* cleanly remove all crash-vol
  (ETH and several alts have idiosyncratic shocks the SMA(200) gate
  misses by days), vol-targeting catches what the gate misses and
  the math works out the other way.

## Implication

Vol-targeting is **not** a universal "always on" layer for retail-
spot crypto trend-following with regime gates.

Recommended decision rules from this dataset:

* `tsmom_medium`: vol-targeting helps Sharpe on 6/7 assets (vol50 wins
  on 5 of those). Default it ON.
* `pma_medium`: vol-targeting helps Sharpe on 6/7 assets (vol30 wins
  on all 6). Default it ON at target=0.30.
* `sma_20_100`: vol-targeting is a wash on Sharpe (mean Δ = +0.02 ±
  0.10) — neither clearly helps nor clearly hurts. Default it OFF.
* **BTC specifically: vol-targeting LOSES on every strategy.** Either
  hard-code an exception or accept that vol-targeting recommendations
  must be configurable per asset.

The decision belongs in **the strategy/asset configuration**, not the
strategy code. The wrapper itself is a generic decorator that should
be parameterized externally.

## Evidence — 7-asset Sharpe / Calmar matrix

(24m train / 6m test / 13 OOS folds, BTC/ETH/BNB/SOL/ADA/XRP/DOGE 1d.)

### Sharpe — winner counts per (strategy, variant)

|  Strategy       | raw | vol30 | vol50 |
|----------------|----:|------:|------:|
| sma_20_100     |  3  |   2   |   2   |  ← no clear winner
| tsmom_medium   |  1  |   1   | **5** |  ← vol-targeting wins
| pma_medium     |  1  | **6** |   0   |  ← vol30 sweeps

### Calmar — winner counts per (strategy, variant)

|  Strategy       | raw | vol30 | vol50 |
|----------------|----:|------:|------:|
| sma_20_100     |**5**|   0   |   2   |  ← raw wins
| tsmom_medium   |**4**|   0   |   3   |  ← raw narrowly wins
| pma_medium     |  3  |   0   | **4** |  ← vol50 narrowly wins

### Mean ΔSharpe (vol30 − raw) across 7 assets

|  Strategy       | mean ΔSharpe | std ΔSharpe |
|----------------|-------------:|------------:|
| sma_20_100     |    +0.02     |    0.10     |
| tsmom_medium   |    +0.12     |    0.11     |
| pma_medium     |  **+0.22**   |    0.17     |

`pma_medium` gains the most Sharpe from vol-targeting *and* has the
largest standard deviation of that gain across assets. Translation:
the vol-targeting effect is strongest on PMA but also least uniform.

## Hit-rate dispersion — partially against an earlier hypothesis

The user hypothesized that vol-targeting affects PMA's hit-rate more
*stably* than SMA's, because PMA generates ~150 signal events/year
vs SMA's ~4. The data are mixed:

|  Strategy       | signals/yr (BTC) | std ΔHit(vol30) | range ΔHit(vol30) |
|----------------|-----------------:|----------------:|------------------:|
| sma_20_100     |        3.7       |       6 pp      |       20 pp        |
| tsmom_medium   |       29.0       |       3 pp      |        8 pp        |
| pma_medium     |      152.5       |       9 pp      |       23 pp        |

Plot twist: **tsmom_medium has the most stable hit-rate** under
vol-targeting, not PMA. PMA's high signal density does *not* yield
the lowest dispersion. A plausible reason: PMA's "signals" are
mostly small ladder steps (1/5 → 2/5 etc.) that are almost cosmetic;
the effective number of *meaningful* hit-rate-shifting events is
closer to TSMOM's.

Practical takeaway: **TSMOM, not PMA, is the variant whose hit-rate
behaves most predictably under vol-targeting**. If we ever build a
weighted ensemble of the three, this matters for stability.

## Caveat

* **N=7 assets is suggestive, not definitive.** Five of the seven are
  USD-quoted top-10 alts as of 2026; the sample is biased toward
  survivors of the 2018-2026 window.
* **Single timeframe (1d).** A 4h or 1h walk-forward might give
  different vol-targeting payoffs because realized vol scales with
  sqrt(time).
* **Single regime-gate spec.** All three strategies use SMA(200) (or
  no regime filter for sma_20_100). With a different gate (Donchian,
  longer SMA, multi-timeframe), the competition story above could
  shift.
* **2020-2026 was a single regime arc.** BTC's behaviour here ("regime
  gate enough, vol-targeting redundant") might not hold in a future
  regime with different vol structure.
* **Fees fixed at 0.10% + 0.05% slippage.** Vol-targeting reduces
  turnover; cheaper fees would shift the wash-out point.

To upgrade this finding from "strong evidence" to "confirmed law"
needs:

1. A second timeframe (4h or 1w) tested on the same 7 assets.
2. A second regime-gate variant (e.g. no gate, or 100-day SMA gate).
3. The PIT cross-sectional universe (deliberately includes LUNA/FTT/
   etc. mid-position vol shocks) re-run with vol-targeting.

The first one is cheap; the third is the most informative.

## Reproducing

```bash
python -c "
import pandas as pd
from trade_lab.backtest import (
    run_strategy_walk_forward, aggregate_walk_forward, ParamGridSpec,
)
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy
from trade_lab.strategies.vol_target_wrapper import VolatilityTargetWrapper

for asset in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE']:
    candles = pd.read_parquet(f'data/binance_{asset}_USDT_1d.parquet')
    for label, factory, warmup in [
        ('raw',   lambda: TimeSeriesMomentumStrategy(use_vol_target=False), 365),
        ('vol30', lambda: VolatilityTargetWrapper(
            TimeSeriesMomentumStrategy(use_vol_target=False),
            annual_vol_target=0.30), 400),
        ('vol50', lambda: VolatilityTargetWrapper(
            TimeSeriesMomentumStrategy(use_vol_target=False),
            annual_vol_target=0.50), 400),
    ]:
        grid = [ParamGridSpec(label=label, factory=factory, warmup_days=warmup)]
        detail = run_strategy_walk_forward(candles, grid)
        s = aggregate_walk_forward(detail)
        print(asset, label, f'{s[\"mean_test_sharpe\"]:+.2f}', f'{s[\"mean_test_calmar\"]:+.2f}')
"
```

Full raw matrix: `outputs/wf_voltarget_7assets.csv` (63 rows: 7 assets
× 3 strategies × 3 variants).
