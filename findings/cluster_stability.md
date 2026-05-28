# Finding — Cluster stability separates robust strategies from cherry-picks

**Status:** strong evidence; demotes SMA crossover, validates the
TSMOM short-ensemble as the most robust deployable family.

## Finding

The cluster-stability check (each variant in a parameter cluster
walk-forwarded *independently*, then we ask: what fraction of the
cluster clears DSR > 0.5?) produces qualitatively different verdicts
across the strategy families. Key result:

| Family / target              | N variants | N passing | Cluster verdict | Median DSR |
|------------------------------|-----------:|----------:|-----------------|-----------:|
| **Market basket**            |            |           |                 |            |
| SMA crossover (19 fast/slow) |    19      |    6      | **FAILS** (32%) |  0.431     |
| TSMOM Han single lookbacks   |     6      |    6      | **PASSES** (100%) | 0.702    |
| **TSMOM short-ensemble**     |     7      |    7      | **PASSES** (100%) | **0.736** |
| PMA ratio ladder             |     6      |    6      | PASSES (100%)   |  0.716     |
| **BTC per-asset**            |            |           |                 |            |
| SMA crossover                |    19      |    0      | **FAILS** (0%)  |  0.204     |
| TSMOM Han single lookbacks   |     6      |    2      | FAILS (33%)     |  0.395     |

The cluster check is the methodology the literature explicitly
recommends (Faber, Carver, Robot Wealth — see
`deep-research-report.md`). It distinguishes families that work
across a band of nearby parameter choices from individual points
that look good in isolation but degrade on either side.

## Three reads on the same data

### 1. SMA crossover is a cherry-pick, not a strategy family

On the market basket, 13 of 19 SMA (fast, slow) variants come in
below DSR 0.5 — including most "canonical" pairs like (20, 100),
(30, 100), (50, 200). The 6 that pass are scattered across the grid
(`(10, 50)`, `(10, 150)`, `(10, 300)`, `(20, 50)`, `(20, 300)`,
`(50, 300)`) with no obvious cluster structure.

On BTC the result is starker: **0 of 19 variants pass DSR 0.5**.
Best is (20, 300) at 0.411 — marginal at best.

What this means: **the previous "best SMA per WF fold" result was
selection noise**. Picking the train-best (fast, slow) pair was
giving us a different cherry each fold, which is exactly why the
walk-forward priority-5 step reported a 0.55-0.82 IS→OOS shrinkage
for SMA. The family is not cluster-stable.

**Implication: drop SMA crossover from the deployable shortlist.**
Keep it in the codebase as a research building block, but don't
treat it as a candidate strategy on its own.

### 2. TSMOM short-ensemble is the most robust family

7 variants of TSMOM with two short lookbacks each (combinations of
21/28/30 and 56/60/84/90):

| Variant            | Concat OOS Sharpe | DSR |
|--------------------|------------------:|----:|
| tsmom (30, 60)     |    +1.49          | 0.782 |
| tsmom (28, 60)     |    +1.48          | 0.770 |
| tsmom (21, 60)     |    +1.45          | 0.750 |
| tsmom (28, 56)     |    +1.44          | 0.736 |
| tsmom (28, 84)     |    +1.42          | 0.716 |
| tsmom (30, 90)     |    +1.39          | 0.693 |
| tsmom (28, 90)     |    +1.37          | 0.671 |

**All 7 pass DSR 0.5. The range is tight (0.671-0.782).** This is
what cluster-stability looks like: parameter choice doesn't matter
within a reasonable band; the family carries the signal.

This is now the **leading recommendation for the paper-trading
candidate**: TSMOM with two short lookbacks (any of 21-30 short +
56-90 medium) on the market basket. The picked deployment doesn't
need to nail the parameters exactly — anywhere in the cluster will
do.

### 3. PMA ladder is also cluster-stable on the basket

| Variant                       | Concat OOS Sharpe | DSR |
|-------------------------------|------------------:|----:|
| pma (20, 50, 100, 200)        |    +1.48          | 0.760 |
| pma (10, 20, 50, 100, 200)    |    +1.46          | 0.744 |
| pma (5, 10, 20, 50, 100)      |    +1.43          | 0.718 |
| pma (10, 20, 50)              |    +1.42          | 0.714 |
| pma (5, 10, 20, 50)           |    +1.40          | 0.692 |
| pma (5, 10, 20)               |    +1.27          | 0.562 |

**All 6 pass DSR 0.5.** The very short (5,10,20) ladder is the
weakest but still passes. The longer ladders (20,50,100,200) are
strongest.

PMA on the basket is a second viable family, with similar median
DSR (0.716) to the TSMOM short-ensemble (0.736). Their per-bar OOS
correlations (not computed here, follow-up) would tell us whether
a basket-level TSMOM + basket-level PMA portfolio is worth
constructing.

## Comparison to the "WF picks best" methodology

Up until this finding the project used `walk_forward_v2.run_strategy_walk_forward`
with a multi-variant grid and picked train-best per fold. That
methodology rewards the lucky cherry. Cluster-stability rewards the
family.

For the families above:

| Family + target          | "WF picks best" peak DSR | Cluster median DSR | Cluster verdict |
|--------------------------|-------------------------:|-------------------:|-----------------|
| SMA basket               |        0.717 (sma_10_50)  |     0.431          | FAILS           |
| SMA BTC                  |        0.411 (sma_20_300) |     0.204          | FAILS           |
| TSMOM short-ens basket   |        0.782 (30,60)      |     **0.736**      | **PASSES**      |
| PMA ladder basket        |        0.760 (20-200)     |     **0.716**      | **PASSES**      |
| TSMOM Han single BTC     |        0.720 (28d)        |     0.395          | FAILS           |

Reading: the **peak DSR overstates robustness**. The median DSR is
the honest representative of "what we should expect deployment to
deliver". For SMA and per-asset TSMOM the gap is huge (peak vs
median). For TSMOM short-ensemble and PMA ladder on the basket the
gap is tight — these are the real families.

## What this displaces in the deployable config

The previous lead (after the Han 28d finding) was **TSMOM single
(28d) on the market basket** at DSR 0.685, or even more pointed
**TSMOM short-ensemble (28, 60)** at DSR 0.770. The cluster-
stability check confirms that "TSMOM short-ensemble" — *with any
reasonable two-window combination* — is the right thing to deploy,
not a specific (28, 60) tuning.

The new ordering of paper-trading candidates:

1. **TSMOM short-ensemble on market basket** (any 21-30 short × 56-90
   medium pair). Robust across the cluster. Median DSR 0.736.
2. **PMA ladder on market basket** (any reasonable 3-5 lookback
   ladder spanning 10 to 100+ days). Robust across cluster.
   Median DSR 0.716.
3. (Implicit) An equal-weight portfolio of #1 and #2 on the basket,
   to test whether they're meaningfully decorrelated. Future check.

The previous lead "BTC Han 28d at DSR 0.720" is **demoted to
cherry-pick status** because its neighbours (Han 21d, Han 35d, Han
60d) don't pass. We do NOT recommend deploying BTC TSMOM by itself.

## Implication

* **Cluster-stability is a sharper test than DSR alone.** A single
  high-DSR variant might be a survivor; a high-fraction cluster is
  the family signal.
* **The strong findings (Han 28d, basket TSMOM) still stand**, but
  with revised interpretation: it's not "28d works" but "short
  lookbacks work, and 28d is one of them". Less precision; more
  honesty.
* **The negative findings (SMA crossover) are sharper too.** "0 of
  19 variants pass" on BTC is unambiguous.
* **For the paper-trading gate**, the deployable artifact is now:
  *"TSMOM with two short lookbacks (28-30 + 60-90 is fine) on the
  market-basket index, with the SMA(200) regime gate."* The
  precise lookback doesn't have to be defended.

## Caveats

* **PROJECT_NUM_TRIALS not bumped**: 19 + 6 + 7 + 6 + 19 + 6 = 63
  trials added in this check, comfortably inside the 350-buffer.
  This is also a methodology test, not a new-strategy search.
* **The cluster threshold (DSR > 0.5) and required fraction (50%)
  are choices.** A stricter run with required_fraction_pass=0.75
  would demote PMA's short (5,10,20) ladder and show only the
  longer PMA ladders as cluster-stable. A run with threshold_dsr=0.7
  would only count the very best variants.
* **Cluster definition is subjective.** We picked "all 19 (fast,
  slow) SMA pairs" as the SMA cluster, but a tighter cluster (say
  only 200-300 slow periods) might pass while the wider cluster
  fails. This is the standard objection to cluster-stability tests
  in the literature; we accept it.

## Reproducing

```python
import pandas as pd
from trade_lab.backtest import (
    run_cluster_stability_check, build_crypto_market_index,
    build_sma_grid, PROJECT_NUM_TRIALS,
)
asset_candles = {s: pd.read_parquet(f'data/binance_{s}_USDT_1d.parquet')
                 for s in ['BTC','ETH','BNB','SOL','ADA','XRP','DOGE']}
basket = build_crypto_market_index(asset_candles)
res = run_cluster_stability_check(
    basket, build_sma_grid(),
    threshold_dsr=0.5, required_fraction_pass=0.5,
    num_trials_for_dsr=PROJECT_NUM_TRIALS,
)
print('passing fraction:', res.fraction_passing,
      'median DSR:', round(res.median_dsr, 3),
      'cluster:', 'PASSES' if res.cluster_passes else 'FAILS')
```
