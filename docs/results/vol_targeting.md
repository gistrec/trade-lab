# Volatility targeting as a wrapper layer

Third of the five honesty checks. Question: **does adding a
volatility-targeting layer on top of `sma_cross` / `tsmom` /
`pma_ratio` close the in-sample / out-of-sample gap we found in
`walk_forward_priority5.md`?**

## What the wrapper does

`VolatilityTargetWrapper(inner, annual_vol_target, vol_lookback)` is
the same mechanic the literature uses (Moreira & Muir 2017,
*Journal of Finance*): take the inner strategy's `[0, 1]` position
signal and multiply by `annual_vol_target / realized_vol(window)`,
clipped at `max_position_size=1`. In high-vol regimes the position
shrinks; in low-vol regimes it grows up to the cap.

Important wiring detail: `TimeSeriesMomentumStrategy` and
`PriceMaRatioStrategy` already had inline vol-targeting from their
original implementations. To keep the wrapper comparison clean,
both strategies grew a `use_vol_target=False` flag in this round —
when set, they return the raw ensemble signal and the wrapper is the
only vol layer.

## Headline matrix — 24m train / 6m test / 13 OOS folds

`Calmar` is `mean(per-fold return / per-fold max DD)`. `hit_rate` is
the share of folds whose test return is positive.

### BTC (full-window realized vol ≈ 65%)

| Strategy       | Variant | OOS Sharpe | OOS Calmar | OOS return | OOS max DD | Hit-rate |
|----------------|---------|-----------:|-----------:|-----------:|-----------:|---------:|
| sma_20_100     | raw     |    **+0.67** |   **+1.40** |  +24.5%  |   21.0%  |   62%    |
| sma_20_100     | vol30   |     +0.54   |    +1.05   |  +12.4%  |   13.9%  |   54%    |
| sma_20_100     | vol50   |     +0.56   |    +1.13   |  +17.9%  |   19.3%  |   54%    |
| tsmom_medium   | raw     |    **+0.78** |   **+1.42** |  +24.1%  |   15.7%  |   46%    |
| tsmom_medium   | vol30   |     +0.69   |    +1.14   |  +13.1%  |   10.1%  |   46%    |
| tsmom_medium   | vol50   |     +0.68   |    +1.17   |  +19.2%  |   14.7%  |   46%    |
| pma_medium     | raw     |    **+0.55** |   **+1.17** |  +20.6%  |   21.4%  |   62%    |
| pma_medium     | vol30   |     +0.54   |    +1.08   |  +12.1%  |   13.5%  |   62%    |
| pma_medium     | vol50   |     +0.51   |    +1.02   |  +16.7%  |   20.1%  |   69%    |

> **On BTC, vol-targeting LOSES on both Sharpe and Calmar at every
> target level (15%, 20%, 30%, 50%, 70% — see TSMOM sensitivity
> below).** Raw wins.

### ETH (full-window realized vol ≈ 86%)

| Strategy       | Variant | OOS Sharpe | OOS Calmar | OOS return | OOS max DD | Hit-rate |
|----------------|---------|-----------:|-----------:|-----------:|-----------:|---------:|
| sma_20_100     | raw     |     +0.46   |    +1.00   |  +32.3%  |   30.0%  |   62%    |
| sma_20_100     | vol30   |     +0.49   |    +1.09   |  +13.6%  |   13.9%  |   62%    |
| sma_20_100     | **vol50** |  **+0.51** | **+1.23** |  +24.7%  |   21.0%  |   62%    |
| tsmom_medium   | raw     |     +0.52   |    +1.01   |  +34.6%  |   23.8%  |   54%    |
| tsmom_medium   | vol30   |    **+0.59** |    +1.07   |  +14.3%  |   10.6%  |   54%    |
| tsmom_medium   | **vol50** |  **+0.59** | **+1.21** |  +25.6%  |   16.5%  |   54%    |
| pma_medium     | raw     |     +0.54   |    +0.74   |  +23.9%  |   27.8%  |   69%    |
| pma_medium     | **vol30** | **+0.63** | **+1.00** |  +11.3%  |   12.7%  |   69%    |
| pma_medium     | vol50   |     +0.60   |    +1.05   |  +18.9%  |   20.0%  |   62%    |

> **On ETH, vol-targeting WINS on both Sharpe AND Calmar across all
> three strategies.** Target=0.50 generally beats target=0.30 by
> Calmar; for PMA, target=0.30 dominates Sharpe. This is the
> Moreira-Muir pattern the literature predicts.

## Why the result is asset-conditional

The two assets see *opposite* outcomes. The mechanism:

* **BTC realized vol ≈ 65% annualized over the test window.** Mean
  trend strategy's raw exposure is 50-60% of bars. The SMA(200)
  regime gate already kicks the strategy into cash during the
  worst-vol *crash* periods. So when the strategy *is* in position,
  the realized vol is closer to BTC's calmer regime (35-50%). Wrapper
  at target=0.30 scales position to 0.30/realized ≈ 0.6-0.8 most of
  the time. The shrinkage is biggest during *bull rallies* (vol of
  big up-moves), and BTC's big-up-moves dominate the OOS-window's
  total return. Result: return shrinks more than DD shrinks, Calmar
  worsens.
* **ETH realized vol ≈ 86%.** The asset has more idiosyncratic
  vol spikes (DeFi exploits, ETH-specific narratives) that occur
  *while the strategy is long*. These are real risk events the SMA
  regime filter doesn't catch quickly enough. The wrapper trims
  exposure precisely during these spikes; the DD savings exceed the
  return cost, and Calmar improves.

The pattern the user expected ("vol-targeting almost always wins on
Calmar") is real — but **on BTC daily over 2020-2026 it doesn't
apply**, because the underlying strategies' DD already comes mostly
from being-out-of-position rather than being-in-position-during-vol-spike.

## TSMOM sensitivity sweep on BTC

To make sure the conclusion isn't an artifact of picking target=0.30
or 0.50 unhappily:

|Variant | OOS Sharpe | OOS Calmar | OOS return | OOS max DD | Hit-rate |
|--------|-----------:|-----------:|-----------:|-----------:|---------:|
| raw    |   **+0.78** |   **+1.42** |  +24.1%  |   15.7%  |   46%    |
| vol15  |    +0.67   |    +1.03   |   +6.0%  |    5.3%  |   54%    |
| vol20  |    +0.67   |    +1.07   |   +8.3%  |    7.0%  |   54%    |
| vol30  |    +0.69   |    +1.14   |  +13.1%  |   10.1%  |   46%    |
| vol50  |    +0.68   |    +1.17   |  +19.2%  |   14.7%  |   46%    |
| vol70  |    +0.72   |    +1.36   |  +24.1%  |   17.0%  |   46%    |

Calmar is **monotonically increasing toward "raw"** as the target
grows. The reason: at target≈realized_vol (~70% for BTC) the wrapper
is approximately identity. The wrapper helps DD only in the strict-
shrinkage regime where target < realized — but on BTC that strict-
shrinkage shrinks return more than DD.

Hit-rate **does increase at low targets** (54% vs 46% for raw at
target=15-20%) — the small-loss folds become small-positive folds
because the strategy is barely in the market. That's the asymmetric
left-tail clipping the literature talks about, but it doesn't survive
the Calmar test on BTC.

## The convexity caveat — partially observed

The user warned in advance: **vol-targeting can hurt hit-rate because
when realized vol spikes during crashes, the position shrinks and we
miss part of the rebound**. We saw a mild version of this on BTC's
`sma_20_100` (hit-rate 62% → 54%) but no clear signal on the other
five strategy×asset combinations. On `pma_medium / vol50 / BTC` the
hit-rate actually *increased* to 69%, the opposite of the predicted
pattern.

Useful reframing: vol-targeting is *not a free lunch*. Its expected
benefit (Calmar improvement) is conditional on the asset having
mid-position vol spikes that the rest of the strategy stack
(regime gate, ensemble, trend filter) doesn't already catch. For BTC
in this window the rest of the stack already does that work; the
wrapper is redundant and dilutes the upside.

## What this means for the wider conclusion

Going back to the OOS Sharpe gap from the previous step:

* Walk-forward OOS Sharpes on BTC: 0.5-0.7.
* Adding vol-targeting to BTC: 0.51-0.69 — **no improvement**.
* Walk-forward OOS Sharpes on ETH: probably similarly 0.5-0.6.
* Adding vol-targeting to ETH: 0.49-0.63 — **a real, modest improvement**.

So vol-targeting **does not close the IS/OOS gap** in any meaningful
sense. The IS/OOS shrinkage from the previous step is a robust
phenomenon, not a "we forgot vol-targeting" artifact.

## Recommendations

1. **Quote BTC and ETH numbers separately.** Aggregating across the
   two assets hides a real qualitative difference. The "right" vol
   target is asset-conditional.
2. **Don't bolt vol-targeting onto BTC if your underlying strategy
   has a regime gate.** It costs more upside than it saves DD on this
   asset in this window.
3. **Do bolt vol-targeting onto ETH** (and presumably higher-vol alts
   — to be confirmed on the PIT universe). target=0.30 to 0.50
   improves Calmar by 0.05-0.30.
4. **Never quote a single "vol-targeting wins" or "loses" claim**
   without specifying asset and time period. The literature's general
   statement is too coarse for retail-scale crypto application.

## Sanity checks

If anyone tweaks the wrapper, these must still hold:

1. `wrapped_signal == 0` everywhere the inner signal is 0. The
   wrapper cannot synthesize exposure out of nothing.
   (Tested.)
2. `wrapped_signal <= max_position_size`. Never lever above the cap.
   (Tested with both vanishing-vol and high-vol synthetic series.)
3. No look-ahead: appending future garbage candles cannot change the
   prefix output. (Tested.)
4. Engine still applies the one-bar shift to the wrapper's signal.
   (Tested via `run_backtest` integration.)
5. `tsmom(use_vol_target=False)` returns the raw `{0, 1/k, ..., 1}`
   ensemble ladder. (Tested.)

## Reproducing

```bash
python -c "
import pandas as pd
from trade_lab.backtest import (
    run_strategy_walk_forward, aggregate_walk_forward, ParamGridSpec,
)
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy
from trade_lab.strategies.vol_target_wrapper import VolatilityTargetWrapper

for asset in ['BTC', 'ETH']:
    candles = pd.read_parquet(f'data/binance_{asset}_USDT_1d.parquet')
    for label, factory, warmup in [
        ('raw',   lambda: TimeSeriesMomentumStrategy(use_vol_target=False), 365),
        ('vol30', lambda: VolatilityTargetWrapper(
            TimeSeriesMomentumStrategy(use_vol_target=False), annual_vol_target=0.30), 400),
    ]:
        grid = [ParamGridSpec(label=label, factory=factory, warmup_days=warmup)]
        detail = run_strategy_walk_forward(candles, grid)
        summary = aggregate_walk_forward(detail)
        print(asset, label, summary['mean_test_sharpe'], summary['mean_test_calmar'])
"
```

## Next on the queue

Per the user's priority list — item #4: **DSR / PSR in the results
table** with explicit per-strategy `num_trials` count, plus an
"unconfirmed" flag for DSR < 0.5 strategies.
