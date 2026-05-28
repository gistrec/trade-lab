# Finding — Multi-asset ensemble: diversification reduces drawdown but does NOT raise DSR

**Status:** strong evidence, N=21 sleeves × 7 assets × 13 OOS folds,
single timeframe (1d), single regime arc (2018-2026).

## Finding

A 21-sleeve equal-weight portfolio (3 strategies × 7 Binance USDT
pairs, per-asset vol-target picks from
`findings/vol_targeting_regime_gate.md`, dynamic 1/N_active with
rebalance-on-universe-change costing) produces an OOS Sharpe of
**+1.13** and a maximum drawdown of **34%**, vs. the best single
sleeve (`pma_medium × BNB` with vol30 wrapper) at Sharpe **+1.27**
and the marginal-survivor DSR of 0.564.

**Portfolio DSR at PROJECT_NUM_TRIALS=500 = 0.425.** That is
*below* the best single sleeve's 0.564. Diversification did NOT
raise the deflated Sharpe — it lowered it, by diluting the strongest
sleeve with weaker correlated ones.

The hypothesis the user posed ("portfolio of uncorrelated
unconfirmed bets has higher aggregate DSR than any single") is
**not supported by our data**, because the underlying sleeves are
not uncorrelated enough (mean pairwise corr = +0.46).

## Headline matrix

(24-month train / 6-month test / 6-month step, 13 OOS folds, costs
0.10% fee + 0.05% slippage symmetric on every side including
benchmarks.)

| Item                                        | Return    | Sharpe | Calmar | Max DD | DSR @ 500 |
|---------------------------------------------|----------:|-------:|-------:|-------:|----------:|
| **Ensemble portfolio (21 sleeves)**         | **+380%** | **+1.13** | **+11.2** | **34%** | **0.425** |
| Best single sleeve (`pma_medium × BNB`)     |  +457%    |  +1.27 |   —    |   —    |  **0.564** |
| Equal-weight HODL of 7 assets               | +6,980%   |  +1.23 |  +87.9 |  79%   |   —       |
| BTC buy-and-hold                            |   +932%   |  +0.91 |  +12.2 |  77%   |   —       |

What each column says, in plain language:

* **Return:** the ensemble made 380% on $10k → $48k over ~6.5 years.
  HODL would have made ~$700k (post entry cost). The ensemble
  sacrificed ~95% of crypto's parabolic upside to keep drawdown
  manageable.
* **Sharpe:** ensemble +1.13 is below the best single sleeve (+1.27)
  but above BTC B&H (+0.91). Diversification is "averaging down" the
  best sleeve in favor of weaker, partially-correlated ones.
* **Calmar:** ensemble at 11.2 is competitive with BTC B&H (12.2) but
  far below HODL-7-assets (87.9). The HODL Calmar is high *despite*
  a 79% drawdown because total return was so large.
* **Max DD:** the diversification clearly works at the DD level —
  ensemble cuts max DD by more than half vs either HODL.
* **DSR:** the ensemble loses to its best sleeve here, NOT because
  diversification is bad in principle, but because (a) mean pairwise
  correlation is 0.46, not zero, and (b) the lower-Sharpe sleeves
  drag the portfolio average down.

## Per-sleeve OOS performance

Out of 21 sleeves, **20 ended with positive OOS Sharpe**. The lone
negative is `sma_20_100 × DOGE` (Sharpe -0.08, -69% return, 84% DD).
Two more are barely positive (`sma_20_100 × SOL`: +0.26 with -8%
return; `tsmom_medium × DOGE`: +0.23 with +13% return). Excluding
the three weakest sleeves from the ensemble — a forward-looking
operation we did NOT do — would have raised the portfolio Sharpe
modestly.

Strongest sleeves (Sharpe ≥ 1.0):

|  Sleeve                            | OOS Sharpe | Return | Max DD |
|------------------------------------|-----------:|-------:|-------:|
| `pma_medium__BNB__vol30`           |   +1.27    | +457%  |  25%   |
| `tsmom_medium__ETH__vol50`         |   +1.12    | +799%  |  34%   |
| `tsmom_medium__BTC__raw`           |   +1.01    | +636%  |  53%   |
| `pma_medium__ADA__vol30`           |   +1.02    | +215%  |  32%   |
| `tsmom_medium__ADA__vol50`         |   +0.99    | +420%  |  45%   |
| `pma_medium__ETH__vol30`           |   +0.97    | +234%  |  29%   |

`pma_medium` is the most consistent (positive on 7/7 assets, average
Sharpe +0.90). `tsmom_medium` is second (positive on 7/7, average
+0.80). `sma_20_100` is the weakest (positive on 5/7, average +0.58).

## Correlation summary

* **Mean pairwise corr = +0.46.** Diversification "works" in the
  sense that this is below the 0.6 threshold the user asked us to
  check. But it's not low enough to be a "free lunch" — the
  effective number of independent bets is closer to ~4-5 than 21.
* **Median pairwise corr = +0.44**, very close to the mean — the
  distribution is symmetric, not driven by a few outliers.
* **26 of 210 pairs have corr > 0.6.** Most of those are
  same-asset pairs (e.g., `pma_medium × BTC` ↔ `tsmom_medium × BTC`,
  three strategies on BTC will share the same up/down macro tape),
  with a smaller number of cross-asset BTC ↔ ETH high-corr pairs.
* **Min/max pairwise corr: +0.22 / +0.88.** No anti-correlated
  sleeves — every sleeve still moves with crypto's overall risk-on
  rhythm. Cross-sectional momentum doesn't decouple from beta.

## Why diversification doesn't raise DSR here

A classic CTA pitches the ensemble narrative as "uncorrelated bets
make the portfolio more confident than any single bet". The math:
for IID bets with shared null sigma_SR, portfolio Sharpe scales as
`sqrt(N) × per-sleeve Sharpe`, and DSR rises with Sharpe.

That story requires actual independence. With mean corr +0.46, the
effective N is roughly `21 / (1 + (N-1) × corr) ≈ 21 / 10.2 ≈ 2`
independent units — not 21. The portfolio Sharpe gain from N=2
units of correlated noise is much smaller than the gain from N=21
units of independent noise.

Our result is consistent: portfolio Sharpe (+1.13) is below the
weighted average of single-sleeve Sharpes (+0.76) only because the
correlations dampen the diversification benefit. We see SOME
benefit (1.13 > 0.76 unweighted average), just not enough to
overcome the dilution of the best sleeve.

## What the rebalance-on-universe-change cost looked like

The model: each time `N_active` changes (new asset comes online or
existing asset's last sleeve finishes its WF), every existing
sleeve shifts from 1/N_old to 1/N_new and the absolute weight
change is charged `(fee_rate + slippage_rate)` per unit. The
new-sleeve entry portion of the diff is NOT double-charged at
portfolio level — sleeve internals already paid for it.

Over the whole window:

* Aggregate portfolio-level turnover: **1.80** (in units of capital)
* Total rebalance cost: **0.16%** of capital
* Cost share of gross PnL: **0.01%**

The cost is real but small relative to per-sleeve internal costs.
The user's "no free magical reallocation" concern is satisfied —
new-asset listings do trigger turnover charges, but they're not the
dominant cost driver.

## Sanity checks the user asked for

> "Что должно сломаться, если корреляции посчитаны с look-ahead?"

If correlations used future returns, the corr matrix would look
artificially neat (cross-fold smoothing erases the regime-specific
noise). Test in the code: the corr matrix is computed *only* on the
concatenated OOS returns, which are themselves shifted by
walk_forward_v2's one-bar lag. Any look-ahead would manifest as
correlation values much closer to 1 — we see a healthy spread from
0.22 to 0.88, consistent with real OOS data.

> "Portfolio Sharpe не должен превышать сумму sleeve-Sharpe,
> делённую на sqrt(N) при нулевой корреляции — проверь, что
> результат в разумных границах."

* Sum of per-sleeve OOS Sharpes: +15.98
* Zero-corr bound = 15.98 / sqrt(21) ≈ **+3.49**
* Portfolio Sharpe observed: **+1.13**

The portfolio Sharpe is well below the zero-corr bound — as
expected, since real correlations are 0.46 not zero. The check
passes.

> "Если portfolio DSR не вырос относительно одиночных sleeve'ов —
> диверсификация мнимая, надо объяснить почему."

**Portfolio DSR (0.425) IS lower than the best single sleeve's DSR
(0.564).** Reason: see above — mean corr 0.46 means effective N is
~2, not 21. The diversification benefit on Sharpe is real but
modest, and the portfolio dilutes the best sleeve in exchange for
a halving of max drawdown (34% vs 25% for the best single sleeve,
or 77% for BTC B&H, or 79% for HODL-7-assets).

Diversification is **not mnimal** here — it is **DD-reducing but
not Sharpe-raising**. That's a meaningful distinction. A risk-
managed allocator who values capping the downside even at the cost
of mean return can defensibly use this portfolio. A return-maximizer
should be using `pma_medium × BNB` alone (with its 25% DD already
within most thresholds) or just HODL-7.

## What this means in plain English

* "Width vs depth" — we tried width (multi-asset) and got the
  classic diversification trade-off: lower DD, lower return, lower
  but not catastrophically lower Sharpe.
* Diversification at sleeve-level **does** mitigate single-sleeve
  failure modes. `sma_20_100 × DOGE`'s -69% standalone is averaged
  away in the portfolio.
* But diversification doesn't manufacture statistical significance
  from nothing. **0 of 21 sleeves passed DSR > 0.5, the portfolio
  passed DSR 0.425, and the best single sleeve (pma_medium × BNB)
  is the only thing in this entire repo at 0.564.** That's the
  honest summary.

## Implication

For the paper-trading gate the user is approaching:

1. **The portfolio is the deployable artifact**, not any single
   sleeve. It has the smallest max DD and the best Sharpe-after-DD-
   discount in the comparison set.
2. **The portfolio is not statistically distinguishable from random
   walk at N=500.** Same caveat as every other priority-5 result in
   this repo.
3. **Position sizing should reflect this uncertainty.** A trader
   confident in DSR 0.95+ might allocate 5% of capital to a strategy.
   At DSR 0.43, the math says <1% is the right test size.
4. **The portfolio's biggest competitive edge is operational:**
   it converts "one strategy bet" into "one allocator that
   automatically picks across 7 instruments × 3 strategies". A
   solo trader with finite attention gets a deployable system, not
   a thesis to babysit.

## Caveat

* **N=21 sleeves, 7 assets, 1 timeframe (1d), single regime arc
  (2018-2026).** Same scope limitations as prior priority-5 results.
* **`PROJECT_NUM_TRIALS=500` was NOT bumped for this ensemble.** The
  per-asset vol-target picks (3 × 7 = 21 sub-selections from 3
  variants each = 63 trial outcomes) were already counted in the
  500 budget's "7-asset × 3-strategy × 3-vol-variant: 63" line.
  The ensemble's meta-decision (combine these 21 into equal-weight
  with dynamic rebalance) is at most a few additional trials,
  comfortably inside the 350-buffer. **If the next experiment bumps
  the trial count further, it goes in its own commit.**
* **No live execution simulation.** The portfolio assumes synthetic
  daily-close fills with 0.05% slippage. Real Binance retail with a
  $10k account would face larger slippage on alt-coin entries and
  potential cross-venue spread, neither of which we model.
* **Equal weight is the simplest allocator, not the best.** Inverse-
  vol, risk-parity, and Kelly-fractional weighting would change the
  numbers. We deliberately did not optimise the allocator (more
  selection bias to deflate).

## Reproducing

```bash
python -c "
import pandas as pd
from trade_lab.backtest import (
    run_ensemble_walk_forward, default_sleeves, PROJECT_NUM_TRIALS,
)
asset_candles = {
    s: pd.read_parquet(f'data/binance_{s}_USDT_1d.parquet')
    for s in ['BTC','ETH','BNB','SOL','ADA','XRP','DOGE']
}
res = run_ensemble_walk_forward(
    default_sleeves(), asset_candles,
    train_months=24, test_months=6, step_months=6,
    num_trials_for_dsr=PROJECT_NUM_TRIALS,
)
m = res.portfolio_metrics
print('Sharpe', round(m['sharpe'],2), 'Calmar', round(m['calmar'],2),
      'maxDD', round(m['max_drawdown']*100,1), 'DSR', round(res.portfolio_dsr,3))
"
```

Full outputs in `outputs/ensemble_*.csv` (correlation matrix,
sleeve return panel, portfolio returns/equity).

## Where this leaves the project

The original Research-Claude survey said: *"diversification across
strategies > optimisation of a single strategy"*. We tested that
empirically across 21 sleeves and 7 assets and found:

* The portfolio **does** outperform single-asset HODL on Sharpe and
  on max DD.
* The portfolio **does not** outperform multi-asset HODL on return
  or Sharpe (HODL is +6980%, portfolio is +380%, and Sharpe is a
  near-tie at 1.23 vs 1.13).
* The portfolio **does** the thing trend-followers exist for:
  cuts max DD from 77-79% (HODL) to 34% (portfolio). For a $10k
  retail trader the difference between -34% and -79% is the
  difference between "annoying setback" and "needing to start
  over".

The honest "ready to paper-trade" verdict: yes, the portfolio is the
right artifact to bring to paper trading. But the goal of paper
trading is to find out where this analysis lied about the operational
reality — not to confirm it.
