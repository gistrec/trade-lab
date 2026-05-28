# Finding — Breadth filter is a substitute for SMA200, not a complement

**Status:** weak / mixed evidence, asset-conditional behaviour, modest
improvement on ETH only.

## Finding

Adding a "breadth-of-universe above own SMA(200)" gate on top of the
existing single-asset SMA(200) regime filter:

* **Does NOT improve** the market-basket TSMOM (already at DSR 0.658).
  The basket close above its own SMA encodes most of the breadth
  information already.
* **Mixed per-asset**: helps ETH meaningfully (DSR 0.28 → 0.45 with
  breadth ≥ 30%), hurts BTC (DSR 0.31 → 0.13), neutral on SOL/XRP/
  DOGE.
* As a *replacement* for SMA(200), breadth ≥ 30% is essentially
  identical to SMA(200) on the basket (DSR 0.662 vs 0.658).

## Headline results

### Market-basket TSMOM × breadth thresholds

| Variant                                          | Concat OOS Sharpe | DSR @ 500 | Time invested |
|--------------------------------------------------|------------------:|----------:|--------------:|
| basket tsmom (SMA200 only)                       |   +1.36           |  **0.658** | 51%          |
| basket tsmom + breadth ≥ 30% (in addition)       |   +1.34           |   0.639    | 51%          |
| basket tsmom + breadth ≥ 50% (in addition)       |   +1.31           |   0.608    | 47%          |
| basket tsmom + breadth ≥ 70% (in addition)       |   +1.27           |   0.570    | 39%          |
| basket tsmom (no regime filter)                  |   +1.32           |   0.618    | 80%          |
| basket tsmom + breadth ≥ 30% (replacing SMA200)  |   +1.36           |  **0.662** | 53%          |
| basket tsmom + breadth ≥ 50% (replacing SMA200)  |   +1.30           |   0.601    | 48%          |

Reading: stacking breadth ON TOP of SMA200 strictly hurts DSR (filter
becomes too restrictive). Replacing SMA200 with breadth ≥ 30% gives
essentially the same result.

### Per-asset TSMOM × breadth (in addition to SMA200)

| Asset | Baseline (SMA200) DSR | + Breadth ≥ 30% DSR | + Breadth ≥ 50% DSR |
|-------|----------------------:|--------------------:|--------------------:|
| BTC   |   0.312               |   0.195             |   0.125             |
| **ETH**   |   0.280           |   **0.446**         |   0.371             |
| SOL   |   0.075               |   0.073             |   0.061             |
| XRP   |   0.010               |   0.011             |   0.021             |
| DOGE  |   0.002               |   0.004             |   0.002             |

**ETH is the only asset where breadth meaningfully helps**, jumping
from 0.28 to 0.45 — still UNCONFIRMED but a 60% relative improvement.
BTC degrades by ~60% in the opposite direction.

## Why the basket result is a wash

On the basket, the price series is already the breadth-weighted
aggregate. "Basket close > basket SMA(200)" essentially asks "is the
breadth of the basket above 50%?" — a slightly different breadth
question (cap-weighted vs equal-weighted), but the two questions
correlate strongly enough that adding both is double-counting. The
breadth gate just over-filters.

## Why ETH benefits but BTC doesn't

Hypothesis: BTC has dominant beta in the universe (it carries the
basket on its back during the regime arcs we're looking at).
Adding a breadth gate to BTC mostly subtracts profitable BTC-only
rallies where the rest of the universe was still washed out. ETH,
by contrast, often led narrative-driven moves (DeFi summer, ETH 2.0,
L2 rollups) where the broader alt set was also confirming. For ETH,
breadth confirmation is value-added; for BTC, it's a beta drag.

This is **the same asset-conditional pattern** we found for vol-
targeting in `findings/vol_targeting_regime_gate.md`: BTC's
peculiar role as crypto's macro instrument means single-instrument
filters work fine for it, while additional confirmations help
narrative-driven alts.

## Implication

* **Don't stack breadth on top of SMA200 on the basket.** Pick one.
  Default to SMA200 since it's cheaper to compute and equally
  effective.
* **Consider breadth-instead-of-SMA200 only if you don't have a
  market-basket already** — the substitution is roughly a wash on
  the basket.
* **For per-asset ETH** specifically, breadth ≥ 30% is a meaningful
  upgrade over plain SMA200 (DSR +0.17). For BTC, do not add it.
  For the rest, the noise dominates.
* The `GatedStrategy` wrapper is **kept in the public API** as a
  generic composition tool — the breadth use case is just one
  example. Future users may want to gate strategies on macro
  signals, custom indicators, or external risk meters that don't
  exist yet.

## Caveats

* **N=21 (per-asset) + 7 (basket) configurations tested.** Adds to
  PROJECT_NUM_TRIALS — counted in the 350-buffer.
* **The 7-asset universe is the same hand-picked basket as everywhere
  else.** Breadth on a true PIT universe would behave differently
  (more historical assets, different SMA cross-overs during 2018-2019
  small-cap rallies).
* **Threshold = 30% was an arbitrary low value.** We did not optimise
  for the best threshold per asset; doing so would just shift the
  selection bias from "best variant per fold" to "best threshold per
  asset", which the DSR formula doesn't catch unless we count it as
  a separate trial.

## What this means for the project queue

Breadth was an "easy win if the literature was right" check. It
wasn't. Skipping breadth in the deployable config is the honest
move; revisiting it for ETH-specific tuning is a paper-trading-era
optimisation, not a backtest-era one.
