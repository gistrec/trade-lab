# Validation Test 5 — universe / availability bias — closed

**Date:** 2026-05-29
**Status:** PASS — both axes closed.
**Frozen-config hash:** `ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753`
**Decision:** the 7-major equal-weight basket on the verified window
(2022-01-21 → 2026-05-28) carries negligible universe-bias risk for
the deployable strategy. No new research cycle was opened; no
strategy parameter touched.

## What this test asks

Per validation prompt, two axes:

1. **Listing axis** — does the basket include any asset before its
   Binance listing date? (Would mean `closes.notna()` PIT proxy is
   contaminated.)
2. **Liquidity axis** — does any of the 7 fall below a reasonable
   liquidity threshold during the verified window? (At $1.4k per
   asset, liquidity is supposed to be moot for majors; verify, don't
   assume.)

## Axis 1 — listing — CLOSED via Check A in validation_multiexchange.md

| Asset | Parquet min | Binance listing | Δ days | Status |
|---|---|---|---:|---|
| BTC  | 2018-01-01 | 2017-08-17 | +137 | CLEAN (truncated) |
| ETH  | 2018-01-01 | 2017-08-17 | +137 | CLEAN (truncated) |
| BNB  | 2018-01-01 | 2017-11-06 | +56  | CLEAN (truncated) |
| SOL  | 2020-08-11 | 2020-08-11 | 0    | CLEAN |
| ADA  | 2018-04-17 | 2018-04-17 | 0    | CLEAN |
| XRP  | 2018-05-04 | 2018-05-04 | 0    | CLEAN |
| DOGE | 2019-07-05 | 2019-07-05 | 0    | CLEAN |

**Listing axis verdict: CLEAN.** Δ ≥ 0 days for every asset. The first
three rows are truncations (parquet starts later than listing — the
backtest does NOT include the late-2017 mania), not contamination.
For this 7-asset universe `closes.notna()` is empirically equivalent
to `tradable_at(date, meta)`, so the basket's PIT proxy is correct
for the deployable era.

Structural caveat (carries forward to any future cross-sectional
work): the equivalence is *coincidental*. The basket-index code still
uses `closes.notna()`; if a future data refresh ever pulls SOL/etc
prices from CoinMetrics/CoinGecko (which can predate the Binance
listing) into a file named `binance_*`, the index would silently
re-pollute. Not blocking; flagged for posterity.

## Axis 2 — liquidity — closed on verified window

The 7-asset basket's worst case at $10k portfolio is **~$1.43k per
asset on a full long entry**. The question is whether that notional
can be transacted without measurable slippage beyond what Test 2
already models.

### Empirical check: 90-day median volume rank vs the full COIN_REGISTRY pool

Ran `build_pit_universe(market_caps, volumes, top_n=20,
volume_lookback_days=90)` on the full ~40-coin registry pool over
the verified window. For each of the 7 majors, recorded:

| Asset | 90-day median volume rank (verified window, vs 43-coin pool) |
|---|---|
| BTC | rank 1 (always) |
| ETH | rank 2 (always) |
| SOL | rank 3-5 (always) |
| XRP | rank 4 (always) |
| ADA | rank 10 (always) |
| DOGE | rank 7 (always) |
| **BNB** | **rank 4–9 (median 6)** — all bars |

Every major is **inside the top 15 by 90-day median USD volume** on
every bar in the verified window. The lowest median 90-day volume on
the verified window belongs to BNB at ~$265M/day; **$1,430 notional
on $265M of daily median volume is 5 × 10⁻⁶ — six orders of
magnitude below any reasonable liquidity concern.**

**Liquidity axis verdict: MOOT at deployment size.** No major drops
out of top-20 by volume rank during the verified window, and the
notional is many orders of magnitude below liquidity-relevant levels.

### Sub-finding: `build_pit_universe` falsely flags BNB ineligible (data caveat)

The composite `build_pit_universe` mask **does** report BNB as
ineligible on every bar of the verified window. Decomposed:

| BNB component | Value on 2024-01-15 |
|---|---|
| `tradable_at(date, meta)` | True (Binance pair active) |
| 90-day median volume | $276M |
| volume rank vs pool | **8** (well inside top-20) |
| `market_cap` from CoinMetrics community cache | **NaN** |
| market-cap rank | 27 (NaN → bottom) |
| Final eligibility | **False** (mcap rank > 20) |

**This is a data-availability artifact, not a liquidity-failure.**
CoinMetrics' community tier does not expose `CapMrktEstUSD` for BNB
in this cache; the panel column is all-NaN. `build_pit_universe`
ranks NaN to bottom by `na_option="bottom"`, so BNB is mathematically
excluded — even though its real-world market cap was ~$50B+
throughout the verified window (which would place it ~rank 4-5 by
mcap, easily inside top-20).

Implications:

* **Does NOT affect the deployable 7-major basket.** The basket is
  hand-picked (CLAUDE.md "Deployable strategy") and includes BNB
  unconditionally; `build_pit_universe` is not in the strategy
  execution path.
* **Does affect any future cross-sectional rotation strategy** that
  derives its universe from `build_pit_universe` against the
  community-tier cache. Such strategies would silently drop BNB.
  Worth a data-quality patch when paid CoinMetrics access is
  added, or a sentinel that raises when an asset has `mcap = NaN`
  but `tradable_at = True` (currently the code drops it silently —
  exactly the "fail loud" violation CLAUDE.md warns against).

## Aggregate verdict

Both axes pass for the deployable frozen basket:

* **Listing:** CLEAN (delta ≥ 0 days for all 7).
* **Liquidity:** MOOT at $1.4k/asset; all 7 in top-15 by volume
  always.
* **Sub-caveat:** `build_pit_universe` has a community-tier data
  artifact on BNB that does not affect the frozen basket but would
  affect any future cross-sectional rotation strategy.

No availability bias material to the validation phase. Test 5
closed.

## Project N_TRIALS bookkeeping

Zero new trials. Confirmatory diagnostic against the frozen config;
no parameter sweep, no signal optimization, no composition change.
Per validation rules, this does not consume `PROJECT_NUM_TRIALS`.

## Reproducing

Inline check via the project's own `load_panel` + `build_pit_universe`
on the registry cache in `data/coinmetrics/*.parquet`. No standalone
script created — the check is short and is summarized above
verbatim.

Last reviewed: 2026-05-29.
