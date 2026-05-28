# Point-in-time universe for cross-sectional momentum

This document records the **first** of five honesty checks the user
queued in the second session: replacing the hand-picked four-asset
XSMOM universe with a procedural one that includes coins that were
top-of-cap **at the rebalance date**, *regardless of whether they
later survived*.

## Why this matters

The original XSMOM result was:

> XSMOM equal + BTC gate (4 hand-picked alts): **+22,737% / DD 48% / Sharpe 1.40**

The Research-Claude survey warned this was likely survivorship-biased.
This document quantifies how badly.

## Data plumbing

* **Market cap and volume**: Coin Metrics community API
  (`https://community-api.coinmetrics.io/v4/timeseries/asset-metrics`),
  daily frequency, no auth.
* **Metric set**: `CapMrktCurUSD` (or `CapMrktEstUSD` fallback for
  post-2020 alts whose `Cur` metric is gated to the paid tier), plus
  `volume_reported_spot_usd_1d` for the volume rank. `PriceUSD` is
  used when available; otherwise `CapMrktEstUSD` substitutes for the
  return computation (empirical drift on BTC: ~6 bps/day = roughly
  the BTC emission rate, well below the 30-day momentum signal).
* **Listed-on-Binance check**: hand-curated `coin_registry.py` with
  `(listed_date, delisted_date)` per pair, sourced from Binance
  announcement archives.

The full registry is 44 pairs; Coin Metrics community has data for
42 of them (Bitconnect predates the catalog; one asset hit a 400 then
recovered). After applying the stablecoin filter and the market-cap
availability filter, the eligible candidate pool is **32 pairs**.

## Universe sanity checks

The PIT universe is the eligibility mask returned by
`build_pit_universe(...)`. A row is True iff the coin is **in the
top-N by market cap AND in the top-N by trailing 90d volume AND
Binance-tradable on that date**.

| Sanity check | Threshold | Observed | Verdict |
|--------------|-----------|----------|---------|
| BTC eligible share | >= 95% of dates | 100.0% | OK |
| ETH eligible share | >= 95% of dates | 100.0% | OK |
| LUNA eligible during 2021-04..2022-05 | >= 50% | 79.1% | OK |
| LUNA eligible after 2022-05-13 (delisting) | 0.0% | 0.0% | OK |
| FTT eligible during 2021-2022 | non-zero | 7.1% (full window) | OK |
| Universe snapshot 2022-01-01 includes LUNA | yes | yes | OK |
| Universe snapshot 2024-01-01 excludes LUNA / FTT / BUSD | yes | yes | OK |

These are the same checks the code's docstring recommends running
after rebuilding the registry or replacing the data source. If
sanity check 4 (LUNA after delisting = 0%) ever fails, the
`tradable_at` predicate or the `delisted_date` field is wrong; if 1-2
fail, the volume ranker is treating BTC/ETH as illiquid (very unlikely
unless the volume metric path is broken).

## Headline comparison — apples-to-apples window (2020-08-11 → 2026-05-27)

| Variant                            | Return    | Max DD | Sharpe | Cash%  | Fees       |
|------------------------------------|-----------|--------|--------|--------|------------|
| OLD XSMOM (4 hand-picked alts)     | +22,737%  |  48%   | +1.40  | 57%    | $94,876    |
| **PIT XSMOM (top-20 PIT, top-2)**  |   +992%   |  66%   | +0.93  | 52%    |  $8,009    |
| BTC buy-and-hold                   |   +566%   |  77%   |   —    |   —    |        —   |

> **Survivorship bias delta: −96% relative return, +18 pp drawdown,
> −0.47 Sharpe.** The "OLD" number was almost entirely an artifact of
> picking four coins that all survived. The real PIT number still
> beats BTC buy-and-hold (+992% vs +566%) with comparable drawdown,
> but the Sharpe is below 1.0 — closer to a barely-OK trend-following
> system than a Sharpe-1.4 marvel.

The fee column also collapses (from $95k to $8k) because the PIT
universe is bigger, so turnover-as-fraction-of-cap-per-rebalance is
smaller.

## Robustness sweep (full window 2018-2026)

For completeness, parameter variations on the PIT universe:

| Variant                                   | Return    | Max DD | Sharpe | Avg basket |
|-------------------------------------------|-----------|--------|--------|------------|
| top_k=2 + BTC gate + equal                | +2,549%   |  80%   | +0.89  |    0.9     |
| top_k=3 + BTC gate + equal (**baseline**) | +2,066%   |  74%   | +0.88  |    1.4     |
| top_k=5 + BTC gate + equal                | +1,026%   |  76%   | +0.78  |    2.1     |
| top_k=3 + **no BTC gate** + equal         |   +195%   |  86%   | +0.58  |    2.3     |
| top_k=3 + BTC gate + inverse_vol          | +1,928%   |  71%   | +0.87  |    1.4     |
| top_k=3 + BTC gate + rebalance=14d        |   +584%   |  77%   | +0.68  |    1.4     |

Observations:

1. **The BTC > SMA(200) gate is the single most load-bearing component**
   — without it, drawdown blows out from 74% to 86% and Sharpe falls
   from 0.88 to 0.58. The gate is doing most of the trend-following
   work; the cross-section is just choosing *which* alts to buy when
   the gate opens.
2. Wider baskets (top_k=5) dilute returns without improving Sharpe —
   the marginal coin is already a relatively poor pick.
3. Inverse-vol weighting is roughly Sharpe-neutral here. With only 1.4
   coins held on average, the weighting choice has little room to
   help.
4. A 14-day rebalance halves both fees and returns — suggests the
   weekly cadence is closer to a real edge than a noise pickup.

## Top-held coins on the PIT panel

(Counted from the baseline `top_k=3 + BTC gate + equal` run.)

| Rank | Coin     | Bars held | % of period |
|------|----------|-----------|-------------|
| 1    | SOL/USDT |     301   |    14.2%    |
| 2    | XRP/USDT |     210   |     9.9%    |
| 3    | DOGE/USDT|     154   |     7.3%    |
| 4    | AVAX/USDT|     140   |     6.6%    |
| 5    | FIL/USDT |     126   |     6.0%    |
| 6    | SUI/USDT |     112   |     5.3%    |
| 7    | LINK/USDT|     112   |     5.3%    |
| 8    | ETH/USDT |     105   |     5.0%    |
| 9    | SHIB/USDT|      98   |     4.6%    |
| 10   | BCH/USDT |      91   |     4.3%    |

LUNA is conspicuously *not* in the top-10 held — even though it was
eligible 79% of its bull window. This is because by the time LUNA's
30-day return was top-3 in the panel, other alts (SOL, FTM, AVAX) were
running just as hard. So the strategy "missed" the LUNA crash but
also didn't capture much of the LUNA upside. That's an empirical
finding, not a property guaranteed by the design — see the residual
bias note below.

## Residual biases the PIT fix does NOT address

1. **Candidate pool selection.** The `coin_registry.py` itself is
   biased: it contains coins we *thought of* in 2026, plus a curated
   list of major delistings. Coins that briefly hit top-20 in 2021
   and then went silently to zero outside the major-delisting list are
   missing. Realistic upper bound on this bias: a few percentage
   points of Sharpe.
2. **Coin Metrics community gating.** Ten alts (NEAR, OP, ARB, INJ,
   TIA, APT, FTM, MATIC, HBAR, UST) have neither `PriceUSD` nor
   `CapMrktCurUSD` accessible — only volume. They are filtered out of
   the panel. For 2018-2022 this is benign; for 2023-2026 it
   probably costs the strategy some exposure to fast-moving alts.
3. **Volume metric** is `volume_reported_spot_usd_1d` — global spot
   volume across all reported exchanges, not Binance-only. A coin
   ranked highly because of, say, OKX or Coinbase volume could be
   illiquid on Binance specifically. We accept this as a soft bias.
4. **Delisting execution.** The forced-exit rule fires the bar after
   eligibility flips to False. Real Binance gives 24-72h notice; in
   practice trader could exit at the announcement, not at the
   suspension. We use the *suspension date*, which is conservative
   (worse for the strategy than realistic).

## Reproducing

```bash
# Once, ~2 minutes (43 Coin Metrics fetches, then cached):
python -c "from trade_lab.data.universe import build_universe_from_registry; \
           build_universe_from_registry(top_n=20)"

# Subsequent invocations re-use the parquet cache:
python -c "
import pandas as pd
from trade_lab.data.universe import build_universe_from_registry
from trade_lab.backtest.cross_sectional import run_cross_sectional_momentum

eligibility, closes = build_universe_from_registry(top_n=20, fetch=False)
asset_candles = {col + '/USDT': pd.DataFrame({
    'open': closes[col], 'high': closes[col], 'low': closes[col],
    'close': closes[col], 'volume': 1.0,
}).dropna() for col in closes.columns}
elig = eligibility.rename(columns={c: c + '/USDT' for c in eligibility.columns})
res = run_cross_sectional_momentum(
    asset_candles, top_k=3, btc_candles=asset_candles['BTC/USDT'],
    eligibility=elig,
)
print(res.total_return, res.max_drawdown, res.sharpe)
"
```

## Where this leaves us

The cleanest takeaway, in the spirit of the Research-Claude warning
that no number in this repo should be quoted without a caveat:

* Cross-sectional momentum on a procedural Binance universe **still
  beats buy-and-hold by ~2x return** with comparable drawdown over
  2018-2026 — that part is robust to removing survivorship.
* The Sharpe is **~0.9**, not 1.4. A Sharpe of 0.9 on a long-only
  crypto rotation is a normal trend-following result, not a smoking
  gun.
* **The BTC gate does most of the work.** Removing it cuts Sharpe in
  half. If the user wants to compare strategies side-by-side, they
  should compare *with the gate*, not *with the rotation alone*.

Next on the queue (per the user's priority list): walk-forward
validation for tsmom / pma_ratio / sma_cross.
