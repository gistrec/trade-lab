# `cross_sectional_momentum` — Top-N rotation across a universe

Cross-sectional momentum (CSMOM) is a *portfolio-level* strategy: each
week, rank the universe by trailing N-day return and hold the top-K.
The Research-Claude survey lists CSMOM at priority 5/5 citing:

- Liu, Tsyvinski, Wu (2022). *Common Risk Factors in Cryptocurrency*.
  **Journal of Finance** 77(2):1133-1177. Identifies market / size /
  momentum as the three factors that explain the cross-section of
  cryptocurrency returns and reports ten significant long-short
  strategies.
- Tzouvanas, Kizys, Tsend-Ayush (2019). *Momentum trading in
  cryptocurrencies: short-term returns and diversification benefits*.
- Starkiller Capital — practical reproduction with a 15-35 day lookback
  and a 7-day rebalance.

Because the strategy makes a portfolio-level decision over multiple
assets simultaneously, it does not fit the single-asset `Strategy`
interface. It lives in `trade_lab/backtest/cross_sectional.py` and is
called directly via `run_cross_sectional_momentum(...)`.

## Rules

1. **Universe.** The caller supplies a `dict` mapping symbol →
   `OHLCV DataFrame`. All assets are outer-joined onto a common date
   index; missed candles inside each asset's listed history are
   forward-filled, but we never invent pre-listing prices.

2. **Trailing-return ranking.** Every `rebalance_days` (default 7),
   compute `close.pct_change(lookback_days)` per asset (default
   30-day lookback). The decision uses only closes through the
   rebalance date.

3. **Selection.** Keep only assets with **positive** trailing return;
   from those, take the top `top_k` (default 3).

4. **Optional BTC regime gate.** Pass `btc_candles=` to add a
   `BTC > SMA(btc_gate_sma_period)` filter (default 200). When the
   gate is closed, the portfolio sits in cash that week regardless of
   the cross-section.

5. **Weighting.**
   - `equal` (default): `1/len(basket)` per selected asset.
   - `inverse_vol`: weight by `1 / realized_vol(vol_lookback)`,
     normalized to sum to 1. Empty / zero vol assets fall back to
     equal weighting within the basket.

6. **Execution.** Target weights from rebalance date `N` apply to bar
   `N+1` onward, exactly like the single-asset engine. Costs are
   `(fee_rate + slippage_rate) * turnover` summed across assets.

## Why long-only

The Liu-Tsyvinski-Wu (2022) original is long-short. We strip the short
leg because spot Binance does not provide it; the long-only retail
analogue is what the Research-Claude survey calls out as priority 5/5.
Expect lower Sharpe than the academic figure (which is a long-short
factor return).

## Output

`CrossSectionalResult` carries:

| Field                      | Description |
|---------------------------|-------------|
| `equity`                  | Portfolio equity curve. |
| `returns`                 | Net per-bar returns (after fees + slippage). |
| `weights`                 | DataFrame: rows = dates, cols = assets, values = held weights ∈ [0, 1]. |
| `rebalance_dates`         | Dates where the weight vector changed. |
| `total_return`            | Final / initial – 1. |
| `max_drawdown`            | Peak-to-trough on equity. |
| `sharpe`                  | Annualized Sharpe (using `annualization_factor`). |
| `num_rebalances`          | Count of weight-vector changes. |
| `average_basket_size`     | Mean number of held assets per bar. |
| `average_cash_fraction`   | Mean `1 - sum(weights)` per bar. |
| `total_fees`, `total_slippage` | Dollar costs. |

## Important caveats

- **Survivorship bias.** A real-world CSMOM in 2020 would have included
  LUNA, FTT, AAVE/early DeFi, etc. — several went to zero or near-zero.
  Running on a hand-picked surviving universe (BTC/ETH/BNB/SOL) gives
  a flattering result.
- **Small universe.** The literature suggests 10-30 coins. With four,
  top-2 selection is highly concentrated; the result is more like
  *"rotation between two of four"* than a cross-section.
- **Fees.** Weekly rebalance × small basket can compound costs quickly
  on a small account; see `total_fees` in the result.
- **No vol scaling at portfolio level.** Each asset's vol affects the
  inverse-vol weights, but the *total* exposure is whatever weights
  sum to. Pair this with `weighting='inverse_vol'` to dampen the
  portfolio vol implicitly.

See `docs/results/strategy_comparison.md` for the side-by-side results.
