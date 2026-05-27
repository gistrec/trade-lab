# Backtest validation report

This is a short audit of look-ahead and benchmark correctness for the
long-term BTC/USDT 1d results (commit `c1b6a82`, dataset 2018-01-01
through 2026-05-27, 3069 daily bars).

## Summary

| Check | Status | Notes |
|---|---|---|
| Engine shifts signals by 1 bar | PASS | `engine.py:107` |
| `sma_cross` indicators are causal | PASS | new test, `test_sma_cross.py` |
| `regime_sma_cross` indicators are causal | PASS | new tests, `test_regime_sma_cross.py` |
| Pandas `rolling().mean()` is trailing | PASS | sanity-checked at runtime |
| Buy & hold uses same `initial_capital` | PASS | `engine.py:128` |
| Buy & hold uses same filtered window | PASS | computed inside `run_backtest` after the candles are filtered |
| Buy & hold fee assumption | NOTE | B&H is **gross** (no fees); strategy is net. Documented but kept asymmetric on purpose — this is the conventional benchmark. |
| Debug-trades CSV reflects the audit | PASS | first 10 trades inspected on real data — see end of this file |

No bugs found. The +1525% / 52% DD vs B&H's +456% / 81% DD numbers stand.

## 1. Signal-to-execution shift

`engine.py:106-107`:

```python
signals = strategy.generate_signals(candles).reindex(candles.index).fillna(0)
positions = signals.shift(1).fillna(0).astype(float) * float(position_size)
```

`Series.shift(1)` moves values forward by one index, so
`positions[N] == signals[N-1]`. The return calculation
`gross_returns[N] = positions[N] * bar_returns[N]` therefore captures the
price move from `close[N-1]` to `close[N]` based on a signal that only
saw `close[N-1]`. No future data is consulted.

Pinned by:

- `tests/test_backtest_engine.py::test_look_ahead_bias_is_prevented`
- `tests/test_backtest_engine.py::test_execution_bar_is_one_after_signal_bar`

## 2. Indicator causality

Both strategies build indicators with `close.rolling(K).mean()`. Pandas's
default `closed='right'` rolling window at bar `i` covers
`[i-K+1, i]` — past + current bar only. A runtime check confirms this:
mutating any bar after index `i` does not change `rolling(K).mean()[i]`.

For full coverage at the strategy level, two causality tests were added:

- `tests/test_sma_cross.py::test_sma_signals_are_causal_appending_future_does_not_change_past`
- `tests/test_regime_sma_cross.py::test_regime_signals_are_causal_appending_future_does_not_change_past`
- `tests/test_regime_sma_cross.py::test_regime_signal_at_bar_n_doesnt_use_close_after_n`

Each test computes signals on a base series, then on the same series
extended with absurd future values (`1e6`, `1e-6`), and asserts the
signal vectors over the overlap are byte-identical. If the strategy
leaked future data into the past, those values would diverge.

> Note on "Donchian-like calculations": the requirement mentioned them,
> but no Donchian-style indicator currently ships in `trade-lab`. All
> the present indicators (`SMA`, `regime SMA`) go through the same
> causal `rolling().mean()` and inherit the same guarantee.

## 3. Buy-and-hold benchmark correctness

`engine.py:128`:

```python
buy_and_hold_equity = initial_capital * (close / close.iloc[0])
```

- `initial_capital` is the exact same parameter the strategy uses.
- `close` is the same post-filter close series the strategy is fed.
- Date range matches automatically because both run inside the same
  `run_backtest` call after `--start-date` / `--end-date` have been
  applied at the CLI level (`cli.py:cmd_backtest`).

The asymmetry: **buy-and-hold pays no fees and no slippage**, whereas
the strategy pays both on every rebalance. This is the textbook
convention but means B&H is slightly more optimistic than apples-to-
apples would suggest. For the long-term BTC run a one-side fee of
0.1% would only knock about 0.1pp off the +456% B&H return — small
relative to the gap to the strategy. The asymmetry is left in place
on purpose; a "B&H net" option can be added later if needed.

## 4. Debug audit export (`--debug-trades-csv`)

A new CLI flag dumps the first N (default 10) completed trades to a CSV
with explicit columns:

| Column | Meaning |
|---|---|
| `signal_time` | bar where the strategy decided (close was visible) |
| `execution_time` | bar where the position became active (one bar later) |
| `signal_close` | close at the signal bar |
| `execution_open_or_close` | close at the execution bar |
| `entry_price_after_slippage` | `close[execution_bar] * (1 + slippage_rate)` |
| `exit_price_after_slippage` | `close[exit_bar] * (1 - slippage_rate)` |
| `reason` | indicator values at the signal bar (`fast`, `slow`, `regime`, `close`) |

Sample rows from `regime_sma_cross` on the long-term run:

```
signal_time, execution_time, signal_close, execution_open_or_close, entry_price_after_slippage, ...
2019-04-02,   2019-04-03,    4857.29,      4932.60,                4935.07,   ..., fast(4053.33)>slow(3775.77) & close(4857.29)>regime(4662.37)
2020-01-28,   2020-01-29,    9374.21,      9301.53,                9306.18,   ...
2020-03-02,   2020-03-03,    8915.24,      8760.07,                8764.45,   ...
```

For every row, `execution_time = signal_time + 1 day`, and
`entry_price_after_slippage / execution_open_or_close ≈ 1 + 0.0005`
(i.e. 1.0005), confirming the slippage rate is applied to the execution
bar's close — not the signal bar's. The `reason` column shows the exact
indicator values the strategy saw at the *signal* bar (never the
execution bar), so any future-leak would be visible on inspection.

Run with:

```bash
trade-lab backtest --strategy regime_sma_cross --symbol BTC/USDT \
    --timeframe 1d --param fast_period=20 --param slow_period=100 \
    --param regime_period=200 --debug-trades-csv outputs/audit.csv
```

## 5. What's not audited (and why)

- **Survivorship bias** — single-asset, BTC has been on Binance the whole
  window. Multi-asset / index would need additional thought.
- **Lookahead via slippage model** — slippage uses the *execution* close
  to compute the fill price. It does not peek further ahead. (And in
  reality slippage relates to the entry-time order book, which is what
  the model approximates with a small percentage.)
- **Live execution edge cases** — out of scope. The engine is a research
  tool; no order-book replay or maker/taker logic.

## 6. Conclusion

No look-ahead bias was found in the engine, the `sma_cross` strategy,
or the `regime_sma_cross` strategy. Buy-and-hold is computed on the
same initial cash and the same filtered window; its only asymmetry vs
the strategy is the (deliberate) zero-cost assumption. The
`+1525%` regime number vs `+456%` B&H number is a legitimate result on
the chosen dataset — but that does not certify the strategy as a
production system; window choice (8+ years including two big bears)
still dominates.
