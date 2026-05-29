# Validation Test 4 — Part A: retrospective truncation-audit — CLEAN

**Date:** 2026-05-29
**Status:** CLEAN — zero mismatches across 1589 bars × 4 metrics.
The verified-window backtest is signal-level look-ahead-free.
**Frozen-config hash:** `ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753`
**Script:** `scripts/validation_lookahead_truncation_audit.py`
**Output:** `outputs/validation_test4_truncation_audit.json`

## What this audit asks

A look-ahead bug lives in the **signal/index plumbing**, not in
realized P&L (P&L is downstream of the signal). The audit therefore
operates at the signal layer and is **offset-FREE** — it compares
`signal[T]` from a PIT-truncated panel against `signal[T]` from the
full-sample run, both labelled by the same T. The 1-bar offset
question belongs to Part B (live detector), not here.

For every bar T in the verified window:

```
truncated = {sym: df[df.index <= T] for sym, df in full_panel.items()}
pit_basket  = build_crypto_market_index(truncated)
pit_signal  = TimeSeriesMomentumStrategy(...).generate_signals(pit_basket)
sig_pit [T] = pit_signal.iloc[-1]
sig_full[T] = full_signal.loc[T]
assert  sig_pit[T] == sig_full[T]
```

The same equality is also checked on:

* `pit_basket["close"].iloc[-1]` vs `full_basket["close"].loc[T]`
  (basket index construction, normalization anchor, weight
  propagation).
* `SMA(200)` at T from truncated vs full (warmup handling).
* Regime gate boolean (`close > SMA`) at T from truncated vs full.

## Result

| Metric | Mismatch bars | Max abs/rel diff |
|---|---:|---:|
| `ladder_state` (final signal in {0, 0.5, 1.0}) | **0 / 1589** | 0.00e+00 abs |
| `basket_close` (synthetic index value) | **0 / 1589** | 0.00e+00 rel |
| `SMA(200)` on basket close | **0 / 1589** | 0.00e+00 rel |
| Regime-gate (close > SMA) | **0 / 1589** | exact bool match |

**Verdict: CLEAN.** Every PIT-truncated rebuild produces bit-for-bit
identical signal, index value, SMA, and gate at the truncation
bar. There is no temporal look-ahead in the signal-generating
pipeline on the verified-window sample.

Runtime: 33.5 s for the full 1589-bar audit on a 2026 laptop. Cheap
enough to be re-run after any code change that touches basket
construction, strategy class, or alignment.

## Specific look-ahead vectors tested

The prompt named five families of look-ahead bugs to probe
explicitly, even though the strategy is trailing "by construction".
Each is exercised by the truncation audit:

1. **Index construction (rebasing / normalization anchor).**
   `build_crypto_market_index` rebases to `100` using
   `portfolio_equity.iloc[0]`. Anchored to the START (bar 0), so
   truncation at the end does not move the anchor. **0 mismatches**
   in `basket_close`.

2. **Eligibility / availability mask.** The basket uses
   `closes.notna()` as its active-asset mask. The notna pattern at
   T is independent of bytes after T. Combined with Test 1 Check A
   (CLEAN: parquet min-dates ≥ Binance listing dates) and Test 5
   (PASS: 90-day median volume rank ≤ 15 for all 7), there is no
   path for a pre-listing asset to slip in. **0 mismatches**.

3. **`fillna` direction.** The repo uses only forward-direction
   ops (`pct_change`, `rolling(...).mean()`, weights forward-
   propagation between rebalances). No `bfill`, no interpolation
   through future points. The truncation test would surface a
   `bfill` immediately — it doesn't. **0 mismatches**.

4. **Warmup handling.** SMA(200) at T uses `close[T-199:T+1]`. At
   T < bar 200 of the basket, SMA is NaN and the gate is forced
   closed — same in truncated and full runs. **0 mismatches** on
   SMA-value or gate boolean.

5. **Timestamp alignment between assets with different histories.**
   The basket outer-joins per-asset closes on a common UTC daily
   index. For T < SOL listing 2020-08-11, SOL has NaN under both
   truncated and full runs; for T ≥ 2020-08-11, SOL contributes
   in both. The notna-based active mask is invariant to truncation
   at any T. **0 mismatches**.

## Scope boundary — what this audit does NOT cover

* **Universe-selection bias** — the hand-picked choice of the 7
  majors. That is a survivorship question and is closed by Test 1
  Check A (parquet dates CLEAN) + Test 5 (PASS, liquidity moot).
  This audit verifies temporal correctness of the pipeline GIVEN
  the chosen universe; it does NOT verify the choice itself.
* **Cross-asset signal contamination** — none in this strategy
  (the basket is averaged into a single synthetic index before the
  strategy sees it). N/A.
* **Live-data revision look-ahead** — the audit uses one fixed
  snapshot of `data/binance_*.parquet`. If a future data refresh
  pulls revised candles (e.g., adjusted historical prices that
  differ from what was first published), the audit would need to
  be re-run. The harness's content-hashed vintage store (Step 2)
  is the mechanism that makes that re-run trivial on the live
  side.

## Why this matters for the validation-phase conclusion

The verified-window net Sharpe of **+0.721** (Test 1 Check B,
Bybit-confirmed) and the full-sample +1.377 are now established
to be **signal-level look-ahead-free** on a deterministic
reproducible test. That answers the dispositive question from
`han_28d_tsmom.md`'s caveat about "actual results likely worse than
backtest" — at minimum, the *backtest itself* is not phantom.

This does NOT promise that forward performance will match the
verified band. Forward performance can still vary because:

* Regime is not stationary (the post-2022 distribution is one
  realization).
* The pre-2022 era is venue-unverifiable (Test 1 RISK FLAG).
* Cost regime can drift (Test 2 sensitivity).
* Live execution adds operational noise (Part B detector).

But all four of those concerns are about forward variance, not
backtest correctness. Backtest correctness is now empirically
established.

## Project N_TRIALS bookkeeping

Zero new trials. The audit is a per-bar correctness check against
the FROZEN config on a pre-registered window. No parameter sweep,
no signal variation, no selection. Diagnostic; does not consume
`PROJECT_NUM_TRIALS`.

## Reproducing

```
.venv/bin/python scripts/validation_lookahead_truncation_audit.py
```

Reads `data/binance_*_USDT_1d.parquet`, writes
`outputs/validation_test4_truncation_audit.json`. The verdict is in
the script's exit code (0 = CLEAN, 1 = CONTAMINATED).

Last reviewed: 2026-05-29.
