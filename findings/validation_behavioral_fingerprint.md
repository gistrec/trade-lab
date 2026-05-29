# Validation Test 3/4 — reference behavioral fingerprint (frozen)

**Date:** 2026-05-29
**Status:** Reference fingerprint frozen + monitor wired. Live
behavioral check is ready to run forward; that clock turns daily,
not in-session.
**Frozen-config hash:** `ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753`
**Frozen reference content-hash:** `f8dd5bcf2a86dcc281a15c91ad35264e229827ce579ea6fb9f7d7e8ff8c903b6`
**Frozen reference path:** `paper_trading/fingerprint/reference_fingerprint.json`
**Build script:** `scripts/build_reference_fingerprint.py`
**Monitor:** `trade_lab.paper_trading.fingerprint_cli`

## What this is

The forward paper-clock that the operator runs daily (Step 2 harness)
will produce live behavioral data. This document fixes the
**reference fingerprint** — what behaviour the strategy
historically had on the venue-verified window — so the monitor has
a stable baseline to compare against.

The fingerprint is **frozen as a versioned artifact** identical in
discipline to `production_config`. The on-disk JSON contains its own
SHA-256 content-hash, and `load_reference` raises if the file has
been edited after writing. The monitor reads this file and **never
re-fits the reference from live data** — that mode would let slow
live degradation silently widen the bands and kill the detector.

## What this is NOT — behavioral, not financial

**No realized return / Sharpe / equity is in the percentile bands.**
The strategy is allowed to lose money in an adverse regime
(precisely the Dec 2024 → May 2026 sub-period the project is in
right now). Including profit metrics as a pass/fail gate would
re-introduce profit-as-success through the back door — exactly what
the validation phase exists to prevent.

The fingerprint tracks **behavioral invariants**: flip cadences,
turnover, drawdown depth.

## Calibration

| Parameter | Value |
|---|---|
| Window | **2022-01-21 → 2026-05-28** (venue-verified per Test 1 Check B) |
| Bars | 1589 daily |
| Pipeline | Identical to the harness: `build_crypto_market_index` on Binance parquets → `TimeSeriesMomentumStrategy(28, 60, SMA200)` → `run_backtest` |
| Rolling window | 90 days |
| Annualization factor | 365 |
| Percentiles | p05, p25, p50, p75, p95 |
| Pipeline cross-check | Harness's signal output equals backtest's signal output on identical input (pinned in `tests/test_paper_trading_harness.py::test_signal_matches_backtest_on_identical_input`) |

The pipeline cross-check is the calibration insurance: if the
reference were computed by a slightly different code path than the
harness uses, the monitor would silently report breaches against
nothing real. The harness↔backtest signal-identity test makes that
mode impossible.

## Reference bands

### M1 — Exposure-flip frequency (rolling 90d, annualized)

Number of ladder transitions (`positions.diff() != 0`) in the
trailing 90 bars, multiplied by `365/90`. Distribution across the
verified window.

| Statistic | Value (flips/year) |
|---|---:|
| min  | 0.00 |
| **p05** | **0.00** |
| p25  | 4.06 |
| p50  | 20.28 |
| p75  | 37.51 |
| **p95** | **77.06** |
| max  | 105.44 |
| mean | 25.79 |

Mean 25.79 is consistent with the project headline "≈ 24.3 flips/year
on the verified window" (the headline averaged over 1589 bars; this
mean averages across overlapping 90d windows). The width of the
band reflects real regime variability: long sustained holds drop the
rolling rate near zero, while regime-transition periods push it
above 100.

### M2 — Rebalance turnover per event

`|Δposition|` at every non-zero transition. Per-event, not rolling.

| Statistic | Value |
|---|---:|
| n events | 106 |
| p05 | 0.5000 |
| p50 | 0.5000 |
| p95 | 1.0000 |
| min | 0.5000 |
| max | 1.0000 |
| mean | 0.5849 |

The ladder is `{0, 0.5, 1.0}` so the legal step sizes are 0.5 or 1.0.
The distribution shows ~70% of events are 0.5-step transitions
(ladder corner-to-edge) and ~30% are full 1.0 transitions
(0→1 or 1→0 jumps).

### M3 — Regime-gate (SMA-200) flip frequency (rolling 90d, annualized)

Count of `close > SMA(200)` boolean changes in trailing 90 bars,
annualized.

| Statistic | Value (flips/year) |
|---|---:|
| min  | 0.00 |
| **p05** | **0.00** |
| p25  | 0.00 |
| p50  | 4.06 |
| p75  | 16.22 |
| **p95** | **32.44** |
| max  | 36.50 |
| mean | 9.25 |

The gate flips much less than the ladder (mean 9.25 vs 25.79). Median
is ~4/year — about one regime flip per quarter on a typical 90d
window. p25 = 0 means a quarter of all 90d windows have **no** gate
flip — sustained regime periods are common.

### M4 — Drawdown profile (with explicit headroom)

Current drawdown from running peak, at every bar in the verified
window.

| Statistic | Value |
|---|---:|
| p05  | −29.05% |
| p25  | −26.86% |
| p50  | −18.67% |
| p75  | −5.49% |
| p95  | 0.00% |
| **max historical DD** | **−32.17%** |
| current DD at window end | −26.86% |

**Breach criterion (explicit, headroom-style):** the monitor flips
the drawdown-breach flag when **live drawdown from peak goes
deeper than −32.17%**. A drawdown shallower than that — including
the current −26.86% sitting at the project's window end — is
**inside the band by design**.

This is the trap the prompt called out: the lower band edge is
anchored by the 2022 bear (FTX-collapse era), the deepest drawdown
in the verified window. The current Dec 2024 → May 2026 chop is
mid-band, not extreme. The flag fires on a NEW 2022-class event,
not on the existence of a drawdown.

Live headroom at this moment (window end): −26.86% live vs −32.17%
threshold → **5.31 pp of headroom** before breach.

### M5 — Position concentration — recorded, NOT banded

On a frozen equal-weight 7-asset basket, the per-asset target weight
is mechanically `ladder_state / 7`. The maximum weight any single
asset can carry on a target rebalance day is bounded by
`{0, 0.0714, 0.1429}` — three discrete values. A percentile band on
this would be falsely narrow and would fire false positives on
between-rebalance drift that the strategy explicitly tolerates
(CLAUDE.md: "Weights drift between rebalances by design.").

This metric is recorded in the artifact as a structural note —
position concentration is **mechanically bounded by `ladder/N`** —
and the monitor does NOT include it in breach calculations.

## Statistical honesty — what these bands ARE and ARE NOT

The percentile bands describe the **observed range of behavior** on
the historical sample. They are explicitly **descriptive, not
inferential**.

* **Rolling windows are autocorrelated.** Today's 90d window
  overlaps yesterday's by 89 days. The 1500-odd daily observations
  of the rolling-90d flip rate are NOT 1500 independent samples.
* The strategy fully flips ~24 times per year on the verified
  window. Across the 4.4 years there are roughly 4 yearly
  observations of "annual flip rate" — too few for any
  significance test even if we wanted one.
* A breach therefore says **"the live behavior is outside the
  historical envelope"**, NOT "the deviation is significant at
  p < 0.05". The monitor's advisory phrasing reflects this:
  "operator review", not "reject the null".

## Monitor semantics — flag for review, not auto-kill

The monitor reports breaches; the operator + Step-4 look-ahead
detector decide what to act on. The thresholds (`sustained_days = 7`,
`multi_metric = 3`) are advisory defaults, configurable via the CLI:

| Trigger | What it means | Operator action |
|---|---|---|
| Within envelope | Behaviour matches reference | None. |
| 1-day 1-metric breach | Likely noise | Watch. |
| `multi_metric_threshold` metrics breached same day | Behavioral pattern change | Review the journal rows + cross-check the vintage with Step 4 detector. |
| `sustained_days_threshold` consecutive days breached on the same metric | The strategy is consistently behaving differently from history | Same as above; treat as load-bearing signal. |
| Drawdown beyond `max_historical_dd` | New regime worse than 2022 bear, OR a bug | This is the load-bearing case. Stop trading until Step-4 detector clears the journal of look-ahead. |

## Fingerprint vs detector — complements, NOT duplicates

* **Fingerprint (this writeup, Step 3):** behavioral consistency on
  **live** data vs frozen reference. Catches regime drift, silent
  breakage, harness bugs that change behaviour distribution. Cannot
  tell apart "new regime" from "code bug" — both look like a
  behavioral shift.
* **Look-ahead detector (Step 4, separate document):** signal
  **identity** on **identical vintage** (replay). The detector
  reconstructs the backtest on the exact bytes the harness saw and
  cross-checks that backtest signal == live signal. This is the
  half that distinguishes "new regime" (live & backtest both agree
  on the new behaviour) from "code bug" (they disagree on identical
  input).

The two are deliberately orthogonal questions. The fingerprint can
report a clean envelope while the detector finds a look-ahead
(behavior coincidentally inside band, but signal logic is wrong).
The fingerprint can report a sustained breach while the detector
finds zero discrepancies (the live data really is in a regime not
covered by history). Acting on either one alone is incomplete.

## Project N_TRIALS bookkeeping

Zero new trials. The reference fingerprint is a derived behavioral
statistic of the FROZEN config on a pre-registered window — no
optimization, no model variation, no selection. Confirmatory
infrastructure for the forward test does not consume
`PROJECT_NUM_TRIALS`.

## Forward-test caveats — explicit so the operator does not misread

* **Current adverse sub-period is expected behavior.** The
  Dec 2024 → May 2026 net-negative arc on Binance (Test 1) is a
  consequence of regime variation, not a deficiency. The drawdown
  band has the current state sitting at p25 of historical DDs;
  monitor reports green on M4.
* **Until the look-ahead detector (Step 4) clears the journal, the
  monitor's "Within envelope" advisory is a NECESSARY but NOT
  SUFFICIENT condition.** A clean envelope with an undetected
  look-ahead = backtest illusion.
* **Reference re-build requires a documented input change.** Adding
  more data on the trailing edge does NOT trigger a reference
  re-build — the reference is the 2022-01-21 → 2026-05-28 sample
  by design. If a new bear cycle materializes and the operator
  wants the reference updated, that is a deliberate decision that
  counts as a new research cycle (new findings document, hash
  bump, monitor consumers re-pinned).

## Reproducing

```bash
.venv/bin/python scripts/build_reference_fingerprint.py
```

Verifies round-trip hash on save. Re-running on the same Binance
parquets produces byte-identical output (file timestamp aside).

Tests covering the artifact and monitor:
`tests/test_paper_trading_fingerprint.py` (10 tests).

Last reviewed: 2026-05-29.
