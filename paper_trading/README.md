# `paper_trading/` — Validation forward-test harness

This directory is the **operational layout** for the validation phase
forward-test harness (validation Tests 3 + 4). The harness *code*
lives at `src/trade_lab/paper_trading/`; this directory holds the
README, the append-only journal, and the immutable content-hashed
vintage snapshots that the harness produces.

## What this is (and what it is NOT)

* **It IS** a daily script that records what the FROZEN strategy
  (`TSMOM(28, 60) + SMA(200)` on the 7-major basket) WOULD do, plus
  a byte-exact snapshot of the OHLCV data it saw on the day it
  decided. No real money. No real orders.
* **It IS NOT** the production execution layer
  (`src/trade_lab/execution/*` / `paper-place-orders` CLI), which
  places real orders on Binance testnet. That stays untouched.

The harness's whole purpose is to make the look-ahead detector
(Test 4) *possible*: replay the backtest against the exact bytes the
harness saw on day T, sanity-check that backtest signal == live
signal on identical data. If they diverge, a look-ahead exists in
the backtest path.

## Hard contract — what makes this useful

1. **Frozen-config hash gate.** The harness reads
   `PRODUCTION_CONFIG` + `CANONICAL_HASH` from
   `src/trade_lab/config/`. If they drift (someone bumps a parameter
   without going through a research-cycle write-up), the harness
   **refuses to run** and surfaces the mismatch. The forward test is
   meaningful only if the strategy under test is the same for the
   whole horizon.
2. **Immutable content-hashed vintage.** Every cycle writes a
   physically separate copy of the OHLCV bytes it used, named after
   the SHA-256 of those bytes (`vintages/{ab}/{abcdef...}.txt`). On
   replay the bytes are verified to still hash to the filename — no
   silent revision, no shared mutable store.
3. **Append-only journal.** One JSONL row per UTC date in
   `logs/journal.jsonl`. Rows are never edited; the journal is
   strict history.
4. **Idempotent.** Re-running the cron command within the same UTC
   day is a no-op (returns the previously-written row). Safe to
   schedule belt-and-suspenders.

## Running the harness

From the repo root, after installing the project into `.venv`:

```bash
.venv/bin/python -m trade_lab.paper_trading.cli
```

Exit codes:
* `0` — wrote a new row, or returned the existing one for today.
* `2` — `HarnessError` (config drift, fetch failure, empty basket).
  The cron job should surface this for human review; do NOT
  blind-retry — fail-loud is the design.

### Scheduling daily

A minimal `crontab(5)` entry that runs once a day at 00:30 UTC
(after the prior day's close has settled):

```cron
30 0 * * *  cd /home/user/trade-lab && .venv/bin/python -m trade_lab.paper_trading.cli >> paper_trading/logs/cron.out 2>&1
```

Hands-on operators can run interactively to debug:

```bash
.venv/bin/python -m trade_lab.paper_trading.cli --asof 2026-05-29
```

### Optional CLI flags

* `--log-path` (default `paper_trading/logs/journal.jsonl`).
* `--vintage-root` (default `paper_trading/vintages`).
* `--asof YYYY-MM-DD` (default: today UTC). The journal row is keyed
  by the last **completed** daily bar: `asof` itself for a past date
  (backfill of a missed cron day), yesterday for a same-day run —
  today's bar is still forming and never participates, and a bar
  after `asof` (e.g. the completed next-day bar during a backfill)
  trips a hard look-ahead guard.
* `--candles-per-asset` (default 400; ≥ 200 needed for SMA(200) warmup).

## Files in this directory

* `README.md` — this document.
* `logs/journal.jsonl` — append-only structured journal (gitignored).
* `logs/cron.out` — optional cron stdout/stderr capture (gitignored).
* `vintages/{xx}/{hash}.txt` — content-hashed OHLCV snapshots
  (gitignored). The two-level layout keeps any single directory
  from growing past a few hundred files even after years of cycles.

## Journal row schema (v1)

One JSON object per line. Field reference:

| Field | Type | Meaning |
|---|---|---|
| `date` | str | ISO UTC date of the cycle |
| `config_hash` | str | `CANONICAL_HASH` at write time (anti-drift) |
| `vintage_content_hash` | str | SHA-256 of OHLCV bytes used |
| `basket_close` | float | basket index close at as-of |
| `sma_value` | float \| null | SMA(200) of basket close |
| `sma_gate_open` | bool | `basket_close > sma_value` |
| `ladder_state` | float | TSMOM signal in `{0.0, 0.5, 1.0}` |
| `prior_ladder_state` | float | yesterday's ladder (0 on bootstrap) |
| `per_lookback_states` | obj | `{"28": 0|1, "60": 0|1}` |
| `per_lookback_returns` | obj | `{"28": pct, "60": pct}` |
| `target_weights` | obj | `{asset: 1/N × ladder}` |
| `current_weights` | obj | prior held weights |
| `intended_trades` | obj | `target - current` per asset |
| `portfolio_equity` | float | virtual USD equity start of cycle |
| `daily_return` | float | basket pct_change since prior cycle |
| `gross_position_return` | float | `prior_ladder × daily_return` |
| `net_position_return` | float | gross minus simulated turnover cost |
| `notes` | str | optional free-text annotation |

### `date` field semantics — anchor for Step 4 detector

The journal's ``date`` field is the **signal date**: the date of the
most recent OHLCV bar (close) used by the strategy to compute
``ladder_state``. In bar-indexed terms: when ``date = T``,
``ladder_state`` is the strategy's output computed from data through
the close of bar T. The intended_trades carried by this row are the
position changes that would be **placed at the open of bar T+1** to
achieve the new target weights (mirroring the backtest engine's
``signal.shift(1)`` convention).

Why this matters: the look-ahead detector (Test 4) replays the
backtest against the vintage data and compares the backtest's
signal[T] against this row's ``ladder_state``. They must match
exactly on identical input. A constant-1-bar offset (every live
signal equals the backtest signal one bar earlier) would mean the
two paths disagree on the convention for ``date`` — that is a
**labeling artifact**, not a real look-ahead, and the detector
should be able to recognize it as such by testing both alignments.
This note is the anchor for that test.

## Frozen reference fingerprint

The reference behavioral fingerprint lives at
``paper_trading/fingerprint/reference_fingerprint.json`` and is a
**versioned frozen artifact**, like the production config. The file
is hash-pinned (the JSON contains its own SHA-256 in the
``content_hash`` field; ``load_reference`` verifies on read).

Rebuilding the reference is a one-time operation:

```bash
.venv/bin/python scripts/build_reference_fingerprint.py
```

The script reads Binance parquets from ``data/`` and writes the
fingerprint. Re-running on the same inputs produces a byte-identical
file; a hash change indicates an input changed.

## Behavioral monitor

```bash
.venv/bin/python -m trade_lab.paper_trading.fingerprint_cli
```

Reports whether live journal behaviour sits inside the reference
bands. **Descriptive only — never auto-kills.** Exit codes:

* `0` — report produced. Default even on breach: descriptive, not
  normative.
* `1` — breach detected (drawdown / sustained / multi-metric) AND
  the opt-in `--fail-on-breach` flag was passed. For cron wrappers
  that alert on non-zero exit.
* `2` — tool error (missing reference, content-hash mismatch, etc.).

The monitor's advisory levels (in increasing seriousness):

* `Within historical envelope` — green.
* `Single-day single-metric breach` — noise.
* `Bootstrap` — fewer than the rolling-window-length of journal
  rows; rolling metrics not yet evaluable.
* `Multi-metric breach` — ≥ 2 of the three daily metrics outside band
  simultaneously on the same day (3 would demand unanimity). Operator
  review.
* `Sustained breach on a behavioral metric` — same metric breached
  for ≥ 7 consecutive days. Operator review.
* `DRAWDOWN BREACH` — live drawdown deeper than the worst observed in
  the reference window (2022 bear). Forward to Step-4 look-ahead
  detector + operator review.

Known limitation: the flip-frequency bands (M1/M3) have `p05 = 0.00`,
so a strategy that goes dormant (flips stop entirely) never breaches
the lower band — dormancy is undetectable by construction.

## Execution tracking (real vs simulation)

Third verification layer (issue #11), next to the fingerprint monitor
and the look-ahead detector: reconciles **real mainnet execution**
with the **simulated forward test**.

```bash
.venv/bin/python -m trade_lab.paper_trading.execution_tracking_cli
```

Reads the mainnet execution journal (default
`data/journal/cycles_mainnet.jsonl`, override `--real-journal`) and
the harness journal (default `paper_trading/logs/journal.jsonl`,
override `--sim-journal`), aligns by signal date, and reports:

* **Equity tracking** — cumulative `sum |Δdaily-return|` plus the
  current level gap in % (both curves normalized to the first aligned
  date). Per date one consistent real cycle is sampled: the daily
  live cycle, falling back to the first dry-run of the date only when
  no live cycle exists (a later 6-hourly dry-run never overwrites the
  live observation). `--gap-threshold-pct` (default 5% — an
  owner-adjustable starting point, not a calibrated bound) sets the
  breach level.
* **Per-symbol trades** — expectations come from the HARNESS rows
  (`intended_trades`), never from the mainnet journal itself (an
  erroneous production signal and its own orders would match each
  other); mainnet supplies only the actual side. Mismatches are
  per-symbol: missing, wrong-direction, partial fill, unexpected.
  Only LIVE-cycle skips with a sub-minimum reason (min-notional /
  lot-step class from `delta.py`; not `pending_*`) may cover a
  missing trade — counted separately, never alerted on.

Exit codes — same contract as the fingerprint monitor:

* `0` — report produced. Default even on breach; an empty overlap
  window (both journals exist but share no dates yet) is a
  descriptive note, not an error. Unknown-schema-version lines in
  the mainnet journal degrade to an explicit incomplete-data warning.
* `1` — tracking threshold breached AND `--fail-on-breach` passed.
* `2` — tool error (missing journal file, unreadable path, corrupt
  mainnet journal lines — a malformed line can hold the very cycle
  under reconciliation — or a harness row that no longer matches the
  `HarnessLogRow` schema).

Daily cron, after the 00:05 UTC live order cycle and the harness run
(one line — crontab has no backslash line continuation; cron does not
`cd`, so pass absolute journal paths):

```cron
50 0 * * * cd /home/user/trade-lab && .venv/bin/python -m trade_lab.paper_trading.execution_tracking_cli --real-journal /home/user/trade-lab/data/journal/cycles_mainnet.jsonl --sim-journal /home/user/trade-lab/paper_trading/logs/journal.jsonl --fail-on-breach >> paper_trading/logs/tracking-cron.out 2>&1
```

## Anti-patterns — DO NOT do these

* **Do NOT edit `journal.jsonl` in place.** The look-ahead detector
  reads it as immutable history. If a row is wrong, write a new
  cycle with corrected notes; never rewrite.
* **Do NOT delete or rename vintage snapshots.** The
  `vintage_content_hash` in the journal points to those files; the
  detector verifies the bytes hash to the filename before using
  them.
* **Do NOT pull "today's prices" from a different data source than
  the harness used.** That defeats the entire look-ahead detector.
* **Do NOT lower the hash gate.** If a config change is intentional,
  open a `findings/` document, count it as a new research cycle,
  re-run walk-forward + DSR, then update both `CANONICAL_HASH` and
  the test pin. The gate is the contract.

## Interpreting the journal

**Until the look-ahead detector (Test 4) runs, the journal is just
data accumulation, not evidence.** Reading day-by-day equity changes
before the detector is ready is structurally inadequate — the
detector is what tells you whether the live signal == backtest
signal on identical data. A green run before the detector is set
up does not validate anything.

The behavioral fingerprint (Test 3, separate writeup) calibrates
percentile bands against the **post-2022 distribution** (NOT
full-sample); live behavior is "in band" when it lives inside those
bands. The current Dec 2024 → May 2026 sub-period is net-negative
on every venue (see `findings/validation_multiexchange.md`) — a red
month is **expected** in that regime, not a signal of failure.

Honest forward-deployment Sharpe expectation: **~0.46 (bear) … 0.90
(bull), centre 0.72**; full-sample 1.38 is venue-unverifiable and
must NOT be the live anchor.
