# CLAUDE.md

Permanent project rules for `trade-lab`. Read first in every session.

## Project context
Python crypto-spot backtester transitioning into paper trading. Solo developer in Serbia, test capital <$10k. Binance is testnet-only here; real money will go to Kraken.

## Current phase
execution-layer #2b is complete (production daily CLI `paper-place-orders` with deterministic clientOrderId idempotency, reconstruction, wait-for-ack, schema v2 journal). Next: extended testnet observation period (≥ 3-5 days of daily cron) to validate signal stability, cycle timing, partial-fill behaviour, and network-error recovery. See `execution/README.md` for current step detail.

## Deployable strategy
TSMOM lookbacks `(28, 60)` on an **equal-weight market-basket index** of 7 assets: BTC, ETH, BNB, SOL, ADA, XRP, DOGE. SMA(200) regime gate on the basket close. Pro-rata ladder signal `{0, 0.5, 1.0}` (average of binary per-lookback states). Basket rebalances monthly (`freq="MS"`) or when `N_active` changes. DSR 0.77, cluster-stable (7/7 neighbor configurations pass DSR > 0.5). Low turnover by design.

## Hard rules (never break without explicit permission)
- **Exchange is the single source of truth.** Never cache balance or positions across cycles — always round-trip to CCXT.
- **Execution must replicate the backtest exactly.** No "improvements." Do not round the ladder into binary, do not swap the basket for a 2-asset proxy, do not turn monthly rebalance into continuous. Weights drift between rebalances by design. Do not "fix" inter-rebalance drift by adding continuous rebalancing — this is a different turnover profile than the backtest measured. Any "simplification" is a different untested strategy.
- **No API keys in code.** Only env / `.env` (in `.gitignore`). `__repr__` of any config object must mask credentials.
- **Mainnet requires a two-flag gate.** `TRADE_LAB_PAPER_SANDBOX=false` AND `TRADE_LAB_PAPER_ALLOW_MAINNET=true`. Never lower this barrier.
- **Live orders only on testnet.** Mainnet migration is a manual process requiring Kraken account setup, KYC, exchange-specific market-constraint / fee validation, and a new code path — NOT a flag-flip on the existing Binance exchange config. Even when the two-flag gate above is satisfied, the project does not currently support mainnet order placement; promoting it requires a deliberate engineering decision and an additional review.
- **Binance is testnet only.** User is in Serbia; mainnet is unavailable. Real money will move to Kraken. Keep all execution code exchange-agnostic via CCXT. Kraken has no CCXT sandbox. The combination `EXCHANGE=kraken` AND `SANDBOX=true` must raise explicitly, not silently proceed (CCXT will ignore `set_sandbox_mode` otherwise).
- **`PROJECT_NUM_TRIALS = 500` is pinned.** Do not change it without a dedicated commit that explicitly acknowledges the shift and its implications for prior DSR numbers.
- **Never recommend live trading** until paper trading on the target exchange is complete.

## Failure handling principles
- **Fail loud, not silent.** Missing candles, missing prices, missing basket assets → raise. Do not fall back to defaults, do not shrink the basket silently.
- **Silent failure is worse than loud failure.** Silent basket shrinkage, silent partial fills, silent skipped orders are all forbidden. Each either raises or emits a structured log.
- **Skipped / divergent actions are first-class outputs.** Log with an explicit reason field and accumulate a cumulative metric (e.g., `total_skipped_quote_drift`).

## Methodological principles
- **Layered honesty.** IS Sharpe → OOS Sharpe → DSR with fixed N. Show each layer separately; never collapse into a single headline number.
- **Walk-forward is mandatory** before any edge claim. In-sample numbers are diagnostic, not result.
- **Cluster stability > single best point.** The deploy candidate is a family whose neighborhood median is high, not the single peak.
- **DSR is necessary but not sufficient** without walk-forward — and vice versa. Neither replaces the other.
- **Multiple testing budget is part of the result.** Every new parametric search adds to `PROJECT_NUM_TRIALS`. Pretending not to test variants does not erase them.

## Code quality rules
- **Tests are mandatory for new modules.** Include defensive tests that fail loudly on regression (e.g., stubs that omit dangerous methods so accidental calls raise immediately).
- **Mock the exchange in tests; never hit the live API.**
- **Idempotency via deterministic `clientOrderId`** — derived from the intent (rebalance date, symbol, side), not from the retry attempt.

## Where things live
- `src/trade_lab/strategies/` — strategy implementations; subclass `Strategy.generate_signals`.
- `src/trade_lab/backtest/` — engine, metrics, walk-forward, DSR, ensemble portfolio, market-basket index.
- `src/trade_lab/data/` — CCXT fetcher, parquet storage, Coin Metrics fetcher, PIT universe.
- `src/trade_lab/execution/` — paper / live execution: `config.py`, `broker.py`, `signal.py`, `allocator.py`, `delta.py`, `dry_run.py`, `journal.py`, `clientorder.py`, `order_state.py`, `orders.py`, `live_cycle.py`.
- `src/trade_lab/monitoring/` — read-only Streamlit dashboard for paper-trading observation (`app.py`, `data_source.py`). Never writes anything, no exchange access, no credentials.
- `findings/` — appendix-style writeups for completed experiments; treat as immutable history, not a current-state index.
- `docs/results/` — formatted results tables tied to specific commits.
- `tests/` — pytest suite covering backtest, strategy, execution, monitoring, and data layers.
- `paper.env.example` — template for paper-trading env vars.

## When to ask vs when to act
- **Ask** before: architectural decisions, changing pinned parameters (`PROJECT_NUM_TRIALS`, basket composition, SMA gate period, ladder semantics), adding or relaxing failure modes, anything that touches the two-flag mainnet gate or the mainnet-migration rule.
- **Act** on: bug fixes, new tests, extending existing modules along an already-agreed plan, behavior-preserving refactors.
- **Ask** when uncertain which category applies.

Last updated: 2026-05-29. Update when phase changes or hard rules are revised.
