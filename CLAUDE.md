# CLAUDE.md

Permanent project rules for `trade-lab`. Read first in every session.

## Project context
Python crypto-spot backtester transitioning into paper trading. Solo developer in Serbia, test capital <$10k. Real money target is **Binance mainnet** (the earlier "Binance is geo-blocked in Serbia, real money goes to Kraken" assumption was wrong — corrected by the user 2026-07-09; the Kraken migration plan is cancelled).

## Current phase
execution-layer #2b is complete (production daily CLI `paper-place-orders` with deterministic clientOrderId idempotency, reconstruction, wait-for-ack, schema v2 journal). Mainnet support shipped 2026-07-09: testnet and mainnet run side by side (symmetric `.env.testnet` (default) / `.env.mainnet` (via `--env-file`), per-environment journal/state with runtime isolation guards, dashboard source switcher, three-flag order gate). Now: mainnet **read-only observation** (6-hourly dry-run cron, API key without trading permission) — mainnet has full kline history, so this validates the SMA(200)/ladder signal path that testnet structurally cannot (testnet wipes candles ~monthly). Live mainnet orders come only after that observation plus a capped smoke test. See `src/trade_lab/execution/README.md` § "Mainnet (real money)" for the rollout ladder.

## Deployable strategy
TSMOM lookbacks `(28, 60)` on an **equal-weight market-basket index** of 7 assets: BTC, ETH, BNB, SOL, ADA, XRP, DOGE. SMA(200) regime gate on the basket close. Pro-rata ladder signal `{0, 0.5, 1.0}` (average of binary per-lookback states). Basket rebalances monthly (`freq="MS"`) or when `N_active` changes. DSR 0.77, cluster-stable (7/7 neighbor configurations pass DSR > 0.5). Low turnover by design.

## Hard rules (never break without explicit permission)
- **Exchange is the single source of truth.** Never cache balance or positions across cycles — always round-trip to CCXT.
- **Execution must replicate the backtest exactly.** No "improvements." Do not round the ladder into binary, do not swap the basket for a 2-asset proxy, do not turn monthly rebalance into continuous. Weights drift between rebalances by design. Do not "fix" inter-rebalance drift by adding continuous rebalancing — this is a different turnover profile than the backtest measured. Any "simplification" is a different untested strategy.
- **No API keys in code.** Only env files (`.env.testnet` / `.env.mainnet` for paper trading, `.env` for everything else — all in `.gitignore`). `__repr__` of any config object must mask credentials.
- **Mainnet requires a three-flag gate.** `TRADE_LAB_PAPER_SANDBOX=false` + `TRADE_LAB_PAPER_ALLOW_MAINNET=true` unlock read paths only (paper-status, paper-dry-run); placing real orders additionally requires `TRADE_LAB_PAPER_MAINNET_LIVE_ORDERS=true`. Never lower this barrier. The live-orders flag on a sandbox config must raise (copy-paste guard).
- **Live mainnet orders follow the rollout ladder, not a flag flip.** Read-only observation (key without trading permission) → capped smoke test (≤ 25 USDT) → daily live cron. Do not skip steps. Enabling the live cron and re-enabling the mainnet daily health check belong in the same commit.
- **Testnet and mainnet never share files.** Symmetric env files `.env.testnet` (default for paper commands) / `.env.mainnet` (explicit `--env-file`; a missing file is a hard error, no fallback to a legacy `.env`), separate journals (`cycles.jsonl` / `cycles_mainnet.jsonl`), separate state (`orders.json` / `orders_mainnet.json`). The runtime guards (`assert_journal_env`, order-state env stamp) enforce this — never weaken them: `clientOrderId` has no environment component, so a shared state file can silently suppress a real mainnet placement.
- **Keep execution exchange-agnostic via CCXT.** Kraken has no CCXT sandbox. The combination `EXCHANGE=kraken` AND `SANDBOX=true` must raise explicitly, not silently proceed (CCXT will ignore `set_sandbox_mode` otherwise).
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
- **Ask** before: architectural decisions, changing pinned parameters (`PROJECT_NUM_TRIALS`, basket composition, SMA gate period, ladder semantics), adding or relaxing failure modes, anything that touches the three-flag mainnet gate, the environment-isolation guards, or the rollout ladder.
- **Act** on: bug fixes, new tests, extending existing modules along an already-agreed plan, behavior-preserving refactors.
- **Ask** when uncertain which category applies.

Last updated: 2026-07-09. Update when phase changes or hard rules are revised.
