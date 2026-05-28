# Execution layer

Paper-trading and live-execution layer for `trade-lab`. Exchange-
agnostic via [CCXT](https://github.com/ccxt/ccxt). Step #1 of the
build (this commit) wires up the connection layer + balance check;
subsequent steps add order execution, reconciliation logging, and
robustness primitives.

## Refuse-by-default to mainnet

This is the *single most important* safety property of this layer:

* `TRADE_LAB_PAPER_SANDBOX=true` → CCXT `set_sandbox_mode(True)` →
  testnet endpoints. **Safe.**
* `TRADE_LAB_PAPER_SANDBOX=false` AND `TRADE_LAB_PAPER_ALLOW_MAINNET=true`
  → mainnet endpoints. Requires both env flags set explicitly.
* Any other combination → `PaperConfigError` at load time. The bot
  refuses to start.

The two-flag requirement means flipping a single value cannot
accidentally point you at mainnet. Two independent conscious decisions
are required.

## Environment variables

See `paper.env.example` at the repo root. **Never commit a populated
`.env`.** The repo's `.gitignore` already excludes it; check before
pushing anyway.

## Quick start (Binance testnet)

```bash
# 1. Generate testnet API key/secret at https://testnet.binance.vision/
# 2. Add to .env:
#    TRADE_LAB_PAPER_EXCHANGE=binance
#    TRADE_LAB_PAPER_SANDBOX=true
#    TRADE_LAB_PAPER_API_KEY=...
#    TRADE_LAB_PAPER_API_SECRET=...
#
# 3. Verify connectivity + balances:
trade-lab paper-status
```

Switching to Kraken later (when you graduate from testnet to a real
exchange that operates in your jurisdiction):

```bash
# 1. Generate Kraken API key/secret at https://www.kraken.com/
# 2. Update .env:
#    TRADE_LAB_PAPER_EXCHANGE=kraken
#    TRADE_LAB_PAPER_SANDBOX=false
#    TRADE_LAB_PAPER_ALLOW_MAINNET=true
#    TRADE_LAB_PAPER_API_KEY=...
#    TRADE_LAB_PAPER_API_SECRET=...
# 3. Same command:
trade-lab paper-status
```

No code change. The whole point.

## What's built in step #1

* `config.py` — `PaperConfig` dataclass, `load_paper_config()`, strict
  env parsing, refuse-by-default to mainnet, repr that masks
  credentials.
* `broker.py` — `Broker` abstraction:
    * `Broker.connect(config)` — sets sandbox mode, opens CCXT session,
      verifies connection with `fetch_balance` round-trip.
    * `fetch_balance_snapshot()` — always live, never cached.
    * `fetch_ticker_price(symbol)` — last-or-close fallback.
    * `estimate_total_equity_usd(snapshot=None)` — mark-to-market.
* CLI: `trade-lab paper-status` — prints connection info, balances,
  mark-to-market equity.

## Failure modes (where this can silently break later)

These are the things I want to fix in subsequent steps. Logged here
so I don't forget.

1. **Two-flag race**: between `set_sandbox_mode(True)` and the first
   real call, the exchange object briefly holds mainnet URLs. CCXT's
   `set_sandbox_mode` is supposed to overwrite them before issuing
   any request, but a bug in a future CCXT version could flip this.
   Detection: the `_verify_connection` call uses `fetch_balance` —
   on testnet that would return a known testnet-style empty balance;
   on mainnet it would surface your live balance. **A future
   diagnostic could print the API URL the CCXT exchange resolved to
   and assert it contains "testnet" when sandbox=true.**
2. **Testnet balance reset mid-session**: Binance testnet wipes
   state periodically (~weekly). The broker is balance-fetch-per-cycle
   already; the execution layer (step #2) needs an explicit "did the
   balance change in a way I didn't predict?" reconciliation alert.
3. **API rate limits**: `enableRateLimit=True` is the default; CCXT
   throttles per the exchange's published limits. Bursting many
   `fetch_ticker` calls in `estimate_total_equity_usd` could be
   throttled at high asset counts. With a 7-asset basket it's fine;
   if the basket grows to 30+, switch to `fetch_tickers` (plural).
4. **Network blip during fetch_balance**: the constructor surfaces it
   as `BrokerError`. The bot at restart time will retry, but if a
   single cycle's `fetch_balance` fails the execution layer must
   skip the cycle, not silently use stale data. (Step #2.)
5. **Auth error vs revoked key**: both surface as
   `ccxt.AuthenticationError`. The bot can't tell "I typed it wrong"
   from "the key was revoked". The CLI surfaces the error verbatim;
   the operator has to look.
6. **Mainnet via mistyped exchange id**: there's no Kraken sandbox in
   CCXT (Kraken doesn't offer one). The instant someone sets
   `EXCHANGE=kraken`, the `SANDBOX=true` flag has no effect — Kraken
   silently goes mainnet. The two-flag gate still prevents trading
   if `ALLOW_MAINNET=false`, but `Broker.connect` should also detect
   `sandbox=True && exchange_id="kraken"` and refuse (Kraken has no
   testnet to point to). **Add this check in a follow-up.**

## Smoke testing the order pipeline

`paper-place-test-order` exercises the order placement plumbing
(`orders.py`, `clientorder.py`, `order_state.py`, `Broker.create_order_safe`)
on the actual testnet, independently of whether TSMOM is currently
producing a non-zero ladder. Run it before each execution-layer
release — every commit that touches the modules above or
`live_cycle.py` (which lands in #2b commit #4).

It uses a dedicated clientOrderId namespace,
`smoke_{YYYYMMDD}_{SYMBOL_NORMALIZED}_{side}`, that **never collides**
with the production `tsmom_…` IDs. Running smoke tests during a
production cron window does not interfere with the scheduled cycle.

### Canonical sequence (all four must pass end-to-end)

```bash
# 1. Buy: places a real testnet order for $20 of BTC.
trade-lab paper-place-test-order \
    --symbol BTC/USDT --side buy --notional 20

# Expect: terminal_status=closed, filled_amount > 0,
#         exchange_order_id set in the output.
```

```bash
# 2. Sell back: reverse the test trade so the testnet balance
#    returns to within +/- 2x slippage of pre-test.
trade-lab paper-place-test-order \
    --symbol BTC/USDT --side sell --notional 20

# Expect: terminal_status=closed. Balance check is manual via
#         `trade-lab paper-status` before and after.
```

```bash
# 3. Idempotency: re-run step 1 with identical args on the same day.
trade-lab paper-place-test-order \
    --symbol BTC/USDT --side buy --notional 20

# Expect ONE of:
#   * "skipping exchange roundtrip" (state cache had the terminal
#      record) — fast path, ideal.
#   * "already exists on exchange" then a wait-for-ack against the
#      pre-existing order — slow path, also correct.
# In both cases the create_order call count on Binance does NOT
# increase. NO duplicate position is opened.
```

```bash
# 4. Sub-minimum preflight: tiny notional below Binance's min_cost.
trade-lab paper-place-test-order \
    --symbol BTC/USDT --side buy --notional 5

# Expect: output starts with "SKIPPED: notional 5.00 USDT < min_cost
#         10.00. Exchange would reject — not sent." The exchange is
#         never contacted for placement.
```

If all four pass: the execution layer's order plumbing is healthy. If
any fail: investigate before relying on `paper-place-orders` (commit
#4) for production cron. The most likely root causes are listed in
each test's expected output above — a mismatch there is a real
regression.

### Optional smoke-test log

`--journal PATH` appends one JSON Lines record per smoke test:

```bash
trade-lab paper-place-test-order \
    --symbol BTC/USDT --side buy --notional 20 \
    --journal data/journal/smoke_tests.jsonl
```

The record format is:

```json
{"kind":"smoke_test","asof":"...","exchange":"binance","sandbox":true,
 "result":{"client_order_id":"smoke_...","terminal_status":"closed",...}}
```

This is a separate file from `cycles.jsonl` by design — smoke tests
are not strategy cycles and should not appear in the monitoring
dashboard's Status / Cycles tabs alongside real bot activity.

## Live paper trading on testnet

`paper-place-orders` is the production daily CLI. Once
`paper-place-test-order` (smoke test) passes, this is what runs
the strategy against the testnet exchange.

```bash
trade-lab paper-place-orders --journal data/journal/cycles.jsonl
```

What it does:

1. Reconstructs the state of any open orders left from a prior cycle
   (`fetch_order` + `fetch_my_trades` fallback). closed/canceled
   become terminal in local state; lost orders get `lost_track` plus
   a warning that survives in the journal.
2. Fetches a fresh balance — now reflecting the reconciliation.
3. Computes signal, target allocation, delta plan — same primitives
   as `paper-dry-run`.
4. Places the plan in sell-first order, 200ms inter-order spacing,
   5-minute per-order wait-for-ack budget.
5. Writes one Cycle entry (schema v2) with `outcome` ∈ {success,
   partial, unknown_orders, failed} and `orders_executed` populated.

Refuses to start when SANDBOX=false unless ALLOW_MAINNET=true is
also set (CLAUDE.md two-flag gate). Even when both flags pass,
prints a loud last-chance warning before connecting.

### Daily cron

```cron
5 0 * * * /opt/trade-lab/.venv/bin/trade-lab paper-place-orders \
    --journal /opt/trade-lab/data/journal/cycles.jsonl \
    >> /opt/trade-lab/data/logs/paper-place-orders.log 2>&1
```

00:05 UTC gives the daily candle a few minutes to settle in
Binance's API before the strategy reads it. Same minute pattern as
the hourly dry-run cron — they don't conflict because they share
the journal file but use different state and `clientOrderId`
namespaces (`tsmom_` for both production and dry-run, `smoke_` for
smoke tests).

### Why dry-run is hourly but live is daily

Dry-run can run as often as you like — it never sends an order.
Hourly gives monitoring a fresh staleness signal every hour, surfaces
testnet balance wipes or manual deposits within the hour, and catches
bot failures (network, revoked key) before the next live cycle.

Live placement is exactly daily, no more often. The backtest computed
the signal at daily resolution against daily candles; placing orders
more often than that is a different turnover profile and an
un-validated strategy — CLAUDE.md hard rule "execution must
replicate the backtest exactly" applied to cadence.
