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
