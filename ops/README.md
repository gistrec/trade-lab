# `ops/` — operational tooling

Deploy-side glue that lives outside the `trade_lab` package: the Netdata
health server and the Netdata drop-in configs that alert on it. Everything
here is a **read-only consumer of the journal** — no credentials, no
exchange access — same contract as the Streamlit dashboard.

```
ops/
  health_server.py              # HTTP health endpoints (stdlib, read-only)
  ecosystem.health.config.js    # pm2 entry for the health server
  netdata/
    go.d/httpcheck.conf.example # httpcheck jobs that probe the endpoints
    health.d/trade_lab.conf     # alarm templates → botcrit (Telegram)
```

## Why a health server (and not just an exit code)

`paper-place-orders` is a **batch** job, not an always-on service, so
"is the process up?" is the wrong question. What matters is **freshness +
outcome**: did the expected cycle run recently and succeed? The server
answers that with HTTP status, so Netdata's `httpcheck` collector — which
you already use for ClearTranscriptBot — consumes it unchanged: a stale or
failed cycle returns **503**, exactly like a refused/timed-out endpoint.
That inversion (absence/failure of the expected signal ⇒ alert) is a
**dead-man's-switch**: a silently dead cron becomes an alert instead of a
job that merely "didn't run" and nobody notices.

## Two cadences ⇒ two endpoints

The bot runs a **6-hourly dry-run** (keeps the journal warm) and a **daily
live** order cycle (see `execution/README.md`). One probe cannot separate
"the whole bot died" from "today's real order cron didn't fire", so:

| Endpoint | Question | 200 when | 503 when |
|---|---|---|---|
| `GET /healthz` | heartbeat | a cycle within `HEARTBEAT_MAX_AGE_S` (default 12h) | journal unreadable/empty, or no cycle in ~12h |
| `GET /healthz/daily` | daily live health | last **main** live cycle (identified by `context.mode=='live'`; reconstruction excluded) within `DAILY_MAX_AGE_S` (26h) with a healthy outcome: `success`, plus `skipped_warmup` on a testnet journal only (see below) | no live cycle in window, stale >26h, or live outcome unhealthy — including a live run that failed even before placing an order (see note) |
| `GET /` | human summary | always (informational, not an alarm target) | — |

`partial` is treated as unhealthy on purpose — CLAUDE.md forbids silent
partial fills. `open_order_incidents` is surfaced in the `/healthz/daily`
body for humans but does **not** gate 503 in v1 (kept out of the paging
path to avoid false positives from stale window entries).

`skipped_warmup` is healthy **only when the cycle's own `context.sandbox`
is `true`**: Binance testnet wipes candles ~monthly, so SMA(200) can never
warm there and the daily cycle records a first-class skip (no orders)
instead of failing. The executor never writes this outcome on mainnet —
if it ever appears in a mainnet journal, the environment guard was
bypassed, and `/daily` pages on it.

**How `/daily` identifies the live run (and two failure modes it closes).**
Every cycle carries a durable `context.mode` (`'live'` / `'dry_run'`), set in
`live_cycle.py` / `dry_run.py`. `/healthz/daily` selects the last cycle with
`mode=='live'`, so:

1. A live run that raises *before* placing any order still has `mode=='live'`
   (with `orders_executed=None`), so it is the latest main-live cycle and its
   `failed` outcome trips 503 — it can no longer hide behind an older success.
2. A **reconstruction** cycle (`outcome=='reconstructed'`) is `mode=='live'`
   but only proves a *prior* cycle's orders were reconciled, so it is
   **excluded** from the daily clock.

Because the marker is exact, a benign 6-hourly **dry-run** failure
(`mode=='dry_run'`) never pages `/daily` — no false positive. (Pre-marker
journal entries fall back to the `orders_executed` heuristic, which cannot see
a fail-before-placement live cycle; this only affects history written before
the marker shipped.)

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `TRADE_LAB_MONITORING_JOURNAL_PATH` | `data/journal/cycles.jsonl` | journal to read (shared with the dashboard) |
| `TRADE_LAB_HEALTH_HOST` | `127.0.0.1` | bind address (localhost only — Netdata is on-host) |
| `TRADE_LAB_HEALTH_PORT` | `7001` | listen port |
| `TRADE_LAB_HEALTH_HEARTBEAT_MAX_AGE_S` | `43200` | heartbeat staleness limit |
| `TRADE_LAB_HEALTH_DAILY_MAX_AGE_S` | `93600` | daily-live staleness limit |
| `TRADE_LAB_HEALTH_DAILY_DISABLED` | `false` | when `true`, `/healthz/daily` returns 200 "disabled by config" — for a journal fed only by dry-run crons (mainnet observation phase, no live cron yet). Self-invalidating: if the journal already contains live cycles, the real verdict is returned instead. Flip to `false` in the same commit that enables the mainnet live cron. |

**Two environments ⇒ two instances.** `ops/ecosystem.health.config.js`
ships `trade-lab-health` (testnet journal, port 7001) and
`trade-lab-health-mainnet` (`cycles_mainnet.jsonl`, port 7002, daily
check disabled during the observation phase). A 200 from
`/healthz/daily` can therefore also mean "disabled by config" — the
JSON `reason` field distinguishes the two.

## Run it

```bash
# local
.venv/bin/python ops/health_server.py
curl -s localhost:7001/healthz | python -m json.tool

# prod (pm2) — deploy.sh does this automatically
pm2 startOrReload ops/ecosystem.health.config.js
```

## Wire Netdata

```bash
sudo cp ops/netdata/go.d/httpcheck.conf.example /etc/netdata/go.d/httpcheck.conf   # merge if it exists
sudo cp ops/netdata/health.d/trade_lab.conf     /etc/netdata/health.d/trade_lab.conf
sudo netdatacli reload-health
```

**Prerequisite — re-scope the existing ClearTranscriptBot alarm first.**
Its `cleartranscript_healthcheck` is a `template: on: httpcheck.status` with
no chart-labels line, so it matches *every* httpcheck job; the moment the
trade-lab jobs appear it also fires on them. Add one line to it:

```
chart labels: _collect_job=cleartranscript
```

The trade-lab alarms scope by `_collect_job` (verified present on this box's
go.d httpcheck charts), so each attaches to exactly its own job. Then
`sudo netdatacli reload-health`.

## Metrics (Prometheus → Netdata)

The health server also serves `GET /metrics` in Prometheus text format
(`ops/metrics.py`), read-only from the journal. Netdata's go.d/prometheus
scraper turns every series into a time-series chart:

| Metric | Type | Meaning |
|---|---|---|
| `tradelab_up` | gauge | 1 when the exporter served the scrape |
| `tradelab_journal_read_error` | gauge | 1 if the journal was unreadable |
| `tradelab_journal_valid_cycles` / `_corrupt_lines` / `_unknown_version_lines` | gauge | journal-parse self-instrumentation |
| `tradelab_last_cycle_age_seconds` / `_timestamp_seconds` | gauge | freshness of the most recent cycle |
| `tradelab_last_live_cycle_age_seconds` / `_timestamp_seconds` | gauge | freshness of the most recent live order cycle |
| `tradelab_cycles_total{outcome=...}` | counter | cycles by outcome over the whole journal |
| `tradelab_cycle_duration_ms{quantile="0.5"\|"0.95"}` / `_max` | gauge | recent cycle duration (last 200) |
| `tradelab_open_order_incidents` | gauge | executed orders not in a resolved terminal state |
| `tradelab_cumulative_skipped_drift_usd` | gauge | cumulative quote drift skipped |
| `tradelab_equity_usd` | gauge | latest paper equity from a successful cycle |
| `tradelab_last_signal_ladder_value` / `tradelab_sma_gate_open` | gauge | latest signal state |

Wire it:

```bash
sudo cp ops/netdata/go.d/prometheus.conf.example /etc/netdata/go.d/prometheus.conf   # merge if it exists
sudo systemctl restart netdata   # go.d job changes need a restart
```

Charts appear under the `trade_lab` job. The exporter is **total**: a corrupt
or missing journal degrades individual metrics (or sets
`tradelab_journal_read_error 1`) rather than failing the scrape.

**Value-based alarms** (`ops/netdata/health.d/trade_lab_metrics.conf`, →
`botcrit`) complement the httpcheck dead-man's-switch: `trade_lab_open_orders`
(any executed order stuck in a non-terminal state — the `lost_track == 0`
SLO), `trade_lab_cycle_latency` (max cycle duration > 60s/120s — slow exchange
round-trips), and `trade_lab_journal_read_error` (journal unreadable). Each
targets a unique `prometheus.trade_lab.<metric>` context, so no scoping is
needed. Install with `sudo cp ... && sudo netdatacli reload-health` (the
metric charts already exist, so no restart).

**Host clock alarms** (`ops/netdata/health.d/trade_lab_clock.conf`, → `botcrit`)
are the infra-side twin of the broker's `_check_clock_skew()` pre-flight guard:
`host_clock_unsynced` (NTP not disciplining the clock) and `host_clock_offset_high`
(offset > 500ms/1s) on Netdata's internal `timex` charts. Clock drift is a
silent-failure class here — the idempotency clientOrderId is UTC-date-derived
and signed requests must be within `recvWindow` of exchange server time. Netdata
ships a stock clock-sync alarm but as `to: silent`, so this routes it to botcrit.

## Known limitation (by design)

On-host Netdata can't alert if the **whole VPS** dies — the agent dies with
it. Close that dimension with a Netdata Cloud *node unreachable* alert (an
external observer), or a single external ping. The endpoints here cover
"cron dead / cycle stale / live outcome failed / dashboard down", which is
the majority of the risk.
