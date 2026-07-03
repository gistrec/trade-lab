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

The bot runs an **hourly dry-run** (keeps the journal warm) and a **daily
live** order cycle (see `execution/README.md`). One probe cannot separate
"the whole bot died" from "today's real order cron didn't fire", so:

| Endpoint | Question | 200 when | 503 when |
|---|---|---|---|
| `GET /healthz` | heartbeat | a cycle within `HEARTBEAT_MAX_AGE_S` (default 2h) | journal unreadable/empty, or no cycle in ~2h |
| `GET /healthz/daily` | daily live health | last **main** live cycle (identified by `context.mode=='live'`; reconstruction excluded) within `DAILY_MAX_AGE_S` (26h) with outcome `success` | no live cycle in window, stale >26h, or live outcome not `success` — including a live run that failed even before placing an order (see note) |
| `GET /` | human summary | always (informational, not an alarm target) | — |

`partial` is treated as unhealthy on purpose — CLAUDE.md forbids silent
partial fills. `open_order_incidents` is surfaced in the `/healthz/daily`
body for humans but does **not** gate 503 in v1 (kept out of the paging
path to avoid false positives from stale window entries).

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

Because the marker is exact, a benign hourly **dry-run** failure
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
| `TRADE_LAB_HEALTH_HEARTBEAT_MAX_AGE_S` | `7200` | heartbeat staleness limit |
| `TRADE_LAB_HEALTH_DAILY_MAX_AGE_S` | `93600` | daily-live staleness limit |

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

## Known limitation (by design)

On-host Netdata can't alert if the **whole VPS** dies — the agent dies with
it. Close that dimension with a Netdata Cloud *node unreachable* alert (an
external observer), or a single external ping. The endpoints here cover
"cron dead / cycle stale / live outcome failed / dashboard down", which is
the majority of the risk.
