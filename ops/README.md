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
| `GET /healthz/daily` | daily live health | last **main** live cycle (real placement; reconstruction excluded) within `DAILY_MAX_AGE_S` (26h) with outcome `success`, **and** no incident cycle newer than it | no live cycle in window, stale >26h, live outcome not `success`, or any cycle failed after the last live success (see notes below) |
| `GET /` | human summary | always (informational, not an alarm target) | — |

`partial` is treated as unhealthy on purpose — CLAUDE.md forbids silent
partial fills. `open_order_incidents` is surfaced in the `/healthz/daily`
body for humans but does **not** gate 503 in v1 (kept out of the paging
path to avoid false positives from stale window entries).

**Blind-spot notes (why `/daily` scans a window, not just the latest entry).**
Two false-negative paths, both rooted in the journal carrying no durable
live/dry marker:

1. A live run that raises *before* placing any order writes
   `orders_executed=None` (`live_cycle.py`), so it reads as non-live and an
   older success would mask it. `/healthz/daily` therefore also 503s on **any
   incident cycle newer than the last successful live run** — a durable scan,
   so a later hourly dry-run can't overwrite the signal away.
2. A **reconstruction** cycle (`outcome=reconstructed`) is live but only
   proves a *prior* cycle's orders were reconciled, so it is **excluded** from
   the daily clock — it can't stand in for today's placement.

Tradeoff: with no live/dry marker, path 1 also fires on a *dry-run* failure
after the last live success — the safe direction (catch a real live failure
rather than stay quiet). The clean fix is a small `mode` field on the cycle
`context` in the execution layer; that is an execution-layer change pending
owner approval.

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

**Before trusting the alarms, read the scoping banner in
`netdata/health.d/trade_lab.conf`.** A `template: on: httpcheck.status`
matches *every* httpcheck job, so each alarm must be scoped to its own job
(via `chart labels`) or it cross-fires with the ClearTranscriptBot alarm —
scope them the same way that working alarm already is, and verify the label
key against your `netdata -v`.

## Known limitation (by design)

On-host Netdata can't alert if the **whole VPS** dies — the agent dies with
it. Close that dimension with a Netdata Cloud *node unreachable* alert (an
external observer), or a single external ping. The endpoints here cover
"cron dead / cycle stale / live outcome failed / dashboard down", which is
the majority of the risk.
