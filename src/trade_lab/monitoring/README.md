# trade-lab monitoring

Read-only dashboard for the paper-trading bot. Reads the bot's
append-only journal and exposes status, signal, portfolio drift, and
recent cycles. **No control buttons, no exchange access, no API
credentials.** The bot writes; this UI reads.

## Architecture

```
[bot process]                    [monitoring process]
  trade-lab paper-dry-run          streamlit run app.py
       │                                   │
       ▼ (append, mode 0640)               ▼ (read-only)
  data/journal/cycles.jsonl ◄───────────────┘
                                            │
                                            ▼
                            Streamlit on 127.0.0.1:7000
```

The monitoring process binds **only to localhost**. Public access via
nginx + TLS + basic auth comes in a follow-up commit. Until then, the
dashboard is reachable from the VPS itself but not from the Internet.

## Prerequisites

* Linux with systemd (Ubuntu 22.04+, Debian 12+, similar).
* Python 3.11 or newer.
* The trade-lab project installed at `/opt/trade-lab` (convention this
  guide assumes; adjust paths if you use a different prefix).
* Root access for user creation and systemd unit installation.

## Permissions setup

Monitoring runs as a dedicated system user that has **read access to
the journal and nothing else** — in particular, no access to the bot's
`.env` file with API keys.

```bash
# 1. Create the monitoring user. No shell, no home directory.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin monitoring

# 2. Bot's .env is owned by the bot user, mode 0600 — monitoring
#    must NOT be able to read this. Replace `botuser` with whatever
#    user owns the bot.
sudo chown botuser:botuser /opt/trade-lab/.env
sudo chmod 600 /opt/trade-lab/.env

# 3. Journal: group-readable by monitoring, not world-readable.
sudo chown -R botuser:monitoring /opt/trade-lab/data/journal
sudo chmod 750 /opt/trade-lab/data/journal
sudo chmod 640 /opt/trade-lab/data/journal/cycles.jsonl
```

After step 3, the bot user can read+write the journal, the monitoring
user can read but not write, and everyone else cannot see it at all.

### Critical verification — do NOT skip

The `.env` file with API keys must remain unreadable to monitoring.

```bash
sudo -u monitoring cat /opt/trade-lab/.env
```

**Expected:** `Permission denied`.

**If you see the file contents:** the deployment is unsafe. The
monitoring user can read your API keys. Fix step 2 above before
proceeding — do not start the systemd unit until this check passes.

## Project installation

```bash
# As root, create /opt/trade-lab if needed.
sudo mkdir -p /opt/trade-lab
sudo chown botuser:botuser /opt/trade-lab

# As the bot user, clone and set up the venv with monitoring extras.
sudo -u botuser bash <<'EOF'
cd /opt/trade-lab
git clone <repo-url> .
python3 -m venv .venv
.venv/bin/pip install -e ".[monitoring]"
EOF
```

The bot itself (`trade-lab paper-dry-run --journal <path>`) must be
running and writing to the journal path configured below. Bot startup
itself is outside this README's scope.

## Systemd setup

The unit template lives at
`src/trade_lab/monitoring/trade-lab-monitoring.service.example`.
Review it before installing — especially the `Environment=` lines
that hold the journal path and the expected cycle interval.

```bash
# Copy the unit into systemd's path.
sudo cp /opt/trade-lab/src/trade_lab/monitoring/trade-lab-monitoring.service.example \
        /etc/systemd/system/trade-lab-monitoring.service

# Tell systemd to re-scan unit files.
sudo systemctl daemon-reload

# Start now and enable on boot.
sudo systemctl enable --now trade-lab-monitoring

# Confirm it came up cleanly.
sudo systemctl status trade-lab-monitoring
```

## Binding verification

This is the most important check after starting the service. Streamlit
MUST be bound to 127.0.0.1 only — otherwise the dashboard is exposed
publicly without TLS or authentication.

```bash
sudo ss -tlnp | grep 7000
```

**Expected** — local address is `127.0.0.1`:

```
LISTEN 0 128 127.0.0.1:7000 0.0.0.0:* users:(("streamlit",pid=...,fd=...))
```

**Red flags — stop and fix the unit file:**

* `0.0.0.0:7000` or `*:7000` → Streamlit is exposed publicly. The
  `--server.address 127.0.0.1` flag is missing or the unit was not
  reloaded after editing. Run `sudo systemctl daemon-reload`,
  `sudo systemctl restart trade-lab-monitoring`, recheck.
* No output → Streamlit failed to start. Check
  `sudo journalctl -u trade-lab-monitoring` for the actual error.

Cross-check from another machine:

```bash
# From a machine that is NOT the VPS:
curl -v http://<your-public-IP>:7000/
```

Expected: `Connection refused` or timeout. Anything else (HTTP 200,
301, etc.) means the binding is wrong — do not proceed until you see
connection-refused.

## Management commands

```bash
# Start / stop / restart
sudo systemctl start trade-lab-monitoring
sudo systemctl stop trade-lab-monitoring
sudo systemctl restart trade-lab-monitoring

# Autostart on boot
sudo systemctl enable trade-lab-monitoring
sudo systemctl disable trade-lab-monitoring

# Status + last few log lines
sudo systemctl status trade-lab-monitoring

# Follow logs in real time
sudo journalctl -fu trade-lab-monitoring

# Logs in a window
sudo journalctl -u trade-lab-monitoring --since "1 hour ago"

# After editing the .service file, ALWAYS reload before restart
sudo systemctl daemon-reload
sudo systemctl restart trade-lab-monitoring
```

## What's not yet done

Public access via nginx + TLS + basic auth is the next commit. Until
that lands, the dashboard is reachable only from the VPS itself
(`curl http://127.0.0.1:7000/` from the VPS shell) and not from the
Internet. This is the intended state for this commit — it isolates
the systemd unit, permissions, and binding-verification work into a
reviewable change with no public exposure surface.
