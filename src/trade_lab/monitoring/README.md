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

The monitoring process binds **only to localhost**. Public access
requires nginx + TLS + basic auth configured outside this project;
the example server block is in the section below.

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

## nginx + TLS + basic auth (your responsibility)

The Streamlit binding above is private to the VPS. Public access — TLS,
authentication, and the exposed hostname — is configured outside this
project, in nginx and certbot. This is a deliberate split: the project
ships the localhost-only baseline; you own the public-facing surface.

**TLS certificate via certbot:**

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d monitoring.example.com
```

**Basic-auth password file:**

```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd_monitoring <username>
# Prompts for password. Use -c only on the first user; omit it to add more.
sudo chmod 640 /etc/nginx/.htpasswd_monitoring
sudo chown root:www-data /etc/nginx/.htpasswd_monitoring
```

**nginx server block** — `/etc/nginx/sites-available/monitoring.example.com`:

```nginx
server {
    listen 443 ssl http2;
    server_name monitoring.example.com;

    ssl_certificate     /etc/letsencrypt/live/monitoring.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/monitoring.example.com/privkey.pem;

    auth_basic           "trade-lab monitoring";
    auth_basic_user_file /etc/nginx/.htpasswd_monitoring;

    location / {
        proxy_pass http://127.0.0.1:7000;

        # WebSocket upgrade — Streamlit uses a WebSocket for its
        # interactive session. WITHOUT these three headers the page
        # loads once and then never updates: tabs, sliders, and
        # the auto-refresh tick all silently break.
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streamlit long-polls the WebSocket; default 60s timeout
        # would tear sessions down every minute.
        proxy_read_timeout 86400;
    }
}

# Redirect plaintext HTTP -> HTTPS.
server {
    listen 80;
    server_name monitoring.example.com;
    return 301 https://$host$request_uri;
}
```

```bash
sudo ln -s /etc/nginx/sites-available/monitoring.example.com \
           /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## Pre-deployment security checklist

Before pointing humans at the dashboard, walk through this list and
tick every box. A miss here is the difference between a private
monitoring tool and an exposed read endpoint into your bot's behaviour.

- [ ] System user `monitoring` exists with no shell and no home
      (`getent passwd monitoring` ends in `/usr/sbin/nologin`).
- [ ] Streamlit listens only on `127.0.0.1:7000`
      (`sudo ss -tlnp | grep 7000` shows `127.0.0.1:7000`, never
      `0.0.0.0:7000` or `*:7000`).
- [ ] Curl from a non-VPS machine to `http://<public-ip>:7000/`
      returns connection refused (not 200, not 301).
- [ ] `sudo -u monitoring cat /opt/trade-lab/.env` returns
      `Permission denied`. If it shows the file contents, the
      monitoring user can read your API keys — **stop**.
- [ ] Journal file is mode 0640 with group `monitoring`
      (`stat -c '%a %U %G' /opt/trade-lab/data/journal/cycles.jsonl`
      shows `640 botuser monitoring`).
- [ ] systemd hardening directives are applied
      (`systemctl show trade-lab-monitoring --property=NoNewPrivileges,ProtectSystem,ReadOnlyPaths`
      shows `yes`, `strict`, `/opt/trade-lab`).
- [ ] nginx server block has `auth_basic` and a populated
      `.htpasswd_monitoring` (curl without `-u user:pass` returns 401).
- [ ] TLS works — `curl -I -u user:pass https://monitoring.example.com/`
      returns HTTP 200 with no cert warnings.
- [ ] nginx has all three WebSocket upgrade headers
      (`proxy_http_version 1.1`, `proxy_set_header Upgrade`,
      `proxy_set_header Connection "upgrade"`). Without these the
      dashboard loads once and never refreshes — visually confirmed
      by clicking a tab and watching nothing change.
- [ ] Mainnet indicator is visibly RED and large in the dashboard.
      Verify by temporarily setting the bot's `sandbox=false` and
      `allow_mainnet=true` in a test config (NEVER on the production
      .env), restarting against that config, and confirming the
      banner. Restore the original config before any real trading.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Status tab shows "no journal entries" | Bot has not written a cycle yet, or the path is wrong. | Verify `TRADE_LAB_MONITORING_JOURNAL_PATH` matches the bot's `--journal` path. Run `ls -la $JOURNAL_PATH` as the monitoring user. |
| Status shows STALE/DOWN despite bot running | Bot writes but not within `MONITORING_EXPECTED_CYCLE_INTERVAL_SECONDS`. | Either raise the threshold to match the bot's cadence, or investigate why cycles are slow. |
| Permission denied when reading the journal | `monitoring` user is not in the journal file's group, or the group bit is not set. | Re-run the permissions step (`chown botuser:monitoring`, `chmod 640`). |
| Dashboard loads once and then stops updating | WebSocket headers missing in nginx. | Add `proxy_http_version 1.1`, `proxy_set_header Upgrade`, `proxy_set_header Connection "upgrade"`. |
| 502 Bad Gateway from nginx | Streamlit not running, or wrong upstream port. | `sudo systemctl status trade-lab-monitoring`. Confirm `127.0.0.1:7000` matches `proxy_pass`. |
| Dashboard publicly reachable on :7000 (CRITICAL) | `--server.address 127.0.0.1` missing or unit not reloaded. | Restore the flag, `daemon-reload`, restart. Run the cross-machine curl check before declaring fixed. |
| Mainnet banner where testnet expected (CRITICAL) | Bot is connecting to mainnet. | **Stop the bot immediately.** Check `.env` for `TRADE_LAB_PAPER_SANDBOX` and `TRADE_LAB_PAPER_ALLOW_MAINNET`. If both flags are set wrong, the bot is executing real orders. |
