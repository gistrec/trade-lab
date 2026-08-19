// pm2 process config for the read-only Streamlit monitoring dashboard.
// Usage (from project root):
//   pm2 start ecosystem.config.js
//   pm2 logs trade-lab-monitoring
//   pm2 restart trade-lab-monitoring
//   pm2 stop trade-lab-monitoring
//   pm2 save            # persist across reboots (with `pm2 startup`)
module.exports = {
  apps: [
    {
      name: "trade-lab-monitoring",
      cwd: __dirname,
      // Run the venv streamlit binary directly; `interpreter: "none"`
      // stops pm2 from trying to execute it with node.
      script: ".venv/bin/streamlit",
      interpreter: "none",
      args: [
        "run",
        "src/trade_lab/monitoring/app.py",
        "--server.port",
        "7000",
        "--server.address",
        "127.0.0.1",
      ],
      env: {
        TRADE_LAB_MONITORING_JOURNAL_PATH: "data/journal/cycles.jsonl",
        // Mainnet journal → enables the testnet/mainnet source switcher.
        // The mainnet bot crons write here (see execution/README.md).
        TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET:
          "data/journal/cycles_mainnet.jsonl",
        // Must match the dry-run heartbeat cron cadence (6h). Dashboard
        // staleness is computed as multiples of this, so a mismatch flags a
        // perfectly healthy bot as stale. Was 3600 (hourly cadence).
        MONITORING_EXPECTED_CYCLE_INTERVAL_SECONDS: "21600",
      },
      autorestart: true,
      // Streamlit watches its own files; don't let pm2 also watch.
      watch: false,
      // Streamlit + pandas/pyarrow sit ~230MB resident at rest; restart
      // only if it grows well past that (guards against a slow leak).
      max_memory_restart: "400M",
    },
    // Netdata health endpoints (ops/health_server.py) — testnet + mainnet.
    // Lived only in a hand-typed `pm2 start` (and its dump) until the
    // russia-03 move lost them; now reproducible from this file.
    {
      name: "trade-lab-health",
      cwd: __dirname,
      script: "ops/health_server.py",
      interpreter: ".venv/bin/python",
      env: {
        TRADE_LAB_MONITORING_JOURNAL_PATH: "data/journal/cycles.jsonl",
        TRADE_LAB_HEALTH_HOST: "127.0.0.1",
        TRADE_LAB_HEALTH_PORT: "7001",
        // 2× the 6h dry-run cadence + slack; daily gets 26h.
        TRADE_LAB_HEALTH_HEARTBEAT_MAX_AGE_S: "43200",
        TRADE_LAB_HEALTH_DAILY_MAX_AGE_S: "93600",
      },
      autorestart: true,
      watch: false,
    },
    {
      name: "trade-lab-health-mainnet",
      cwd: __dirname,
      script: "ops/health_server.py",
      interpreter: ".venv/bin/python",
      env: {
        TRADE_LAB_MONITORING_JOURNAL_PATH: "data/journal/cycles_mainnet.jsonl",
        TRADE_LAB_HEALTH_HOST: "127.0.0.1",
        TRADE_LAB_HEALTH_PORT: "7002",
        TRADE_LAB_HEALTH_HEARTBEAT_MAX_AGE_S: "43200",
        TRADE_LAB_HEALTH_DAILY_MAX_AGE_S: "93600",
        // Daily probe off for mainnet — matches the pre-move live config.
        TRADE_LAB_HEALTH_DAILY_DISABLED: "true",
      },
      autorestart: true,
      watch: false,
    },
  ],
};
