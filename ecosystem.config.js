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
  ],
};
