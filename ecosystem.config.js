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
        // Served under /monitoring/ so the research site can own the domain
        // root (nginx: / -> static site/, /monitoring/ -> this app). Streamlit
        // then serves its app, assets, websocket AND /_stcore/health under
        // /monitoring/ — the Netdata dashboard probe URL is updated to match.
        "--server.baseUrlPath",
        "/monitoring",
      ],
      env: {
        TRADE_LAB_MONITORING_JOURNAL_PATH: "data/journal/cycles.jsonl",
        MONITORING_EXPECTED_CYCLE_INTERVAL_SECONDS: "3600",
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
