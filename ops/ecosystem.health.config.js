// pm2 process config for the read-only Netdata health server.
// Sibling to the monitoring dashboard's ecosystem.config.js at the repo root.
//
// Usage (from repo root):
//   pm2 startOrReload ops/ecosystem.health.config.js
//   pm2 logs trade-lab-health
// deploy.sh loads this automatically after the monitoring dashboard.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "trade-lab-health",
      // Run from the repo root so the editable-installed `trade_lab` package
      // imports and the default relative journal path both resolve.
      cwd: path.resolve(__dirname, ".."),
      script: "ops/health_server.py",
      // Execute with the project venv's Python, not node. Resolve to an
      // absolute path (like the dashboard config) so it does not depend on
      // libuv chdir-before-exec semantics at start or after a pm2 resurrect.
      interpreter: path.resolve(__dirname, "..", ".venv/bin/python"),
      env: {
        // Same journal the dashboard reads — one source of truth.
        TRADE_LAB_MONITORING_JOURNAL_PATH: "data/journal/cycles.jsonl",
        TRADE_LAB_HEALTH_HOST: "127.0.0.1",
        TRADE_LAB_HEALTH_PORT: "7001",
        // Heartbeat: hourly dry-run + grace. Daily: 24h + 2h grace.
        TRADE_LAB_HEALTH_HEARTBEAT_MAX_AGE_S: "7200",
        TRADE_LAB_HEALTH_DAILY_MAX_AGE_S: "93600",
      },
      autorestart: true,
      watch: false,
      // A tiny stdlib HTTP server; it should sit in the low tens of MB.
      max_memory_restart: "150M",
    },
  ],
};
