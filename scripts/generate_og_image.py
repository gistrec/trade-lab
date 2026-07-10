"""Regenerate the Open Graph link-preview image (ops/static/og-image.png).

The 1200x630 card is served by nginx at /og-image.png on the public
monitoring deployment (see monitoring/README.md § Link preview). It plots
the real equal-weight basket index from the local parquet snapshot, so
the preview honestly depicts the product.

Usage:
    .venv/bin/python scripts/generate_og_image.py [--end 2026-05-28]

--end trims the candle history (exclusive) — useful when the local
snapshot has a trailing gap that the fail-loud index builder refuses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from trade_lab.backtest.market_index import (  # noqa: E402
    build_crypto_market_index_with_weights,
)

REPO = Path(__file__).resolve().parents[1]
ASSETS = ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")
OUT = REPO / "ops" / "static" / "og-image.png"

BG = "#0e1117"        # Streamlit dark background
FG = "#fafafa"
GREY = "#9aa0a6"
GREEN = "#2e7d32"     # dashboard testnet banner palette
RED = "#b71c1c"       # dashboard mainnet banner palette
LINE = "#4caf50"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end", default=None,
        help="Trim candles strictly before this date (e.g. 2026-05-28) "
             "when the local snapshot has a trailing gap.",
    )
    args = parser.parse_args()

    candles = {}
    for a in ASSETS:
        df = pd.read_parquet(REPO / "data" / f"binance_{a}_USDT_1d.parquet")
        if args.end:
            df = df[df.index < args.end]
        candles[a] = df

    market = build_crypto_market_index_with_weights(candles)
    close = market.index["close"].dropna().tail(365)
    close = close / close.iloc[0]

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    ax = fig.add_axes([0.07, 0.10, 0.86, 0.38])
    ax.set_facecolor(BG)
    ax.plot(close.index, close.values, color=LINE, linewidth=2.2)
    ax.fill_between(close.index, close.values, close.values.min(),
                    color=LINE, alpha=0.12)
    ax.axis("off")

    fig.text(0.07, 0.84, "trade-lab", color=FG, fontsize=54,
             fontweight="bold", family="DejaVu Sans")
    fig.text(0.07, 0.735, "TSMOM crypto strategy — live paper-trading monitor",
             color=FG, fontsize=22, family="DejaVu Sans")
    fig.text(0.07, 0.655, "signal · portfolio · cycle journal · execution health",
             color=GREY, fontsize=17, family="DejaVu Sans")
    fig.text(0.07, 0.545, "BTC · ETH · BNB · SOL · ADA · XRP · DOGE",
             color=GREY, fontsize=15, family="DejaVu Sans")

    # Environment chips, mirroring the dashboard banner palette. Right-
    # aligned in the empty area between the text block and the sparkline.
    fig.text(0.93, 0.66, " TESTNET — PAPER TRADING ", color="white",
             fontsize=13, family="DejaVu Sans", fontweight="bold", ha="right",
             bbox=dict(boxstyle="round,pad=0.55", facecolor=GREEN,
                       edgecolor="none"))
    fig.text(0.93, 0.555, " MAINNET — REAL MONEY ", color="white",
             fontsize=13, family="DejaVu Sans", fontweight="bold", ha="right",
             bbox=dict(boxstyle="round,pad=0.55", facecolor=RED,
                       edgecolor="none"))

    fig.text(0.07, 0.455,
             "Equal-weight 7-asset market-basket index — last 12 months",
             color=GREY, fontsize=12, family="DejaVu Sans")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=100)
    print(f"{OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
