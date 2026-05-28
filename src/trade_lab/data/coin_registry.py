"""Curated registry of Binance USDT pairs we use for the PIT universe.

The point of this file is to enumerate every coin we *might* want to
consider in the cross-sectional momentum universe, regardless of
whether it is currently tradable. Inclusion is determined by two
filters:

1. The coin had a Binance USDT pair at some point (so the historical
   PnL is realistic against Binance fees + slippage).
2. The coin was ever in the top 30 by CoinGecko market cap during the
   2018-2026 window (so we don't bloat the panel with permanently
   small-cap tickers).

Every entry records:

* ``coingecko_id`` — id used by the CoinGecko API.
* ``binance_symbol`` — pair label as ccxt would format it.
* ``listed_date`` — date the pair started trading on Binance (best-effort
  from Binance announcement archive).
* ``delisted_date`` — date the pair stopped trading. ``None`` = still
  listed at the time of this file's writing.
* ``notes`` — free text on rename/migration, e.g. MATIC -> POL.

**Survivorship-bias caveat that this registry does NOT fully fix:**
the *current* CoinGecko top-30 by market cap is a survivor-biased
sample. To temper that, we explicitly include 15+ coins that fell out
of the top 30, were delisted from Binance, or went to ~zero. The
registry is **not** intended to be exhaustive — see
``docs/results/pit_universe.md`` for residual coverage estimates.

Dates are best-effort from publicly observable Binance announcements
and CoinGecko launch records. Where exact date is uncertain, the entry
notes that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CoinMeta:
    coingecko_id: str
    binance_symbol: str           # ccxt format: "BASE/QUOTE"
    listed_date: Optional[str]    # ISO date when the pair listed on Binance
    delisted_date: Optional[str]  # None if still listed
    notes: str = ""
    # Coin Metrics community asset id. Defaults to the lowercase Binance
    # base symbol, which matches their convention for ~90% of majors.
    # Override only when the convention differs (e.g. wluna for pre-fork
    # Terra LUNA, vs the new chain's luna2).
    coin_metrics_id: Optional[str] = None

    @property
    def base(self) -> str:
        return self.binance_symbol.split("/")[0]

    @property
    def cm_id(self) -> str:
        return self.coin_metrics_id or self.base.lower()


COIN_REGISTRY: dict[str, CoinMeta] = {
    # ------------------------------------------------------------------
    # Tier 1: always-in-top-30 majors. Currently listed.
    # ------------------------------------------------------------------
    "BTC":  CoinMeta("bitcoin",          "BTC/USDT",  "2017-08-17", None),
    "ETH":  CoinMeta("ethereum",         "ETH/USDT",  "2017-08-17", None),
    "BNB":  CoinMeta("binancecoin",      "BNB/USDT",  "2017-11-06", None),
    "XRP":  CoinMeta("ripple",           "XRP/USDT",  "2018-05-04", None),
    "ADA":  CoinMeta("cardano",          "ADA/USDT",  "2018-04-17", None),
    "SOL":  CoinMeta("solana",           "SOL/USDT",  "2020-08-11", None),
    "DOGE": CoinMeta("dogecoin",         "DOGE/USDT", "2019-07-05", None),
    "TRX":  CoinMeta("tron",             "TRX/USDT",  "2018-06-11", None),
    "DOT":  CoinMeta("polkadot",         "DOT/USDT",  "2020-08-19", None),
    "LTC":  CoinMeta("litecoin",         "LTC/USDT",  "2017-12-13", None),
    "AVAX": CoinMeta("avalanche-2",      "AVAX/USDT", "2020-09-22", None),
    "LINK": CoinMeta("chainlink",        "LINK/USDT", "2019-01-16", None),
    "ATOM": CoinMeta("cosmos",           "ATOM/USDT", "2019-04-29", None),
    "ETC":  CoinMeta("ethereum-classic", "ETC/USDT",  "2017-08-17", None),
    "XLM":  CoinMeta("stellar",          "XLM/USDT",  "2018-06-11", None),
    "FIL":  CoinMeta("filecoin",         "FIL/USDT",  "2020-10-15", None),
    "NEAR": CoinMeta("near",             "NEAR/USDT", "2020-10-14", None),
    "ALGO": CoinMeta("algorand",         "ALGO/USDT", "2019-06-21", None),
    "VET":  CoinMeta("vechain",          "VET/USDT",  "2018-07-25", None),
    "ICP":  CoinMeta("internet-computer", "ICP/USDT", "2021-05-10", None),
    "HBAR": CoinMeta("hedera-hashgraph", "HBAR/USDT", "2019-09-29", None),
    "AAVE": CoinMeta("aave",             "AAVE/USDT", "2020-10-15", None),
    "UNI":  CoinMeta("uniswap",          "UNI/USDT",  "2020-09-17", None),
    "MKR":  CoinMeta("maker",            "MKR/USDT",  "2018-08-16", None),
    "BCH":  CoinMeta("bitcoin-cash",     "BCH/USDT",  "2017-08-17", None,
                     notes="originally BCC; renamed by Binance to BCH on 2017-12-21"),
    "SHIB": CoinMeta("shiba-inu",        "SHIB/USDT", "2021-05-10", None),
    "OP":   CoinMeta("optimism",         "OP/USDT",   "2022-06-01", None),
    "ARB":  CoinMeta("arbitrum",         "ARB/USDT",  "2023-03-23", None),
    "INJ":  CoinMeta("injective-protocol", "INJ/USDT", "2020-10-21", None),
    "TIA":  CoinMeta("celestia",         "TIA/USDT",  "2023-10-31", None),
    "SUI":  CoinMeta("sui",              "SUI/USDT",  "2023-05-03", None),
    "APT":  CoinMeta("aptos",            "APT/USDT",  "2022-10-19", None),

    # ------------------------------------------------------------------
    # Tier 2: renamed / migrated. Treated as one continuous series
    # under the new ticker — coingecko_id points to the active record.
    # ------------------------------------------------------------------
    "MATIC": CoinMeta("polygon",         "POL/USDT",  "2019-04-26", None,
                      notes="MATIC -> POL migration completed 2024-09; volumes "
                            "merged. Listed on Binance as MATIC originally."),
    "FTM":   CoinMeta("fantom",          "FTM/USDT",  "2019-06-12", None,
                      notes="Rebranding to Sonic announced 2024-08; still listed."),

    # ------------------------------------------------------------------
    # Tier 3: notable delistings. These are the survivorship-bias
    # patches. Dates from Binance delisting announcements.
    # ------------------------------------------------------------------
    # Pre-collapse LUNA. Coin Metrics has the pre-fork chain under
    # ``luna`` and the new chain under ``luna2``; the wrapped Ethereum
    # version (``wluna``) tracks the same price but on Ethereum.
    "LUNA":  CoinMeta("terra-luna",      "LUNA/USDT", "2020-08-21", "2022-05-13",
                      notes="Pre-collapse LUNA. Series sourced from Coin Metrics 'luna'.",
                      coin_metrics_id="luna"),
    "UST":   CoinMeta("terrausd",        "UST/USDT",  "2020-10-01", "2022-05-31",
                      notes="Algorithmic stablecoin; depegged May 2022. "
                            "Excluded from momentum by the stablecoin filter."),
    "FTT":   CoinMeta("ftx-token",       "FTT/USDT",  "2019-12-19", "2022-11-13",
                      notes="FTX exchange token. Binance suspended trading "
                            "and withdrawals 2022-11-13 amid the FTX collapse."),
    "BUSD":  CoinMeta("binance-usd",     "BUSD/USDT", "2019-09-15", "2024-02-06",
                      notes="Stablecoin discontinued by Paxos. Excluded from "
                            "momentum panel by stablecoin filter."),
    "WAVES": CoinMeta("waves",           "WAVES/USDT", "2018-04-04", "2024-09-05",
                      notes="Delisted from Binance during the 2024 cleanup."),
    "SRM":   CoinMeta("serum",           "SRM/USDT",  "2020-08-11", "2023-08-25",
                      notes="FTX/Alameda-linked DEX token. Delisted by Binance."),
    "ANT":   CoinMeta("aragon",          "ANT/USDT",  "2019-04-25", "2024-10-25",
                      notes="One of the 2024 Q4 batch delistings."),
    "OMG":   CoinMeta("omisego",         "OMG/USDT",  "2018-09-11", "2024-09-05",
                      notes="Renamed to BOBA on Ethereum; original token delisted."),
    "BCC":   CoinMeta("bitconnect",      "BCC/USDT",  "2017-11-15", "2018-01-09",
                      notes="Bitconnect — explicit fraud. Delisted Q1 2018. "
                            "Outside our usual backtest window but listed for "
                            "survivorship-bias visibility."),
    "ZEC":   CoinMeta("zcash",           "ZEC/USDT",  "2018-04-23", None,
                      notes="Privacy coin; still listed on Binance Global as of "
                            "the 2026-05-28 snapshot of this file."),
}


def stablecoins() -> set[str]:
    """Base assets we exclude from a momentum universe by convention.

    Stablecoins dominate volume rankings but have no momentum signal
    (peg). They also produce a long string of zero returns that depress
    the realized-vol denominator in inverse-vol weighting.
    """
    return {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "UST", "DAI"}


def tradable_at(date: str, meta: CoinMeta) -> bool:
    """Return True if ``meta.binance_symbol`` was tradable on ``date``.

    ``date`` is an ISO string like ``"2021-08-15"``. Dates inside the
    [listed_date, delisted_date) interval are tradable; on the
    delisted_date itself we treat the pair as already suspended (Binance
    typically suspends trading at midnight UTC of the announced date).
    """
    if meta.listed_date is None:
        return False
    if date < meta.listed_date:
        return False
    if meta.delisted_date is not None and date >= meta.delisted_date:
        return False
    return True
