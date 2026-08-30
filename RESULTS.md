# Strategy results — master index

Single navigable summary of every strategy, overlay, and cross-section
construction the project has tested. Each row points at the full
finding for details. Project-wide convention: every Sharpe / DSR
quoted is **net of cost** and **OOS** unless explicitly marked
otherwise. `PROJECT_NUM_TRIALS = 500` (pinned, see CLAUDE.md).

**DSR convention — read this before any DSR number in this file.**
There is **no single convention across this file**. Earlier revisions
of this paragraph claimed one and were wrong. The list below is the
label, row by row, and it is exhaustive over the DSR figures quoted
here:

* **Minimal `1/sqrt(T)` at N=500 — the project's *gate* convention.**
  Rows **#2, #3, #4, #5, #6, #8, #9**; the *secondary* figures of row
  **#1** (0.770, cluster median 0.736, peak 0.782, neighbour-only
  0.726); every "DSR median …, N of M pass" phrase; and every
  "DSR > 0.5" used as a status or a threshold. All of these come out
  of `walk_forward_v2.aggregate_walk_forward` or `ensemble.py`, which
  hardcode `sharpe_std_dev = 1/sqrt(T)`.
* **Conservative pooled `sd_trial_sharpes = 0.7` — the project's
  primary *reported* convention.** Row **#1's headline** (0.037 on the
  concat-OOS series, 0.0016 on the venue-verified replay window,
  computed with the de-annualized `0.7/sqrt(365) ≈ 0.0366` per-period)
  **and rows #12 (CTREND), #13 (MVRV), #14 (HMM)** — those three
  verdicts were argued in their own findings against the pooled-0.7
  bar `E[max Sharpe over 500 trials] ≈ 2.14` (annualized), and were
  **never** computed under `1/sqrt(T)`:
  `findings/ctrend_proxy_price_only.md` § DSR,
  `findings/mvrv_overlay.md` § DSR,
  `findings/hmm_regime_overlay.md` § DSR, and
  `findings/strategy_test_session_2026_05_29.md` § 6.
* **No convention stated by the source.** Row **#10**
  (cross-sectional reversal, DSR 0.001):
  `findings/cross_sectional_reversal.md` records the number with no
  deflator named, and its reproduction block computes no DSR at all.
  Read it as unlabelled — not as minimal.

Rows #7 and #11 quote no DSR. Footnote ¹ under the table repeats this
split; the policy and its scope live in "DSR reporting convention" at
the bottom.

One standing exception to the "OOS unless marked" rule, marked at
every use below: the **venue-verified replay window** figures
(Sharpe +0.72 and the DSRs computed on it) are a **fixed-config
historical replay**, not selection-OOS and not forward data. They
come from re-running the already-frozen config (`ac8919…`) over
2022-01-21 → 2026-05-27 — a window that *overlaps* the walk-forward
OOS sample — in order to check that Binance prices and independently
sourced Bybit prices give the same answer
(`findings/validation_multiexchange.md`). Venue agreement is what
that number tests; it is not a second, independent out-of-sample
result.

## What "LIVE" status actually means (and what it does not)

A strategy that reaches LIVE status has cleared the backtest-side gates and the rollout ladder. It has **not** cleared the forward test — that one is running. Specifically:

* **The deploy gate is "DSR > 0.5 at N=500, cluster-stable" evaluated UNDER THE MINIMAL 1/sqrt(T) DEFLATOR CONVENTION.** That is the convention every gate decision in this repo was made under, and it is still the gate. Read literally, passing it means: after correcting for the project's effective multiple-testing budget (500 trials) with a trial dispersion set to the sampling noise of one Sharpe estimate, the strategy's Sharpe is unlikely to be pure selection noise. **It does NOT mean the strategy is profitable going forward, only that the historical edge is unlikely to be a statistical artifact under that convention.**
* **The gate convention and the reporting convention are deliberately different — this is not an oversight, and there is no claim here that both were passed.** The 2026-08-25 owner decision (see "DSR reporting convention" at the bottom of this file and `findings/dsr_convention_2026_08.md`) made the *conservative* deflator the primary **reported** figure. Under it the deployed config scores DSR 0.037 ≈ 0, and so do all **six** of its cluster neighbours (seven cluster members in total, the deployed config included — see the counting note in section #2 below). That decision changed **reporting only**: no gate was re-run, no threshold was re-set, and a 0.5 bar was never restated for the conservative deflator. So the honest statement is: **strategy #1 passed the gate as the gate is defined (minimal convention), and would NOT pass a gate restated on the conservative convention — under which no config in this project would pass, which is why the bar was not restated.** The deploy case therefore does not rest on the DSR headline at all; it rests on parameter stability (a convention-free raw-Sharpe plateau) plus the forward test.
* Consequence for reading this file: wherever **"DSR > 0.5"** appears as a *status* or a *threshold*, it means the minimal-convention gate. Wherever a DSR is quoted as the **headline for the deployed config**, it is the conservative figure. The two never mean the same number.
* The **remaining honest gate is the forward test itself** — live behaviour against the actual order-placement pipeline. Sources of failure that DSR cannot rule out: signal stability under live data-feed jitter, slippage divergence from the modelled rate, partial fills, exchange-side rejections, network reliability, regime shifts the historical sample never saw. This gate is **open, not passed**: it is being walked now, with capped capital.

**Status of strategy #1 as of 2026-08-30: LIVE on Binance mainnet with real money (~148 USDT), since 2026-08-24.**

That is a deliberately small allocation, and the ladder that got there was walked step by step rather than flag-flipped:

1. **Testnet paper trading** — validated the order pipeline (idempotent `clientOrderId`, reconstruction, wait-for-ack, journal schema). Note that testnet **cannot** validate the signal path: Binance testnet wipes its candle history roughly monthly, so SMA(200) never warms and no buy order is ever placed there.
2. **Mainnet read-only observation** — 6-hourly dry-run cron against a key with no trading permission. Mainnet has full kline history, so this is where the SMA(200) / ladder signal path was actually exercised.
3. **Capped smoke test** (≤ 25 USDT) — one real order end to end.
4. **Daily live cron** — since 2026-08-24; the first cycle filled 7 of 7 basket legs.

Real-money placement is gated by three separate environment flags (`SANDBOX=false` + `ALLOW_MAINNET=true` unlock read paths; `MAINNET_LIVE_ORDERS=true` is additionally required to place). See `src/trade_lab/execution/README.md` for the full ladder and the environment-isolation guards.

**LIVE status is not a profit claim.** It means the config passed the minimal-convention DSR gate, cleared the rollout ladder, and is now being observed with an amount the owner is willing to lose. Under the primary conservative convention this same config scores DSR 0.037 ≈ 0 — the deploy case rests on the parameter plateau plus this forward test, not on the DSR headline.

## All strategies at a glance

| # | Strategy / variant | Class | Status | Key metric ¹ | Finding |
|---|---|---|---|---|---|
| 1 | **TSMOM (28, 60) + SMA(200) gate on market-basket** | Single-signal trend, 7-asset basket | **LIVE (Binance mainnet, capped)** | Concat-OOS Sharpe +1.48 (venue-verified *replay* window 0.72 — fixed-config historical replay, not selection-OOS); conservative-deflator DSR 0.037 ≈ 0 — deploy case = parameter plateau (deployed config + **6** neighbors, raw Sharpe +1.37…+1.49, one shared return stream) + forward test; DSR 0.770 / cluster median 0.736 over all 7 members, **6/6 neighbors** > 0.5, under 1/sqrt(T) (secondary, = the gate convention) | `findings/han_28d_tsmom.md` |
| 2 | TSMOM short-ensemble (lookbacks 28/60/120, etc) | Strategy family | Cluster-stable | DSR median 0.736 over 7 cluster members = deployed (28, 60) + 6 neighbors (6/6 neighbors pass) | `findings/cluster_stability.md` |
| 3 | TSMOM Han single lookbacks | Strategy family | Cluster-stable | DSR median 0.702 (6/6 pass) | `findings/cluster_stability.md` |
| 4 | PMA ratio ladder | Strategy family | Cluster-stable | DSR median 0.716 (6/6 pass) | `findings/cluster_stability.md` |
| 5 | SMA crossover ensemble | Strategy family | Cluster-FAILS | DSR median 0.431 (6/19 pass) | `findings/cluster_stability.md` |
| 6 | Market-basket index construction (pre-Han) | Aggregation primitive | Building block of #1 | DSR 0.658 (now superseded by #1) | `findings/market_basket_tsmom.md` |
| 7 | VolatilityTargetWrapper | Overlay (any strategy) | Asset-conditional | Helps ETH/SOL on Sharpe; hurts BTC on Calmar | `findings/vol_targeting_regime_gate.md` |
| 8 | Breadth filter (`GatedStrategy`) | Overlay (sequence count) | Does NOT improve on SMA200 | Basket DSR identical to SMA200 | `findings/breadth_filter.md` |
| 9 | 21-sleeve ensemble portfolio (3 strats × 7 assets) | Portfolio aggregation | Below threshold | DSR 0.425 (below 0.5; DD halved but DSR lowered) | `findings/ensemble_portfolio.md` |
| 10 | Cross-sectional one-day reversal | Cross-section rotation | REJECT | Sharpe +0.01, DSR 0.001 (deflator convention not stated by the finding) | `findings/cross_sectional_reversal.md` |
| 11 | Cross-sectional momentum (rotation top-K) | Cross-section rotation | Available, used as benchmark | Sharpe 0.70–0.95 on 24-coin universe (benchmark in CTREND test) | (no standalone finding; lives in `backtest/cross_sectional.py`) |
| 12 | CTREND-proxy (price-only) | Cross-section ML ranker | **REJECT** | Sharpe 0.32–0.50; underperforms BH-BTC and CSM | `findings/ctrend_proxy_price_only.md` |
| 13 | MVRV-ratio overlay | BTC weekly tilt (on-chain) | **INCONCLUSIVE** (REJECT-leaning) | Sharpe 0.58–0.93 vs BH 0.65–1.11 | `findings/mvrv_overlay.md` |
| 14 | HMM 2-state regime overlay | BTC regime gate (Markov-switching) | **REJECT** | Loses 5/6 cuts to existing VolTarget; Sharpe 0.46–0.77 | `findings/hmm_regime_overlay.md` |

¹ **Convention label for the table and every section below it — the per-row
split is spelled out in "DSR convention" at the top of this file.** Short form,
and it is deliberately *not* a blanket, because a blanket would be false:
rows #2–#6, #8 and #9, row #1's secondary figures, and every
"DSR median …, N of M pass" phrase are **minimal 1/sqrt(T)** numbers at N=500
(the gate convention). Row #1's headline (0.037 / 0.0016) and rows **#12, #13
and #14** are **conservative pooled `sd_trial_sharpes = 0.7`** figures — the
last three were never computed under 1/sqrt(T). Row #10's DSR 0.001 names no
convention in its source. Treat this as a footnote attached to each of those
numbers rather than expecting a per-row repeat; the minimal rows have never been
recomputed conservatively, and under that deflator they would read ≈ 0, which is
why they are kept on the gate convention instead of silently mixing the two.
Full statement of the policy and its scope: "DSR reporting convention" at the
bottom of this file.

Status legend:
* **LIVE (Binance mainnet, capped)** — passes DSR > 0.5 at N=500 **under the minimal 1/sqrt(T) deflator (the gate convention — NOT the primary reporting convention, under which this same config scores 0.037 ≈ 0)**, cluster-stable, walked the full rollout ladder, and placing real orders daily with capped capital since 2026-08-24. This is the forward test running, **not** a verdict that it passed. See "What LIVE status actually means" above.
* Cluster-stable / Cluster-FAILS — see `findings/cluster_stability.md` for the rule.
* Available / benchmark — implemented in code; not a standalone deploy candidate, used to verify other tests.
* REJECT — net of cost and OOS, does not beat the relevant in-stack benchmark.
* INCONCLUSIVE — empirical data does not support KEEP; sample size too small for hard REJECT.

---

## Currently trading live (Binance mainnet, real money, capped)

### 1. TSMOM (28, 60) on market-basket index with SMA(200) gate
* **Universe:** equal-weight market-basket of 7 majors (BTC, ETH, BNB, SOL, ADA, XRP, DOGE). Monthly rebalance + on-`N_active`-change rebalance.
* **Signal:** TSMOM ladder `{0, 0.5, 1.0}` = mean of binary `sign(28d return), sign(60d return)`. SMA(200) gate zeroes the ladder when basket close < SMA.
* **Concatenated OOS Sharpe = +1.48** on the market-basket. Earlier revisions of this file quoted +1.81; the finding's own table says +1.48 — the finding is authoritative. This is the genuinely selection-OOS number.
* **Venue-verified replay window Sharpe = +0.72** — **a fixed-config historical replay, NOT a second OOS result.** The frozen config (`ac8919…`) is re-run over 2022-01-21 → 2026-05-27, a window that overlaps the walk-forward OOS sample, purely to check that Binance and independently sourced Bybit prices agree (`findings/validation_multiexchange.md`). It carries no additional selection-bias protection beyond what the concat-OOS number already carries, and it is not forward data. Treat it as "does the edge survive a different venue's prices, and how much of it lives in the venue-verifiable era" — nothing more.
* **DSR, two-layer convention (primary statement).** Under the project's own conservative deflator (pinned conservative assumption `sd_trial_sharpes ≈ 0.7` annualized → `0.7/sqrt(365) ≈ 0.0366` per-period; an a-priori pool dispersion, no committed trial panel; `deflated_sharpe_ratio` compares per-period quantities), DSR = **0.037** on the concat-OOS series (Sharpe +1.48, T = 2339) and **0.0016 ≈ 0.002** on the **venue-verified replay window** (again: fixed-config historical replay, not selection-OOS) — the latter recomputed from the frozen-config backtest of `findings/validation_multiexchange.md` (+0.72, 2022-01-21 → 2026-05-27, 1588 bars), not from a slice of the stitched series. **The deploy case rests on parameter stability stated convention-free — the deployed config plus its 6 neighbor configs (7 cluster members, not 7 independent confirmations) land in the raw concat-OOS Sharpe band +1.37…+1.49; the 6 neighbors alone span the same band, so the deployed +1.48 is inside a plateau it does not define (one shared return stream, correlation ≥ 0.97; no lone peak) — plus the forward test**, not on a DSR headline. The "median DSR 0.736, 6/6 neighbors > 0.5" form of that stability holds only under the secondary 1/sqrt(T) convention.
* **Secondary figure:** DSR = 0.770 at N=500 under the minimal 1/sqrt(T) deflator; cluster median 0.736 under the same convention, computed over all 7 cluster members with the deployed config among them (`(30, 60)` at 0.782 is the actual cluster peak, not the deployed point). Note what "minimal" does and does not mean: the N=500 extreme-value correction **is** applied in both conventions (`deflated_sharpe_ratio` passes `sharpe_std_dev` into `expected_max_sharpe`, which scales it by the Bailey-LdP factor for 500 trials). What is minimal is the *dispersion* fed into that factor — 1/sqrt(T) ≈ 0.021, the sampling noise of a single Sharpe estimate, instead of the wider dispersion of the pool of strategies actually searched. Both figures are reproducible from the same code and from the committed return series in `docs/results/` (see below); `findings/dsr_convention_2026_08.md` has the convention decision.
* **Survivorship caveat:** the basket is seven majors known ex post — the composition axis (which 7 coins) is untested pending the PIT diagnostic run, and the project's own cross-sectional-momentum measurement showed Sharpe 1.40 → 0.93 when moving to a PIT universe (deep review 2026-08-24).

**What DSR under either convention actually says.** Both figures correct for the 500-trial selection budget with the same extreme-value machinery; they differ only in how dispersed the trial pool is assumed to be. The minimal-deflator figure (0.770) says the historical Sharpe clears the expected-max bar of 500 trials whose spread is nothing but the sampling noise of one Sharpe estimate. The conservative-deflator figure (0.037) says it does **not** clear the bar once the pool is assumed as dispersed as the project's pinned `sd ≈ 0.7`. Neither says the strategy will be profitable going forward. Backtest survival is the *previous* gate, not the *final* one.

**Current state and what comes next.**
* Now: **live on Binance mainnet with real money since 2026-08-24**, ~148 USDT, placed daily by the `paper-place-orders` cron. The first live cycle filled 7 of 7 basket legs. See `src/trade_lab/execution/README.md` for the pipeline and the rollout ladder that was walked (testnet pipeline validation → mainnet read-only observation → capped smoke test → daily live cron).
* An earlier revision of this file promised a "code-path migration to Kraken" and cited a CLAUDE.md rule "Live orders only on testnet". Both are obsolete: the Kraken plan was cancelled on 2026-07-09 (the premise that Binance is geo-blocked for the owner was simply wrong), and that hard rule no longer exists. What replaced it is the three-flag mainnet gate plus the rollout ladder — mainnet is still not a flag-flip.
* Next: accumulate forward-test evidence at this size. The live sample is weeks old and proves the *pipeline*, not the *edge* — at this horizon it cannot distinguish a working strategy from a lucky one, and it is not expected to.
* Testnet remains configured as a second, parallel paper environment, with its own journal and state. Note it cannot validate the signal path at all: the testnet wipes candle history roughly monthly, so SMA(200) never warms there.

* Finding: `findings/han_28d_tsmom.md`.

---

## Tested at the strategy family level (cluster-stability discipline)

Footnote ¹ from the table above applies to this whole section: every "median DSR
… , N of M pass" below is a **minimal 1/sqrt(T)-convention** figure. The
DSR-threshold *form* of cluster stability exists only under that convention —
under the conservative deflator all of these sit at ≈ 0. For the deployed config
the convention-free form of the same fact is the raw concat-OOS Sharpe band
+1.37…+1.49 across the deployed config and its 6 neighbors.

### 2. TSMOM short-ensemble (multiple lookback pairs)
**Counting note (applies wherever this cluster is cited).** The cluster has
**7 members and the deployed `(28, 60)` is one of them**, so the familiar
"7 of 7 pass" is *the deployed config plus 6 neighbours* — six confirmations,
not seven. The deployed point cannot be evidence for itself. Stated without the
double count: median DSR 0.736 across all 7 members; **6 of 6 neighbours pass**
DSR > 0.5, neighbour-only median 0.726 (plain arithmetic on the per-variant
table in `findings/cluster_stability.md` — nothing re-run, no new trials). The
deployed config is not the family's peak either: `(30, 60)` leads on both
metrics (Sharpe +1.49, DSR 0.782) versus `(28, 60)` at +1.48 / 0.770 — an
earlier revision of this line called `(28, 60)` the best individual, which the
finding's own table contradicts.

### 3. TSMOM Han single lookbacks
Median DSR 0.702 across 6 cluster neighbours; **6 of 6 pass** DSR > 0.5. Adding the short-ensemble (combining two lookbacks) gives a small Sharpe lift over the best single lookback.

### 4. PMA ratio ladder (Detzel et al. 2021)
Median DSR 0.716 across 6 cluster neighbours; **6 of 6 pass**. PMA is a legitimate trend signal and cluster-stable, but the deployed TSMOM (28, 60) on the basket clears the bar with more margin and is simpler. PMA is implemented in `strategies/pma_ratio.py` for research use; not deployed.

### 5. SMA crossover ensemble
**FAILS cluster test**: median DSR 0.431, only 6/19 individual configs pass. Single best is cherry-picked. Lesson: SMA crossover survives single-config DSR but not cluster-stability.

All four families documented in `findings/cluster_stability.md` — the core methodology document that demotes SMA crossover and validates the TSMOM family.

---

## Overlays / wrappers (apply to a base strategy)

### 7. VolatilityTargetWrapper (Moreira-Muir style)
* Wraps any base strategy with `signal × (annual_vol_target / realized_vol)` capped at 1.0.
* **Asset-conditional**: helps ETH / SOL / most alts on Sharpe and Calmar; **hurts BTC on Calmar specifically** (BTC has fat right tail vol-targeting cuts off).
* Used as an in-stack benchmark in the HMM regime overlay test (#14) — HMM is essentially a duplicate of vol-target, and loses to it.
* Finding: `findings/vol_targeting_regime_gate.md`.

### 8. Breadth filter (`GatedStrategy` wrapper)
Adds a "breadth ≥ K%" filter on top of the SMA200 regime gate. **Does NOT improve** the market-basket TSMOM (basket DSR identical at 0.658). **Asset-conditional improvements only**: helps ETH (DSR 0.28 → 0.45), hurts BTC (0.31 → 0.13). Stacking breadth on top of SMA200 strictly hurts: the filters are substitutes, not complements. Finding: `findings/breadth_filter.md`.

---

## Portfolio-level aggregation

### 9. 21-sleeve ensemble portfolio (3 strategies × 7 assets)
* Equal-weight, dynamic 1/N_active, rebalance-on-universe-change costing.
* Concatenated OOS Sharpe **+1.13**, DSR **0.425** at N=500.
* Single best sleeve (`pma_medium × BNB` with vol30) achieves Sharpe +1.27, DSR 0.564 — **better than the portfolio**. Diversification halved max DD (good) but lowered DSR (the "many unconfirmed bets has higher DSR than one" intuition is false here).
* Useful as a sanity benchmark; not a deploy candidate. Finding: `findings/ensemble_portfolio.md`.

---

## Cross-sectional rotation strategies

### 10. Cross-sectional one-day reversal
**REJECT.** Annualized Sharpe +0.01, DSR @ N=500 = 0.001 — **the finding names no deflator convention** and its reproduction block computes no DSR, so this figure is unlabelled, neither minimal nor conservative. The verdict does not turn on it: +0.01 Sharpe fails under any deflator. The literature exists (Zaremba 2021, Bianchi 2022) but does not survive on Binance majors with realistic costs and our universe. Critical failure mode: "buy losers" → strategy keeps buying LUNA / FTT / UST as they fall to zero. Documented as a verdict that prevents future re-attempts of the same shape. Finding: `findings/cross_sectional_reversal.md`.

### 11. Cross-sectional momentum (top-K rotation by past return)
Implemented in `backtest/cross_sectional.py` (`run_cross_sectional_momentum`). Used as a benchmark in CTREND-proxy test on the 24-coin coinmetrics universe: Sharpe 0.70–0.95 across (cost × subperiod) cuts. **Not deployed standalone** — the basket-level TSMOM (#1) outperforms with simpler operations.

### 12. CTREND-proxy (price-only) — Fieberg et al. JFQA 2024
**REJECT.** Pooled Ridge on 6 price MA-ratio features at 6 windows, weekly top-quintile, 730-day train. Underperforms BH-BTC and existing CSM on every subperiod × cost cut. Kraken net-negative (CAGR −3.3% full). DSR ≈ 0 at N=500 **under the conservative pooled `sd_trial_sharpes = 0.7`** (Sharpes 0.27–0.50 against the annualized bar `E[max] ≈ 2.14`, `findings/ctrend_proxy_price_only.md` § DSR) — **not** the minimal 1/sqrt(T) gate convention, under which this row was never computed. **Important asymmetry caveat:** rejection is of the proxy, not the paper — volume half omitted (Coin Metrics community has only reported volume; trusted volume paid-tier), and pooled Ridge is not Fieberg's Fama-MacBeth estimator. Faithful V2 would need paid data + FMB. Finding: `findings/ctrend_proxy_price_only.md`.

---

## On-chain overlays

### 13. MVRV-ratio overlay (weekly BTC tilt)
**INCONCLUSIVE** (empirically tilting REJECT). Underperforms BH-BTC on Sharpe / Calmar / CAGR on every subperiod × cost. DD reduction is real (−64% vs BTC's −84% pre-ETF) but disproportionately costs return. DSR ≈ 0 at N=500 **under the conservative pooled `sd_trial_sharpes = 0.7`** (Sharpes 0.58–0.93 against the annualized bar `E[max] ≈ 2.14`, `findings/mvrv_overlay.md` § DSR) — **not** the minimal 1/sqrt(T) gate convention, under which this row was never computed; ~2–3 BTC market cycles in available data is too small for a confident hard REJECT. Compass-artifact prior of INCONCLUSIVE confirmed. **Important caveats:** ratio thresholds approximate the canonical Z-score (paid-tier `CapMVRVZ` / `CapRealUSD` not available on community tier); a faithful Z-score implementation might find different thresholds, but the binding constraint is sample size. Finding: `findings/mvrv_overlay.md`.

---

## Regime-switching overlays

### 14. HMM 2-state regime overlay
**REJECT.** Gaussian HMM on daily log-returns, refit weekly on trailing 730d, long when **filtered** (not smoothed) P(bull) > 0.5 else cash. Per the user's operative decision rule for this candidate ("must beat existing VolTarget — duplicate test"), HMM loses 5 of 6 (cost × subperiod) cuts. The one HMM win is Binance-only / post-ETF-only (2.4y, Sharpe 0.77 vs VolTarget 0.49) but vanishes on Kraken (HMM ties VolTarget at 0.46) and falls below the DSR bar at N=500 **under the conservative pooled `sd_trial_sharpes = 0.7`** (Sharpes 0.46–0.77 against the annualized bar `E[max] ≈ 2.14`, `findings/hmm_regime_overlay.md` § DSR). That bar is **not** the minimal-convention "DSR > 0.5" gate used elsewhere in this file: this row was never computed under 1/sqrt(T). **Critical implementation invariant**: uses forward-only `_hmmc.forward_log(...)` for filtered probabilities; never `predict_proba` (which is smoothed and uses future data). Finding: `findings/hmm_regime_overlay.md`.

---

## Methodology & session writeups (not strategies)

* `findings/strategy_test_session_2026_05_29.md` — sweep of three candidates (CTREND, MVRV, HMM) in one sitting; reusable methodology lessons.
* `findings/literature_review_v1.md`, `_v2.md`, `_v3.md` — three external survey indexes; navigation maps, not deploy candidates.
* `findings/cluster_stability.md` — the core cluster-stability discipline that demotes single-config DSR survivors.
* `findings/buy_and_hold_cost_symmetry.md` — engine consistency check: after applying symmetric entry costs to the B&H benchmark, **Δ Sharpe ≤ 0.05 across all (strategy × asset) pairs and 0 prior verdicts flipped**.

## Pinned constants

* `PROJECT_NUM_TRIALS = 500` (CLAUDE.md hard rule)
* Conservative pooled `sd_trial_sharpes ≈ 0.7` — a **pinned a-priori assumption** about cross-project trial dispersion, never estimated from a committed trial-Sharpe panel; `E[max Sharpe over 500 trials] ≈ 2.14` (both annualized; `deflated_sharpe_ratio` works per-period, so de-annualize by `sqrt(365)` before passing — see `findings/dsr_convention_2026_08.md`)
* Single-config DSR threshold for deployment: 0.5 cluster-median —
  **evaluated under the minimal 1/sqrt(T) convention, which is the
  gate convention.** The 2026-08-25 reporting decision did NOT restate
  this threshold for the conservative deflator: under that deflator
  every config in this project, the deployed one included, sits at
  DSR ≈ 0, so a 0.5 bar would be vacuous rather than strict. The
  threshold and the primary reported DSR figure are therefore quoted
  on different conventions on purpose — see "What LIVE status
  actually means" above and the section below.

## DSR reporting convention (owner decision 2026-08-25)

Two deflator conventions coexist. They serve different jobs and are
never merged: the **conservative** convention is what the project
*reports* about the deployed config, the **minimal** convention is
what the project's *deploy gate and thresholds* are defined on. The
two-layer statement is primary everywhere alive, and the 1/sqrt(T)
figure is secondary and must be labeled as such:

* **Primary (conservative):** pinned conservative assumption
  `sd_trial_sharpes ≈ 0.7` annualized (not an empirical estimate),
  de-annualized to `0.7/sqrt(365) ≈ 0.0366` per-period
  for `deflated_sharpe_ratio` → DSR ≈ 0 for the deployed config
  (computed 0.037 on the concat-OOS series at Sharpe +1.48,
  T = 2339; 0.0016 on the **venue-verified replay window** — a
  fixed-config historical replay of the frozen config, not
  selection-OOS and not forward data — at +0.72, 1588 bars). The
  deploy case rests on
  parameter stability stated convention-free (the deployed config
  plus 6 neighbor configs — 7 cluster members, so six confirmations
  and not seven — in the raw concat-OOS Sharpe band +1.37…+1.49, one
  shared return stream — no lone peak) plus the forward test.
* **Secondary (minimal), and the gate convention:** 1/sqrt(T) null
  sampling std → DSR 0.770 for the deployed config, 0.736 cluster
  median over its 7 members (6/6 neighbours > 0.5 — the DSR-threshold
  form of cluster stability exists only under this convention). It
  applies the *same* N=500
  extreme-value correction as the conservative convention; what is
  minimal is the assumed dispersion of the trial pool (the sampling
  noise of one Sharpe estimate, 1/sqrt(T) ≈ 0.021) rather than the
  wider dispersion of the strategies actually searched.

**Scope of the labeling rule.** "Label the convention" binds every
place this file speaks about the **deployed config** (#1) and every
place a **gate or threshold** is defined — those are the statements the
project acts on. It does *not* require re-deriving the historical
per-strategy rows; it does require that their label be *correct*, and
the correct label is not one blanket. Rows #2–#6, #8 and #9 are
minimal-convention figures never recomputed conservatively; rows
**#12–#14 are conservative pooled-0.7 figures never computed under
1/sqrt(T)**; row #10 states no convention at all. The enumerated split
in "DSR convention" at the top of this file is the label, and it is
repeated at the point of use for the three exceptions — no bare,
unscoped DSR is left anywhere.

`findings/dsr_convention_2026_08.md` § "Scope of that labeling rule"
predates this correction and calls the whole #2–#14 block
minimal-convention. On rows #12–#14 it is superseded by the split
above; that file is a dated artifact and is not rewritten.

Both figures stay reproducible from `walk_forward_v2.py` (computation
unchanged) on the concat-OOS series; the venue-verified replay figure
is reproduced from the frozen-config backtest path of
`scripts/validation_test1_multiexchange.py` instead — the two are
different statistical objects and are never mixed. Because `data/` is
gitignored, the two derived return series are committed so the numbers
survive without the parquet snapshot:
`docs/results/dsr_convention_2026_08_concat_oos_returns.csv` and
`docs/results/dsr_convention_2026_08_venue_replay_returns.csv` (each
carries a `#` provenance header naming the frozen-config hash, the data
vintage, and the commit). Full rationale:
`findings/dsr_convention_2026_08.md`.

Last updated: 2026-08-30 (strategy #1 status corrected from "PAPER
(Binance testnet)" to LIVE on Binance mainnet with capped real money
since 2026-08-24; the obsolete Kraken-migration plan and the retired
CLAUDE.md rule "Live orders only on testnet" removed; the rollout
ladder actually walked is now described. Earlier the same day: the
file-wide "everything is minimal 1/sqrt(T)" claim replaced by a
per-row enumeration. Previously 2026-08-26: DSR two-layer convention +
survivorship caveat + gate/reporting-convention split, replay-vs-OOS
labeling, committed return-series artifacts, and cluster counting
restated as deployed + 6 neighbours. Strategy table content otherwise
as of 2026-05-29; the REJECT verdicts are unchanged.)
