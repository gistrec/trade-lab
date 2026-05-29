# Validation discipline for retail crypto algo trading — what a single solo run can actually claim

**Draft. 2026-05-29.** Methodology write-up based on `trade-lab`'s
validation phase. The reader is assumed to know what a backtest is
and roughly how time-series momentum works; everything else is
self-contained.

> *The claim of this article is not "I found an edge in crypto
> momentum." Han et al. (2024) already published that, and 28-day
> TSMOM has been in the literature for years. The claim is about
> the **set of discipline checks** a solo researcher with a public
> data tier can run between "the backtest looks promising" and
> "I am about to wire real money", and the **specific failure modes
> each check catches that the others cannot.**

## 0. Why this write-up exists

Most crypto-strategy blog posts — and a non-trivial slice of crypto
finance papers — stop at:

> *Strategy X has Sharpe Y on Z years of data, here is the equity
> curve.*

A solo retail researcher reading that has no way to distinguish
between (a) a real edge, (b) a backtest with a hidden look-ahead in
the index construction, (c) an edge that lives entirely in a sample
period the writer happens to like, and (d) an edge that survives
only on a single exchange's fee schedule. Each of those failure
modes has a different fix; conflating them produces "I deployed and
got rugged but the paper said Sharpe 1.5".

The methodology below was developed during a 7-day validation phase
on a TSMOM(28, 60) + SMA(200) regime-gated basket strategy. The
goal of the phase was **to falsify the strategy**, not to confirm
it — every test was designed so that "PASS" would be the worst
possible outcome of running it. The strategy survived four of five
falsification attempts; the fifth produced a documented per-venue
verdict. I think the **shape of the discipline** is more transferable
than the result.

## 1. Freeze before you measure

Selection bias is the easiest way to fool yourself, and the hardest
to detect after the fact. The countermeasure is mechanical: before
the first validation test runs, the strategy parameters are written
to a single dataclass with a SHA-256 hash, and a unit test pins that
hash:

```python
@dataclass(frozen=True)
class ProductionConfig:
    assets: Tuple[str, ...] = ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")
    lookbacks: Tuple[int, ...] = (28, 60)
    sma_filter_periods: Tuple[int, ...] = (200,)
    use_vol_target: bool = False
    # ... including knobs that are inactive in the current config
    vol_lookback: int = 30
    # ... and the cost model
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005

CANONICAL_HASH = sha256(canonical_json(PRODUCTION_CONFIG)).hexdigest()
```

The test:

```python
def test_canonical_hash_pinned():
    assert CANONICAL_HASH == "ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753"
```

Three things make this load-bearing rather than ceremonial:

1. **Inactive knobs are in the hash.** The strategy uses
   `use_vol_target = False`. If you flip it to `True` later, the
   strategy now consults `annual_vol_target = 0.25` — a knob you may
   have set on a whim months ago. Including it in the hash means the
   flip becomes a hash change, which becomes a test failure, which
   becomes a forced documentation step.
2. **The hash is checked by every consumer.** The forward-test
   harness verifies the hash at startup and refuses to run on
   drift. The DSR diagnostic verifies the hash before producing a
   number. The reference behavioral fingerprint embeds the hash in
   its frozen artifact. If you change the config without going
   through the documented procedure, *something* in the validation
   chain breaks loudly.
3. **The procedure to change it is documented in the test file.**
   A future you (or contributor) editing the dataclass first sees:
   "if this change is intentional, document a new research cycle,
   re-run walk-forward + DSR, *only then* update this pin."

This costs ~30 lines of code and converts "I'm going to tweak
this and see if it improves" — which is the entire pathology of
selection bias — into a deliberate research action.

## 2. Survive at least one independent venue before trusting the number

A retail backtest typically uses one exchange's data
(Binance is over-represented; it has the cleanest public REST and
the longest spot history). A Sharpe ratio computed on that data
silently assumes the exchange's price construction is a faithful
representation of "the market". When the strategy moves to a
different exchange — or to live execution — the assumption may
break.

The test: replay the **frozen** strategy on independent daily OHLCV
from at least one other venue. Three numbers come out:

| Metric | What it measures |
|---|---|
| Signal agreement % (per bar) | Whether the strategy *decides* the same thing on independently-sourced prices |
| Same-window equity Sharpe per venue | Whether the strategy *would have done* the same thing on independently-sourced prices |
| Per-asset daily-return correlation | Whether the data layer itself is comparable |

In the trade-lab case, on a 4.4-year venue-confirmed sub-period:

* Binance ↔ Bybit signal agreement: **100.0%** on 1589 bars.
* Net Sharpe: Binance +0.721, Bybit +0.719 (Δ +0.002).
* Per-asset return correlation: ≥ 0.9989 for 6 of 7 majors.

A research practitioner working only with Binance would have
reported the full-sample Sharpe (+1.377) and stopped. The same
practitioner running the venue-replay test discovers that **the
verified-window Sharpe is 0.72**, and that the 1.38 number is
dominated by a 2018-2022 sub-period that no independent venue can
confirm at the public REST tier (Bybit spot did not exist before
mid-2021; Kraken's public OHLC API hard-caps at the trailing 720
days).

The honest deploy-time expectation moves from 1.38 to 0.72. Not
because anything is wrong with the 1.38 — the math on Binance is
real — but because the part you cannot independently verify should
not anchor forward expectations.

A note on "the test is short". Kraken's 720-bar cap is a permanent
data-access limit at the retail tier; it is not a deficiency of the
test. The right write-up move is to **report the limit explicitly**
and not pretend the test covered 8 years when it covered ~2 on
that venue. Three observers agreeing on a short period beats one
observer agreeing with itself on a long one.

## 3. Cost-tax as a separate axis from "does the signal work"

Test 2 in the validation phase only swaps engine-level fee and
slippage parameters; it does NOT swap the data, recompute signals,
or rebuild the basket. The point is to isolate the **marginal
cost-tax** of moving from one venue's fee schedule to another, on
identical signals.

This isolation matters because it produces an interpretable result:

> *The Kraken-vs-Binance tax is structurally constant at ≈ −0.14
> Sharpe across every regime block tested, dominated by the 4× taker
> fee delta. It is not regime-absorbed.*

That sentence is actionable. "It is fee-dominated, not
slippage-dominated" tells you: a maker-priced Kraken tier (or a
fee-reduction promo) closes most of the gap. "It is regime-
independent" tells you: even a sustained bull run will not paper
over the tax. Compared to a single bottom-line number like "Kraken
Sharpe = 0.58", the decomposed answer survives interrogation about
"what if X changes".

Reporting per-venue verdicts rather than a single PASS/FAIL is the
correct corollary: **Binance PASS, Kraken fee-fragile-not-advisable
at the current 0.40% taker, with a documented conditional re-entry
path (maker or ≤ 0.20% taker)**.

## 4. Universe-bias closure is two axes, not one

"Universe bias" gets used as a single phrase, but it decomposes
into:

* **Listing axis**: is any asset in the basket on a date where the
  asset was not yet listed on the assumed exchange?
* **Liquidity axis**: is any asset in the basket on a date where its
  realistic liquidity could not absorb the deploy notional?

For a frozen hand-picked top-7 basket at $10k notional, both axes
turned out to be moot — but only after explicit per-axis checks:

1. For listing axis: read each asset's parquet's `min(date)` and
   compare against the exchange's official listing date. For the
   trade-lab data the delta was ≥ 0 days for every asset (some
   parquets were *truncated* relative to listing, never extended).
   `closes.notna()` is therefore empirically equivalent to
   `tradable_at(date, listing_metadata)` on this universe.
2. For liquidity axis: rank each asset against a wider candidate
   pool by trailing-90-day median USD volume. Every major in the
   basket sat in the top-15 by volume on every bar of the verified
   window. The lowest-volume major had $265M+ daily median. At a
   $1.4k per-asset notional that is 5 × 10⁻⁶ of daily median — six
   orders of magnitude below any liquidity-relevant level.

A subtler point this check surfaced: the `build_pit_universe`
function in the codebase silently dropped BNB from the universe in
every bar of the verified window, because the CoinMetrics
community-tier cache reported `market_cap = NaN` for BNB, and the
ranking code used `na_option="bottom"` which sorted NaN to the end
of the universe. This **does not affect the deployable
hand-picked basket** (BNB is included unconditionally) — but it
would silently shrink the universe for any future cross-sectional
rotation strategy derived from the same code path.

The lesson — and this is one of the few that I think transfers
without modification — is: **NaN in an eligibility computation
must fail loudly, not silently re-rank the universe.** The bug here
was harmless because the candidate strategy bypassed the affected
code path; a different candidate would have been silently
evaluated against a 6-asset universe instead of 7 and the operator
would not have known.

## 5. Behavioral fingerprint is not a Sharpe gate

The hardest principle to hold to in validation discipline is that
**a behavioral monitor must not measure profit.** A correctly-
behaving strategy is allowed to lose money in an adverse regime.
Putting Sharpe or equity into the monitor's percentile bands
reintroduces "did we make money this month" as the pass-fail
criterion through the back door, which is exactly what disciplined
validation exists to avoid.

What the fingerprint *does* measure are behavioral invariants that
should persist regardless of which sub-regime the strategy is in:

* Exposure-flip frequency (how often does the position change).
* Per-event rebalance turnover (how big are the position changes).
* Regime-gate flip frequency (how often does the SMA-200 gate
  cross).
* Drawdown profile (distribution of current-DD-from-peak, with
  the **max historical DD as an explicit breach threshold**).
* Position concentration — recorded as a *structural note* on an
  equal-weight basket because per-asset target weight is
  mechanically `ladder / N`, so a percentile band on it would be
  falsely narrow.

Two things make this hard to do correctly:

1. **The bands must be frozen.** If the monitor refits its bands
   from live data — by averaging in incoming days, by sliding the
   rolling window, by "adapting to the new normal" — slow live
   degradation widens the bands and the monitor never fires. The
   bands are computed once from the venue-verified backtest window,
   stored as a versioned hash-pinned artifact, and the monitor
   loads them as-is.
2. **The statistical honesty must be load-bearing.** Daily rolling
   90-day windows are not 1500 independent samples; they are
   ~16 effective observations of "what does a 90-day window look
   like". The percentile bands describe **the observed range of
   behavior on the historical sample**. They are descriptive, not
   inferential. The monitor's advisory wording reflects this:
   "operator review", not "reject the null".

The drawdown band has a particular trap. Its lower edge is anchored
by the worst observed drawdown in the verified window
(−32.17% in the trade-lab case, dating to the 2022 bear / FTX
collapse). The breach criterion is "live DD goes deeper than
−32.17%", **not** "DD exists". A live drawdown of −20% with
plenty of historical depth above it is well within the band.
Reporting headroom — "live DD is currently at −26.86%, with 5.31
pp of room before breach" — gives the operator the calibrated
state.

## 6. Look-ahead audit at the signal layer, not the P&L layer

This is the methodological move I think is most under-used and most
worth copying.

A look-ahead bug lives in the **signal-generating pipeline**: index
construction, eligibility masking, `fillna` direction, warmup
handling, timestamp alignment. P&L is downstream of the signal —
if the signal is correct on identical input, the P&L is correct;
if the signal is wrong, the P&L is wrong by the same amount the
signal is wrong, and chasing the bug in the P&L layer means
debugging through a return-times-position composition.

The audit:

```
for each bar T in the verified window:
    truncated_panel = {asset: data[:T] for asset, data in full_panel.items()}
    pit_basket = build_basket_index(truncated_panel)
    pit_signal = strategy.generate_signals(pit_basket)
    sig_pit [T] = pit_signal.iloc[-1]
    sig_full[T] = full_sample_signal.loc[T]
    assert sig_pit[T] == sig_full[T]
```

The same equality is checked at:

* The basket index value at T (catches normalization look-ahead).
* The SMA-200 value at T (catches warmup-handling bugs).
* The regime-gate boolean at T (catches gate-flip look-ahead).

This audit is **offset-FREE** — it compares `signal[T]` to
`signal[T]`, same convention on both sides. The notorious 1-bar-
offset question (do you use signal at T to position at T or T+1)
belongs to a separate live detector that compares forward live
signals to backtest replays on identical vintage. Mixing the two
concerns produces a test that is impossible to interpret.

On the trade-lab pipeline: **zero mismatches across 1589 bars × 4
metrics, max absolute difference exactly 0.00e+00.** This means
the verified-window backtest is signal-level look-ahead-free on a
deterministic reproducible test. It does NOT promise that forward
P&L matches backtest P&L (regime drift, execution noise, and venue
artifacts all still apply) — but it does close one specific
failure mode that is otherwise undetectable from the equity curve.

The scope boundary deserves writing down:

* Catches: temporal look-ahead in the signal / index / SMA / gate.
* Does NOT catch: universe-selection bias (closed separately,
  Section 4), live data-revision look-ahead (the harness's
  content-hashed vintage store is the mechanism for the latter).

The audit takes 33 seconds on a 2026 laptop. Cheap enough to be
re-run after any code change that touches the strategy, index, or
config modules.

## 7. Forward replay infrastructure must be byte-immutable from day one

The look-ahead audit above runs on backtest data. The complementary
forward check needs to compare a *live* signal (computed by the
harness on a real production day) against a *replay* signal
(computed by the backtest on identical input bytes). For this to be
meaningful, the harness must save the exact OHLCV bytes it saw on
each decision day in a form that survives later data revisions.

The pattern I think is correct:

1. The harness writes a **physically separate immutable copy** of
   the OHLCV bytes used for each decision — not a pointer to a
   mutable shared store.
2. The copy is addressed by the **SHA-256 hash of its own bytes**
   (content-hash). Two-level directory layout (`{hash[:2]}/{hash}.txt`)
   keeps any single directory manageable after years of cycles.
3. The content-hash is recorded in the journal row written that
   day.
4. Loading by hash **verifies on read** that the bytes still hash
   to the same value. A bit-flip on disk, an editor accidentally
   rewriting the file, or an operator inadvertently sed-replacing
   a price — all of these surface at read time, not silently.
5. Writes are atomic (tmpfile + rename), so a crash mid-write
   cannot produce a partially-written snapshot whose hash
   mismatches its filename.

Canonical serialization is text, not parquet. Parquet's byte
representation is not stable across `pyarrow` versions or
compression settings — the same logical data can hash to different
values depending on the runtime, which defeats the entire purpose.
Canonical text (sorted asset keys, ISO UTC timestamps,
8-decimal floats, fixed separators) is byte-deterministic and
human-readable.

The single feature this enables — and that nothing else does — is
**the look-ahead detector for live data**. On each forward cycle:

```
live_row = read_latest_journal_row()
vintage = load_vintage(live_row.vintage_content_hash)
replay_signal = strategy.generate_signals(build_basket(vintage)).iloc[-1]
assert live_row.ladder_state == replay_signal
```

A mismatch on identical bytes is a backtest look-ahead in the live
path. A constant 1-bar offset on every mismatch is a labeling
artifact (the journal's `date` field convention drifted from the
backtest replay convention). A random-pattern mismatch is the
load-bearing signal that the strategy is consulting bytes it
should not have access to.

Until forward data accumulates, the live detector has nothing to
check; the backtest audit (Section 6) is the dispositive look-ahead
test for the backtest path itself. The two are complementary, not
redundant.

## 8. Honest DSR on the venue-verified sample

The headline number that justified entering validation in the
trade-lab case was a Deflated Sharpe Ratio (DSR) of 0.770 — the
Bailey-López de Prado correction for multiple-testing on a Sharpe
ratio, given a known pool of trials. 0.770 is well above the
"more likely real than not" threshold of 0.5.

That number was computed on **walk-forward concatenated
out-of-sample returns** on the full Binance sample. It is
mathematically real on its terms.

But: rerunning the project's existing DSR machinery on the
**venue-verified post-2022 sample** (1589 bars, annualized SR
+0.721) with `PROJECT_NUM_TRIALS = 500` and a conservative pool
dispersion `sd = 0.7`:

```
expected_max_sharpe(N=500, sd=0.7) ≈ 2.137 per-period
DSR(verified_window, N=500, sd=0.7)   ≈ 0.000
```

Per-period SR of 0.038 (annualized 0.72) is well below the
expected-max bar. DSR floors to zero.

This is not a contradiction with the original 0.770. The two
numbers measure different things:

* The original DSR was on walk-forward folds on a different sample.
* The new DSR is on the direct backtest on the venue-verified
  shorter sample.

Both are honest. The **deployment-relevance** number is the one
on the sample whose Sharpe you can actually anchor to forward
expectations — i.e., the venue-verified one. That number says:
under the project's conservative multiple-testing penalty, the
observed Sharpe does not clear the trial-pool expected-max bar.

This does not void the strategy. It frames the forward expectation:
there is a measurable raw edge (+0.72 Sharpe over 4.4 years), but
under the project's own multiple-testing discipline the
deploy-confidence is modest, not high.

I think the right way to read this in a methodology context is:
**stating DSR ≈ 0 on the venue-verified sample alongside the
+0.72 Sharpe is more honest than not stating either.** A
practitioner who reports only Sharpe overstates confidence; one
who reports only the failing DSR overstates pessimism. Both
together calibrate the forward expectation.

## 9. The change-management contract

Every check above is conditional on the strategy being a fixed
target. The contract that holds across them:

> **Any change to the frozen strategy parameters is a new research
> cycle.** That includes basket composition, lookbacks, SMA period,
> vol-targeting toggle, cost-model rates, rebalance frequency, and
> any inactive-but-recorded knob.
>
> Procedure: open a `findings/<descriptive_name>.md` documenting
> the change as a new trial; re-run walk-forward + DSR on the new
> config; update the consumers' pinned hashes only then.

The forward paper-trade horizon is **precisely** the time during
which the strategy must not change, because the harness, monitor,
and detector are all gathering evidence about it. A tweak in the
middle of forward testing forks the timeline: the monitor's
reference no longer corresponds to the strategy being run, the
detector's replay no longer compares to the same code path, the
forward Sharpe is a mixture of two strategies.

The change-management rule is what makes the rest of the
discipline coherent.

## 10. What this discipline does NOT promise

Even with all of the above PASSed and the DSR diagnostic reported
honestly:

* **Regime is not stationary.** The post-2022 distribution is one
  realization of how crypto trends and reverses. The next regime
  may look unlike anything in the verified window.
* **Pre-venue-history era is unverifiable at the public REST tier.**
  The honest move is to anchor on the verified-window number, not
  the full-sample one — but the operator should know that a
  meaningful chunk of the literature's reported Sharpe lives in
  the unverifiable era and that the public-tier verification has
  structural limits.
* **Forward execution adds operational noise.** Network jitter,
  exchange downtime, partial fills, latency on rebalance — none of
  these are modeled in the audit chain above. The forward harness
  is where they show up.
* **Single-strategy validation is not a portfolio.** A diversified
  retail allocation should not put all of its capital into one
  strategy whose DSR floors to zero under conservative
  multiple-testing.

Putting these limits up-front is the closest thing to a guarantee
the discipline can offer. The strategy survived the falsification
attempts that the methodology covered. The methodology did not
falsify *more* than it could.

## 11. A practitioner's checklist

If you are running a single retail crypto strategy through a
similar discipline, the order I think is correct:

1. **Freeze** the strategy parameters in a hash-pinned dataclass
   before the first validation test. Include inactive knobs.
2. **Independently replay** the strategy on at least one other
   venue. Report signal agreement, same-window Sharpe, and the
   structural limits of the data access. Do not pretend the
   replay covered more than it did.
3. **Decompose the cost regime** by venue. Report per-venue
   verdicts, not a single PASS / FAIL. State the conditional path
   under which a fee-fragile verdict could re-enter (maker tier,
   lower-fee promo).
4. **Check universe bias on both axes** (listing + liquidity).
   Use NaN in an eligibility computation as a fail-loud signal,
   not a silent re-rank.
5. **Run a per-bar truncation audit** at the signal and index
   layers on the verified window. Report 0 mismatches and the
   scope boundary, or stop and re-think the pipeline.
6. **Build forward infrastructure with byte-immutable
   content-hashed vintages** from day one. Verify on read.
7. **Calibrate the behavioral fingerprint on the verified
   window** including the current adverse sub-period. Freeze it.
   Report headroom on each metric. Do not include realized
   return / Sharpe / equity in the bands.
8. **Report DSR on the verified sample** alongside the raw
   Sharpe. Do not anchor forward expectations on the
   venue-unverifiable era.
9. **Pin the change-management rule** by tying every future
   parameter change to a documented research cycle.
10. **Run forward paper trading** for at least one full regime
    transition before considering real money, and treat any
    drawdown beyond the band edge as a stop-and-investigate signal
    not a tactical drawup.

## 12. What I think is novel here

I think most of the techniques above exist somewhere in the
academic literature (DSR is from Bailey & López de Prado, behavioral
fingerprinting is informally common in industry research, the
look-ahead audit pattern is described in Lopez de Prado's *Advances
in Financial Machine Learning*). What I do not think is common is
running them **all together** on a single solo-research validation
chain, **disclosing the failure modes** alongside the headline
result (Kraken not advisable, edge concentration in
venue-unverifiable era, DSR-on-verified ≈ 0), and treating the
**no realized P&L in the behavioral fingerprint** rule as a hard
constraint rather than a guideline.

If I had to name what I would want a reader to take away from this
write-up, in priority order:

1. **Freeze the strategy before you start validating.** The hash
   discipline costs nothing and prevents the most common solo-
   research mistake (tweaking the strategy in response to test
   results, which converts validation into selection).
2. **Disclose the failure modes alongside the headline number.**
   The forward operator inherits all of them; burying them in
   appendix footnotes guarantees that "I deployed and got rugged"
   is the next outcome.
3. **Behavior, not profit, is what the live monitor measures.**
   The strategy is allowed to lose money in an adverse regime;
   what is not allowed is for it to behave differently from how
   the backtest behaved on identically-shaped data. Conflating
   these breaks the monitor.

I will revisit this article after 3-6 months of forward paper
trading and report what the live detector and behavioral
fingerprint actually found. The article above is the
infrastructure write-up; the forward write-up is the empirical
validation of the infrastructure itself.

---

*The complete code, configuration, and per-test findings for the
trade-lab run that motivated this write-up are in
`findings/production_config_v1.md` and the supporting `findings/
validation_*.md` documents in the same repository.*
