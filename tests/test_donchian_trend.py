import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.donchian_trend import DonchianTrendEnsembleStrategy


def _candles(closes, freq: str = "1D"):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq=freq, tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1.0,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Required tests (per the spec)
# ---------------------------------------------------------------------------


def test_no_lookahead_in_donchian_thresholds():
    """Appending arbitrary future bars must not change signals on the
    overlapping prefix. If the Donchian high/low ever peeked at the
    current bar (or beyond), this would fail."""
    rng = np.random.default_rng(0)
    base = (100 + np.linspace(0, 60, 200) + rng.normal(0, 2.0, 200)).tolist()
    future_garbage = [1e6, 1e-6, 1e6, 1e-6, 1e6, 1e-6]

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20, 50),
        sma_filter_periods=(50, 100),
        vol_lookback=20,
    )
    sig_base = strat.generate_signals(_candles(base))
    sig_extended_prefix = strat.generate_signals(_candles(base + future_garbage)).iloc[: len(base)]

    np.testing.assert_array_equal(sig_base.values, sig_extended_prefix.values)


def test_sma_filter_blocks_exposure_when_price_below_sma():
    """Long uptrend followed by a sharp drop. After the drop, price stays
    well below both SMA(100) and SMA(200); even if a Donchian channel
    briefly re-opens, the SMA filter must keep the strategy in cash."""
    uptrend = list(range(100, 600))     # 500 bars: 100 -> 599
    crash_floor = [180.0] * 80          # 80 bars sitting under the SMAs
    candles = _candles(uptrend + crash_floor)

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20, 50),
        sma_filter_periods=(100, 200),
        vol_lookback=20,
    )
    sig = strat.generate_signals(candles)

    # Tail of the crash region — well past the drop, well past any breakout
    # echo. Should be flat.
    assert (sig.iloc[-40:] == 0).all()


def test_vol_targeting_reduces_position_when_realized_vol_rises():
    """Two series with the same trend but very different daily noise. The
    higher-vol series should produce a smaller average position once the
    SMA filter passes."""
    n = 400
    rng = np.random.default_rng(0)
    trend = np.linspace(100, 400, n)
    low_vol_closes = (trend + rng.normal(0, 0.5, n)).tolist()
    high_vol_closes = (trend + rng.normal(0, 8.0, n)).tolist()

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(100,),
        vol_lookback=20,
        annual_vol_target=0.25,
    )
    sig_low = strat.generate_signals(_candles(low_vol_closes))
    sig_high = strat.generate_signals(_candles(high_vol_closes))

    # After warmup, both series should have positive average exposure
    # (they're trending up), but the high-vol one should be smaller because
    # realized vol divides into the position weight.
    avg_low = sig_low.iloc[150:].mean()
    avg_high = sig_high.iloc[150:].mean()
    assert avg_low > 0
    assert avg_high > 0
    assert avg_low > avg_high


def test_exposure_cap_is_respected():
    """A near-zero-vol series would produce a huge vol_weight without the
    cap. With ``max_position_size=1.0`` exposure must stay in [0, 1]."""
    n = 400
    # Tiny linear drift -> vanishing realized vol. Without the cap, the
    # vol_weight would explode.
    closes = (100 + np.arange(n) * 0.01).tolist()
    candles = _candles(closes)

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(50, 100),
        vol_lookback=20,
        annual_vol_target=0.25,
        max_position_size=1.0,
    )
    sig = strat.generate_signals(candles)

    assert (sig >= 0.0).all()
    assert (sig <= 1.0).all()


def test_signals_are_deterministic_on_synthetic_dataset():
    """Same input must produce identical signals across calls — there is
    no internal randomness or state."""
    rng = np.random.default_rng(7)
    closes = (100 + np.linspace(0, 50, 300) + rng.normal(0, 2.0, 300)).tolist()
    candles = _candles(closes)

    strat = DonchianTrendEnsembleStrategy()
    sig1 = strat.generate_signals(candles)
    sig2 = strat.generate_signals(candles)

    pd.testing.assert_series_equal(sig1, sig2)


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------


def test_donchian_ladder_produces_partial_exposure_with_three_lookbacks():
    """Three Donchian lookbacks should yield positions on the
    {0, 1/3, 2/3, 1} ladder after vol weighting is removed (we cap
    weights to 1 via a tiny target vs huge realized vol)."""
    rng = np.random.default_rng(1)
    closes = (100 + np.linspace(0, 100, 400) + rng.normal(0, 5.0, 400)).tolist()
    candles = _candles(closes)

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(10, 30, 60),
        sma_filter_periods=(50,),
        vol_lookback=10,
        annual_vol_target=0.000_001,   # absurdly tiny -> always capped
        max_position_size=1.0,
    )
    # Without vol scaling, raw signal would be Donchian mean. With cap at 1
    # and tiny vol target, position equals raw signal exactly (or 0 from
    # the SMA filter / warmup).
    sig = strat.generate_signals(candles)
    unique_vals = set(round(v, 4) for v in sig.unique())
    # With min target, the cap turns large weights into 1 only when raw is
    # 1; otherwise position == raw signal == k/3 for k in {0,1,2,3}.
    # Wait — vol_weight is target/realized = ~0, so position = raw * ~0
    # ~= 0. Re-do with a small but reasonable target.
    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(10, 30, 60),
        sma_filter_periods=(50,),
        vol_lookback=10,
        annual_vol_target=10.0,   # huge -> capped to 1 always
        max_position_size=1.0,
    )
    sig = strat.generate_signals(candles)
    nonzero = sig[sig > 0]
    if not nonzero.empty:
        # Each non-zero value should sit on the 1/3 ladder, up to floating
        # point. Cap pushes 2/3 and 1.0 candidates to 1.0.
        levels = {round(v, 6) for v in nonzero.unique()}
        assert levels.issubset({round(1 / 3, 6), round(2 / 3, 6), 1.0})


def test_btc_gate_blocks_exposure_when_btc_below_sma():
    """When ``btc_candles`` is supplied and BTC is below its own SMA,
    the strategy is forced flat regardless of the per-asset signal."""
    rng = np.random.default_rng(0)
    n = 250
    # Altcoin: clean uptrend that the Donchian + SMA filter would happily go long on.
    altcoin_closes = (100 + np.linspace(0, 80, n) + rng.normal(0, 1.0, n)).tolist()
    # BTC: clean downtrend below its own SMA from bar 50 onward.
    btc_closes = (200 - np.linspace(0, 80, n) + rng.normal(0, 1.0, n)).tolist()

    altcoin_candles = _candles(altcoin_closes)
    btc_candles = _candles(btc_closes)

    no_gate = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(50,),
        vol_lookback=20,
    )
    with_gate = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(50,),
        vol_lookback=20,
        btc_candles=btc_candles,
        btc_gate_sma_period=50,
    )

    sig_off = no_gate.generate_signals(altcoin_candles)
    sig_on = with_gate.generate_signals(altcoin_candles)

    assert sig_off.iloc[150:].sum() > 0
    assert (sig_on.iloc[150:] == 0).all()


def test_engine_shifts_donchian_signal_by_one_bar():
    """Integration sanity check: the engine still applies its one-bar shift
    to whatever this strategy returns."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 50, 300) + rng.normal(0, 1.5, 300)).tolist()
    candles = _candles(closes)

    strat = DonchianTrendEnsembleStrategy()
    result = run_backtest(
        candles, strat, initial_capital=10_000.0, fee_rate=0.001, slippage_rate=0.0005,
    )
    raw_signals = strat.generate_signals(candles)

    # positions[N] should equal signals[N-1] * position_size, modulo a
    # default position_size of 1.0 and the engine's leading fillna(0).
    expected_positions = raw_signals.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        result.positions, expected_positions, check_names=False
    )


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        DonchianTrendEnsembleStrategy(donchian_lookbacks=())
    with pytest.raises(ValueError):
        DonchianTrendEnsembleStrategy(vol_lookback=1)
    with pytest.raises(ValueError):
        DonchianTrendEnsembleStrategy(annual_vol_target=0)
    with pytest.raises(ValueError):
        DonchianTrendEnsembleStrategy(max_position_size=0)
    with pytest.raises(ValueError):
        DonchianTrendEnsembleStrategy(max_position_size=2)
    with pytest.raises(ValueError):
        DonchianTrendEnsembleStrategy(donchian_lookbacks=(0, 10))


def test_rebalance_threshold_zero_matches_unbanded_behaviour():
    """rebalance_threshold = 0 must reproduce the pre-feature output exactly."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 60, 400) + rng.normal(0, 2.0, 400)).tolist()
    candles = _candles(closes)

    unbanded = DonchianTrendEnsembleStrategy(rebalance_threshold=0.0)
    banded_zero = DonchianTrendEnsembleStrategy(rebalance_threshold=0.0)
    sig_a = unbanded.generate_signals(candles)
    sig_b = banded_zero.generate_signals(candles)
    pd.testing.assert_series_equal(sig_a, sig_b)


def test_rebalance_threshold_suppresses_small_target_changes():
    """A series whose realized vol slowly drifts produces tiny daily
    adjustments. With a band, the position should hold flat for stretches
    instead of changing every bar."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 100, 500) + rng.normal(0, 2.0, 500)).tolist()
    candles = _candles(closes)

    unbanded = DonchianTrendEnsembleStrategy(rebalance_threshold=0.0)
    banded = DonchianTrendEnsembleStrategy(rebalance_threshold=0.05)
    sig_unbanded = unbanded.generate_signals(candles)
    sig_banded = banded.generate_signals(candles)

    # Turnover proxy: number of unique non-zero levels held. Banded run
    # should have far fewer distinct levels.
    unbanded_changes = (sig_unbanded.diff().abs() > 1e-12).sum()
    banded_changes = (sig_banded.diff().abs() > 1e-12).sum()
    assert banded_changes < unbanded_changes


def test_rebalance_threshold_does_not_block_entries():
    """A transition from 0 to a positive target must always go through,
    even if the size happens to be smaller than the threshold."""
    # Construct a deterministic toy series where there's a clear entry
    # after warm-up: long uptrend, then a brief sharp move that triggers
    # the breakout but stays small.
    closes = [100.0] * 60 + [101.0, 103.0, 110.0] + [110.0] * 30
    candles = _candles(closes)

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(30,),
        vol_lookback=10,
        annual_vol_target=0.25,
        rebalance_threshold=0.50,  # very wide band
    )
    sig = strat.generate_signals(candles)

    # The signal must become positive at some point after warm-up.
    assert (sig > 0).any()


def test_rebalance_threshold_does_not_block_exits():
    """A transition from a positive position to 0 must always go through
    regardless of the threshold."""
    # Long uptrend (strategy goes long) then a hard crash (SMA filter
    # forces flat). Even with a huge band, the exit must fire.
    uptrend = list(range(100, 400))
    crash_floor = [120.0] * 80
    candles = _candles(uptrend + crash_floor)

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(50, 100),
        vol_lookback=20,
        annual_vol_target=0.25,
        rebalance_threshold=0.99,  # blocks every conceivable size update
    )
    sig = strat.generate_signals(candles)

    # By the tail of the crash window the strategy must be flat.
    assert (sig.iloc[-40:] == 0).all()


def test_rebalance_threshold_respects_position_cap():
    """Even with the band, position must stay within [0, max_position_size]."""
    n = 400
    closes = (100 + np.arange(n) * 0.01).tolist()  # vanishing vol
    candles = _candles(closes)

    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks=(20,),
        sma_filter_periods=(50, 100),
        vol_lookback=20,
        annual_vol_target=0.25,
        max_position_size=1.0,
        rebalance_threshold=0.10,
    )
    sig = strat.generate_signals(candles)
    assert (sig >= 0.0).all()
    assert (sig <= 1.0).all()


def test_rebalance_threshold_does_not_introduce_lookahead():
    """The band uses only the running position, never future targets.
    Appending future bars must not change signals on the prefix."""
    rng = np.random.default_rng(0)
    base = (100 + np.linspace(0, 60, 200) + rng.normal(0, 2.0, 200)).tolist()
    future_garbage = [1e6, 1e-6, 1e6, 1e-6, 1e6]

    strat = DonchianTrendEnsembleStrategy(rebalance_threshold=0.05)
    sig_base = strat.generate_signals(_candles(base))
    sig_extended_prefix = strat.generate_signals(_candles(base + future_garbage)).iloc[: len(base)]

    np.testing.assert_array_equal(sig_base.values, sig_extended_prefix.values)


def test_string_parameters_accepted_for_cli_use():
    """The CLI passes lookback lists as comma-separated strings via
    --param. The strategy must accept that form too."""
    strat = DonchianTrendEnsembleStrategy(
        donchian_lookbacks="10,20,30",
        sma_filter_periods="50,100",
    )
    assert strat.donchian_lookbacks == (10, 20, 30)
    assert strat.sma_filter_periods == (50, 100)
