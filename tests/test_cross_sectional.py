import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.cross_sectional import (
    CrossSectionalResult,
    run_cross_sectional_momentum,
)


def _candles(closes, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC")
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


def _universe(n_assets: int = 3, n_bars: int = 400, seed: int = 0):
    """Build a multi-asset universe with distinct momentum profiles."""
    rng = np.random.default_rng(seed)
    closes = {}
    for i in range(n_assets):
        # Distinct slopes so the rank order is non-trivial.
        slope = 0.5 + i * 0.3
        noise = rng.normal(0, 1.0, n_bars)
        path = 100 + slope * np.arange(n_bars) + noise
        closes[f"ASSET_{i}/USDT"] = _candles(path.tolist())
    return closes


def test_empty_input_returns_empty_result():
    res = run_cross_sectional_momentum({})
    assert isinstance(res, CrossSectionalResult)
    assert res.equity.empty
    assert res.weights.empty


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        run_cross_sectional_momentum(_universe(), lookback_days=0)
    with pytest.raises(ValueError):
        run_cross_sectional_momentum(_universe(), rebalance_days=0)
    with pytest.raises(ValueError):
        run_cross_sectional_momentum(_universe(), top_k=0)
    with pytest.raises(ValueError):
        run_cross_sectional_momentum(_universe(), weighting="something_weird")
    with pytest.raises(ValueError):
        run_cross_sectional_momentum(_universe(), vol_lookback=1)


def test_weights_never_exceed_one_in_sum():
    res = run_cross_sectional_momentum(
        _universe(n_assets=5),
        lookback_days=30,
        rebalance_days=7,
        top_k=3,
        weighting="equal",
    )
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (res.weights.to_numpy() >= 0.0).all()


def test_equal_weight_basket_sums_to_one_when_filter_passes():
    """When ``top_k`` assets are eligible and weighted equally, the row
    sum is exactly 1.0 (apart from warmup rows held flat)."""
    res = run_cross_sectional_momentum(
        _universe(n_assets=4),
        lookback_days=30,
        rebalance_days=7,
        top_k=3,
        weighting="equal",
    )
    invested = res.weights[res.weights.sum(axis=1) > 0]
    if not invested.empty:
        # All invested rows sum to exactly 1.0 with equal weighting.
        np.testing.assert_allclose(invested.sum(axis=1).to_numpy(), 1.0)


def test_top_k_selects_strongest_assets():
    """Among assets with distinct slopes, the top_k=2 basket on a fully
    eligible week should be the two highest-slope assets."""
    closes = {
        "WEAK/USDT": _candles(np.linspace(100, 110, 200).tolist()),     # +10%
        "MID/USDT": _candles(np.linspace(100, 150, 200).tolist()),      # +50%
        "STRONG/USDT": _candles(np.linspace(100, 300, 200).tolist()),   # +200%
    }
    res = run_cross_sectional_momentum(
        closes,
        lookback_days=30,
        rebalance_days=7,
        top_k=2,
        weighting="equal",
    )
    invested = res.weights[res.weights.sum(axis=1) > 0]
    # On every invested row the WEAK asset should not appear, but MID
    # and STRONG should each have 0.5.
    assert (invested["WEAK/USDT"] == 0.0).all()
    np.testing.assert_allclose(invested["MID/USDT"].to_numpy(), 0.5)
    np.testing.assert_allclose(invested["STRONG/USDT"].to_numpy(), 0.5)


def test_btc_gate_forces_cash():
    """When BTC is below its SMA, the portfolio sits in cash regardless
    of the cross-section."""
    universe = _universe(n_assets=3)
    # BTC: monotone downtrend over the entire window so it's always
    # under its own SMA(50).
    btc = _candles(np.linspace(400, 50, 400).tolist())
    res = run_cross_sectional_momentum(
        universe,
        lookback_days=30,
        rebalance_days=7,
        top_k=3,
        weighting="equal",
        btc_candles=btc,
        btc_gate_sma_period=50,
    )
    # No exposure at all after warmup — BTC gate is closed.
    invested_rows = (res.weights.sum(axis=1) > 0).sum()
    assert invested_rows == 0
    assert res.total_return == 0.0


def test_inverse_vol_weights_assets_inversely():
    """An asset that is exactly 4x more volatile should get ~1/4 the
    weight in an inverse-vol basket."""
    n = 400
    rng = np.random.default_rng(0)
    low_vol = 100 + np.linspace(0, 100, n) + rng.normal(0, 0.5, n)
    high_vol = 100 + np.linspace(0, 100, n) + rng.normal(0, 2.0, n)
    closes = {
        "CALM/USDT": _candles(low_vol.tolist()),
        "WILD/USDT": _candles(high_vol.tolist()),
    }
    res = run_cross_sectional_momentum(
        closes,
        lookback_days=30,
        rebalance_days=14,
        top_k=2,
        weighting="inverse_vol",
        vol_lookback=30,
    )
    invested = res.weights[res.weights.sum(axis=1) > 0]
    # CALM should consistently take more weight than WILD.
    assert (invested["CALM/USDT"] > invested["WILD/USDT"]).mean() > 0.9


def test_no_lookahead_in_portfolio_weights():
    """Appending future bars must not change weights on the prefix."""
    rng = np.random.default_rng(2)
    base = {}
    for i in range(3):
        path = 100 + 0.3 * np.arange(300) + rng.normal(0, 1.0, 300)
        base[f"ASSET_{i}/USDT"] = _candles(path.tolist())
    extended = {
        k: _candles(v["close"].tolist() + [1e6, 1e-6, 1e6, 1e-6, 1e6])
        for k, v in base.items()
    }
    res_base = run_cross_sectional_momentum(base, lookback_days=30, rebalance_days=7)
    res_ext = run_cross_sectional_momentum(extended, lookback_days=30, rebalance_days=7)
    common = min(len(res_base.weights), len(res_ext.weights))
    np.testing.assert_array_equal(
        res_base.weights.iloc[:common].to_numpy(),
        res_ext.weights.iloc[:common].to_numpy(),
    )


def test_equity_grows_when_universe_trends_up():
    res = run_cross_sectional_momentum(
        _universe(n_assets=4, n_bars=500),
        lookback_days=30,
        rebalance_days=7,
        top_k=2,
    )
    assert res.total_return > 0
    assert res.equity.iloc[-1] > res.initial_capital


def test_fees_and_slippage_are_tracked():
    res = run_cross_sectional_momentum(
        _universe(n_assets=4, n_bars=400),
        lookback_days=30,
        rebalance_days=7,
        top_k=2,
        fee_rate=0.001,
        slippage_rate=0.0005,
    )
    # Any non-trivial run should incur some turnover -> non-zero costs.
    assert res.total_fees > 0
    assert res.total_slippage > 0


def test_no_eligible_asset_means_cash():
    """When all candidate assets have a negative trailing return, the
    basket is empty and the portfolio sits in cash."""
    n = 400
    closes = {
        f"DOWN_{i}/USDT": _candles(np.linspace(500, 50, n).tolist())
        for i in range(3)
    }
    res = run_cross_sectional_momentum(
        closes, lookback_days=30, rebalance_days=7, top_k=2,
    )
    assert (res.weights.sum(axis=1) == 0.0).all()
    assert res.total_return == 0.0
