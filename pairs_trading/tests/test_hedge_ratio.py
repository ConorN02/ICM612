"""Unit tests for hedge_ratio.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading import hedge_ratio


def test_static_ols_hedge_ratio_recovers_known_alpha_beta() -> None:
    """A synthetic pair with Y = alpha + beta*X + small noise should recover
    alpha/beta close to the true generating values, and report which ticker
    was the dependent/independent leg."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")

    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(dates))), index=dates, name="X")
    true_alpha, true_beta = 5.0, 1.5
    noise = rng.normal(0, 0.1, len(dates))
    y = pd.Series(true_alpha + true_beta * x.to_numpy() + noise, index=dates, name="Y")

    price_panel = pd.concat([y, x], axis=1)

    result = hedge_ratio.static_ols_hedge_ratio(
        price_panel, "Y", "X", dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    )

    assert result["dependent_ticker"] == "Y"
    assert result["independent_ticker"] == "X"
    assert result["alpha"] == pytest.approx(true_alpha, abs=0.2)
    assert result["beta"] == pytest.approx(true_beta, abs=0.05)
    assert result["r_squared"] > 0.99


def test_static_ols_hedge_ratio_honours_explicit_direction() -> None:
    """Passing direction="X~Y" should regress X on Y (X dependent), the
    reverse of the default ticker_a-dependent convention -- this is what
    lets the caller match whatever direction screening.py's cointegration
    test actually found evidence for."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")

    y = pd.Series(50 + np.cumsum(rng.normal(0, 1, len(dates))), index=dates, name="Y")
    x = pd.Series(2.0 + 0.5 * y.to_numpy() + rng.normal(0, 0.1, len(dates)), index=dates, name="X")

    price_panel = pd.concat([y, x], axis=1)

    result = hedge_ratio.static_ols_hedge_ratio(
        price_panel,
        "Y",
        "X",
        dates[0].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d"),
        direction="X~Y",
    )

    assert result["dependent_ticker"] == "X"
    assert result["independent_ticker"] == "Y"
    assert result["beta"] == pytest.approx(0.5, abs=0.05)


def test_spread_zscore_uses_supplied_mean_std_not_its_own() -> None:
    """spread_zscore must standardise using the formation_mean/formation_std
    arguments, never the input series' own mean/std -- this is the
    no-lookahead-bias contract signals.py depends on."""
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    spread = pd.Series([10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 10.2, 9.8, 10.1, 9.9], index=dates)

    # Deliberately pass formation stats wildly different from the series' own
    # (own mean/std is ~10.0/~0.6).
    formation_mean, formation_std = 0.0, 1.0

    z = hedge_ratio.spread_zscore(spread, formation_mean, formation_std)

    expected = (spread - formation_mean) / formation_std
    pd.testing.assert_series_equal(z, expected, check_names=False)

    own_mean, own_std = spread.mean(), spread.std()
    z_if_recomputed = (spread - own_mean) / own_std
    assert not np.allclose(z.to_numpy(), z_if_recomputed.to_numpy())


def test_spread_zscore_rejects_non_positive_formation_std() -> None:
    """A zero or negative formation_std would divide by zero/flip sign
    silently; this must be rejected rather than produce garbage z-scores."""
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)

    with pytest.raises(ValueError):
        hedge_ratio.spread_zscore(spread, formation_mean=0.0, formation_std=0.0)


def test_estimate_half_life_recovers_known_half_life_from_ou_process() -> None:
    """A synthetic OU/AR(1) spread simulated with a known mean-reversion
    speed theta should yield a half-life close to ln(2)/theta."""
    rng = np.random.default_rng(7)
    n = 3000
    theta = 0.05  # mean-reversion speed per step
    mu = 0.0
    sigma = 0.2

    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] + theta * (mu - spread[t - 1]) + rng.normal(0, sigma)

    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    spread_series = pd.Series(spread, index=dates)

    result = hedge_ratio.estimate_half_life(spread_series)

    expected_half_life = np.log(2) / theta
    assert result["half_life_days"] == pytest.approx(expected_half_life, rel=0.25)
    assert 0.0 <= result["r_squared"] <= 1.0


def test_estimate_half_life_is_infinite_for_a_random_walk() -> None:
    """A pure random walk (no mean reversion) should report an infinite
    half-life rather than a misleading finite number."""
    rng = np.random.default_rng(3)
    n = 500
    spread = np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    spread_series = pd.Series(spread, index=dates)

    result = hedge_ratio.estimate_half_life(spread_series)

    # A pure random walk has lambda ~ 0; allow either sign of noise to push
    # the estimate slightly positive (finite but very long) or non-negative
    # (infinite) -- the key contract is it must not report a short,
    # confidently mean-reverting half-life.
    assert result["half_life_days"] > 50 or result["half_life_days"] == float("inf")
