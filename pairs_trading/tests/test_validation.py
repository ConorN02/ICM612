"""Unit tests for validation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading import validation


def test_bootstrap_ci_contains_observed_statistic() -> None:
    """The 95% CI from bootstrapping a return series should contain that
    series' own observed cumulative return and Sharpe ratio (basic CI
    coverage sanity check) on a known synthetic distribution."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    returns = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)

    result = validation.bootstrap_significance_test(returns, n_simulations=2000, random_seed=7, block_size=20)

    assert len(result["bootstrapped_cumulative_return"]) == 2000
    assert len(result["bootstrapped_sharpe"]) == 2000
    assert (
        result["cumulative_return_ci_lower"]
        <= result["observed_cumulative_return"]
        <= result["cumulative_return_ci_upper"]
    )
    assert result["sharpe_ci_lower"] <= result["observed_sharpe"] <= result["sharpe_ci_upper"]


def test_bootstrap_all_zero_returns_handled_without_raising() -> None:
    """An all-zero daily return series (e.g. DAL/UAL's trading_period_1,
    zero trades) must return NaN Sharpe stats without raising, while
    cumulative-return stats remain well-defined at exactly zero."""
    dates = pd.date_range("2022-01-01", periods=250, freq="B")
    returns = pd.Series([0.0] * 250, index=dates)

    result = validation.bootstrap_significance_test(returns, n_simulations=500, random_seed=1)

    assert pd.isna(result["observed_sharpe"])
    assert pd.isna(result["mean_sharpe"])
    assert pd.isna(result["sharpe_ci_lower"])
    assert pd.isna(result["sharpe_ci_upper"])
    assert result["observed_cumulative_return"] == pytest.approx(0.0)
    assert result["mean_cumulative_return"] == pytest.approx(0.0)
    assert result["cumulative_return_ci_lower"] == pytest.approx(0.0)
    assert result["cumulative_return_ci_upper"] == pytest.approx(0.0)
    assert result["p_value_cumulative_return_le_zero"] == pytest.approx(1.0)


def test_bootstrap_empty_series_handled_without_raising() -> None:
    """An entirely empty/all-NaN return series must also return NaNs
    rather than raising (e.g. index errors on an empty array)."""
    returns = pd.Series([], dtype=float)

    result = validation.bootstrap_significance_test(returns, n_simulations=100, random_seed=1)

    assert pd.isna(result["observed_sharpe"])
    assert pd.isna(result["mean_cumulative_return"])
    assert len(result["bootstrapped_sharpe"]) == 0


def test_percentile_and_p_value_small_synthetic_case() -> None:
    """A small, hand-checkable synthetic null distribution should produce
    the exact expected percentile and p-value for a known observed value."""
    simulated = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    # 0.035 falls between 0.03 and 0.04: 3 of 5 simulated values are below
    # it (60th percentile), and 2 of 5 are >= it (p-value 0.4).
    percentile, p_value = validation._percentile_and_p_value(simulated, 0.035)
    assert percentile == pytest.approx(60.0)
    assert p_value == pytest.approx(2 / 5)

    # Below every simulated value -> 0th percentile, p-value 1 (nothing
    # beats it, so it's not "unusually good" -- everything is >= it).
    percentile_low, p_value_low = validation._percentile_and_p_value(simulated, 0.0)
    assert percentile_low == pytest.approx(0.0)
    assert p_value_low == pytest.approx(1.0)

    # Above every simulated value -> 100th percentile, p-value 0 (nothing
    # in the null distribution matches or beats it).
    percentile_high, p_value_high = validation._percentile_and_p_value(simulated, 1.0)
    assert percentile_high == pytest.approx(100.0)
    assert p_value_high == pytest.approx(0.0)


def test_compare_to_buy_and_hold_computes_excess_return() -> None:
    """excess_return should equal strategy minus benchmark cumulative
    return, and benchmark_cumulative_return should match the simple
    final/initial - 1 buy-and-hold formula."""
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    strategy_cumulative_return = pd.Series([0.0, 0.01, 0.02, 0.015, 0.03], index=dates)
    benchmark_prices = pd.Series([100.0, 101.0, 102.0, 101.0, 105.0], index=dates)

    result = validation.compare_to_buy_and_hold(strategy_cumulative_return, benchmark_prices)

    expected_benchmark_return = 105.0 / 100.0 - 1.0
    assert result["benchmark_cumulative_return"] == pytest.approx(expected_benchmark_return)
    assert result["strategy_cumulative_return"] == pytest.approx(0.03)
    assert result["excess_return"] == pytest.approx(0.03 - expected_benchmark_return)
