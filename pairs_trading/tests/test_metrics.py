"""Unit tests for metrics.py.

Stubs pending implementation of the metrics functions. Fill these in
alongside the implementation in Phase 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading import metrics


@pytest.fixture
def rising_equity_curve() -> pd.Series:
    """Monotonically rising equity curve with a known drawdown in the middle.

    Returns:
        Series indexed by a 6-day business date range: 1.0, 1.1, 1.05, 1.2, 1.15, 1.3.
    """
    dates = pd.date_range("2021-01-01", periods=6, freq="B")
    return pd.Series([1.0, 1.1, 1.05, 1.2, 1.15, 1.3], index=dates)


def test_cumulative_return_matches_start_end_ratio(rising_equity_curve: pd.Series) -> None:
    """Cumulative return should equal (final / initial) - 1."""
    pytest.skip("Pending metrics.cumulative_return implementation")


def test_max_drawdown_is_negative_and_correct(rising_equity_curve: pd.Series) -> None:
    """Known drawdown from 1.1 to 1.05 should be -1/22 (~-4.5%)."""
    pytest.skip("Pending metrics.max_drawdown implementation")


def test_sharpe_ratio_zero_for_zero_excess_return() -> None:
    """A constant-return series with zero volatility should not raise, and mean
    excess return of zero should give a Sharpe ratio of zero."""
    pytest.skip("Pending metrics.sharpe_ratio implementation")


def test_half_life_positive_for_mean_reverting_series() -> None:
    """A synthetic AR(1) mean-reverting series should yield a positive, finite half-life."""
    pytest.skip("Pending metrics.half_life_of_mean_reversion implementation")
