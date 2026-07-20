"""Unit tests for metrics.py."""

from __future__ import annotations

import pandas as pd
import pytest

from pairs_trading import metrics


def test_sharpe_ratio_matches_hand_computed_value() -> None:
    """sharpe_ratio should match the formula computed independently from
    the same daily return series, on synthetic (non-constant) returns."""
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    returns = pd.Series([0.001, 0.002, -0.001, 0.0015, 0.0005], index=dates)
    risk_free_rate_annual = 0.02
    trading_days_per_year = 252

    expected = (returns.mean() * trading_days_per_year - risk_free_rate_annual) / (
        returns.std() * (trading_days_per_year**0.5)
    )

    result = metrics.sharpe_ratio(
        returns, risk_free_rate_annual=risk_free_rate_annual, trading_days_per_year=trading_days_per_year
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_returns_nan_for_zero_variance_series() -> None:
    """A zero-variance return series (e.g. a pair with zero trades in a
    period) must return NaN, not raise or return +/-inf."""
    dates = pd.date_range("2022-01-01", periods=6, freq="B")
    returns = pd.Series([0.0] * 6, index=dates)

    result = metrics.sharpe_ratio(returns)

    assert pd.isna(result)


def test_max_drawdown_identifies_known_peak_and_trough() -> None:
    """A synthetic additive cumulative-return series with a known largest
    decline should report the correct magnitude, peak_date, and trough_date."""
    dates = pd.date_range("2022-01-01", periods=6, freq="B")
    # Peaks at index1 (0.05) and index3 (0.08); largest decline is
    # index3 -> index4 (0.08 -> 0.02, a drop of 0.06), bigger than
    # index1 -> index2 (0.05 -> 0.03, a drop of 0.02).
    cumulative = pd.Series([0.0, 0.05, 0.03, 0.08, 0.02, 0.10], index=dates)

    result = metrics.max_drawdown(cumulative)

    assert result["max_drawdown"] == pytest.approx(-0.06)
    assert result["peak_date"] == dates[3]
    assert result["trough_date"] == dates[4]


def test_max_drawdown_zero_for_flat_series() -> None:
    """A flat (zero-trade) cumulative-return series should report zero
    drawdown rather than raising or dividing by a zero/negative peak."""
    dates = pd.date_range("2022-01-01", periods=4, freq="B")
    cumulative = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates)

    result = metrics.max_drawdown(cumulative)

    assert result["max_drawdown"] == pytest.approx(0.0)


def test_trade_stats_handles_zero_trades_without_raising() -> None:
    """A fully flat position/exit_reason series (e.g. DAL/UAL's
    trading_period_1) must return zeros/NaNs, not raise."""
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    positions_df = pd.DataFrame(
        {
            "date": dates,
            "position": [0, 0, 0, 0, 0],
            "exit_reason": [None, None, None, None, None],
            "net_return": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = metrics.trade_stats(positions_df)

    assert result["n_trades"] == 0
    assert pd.isna(result["win_rate"])
    assert pd.isna(result["avg_holding_days"])
    assert result["exit_reason_counts"] == {"mean_reversion": 0, "max_holding": 0, "stop_loss": 0}


def test_trade_stats_computes_correct_win_rate_and_holding_days() -> None:
    """A two-trade synthetic series (one winning, one losing, different
    exit reasons) should produce the correct trade count, win rate,
    average holding period, and exit_reason breakdown."""
    dates = pd.date_range("2022-01-01", periods=8, freq="B")
    # Trade 1: open days 1-2 (position=1), closes day 3 (mean_reversion).
    # Trade 2: open day 5 only (position=-1), closes day 6 (stop_loss).
    positions = [0, 1, 1, 0, 0, -1, 0, 0]
    exit_reason = [None, None, None, "mean_reversion", None, None, "stop_loss", None]
    # Per backtest.py's convention, a trade's P&L is the sum of net_return
    # from the day after entry through the close day inclusive.
    net_return = [0.0, 0.0, 0.02, 0.03, 0.0, 0.0, -0.05, 0.0]

    positions_df = pd.DataFrame(
        {"date": dates, "position": positions, "exit_reason": exit_reason, "net_return": net_return}
    )

    result = metrics.trade_stats(positions_df)

    assert result["n_trades"] == 2
    assert result["win_rate"] == pytest.approx(0.5)
    assert result["avg_holding_days"] == pytest.approx(1.5)
    assert result["exit_reason_counts"] == {"mean_reversion": 1, "max_holding": 0, "stop_loss": 1}
