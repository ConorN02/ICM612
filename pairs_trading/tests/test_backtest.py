"""Unit tests for backtest.py."""

from __future__ import annotations

import pandas as pd
import pytest

from pairs_trading import backtest, config


def test_compute_pair_daily_returns_known_price_move() -> None:
    """A position held for exactly one day, with a known price move, must
    produce the expected dollar P&L using the pair's beta."""
    dates = pd.date_range("2022-01-01", periods=3, freq="B")
    price_panel = pd.DataFrame({"Y": [100.0, 100.0, 103.0], "X": [50.0, 50.0, 52.0]}, index=dates)
    hedge_ratio_row = pd.Series({"dependent_ticker": "Y", "independent_ticker": "X", "beta": 2.0})
    # Enters (position=1) on day 1, closes on day 2 -- held for exactly one day.
    positions_df = pd.DataFrame({"position": [0, 1, 0]}, index=dates)

    result = backtest.compute_pair_daily_returns(price_panel, positions_df, hedge_ratio_row, "Y", "X")

    y_return = 103.0 / 100.0 - 1.0
    x_return = 52.0 / 50.0 - 1.0
    expected_day2_pnl = 1 * (y_return - 2.0 * x_return) * config.CAPITAL_PER_PAIR

    assert result.iloc[0] == pytest.approx(0.0)  # no position going into day 0
    assert result.iloc[1] == pytest.approx(0.0)  # position just opened on day 1, no P&L yet
    assert result.iloc[2] == pytest.approx(expected_day2_pnl)


def test_compute_pair_daily_returns_no_lookahead() -> None:
    """Today's P&L must be driven by the position held going INTO today
    (yesterday's close), never today's own position -- verified with a
    deliberately lagged synthetic check where using today's position
    instead would give a different, wrong answer."""
    dates = pd.date_range("2022-01-01", periods=2, freq="B")
    price_panel = pd.DataFrame({"Y": [100.0, 110.0], "X": [50.0, 50.0]}, index=dates)
    hedge_ratio_row = pd.Series({"dependent_ticker": "Y", "independent_ticker": "X", "beta": 0.0})
    # Position is 1 on day 0, then flat on day 1: a lookahead bug (using
    # position[t] instead of position[t-1]) would read the day-1 return
    # (10%) as belonging to a flat (0) position and report 0.0 P&L on day
    # 1, instead of correctly attributing it to the position of 1 held
    # going into day 1.
    positions_df = pd.DataFrame({"position": [1, 0]}, index=dates)

    result = backtest.compute_pair_daily_returns(price_panel, positions_df, hedge_ratio_row, "Y", "X")

    assert result.iloc[0] == pytest.approx(0.0)  # no position existed before day 0
    assert result.iloc[1] == pytest.approx(0.10 * config.CAPITAL_PER_PAIR)  # driven by day 0's position


def test_apply_transaction_costs_charges_only_on_position_change() -> None:
    """Transaction costs must be zero on days the position is held
    unchanged, and nonzero exactly on days it opens, closes, or changes."""
    dates = pd.date_range("2022-01-01", periods=6, freq="B")
    # 0 -> 1 (open, day1), hold, 1 -> 0 (close, day3), hold, 0 -> -1 (open, day5).
    positions_df = pd.DataFrame({"position": [0, 1, 1, 0, 0, -1]}, index=dates)
    gross_returns = pd.Series(0.0, index=dates)
    cost_bps = 10.0

    net_returns, cost_series = backtest.apply_transaction_costs(gross_returns, positions_df, cost_bps)

    expected_cost_per_change = 2 * (cost_bps / 10_000) * config.CAPITAL_PER_PAIR
    expected_costs = [0.0, expected_cost_per_change, 0.0, expected_cost_per_change, 0.0, expected_cost_per_change]

    assert cost_series.tolist() == pytest.approx(expected_costs)
    assert net_returns.tolist() == pytest.approx([-c for c in expected_costs])


def test_backtest_pair_produces_expected_columns_and_cumulative_sums() -> None:
    """backtest_pair should orchestrate gross P&L and costs into the tidy
    output schema, with cumulative columns as running sums of the daily
    gross/net series."""
    dates = pd.date_range("2022-01-01", periods=3, freq="B")
    price_panel = pd.DataFrame({"Y": [100.0, 102.0, 101.0], "X": [50.0, 50.0, 50.0]}, index=dates)
    hedge_ratio_row = pd.Series({"dependent_ticker": "Y", "independent_ticker": "X", "beta": 0.0})
    positions_df = pd.DataFrame({"position": [1, 1, 0]}, index=dates)

    result = backtest.backtest_pair(price_panel, positions_df, hedge_ratio_row, "Y", "X", cost_bps=0.0)

    assert list(result.columns) == [
        "date",
        "position",
        "gross_return",
        "cost",
        "net_return",
        "cumulative_gross_return",
        "cumulative_net_return",
    ]
    assert result["cumulative_gross_return"].iloc[-1] == pytest.approx(result["gross_return"].sum())
    assert result["cumulative_net_return"].iloc[-1] == pytest.approx(result["net_return"].sum())
    # With cost_bps=0, net and gross must be identical.
    pd.testing.assert_series_equal(
        result["gross_return"], result["net_return"], check_names=False
    )
