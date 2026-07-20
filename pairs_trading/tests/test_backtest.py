"""Unit tests for backtest.py.

Stubs pending implementation of backtest.run_backtest and
backtest.apply_transaction_costs. Fill these in alongside the
implementation in Phase 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading import backtest


@pytest.fixture
def simple_pair_data() -> dict[str, pd.Series]:
    """Small synthetic price/hedge-ratio/position series for backtest tests.

    Returns:
        Dict with keys "price_y", "price_x", "hedge_ratio", "positions",
        each a pd.Series over a 10-day synthetic date range.
    """
    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    price_y = pd.Series(np.linspace(100, 110, 10), index=dates, name="Y")
    price_x = pd.Series(np.linspace(50, 55, 10), index=dates, name="X")
    hedge_ratio = pd.Series(1.0, index=dates)
    positions = pd.Series([0, 0, 1, 1, 1, 0, 0, -1, -1, 0], index=dates)
    return {
        "price_y": price_y,
        "price_x": price_x,
        "hedge_ratio": hedge_ratio,
        "positions": positions,
    }


def test_run_backtest_no_lookahead(simple_pair_data: dict[str, pd.Series]) -> None:
    """A position decided at time t should only affect P&L realised after t."""
    pytest.skip("Pending backtest.run_backtest implementation")


def test_run_backtest_flat_positions_produce_zero_pnl(simple_pair_data: dict[str, pd.Series]) -> None:
    """An all-zero position series should produce zero net P&L throughout."""
    pytest.skip("Pending backtest.run_backtest implementation")


def test_apply_transaction_costs_charges_only_on_turnover() -> None:
    """Transaction costs should be zero wherever the position is unchanged."""
    pytest.skip("Pending backtest.apply_transaction_costs implementation")
