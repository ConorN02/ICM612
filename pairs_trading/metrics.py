"""Performance metrics for backtested pair (or portfolio) equity curves."""

from __future__ import annotations

import pandas as pd


def cumulative_return(equity_curve: pd.Series) -> float:
    """Compute total cumulative return over the equity curve.

    Args:
        equity_curve: Equity/NAV series indexed by date.

    Returns:
        Cumulative return as a fraction (e.g. 0.25 for +25%).
    """
    raise NotImplementedError


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float,
    periods_per_year: int,
) -> float:
    """Compute the annualised Sharpe ratio of a returns series.

    Args:
        returns: Periodic (e.g. daily) return series.
        risk_free_rate_annual: Annualised risk-free rate, as a fraction.
        periods_per_year: Number of return periods per year (e.g. 252 for daily).

    Returns:
        Annualised Sharpe ratio.
    """
    raise NotImplementedError


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown of an equity curve.

    Args:
        equity_curve: Equity/NAV series indexed by date.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.15 for -15%).
    """
    raise NotImplementedError


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """Estimate the half-life of mean reversion of a spread series.

    Fits spread_t - spread_{t-1} = theta * (spread_{t-1} - mean) + epsilon_t
    via OLS (Ornstein-Uhlenbeck discretisation) and converts the mean
    reversion speed theta into a half-life in trading days.

    Args:
        spread: Spread series, indexed by date.

    Returns:
        Half-life in trading days.
    """
    raise NotImplementedError


def trade_statistics(positions: pd.Series, net_pnl: pd.Series) -> dict[str, float]:
    """Compute discrete trade-level statistics from a position and P&L series.

    A "trade" is a contiguous run of non-zero position between two flat
    periods.

    Args:
        positions: Position series (+1/0/-1), indexed by date.
        net_pnl: Per-period net P&L series, aligned to `positions`.

    Returns:
        Dict with keys "n_trades", "win_rate", "avg_holding_period_days",
        "avg_pnl_per_trade".
    """
    raise NotImplementedError
