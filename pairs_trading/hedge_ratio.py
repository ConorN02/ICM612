"""Hedge ratio estimation: static OLS and rolling Kalman filter.

The hedge ratio (beta) defines the spread for a pair: spread_t = y_t - (alpha + beta * x_t).
`static_ols_hedge_ratio` estimates a single beta over the formation period.
`kalman_filter_hedge_ratio` estimates a time-varying beta, updated at each
time step using only information available up to and including that step
(no lookahead).
"""

from __future__ import annotations

import pandas as pd


def static_ols_hedge_ratio(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """Estimate a single, static hedge ratio via OLS over the given window.

    Fits y_t = alpha + beta * x_t + epsilon_t by ordinary least squares.
    Intended to be fit on the formation period and then held fixed when
    used for a subsequent out-of-sample trading period.

    Args:
        y: Dependent price series.
        x: Independent price series.

    Returns:
        Dict with keys "alpha", "beta", "residuals" (pd.Series indexed like y/x).
    """
    raise NotImplementedError


def kalman_filter_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float,
    obs_covariance: float,
) -> pd.DataFrame:
    """Estimate a time-varying hedge ratio via a Kalman filter.

    Models [alpha_t, beta_t] as a random walk (state transition covariance
    controlled by `delta`) with observation equation y_t = alpha_t + beta_t * x_t
    + epsilon_t (variance `obs_covariance`). At each time step, the filter
    uses only data up to and including that step, so the resulting series
    can be used directly in a no-lookahead backtest.

    Args:
        y: Dependent price series.
        x: Independent price series.
        delta: State transition covariance tuning parameter (smaller values
            produce a smoother, slower-moving beta).
        obs_covariance: Observation noise variance.

    Returns:
        DataFrame indexed like y/x with columns ["alpha", "beta", "spread"],
        where "spread" is the filtered residual y_t - (alpha_t + beta_t * x_t).
    """
    raise NotImplementedError
