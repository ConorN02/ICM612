"""Pair screening over the formation period.

Implements the Gatev, Goetzmann & Rouwenhorst (2006) sum-of-squared-deviations
(SSD) distance metric, correlation screening, and Engle-Granger cointegration
testing (both a "generic" ADF critical-value comparison and the statistically
correct Engle-Granger/MacKinnon critical values via statsmodels).
"""

from __future__ import annotations

import pandas as pd


def compute_pairwise_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the pairwise Pearson correlation matrix of daily returns.

    Args:
        prices: DataFrame of price levels, columns=tickers, indexed by date,
            restricted to the formation period.

    Returns:
        Symmetric DataFrame of pairwise return correlations, indexed and
        columned by ticker.
    """
    raise NotImplementedError


def compute_ssd(prices: pd.DataFrame, candidate_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Compute the sum of squared deviations (SSD) distance for each candidate pair.

    Follows Gatev et al. (2006): prices are normalised to a cumulative
    return index (starting at 1.0) over the formation period, and the SSD
    is the sum of squared differences between the two normalised series.

    Args:
        prices: DataFrame of price levels, columns=tickers, indexed by date,
            restricted to the formation period.
        candidate_pairs: Pairs to score, as (ticker_a, ticker_b) tuples.

    Returns:
        DataFrame with one row per pair and columns ["ticker_a", "ticker_b", "ssd"],
        sorted ascending by "ssd" (lowest distance first).
    """
    raise NotImplementedError


def engle_granger_test_generic(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """Run a naive Engle-Granger cointegration test using generic ADF critical values.

    Regresses y on x via OLS, then tests the residuals for a unit root
    using standard (non-cointegration-adjusted) ADF critical values from
    `config.GENERIC_ADF_CRITICAL_VALUES`. Included for pedagogical
    comparison against `engle_granger_test_proper`, since using generic ADF
    critical values on regression residuals is a common but statistically
    invalid shortcut (Engle & Granger 1987 show the correct critical values
    are more negative).

    Args:
        y: Dependent price series (formation period).
        x: Independent price series (formation period).

    Returns:
        Dict with keys "adf_statistic", "p_value", "critical_values",
        "is_cointegrated" (bool, using the 5% generic critical value).
    """
    raise NotImplementedError


def engle_granger_test_proper(y: pd.Series, x: pd.Series) -> dict[str, float]:
    """Run the Engle-Granger cointegration test with correct MacKinnon critical values.

    Uses `statsmodels.tsa.stattools.coint`, which applies the correct
    response-surface (MacKinnon) critical values for the two-step
    Engle-Granger residual-based cointegration test.

    Args:
        y: Dependent price series (formation period).
        x: Independent price series (formation period).

    Returns:
        Dict with keys "test_statistic", "p_value", "critical_values",
        "is_cointegrated" (bool, using `config.COINTEGRATION_SIGNIFICANCE`).
    """
    raise NotImplementedError


def screen_pairs(
    prices: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Run the full formation-period screen over all candidate pairs.

    Orchestrates correlation, SSD, and both cointegration tests, applying
    the thresholds in config.py (MIN_CORRELATION, SSD_TOP_N,
    COINTEGRATION_SIGNIFICANCE) to shortlist tradeable pairs.

    Args:
        prices: DataFrame of price levels, columns=tickers, indexed by date,
            restricted to the formation period.
        candidate_pairs: Pairs to screen, as (ticker_a, ticker_b) tuples.

    Returns:
        DataFrame with one row per candidate pair, including correlation,
        SSD, both Engle-Granger test results, and a final "passes_screen"
        boolean column.
    """
    raise NotImplementedError
