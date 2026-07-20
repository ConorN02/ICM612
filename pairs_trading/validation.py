"""Statistical validation of backtest results.

Tests whether the screened pairs' performance is genuine mean-reversion
signal rather than noise, via bootstrap/permutation resampling against a
random-pair benchmark, and compares strategy performance to a passive
SPY buy-and-hold benchmark.
"""

from __future__ import annotations

import pandas as pd


def bootstrap_significance_test(
    strategy_returns: pd.Series,
    n_samples: int,
    random_seed: int,
) -> dict[str, float]:
    """Bootstrap the strategy's Sharpe ratio to build a confidence interval.

    Resamples blocks of `strategy_returns` with replacement `n_samples`
    times, recomputes the Sharpe ratio on each resample, and reports the
    empirical distribution.

    Args:
        strategy_returns: Periodic strategy return series.
        n_samples: Number of bootstrap resamples to draw.
        random_seed: Seed for reproducibility.

    Returns:
        Dict with keys "mean_sharpe", "ci_lower", "ci_upper", "p_value_gt_zero".
    """
    raise NotImplementedError


def random_pair_benchmark(
    prices: pd.DataFrame,
    n_random_pairs: int,
    random_seed: int,
) -> pd.DataFrame:
    """Backtest a sample of randomly-formed (non-screened) pairs as a null benchmark.

    Draws `n_random_pairs` random ticker pairs from `prices`, runs each
    through the same signal/backtest pipeline as the screened pairs, and
    returns their performance for comparison against the screened pairs'
    results (i.e. does formation-period screening actually add value?).

    Args:
        prices: DataFrame of price levels, columns=tickers, indexed by date.
        n_random_pairs: Number of random pairs to sample and backtest.
        random_seed: Seed for reproducibility.

    Returns:
        DataFrame with one row per random pair and performance metric columns
        (matching the schema produced by metrics.py functions).
    """
    raise NotImplementedError


def compare_to_buy_and_hold(
    strategy_equity_curve: pd.Series,
    benchmark_prices: pd.Series,
) -> dict[str, float]:
    """Compare strategy performance to a passive buy-and-hold benchmark.

    Args:
        strategy_equity_curve: Strategy equity/NAV series, indexed by date.
        benchmark_prices: Benchmark (e.g. SPY) price series, indexed by date,
            covering the same period.

    Returns:
        Dict with keys "strategy_cumulative_return", "benchmark_cumulative_return",
        "strategy_sharpe", "benchmark_sharpe", "excess_return".
    """
    raise NotImplementedError
