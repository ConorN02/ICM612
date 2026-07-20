"""Statistical validation of backtest results.

Tests whether the screened pairs' performance is genuine mean-reversion
signal rather than noise, via:

- `bootstrap_significance_test`: a circular block bootstrap on a single
  daily return series (a pair, or the equal-weighted portfolio), asking
  "how much does resampling the SAME return series jitter the Sharpe
  ratio / cumulative return?" -- i.e. how much of the result could be
  luck in the ORDERING/composition of the actual trades observed.
- `random_pair_benchmark`: a permutation-style test that re-runs the full
  hedge_ratio -> signals -> backtest pipeline on randomly formed pairs
  (not screened for cointegration), asking "does our screening process
  itself add value over picking pairs at random?"
- `compare_to_buy_and_hold`: a passive SPY benchmark comparison, asking
  "is the strategy even competitive with doing nothing sophisticated?"

All three operate on the pipeline's established conventions: daily return
series are dollar P&L per $1 of `config.CAPITAL_PER_PAIR` notional, and
cumulative return series are the running SUM of those daily figures
(additive, not compounded) -- see backtest.py's module docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from pairs_trading import backtest, config, hedge_ratio, metrics, signals


def bootstrap_significance_test(
    strategy_returns: pd.Series,
    n_simulations: int | None = None,
    random_seed: int | None = None,
    block_size: int | None = None,
    risk_free_rate_annual: float | None = None,
    trading_days_per_year: int | None = None,
) -> dict[str, object]:
    """Circular block-bootstrap a daily return series to test how robust its Sharpe/return are.

    Resamples `strategy_returns` `n_simulations` times using a CIRCULAR
    BLOCK bootstrap: rather than resampling individual days (which would
    destroy the autocorrelation structure of the return series -- e.g.
    the run of days that make up one held trade), contiguous blocks of
    `block_size` consecutive days are drawn (with wrap-around at the end
    of the series, hence "circular") and concatenated to build each
    resampled path. This is standard practice for time-series bootstrap
    (Politis & Romano-style block bootstrap) precisely because pairs
    trading returns are strongly autocorrelated within a trade.

    Fully vectorised: every resampled path for every simulation is built
    as a single NumPy index array in one shot (`returns[idx]`, `idx` of
    shape `(n_simulations, len(strategy_returns))`) -- there is no
    Python-level loop over simulations.

    Handles the zero-variance case (e.g. DAL/UAL's trading_period_1, which
    had zero trades and therefore an all-zero daily return series) without
    raising: every resampled path is then also all zeros, so the Sharpe
    ratio (0/0) is reported as NaN for every simulation via
    `metrics.sharpe_ratio`'s own zero-variance convention, while the
    cumulative-return statistics (well-defined even at exactly zero) are
    still reported normally.

    Args:
        strategy_returns: Daily return series (dollar P&L per $1 notional)
            for one pair/period, or the equal-weighted portfolio.
        n_simulations: Number of bootstrap resamples. Defaults to
            `config.N_BOOTSTRAP_SAMPLES`.
        random_seed: Seed for reproducibility. Defaults to
            `config.BOOTSTRAP_RANDOM_SEED`.
        block_size: Block length in trading days. Defaults to
            `config.BOOTSTRAP_BLOCK_SIZE`. Capped at `len(strategy_returns)`.
        risk_free_rate_annual: Passed through to `metrics.sharpe_ratio`.
            Defaults to `config.RISK_FREE_RATE_ANNUAL`.
        trading_days_per_year: Passed through to `metrics.sharpe_ratio`.
            Defaults to `config.TRADING_DAYS_PER_YEAR`.

    Returns:
        Dict with keys "n_simulations", "block_size", "observed_sharpe",
        "observed_cumulative_return" (computed on the actual, un-resampled
        series), "bootstrapped_sharpe", "bootstrapped_cumulative_return"
        (the raw `n_simulations`-length NumPy arrays, for plotting a
        histogram in the report), "mean_sharpe", "sharpe_ci_lower",
        "sharpe_ci_upper", "mean_cumulative_return",
        "cumulative_return_ci_lower", "cumulative_return_ci_upper" (95%
        empirical CIs), and "p_value_cumulative_return_le_zero" (the
        proportion of bootstrapped cumulative returns <= 0 -- a small
        value is evidence the positive return is not just luck).
    """
    if n_simulations is None:
        n_simulations = config.N_BOOTSTRAP_SAMPLES
    if random_seed is None:
        random_seed = config.BOOTSTRAP_RANDOM_SEED
    if block_size is None:
        block_size = config.BOOTSTRAP_BLOCK_SIZE
    if risk_free_rate_annual is None:
        risk_free_rate_annual = config.RISK_FREE_RATE_ANNUAL
    if trading_days_per_year is None:
        trading_days_per_year = config.TRADING_DAYS_PER_YEAR

    returns = strategy_returns.dropna().to_numpy(dtype=float)
    n_obs = len(returns)

    if n_obs == 0:
        empty = np.array([])
        return {
            "n_simulations": n_simulations,
            "block_size": block_size,
            "observed_sharpe": float("nan"),
            "observed_cumulative_return": float("nan"),
            "bootstrapped_sharpe": empty,
            "bootstrapped_cumulative_return": empty,
            "mean_sharpe": float("nan"),
            "sharpe_ci_lower": float("nan"),
            "sharpe_ci_upper": float("nan"),
            "mean_cumulative_return": float("nan"),
            "cumulative_return_ci_lower": float("nan"),
            "cumulative_return_ci_upper": float("nan"),
            "p_value_cumulative_return_le_zero": float("nan"),
        }

    block_size = min(block_size, n_obs)
    n_blocks = int(np.ceil(n_obs / block_size))

    rng = np.random.default_rng(random_seed)
    block_starts = rng.integers(0, n_obs, size=(n_simulations, n_blocks))
    offsets = np.arange(block_size)
    idx = (block_starts[:, :, None] + offsets[None, None, :]) % n_obs
    idx = idx.reshape(n_simulations, -1)[:, :n_obs]

    resampled = returns[idx]  # shape (n_simulations, n_obs)

    cumulative_returns = resampled.sum(axis=1)
    means = resampled.mean(axis=1)
    stds = resampled.std(axis=1, ddof=1)

    annualised_mean = means * trading_days_per_year
    annualised_std = stds * np.sqrt(trading_days_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(
            annualised_std > 0,
            (annualised_mean - risk_free_rate_annual) / annualised_std,
            np.nan,
        )

    valid_sharpes = sharpes[~np.isnan(sharpes)]
    if len(valid_sharpes) > 0:
        mean_sharpe = float(valid_sharpes.mean())
        sharpe_ci_lower = float(np.percentile(valid_sharpes, 2.5))
        sharpe_ci_upper = float(np.percentile(valid_sharpes, 97.5))
    else:
        mean_sharpe = float("nan")
        sharpe_ci_lower = float("nan")
        sharpe_ci_upper = float("nan")

    observed_sharpe = metrics.sharpe_ratio(
        pd.Series(returns), risk_free_rate_annual, trading_days_per_year
    )

    return {
        "n_simulations": n_simulations,
        "block_size": block_size,
        "observed_sharpe": observed_sharpe,
        "observed_cumulative_return": float(returns.sum()),
        "bootstrapped_sharpe": sharpes,
        "bootstrapped_cumulative_return": cumulative_returns,
        "mean_sharpe": mean_sharpe,
        "sharpe_ci_lower": sharpe_ci_lower,
        "sharpe_ci_upper": sharpe_ci_upper,
        "mean_cumulative_return": float(cumulative_returns.mean()),
        "cumulative_return_ci_lower": float(np.percentile(cumulative_returns, 2.5)),
        "cumulative_return_ci_upper": float(np.percentile(cumulative_returns, 97.5)),
        "p_value_cumulative_return_le_zero": float((cumulative_returns <= 0).mean()),
    }


def _build_random_pair_hedge_row(
    price_panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    formation_start: str,
    formation_end: str,
) -> pd.Series:
    """Fit a formation-period hedge ratio for an arbitrary (non-screened) pair.

    Uses `hedge_ratio.static_ols_hedge_ratio`'s default direction
    (`ticker_a` as the dependent leg) since, unlike the real selected
    pairs, a randomly drawn pair has no `screening.cointegration_test`
    result to say which regression direction showed cointegration
    evidence -- there is no such evidence being tested for here at all.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        ticker_a: First leg of the random pair (regressed as dependent).
        ticker_b: Second leg of the random pair (independent).
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".

    Returns:
        Series with the same schema as one row of
        `hedge_ratio.build_hedge_ratios_for_selected_pairs`'s output
        ("pair", "dependent_ticker", "independent_ticker", "alpha",
        "beta", "formation_spread_mean", "formation_spread_std"), suitable
        for `signals.generate_zscore_series`/`generate_positions` and
        `backtest.backtest_pair`.
    """
    hedge = hedge_ratio.static_ols_hedge_ratio(price_panel, ticker_a, ticker_b, formation_start, formation_end)
    formation_spread = hedge_ratio.compute_spread(
        price_panel,
        hedge["dependent_ticker"],
        hedge["independent_ticker"],
        hedge["alpha"],
        hedge["beta"],
        formation_start,
        formation_end,
    )
    return pd.Series(
        {
            "pair": f"{ticker_a}/{ticker_b}",
            "dependent_ticker": hedge["dependent_ticker"],
            "independent_ticker": hedge["independent_ticker"],
            "alpha": hedge["alpha"],
            "beta": hedge["beta"],
            "formation_spread_mean": float(formation_spread.mean()),
            "formation_spread_std": float(formation_spread.std()),
        }
    )


def _percentile_and_p_value(simulated_returns: np.ndarray, actual_value: float) -> tuple[float, float]:
    """Locate an observed value within a simulated null distribution.

    Args:
        simulated_returns: Array of simulated values under the null
            (e.g. `random_pair_benchmark`'s randomly-formed-pair
            portfolio returns).
        actual_value: The observed value to locate (e.g. the real
            screened-pairs portfolio's cumulative return).

    Returns:
        Tuple of `(percentile, p_value)`: `percentile` (0-100) is the
        fraction of `simulated_returns` strictly below `actual_value`;
        `p_value` is the fraction of `simulated_returns` >= `actual_value`
        (small means `actual_value` is unusually good under the null).
    """
    percentile = float((simulated_returns < actual_value).mean() * 100)
    p_value = float((simulated_returns >= actual_value).mean())
    return percentile, p_value


def random_pair_benchmark(
    price_panel: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    period_start: str,
    period_end: str,
    actual_portfolio_return: float,
    n_simulations: int | None = None,
    n_pairs_per_simulation: int = config.N_PAIRS_TO_SELECT,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Permutation test: does formation-period screening add value over random pair selection?

    Each simulation draws `n_pairs_per_simulation` (default 3, matching
    our real selection) random pairs from the FULL 10-ticker candidate
    universe (`config.CANDIDATE_PAIRS`' tickers), NOT requiring
    cointegration -- this is deliberate: the question being tested is
    whether our screening process (correlation + SSD + cointegration
    gate-then-rank) specifically added value over picking pairs
    arbitrarily, so the null-hypothesis draws must not themselves be
    pre-filtered for cointegration. Each ticker pair within a draw is
    distinct, but the same ticker may appear in more than one of the
    `n_pairs_per_simulation` pairs within one simulation (a minor
    simplification -- see Limitations below).

    For each random pair, runs the SAME hedge_ratio -> signals -> backtest
    pipeline used for the real selected pairs: `static_ols_hedge_ratio` on
    [formation_start, formation_end], then `generate_positions` over
    [period_start, period_end]. Random pairs deliberately use
    `config.DEFAULT_ENTRY_THRESHOLD`/`config.DEFAULT_EXIT_THRESHOLD`
    rather than a per-pair `threshold_sensitivity_grid` search: giving
    every randomly drawn pair an individually formation-optimised
    threshold would let the null distribution benefit from the same
    parameter-tuning our real pairs get, which would understate how much
    of our result comes from screening specifically (as opposed to
    threshold tuning, which every random draw would then also enjoy).
    Using a single fixed threshold for the whole null distribution keeps
    that source of edge attributed correctly.

    The `n_pairs_per_simulation` random pairs' cumulative net returns are
    equal-weighted (same convention as `backtest.compute_average_pair_return`)
    into one simulated portfolio return per simulation, building a null
    distribution against which `actual_portfolio_return` is compared.

    Limitations (documented per the brief's request, for accurate
    reporting): (1) allowing ticker reuse across the `n_pairs_per_simulation`
    pairs within one simulation is a simplification -- excluding it would
    require sampling without replacement across pairs, which is a minor
    change but was not made here; (2) using a single fixed threshold for
    every random pair, while methodologically deliberate (see above), does
    mean the comparison is "screening + fixed thresholds" vs. "no
    screening + fixed thresholds", not "screening + tuned thresholds" vs.
    "no screening + tuned thresholds" -- the latter would be more
    expensive (one grid search per random pair per simulation) and was
    judged not to change the qualitative conclusion enough to justify the
    cost.

    Args:
        price_panel: Full price panel, columns=tickers, indexed by date,
            covering formation and the trading period being tested.
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".
        period_start: Trading period start date, "YYYY-MM-DD".
        period_end: Trading period end date, "YYYY-MM-DD".
        actual_portfolio_return: The real, screened-pairs equal-weighted
            portfolio's cumulative net return over [period_start, period_end]
            (e.g. from `backtest.compute_average_pair_return`), to compare
            against the null distribution.
        n_simulations: Number of random-pair-portfolio draws. Defaults to
            `config.N_RANDOM_PAIRS_BENCHMARK`.
        n_pairs_per_simulation: Pairs drawn per simulation. Defaults to
            `config.N_PAIRS_TO_SELECT` (3), matching the real portfolio.
        random_seed: Seed for reproducibility. Defaults to
            `config.BOOTSTRAP_RANDOM_SEED`.

    Returns:
        Dict with keys "n_simulations", "n_pairs_per_simulation",
        "actual_portfolio_return", "simulated_returns" (the raw
        `n_simulations`-length NumPy array of simulated portfolio
        returns), "simulated_mean", "simulated_std",
        "percentile_of_actual" (0-100, where `actual_portfolio_return`
        falls in the simulated distribution), "p_value_actual_ge_random"
        (proportion of simulated returns >= the actual return -- small
        means the actual result is unusually good relative to random pair
        formation, i.e. evidence screening added value).
    """
    if n_simulations is None:
        n_simulations = config.N_RANDOM_PAIRS_BENCHMARK
    if random_seed is None:
        random_seed = config.BOOTSTRAP_RANDOM_SEED

    universe = np.array(sorted({ticker for pair in config.CANDIDATE_PAIRS for ticker in pair}))
    rng = np.random.default_rng(random_seed)

    simulated_returns = np.empty(n_simulations, dtype=float)

    for sim_idx in tqdm(range(n_simulations), desc="random_pair_benchmark"):
        pair_returns = np.empty(n_pairs_per_simulation, dtype=float)

        for pair_idx in range(n_pairs_per_simulation):
            ticker_a, ticker_b = rng.choice(universe, size=2, replace=False)

            hedge_row = _build_random_pair_hedge_row(
                price_panel, ticker_a, ticker_b, formation_start, formation_end
            )
            zscore_series = signals.generate_zscore_series(price_panel, hedge_row, period_start, period_end)
            positions_df = signals.generate_positions(
                zscore_series,
                entry_threshold=config.DEFAULT_ENTRY_THRESHOLD,
                exit_threshold=config.DEFAULT_EXIT_THRESHOLD,
                max_holding_days=config.MAX_HOLDING_DAYS,
                stop_loss_threshold=config.STOP_LOSS_THRESHOLD,
            )
            pair_result = backtest.backtest_pair(
                price_panel,
                positions_df,
                hedge_row,
                hedge_row["dependent_ticker"],
                hedge_row["independent_ticker"],
            )
            pair_returns[pair_idx] = pair_result["cumulative_net_return"].iloc[-1]

        simulated_returns[sim_idx] = pair_returns.mean()

    percentile_of_actual, p_value = _percentile_and_p_value(simulated_returns, actual_portfolio_return)

    return {
        "n_simulations": n_simulations,
        "n_pairs_per_simulation": n_pairs_per_simulation,
        "actual_portfolio_return": actual_portfolio_return,
        "simulated_returns": simulated_returns,
        "simulated_mean": float(simulated_returns.mean()),
        "simulated_std": float(simulated_returns.std()),
        "percentile_of_actual": percentile_of_actual,
        "p_value_actual_ge_random": p_value,
    }


def compare_to_buy_and_hold(
    strategy_cumulative_return: pd.Series,
    benchmark_prices: pd.Series,
    risk_free_rate_annual: float | None = None,
    trading_days_per_year: int | None = None,
) -> dict[str, float]:
    """Compare the strategy's cumulative return/Sharpe against a passive buy-and-hold benchmark.

    `strategy_cumulative_return` is the pipeline's additive cumulative
    net return series (e.g. from `backtest.compute_average_pair_return`);
    `benchmark_prices` is a raw price series (typically
    `config.BENCHMARK_TICKER`, SPY) over the same window, whose
    buy-and-hold return naturally compounds (`final/initial - 1`) rather
    than accumulating additively -- this is how a real buy-and-hold
    position behaves, so it is not forced into the strategy's additive
    convention. Comparing an additive strategy return against a
    compounding benchmark return introduces a small apples-to-apples
    discrepancy at large return magnitudes, but is standard practice and
    negligible over the ~2-year trading periods used here.

    Args:
        strategy_cumulative_return: Strategy's cumulative net return
            series, indexed by date.
        benchmark_prices: Benchmark price series, indexed by date,
            covering the same period.
        risk_free_rate_annual: Passed through to `metrics.sharpe_ratio`.
            Defaults to `config.RISK_FREE_RATE_ANNUAL`.
        trading_days_per_year: Passed through to `metrics.sharpe_ratio`.
            Defaults to `config.TRADING_DAYS_PER_YEAR`.

    Returns:
        Dict with keys "strategy_cumulative_return",
        "benchmark_cumulative_return", "strategy_sharpe",
        "benchmark_sharpe", "excess_return" (strategy minus benchmark
        cumulative return).
    """
    strategy_cumulative_return = strategy_cumulative_return.sort_index()
    strategy_daily_returns = strategy_cumulative_return.diff()
    strategy_daily_returns.iloc[0] = strategy_cumulative_return.iloc[0]

    benchmark_prices = benchmark_prices.sort_index()
    benchmark_daily_returns = benchmark_prices.pct_change().dropna()
    benchmark_cumulative_return = float(benchmark_prices.iloc[-1] / benchmark_prices.iloc[0] - 1.0)

    strategy_sharpe = metrics.sharpe_ratio(strategy_daily_returns, risk_free_rate_annual, trading_days_per_year)
    benchmark_sharpe = metrics.sharpe_ratio(benchmark_daily_returns, risk_free_rate_annual, trading_days_per_year)

    strategy_cumulative_return_value = float(strategy_cumulative_return.iloc[-1])

    return {
        "strategy_cumulative_return": strategy_cumulative_return_value,
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "excess_return": strategy_cumulative_return_value - benchmark_cumulative_return,
    }
