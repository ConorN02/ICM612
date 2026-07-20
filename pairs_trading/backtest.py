"""Backtest engine: converts each pair's position series into dollar P&L.

Uses the Gatev, Goetzmann & Rouwenhorst (2006) $1-per-pair sizing
convention, extended with the regression hedge ratio from hedge_ratio.py
in place of the classic distance method's naive 1:1 hedge: each day a
position is held, it is treated as $config.CAPITAL_PER_PAIR notional long
the dependent leg financed by beta * $config.CAPITAL_PER_PAIR (short) in
the independent leg, re-marked fresh each day (no compounding/rebalancing
state to track) rather than growing/shrinking a single invested balance.
Concretely, the daily dollar P&L per $1 of capital is:

    (dependent_return_t - beta * independent_return_t) * CAPITAL_PER_PAIR

which is exactly the day-over-day dollar change in the pair's spread
(since spread_t = dependent_t - alpha - beta * independent_t, and alpha is
constant), scaled to a $1 reference notional. Cumulative return is the
running SUM of these daily dollar P&L figures (additive, not compounded),
consistent with the fixed daily notional convention above.

Every function here operates on the position held going INTO day t (i.e.
positions_df["position"].shift(1)) applied to day t's realised price
change, never day t's own (still-unknown-at-the-time) position -- this is
the no-lookahead contract the whole pipeline depends on.
"""

from __future__ import annotations

import pandas as pd

from pairs_trading import config


def compute_pair_daily_returns(
    price_panel: pd.DataFrame,
    positions_df: pd.DataFrame,
    hedge_ratio_row: pd.Series,
    ticker_a: str,
    ticker_b: str,
) -> pd.Series:
    """Compute one pair's daily dollar P&L (gross, before transaction costs) for one period.

    On each day, the position HELD GOING INTO that day (yesterday's close,
    via `positions_df["position"].shift(1)`) is applied to that day's
    price return: `position_{t-1} * (dependent_return_t - beta *
    independent_return_t) * config.CAPITAL_PER_PAIR`. Today's own position
    (which may reflect a trade decided using today's z-score) never
    affects today's P&L -- it only starts contributing from tomorrow.

    Price returns are computed from the full `price_panel` (not sliced to
    `positions_df`'s window) so the first day of a trading period still
    gets a correctly computed return from the preceding trading day (which
    may fall in a different period); this is harmless even when it can't
    be computed (e.g. the very first day of the whole dataset) since the
    position going into that day is always 0 regardless.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        positions_df: DataFrame indexed by date with a "position" column
            (-1/0/+1), e.g. the direct output of `signals.generate_positions`
            for this pair and period.
        hedge_ratio_row: One row of
            `hedge_ratio.build_hedge_ratios_for_selected_pairs()`'s output,
            with "dependent_ticker", "independent_ticker", "beta".
        ticker_a: Dependent-side ticker; must equal
            `hedge_ratio_row["dependent_ticker"]`.
        ticker_b: Independent-side ticker; must equal
            `hedge_ratio_row["independent_ticker"]`.

    Returns:
        Series of daily dollar P&L, indexed like `positions_df`, named
        "gross_return".

    Raises:
        ValueError: If {ticker_a, ticker_b} doesn't match
            hedge_ratio_row's dependent/independent tickers.
    """
    dependent_ticker = hedge_ratio_row["dependent_ticker"]
    independent_ticker = hedge_ratio_row["independent_ticker"]
    if {ticker_a, ticker_b} != {dependent_ticker, independent_ticker}:
        raise ValueError(
            f"ticker_a/ticker_b ({ticker_a}, {ticker_b}) do not match hedge_ratio_row's "
            f"dependent/independent tickers ({dependent_ticker}, {independent_ticker})."
        )
    beta = float(hedge_ratio_row["beta"])

    dependent_returns = price_panel[dependent_ticker].pct_change()
    independent_returns = price_panel[independent_ticker].pct_change()
    spread_return = (dependent_returns - beta * independent_returns).reindex(positions_df.index)

    position_going_into_day = positions_df["position"].shift(1).fillna(0)

    daily_dollar_pnl = (position_going_into_day * spread_return * config.CAPITAL_PER_PAIR).fillna(0.0)
    return daily_dollar_pnl.rename("gross_return")


def apply_transaction_costs(
    daily_returns: pd.Series,
    positions_df: pd.DataFrame,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    """Deduct transaction costs from gross daily returns on position-change days only.

    `cost_bps` is charged on BOTH legs (a factor of 2) as a fraction of
    `config.CAPITAL_PER_PAIR`, on any day the position changes relative to
    the day before -- opening from flat, closing to flat, or (were it
    possible; `signals.generate_positions` never flips directly between
    +1 and -1 in one day) flipping. Days the position is simply held
    unchanged incur zero cost, matching the brief's requirement that
    holding is free and only trading incurs cost.

    Args:
        daily_returns: Gross daily dollar P&L series (e.g. from
            `compute_pair_daily_returns`), indexed by date.
        positions_df: DataFrame indexed by date with a "position" column
            (-1/0/+1), aligned to `daily_returns`.
        cost_bps: Transaction cost in basis points per leg, per trade
            (entry or exit). Typically `config.TRANSACTION_COST_BPS`.

    Returns:
        Tuple `(net_returns, total_cost_series)`: `net_returns` is
        `daily_returns` minus `total_cost_series`; `total_cost_series` is
        the per-day dollar cost (0 on no-change days), both indexed like
        `daily_returns`.
    """
    position = positions_df["position"]
    prior_position = position.shift(1).fillna(0)
    position_changed = (position - prior_position).abs()

    cost_per_leg_change = (cost_bps / 10_000) * config.CAPITAL_PER_PAIR
    total_cost_series = (position_changed * 2 * cost_per_leg_change).rename("cost")

    net_returns = (daily_returns - total_cost_series).rename("net_return")
    return net_returns, total_cost_series


def backtest_pair(
    price_panel: pd.DataFrame,
    positions_df: pd.DataFrame,
    hedge_ratio_row: pd.Series,
    ticker_a: str,
    ticker_b: str,
    cost_bps: float | None = None,
) -> pd.DataFrame:
    """Run the full daily P&L backtest for one pair over one period.

    Orchestrates `compute_pair_daily_returns` (gross P&L) and
    `apply_transaction_costs` (net of trading costs), then accumulates
    both into running (additive, not compounded) cumulative return series
    -- see the module docstring for why cumulative return is a simple sum
    here rather than a compounded product.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        positions_df: DataFrame indexed by date with a "position" column,
            for this pair and period (e.g. `signals.generate_positions`'s
            output, or a single pair/period slice of
            `signals.build_signals_for_selected_pairs()["positions"]`
            re-indexed by date).
        hedge_ratio_row: One row of
            `hedge_ratio.build_hedge_ratios_for_selected_pairs()`'s output.
        ticker_a: Dependent-side ticker; must equal
            `hedge_ratio_row["dependent_ticker"]`.
        ticker_b: Independent-side ticker; must equal
            `hedge_ratio_row["independent_ticker"]`.
        cost_bps: Transaction cost in basis points per leg, per trade.
            Defaults to `config.TRANSACTION_COST_BPS` if not given.

    Returns:
        Tidy DataFrame with columns "date", "position", "gross_return",
        "cost", "net_return", "cumulative_gross_return",
        "cumulative_net_return", one row per date in `positions_df`.
    """
    if cost_bps is None:
        cost_bps = config.TRANSACTION_COST_BPS

    gross_return = compute_pair_daily_returns(price_panel, positions_df, hedge_ratio_row, ticker_a, ticker_b)
    net_return, cost = apply_transaction_costs(gross_return, positions_df, cost_bps)

    result = pd.DataFrame(
        {
            "position": positions_df["position"],
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
        }
    )
    result["cumulative_gross_return"] = result["gross_return"].cumsum()
    result["cumulative_net_return"] = result["net_return"].cumsum()

    return result.reset_index().rename(columns={"index": "date"})


def backtest_all_selected_pairs(
    price_panel: pd.DataFrame,
    signals_dict: dict[str, pd.DataFrame],
    hedge_ratios_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run `backtest_pair` for every selected pair across all three periods.

    For each pair in `hedge_ratios_df`, slices
    `signals_dict["positions"]` (the long-format table from
    `signals.build_signals_for_selected_pairs`) down to that pair for
    each of "formation", "trading_period_1", "trading_period_2" in turn,
    and backtests it independently -- exactly as `signals.py` generated
    positions independently per period, with no state carried across a
    period boundary.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        signals_dict: Output of `signals.build_signals_for_selected_pairs`,
            used for its "positions" table (columns "pair", "period",
            "date", "position", ...).
        hedge_ratios_df: Output of
            `hedge_ratio.build_hedge_ratios_for_selected_pairs`.

    Returns:
        Long-format DataFrame with columns "pair", "period", "date",
        "position", "gross_return", "cost", "net_return",
        "cumulative_gross_return", "cumulative_net_return" -- one row per
        (pair, period, date).
    """
    all_positions = signals_dict["positions"]
    periods = list(all_positions["period"].unique())

    result_tables = []
    for _, hedge_row in hedge_ratios_df.iterrows():
        pair = hedge_row["pair"]
        ticker_a = hedge_row["dependent_ticker"]
        ticker_b = hedge_row["independent_ticker"]

        for period in periods:
            period_positions = all_positions[
                (all_positions["pair"] == pair) & (all_positions["period"] == period)
            ].set_index("date")

            pair_result = backtest_pair(price_panel, period_positions, hedge_row, ticker_a, ticker_b)
            pair_result.insert(0, "pair", pair)
            pair_result.insert(1, "period", period)
            result_tables.append(pair_result)

    return pd.concat(result_tables, ignore_index=True)


def compute_average_pair_return(backtest_results_df: pd.DataFrame, period: str) -> pd.Series:
    """Equal-weight the cumulative net return across the 3 selected pairs for one period.

    Per Gatev et al. (2006)'s $1-per-pair convention, the "portfolio"
    result is simply the equal-weighted average across pairs (each pair
    already sized to the same $1 notional in `compute_pair_daily_returns`)
    -- this is the headline result for the report.

    Args:
        backtest_results_df: Output of `backtest_all_selected_pairs`.
        period: Period to average over, e.g. "trading_period_1".

    Returns:
        Series of the equal-weighted average cumulative net return across
        pairs, indexed by date, named
        "average_cumulative_net_return_{period}".

    Raises:
        ValueError: If no rows in `backtest_results_df` match `period`.
    """
    period_df = backtest_results_df[backtest_results_df["period"] == period]
    if period_df.empty:
        raise ValueError(f"No backtest results found for period={period!r}.")

    pivoted = period_df.pivot(index="date", columns="pair", values="cumulative_net_return")
    average_return = pivoted.mean(axis=1)
    return average_return.rename(f"average_cumulative_net_return_{period}")
