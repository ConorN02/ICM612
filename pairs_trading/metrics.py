"""Performance and risk metrics computed from backtest.py's output.

Operates on the pipeline's established conventions: daily return series are
dollar P&L per $1 of `config.CAPITAL_PER_PAIR` notional (see backtest.py),
and cumulative return series are the running SUM of those daily figures
(additive, not compounded) -- so `max_drawdown` here measures the largest
absolute decline in cumulative return from its running peak, not a
percentage-of-peak drawdown (which wouldn't be meaningful for a series that
can cross zero).

Functions operate on either a single pair/period (`sharpe_ratio`,
`max_drawdown`, `trade_stats`, `summarise_pair_period`) or across every
selected pair and period (`summarise_all_pairs`) or the equal-weighted
portfolio (`summarise_portfolio`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading import config


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate_annual: float | None = None,
    trading_days_per_year: int | None = None,
) -> float:
    """Compute the annualised Sharpe ratio of a daily return series.

    `(mean(daily_returns) * trading_days_per_year - risk_free_rate_annual)
    / (std(daily_returns) * sqrt(trading_days_per_year))`.

    Returns NaN (rather than raising, or returning +/-inf) when the return
    series has zero variance -- e.g. DAL/UAL's trading_period_1, which had
    zero trades and therefore a daily return series of all zeros. A Sharpe
    ratio is undefined, not "very good" or "very bad", when there is no
    variability to divide by; reporting NaN keeps that distinction visible
    in downstream tables/report text rather than silently emitting +/-inf
    or crashing the summary pipeline.

    Args:
        daily_returns: Daily return (dollar P&L per $1 notional) series.
        risk_free_rate_annual: Annualised risk-free rate, as a fraction.
            Defaults to `config.RISK_FREE_RATE_ANNUAL` if not given.
        trading_days_per_year: Number of trading days per year, for
            annualising. Defaults to `config.TRADING_DAYS_PER_YEAR` if not
            given.

    Returns:
        Annualised Sharpe ratio, or NaN if the return series is empty or
        has zero (or undefined, e.g. single-observation) variance.
    """
    if risk_free_rate_annual is None:
        risk_free_rate_annual = config.RISK_FREE_RATE_ANNUAL
    if trading_days_per_year is None:
        trading_days_per_year = config.TRADING_DAYS_PER_YEAR

    daily_returns = daily_returns.dropna()
    std = daily_returns.std()

    if daily_returns.empty or pd.isna(std) or std == 0:
        return float("nan")

    annualised_mean = daily_returns.mean() * trading_days_per_year
    annualised_std = std * np.sqrt(trading_days_per_year)

    return float((annualised_mean - risk_free_rate_annual) / annualised_std)


def max_drawdown(cumulative_return_series: pd.Series) -> dict[str, object]:
    """Compute the largest peak-to-trough decline in a cumulative return series.

    Defined as an ABSOLUTE decline (`value - running_max`), not a
    percentage of the peak: `cumulative_return_series` is an additive
    dollar P&L series (see module docstring) that can be zero, positive,
    or negative, so a peak-relative percentage drawdown is not meaningful
    here the way it would be for a multiplicative NAV/equity index.

    Args:
        cumulative_return_series: Cumulative return series, indexed by date.

    Returns:
        Dict with keys "max_drawdown" (<=0, or NaN if the series is
        empty), "peak_date" (the date of the running peak that preceded
        the worst decline), "trough_date" (the date of the worst point).
        For a series with no decline at all (e.g. a flat, zero-trade
        period), max_drawdown is 0.0 and peak_date == trough_date.
    """
    series = cumulative_return_series.dropna()
    if series.empty:
        return {"max_drawdown": float("nan"), "peak_date": None, "trough_date": None}

    running_max = series.cummax()
    drawdown = series - running_max

    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_date])

    peak_value = running_max.loc[trough_date]
    peak_date = series.loc[:trough_date][series.loc[:trough_date] == peak_value].index[0]

    return {"max_drawdown": max_dd, "peak_date": peak_date, "trough_date": trough_date}


def trade_stats(positions_df: pd.DataFrame) -> dict[str, object]:
    """Compute discrete trade-level statistics for one pair/period.

    A "trade" is one contiguous run of non-zero `position`, closed on the
    row where `exit_reason` is set (per `signals.generate_positions`'s
    convention: the close row itself has `position == 0`, with
    `exit_reason` recorded on that same row). Trade P&L is summed over
    `net_return` from the day AFTER entry through the close row inclusive
    -- not the nonzero-position rows themselves -- because
    `backtest.compute_pair_daily_returns` attributes day t's P&L to the
    position held going INTO day t (yesterday's close), so the close row's
    own `net_return` still reflects the position that was open the day
    before it closed, and the entry row's `net_return` reflects the day
    before entry (i.e. zero contribution from this trade).

    `positions_df` must be a MERGED table (position + exit_reason from
    signals.py, joined with net_return from backtest.py for the same
    pair/period) -- e.g. what `summarise_pair_period` builds before
    calling this function -- not the raw output of either module alone.

    Args:
        positions_df: DataFrame with "position", "exit_reason", and
            "net_return" columns, either indexed by date or with a "date"
            column, for one pair and one period.

    Returns:
        Dict with keys "n_trades" (count of CLOSED trades; a position
        still open at the end of the period is not counted), "win_rate"
        (fraction of closed trades with positive summed P&L; NaN if
        n_trades == 0), "avg_holding_days" (mean trading days held per
        closed trade; NaN if n_trades == 0), "exit_reason_counts" (dict
        with keys "mean_reversion", "max_holding", "stop_loss", each a
        count, 0 if none occurred).
    """
    if "date" in positions_df.columns:
        df = positions_df.sort_values("date").reset_index(drop=True)
    else:
        df = positions_df.sort_index().reset_index(drop=True)

    positions = df["position"].to_numpy()
    exit_reasons = df["exit_reason"].to_numpy()
    net_returns = df["net_return"].to_numpy()

    trades: list[dict[str, object]] = []
    entry_idx: int | None = None

    for i in range(len(df)):
        if positions[i] != 0 and entry_idx is None:
            entry_idx = i

        if pd.notna(exit_reasons[i]):
            if entry_idx is not None:
                holding_days = i - entry_idx
                pnl = float(net_returns[entry_idx + 1 : i + 1].sum())
                trades.append({"holding_days": holding_days, "pnl": pnl, "exit_reason": exit_reasons[i]})
            entry_idx = None

    exit_reason_counts = {"mean_reversion": 0, "max_holding": 0, "stop_loss": 0}
    for trade in trades:
        exit_reason_counts[trade["exit_reason"]] = exit_reason_counts.get(trade["exit_reason"], 0) + 1

    n_trades = len(trades)
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_holding_days": float("nan"),
            "exit_reason_counts": exit_reason_counts,
        }

    win_rate = sum(1 for trade in trades if trade["pnl"] > 0) / n_trades
    avg_holding_days = sum(trade["holding_days"] for trade in trades) / n_trades

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_holding_days": avg_holding_days,
        "exit_reason_counts": exit_reason_counts,
    }


def summarise_pair_period(
    backtest_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    period_label: str,
) -> dict[str, object]:
    """Summarise one pair's performance over one period into a single report row.

    Merges `positions_df`'s "exit_reason" into `backtest_df` (on "date")
    before calling `trade_stats`, since `backtest_df` alone (from
    `backtest.backtest_pair`/`backtest_all_selected_pairs`) already has
    "position" and "net_return" but not "exit_reason".

    Args:
        backtest_df: One pair's rows for `period_label`, from
            `backtest.backtest_all_selected_pairs`'s output (must include
            "pair", "date", "position", "net_return",
            "cumulative_net_return"; "pair" must be single-valued).
        positions_df: The matching pair/period slice of
            `signals.build_signals_for_selected_pairs()["positions"]`
            (must include "date", "exit_reason").
        period_label: Period name, e.g. "trading_period_1", used verbatim
            in the returned row rather than inferred from the data.

    Returns:
        Dict with keys "pair", "period", "cumulative_net_return",
        "sharpe", "max_drawdown", "max_drawdown_peak_date",
        "max_drawdown_trough_date", "n_trades", "win_rate",
        "avg_holding_days", "exit_reason_counts".
    """
    pair = backtest_df["pair"].iloc[0] if "pair" in backtest_df.columns and not backtest_df.empty else None

    merged = backtest_df.merge(positions_df[["date", "exit_reason"]], on="date", how="left")

    cumulative_net_return = (
        float(backtest_df["cumulative_net_return"].iloc[-1]) if not backtest_df.empty else float("nan")
    )

    sharpe = sharpe_ratio(backtest_df["net_return"])
    drawdown_result = max_drawdown(backtest_df.set_index("date")["cumulative_net_return"])
    stats = trade_stats(merged)

    return {
        "pair": pair,
        "period": period_label,
        "cumulative_net_return": cumulative_net_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown_result["max_drawdown"],
        "max_drawdown_peak_date": drawdown_result["peak_date"],
        "max_drawdown_trough_date": drawdown_result["trough_date"],
        "n_trades": stats["n_trades"],
        "win_rate": stats["win_rate"],
        "avg_holding_days": stats["avg_holding_days"],
        "exit_reason_counts": stats["exit_reason_counts"],
    }


def summarise_all_pairs(
    backtest_results_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    periods: list[str] | None = None,
) -> pd.DataFrame:
    """Run `summarise_pair_period` for every pair x period combination present in the data.

    Args:
        backtest_results_df: Output of `backtest.backtest_all_selected_pairs`
            (columns "pair", "period", "date", "position", "net_return",
            "cumulative_net_return", ...).
        positions_df: Output of
            `signals.build_signals_for_selected_pairs()["positions"]`
            (columns "pair", "period", "date", "exit_reason", ...).
        periods: Periods to summarise. Defaults to every unique value of
            `backtest_results_df["period"]`.

    Returns:
        Tidy DataFrame, one row per (pair, period) combination present in
        `backtest_results_df`, with the columns from
        `summarise_pair_period`.
    """
    if periods is None:
        periods = sorted(backtest_results_df["period"].unique())

    pairs = sorted(backtest_results_df["pair"].unique())

    rows = []
    for pair in pairs:
        for period in periods:
            backtest_slice = backtest_results_df[
                (backtest_results_df["pair"] == pair) & (backtest_results_df["period"] == period)
            ]
            if backtest_slice.empty:
                continue

            positions_slice = positions_df[(positions_df["pair"] == pair) & (positions_df["period"] == period)]
            rows.append(summarise_pair_period(backtest_slice, positions_slice, period))

    return pd.DataFrame(rows)


def summarise_portfolio(average_return_series_by_period: dict[str, pd.Series]) -> pd.DataFrame:
    """Summarise the equal-weighted portfolio's performance for each trading period.

    Sharpe ratio and max drawdown are computed directly from the
    portfolio's OWN cumulative return series (recovering daily returns via
    `.diff()`), not averaged from the individual pairs' Sharpe ratios --
    Sharpe ratios don't combine linearly across positions (the portfolio's
    variance depends on the pairs' covariance, not just their individual
    variances), so averaging per-pair Sharpes would misstate the
    portfolio's actual risk-adjusted return.

    Args:
        average_return_series_by_period: Mapping from period name (e.g.
            "trading_period_1") to that period's equal-weighted cumulative
            net return series (e.g. one call to
            `backtest.compute_average_pair_return` per period), indexed
            by date.

    Returns:
        DataFrame with one row per period and columns "period",
        "cumulative_return", "sharpe", "max_drawdown",
        "max_drawdown_peak_date", "max_drawdown_trough_date".
    """
    rows = []
    for period, cumulative_series in average_return_series_by_period.items():
        cumulative_series = cumulative_series.sort_index()

        if cumulative_series.empty:
            daily_returns = cumulative_series
        else:
            daily_returns = cumulative_series.diff()
            daily_returns.iloc[0] = cumulative_series.iloc[0]

        sharpe = sharpe_ratio(daily_returns)
        drawdown_result = max_drawdown(cumulative_series)

        rows.append(
            {
                "period": period,
                "cumulative_return": float(cumulative_series.iloc[-1]) if not cumulative_series.empty else float("nan"),
                "sharpe": sharpe,
                "max_drawdown": drawdown_result["max_drawdown"],
                "max_drawdown_peak_date": drawdown_result["peak_date"],
                "max_drawdown_trough_date": drawdown_result["trough_date"],
            }
        )

    return pd.DataFrame(rows)
