"""Z-score signal generation for pairs trading.

Converts each pair's spread into a z-score against a FIXED formation-period
mean/std (via `hedge_ratio.compute_spread` + `hedge_ratio.spread_zscore`),
then translates that z-score into discrete long/short/flat positions with
configurable entry/exit thresholds, an optional max holding period, and a
stop-loss risk control.

Threshold selection (`threshold_sensitivity_grid`) is deliberately restricted
to the formation period: this is the pipeline's safeguard against picking
entry/exit thresholds with hindsight from trading-period performance. See
that function's docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading import config, hedge_ratio


def generate_zscore_series(
    price_panel: pd.DataFrame,
    hedge_ratio_row: pd.Series,
    start: str,
    end: str,
) -> pd.Series:
    """Compute a pair's z-scored spread over [start, end] using its formation-period baseline.

    Thin composition of `hedge_ratio.compute_spread` (fixed alpha/beta) and
    `hedge_ratio.spread_zscore` (fixed formation mean/std): both come from
    `hedge_ratio_row`, never re-estimated from [start, end] itself. This is
    what makes the function correct whether [start, end] is the formation
    period (reproducing the baseline the thresholds were chosen against)
    or either out-of-sample trading period (scoring genuinely new data
    against a baseline fixed before that data existed).

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        hedge_ratio_row: One row (e.g. via `.iloc[i]`) of
            `hedge_ratio.build_hedge_ratios_for_selected_pairs()`'s output,
            with "dependent_ticker", "independent_ticker", "alpha", "beta",
            "formation_spread_mean", "formation_spread_std".
        start: Window start date, "YYYY-MM-DD".
        end: Window end date, "YYYY-MM-DD".

    Returns:
        Z-score series indexed by date, named "zscore".
    """
    spread = hedge_ratio.compute_spread(
        price_panel,
        hedge_ratio_row["dependent_ticker"],
        hedge_ratio_row["independent_ticker"],
        hedge_ratio_row["alpha"],
        hedge_ratio_row["beta"],
        start,
        end,
    )
    return hedge_ratio.spread_zscore(
        spread,
        hedge_ratio_row["formation_spread_mean"],
        hedge_ratio_row["formation_spread_std"],
    )


def generate_positions(
    zscore_series: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
    max_holding_days: int | None = None,
    stop_loss_threshold: float | None = None,
) -> pd.DataFrame:
    """Translate a z-score series into discrete long/short/flat positions.

    Position convention: +1 = long the spread (long the dependent leg,
    short beta units of the independent leg), -1 = short the spread, 0 =
    flat.

    Rule, evaluated sequentially one day at a time:
        - While flat: enter LONG when z <= -entry_threshold, enter SHORT
          when z >= +entry_threshold.
        - While in a position: close to FLAT the first day the position's
          exit condition becomes true, in this priority order:
            1. "stop_loss" -- |z| >= stop_loss_threshold (if set).
            2. "mean_reversion" -- z has crossed back through the exit
               band on the relevant side (z >= -exit_threshold for a long
               position, z <= exit_threshold for a short position). Since
               this is evaluated once per day and the position was still
               open on the prior day (meaning the condition was false
               then), the first day it becomes true IS the crossing day --
               not a repeated "touch" of a level already passed.
            3. "max_holding" -- the position has been open for
               max_holding_days trading days (if set), regardless of where
               z is, so a single non-reverting trade cannot run forever.

    Re-entry policy differs by exit type:
        - After a "max_holding" close: entry is re-evaluated fresh the very
          next day, with no memory of the prior trade -- if z is still
          beyond entry_threshold, a new position opens immediately.
          max_holding is a time-based constraint, not a signal that the
          spread relationship has failed, so there is no reason to wait.
        - After a "stop_loss" close: the pair enters a "cooling_down"
          state (see "state" in Returns) and is BLOCKED from opening a new
          position until z first returns inside the entry band
          (|z| < entry_threshold) at least once. A stop-loss firing is a
          risk-control signal that the model of the spread may currently
          be broken (e.g. a structural break in the pair's relationship);
          immediately re-opening into the same extreme reading would
          defeat the purpose of having a stop-loss at all, and -- if the
          extreme persists -- would otherwise produce a rapid, misleading
          sequence of one-or-two-day stop-outs that looks like a bug
          rather than the genuine out-of-sample breakdown it represents.
          Once z re-enters the band, cooldown clears and ordinary entry
          evaluation resumes (note entry itself requires |z| >=
          entry_threshold, so clearing cooldown and opening a new position
          can never happen on the same day).

    No lookahead: the position (and any exit/cooldown decision) on day t is
    decided using only zscore_series.iloc[t] and state carried forward from
    days before t (current position, days held, cooling_down) -- never a
    later value.

    Args:
        zscore_series: Z-score series (e.g. from `generate_zscore_series`),
            indexed by date.
        entry_threshold: Absolute z-score level to open a position, and
            (after a stop-loss) the band |z| must first fall back inside
            before a new position can open again.
        exit_threshold: Absolute z-score level defining the mean-reversion
            exit band. 0.0 means "exit once the spread reverts back through
            its formation-period mean".
        max_holding_days: If set, force-close any position open this many
            trading days or longer.
        stop_loss_threshold: If set, force-close a position the day |z|
            reaches or exceeds this level, tracked as "stop_loss" (and
            triggering the re-entry cooldown described above) so how often
            the stop actually fires can be reported separately from
            ordinary mean-reversion exits.

    Returns:
        DataFrame indexed like `zscore_series` with columns "zscore",
        "position" (-1/0/+1), "state" ("long"/"short"/"flat"/"cooling_down"
        -- "cooling_down" is a flat, no-exposure state like "flat" but
        distinguishes "blocked from re-entry after a stop-loss" from
        ordinary flat, since `position` alone can't), and "exit_reason"
        (one of "mean_reversion", "max_holding", "stop_loss" on the day a
        position closes; None on every other day).
    """
    zscore_series = zscore_series.sort_index()
    dates = zscore_series.index
    values = zscore_series.to_numpy()

    positions = np.zeros(len(dates), dtype=int)
    exit_reasons: list[str | None] = [None] * len(dates)
    states: list[str] = [""] * len(dates)

    current_position = 0
    days_held = 0
    cooling_down = False

    for i in range(len(dates)):
        z = values[i]

        if current_position == 0:
            if cooling_down:
                if abs(z) < entry_threshold:
                    cooling_down = False
                positions[i] = 0
                states[i] = "cooling_down" if cooling_down else "flat"
                continue

            if z >= entry_threshold:
                current_position = -1
                days_held = 1
            elif z <= -entry_threshold:
                current_position = 1
                days_held = 1
            positions[i] = current_position
            states[i] = "short" if current_position == -1 else "long" if current_position == 1 else "flat"
            continue

        days_held += 1

        crossed_back_to_mean = (current_position == 1 and z >= -exit_threshold) or (
            current_position == -1 and z <= exit_threshold
        )
        hit_stop_loss = stop_loss_threshold is not None and abs(z) >= stop_loss_threshold
        hit_max_holding = max_holding_days is not None and days_held >= max_holding_days

        if hit_stop_loss:
            exit_reason: str | None = "stop_loss"
        elif crossed_back_to_mean:
            exit_reason = "mean_reversion"
        elif hit_max_holding:
            exit_reason = "max_holding"
        else:
            exit_reason = None

        if exit_reason is not None:
            exit_reasons[i] = exit_reason
            positions[i] = 0
            current_position = 0
            days_held = 0
            if exit_reason == "stop_loss":
                cooling_down = True
                states[i] = "cooling_down"
            else:
                states[i] = "flat"
        else:
            positions[i] = current_position
            states[i] = "short" if current_position == -1 else "long"

    return pd.DataFrame(
        {"zscore": zscore_series, "position": positions, "state": states, "exit_reason": exit_reasons},
        index=dates,
    )


def _formation_sharpe_for_positions(zscore_series: pd.Series, positions_df: pd.DataFrame) -> float:
    """Compute a lightweight, cost-free formation-period Sharpe ratio for one position series.

    Uses `position_{t-1}` applied to the z-score change realised over day
    t as a scale-free proxy for the pair's daily spread P&L: since every
    day's actual spread P&L was divided by the same positive
    `formation_spread_std` to build the z-score, and the Sharpe ratio is
    scale invariant (mean/std is unchanged by multiplying every return by
    the same positive constant), ranking threshold combinations by this
    proxy gives the same ordering as ranking by raw spread P&L would,
    without needing to reconstruct dollar P&L here. This is deliberately
    not the real backtest: it ignores transaction costs and $-sizing (both
    handled properly in backtest.py) and exists purely to rank threshold
    combinations against each other in `threshold_sensitivity_grid`.

    Args:
        zscore_series: Z-score series the positions were generated from.
        positions_df: Output of `generate_positions` on `zscore_series`.

    Returns:
        Annualised Sharpe ratio (using `config.TRADING_DAYS_PER_YEAR`), or
        0.0 if the position was flat throughout or the daily P&L proxy has
        zero variance.
    """
    zscore_diff = zscore_series.diff()
    daily_pnl_proxy = (positions_df["position"].shift(1).fillna(0) * zscore_diff).dropna()

    if daily_pnl_proxy.empty or daily_pnl_proxy.std(ddof=0) == 0:
        return 0.0

    return float(daily_pnl_proxy.mean() / daily_pnl_proxy.std(ddof=0) * np.sqrt(config.TRADING_DAYS_PER_YEAR))


def threshold_sensitivity_grid(
    price_panel: pd.DataFrame,
    hedge_ratio_row: pd.Series,
    formation_start: str,
    formation_end: str,
    entry_thresholds: list[float] = config.ENTRY_THRESHOLD_GRID,
    exit_thresholds: list[float] = config.EXIT_THRESHOLD_GRID,
) -> dict[str, pd.DataFrame]:
    """Grid-search entry/exit thresholds on the FORMATION PERIOD ONLY, and rank them.

    This is the pipeline's safeguard against parameter overfitting to
    out-of-sample data: every (entry_threshold, exit_threshold) combination
    is evaluated exclusively on `[formation_start, formation_end]`. Neither
    trading period is touched here, and this function has no way to see
    them. Locking in threshold choices from formation-period performance
    alone, before any trading-period result is observed, is what makes the
    later trading-period backtest a genuine out-of-sample test rather than
    a threshold chosen with hindsight — report it that way: the
    methodological control against look-ahead / parameter-selection bias,
    not merely a convenient default.

    For each combination, ranks by a lightweight, cost-free formation
    Sharpe ratio (`_formation_sharpe_for_positions`) — enough to choose
    between thresholds; the real, cost-aware backtest lives in backtest.py
    and is only ever run with the thresholds locked in here.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        hedge_ratio_row: One row of
            `hedge_ratio.build_hedge_ratios_for_selected_pairs()`'s output.
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".
        entry_thresholds: Entry threshold values to grid over. Defaults to
            `config.ENTRY_THRESHOLD_GRID`.
        exit_thresholds: Exit threshold values to grid over. Defaults to
            `config.EXIT_THRESHOLD_GRID`.

    Returns:
        Dict with keys:
            "grid": tidy DataFrame with columns "pair", "entry_threshold",
                "exit_threshold", "formation_sharpe" — one row per
                combination, ready for a heatmap plot.
            "best": single-row DataFrame with the combination achieving
                the highest "formation_sharpe" for this pair.
    """
    zscore_series = generate_zscore_series(price_panel, hedge_ratio_row, formation_start, formation_end)

    rows = []
    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            positions_df = generate_positions(
                zscore_series,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                max_holding_days=config.MAX_HOLDING_DAYS,
                stop_loss_threshold=config.STOP_LOSS_THRESHOLD,
            )
            sharpe = _formation_sharpe_for_positions(zscore_series, positions_df)
            rows.append(
                {
                    "pair": hedge_ratio_row["pair"],
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    "formation_sharpe": sharpe,
                }
            )

    grid = pd.DataFrame(rows)
    best = grid.loc[[grid["formation_sharpe"].idxmax()]].reset_index(drop=True)

    return {"grid": grid, "best": best}


def build_signals_for_selected_pairs(
    price_panel: pd.DataFrame,
    hedge_ratios_df: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    trading1_start: str,
    trading1_end: str,
    trading2_start: str,
    trading2_end: str,
    use_threshold_grid: bool = config.USE_THRESHOLD_GRID,
) -> dict[str, pd.DataFrame]:
    """Build position series for every selected pair across formation and both trading periods.

    For each pair in `hedge_ratios_df`: optionally runs
    `threshold_sensitivity_grid` on the formation period to pick that
    pair's entry/exit thresholds (falling back to
    `config.DEFAULT_ENTRY_THRESHOLD`/`config.DEFAULT_EXIT_THRESHOLD` for
    every pair when `use_threshold_grid` is False), then calls
    `generate_positions` on the formation period and both trading periods
    separately with those locked-in thresholds. The same formation-period
    hedge ratio (alpha/beta) and formation mean/std from `hedge_ratios_df`
    are reused for every period for a given pair — nothing is
    re-estimated from either trading period.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed
            by date, covering formation and both trading periods.
        hedge_ratios_df: Output of
            `hedge_ratio.build_hedge_ratios_for_selected_pairs`.
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".
        trading1_start: Trading period 1 start date, "YYYY-MM-DD".
        trading1_end: Trading period 1 end date, "YYYY-MM-DD".
        trading2_start: Trading period 2 start date, "YYYY-MM-DD".
        trading2_end: Trading period 2 end date, "YYYY-MM-DD".
        use_threshold_grid: If True, select thresholds per pair via
            `threshold_sensitivity_grid` on the formation period. If
            False, use `config.DEFAULT_ENTRY_THRESHOLD`/
            `config.DEFAULT_EXIT_THRESHOLD` for every pair. Defaults to
            `config.USE_THRESHOLD_GRID`.

    Returns:
        Dict with keys:
            "positions": tidy long-format DataFrame with columns "pair",
                "period" ("formation"/"trading_period_1"/"trading_period_2"),
                "date", "zscore", "position", "exit_reason" — ready for
                backtest.py to consume.
            "thresholds": one row per pair with columns "pair",
                "entry_threshold", "exit_threshold", "selection_method"
                ("threshold_grid" or "config_default"), and
                "formation_sharpe" of the chosen combination (NaN when
                `selection_method` is "config_default", since no grid was
                run).
    """
    periods = {
        "formation": (formation_start, formation_end),
        "trading_period_1": (trading1_start, trading1_end),
        "trading_period_2": (trading2_start, trading2_end),
    }

    position_tables = []
    threshold_rows = []

    for _, hedge_row in hedge_ratios_df.iterrows():
        if use_threshold_grid:
            grid_result = threshold_sensitivity_grid(price_panel, hedge_row, formation_start, formation_end)
            best = grid_result["best"].iloc[0]
            entry_threshold = float(best["entry_threshold"])
            exit_threshold = float(best["exit_threshold"])
            threshold_rows.append(
                {
                    "pair": hedge_row["pair"],
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    "selection_method": "threshold_grid",
                    "formation_sharpe": float(best["formation_sharpe"]),
                }
            )
        else:
            entry_threshold = config.DEFAULT_ENTRY_THRESHOLD
            exit_threshold = config.DEFAULT_EXIT_THRESHOLD
            threshold_rows.append(
                {
                    "pair": hedge_row["pair"],
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    "selection_method": "config_default",
                    "formation_sharpe": float("nan"),
                }
            )

        for period_name, (period_start, period_end) in periods.items():
            zscore_series = generate_zscore_series(price_panel, hedge_row, period_start, period_end)
            positions_df = generate_positions(
                zscore_series,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                max_holding_days=config.MAX_HOLDING_DAYS,
                stop_loss_threshold=config.STOP_LOSS_THRESHOLD,
            )
            positions_df = positions_df.reset_index().rename(columns={"index": "date"})
            positions_df.insert(0, "pair", hedge_row["pair"])
            positions_df.insert(1, "period", period_name)
            position_tables.append(positions_df)

    all_positions = pd.concat(position_tables, ignore_index=True)
    thresholds_df = pd.DataFrame(threshold_rows)

    return {"positions": all_positions, "thresholds": thresholds_df}
