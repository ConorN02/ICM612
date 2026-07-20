"""Vectorised backtest engine for a single pair.

Simulates $1-per-pair sizing (Gatev et al. 2006 convention): at entry, the
long and short legs are each sized to $1 of notional (dollar-neutral),
scaled by the prevailing hedge ratio. Applies a transaction cost model on
each position change and produces a daily P&L / equity curve using only
information available at or before time t (positions decided on zscore_t
are applied to returns realised over (t, t+1], i.e. execution happens on
the next bar to avoid lookahead bias).
"""

from __future__ import annotations

import pandas as pd


def apply_transaction_costs(
    positions: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    """Compute per-period transaction cost drag from position changes.

    Charges `transaction_cost_bps` (in basis points) per unit of position
    change (i.e. cost is proportional to |position_t - position_{t-1}|),
    applied once per leg-equivalent turnover.

    Args:
        positions: Position series (+1/0/-1), indexed by date.
        transaction_cost_bps: Transaction cost in basis points per unit of turnover.

    Returns:
        Series of per-period transaction cost drag (as a fraction of
        capital), same index as `positions`.
    """
    raise NotImplementedError


def run_backtest(
    price_y: pd.Series,
    price_x: pd.Series,
    hedge_ratio: pd.Series,
    positions: pd.Series,
    capital_per_pair: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Run a vectorised backtest for a single pair.

    Positions decided using information available at time t are executed
    at t+1 (no lookahead). P&L on the long/short legs is computed from
    period returns scaled by `hedge_ratio`, sized to `capital_per_pair`
    dollars of exposure, net of transaction costs from `apply_transaction_costs`.

    Args:
        price_y: Price series for the dependent leg.
        price_x: Price series for the independent leg.
        hedge_ratio: Hedge ratio series (static or time-varying), aligned to
            price_y/price_x.
        positions: Position series (+1/0/-1) from signals.generate_signals,
            aligned to price_y/price_x.
        capital_per_pair: Dollar notional allocated to this pair.
        transaction_cost_bps: Transaction cost in basis points per unit of turnover.

    Returns:
        DataFrame indexed like the inputs with columns
        ["position", "gross_pnl", "transaction_cost", "net_pnl", "equity"].
    """
    raise NotImplementedError


def run_backtest_multi_pair(
    pair_backtests: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate individual pair backtests into a portfolio-level equity curve.

    Args:
        pair_backtests: Mapping from (ticker_a, ticker_b) to the DataFrame
            returned by `run_backtest` for that pair.

    Returns:
        DataFrame indexed by date with columns ["net_pnl", "equity"]
        aggregated (summed) across all pairs.
    """
    raise NotImplementedError
