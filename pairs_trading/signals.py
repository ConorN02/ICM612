"""Z-score signal generation for pairs trading.

Converts a spread series into a rolling z-score and translates that into
long/short/flat positions using configurable entry/exit thresholds, with
an optional volatility-adjusted threshold variant.
"""

from __future__ import annotations

import pandas as pd


def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Compute a rolling z-score of the spread.

    z_t = (spread_t - rolling_mean_t) / rolling_std_t, where the rolling
    mean/std are computed over the trailing `lookback` observations up to
    and including t (no lookahead).

    Args:
        spread: Spread series (e.g. from hedge_ratio.py), indexed by date.
        lookback: Rolling window length in trading days.

    Returns:
        Z-score series, same index as `spread`, with NaN for the initial
        `lookback - 1` observations.
    """
    raise NotImplementedError


def compute_volatility_adjusted_zscore(
    spread: pd.Series,
    lookback: int,
    volatility_lookback: int,
) -> pd.Series:
    """Compute a z-score whose effective threshold adapts to recent spread volatility.

    Args:
        spread: Spread series, indexed by date.
        lookback: Rolling window length (trading days) for the spread mean/std.
        volatility_lookback: Rolling window length (trading days) used to
            estimate recent spread volatility for the adjustment.

    Returns:
        Volatility-adjusted z-score series, same index as `spread`.
    """
    raise NotImplementedError


def generate_signals(
    zscore: pd.Series,
    entry_zscore: float,
    exit_zscore: float,
    stop_loss_zscore: float | None = None,
) -> pd.DataFrame:
    """Translate a z-score series into pair positions.

    Position convention: +1 = long the spread (long y, short beta*x),
    -1 = short the spread, 0 = flat. Enters long when z <= -entry_zscore,
    enters short when z >= entry_zscore, exits to flat when |z| <= exit_zscore,
    and (if `stop_loss_zscore` is provided) force-exits to flat when
    |z| >= stop_loss_zscore. Only uses zscore_t to decide position_t (no
    lookahead).

    Args:
        zscore: Z-score series, indexed by date.
        entry_zscore: Absolute z-score threshold to open a position.
        exit_zscore: Absolute z-score threshold to close a position.
        stop_loss_zscore: Optional absolute z-score threshold to force-close
            a position regardless of the entry/exit logic.

    Returns:
        DataFrame indexed like `zscore` with columns ["zscore", "position"].
    """
    raise NotImplementedError
