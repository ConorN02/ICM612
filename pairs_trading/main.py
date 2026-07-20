"""End-to-end pipeline orchestration.

Runs, in order: data loading -> formation-period screening -> hedge ratio
estimation -> signal generation -> backtest over each trading period ->
metrics -> validation, then produces the final summary tables/figures.
"""

from __future__ import annotations

import pandas as pd

from pairs_trading import config


def load_and_split_data() -> dict[str, pd.DataFrame]:
    """Load the full candidate price panel and split into formation/trading windows.

    Returns:
        Dict with keys "formation", "trading_period_1", "trading_period_2",
        each a DataFrame of prices sliced to the corresponding date range
        in config.py.
    """
    raise NotImplementedError


def screen_and_select_pairs(formation_prices: pd.DataFrame) -> pd.DataFrame:
    """Run formation-period screening and select the final tradeable pairs.

    Args:
        formation_prices: Price panel restricted to the formation period.

    Returns:
        DataFrame of pairs that passed the screen, as produced by
        `screening.screen_pairs`.
    """
    raise NotImplementedError


def run_pipeline() -> dict[str, pd.DataFrame]:
    """Run the full pipeline end-to-end and produce final tables/figures.

    Orchestrates: data.load_all_candidate_prices -> screening.screen_pairs ->
    hedge_ratio (static + Kalman) -> signals.generate_signals ->
    backtest.run_backtest for each trading period -> metrics -> validation.

    Returns:
        Dict of result tables keyed by name (e.g. "screened_pairs",
        "backtest_summary", "validation_summary"), suitable for writing to
        config.RESULTS_DIR.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run_pipeline()
