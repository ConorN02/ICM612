"""End-to-end pipeline orchestration.

Runs, in order: data loading -> formation-period screening -> hedge ratio
estimation -> signal generation -> backtest over each trading period ->
metrics -> validation, then produces the final summary tables/figures.

Only the data-loading, screening, and formation-period hedge ratio stages
are implemented so far (`load_and_split_data`, `screen_and_select_pairs`,
`build_hedge_ratios`, `run_screening_phase`). `run_pipeline` remains a stub
until signals.py, backtest.py, metrics.py, and validation.py are
implemented in later phases.
"""

from __future__ import annotations

import logging

import pandas as pd

from pairs_trading import config, data, hedge_ratio, screening

logger = logging.getLogger(__name__)


def load_and_split_data() -> dict[str, pd.DataFrame]:
    """Load the full candidate price panel and split into formation/trading windows.

    Returns:
        Dict with keys "formation", "trading_period_1", "trading_period_2",
        each a DataFrame of prices sliced to the corresponding date range
        in config.py.
    """
    prices = data.load_all_candidate_prices()
    return {
        "formation": prices.loc[config.FORMATION_START : config.FORMATION_END],
        "trading_period_1": prices.loc[config.TRADING_PERIOD_1_START : config.TRADING_PERIOD_1_END],
        "trading_period_2": prices.loc[config.TRADING_PERIOD_2_START : config.TRADING_PERIOD_2_END],
    }


def screen_and_select_pairs(formation_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run formation-period screening, COVID sensitivity, and gated selection, saving CSVs.

    Orchestrates `screening.screen_covid_sensitivity` (which itself runs
    `screening.screen_all_candidate_pairs` on both the unmodified formation
    period and the period with the COVID-19 window excised) and
    `screening.rank_and_select_pairs`, then writes every resulting table to
    `config.RESULTS_DIR` as CSV so they can be dropped directly into the
    report.

    Final selection runs on the ex-COVID table because
    `config.USE_COVID_EXCLUDED_SCREENING` is True: none of the 5
    candidates are cointegrated at 5% over the full 2017-2021 formation
    period, but HD/LOW and KO/PEP clear 5% once the COVID crash window is
    excluded, and DAL/UAL is carried as a named, justified exception at the
    10% level (see config.py). This is a deliberate, documented modelling
    choice, not a silent default — flip `config.USE_COVID_EXCLUDED_SCREENING`
    to False to select from the full-period table instead.

    Args:
        formation_prices: Price panel restricted to the formation period
            (`config.FORMATION_START` to `config.FORMATION_END`).

    Returns:
        Dict with keys "full_period", "ex_covid", "covid_comparison",
        "selected_pairs", "selection_audit" — each also written to
        `config.RESULTS_DIR / f"screening_{key}.csv"`.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    covid_sensitivity = screening.screen_covid_sensitivity(
        formation_prices, config.FORMATION_START, config.FORMATION_END
    )
    full_results = covid_sensitivity["full_period"]
    ex_covid_results = covid_sensitivity["ex_covid"]

    source_label = "ex_covid" if config.USE_COVID_EXCLUDED_SCREENING else "full_period"
    screening_df = ex_covid_results if config.USE_COVID_EXCLUDED_SCREENING else full_results
    selection = screening.rank_and_select_pairs(screening_df, source_label=source_label)

    results = {
        "full_period": full_results,
        "ex_covid": ex_covid_results,
        "covid_comparison": covid_sensitivity["comparison"],
        "selected_pairs": selection["selected_pairs"],
        "selection_audit": selection["selection_audit"],
    }

    for name, table in results.items():
        out_path = config.RESULTS_DIR / f"screening_{name}.csv"
        table.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows) to %s", name, len(table), out_path)

    return results


def build_hedge_ratios(
    formation_prices: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit formation-period hedge ratios for the selected pairs and save to CSV.

    Thin wrapper around `hedge_ratio.build_hedge_ratios_for_selected_pairs`
    that also writes the result to `config.RESULTS_DIR / "hedge_ratios.csv"`,
    so signals.py/backtest.py (and the report) can read a single,
    already-computed table rather than each re-fitting alpha/beta itself.

    Args:
        formation_prices: Price panel restricted to the formation period
            (`config.FORMATION_START` to `config.FORMATION_END`).
        selected_pairs_df: Output of
            `screen_and_select_pairs(...)["selected_pairs"]`.

    Returns:
        The DataFrame from `hedge_ratio.build_hedge_ratios_for_selected_pairs`,
        also written to `config.RESULTS_DIR / "hedge_ratios.csv"`.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hedge_ratios = hedge_ratio.build_hedge_ratios_for_selected_pairs(
        formation_prices, selected_pairs_df, config.FORMATION_START, config.FORMATION_END
    )

    out_path = config.RESULTS_DIR / "hedge_ratios.csv"
    hedge_ratios.to_csv(out_path, index=False)
    logger.info("Wrote hedge_ratios (%d rows) to %s", len(hedge_ratios), out_path)

    return hedge_ratios


def run_screening_phase() -> dict[str, pd.DataFrame]:
    """Load cached candidate price data and run screening plus formation-period hedge ratio estimation.

    This is the current entry point for the screening and hedge-ratio
    stages of the pipeline. `run_pipeline` will eventually call this as
    its first stages once signals.py, backtest.py, metrics.py, and
    validation.py are implemented.

    Returns:
        The dict returned by `screen_and_select_pairs`, with an added
        "hedge_ratios" key holding the output of `build_hedge_ratios`.
    """
    data_splits = load_and_split_data()
    results = screen_and_select_pairs(data_splits["formation"])

    selected = results["selected_pairs"]
    source_desc = "ex-COVID" if config.USE_COVID_EXCLUDED_SCREENING else "full-period"
    print(f"\n=== Selected {len(selected)} pair(s) (source: {source_desc} screening) ===")
    for _, row in selected.iterrows():
        print(f"  {row['pair']}: {row['reason']}")

    print("\n=== COVID window sensitivity (full period vs. COVID excluded) ===")
    print(
        results["covid_comparison"][
            ["pair", "correlation_change", "ssd_normalised_spread_change", "eg_p_value_change"]
        ].to_string(index=False)
    )

    results["hedge_ratios"] = build_hedge_ratios(data_splits["formation"], selected)

    print("\n=== Formation-period hedge ratios ===")
    print(
        results["hedge_ratios"][
            [
                "pair",
                "dependent_ticker",
                "independent_ticker",
                "alpha",
                "beta",
                "r_squared",
                "half_life_days",
            ]
        ].to_string(index=False)
    )

    return results


def run_pipeline() -> dict[str, pd.DataFrame]:
    """Run the full pipeline end-to-end and produce final tables/figures.

    Orchestrates: data.load_all_candidate_prices -> screening + hedge_ratio
    (via `run_screening_phase`) -> signals.build_signals_for_selected_pairs ->
    backtest.run_backtest for each trading period -> metrics -> validation.

    Returns:
        Dict of result tables keyed by name (e.g. "screened_pairs",
        "backtest_summary", "validation_summary"), suitable for writing to
        config.RESULTS_DIR.
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_screening_phase()
