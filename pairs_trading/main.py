"""End-to-end pipeline orchestration.

Runs, in order: data loading -> formation-period screening -> hedge ratio
estimation -> signal generation -> backtest over each trading period ->
metrics -> validation, then produces the final summary tables/figures.

Data loading, screening, hedge ratio estimation, signal generation,
backtesting, and metrics summarisation are all implemented
(`load_and_split_data`, `screen_and_select_pairs`, `build_hedge_ratios`,
`build_signals`, `run_backtest_phase`, `run_metrics_phase`,
`run_pipeline_through_metrics`). `run_pipeline` remains a stub until
validation.py is implemented.
"""

from __future__ import annotations

import logging

import pandas as pd

from pairs_trading import backtest, config, data, hedge_ratio, metrics, screening, signals

logger = logging.getLogger(__name__)


def load_and_split_data() -> dict[str, pd.DataFrame]:
    """Load the full candidate price panel and split into formation/trading windows.

    Returns:
        Dict with keys "full" (the entire loaded panel, unsliced),
        "formation", "trading_period_1", "trading_period_2" (each a
        DataFrame of prices sliced to the corresponding date range in
        config.py).
    """
    prices = data.load_all_candidate_prices()
    return {
        "full": prices,
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


def build_signals(
    price_panel: pd.DataFrame,
    hedge_ratios_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate position signals for the selected pairs across all three periods, saving CSVs.

    Thin wrapper around `signals.build_signals_for_selected_pairs` that
    also writes "positions" and "thresholds" to `config.RESULTS_DIR` as
    CSV, using the date ranges in config.py for all three periods.

    Args:
        price_panel: Full price panel, columns=tickers, indexed by date,
            covering formation and both trading periods (e.g.
            `load_and_split_data()["full"]`).
        hedge_ratios_df: Output of `build_hedge_ratios`.

    Returns:
        The dict from `signals.build_signals_for_selected_pairs`
        ("positions", "thresholds"), each also written to
        `config.RESULTS_DIR / "signals_{key}.csv"`.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    signals_dict = signals.build_signals_for_selected_pairs(
        price_panel,
        hedge_ratios_df,
        config.FORMATION_START,
        config.FORMATION_END,
        config.TRADING_PERIOD_1_START,
        config.TRADING_PERIOD_1_END,
        config.TRADING_PERIOD_2_START,
        config.TRADING_PERIOD_2_END,
    )

    for name, table in signals_dict.items():
        out_path = config.RESULTS_DIR / f"signals_{name}.csv"
        table.to_csv(out_path, index=False)
        logger.info("Wrote signals_%s (%d rows) to %s", name, len(table), out_path)

    return signals_dict


def run_backtest_phase(
    price_panel: pd.DataFrame,
    signals_dict: dict[str, pd.DataFrame],
    hedge_ratios_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run the full backtest for the selected pairs and save results/summaries to CSV.

    Runs `backtest.backtest_all_selected_pairs` once (covering formation
    and both trading periods), then `backtest.compute_average_pair_return`
    for each trading period to get the equal-weighted "portfolio" result
    the brief asks for. Only the two trading periods are written as their
    own CSVs: formation-period performance already informed threshold
    selection (via signals.py's grid search), so it is not a second
    out-of-sample result and is reported separately, not alongside the
    genuinely out-of-sample trading-period figures.

    Args:
        price_panel: Full price panel, columns=tickers, indexed by date.
        signals_dict: Output of `build_signals`.
        hedge_ratios_df: Output of `build_hedge_ratios`.

    Returns:
        Dict with keys:
            "all_periods": full `backtest_all_selected_pairs` output
                (formation and both trading periods).
            "trading_period_1", "trading_period_2": `all_periods` filtered
                to that period, written to
                `config.RESULTS_DIR / "backtest_trading{1,2}.csv"`.
            "average_returns": tidy DataFrame with columns "period",
                "date", "average_cumulative_net_return" for both trading
                periods, written to
                `config.RESULTS_DIR / "backtest_average_returns.csv"`.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = backtest.backtest_all_selected_pairs(price_panel, signals_dict, hedge_ratios_df)

    trading1 = all_results[all_results["period"] == "trading_period_1"]
    trading2 = all_results[all_results["period"] == "trading_period_2"]

    average_returns = pd.concat(
        [
            backtest.compute_average_pair_return(all_results, period_name)
            .rename("average_cumulative_net_return")
            .reset_index()
            .assign(period=period_name)
            for period_name in ("trading_period_1", "trading_period_2")
        ],
        ignore_index=True,
    )[["period", "date", "average_cumulative_net_return"]]

    trading1.to_csv(config.RESULTS_DIR / "backtest_trading1.csv", index=False)
    trading2.to_csv(config.RESULTS_DIR / "backtest_trading2.csv", index=False)
    average_returns.to_csv(config.RESULTS_DIR / "backtest_average_returns.csv", index=False)
    logger.info(
        "Wrote backtest_trading1 (%d rows), backtest_trading2 (%d rows), "
        "backtest_average_returns (%d rows) to %s",
        len(trading1),
        len(trading2),
        len(average_returns),
        config.RESULTS_DIR,
    )

    return {
        "all_periods": all_results,
        "trading_period_1": trading1,
        "trading_period_2": trading2,
        "average_returns": average_returns,
    }


def run_metrics_phase(
    backtest_results: dict[str, pd.DataFrame],
    signals_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Summarise per-pair and portfolio performance metrics and save to CSV.

    Restricted to the two trading periods (not formation), consistent with
    `run_backtest_phase`'s CSVs: formation-period performance already
    informed threshold selection, so it is not a second out-of-sample
    result to report alongside the genuine out-of-sample figures.

    Args:
        backtest_results: Output of `run_backtest_phase`.
        signals_dict: Output of `build_signals`.

    Returns:
        Dict with keys "by_pair" (output of `metrics.summarise_all_pairs`)
        and "portfolio" (output of `metrics.summarise_portfolio`), written
        to `config.RESULTS_DIR / "metrics_by_pair.csv"` and
        `config.RESULTS_DIR / "metrics_portfolio.csv"` respectively. The
        saved/printed "by_pair" table flattens "exit_reason_counts" into
        separate "exit_mean_reversion"/"exit_max_holding"/"exit_stop_loss"
        columns for CSV/terminal friendliness.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trading_periods = ["trading_period_1", "trading_period_2"]

    by_pair = metrics.summarise_all_pairs(
        backtest_results["all_periods"], signals_dict["positions"], periods=trading_periods
    )

    average_returns = backtest_results["average_returns"]
    average_series_by_period = {
        period_name: average_returns.loc[
            average_returns["period"] == period_name
        ].set_index("date")["average_cumulative_net_return"]
        for period_name in trading_periods
    }
    portfolio = metrics.summarise_portfolio(average_series_by_period)

    exit_counts_expanded = pd.json_normalize(by_pair["exit_reason_counts"]).add_prefix("exit_")
    by_pair_flat = pd.concat([by_pair.drop(columns="exit_reason_counts"), exit_counts_expanded], axis=1)

    by_pair_flat.to_csv(config.RESULTS_DIR / "metrics_by_pair.csv", index=False)
    portfolio.to_csv(config.RESULTS_DIR / "metrics_portfolio.csv", index=False)
    logger.info(
        "Wrote metrics_by_pair (%d rows), metrics_portfolio (%d rows) to %s",
        len(by_pair_flat),
        len(portfolio),
        config.RESULTS_DIR,
    )

    print("\n=== Metrics: per-pair summary (trading periods only) ===")
    print(
        by_pair_flat[
            [
                "pair",
                "period",
                "cumulative_net_return",
                "sharpe",
                "max_drawdown",
                "n_trades",
                "win_rate",
                "avg_holding_days",
            ]
        ].to_string(index=False)
    )

    print("\n=== Metrics: equal-weighted portfolio summary ===")
    print(portfolio[["period", "cumulative_return", "sharpe", "max_drawdown"]].to_string(index=False))

    return {"by_pair": by_pair, "portfolio": portfolio}


def run_screening_phase() -> dict[str, pd.DataFrame]:
    """Load cached candidate price data and run screening plus formation-period hedge ratio estimation.

    Standalone convenience entry point for just the screening and
    hedge-ratio stages. `run_pipeline_through_metrics` runs this same
    work (without a second data load) plus signal generation,
    backtesting, and metrics summarisation.

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


def run_pipeline_through_metrics() -> dict[str, pd.DataFrame]:
    """Load data once and run every implemented stage: screening, hedge ratios, signals, backtest, metrics.

    This is the current top-level entry point (called from `__main__`).
    Loads the price panel a single time (unlike calling
    `run_screening_phase` followed by separate signals/backtest/metrics
    steps, which would reload it) and threads it through
    `screen_and_select_pairs` -> `build_hedge_ratios` -> `build_signals`
    -> `run_backtest_phase` -> `run_metrics_phase`. `run_pipeline` will
    eventually extend this with validation.py once implemented.

    Returns:
        Dict merging: `screen_and_select_pairs`'s keys, "hedge_ratios"
        (from `build_hedge_ratios`), "signals_positions"/
        "signals_thresholds" (from `build_signals`),
        "backtest_all_periods"/"backtest_trading_period_1"/
        "backtest_trading_period_2"/"backtest_average_returns" (from
        `run_backtest_phase`), and "metrics_by_pair"/"metrics_portfolio"
        (from `run_metrics_phase`).
    """
    data_splits = load_and_split_data()
    full_prices = data_splits["full"]

    results = screen_and_select_pairs(data_splits["formation"])
    selected = results["selected_pairs"]

    hedge_ratios = build_hedge_ratios(data_splits["formation"], selected)
    results["hedge_ratios"] = hedge_ratios

    signals_dict = build_signals(full_prices, hedge_ratios)
    results["signals_positions"] = signals_dict["positions"]
    results["signals_thresholds"] = signals_dict["thresholds"]

    backtest_results = run_backtest_phase(full_prices, signals_dict, hedge_ratios)
    results["backtest_all_periods"] = backtest_results["all_periods"]
    results["backtest_trading_period_1"] = backtest_results["trading_period_1"]
    results["backtest_trading_period_2"] = backtest_results["trading_period_2"]
    results["backtest_average_returns"] = backtest_results["average_returns"]

    metrics_results = run_metrics_phase(backtest_results, signals_dict)
    results["metrics_by_pair"] = metrics_results["by_pair"]
    results["metrics_portfolio"] = metrics_results["portfolio"]

    return results


def run_pipeline() -> dict[str, pd.DataFrame]:
    """Run the full pipeline end-to-end and produce final tables/figures.

    Orchestrates `run_pipeline_through_metrics` (data loading, screening,
    hedge ratios, signals, backtest, metrics) plus validation.py, once
    implemented.

    Returns:
        Dict of result tables keyed by name (e.g. "screened_pairs",
        "backtest_summary", "validation_summary"), suitable for writing to
        config.RESULTS_DIR.
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline_through_metrics()
