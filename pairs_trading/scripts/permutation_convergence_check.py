"""One-off robustness check: how much does random_pair_benchmark's percentile/
p-value estimate move as n_simulations scales past the pipeline's reported
default of 10,000?

Standalone script, separate from the main pipeline (main.py) -- running it
does NOT change config.N_RANDOM_PAIRS_BENCHMARK (which stays at 10,000 as the
reported default) and does NOT overwrite any of the main pipeline's result
CSVs. It reuses the existing screening/hedge_ratio/signals/backtest/validation
functions as-is; no pipeline logic is reimplemented here.

Loads cached price data and re-derives the actual selected-pairs portfolio
returns once, then calls `validation.random_pair_benchmark` directly at
n_simulations = 50,000 and 100,000 for each trading period, timing each call.
The already-known n=10,000 result (from the most recent full pipeline run) is
included in the final comparison table for reference.

Usage:
    python3 pairs_trading/scripts/permutation_convergence_check.py

Expect this to run for well over an hour: at the ~43-44 simulations/second
measured for n=10,000 (see config.py's N_RANDOM_PAIRS_BENCHMARK comment), the
50,000-simulation pass alone takes roughly 35-40 minutes across both trading
periods, and the 100,000-simulation pass roughly 70-80 minutes -- run it in a
terminal you can leave alone, not inline in a short session.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Allow running this script directly (`python3 pairs_trading/scripts/...py`)
# regardless of the current working directory, by putting the repo root
# (parent of the pairs_trading package) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pairs_trading import backtest, config, data, hedge_ratio, screening, signals, validation

# Sample sizes to test, beyond the pipeline's reported default of 10,000.
SAMPLE_SIZES: list[int] = [50_000, 100_000]

# Already-known result from the most recent full pipeline run at
# config.N_RANDOM_PAIRS_BENCHMARK = 10,000 (see main.py's run_validation_phase
# output / results/validation_permutation.csv). Included here for reference
# so the convergence table shows all three sample sizes side by side.
# wall_clock_seconds is left as NaN for this baseline: the prior run's timing
# covered the entire pipeline (screening through validation, both periods,
# plus the bootstrap step) in one combined figure, not this permutation test
# in isolation, so there is no comparable per-call timing to report.
BASELINE_N_SIMULATIONS = 10_000
BASELINE_RESULTS: dict[str, dict[str, float]] = {
    "trading_period_1": {"percentile": 53.98, "p_value": 0.4602},
    "trading_period_2": {"percentile": 80.27, "p_value": 0.1973},
}

TRADING_PERIODS: list[tuple[str, str, str]] = [
    ("trading_period_1", config.TRADING_PERIOD_1_START, config.TRADING_PERIOD_1_END),
    ("trading_period_2", config.TRADING_PERIOD_2_START, config.TRADING_PERIOD_2_END),
]


def build_actual_portfolio_returns(prices: pd.DataFrame) -> dict[str, float]:
    """Re-derive the actual selected-pairs portfolio's cumulative net return per trading period.

    Reuses the existing screening -> hedge_ratio -> signals -> backtest
    pipeline functions exactly as main.py does, without reimplementing any
    of that logic and without writing any of the CSVs those functions'
    main.py wrappers normally produce (this script only ever writes its
    own convergence-comparison CSV).

    Args:
        prices: Full candidate price panel, columns=tickers, indexed by
            date (`data.load_all_candidate_prices()`'s output).

    Returns:
        Dict mapping period name ("trading_period_1"/"trading_period_2")
        to that period's actual equal-weighted portfolio cumulative net
        return.
    """
    formation_prices = prices.loc[config.FORMATION_START : config.FORMATION_END]

    covid_sensitivity = screening.screen_covid_sensitivity(
        formation_prices, config.FORMATION_START, config.FORMATION_END
    )
    source_label = "ex_covid" if config.USE_COVID_EXCLUDED_SCREENING else "full_period"
    screening_df = covid_sensitivity[source_label]
    selection = screening.rank_and_select_pairs(screening_df, source_label=source_label)
    selected_pairs = selection["selected_pairs"]

    hedge_ratios = hedge_ratio.build_hedge_ratios_for_selected_pairs(
        formation_prices, selected_pairs, config.FORMATION_START, config.FORMATION_END
    )

    signals_dict = signals.build_signals_for_selected_pairs(
        prices,
        hedge_ratios,
        config.FORMATION_START,
        config.FORMATION_END,
        config.TRADING_PERIOD_1_START,
        config.TRADING_PERIOD_1_END,
        config.TRADING_PERIOD_2_START,
        config.TRADING_PERIOD_2_END,
    )

    backtest_results = backtest.backtest_all_selected_pairs(prices, signals_dict, hedge_ratios)

    return {
        period_name: float(backtest.compute_average_pair_return(backtest_results, period_name).iloc[-1])
        for period_name, _, _ in TRADING_PERIODS
    }


def main() -> None:
    """Run the convergence check and save/print the comparison table."""
    print("[permutation_convergence_check] Loading cached price data...", flush=True)
    prices = data.load_all_candidate_prices()

    print("[permutation_convergence_check] Re-deriving actual selected-pairs portfolio returns...", flush=True)
    actual_portfolio_return = build_actual_portfolio_returns(prices)
    for period_name, value in actual_portfolio_return.items():
        print(f"  {period_name}: actual_portfolio_return={value:.6f}", flush=True)

    results_rows: list[dict[str, object]] = []
    for period_name, _, _ in TRADING_PERIODS:
        baseline = BASELINE_RESULTS[period_name]
        results_rows.append(
            {
                "n_simulations": BASELINE_N_SIMULATIONS,
                "period": period_name,
                "percentile": baseline["percentile"],
                "p_value": baseline["p_value"],
                "wall_clock_seconds": float("nan"),
            }
        )

    for n_simulations in SAMPLE_SIZES:
        for period_name, period_start, period_end in TRADING_PERIODS:
            print(
                f"\n[permutation_convergence_check] Running n_simulations={n_simulations:,} "
                f"for {period_name}...",
                flush=True,
            )
            start = time.perf_counter()
            result = validation.random_pair_benchmark(
                prices,
                config.FORMATION_START,
                config.FORMATION_END,
                period_start,
                period_end,
                actual_portfolio_return[period_name],
                n_simulations=n_simulations,
            )
            elapsed = time.perf_counter() - start

            print(
                f"[permutation_convergence_check] Done: n_simulations={n_simulations:,}, "
                f"period={period_name}, percentile={result['percentile_of_actual']:.2f}, "
                f"p_value={result['p_value_actual_ge_random']:.4f}, elapsed={elapsed:.1f}s",
                flush=True,
            )

            results_rows.append(
                {
                    "n_simulations": n_simulations,
                    "period": period_name,
                    "percentile": result["percentile_of_actual"],
                    "p_value": result["p_value_actual_ge_random"],
                    "wall_clock_seconds": elapsed,
                }
            )

    comparison_df = pd.DataFrame(results_rows)[
        ["n_simulations", "period", "percentile", "p_value", "wall_clock_seconds"]
    ]
    comparison_df = comparison_df.sort_values(["period", "n_simulations"]).reset_index(drop=True)

    print("\n=== Permutation test convergence comparison ===")
    print(comparison_df.to_string(index=False))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "validation_permutation_convergence.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"\n[permutation_convergence_check] Saved comparison table to {out_path}")


if __name__ == "__main__":
    main()
