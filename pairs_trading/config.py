"""Central configuration for the pairs trading research pipeline.

Every tunable parameter used by the logic modules (screening, hedge_ratio,
signals, backtest, metrics, validation) lives here. No other module should
contain magic numbers or hardcoded date ranges/thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data_cache"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# 5 candidate pairs from S&P 500 components, as (ticker_a, ticker_b) tuples.
# Confirmed with the user 2026-07-20:
#   JPM / BAC  - financials, sector-peer money-center banks
#   XOM / CVX  - energy, integrated oil majors on the same commodity driver
#                (note: April 2020 negative oil prices makes this a likely
#                structural-break candidate around COVID)
#   HD  / LOW  - consumer discretionary, near-identical home-improvement retail
#   KO  / PEP  - consumer staples, classic textbook pair; expected to be the
#                "clean" control pair with a well-behaved spread
#   DAL / UAL  - industrials/travel, volatile COVID collapse/recovery narrative
CANDIDATE_PAIRS: Final[list[tuple[str, str]]] = [
    ("JPM", "BAC"),
    ("XOM", "CVX"),
    ("HD", "LOW"),
    ("KO", "PEP"),
    ("DAL", "UAL"),
]

BENCHMARK_TICKER: Final[str] = "SPY"

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------
DATA_START: Final[str] = "2017-01-01"
DATA_END: Final[str] = "2025-12-31"

# Formation period: used to screen pairs and estimate static hedge ratios.
FORMATION_START: Final[str] = "2017-01-01"
FORMATION_END: Final[str] = "2021-12-31"

# Trading (out-of-sample) periods.
TRADING_PERIOD_1_START: Final[str] = "2022-01-01"
TRADING_PERIOD_1_END: Final[str] = "2023-12-31"

TRADING_PERIOD_2_START: Final[str] = "2024-01-01"
TRADING_PERIOD_2_END: Final[str] = "2025-12-31"

# ---------------------------------------------------------------------------
# Screening thresholds (screening.py)
# ---------------------------------------------------------------------------
MIN_CORRELATION: Final[float] = 0.80
SSD_TOP_N: Final[int] = 20  # number of lowest-SSD pairs to shortlist before cointegration testing
COINTEGRATION_SIGNIFICANCE: Final[float] = 0.05
# Critical values for the "generic" (naive) Engle-Granger residual ADF test,
# as commonly (and incorrectly) taken from standard ADF tables rather than
# the correct Engle-Granger/MacKinnon tables. Kept for comparison purposes.
GENERIC_ADF_CRITICAL_VALUES: Final[dict[str, float]] = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

# COVID-19 crash window. Used by screening.screen_covid_sensitivity to test
# whether excluding this window from the formation period materially changes
# correlation, SSD, or cointegration results for each candidate pair, per the
# brief's requirement to justify including/excluding the crash.
COVID_EXCLUSION_START: Final[str] = "2020-02-01"
COVID_EXCLUSION_END: Final[str] = "2020-06-30"

# Number of pairs carried forward from formation-period screening into the
# trading periods.
N_PAIRS_TO_SELECT: Final[int] = 3

# Deliberate, evidence-based screening decision (see
# screen_covid_sensitivity() results, pairs_trading/results/screening_covid_comparison.csv):
# none of the 5 candidate pairs are cointegrated at 5% over the full
# 2017-2021 formation period, but excluding the COVID crash window causes
# HD/LOW and KO/PEP to clear cointegration at 5%. Final pair selection is
# therefore run on the ex-COVID screening table, not the full-period one.
# This is stated explicitly here (rather than left as an unstated default)
# so it is visible in the report and must be revisited if the candidate
# universe or date ranges change.
USE_COVID_EXCLUDED_SCREENING: Final[bool] = True

# Primary cointegration bar every candidate pair is judged against.
PRIMARY_COINTEGRATION_THRESHOLD: Final[float] = 0.05

# Looser bar applied only to the named exceptions in
# PAIRS_REQUIRING_RELAXED_THRESHOLD below, never to the universe as a whole.
RELAXED_COINTEGRATION_THRESHOLD: Final[float] = 0.10

# DAL/UAL clears cointegration only at the 10% level even ex-COVID (not 5%),
# but shows the largest absolute improvement in Engle-Granger p-value of any
# candidate pair when the COVID window is excluded (0.292 -> 0.074), and
# travel was the most COVID-disrupted of the five candidate sectors. It is
# therefore carried as a named, justified exception to the primary
# threshold rather than lowering the bar for every pair.
PAIRS_REQUIRING_RELAXED_THRESHOLD: Final[list[str]] = ["DAL/UAL"]

# ---------------------------------------------------------------------------
# Hedge ratio estimation (hedge_ratio.py)
# ---------------------------------------------------------------------------
KALMAN_DELTA: Final[float] = 1e-4  # state transition covariance tuning parameter (smaller = smoother beta)
KALMAN_OBS_COVARIANCE: Final[float] = 1e-3  # observation noise variance
KALMAN_INITIAL_STATE_MEAN: Final[list[float]] = [0.0, 1.0]  # [intercept, beta] prior

# ---------------------------------------------------------------------------
# Signal generation (signals.py)
# ---------------------------------------------------------------------------
# Z-scoring uses the FIXED formation-period mean/std from hedge_ratio.py
# (hedge_ratio.spread_zscore), not a rolling window, so there is no
# "lookback" parameter here by design: re-centring against a rolling
# window would itself leak local (including future-adjacent) information
# into the signal.
DEFAULT_ENTRY_THRESHOLD: Final[float] = 2.0  # |z| to open a position
DEFAULT_EXIT_THRESHOLD: Final[float] = 0.0  # |z| to close on mean reversion (0 = revert to formation mean)

# Force-close any position still open after this many trading days, so a
# single non-reverting trade cannot dominate the backtest. Set comfortably
# above the longest formation-period half-life observed across the 3
# selected pairs (~60 trading days for DAL/UAL; see results/hedge_ratios.csv).
MAX_HOLDING_DAYS: Final[int] = 120

# Force-close a position if |z| reaches this level, as a risk control
# tracked separately from (and reported separately from) a normal
# mean-reversion exit. A stop-loss also blocks re-entry into the same pair
# ("cooling_down" in signals.generate_positions) until z first returns
# inside +/-DEFAULT_ENTRY_THRESHOLD's entry_threshold band, unlike a
# max-holding close which allows immediate re-entry -- see that function's
# docstring for the reasoning.
STOP_LOSS_THRESHOLD: Final[float] = 4.0

# If True, signals.build_signals_for_selected_pairs selects entry/exit
# thresholds per pair via a formation-period-only grid search
# (signals.threshold_sensitivity_grid) rather than using
# DEFAULT_ENTRY_THRESHOLD/DEFAULT_EXIT_THRESHOLD for every pair.
USE_THRESHOLD_GRID: Final[bool] = True

# Grids searched by signals.threshold_sensitivity_grid.
ENTRY_THRESHOLD_GRID: Final[list[float]] = [1.5, 2.0, 2.5, 3.0]
EXIT_THRESHOLD_GRID: Final[list[float]] = [0.0, 0.5]

# ---------------------------------------------------------------------------
# Backtest (backtest.py)
# ---------------------------------------------------------------------------
CAPITAL_PER_PAIR: Final[float] = 1.0  # $1-per-pair sizing convention (Gatev et al. 2006)
TRANSACTION_COST_BPS: Final[float] = 5.0  # basis points per leg, charged on each trade entry/exit
RISK_FREE_RATE_ANNUAL: Final[float] = 0.02

# ---------------------------------------------------------------------------
# Validation (validation.py)
# ---------------------------------------------------------------------------
# Number of circular block-bootstrap resamples per return series in
# bootstrap_significance_test. Updated from an initial placeholder of 1000
# to 10000 for a smoother empirical distribution/CI, now that the resampling
# is fully vectorised (see validation.py) and the extra draws are cheap.
N_BOOTSTRAP_SAMPLES: Final[int] = 10_000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42

# Block length (trading days) for the circular block bootstrap, chosen to
# be long enough to preserve short-run autocorrelation in the daily return
# series (roughly a trading month) without being so long that too few
# independent blocks are available to resample from.
BOOTSTRAP_BLOCK_SIZE: Final[int] = 20

# Number of Monte Carlo simulations in random_pair_benchmark. Each
# simulation re-runs the full hedge_ratio -> signals -> backtest pipeline
# on N_PAIRS_TO_SELECT randomly drawn pairs (unlike N_BOOTSTRAP_SAMPLES,
# which only needs cheap vectorised numpy resampling). Measured at ~1.1s
# per 50 simulations, so 10,000 simulations costs roughly 3-4 minutes --
# affordable, and worth it for a much less noisy p-value/percentile
# estimate, since this number is cited directly in the report as evidence
# for or against the selection methodology adding value.
N_RANDOM_PAIRS_BENCHMARK: Final[int] = 10_000

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: Final[int] = 252
