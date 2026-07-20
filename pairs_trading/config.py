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

# ---------------------------------------------------------------------------
# Hedge ratio estimation (hedge_ratio.py)
# ---------------------------------------------------------------------------
KALMAN_DELTA: Final[float] = 1e-4  # state transition covariance tuning parameter (smaller = smoother beta)
KALMAN_OBS_COVARIANCE: Final[float] = 1e-3  # observation noise variance
KALMAN_INITIAL_STATE_MEAN: Final[list[float]] = [0.0, 1.0]  # [intercept, beta] prior

# ---------------------------------------------------------------------------
# Signal generation (signals.py)
# ---------------------------------------------------------------------------
ZSCORE_LOOKBACK: Final[int] = 30  # rolling window (trading days) for spread mean/std
ENTRY_ZSCORE: Final[float] = 2.0
EXIT_ZSCORE: Final[float] = 0.5
STOP_LOSS_ZSCORE: Final[float] = 4.0
USE_VOLATILITY_ADJUSTED_THRESHOLD: Final[bool] = False
VOLATILITY_LOOKBACK: Final[int] = 60  # window for the volatility-adjusted threshold variant

# Grids used for threshold sensitivity analysis.
ENTRY_ZSCORE_GRID: Final[list[float]] = [1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_ZSCORE_GRID: Final[list[float]] = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Backtest (backtest.py)
# ---------------------------------------------------------------------------
CAPITAL_PER_PAIR: Final[float] = 1.0  # $1-per-pair sizing convention (Gatev et al. 2006)
TRANSACTION_COST_BPS: Final[float] = 5.0  # basis points per leg, charged on each trade entry/exit
RISK_FREE_RATE_ANNUAL: Final[float] = 0.02

# ---------------------------------------------------------------------------
# Validation (validation.py)
# ---------------------------------------------------------------------------
N_BOOTSTRAP_SAMPLES: Final[int] = 1000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42
N_RANDOM_PAIRS_BENCHMARK: Final[int] = 50

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: Final[int] = 252
