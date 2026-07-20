# Pairs Trading Research Pipeline

ICM612 Algorithmic and High Frequency Trading, Henley Business School.

A simplified implementation of the Gatev, Goetzmann & Rouwenhorst (2006)
distance-method pairs trading strategy, extended with a Kalman filter
dynamic hedge ratio, threshold sensitivity analysis, and bootstrap
significance testing.

## Project status

Phase 1 (scaffolding) complete. `data.py` is fully implemented; all other
pipeline modules (`screening.py`, `hedge_ratio.py`, `signals.py`,
`backtest.py`, `metrics.py`, `validation.py`, `main.py`) are stubs with
finalised signatures and docstrings, to be implemented module-by-module.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All tunable parameters (date ranges, thresholds, candidate universe) live
in `config.py`. In particular, `config.CANDIDATE_PAIRS` must be populated
with the 5 confirmed (ticker_a, ticker_b) pairs before running the pipeline.

- **Data window:** 2017-01-01 to 2025-12-31
- **Formation period:** 2017-2021 (used for screening and static hedge ratios)
- **Trading periods:** 2022-2023 and 2024-2025 (out-of-sample)

## Running the pipeline

```bash
python -m pairs_trading.main
```

This downloads/caches price data, screens candidate pairs over the
formation period, estimates hedge ratios, generates signals, backtests
each trading period, and produces summary tables/figures.

## Data caching

Adjusted close prices are downloaded via `yfinance` (no API key required)
and cached as parquet files under `pairs_trading/data_cache/`, keyed by
ticker. Delete a ticker's cache file (or pass `force_refresh=True`) to
re-download.

## Tests

```bash
pytest pairs_trading/tests
```

## Notebooks

`notebooks/exploration.ipynb` is for ad-hoc plots and scratch analysis. It
is not imported by the pipeline and is not required to run `main.py`.
