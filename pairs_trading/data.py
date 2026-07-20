"""Data acquisition and local caching of daily adjusted close prices.

Downloads prices via yfinance (no API key required) and caches them as
parquet files under `config.DATA_DIR` so the pipeline does not hit the
network on every run. All downstream modules should consume prices through
`load_price_panel` / `load_all_candidate_prices` rather than calling
yfinance directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from pairs_trading import config

logger = logging.getLogger(__name__)

# Drop a ticker from a panel if more than this fraction of aligned trading
# days are missing after the outer join (e.g. a late IPO or delisting).
MAX_MISSING_FRACTION: float = 0.05


def _cache_path(ticker: str) -> Path:
    """Return the parquet cache path for a single ticker.

    Args:
        ticker: Ticker symbol, e.g. "KO".

    Returns:
        Path to the cached parquet file for this ticker.
    """
    return config.DATA_DIR / f"{ticker}.parquet"


def _download_single_ticker(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices for one ticker from yfinance.

    Uses `auto_adjust=True` so the returned "Close" column is already
    split/dividend adjusted.

    Args:
        ticker: Ticker symbol to download.
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD" (exclusive, per yfinance convention).

    Returns:
        Series of adjusted close prices indexed by date, named `ticker`.

    Raises:
        ValueError: If yfinance returns no data for the ticker.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        # yfinance can return a single-column MultiIndex frame depending on version.
        close = close.iloc[:, 0]
    close = close.rename(ticker)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close


def get_price_series(
    ticker: str,
    start: str = config.DATA_START,
    end: str = config.DATA_END,
    force_refresh: bool = False,
) -> pd.Series:
    """Get an adjusted close price series for one ticker, using the local cache.

    On first request for a ticker, downloads from yfinance and writes a
    parquet cache file. Subsequent calls read from the cache unless
    `force_refresh` is set.

    Args:
        ticker: Ticker symbol to fetch.
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        force_refresh: If True, bypass the cache and re-download from yfinance.

    Returns:
        Series of adjusted close prices indexed by date, named `ticker`,
        sliced to [start, end].
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(ticker)

    if cache_file.exists() and not force_refresh:
        logger.info("Loading %s from cache: %s", ticker, cache_file)
        series = pd.read_parquet(cache_file).iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        series.name = ticker
        return series.loc[start:end]

    logger.info("Downloading %s from yfinance (%s to %s)", ticker, start, end)
    series = _download_single_ticker(ticker, config.DATA_START, config.DATA_END)
    series.to_frame().to_parquet(cache_file)
    return series.loc[start:end]


def load_price_panel(
    tickers: list[str],
    start: str = config.DATA_START,
    end: str = config.DATA_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build an aligned panel of adjusted close prices for multiple tickers.

    Downloads/caches each ticker individually, aligns them on a common
    date index via an outer join, forward-fills short gaps (e.g. a single
    exchange holiday not shared across listings), and drops any ticker
    whose remaining missing fraction exceeds `MAX_MISSING_FRACTION`.

    Args:
        tickers: Ticker symbols to include in the panel.
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        force_refresh: If True, bypass the cache and re-download every ticker.

    Returns:
        DataFrame of adjusted close prices, columns=tickers (surviving
        tickers only), indexed by date, with no remaining NaNs.

    Raises:
        ValueError: If fewer than 2 tickers remain after cleaning.
    """
    series_list = [
        get_price_series(ticker, start=start, end=end, force_refresh=force_refresh) for ticker in tickers
    ]
    panel = pd.concat(series_list, axis=1, join="outer").sort_index()

    missing_fraction = panel.isna().mean()
    dropped = missing_fraction[missing_fraction > MAX_MISSING_FRACTION].index.tolist()
    if dropped:
        logger.warning(
            "Dropping tickers with >%.0f%% missing data after alignment: %s",
            MAX_MISSING_FRACTION * 100,
            dropped,
        )
        panel = panel.drop(columns=dropped)

    panel = panel.ffill().dropna()

    if panel.shape[1] < 2:
        raise ValueError("Fewer than 2 tickers remain after cleaning; cannot form pairs.")

    return panel


def load_all_candidate_prices(force_refresh: bool = False) -> pd.DataFrame:
    """Load the aligned price panel for every ticker referenced in config.

    Includes both legs of every pair in `config.CANDIDATE_PAIRS` plus
    `config.BENCHMARK_TICKER`, over the full `config.DATA_START` to
    `config.DATA_END` range.

    Args:
        force_refresh: If True, bypass the cache and re-download every ticker.

    Returns:
        DataFrame of adjusted close prices, columns=tickers, indexed by date.

    Raises:
        ValueError: If `config.CANDIDATE_PAIRS` is empty.
    """
    if not config.CANDIDATE_PAIRS:
        raise ValueError(
            "config.CANDIDATE_PAIRS is empty. Confirm the 5 candidate pairs and "
            "populate config.py before loading data."
        )
    tickers = sorted({t for pair in config.CANDIDATE_PAIRS for t in pair} | {config.BENCHMARK_TICKER})
    return load_price_panel(tickers)


def plot_normalised_prices(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot price series normalised to 1.0 at the start of the sample.

    Quick visual gut-check of co-movement between candidate pair legs
    before running formal screening.

    Args:
        prices: DataFrame of price levels, columns=tickers, indexed by date.
        tickers: Subset of columns to plot. Defaults to all columns.
        ax: Optional existing matplotlib Axes to draw on. A new figure is
            created if not provided.

    Returns:
        The matplotlib Axes the plot was drawn on.
    """
    if tickers is None:
        tickers = list(prices.columns)

    normalised = prices[tickers] / prices[tickers].iloc[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    normalised.plot(ax=ax)
    ax.set_title("Normalised Adjusted Close Prices")
    ax.set_ylabel("Price (normalised to 1.0 at start)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")

    return ax
