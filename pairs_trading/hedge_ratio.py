"""Hedge ratio estimation: static OLS and rolling Kalman filter.

The hedge ratio (beta) defines the spread for a pair:
`spread_t = dependent_t - (alpha + beta * independent_t)`.
`static_ols_hedge_ratio` estimates a single, fixed alpha/beta over the
formation period; `compute_spread` and `spread_zscore` then apply that fixed
relationship unchanged to any window (formation or a later out-of-sample
trading period) so no re-estimation ever leaks future information into a
supposedly historical hedge ratio.

`kalman_filter_hedge_ratio` (rolling, time-varying beta) remains a stub for
a later phase.

`ols_regress` is a small shared regression primitive used both here and by
`screening.cointegration_test`, which needs OLS residuals for its manual
generic-ADF cointegration check. Centralising it here (rather than each
module fitting its own `statsmodels.api.OLS` call) means the estimation
logic — and any future change to it — exists in exactly one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def ols_regress(y: pd.Series, x: pd.Series) -> dict[str, float | pd.Series]:
    """Fit y = alpha + beta * x by OLS and return the fit and its residuals.

    Rows where either `y` or `x` is missing are dropped before fitting, so
    callers do not need to pre-align the two series.

    Args:
        y: Dependent series.
        x: Independent series. Does not need to already share `y`'s index.

    Returns:
        Dict with keys "alpha" (intercept), "beta" (slope), "r_squared",
        and "residuals" (`y - (alpha + beta * x)`, indexed like the
        aligned, NaN-dropped input).

    Raises:
        ValueError: If fewer than 2 overlapping, non-missing observations
            remain between `y` and `x`.
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("Need at least 2 overlapping observations to fit an OLS hedge ratio.")

    fitted = sm.OLS(aligned["y"], sm.add_constant(aligned["x"])).fit()

    return {
        "alpha": float(fitted.params.iloc[0]),
        "beta": float(fitted.params.iloc[1]),
        "r_squared": float(fitted.rsquared),
        "residuals": fitted.resid.rename("residuals"),
    }


def _parse_direction(direction: str, ticker_a: str, ticker_b: str) -> tuple[str, str]:
    """Parse a "DEPENDENT~INDEPENDENT" direction string into (dependent, independent) tickers.

    Args:
        direction: String in the format produced by
            `screening.cointegration_test`'s "eg_direction" column, e.g.
            "KO~PEP" meaning KO was regressed on PEP.
        ticker_a: One leg of the pair, for validation.
        ticker_b: The other leg of the pair, for validation.

    Returns:
        Tuple of (dependent_ticker, independent_ticker).

    Raises:
        ValueError: If `direction` isn't of the form "X~Y", or {X, Y} != {ticker_a, ticker_b}.
    """
    parts = direction.split("~")
    if len(parts) != 2:
        raise ValueError(f"direction must be formatted 'DEPENDENT~INDEPENDENT', got {direction!r}")

    dependent, independent = parts
    if {dependent, independent} != {ticker_a, ticker_b}:
        raise ValueError(
            f"direction {direction!r} does not reference the given pair ({ticker_a}, {ticker_b})"
        )
    return dependent, independent


def static_ols_hedge_ratio(
    price_panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    direction: str | None = None,
) -> dict[str, float | str]:
    """Estimate a single, static OLS hedge ratio for one pair over a fixed window.

    Fits `dependent = alpha + beta * independent` over [start, end].
    Intended to be fit on the formation period and then held fixed
    (via `compute_spread`) for a subsequent out-of-sample trading period.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        ticker_a: One leg of the pair.
        ticker_b: The other leg of the pair.
        start: Window start date, "YYYY-MM-DD". Intended to be the
            formation period start, since the resulting alpha/beta are
            meant to be held fixed for later out-of-sample use.
        end: Window end date, "YYYY-MM-DD".
        direction: Which ticker to regress as the dependent variable,
            formatted "DEPENDENT~INDEPENDENT" (matching
            `screening.cointegration_test`'s "eg_direction" output).
            Defaults to `f"{ticker_a}~{ticker_b}"` (ticker_a dependent) if
            not supplied. Always pass the pair's actual `eg_direction`
            from screening when one is available: `cointegration_test`
            already determined which direction showed cointegration
            evidence, and regressing the other way can produce a spread
            with materially different (and less mean-reverting) behaviour
            even for a genuinely cointegrated pair.

    Returns:
        Dict with keys "dependent_ticker", "independent_ticker", "alpha",
        "beta", "r_squared". Pass "dependent_ticker"/"independent_ticker"
        (not necessarily `ticker_a`/`ticker_b` in that order) as the
        `ticker_a`/`ticker_b` arguments to `compute_spread`, so the spread
        formula matches the orientation this hedge ratio was actually fit in.

    Raises:
        ValueError: If `direction` is malformed or doesn't reference
            {ticker_a, ticker_b}, or if fewer than 2 overlapping
            observations exist in [start, end].
    """
    if direction is None:
        direction = f"{ticker_a}~{ticker_b}"
    dependent_ticker, independent_ticker = _parse_direction(direction, ticker_a, ticker_b)

    window = price_panel.loc[start:end, [ticker_a, ticker_b]].dropna()
    if len(window) < 2:
        raise ValueError(
            f"Insufficient overlapping price data for {ticker_a}/{ticker_b} in [{start}, {end}] "
            f"({len(window)} observations)."
        )

    fit = ols_regress(window[dependent_ticker], window[independent_ticker])

    return {
        "dependent_ticker": dependent_ticker,
        "independent_ticker": independent_ticker,
        "alpha": fit["alpha"],
        "beta": fit["beta"],
        "r_squared": fit["r_squared"],
    }


def compute_spread(
    price_panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    alpha: float,
    beta: float,
    start: str,
    end: str,
) -> pd.Series:
    """Compute the spread `ticker_a - (alpha + beta * ticker_b)` over [start, end].

    `alpha`/`beta` are always FIXED inputs — typically the output of
    `static_ols_hedge_ratio` run on the formation period only — and are
    never re-estimated here. This is what lets the same function be used
    unchanged on the formation period (to establish the baseline spread
    mean/std) and on later out-of-sample trading periods (to generate live
    signals): the regression that defines the spread is always anchored to
    what was known at the end of formation, never to whatever window is
    currently being evaluated. Re-fitting alpha/beta on the trading-period
    data itself would leak out-of-sample information into a supposedly
    historical hedge ratio.

    `ticker_a` must be the same ticker used as the *dependent* variable
    when `alpha`/`beta` were estimated (`static_ols_hedge_ratio`'s
    "dependent_ticker"), and `ticker_b` the independent one
    ("independent_ticker") — passing them in the wrong order silently
    produces a meaningless spread.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        ticker_a: Dependent-side ticker (matching the alpha/beta fit).
        ticker_b: Independent-side ticker (matching the alpha/beta fit).
        alpha: Fixed intercept, from a formation-period OLS fit.
        beta: Fixed hedge ratio (slope), from a formation-period OLS fit.
        start: Window start date, "YYYY-MM-DD".
        end: Window end date, "YYYY-MM-DD".

    Returns:
        Series of spread values indexed by date, named "spread".

    Raises:
        ValueError: If no overlapping observations exist for the pair in
            [start, end].
    """
    window = price_panel.loc[start:end, [ticker_a, ticker_b]].dropna()
    if window.empty:
        raise ValueError(f"No overlapping price data for {ticker_a}/{ticker_b} in [{start}, {end}].")

    spread = window[ticker_a] - (alpha + beta * window[ticker_b])
    return spread.rename("spread")


def spread_zscore(
    spread_series: pd.Series,
    formation_mean: float,
    formation_std: float,
) -> pd.Series:
    """Standardise a spread series using a FIXED formation-period mean/std.

    This is the single most important no-lookahead-bias rule in the whole
    pipeline: `formation_mean`/`formation_std` must always come from the
    spread's own formation-period statistics — computed once (e.g.
    `compute_spread(...).mean()` / `.std()` over the formation window, as
    `build_hedge_ratios_for_selected_pairs` does) — and never recomputed
    from whatever window `spread_series` itself happens to cover. If a
    trading-period z-score were instead judged against that trading
    period's own mean/std, the score would be centred on data the strategy
    could not have known in advance, and the backtest would silently
    cheat: the z-scores would look artificially well-behaved (mean-reverting
    to 0 by construction) regardless of whether the pair still tracks its
    formation-period relationship at all.

    Args:
        spread_series: Spread series (e.g. from `compute_spread`), indexed
            by date, over any window (formation or a trading period).
        formation_mean: Mean of the spread over the formation period only.
        formation_std: Standard deviation of the spread over the
            formation period only. Must be strictly positive.

    Returns:
        Series `(spread_series - formation_mean) / formation_std`, same
        index as `spread_series`, named "zscore".

    Raises:
        ValueError: If `formation_std` is zero or negative.
    """
    if formation_std <= 0:
        raise ValueError(f"formation_std must be positive, got {formation_std}")

    return ((spread_series - formation_mean) / formation_std).rename("zscore")


def estimate_half_life(spread_series: pd.Series) -> dict[str, float]:
    """Estimate the half-life of mean reversion of a spread series.

    Regresses the change in spread against the lagged spread,
    `Δspread_t = c + lambda * spread_{t-1} + epsilon_t` (the standard
    discretisation of an Ornstein-Uhlenbeck process), and converts the
    mean-reversion speed into a half-life: `half_life = -ln(2) / lambda`.
    A more negative `lambda` means faster reversion and a shorter
    half-life; `lambda >= 0` implies the series is not mean-reverting
    (random walk or explosive) over the window tested, in which case
    half-life is reported as infinite rather than a misleading negative
    number.

    Note: this is a descriptive diagnostic, used in the report to justify
    the trading signal's entry/exit/max-holding choices
    (`config.DEFAULT_ENTRY_THRESHOLD`, `config.MAX_HOLDING_DAYS`, etc. in
    signals.py) — it is not itself a trading rule and is not used anywhere
    in the live signal/backtest logic.

    Args:
        spread_series: Spread series (e.g. from `compute_spread`), indexed
            by date. Typically the formation-period spread, so the
            half-life reflects only information available at formation end.

    Returns:
        Dict with keys "half_life_days" (float, in the same units as
        `spread_series`'s index frequency — trading days for daily data;
        `float("inf")` if lambda >= 0) and "r_squared" (fit quality of the
        underlying regression, as a rough diagnostic only).

    Raises:
        ValueError: If fewer than 2 valid (delta, lagged) observation
            pairs remain after differencing and lagging.
    """
    spread_series = spread_series.dropna()
    delta = spread_series.diff()
    lagged = spread_series.shift(1)

    regression_df = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1).dropna()
    if len(regression_df) < 2:
        raise ValueError("Need at least 2 valid (delta, lagged) observation pairs to estimate half-life.")

    fit = ols_regress(regression_df["delta"], regression_df["lagged"])
    lambda_coef = fit["beta"]

    half_life_days = float("inf") if lambda_coef >= 0 else float(-np.log(2) / lambda_coef)

    return {"half_life_days": half_life_days, "r_squared": fit["r_squared"]}


def build_hedge_ratios_for_selected_pairs(
    price_panel: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
    formation_start: str,
    formation_end: str,
) -> pd.DataFrame:
    """Compute and cache formation-period hedge ratios for every selected pair.

    For each pair in `selected_pairs_df` (typically
    `screening.rank_and_select_pairs(...)["selected_pairs"]`), fits a
    static OLS hedge ratio over [formation_start, formation_end] using the
    exact regression direction ("eg_direction") that pair was screened
    under, computes the resulting formation-period spread and its
    mean/std (the fixed baseline `spread_zscore` must use for every later
    trading-period window), and estimates the spread's half-life of mean
    reversion.

    Everything downstream (signals.py for z-score thresholds, backtest.py
    for spread reconstruction and position sizing) should read from this
    table rather than recomputing any of these figures itself, so exactly
    the same alpha/beta/mean/std is used consistently everywhere a given
    pair's spread is referenced.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed
            by date, covering at least [formation_start, formation_end].
        selected_pairs_df: DataFrame with one row per pair and at least
            "pair", "ticker_a", "ticker_b", "eg_direction" columns (the
            schema produced by `screening.screen_all_candidate_pairs`
            and carried through by `screening.rank_and_select_pairs`).
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".

    Returns:
        DataFrame with one row per pair and columns: "pair",
        "dependent_ticker", "independent_ticker", "alpha", "beta",
        "r_squared", "half_life_days", "half_life_r_squared",
        "formation_spread_mean", "formation_spread_std".
    """
    rows = []
    for _, pair_row in selected_pairs_df.iterrows():
        ticker_a = pair_row["ticker_a"]
        ticker_b = pair_row["ticker_b"]
        direction = pair_row["eg_direction"]

        hedge = static_ols_hedge_ratio(
            price_panel, ticker_a, ticker_b, formation_start, formation_end, direction=direction
        )
        formation_spread = compute_spread(
            price_panel,
            hedge["dependent_ticker"],
            hedge["independent_ticker"],
            hedge["alpha"],
            hedge["beta"],
            formation_start,
            formation_end,
        )
        half_life = estimate_half_life(formation_spread)

        rows.append(
            {
                "pair": pair_row["pair"],
                "dependent_ticker": hedge["dependent_ticker"],
                "independent_ticker": hedge["independent_ticker"],
                "alpha": hedge["alpha"],
                "beta": hedge["beta"],
                "r_squared": hedge["r_squared"],
                "half_life_days": half_life["half_life_days"],
                "half_life_r_squared": half_life["r_squared"],
                "formation_spread_mean": float(formation_spread.mean()),
                "formation_spread_std": float(formation_spread.std()),
            }
        )

    return pd.DataFrame(rows)


def kalman_filter_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float,
    obs_covariance: float,
) -> pd.DataFrame:
    """Estimate a time-varying hedge ratio via a Kalman filter.

    Models [alpha_t, beta_t] as a random walk (state transition covariance
    controlled by `delta`) with observation equation y_t = alpha_t + beta_t * x_t
    + epsilon_t (variance `obs_covariance`). At each time step, the filter
    uses only data up to and including that step, so the resulting series
    can be used directly in a no-lookahead backtest.

    Args:
        y: Dependent price series.
        x: Independent price series.
        delta: State transition covariance tuning parameter (smaller values
            produce a smoother, slower-moving beta).
        obs_covariance: Observation noise variance.

    Returns:
        DataFrame indexed like y/x with columns ["alpha", "beta", "spread"],
        where "spread" is the filtered residual y_t - (alpha_t + beta_t * x_t).
    """
    raise NotImplementedError
