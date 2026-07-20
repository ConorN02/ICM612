"""Formation-period screening of candidate pairs.

Implements the screening criteria required by the assignment brief:
correlation of daily returns, average absolute and sum-of-squared-deviations
(SSD) of the normalised price spread (the Gatev, Goetzmann & Rouwenhorst
2006 distance metric), average daily dollar trading volume, and Engle-Granger
cointegration testing.

Two cointegration critical-value regimes are reported side by side for
comparison, as required by the brief's appendix example:

- The *proper* regime, via `statsmodels.tsa.stattools.coint`, which uses the
  Engle-Granger/MacKinnon response-surface critical values that correctly
  account for the fact that the cointegrating regression's residuals are
  themselves estimated (not directly observed).
- The *generic* regime, via a manual OLS regression followed by an
  Augmented Dickey-Fuller test on the residuals, compared against the
  standard (non-cointegration-adjusted) ADF critical values in
  `config.GENERIC_ADF_CRITICAL_VALUES`. This is a common but statistically
  invalid shortcut: because the generic critical values are less negative
  than the correct Engle-Granger ones, using them makes it too easy to
  reject the null of no cointegration, biasing the test toward false
  positives. It is included here purely so the two can be compared and the
  difference discussed in the report.

Every screening function operates on a `price_panel` (adjusted close prices,
columns=tickers, indexed by date) as produced by `data.load_all_candidate_prices`,
restricted by the caller to whatever date window is under test (full
formation period, or the formation period with the COVID-19 window excised).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from pairs_trading import config, hedge_ratio

logger = logging.getLogger(__name__)


def normalise_prices(price_panel: pd.DataFrame, base_date: str | pd.Timestamp) -> pd.DataFrame:
    """Rescale every price series in the panel to 1.0 at `base_date`.

    This is the normalisation step underlying the Gatev et al. (2006)
    distance method: expressing each stock as a cumulative-return index
    from a common starting point makes the SSD between two series a
    meaningful measure of co-movement, independent of the two stocks'
    absolute price levels.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        base_date: Date to normalise to. If not present in the index (e.g.
            a weekend or holiday), the next available trading day on or
            after `base_date` is used instead.

    Returns:
        DataFrame of the same shape as `price_panel`, where every column
        equals 1.0 at the (possibly adjusted) base date.

    Raises:
        ValueError: If no index date on or after `base_date` exists.
    """
    base_date = pd.Timestamp(base_date)

    if base_date not in price_panel.index:
        available_on_or_after = price_panel.index[price_panel.index >= base_date]
        if available_on_or_after.empty:
            raise ValueError(f"No price data on or after base_date {base_date.date()}")
        base_date = available_on_or_after[0]

    return price_panel / price_panel.loc[base_date]


def compute_pair_metrics(
    price_panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
) -> dict[str, float | str | int]:
    """Compute correlation, spread, and volume metrics for one pair.

    Correlation is computed on daily returns (not price levels) to avoid
    the spurious-correlation problem of two independently trending,
    non-stationary price series. The spread metrics follow Gatev et al.
    (2006): prices are normalised to 1.0 at the start of the window, and
    the spread is the difference between the two normalised series at each
    date.

    Limitation: `price_panel` (as produced by `data.py`) contains adjusted
    close prices only — `yf.download` is not asked for the "Volume" field,
    so daily dollar trading volume cannot currently be computed from the
    cached data. The two volume columns below are returned as NaN rather
    than silently omitted, so this limitation is visible in any report
    table built from this function's output rather than hidden by a
    missing column. Extending `data.py` to also cache volume would remove
    this limitation.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        ticker_a: First leg of the pair.
        ticker_b: Second leg of the pair.
        start: Window start date, "YYYY-MM-DD".
        end: Window end date, "YYYY-MM-DD".

    Returns:
        Dict with keys "ticker_a", "ticker_b", "n_obs", "correlation",
        "avg_abs_normalised_spread", "ssd_normalised_spread",
        "avg_dollar_volume_a", "avg_dollar_volume_b".

    Raises:
        ValueError: If fewer than 2 overlapping observations exist for the
            pair in [start, end].
    """
    window = price_panel.loc[start:end, [ticker_a, ticker_b]].dropna()
    if len(window) < 2:
        raise ValueError(
            f"Insufficient overlapping price data for {ticker_a}/{ticker_b} in [{start}, {end}] "
            f"({len(window)} observations)."
        )

    returns = window.pct_change().dropna()
    correlation = float(returns[ticker_a].corr(returns[ticker_b]))

    normalised = normalise_prices(window, window.index[0])
    spread = normalised[ticker_a] - normalised[ticker_b]

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "n_obs": len(window),
        "correlation": correlation,
        "avg_abs_normalised_spread": float(spread.abs().mean()),
        "ssd_normalised_spread": float((spread**2).sum()),
        # See docstring: not computable from the current price-only cache.
        "avg_dollar_volume_a": float("nan"),
        "avg_dollar_volume_b": float("nan"),
    }


def _generic_adf_critical_key(significance: float) -> str:
    """Map a significance level to the nearest key in `config.GENERIC_ADF_CRITICAL_VALUES`.

    Args:
        significance: Significance level as a fraction (e.g. 0.05).

    Returns:
        The matching key ("1%", "5%", or "10%"), defaulting to "5%" if the
        significance level doesn't exactly match a tabulated value.
    """
    mapping = {0.01: "1%", 0.05: "5%", 0.10: "10%"}
    return mapping.get(round(significance, 2), "5%")


def cointegration_test(
    price_panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
) -> dict[str, float | str | bool]:
    """Test a pair for cointegration using both a proper and a generic critical-value regime.

    Runs `statsmodels.tsa.stattools.coint` in both regression directions —
    a on b, and b on a — since the Engle-Granger test is not direction
    invariant in finite samples. The direction producing the lower p-value
    (stronger evidence against the null of no cointegration) is kept as
    "the" result; the other is discarded. This choice, and the reason for
    it, is recorded in the output so it can be justified in the report
    rather than presented as an arbitrary pick.

    For the *same* chosen direction, a second, manual test is run: OLS
    regress the dependent leg on the independent leg (via
    `hedge_ratio.ols_regress`, the same regression primitive
    `hedge_ratio.static_ols_hedge_ratio` uses, so this module never fits
    its own separate copy of that logic), then run an Augmented
    Dickey-Fuller test on the residuals, compared against the *generic* ADF
    critical values in `config.GENERIC_ADF_CRITICAL_VALUES` rather than the
    correct Engle-Granger/MacKinnon values `coint` already applied. This
    lets the report show, side by side, how much more conservative the
    proper cointegration critical values are than naively reusing a
    standard ADF table (see module docstring).

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        ticker_a: First leg of the pair.
        ticker_b: Second leg of the pair.
        start: Window start date, "YYYY-MM-DD".
        end: Window end date, "YYYY-MM-DD".

    Returns:
        Dict with keys "ticker_a", "ticker_b", "eg_direction",
        "eg_direction_reason", "eg_test_statistic", "eg_p_value",
        "eg_critical_value_1pct"/"_5pct"/"_10pct", "eg_is_cointegrated"
        (using `config.COINTEGRATION_SIGNIFICANCE` against the proper
        p-value), "ols_alpha", "ols_beta" (from the chosen-direction OLS
        regression), "generic_adf_statistic", "generic_adf_p_value",
        "generic_adf_critical_value_1pct"/"_5pct"/"_10pct",
        "generic_adf_is_cointegrated" (comparing the ADF statistic against
        the generic critical value nearest `config.COINTEGRATION_SIGNIFICANCE`).

    Raises:
        ValueError: If fewer than 20 overlapping observations exist for the
            pair in [start, end] (below this, both tests are unreliable).
    """
    window = price_panel.loc[start:end, [ticker_a, ticker_b]].dropna()
    if len(window) < 20:
        raise ValueError(
            f"Insufficient overlapping price data for {ticker_a}/{ticker_b} in [{start}, {end}] "
            f"({len(window)} observations); need at least 20 for a reliable cointegration test."
        )

    price_a, price_b = window[ticker_a], window[ticker_b]

    stat_ab, pvalue_ab, crit_ab = coint(price_a, price_b)
    stat_ba, pvalue_ba, crit_ba = coint(price_b, price_a)

    if pvalue_ab <= pvalue_ba:
        direction = f"{ticker_a}~{ticker_b}"
        dependent, independent = price_a, price_b
        eg_stat, eg_pvalue, eg_crit = stat_ab, pvalue_ab, crit_ab
        direction_reason = (
            f"Regressing {ticker_a} on {ticker_b} gave the lower Engle-Granger p-value "
            f"({pvalue_ab:.4f} vs {pvalue_ba:.4f} for the reverse regression), i.e. stronger "
            f"evidence against the null of no cointegration in that direction."
        )
    else:
        direction = f"{ticker_b}~{ticker_a}"
        dependent, independent = price_b, price_a
        eg_stat, eg_pvalue, eg_crit = stat_ba, pvalue_ba, crit_ba
        direction_reason = (
            f"Regressing {ticker_b} on {ticker_a} gave the lower Engle-Granger p-value "
            f"({pvalue_ba:.4f} vs {pvalue_ab:.4f} for the reverse regression), i.e. stronger "
            f"evidence against the null of no cointegration in that direction."
        )

    ols_fit = hedge_ratio.ols_regress(dependent, independent)
    generic_adf_stat, generic_adf_pvalue = adfuller(ols_fit["residuals"], autolag="AIC")[:2]

    generic_key = _generic_adf_critical_key(config.COINTEGRATION_SIGNIFICANCE)
    generic_critical_value = config.GENERIC_ADF_CRITICAL_VALUES[generic_key]

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "eg_direction": direction,
        "eg_direction_reason": direction_reason,
        "eg_test_statistic": float(eg_stat),
        "eg_p_value": float(eg_pvalue),
        "eg_critical_value_1pct": float(eg_crit[0]),
        "eg_critical_value_5pct": float(eg_crit[1]),
        "eg_critical_value_10pct": float(eg_crit[2]),
        "eg_is_cointegrated": bool(eg_pvalue <= config.COINTEGRATION_SIGNIFICANCE),
        "ols_alpha": ols_fit["alpha"],
        "ols_beta": ols_fit["beta"],
        "generic_adf_statistic": float(generic_adf_stat),
        "generic_adf_p_value": float(generic_adf_pvalue),
        "generic_adf_critical_value_1pct": config.GENERIC_ADF_CRITICAL_VALUES["1%"],
        "generic_adf_critical_value_5pct": config.GENERIC_ADF_CRITICAL_VALUES["5%"],
        "generic_adf_critical_value_10pct": config.GENERIC_ADF_CRITICAL_VALUES["10%"],
        "generic_adf_is_cointegrated": bool(generic_adf_stat <= generic_critical_value),
    }


def screen_all_candidate_pairs(
    price_panel: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    candidate_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Run the full formation-period screen over all candidate pairs.

    Combines `compute_pair_metrics` and `cointegration_test` for each pair
    into a single tidy row, adds pass/fail flags against the config
    thresholds (`config.MIN_CORRELATION`, `config.COINTEGRATION_SIGNIFICANCE`),
    and returns one DataFrame suitable for dropping directly into a report
    table. Row order matches `candidate_pairs` (no sorting/ranking is
    applied here — that is `rank_and_select_pairs`'s job).

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".
        candidate_pairs: Pairs to screen, as (ticker_a, ticker_b) tuples.
            Defaults to `config.CANDIDATE_PAIRS`.

    Returns:
        DataFrame with one row per candidate pair and columns: "pair",
        "ticker_a", "ticker_b", "n_obs", "correlation",
        "avg_abs_normalised_spread", "ssd_normalised_spread",
        "avg_dollar_volume_a", "avg_dollar_volume_b", the cointegration
        columns from `cointegration_test`, and "passes_correlation_threshold",
        "passes_cointegration", "passes_screen".
    """
    if candidate_pairs is None:
        candidate_pairs = config.CANDIDATE_PAIRS

    rows = []
    for ticker_a, ticker_b in candidate_pairs:
        metrics = compute_pair_metrics(price_panel, ticker_a, ticker_b, formation_start, formation_end)
        coint_result = cointegration_test(price_panel, ticker_a, ticker_b, formation_start, formation_end)

        row = {"pair": f"{ticker_a}/{ticker_b}", **metrics}
        row.update({k: v for k, v in coint_result.items() if k not in ("ticker_a", "ticker_b")})

        row["passes_correlation_threshold"] = row["correlation"] >= config.MIN_CORRELATION
        row["passes_cointegration"] = row["eg_is_cointegrated"]
        row["passes_screen"] = row["passes_correlation_threshold"] and row["passes_cointegration"]

        rows.append(row)

    return pd.DataFrame(rows)


def _exclude_date_range(panel: pd.DataFrame, exclude_start: str, exclude_end: str) -> pd.DataFrame:
    """Drop all rows whose date index falls within [exclude_start, exclude_end].

    Args:
        panel: DataFrame indexed by date.
        exclude_start: Start of the window to exclude, "YYYY-MM-DD" (inclusive).
        exclude_end: End of the window to exclude, "YYYY-MM-DD" (inclusive).

    Returns:
        DataFrame with the excluded rows removed; the surrounding pre- and
        post-window segments are concatenated with the gap simply absent
        from the index (not filled or interpolated).
    """
    outside_window = (panel.index < pd.Timestamp(exclude_start)) | (panel.index > pd.Timestamp(exclude_end))
    return panel.loc[outside_window]


def screen_covid_sensitivity(
    price_panel: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    covid_start: str = config.COVID_EXCLUSION_START,
    covid_end: str = config.COVID_EXCLUSION_END,
    candidate_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Re-run formation-period screening with and without the COVID-19 crash window.

    The Feb-Jun 2020 crash and recovery is an extreme, arguably one-off
    structural break. Including it in the formation period could either
    strengthen an apparent relationship (if both legs crashed and recovered
    together) or weaken it (if the two legs' liquidity/beta diverged
    sharply during the crash, e.g. DAL/UAL). This function makes that
    sensitivity visible rather than leaving it as an unstated modelling
    choice, directly supporting the brief's requirement to justify
    including or excluding the crash from the formation period.

    Args:
        price_panel: DataFrame of price levels, columns=tickers, indexed by date.
        formation_start: Formation window start date, "YYYY-MM-DD".
        formation_end: Formation window end date, "YYYY-MM-DD".
        covid_start: Start of the window to exclude in the "ex_covid" run.
            Defaults to `config.COVID_EXCLUSION_START`.
        covid_end: End of the window to exclude in the "ex_covid" run.
            Defaults to `config.COVID_EXCLUSION_END`.
        candidate_pairs: Pairs to screen, as (ticker_a, ticker_b) tuples.
            Defaults to `config.CANDIDATE_PAIRS`.

    Returns:
        Dict with keys:
            "full_period": output of `screen_all_candidate_pairs` on the
                unmodified formation period.
            "ex_covid": output of `screen_all_candidate_pairs` on the
                formation period with [covid_start, covid_end] excised.
            "comparison": one row per pair with correlation, SSD, and
                Engle-Granger p-value from both runs plus their differences
                (ex_covid minus full_period), for a direct include/exclude
                justification table.
    """
    if candidate_pairs is None:
        candidate_pairs = config.CANDIDATE_PAIRS

    formation_panel = price_panel.loc[formation_start:formation_end]
    full_results = screen_all_candidate_pairs(formation_panel, formation_start, formation_end, candidate_pairs)

    ex_covid_panel = _exclude_date_range(formation_panel, covid_start, covid_end)
    ex_covid_results = screen_all_candidate_pairs(
        ex_covid_panel, formation_start, formation_end, candidate_pairs
    )

    compare_cols = ["pair", "correlation", "ssd_normalised_spread", "eg_p_value"]
    comparison = full_results[compare_cols].merge(
        ex_covid_results[compare_cols],
        on="pair",
        suffixes=("_full_period", "_ex_covid"),
    )
    comparison["correlation_change"] = (
        comparison["correlation_ex_covid"] - comparison["correlation_full_period"]
    )
    comparison["ssd_normalised_spread_change"] = (
        comparison["ssd_normalised_spread_ex_covid"] - comparison["ssd_normalised_spread_full_period"]
    )
    comparison["eg_p_value_change"] = comparison["eg_p_value_ex_covid"] - comparison["eg_p_value_full_period"]

    return {"full_period": full_results, "ex_covid": ex_covid_results, "comparison": comparison}


def _evaluate_gate(pair: str, eg_p_value: float) -> tuple[bool, str, str]:
    """Decide whether one pair clears the cointegration gate, and record why.

    A pair passes if its Engle-Granger p-value clears
    `config.PRIMARY_COINTEGRATION_THRESHOLD`. Pairs named in
    `config.PAIRS_REQUIRING_RELAXED_THRESHOLD` get a second chance against
    the looser `config.RELAXED_COINTEGRATION_THRESHOLD` — but only those
    named pairs; everyone else is judged solely against the primary bar.

    Args:
        pair: Pair label, "ticker_a/ticker_b" (matching the "pair" column
            produced by `screen_all_candidate_pairs`).
        eg_p_value: Engle-Granger cointegration p-value for this pair.

    Returns:
        Tuple of (gate_passed, threshold_applied, reason), where
        threshold_applied is a short human-readable label and reason is a
        one-line plain-language explanation for the report appendix.
    """
    is_relaxed_eligible = pair in config.PAIRS_REQUIRING_RELAXED_THRESHOLD

    if eg_p_value <= config.PRIMARY_COINTEGRATION_THRESHOLD:
        threshold_applied = f"primary ({config.PRIMARY_COINTEGRATION_THRESHOLD:.0%})"
        reason = (
            f"Cleared the primary cointegration threshold "
            f"(p={eg_p_value:.4f} <= {config.PRIMARY_COINTEGRATION_THRESHOLD:.0%})."
        )
        return True, threshold_applied, reason

    if is_relaxed_eligible and eg_p_value <= config.RELAXED_COINTEGRATION_THRESHOLD:
        threshold_applied = f"relaxed ({config.RELAXED_COINTEGRATION_THRESHOLD:.0%}, justified exception)"
        reason = (
            f"Failed the primary {config.PRIMARY_COINTEGRATION_THRESHOLD:.0%} threshold but is a "
            f"pre-approved exception (see config.PAIRS_REQUIRING_RELAXED_THRESHOLD); cleared the "
            f"relaxed {config.RELAXED_COINTEGRATION_THRESHOLD:.0%} threshold (p={eg_p_value:.4f})."
        )
        return True, threshold_applied, reason

    if is_relaxed_eligible:
        threshold_applied = f"relaxed ({config.RELAXED_COINTEGRATION_THRESHOLD:.0%}, justified exception)"
        reason = (
            f"Pre-approved exception pair but still failed even the relaxed "
            f"{config.RELAXED_COINTEGRATION_THRESHOLD:.0%} threshold (p={eg_p_value:.4f})."
        )
        return False, threshold_applied, reason

    threshold_applied = f"primary ({config.PRIMARY_COINTEGRATION_THRESHOLD:.0%})"
    reason = (
        f"Failed the primary cointegration threshold "
        f"(p={eg_p_value:.4f} > {config.PRIMARY_COINTEGRATION_THRESHOLD:.0%}) and is not on the "
        f"relaxed-threshold exception list."
    )
    return False, threshold_applied, reason


def rank_and_select_pairs(
    screening_df: pd.DataFrame,
    n_select: int = config.N_PAIRS_TO_SELECT,
    source_label: str = "ex_covid",
) -> dict[str, pd.DataFrame]:
    """Gate candidate pairs on cointegration, then rank eligible pairs by SSD.

    Deliberately a two-stage "gate, then rank" procedure rather than a
    blended composite score. Correlation and SSD measure historical
    co-movement, but only cointegration is evidence that a spread actually
    mean-reverts; a pair with excellent correlation/SSD but no
    cointegration evidence has no basis for a mean-reversion strategy, so
    it must not be tradeable into the top n_select merely by scoring well
    on the other two metrics. Concretely:

    Step 1 (gate): a pair is eligible only if it clears
    `config.PRIMARY_COINTEGRATION_THRESHOLD`, or — for the specific,
    pre-justified exceptions in `config.PAIRS_REQUIRING_RELAXED_THRESHOLD`
    — clears the looser `config.RELAXED_COINTEGRATION_THRESHOLD` instead
    (see `_evaluate_gate`).

    Step 2 (rank): among eligible pairs only, rank by
    `ssd_normalised_spread` ascending — tighter historical tracking is
    more attractive, per Gatev et al. (2006).

    Step 3 (select): take the top `n_select` eligible pairs. If fewer than
    `n_select` pairs are eligible, all eligible pairs are returned and a
    warning is logged; ineligible pairs are never used to pad the count,
    since that would silently select a pair with no cointegration
    evidence.

    This is intended to run on the ex-COVID screening table
    (`screen_covid_sensitivity()["ex_covid"]`) when
    `config.USE_COVID_EXCLUDED_SCREENING` is True — the project's current,
    evidence-based default (see config.py: none of the 5 candidates are
    cointegrated at 5% over the full formation period, but two clear 5%
    once the COVID window is excluded). `source_label` must be passed
    explicitly stating which table `screening_df` is, and is checked
    against that config flag, so this function can never be silently
    pointed at the wrong table by accident.

    Args:
        screening_df: Output of `screen_all_candidate_pairs` (or one of
            the tables from `screen_covid_sensitivity`), with at least
            "pair", "eg_p_value", and "ssd_normalised_spread" columns.
        n_select: Number of top-ranked eligible pairs to select. Defaults
            to `config.N_PAIRS_TO_SELECT`.
        source_label: Which screening table `screening_df` is — either
            "full_period" or "ex_covid". Must match
            `config.USE_COVID_EXCLUDED_SCREENING`.

    Returns:
        Dict with keys:
            "selected_pairs": up to `n_select` eligible pairs (all of
                `screening_df`'s original columns, plus "gate_passed",
                "threshold_applied", "reason", "ssd_rank"), sorted by
                SSD ascending.
            "selection_audit": every candidate pair from `screening_df`
                with the same added columns plus "selected" — for the
                report appendix, so every pair's fate is traceable.

    Raises:
        ValueError: If `source_label` doesn't match
            `config.USE_COVID_EXCLUDED_SCREENING`.
    """
    expected_label = "ex_covid" if config.USE_COVID_EXCLUDED_SCREENING else "full_period"
    if source_label != expected_label:
        raise ValueError(
            f"rank_and_select_pairs expected a '{expected_label}' screening table "
            f"(config.USE_COVID_EXCLUDED_SCREENING={config.USE_COVID_EXCLUDED_SCREENING}), "
            f"but source_label='{source_label}'. Pass the matching table from "
            f"screen_covid_sensitivity() and its matching label explicitly."
        )

    gate_results = [
        _evaluate_gate(pair, eg_p_value)
        for pair, eg_p_value in zip(screening_df["pair"], screening_df["eg_p_value"])
    ]

    audit = screening_df.copy()
    audit["gate_passed"] = [passed for passed, _, _ in gate_results]
    audit["threshold_applied"] = [threshold for _, threshold, _ in gate_results]
    audit["reason"] = [reason for _, _, reason in gate_results]

    eligible_mask = audit["gate_passed"]
    audit["ssd_rank"] = np.nan
    audit.loc[eligible_mask, "ssd_rank"] = audit.loc[eligible_mask, "ssd_normalised_spread"].rank(
        ascending=True, method="average"
    )

    n_eligible = int(eligible_mask.sum())
    if n_eligible < n_select:
        logger.warning(
            "Only %d of %d requested pairs cleared the cointegration gate (source=%s); "
            "returning all %d eligible pairs rather than padding with ineligible ones.",
            n_eligible,
            n_select,
            source_label,
            n_eligible,
        )

    audit = audit.sort_values(
        by=["gate_passed", "ssd_rank"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    selected_labels = audit.loc[audit["gate_passed"]].head(n_select)["pair"].tolist()
    audit["selected"] = audit["pair"].isin(selected_labels)

    selected_pairs = (
        audit[audit["selected"]].drop(columns="selected").sort_values("ssd_rank").reset_index(drop=True)
    )

    return {"selected_pairs": selected_pairs, "selection_audit": audit}
