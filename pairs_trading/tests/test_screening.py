"""Unit tests for screening.py's gate-then-rank pair selection logic.

These tests drive `rank_and_select_pairs` from a small, hand-built
screening-results table rather than a raw price panel. `rank_and_select_pairs`
operates entirely on already-computed metrics (pair label, Engle-Granger
p-value, SSD), and the behaviour under test — which pairs clear the
cointegration gate, and how eligible pairs are ranked — is fully determined
by those metrics. Reverse-engineering a synthetic price series that produces
an exact target p-value (e.g. 0.03 or 0.08) is not analytically tractable
and would make the test flaky; testing at the `screening_df` boundary is
both the correct unit and a deterministic one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pairs_trading import config, screening


def _make_screening_df(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Build a minimal synthetic screening-results table for selection tests.

    Args:
        rows: One dict per pair, each with keys "pair", "eg_p_value", and
            "ssd_normalised_spread". "correlation" defaults to 0.9 if not
            supplied, since it is not exercised by the gate/rank logic
            under test.

    Returns:
        DataFrame with one row per dict, matching the subset of
        `screen_all_candidate_pairs`'s output columns that
        `rank_and_select_pairs` actually reads.
    """
    return pd.DataFrame([{"correlation": 0.9, **row} for row in rows])


@pytest.fixture(autouse=True)
def _pin_screening_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the selection-relevant config values so tests don't depend on config.py's current settings."""
    monkeypatch.setattr(config, "USE_COVID_EXCLUDED_SCREENING", True)
    monkeypatch.setattr(config, "PRIMARY_COINTEGRATION_THRESHOLD", 0.05)
    monkeypatch.setattr(config, "RELAXED_COINTEGRATION_THRESHOLD", 0.10)
    monkeypatch.setattr(config, "PAIRS_REQUIRING_RELAXED_THRESHOLD", ["C/D"])


def test_primary_threshold_pair_beats_non_relaxed_marginal_pair() -> None:
    """A pair clearing the primary 5% threshold (p=0.03) should be selected;
    a pair at p=0.08 that is NOT on the relaxed-threshold exception list
    should be gated out even though it has a tighter (lower) SSD."""
    screening_df = _make_screening_df(
        [
            {"pair": "A/B", "eg_p_value": 0.03, "ssd_normalised_spread": 20.0},
            {"pair": "E/F", "eg_p_value": 0.08, "ssd_normalised_spread": 5.0},
        ]
    )

    result = screening.rank_and_select_pairs(screening_df, n_select=3, source_label="ex_covid")

    assert result["selected_pairs"]["pair"].tolist() == ["A/B"]

    audit = result["selection_audit"].set_index("pair")
    assert bool(audit.loc["A/B", "gate_passed"])
    assert not bool(audit.loc["E/F", "gate_passed"])
    assert "primary" in audit.loc["E/F", "threshold_applied"]


def test_relaxed_threshold_pair_included_when_short_of_quota() -> None:
    """When fewer pairs clear the primary 5% threshold than n_select, a pair
    on the relaxed-threshold exception list at p=0.08 (between the primary
    and relaxed thresholds) should still be selected — but a non-exception
    pair that fails even the relaxed threshold must not be used to pad the
    count."""
    screening_df = _make_screening_df(
        [
            {"pair": "A/B", "eg_p_value": 0.03, "ssd_normalised_spread": 20.0},
            {"pair": "C/D", "eg_p_value": 0.08, "ssd_normalised_spread": 5.0},  # relaxed exception
            {"pair": "E/F", "eg_p_value": 0.30, "ssd_normalised_spread": 1.0},  # fails even relaxed
        ]
    )

    result = screening.rank_and_select_pairs(screening_df, n_select=3, source_label="ex_covid")

    assert set(result["selected_pairs"]["pair"]) == {"A/B", "C/D"}

    audit = result["selection_audit"].set_index("pair")
    assert bool(audit.loc["C/D", "gate_passed"])
    assert "relaxed" in audit.loc["C/D", "threshold_applied"]
    assert not bool(audit.loc["E/F", "gate_passed"])
    assert not bool(audit.loc["E/F", "selected"])


def test_source_label_must_match_config_flag() -> None:
    """Calling with a source_label that doesn't match
    config.USE_COVID_EXCLUDED_SCREENING must raise, so the function can
    never be silently pointed at the wrong screening table."""
    screening_df = _make_screening_df([{"pair": "A/B", "eg_p_value": 0.03, "ssd_normalised_spread": 20.0}])

    with pytest.raises(ValueError):
        screening.rank_and_select_pairs(screening_df, n_select=1, source_label="full_period")
