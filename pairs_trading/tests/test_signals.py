"""Unit tests for signals.py's position-generation logic."""

from __future__ import annotations

import pandas as pd

from pairs_trading import signals


def test_generate_positions_short_then_mean_reversion_exit() -> None:
    """A z-score series that rises above +entry_threshold then reverts back
    through 0 should produce exactly one short trade that closes with
    exit_reason='mean_reversion', and no other position opens or closes."""
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    zscore = pd.Series([0.5, 1.0, 2.5, 2.0, 1.0, 0.5, 0.0, 0.1, -0.2, 0.0], index=dates)

    result = signals.generate_positions(
        zscore, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=None, stop_loss_threshold=None
    )

    assert result["position"].tolist() == [0, 0, -1, -1, -1, -1, 0, 0, 0, 0]

    exits = result[result["exit_reason"].notna()]
    assert len(exits) == 1
    assert exits.iloc[0]["exit_reason"] == "mean_reversion"
    assert exits.index[0] == dates[6]


def test_generate_positions_force_closes_at_max_holding_days() -> None:
    """A position that never reverts back through the exit threshold must be
    force-closed exactly when days_held reaches max_holding_days, with
    exit_reason='max_holding'."""
    dates = pd.date_range("2022-01-01", periods=8, freq="B")
    # Enters short on day 1 (z=2.5) and stays far above the exit band
    # through day 4 (forced close), then drops back below entry_threshold
    # so no new position re-opens afterward -- isolates the force-close
    # behaviour from the re-entry policy (covered separately below).
    zscore = pd.Series([0.5, 2.5, 3.0, 3.0, 3.0, 1.0, 0.5, 0.3], index=dates)

    result = signals.generate_positions(
        zscore, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=4, stop_loss_threshold=None
    )

    assert result["position"].tolist() == [0, -1, -1, -1, 0, 0, 0, 0]

    exits = result[result["exit_reason"].notna()]
    assert len(exits) == 1
    assert exits.iloc[0]["exit_reason"] == "max_holding"
    assert exits.index[0] == dates[4]


def test_generate_positions_allows_immediate_reentry_after_forced_close() -> None:
    """Policy decision (confirmed 2026-07-20): entry is re-evaluated fresh
    every day the position is flat, with no memory of a prior forced
    close. If z is still beyond entry_threshold the day immediately after
    a max-holding or stop-loss close, a new position opens right away --
    this is not a bug, it's the deliberate no-cooldown convention."""
    dates = pd.date_range("2022-01-01", periods=8, freq="B")
    # Stays at z=3.0 straight through the forced close at day 4 and beyond,
    # so day 5 should re-enter immediately rather than waiting for z to
    # first return inside the entry band.
    zscore = pd.Series([0.5, 2.5, 3.0, 3.0, 3.0, 3.0, 1.0, 0.5], index=dates)

    result = signals.generate_positions(
        zscore, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=4, stop_loss_threshold=None
    )

    assert result["position"].tolist() == [0, -1, -1, -1, 0, -1, -1, -1]
    assert result.loc[dates[5], "position"] == -1  # re-entered the very next day

    exits = result[result["exit_reason"].notna()]
    assert len(exits) == 1  # only the forced close counts as an "exit"; the re-entry is not one
    assert exits.iloc[0]["exit_reason"] == "max_holding"
    assert exits.index[0] == dates[4]


def test_generate_positions_long_entry_and_stop_loss_priority() -> None:
    """A long position (entered on z <= -entry_threshold) that blows through
    the stop-loss level must close with exit_reason='stop_loss'."""
    dates = pd.date_range("2022-01-01", periods=6, freq="B")
    zscore = pd.Series([0.0, -2.5, -3.0, -4.5, 0.0, 0.0], index=dates)

    result = signals.generate_positions(
        zscore, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=10, stop_loss_threshold=4.0
    )

    assert result["position"].tolist() == [0, 1, 1, 0, 0, 0]

    exits = result[result["exit_reason"].notna()]
    assert len(exits) == 1
    assert exits.iloc[0]["exit_reason"] == "stop_loss"
    assert exits.index[0] == dates[3]


def test_generate_positions_stop_loss_triggers_cooldown_before_reentry() -> None:
    """Unlike a max-holding close, a stop-loss exit must block re-entry
    even if z remains beyond entry_threshold: the pair must first see z
    return inside the entry band (a distinct 'cooling_down' state) before
    a new position can open again."""
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    # Enter short (day1, z=2.5), stop out on day2 (z=5.0 >= stop_loss=4.0),
    # then stay extreme (4.5, 4.2) -- must NOT re-enter -- before finally
    # dropping inside the entry band (z=1.5) on day5, after which entry is
    # eligible again and fires on day6 (z=2.5).
    zscore = pd.Series([0.5, 2.5, 5.0, 4.5, 4.2, 1.5, 2.5, 2.6, 0.0, 0.0], index=dates)

    result = signals.generate_positions(
        zscore, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=None, stop_loss_threshold=4.0
    )

    assert result["position"].tolist() == [0, -1, 0, 0, 0, 0, -1, -1, 0, 0]
    assert result["state"].tolist() == [
        "flat",
        "short",
        "cooling_down",
        "cooling_down",
        "cooling_down",
        "flat",
        "short",
        "short",
        "flat",
        "flat",
    ]

    stop_loss_exits = result[result["exit_reason"] == "stop_loss"]
    assert len(stop_loss_exits) == 1
    assert stop_loss_exits.index[0] == dates[2]

    # Still-extreme z on days 3-4 must not re-open a position while cooling down.
    assert result.loc[dates[3], "position"] == 0
    assert result.loc[dates[4], "position"] == 0
    # Re-entry only happens once z is back inside the band and rises past
    # entry_threshold again (day6), not on the day cooldown itself clears (day5).
    assert result.loc[dates[5], "position"] == 0
    assert result.loc[dates[6], "position"] == -1


def test_generate_positions_has_no_lookahead() -> None:
    """Positions and exit reasons up to day t must be identical regardless
    of what the z-score does strictly after day t: two series sharing an
    identical prefix but diverging afterward must produce identical
    position/exit_reason values over that shared prefix."""
    dates = pd.date_range("2022-01-01", periods=12, freq="B")
    shared_prefix = [0.5, 2.5, 2.6, 2.4, 1.0, 0.3, -0.1, 0.2]

    zscore_a = pd.Series(shared_prefix + [-3.0, -3.0, -3.0, -3.0], index=dates)
    zscore_b = pd.Series(shared_prefix + [10.0, -10.0, 5.0, -5.0], index=dates)

    result_a = signals.generate_positions(
        zscore_a, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=None, stop_loss_threshold=4.0
    )
    result_b = signals.generate_positions(
        zscore_b, entry_threshold=2.0, exit_threshold=0.0, max_holding_days=None, stop_loss_threshold=4.0
    )

    n_shared = len(shared_prefix)
    pd.testing.assert_series_equal(
        result_a["position"].iloc[:n_shared], result_b["position"].iloc[:n_shared], check_names=False
    )
    pd.testing.assert_series_equal(
        result_a["exit_reason"].iloc[:n_shared], result_b["exit_reason"].iloc[:n_shared], check_names=False
    )
