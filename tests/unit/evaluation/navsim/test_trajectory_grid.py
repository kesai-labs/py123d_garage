"""The sampling grid PDM scores trajectories and proposals on."""

from __future__ import annotations

import pytest
from jaxtyping import TypeCheckError

from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid


def test_sampling_deduces_the_missing_quantity() -> None:
    from_interval = TrajectoryGrid.from_num_poses_and_interval(num_poses=8, interval_us=250_000)
    assert from_interval.horizon_us == 2_000_000

    from_horizon = TrajectoryGrid.from_horizon_and_interval(horizon_us=3_000_000, interval_us=500_000)
    assert from_horizon.num_poses == 6
    assert from_horizon.interval_s == 0.5


def test_sampling_accepts_consistent_triple() -> None:
    trajectory_grid = TrajectoryGrid(
        num_poses=4,
        horizon_us=2_000_000,
        interval_us=500_000,
    )
    assert trajectory_grid.num_poses == 4
    assert trajectory_grid.horizon_us == 2_000_000
    assert trajectory_grid.interval_us == 500_000


def test_sampling_rejects_invalid_configurations() -> None:
    with pytest.raises(ValueError, match="is not num_poses"):
        TrajectoryGrid(
            num_poses=4,
            horizon_us=2_000_000,
            interval_us=300_000,
        )
    with pytest.raises(ValueError, match="multiple"):
        TrajectoryGrid.from_horizon_and_interval(horizon_us=1_000_000, interval_us=300_000)
    # Rejected by the runtime type-checking hook, or by __post_init__ without it.
    with pytest.raises((TypeCheckError, ValueError)):
        TrajectoryGrid.from_num_poses_and_interval(num_poses=4, interval_us=0.5)  # type: ignore[arg-type]
    with pytest.raises((TypeCheckError, ValueError)):
        TrajectoryGrid.from_num_poses_and_interval(num_poses=0, interval_us=500_000)


def test_sampling_equality_and_hash() -> None:
    left = TrajectoryGrid.from_horizon_and_interval(horizon_us=5_000_000, interval_us=500_000)
    right = TrajectoryGrid.from_num_poses_and_interval(num_poses=10, interval_us=500_000)
    assert left == right
    assert hash(left) == hash(right)
    assert left != TrajectoryGrid.from_num_poses_and_interval(num_poses=8, interval_us=500_000)


def test_grid_interval_in_seconds() -> None:
    grid = TrajectoryGrid.from_num_poses_and_interval(num_poses=4, interval_us=250_000)
    assert grid.interval_s == pytest.approx(0.25)
