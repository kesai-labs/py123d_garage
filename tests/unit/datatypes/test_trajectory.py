"""Unit tests for the trajectory containers, interpolation and the yaw derivation."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
import torch
from py123d.geometry.geometry_index import PoseSE2Index

from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
    TrajectoryXY,
    derive_yaw_from_positions,
)

_EPOCH_US = 1_700_000_000_000_000  # realistic unix-microsecond origin


def test_interpolate_straight_constant_velocity() -> None:
    """Midway between two poses of a constant-velocity segment lies the exact midpoint pose."""
    yaw = math.pi / 4.0
    poses = np.array([[0.0, 0.0, yaw], [4.0, 4.0, yaw]], dtype=np.float64)
    timestamps = np.array([_EPOCH_US, _EPOCH_US + 1_000_000], dtype=np.int64)
    trajectory = TrajectorySE2(poses, timestamps)

    pose = trajectory.interpolate(_EPOCH_US + 250_000)
    assert np.allclose(pose, [1.0, 1.0, yaw], atol=1e-9)


def test_interpolate_at_exact_knot_returns_knot_pose() -> None:
    poses = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.3], [5.0, -1.0, -0.2]],
        dtype=np.float64,
    )
    timestamps = _EPOCH_US + np.array(
        [0, 300_000, 900_000],
        dtype=np.int64,
    )
    trajectory = TrajectorySE2(poses.copy(), timestamps)

    assert np.allclose(
        trajectory.interpolate(timestamps[1]),
        poses[1],
        atol=1e-9,
    )


def test_interpolate_vector_query() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
    timestamps = np.array([_EPOCH_US, _EPOCH_US + 1_000_000], dtype=np.int64)
    trajectory = TrajectorySE2(poses, timestamps)

    query = _EPOCH_US + np.array([250_000, 750_000], dtype=np.int64)
    result = trajectory.interpolate(query)
    assert result.shape == (2, 3)
    assert np.allclose(result[:, 0], [2.5, 7.5], atol=1e-9)


def test_interpolate_rejects_out_of_range_timestamps() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.1]], dtype=np.float64)
    timestamps = np.array([_EPOCH_US, _EPOCH_US + 1_000_000], dtype=np.int64)
    trajectory = TrajectorySE2(poses.copy(), timestamps)

    with pytest.raises(ValueError, match="beyond the"):
        trajectory.interpolate(_EPOCH_US + 2_000_000)
    with pytest.raises(ValueError, match="beyond the"):
        trajectory.interpolate(_EPOCH_US - 1)


def test_interpolate_yaw_takes_the_short_way_across_pi() -> None:
    """From yaw 2.9 to -2.9 the short sweep passes through π, not through 0."""
    poses = np.array([[0.0, 0.0, 2.9], [1.0, 0.0, -2.9]], dtype=np.float64)
    timestamps = np.array([_EPOCH_US, _EPOCH_US + 1_000_000], dtype=np.int64)
    trajectory = TrajectorySE2(poses, timestamps)

    at_quarter = trajectory.interpolate(_EPOCH_US + 250_000)
    expected_quarter = 2.9 + 0.25 * (2.0 * math.pi - 5.8)
    assert np.isclose(at_quarter[2], expected_quarter, atol=1e-9)

    at_half = trajectory.interpolate(_EPOCH_US + 500_000)
    assert np.isclose(abs(at_half[2]), math.pi, atol=1e-9)


@pytest.mark.xfail(
    reason="single-pose trajectory silently yields NaN from interp1d",
    strict=True,
)
def test_single_pose_trajectory_interpolates_to_its_pose() -> None:
    trajectory = TrajectorySE2(
        np.array([[1.0, 2.0, 0.5]], dtype=np.float64),
        np.array([_EPOCH_US], dtype=np.int64),
    )
    with np.errstate(invalid="ignore"):
        pose = trajectory.interpolate(_EPOCH_US)
    assert np.allclose(pose, [1.0, 2.0, 0.5], atol=1e-9)


def test_polyline_property_carries_the_poses() -> None:
    poses = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.3], [5.0, -1.0, -0.2]],
        dtype=np.float64,
    )
    timestamps = _EPOCH_US + np.array([0, 100_000, 200_000], dtype=np.int64)
    trajectory = TrajectorySE2(poses.copy(), timestamps)
    assert np.allclose(trajectory.polyline_se2.array, poses, atol=1e-9)


def _xy(positions: list[list[float]]) -> TrajectoryXY:
    """A TrajectoryXY over the given ego-frame positions, sampled every 0.5 s."""
    position_xy_array = np.array(positions, dtype=np.float64)
    timestamps_us = _EPOCH_US + np.arange(1, len(positions) + 1, dtype=np.int64) * 500_000
    return TrajectoryXY(position_xy_array, timestamps_us)


def test_derive_yaw_straight_ahead_is_zero() -> None:
    """A path straight down the ego x axis keeps the ego's current heading throughout."""
    trajectory = _xy([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]).to_se2()

    assert np.allclose(trajectory.pose_se2_array[:, PoseSE2Index.YAW], 0.0, atol=1e-12)


def test_derive_yaw_positions_are_untouched() -> None:
    """to_se2 only appends a column; the xy it was given survives unchanged."""
    positions = [[1.0, 0.5], [2.0, 1.5], [3.5, 1.5]]
    trajectory = _xy(positions).to_se2()

    assert np.allclose(trajectory.pose_se2_array[:, :2], np.array(positions), atol=1e-12)


def test_derive_yaw_first_pose_measures_from_the_ego_origin() -> None:
    """The first step is measured from (0, 0), the origin the trajectory is relative to."""
    trajectory = _xy([[1.0, 1.0], [2.0, 2.0]]).to_se2()

    assert math.isclose(
        float(trajectory.pose_se2_array[0, PoseSE2Index.YAW]),
        math.pi / 4.0,
        abs_tol=1e-12,
    )


def test_derive_yaw_follows_the_path_tangent() -> None:
    """Each pose takes the heading of the step that reached it, wrapped to -pi not +pi."""
    trajectory = _xy([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]).to_se2()

    assert np.allclose(
        trajectory.pose_se2_array[:, PoseSE2Index.YAW],
        [0.0, math.pi / 2.0, -math.pi],
        atol=1e-12,
    )


def test_derive_yaw_holds_last_trusted_heading_when_stopped() -> None:
    """A step below the displacement threshold keeps the heading of the last real step."""
    trajectory = _xy([[0.0, 1.0], [0.0, 2.0], [0.0, 2.0 + 1e-9]]).to_se2()

    assert np.allclose(
        trajectory.pose_se2_array[:, PoseSE2Index.YAW],
        math.pi / 2.0,
        atol=1e-12,
    )


def test_derive_yaw_standstill_from_the_start_holds_the_ego_heading() -> None:
    """With no trusted step yet, yaw is 0.0: in the ego frame, the current heading."""
    trajectory = _xy([[0.0, 0.0], [1e-9, 0.0], [0.0, 1e-9]]).to_se2()

    assert np.allclose(trajectory.pose_se2_array[:, PoseSE2Index.YAW], 0.0, atol=1e-12)


def test_derive_yaw_noise_at_standstill_does_not_reach_the_output() -> None:
    """Sub-threshold jitter must not be amplified into arbitrary headings by atan2."""
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=1e-6, size=(16, 2))
    positions = np.array([[3.0, 0.0]] * 16, dtype=np.float64) + jitter
    trajectory = _xy([[1.0, 0.0], [3.0, 0.0], *positions.tolist()]).to_se2()

    assert np.allclose(trajectory.pose_se2_array[:, PoseSE2Index.YAW], 0.0, atol=1e-12)


def test_derive_yaw_carries_the_timestamps_over() -> None:
    trajectory_xy = _xy([[1.0, 0.0], [2.0, 0.0]])
    trajectory_se2 = trajectory_xy.to_se2()

    assert np.array_equal(trajectory_se2.timestamps_us, trajectory_xy.timestamps_us)


def test_derive_yaw_result_interpolates_as_se2() -> None:
    """The lift produces a real TrajectorySE2, usable by everything downstream."""
    trajectory = _xy([[1.0, 0.0], [2.0, 0.0]]).to_se2()

    pose = trajectory.interpolate(_EPOCH_US + 750_000)
    assert np.allclose(pose, [1.5, 0.0, 0.0], atol=1e-9)


def test_se2_to_se2_returns_itself() -> None:
    """Both trajectory types answer to_se2(), so a benchmark can take either."""
    poses = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    timestamps = _EPOCH_US + np.array([500_000, 1_000_000], dtype=np.int64)
    trajectory = TrajectorySE2(poses, timestamps)

    assert trajectory.to_se2() is trajectory


def test_to_se2_honours_a_wider_displacement_threshold() -> None:
    """Raising the threshold makes a real but short step untrusted."""
    trajectory = _xy([[0.0, 1.0], [0.1, 1.0]])

    assert trajectory.to_se2().pose_se2_array[1, PoseSE2Index.YAW] == pytest.approx(0.0)
    lifted = trajectory.to_se2(min_displacement_m=1.0)
    assert lifted.pose_se2_array[1, PoseSE2Index.YAW] == pytest.approx(math.pi / 2.0)


def test_derive_yaw_batched_matches_each_row_alone() -> None:
    """The predictions properties call this batched; rows must not leak into each other."""
    rows = [
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[0.0, 1.0], [0.0, 2.0], [-1.0, 2.0]],
        [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
    ]
    batch = torch.tensor(rows, dtype=torch.float64)

    batched = derive_yaw_from_positions(batch)
    assert batched.shape == (len(rows), 3)
    for index, row in enumerate(rows):
        alone = derive_yaw_from_positions(torch.tensor(row, dtype=torch.float64))
        assert torch.allclose(batched[index], alone, atol=1e-12)


def test_derive_yaw_unbatched_keeps_its_shape() -> None:
    yaw = derive_yaw_from_positions(torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64))

    assert yaw.shape == (2,)
    assert torch.allclose(yaw, torch.tensor([0.0, math.pi / 2.0], dtype=torch.float64), atol=1e-12)


def test_derive_yaw_matches_the_trajectory_lift() -> None:
    """to_se2 delegates here, so the two must agree exactly."""
    positions = [[1.0, 0.5], [2.0, 1.5], [2.0, 1.5], [3.5, 1.5]]
    lifted = _xy(positions).to_se2().pose_se2_array[:, PoseSE2Index.YAW]
    direct = derive_yaw_from_positions(torch.tensor(positions, dtype=torch.float64)).numpy()

    assert np.allclose(lifted, direct, atol=1e-12)


def _turning_further_than_a_full_circle() -> TrajectoryXY:
    """A path whose heading sweeps past +pi, where wrapped and unwrapped disagree."""
    angles = np.linspace(0.0, 1.6 * np.pi, 12)
    positions = np.stack([np.cumsum(np.cos(angles)), np.cumsum(np.sin(angles))], axis=1)
    return TrajectoryXY(positions, _EPOCH_US + np.arange(1, 13, dtype=np.int64) * 500_000)


def _is_wrapped(yaw: npt.NDArray[np.float64]) -> bool:
    return bool(np.all((yaw >= -np.pi) & (yaw < np.pi)))


def test_stored_yaw_is_wrapped() -> None:
    trajectory = _turning_further_than_a_full_circle().to_se2()

    assert _is_wrapped(trajectory.pose_se2_array[:, PoseSE2Index.YAW])


def test_interpolated_yaw_is_wrapped() -> None:
    trajectory = _turning_further_than_a_full_circle().to_se2()
    dense = trajectory.interpolate(
        np.linspace(
            int(trajectory.timestamps_us[0]),
            int(trajectory.timestamps_us[-1]),
            97,
        ).astype(np.int64),
    )

    assert _is_wrapped(dense[:, PoseSE2Index.YAW])


def test_derived_yaw_is_wrapped() -> None:
    positions = _turning_further_than_a_full_circle().position_xy_array
    yaw = derive_yaw_from_positions(torch.from_numpy(positions)).numpy()

    assert _is_wrapped(yaw)


def test_derived_yaw_never_returns_positive_pi() -> None:
    """atan2 reaches +pi; the wrap must fold it onto -pi so one form is left."""
    straight_back = torch.tensor([[-1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64)
    yaw = derive_yaw_from_positions(straight_back)

    assert torch.allclose(yaw, torch.full_like(yaw, -math.pi))


def test_constructor_leaves_the_callers_array_alone() -> None:
    """Wrapping must not reach back into the array the caller still holds."""
    poses = np.array([[0.0, 0.0, 3.0 * math.pi]], dtype=np.float64)
    original = poses.copy()
    TrajectorySE2(poses, _EPOCH_US + np.array([500_000], dtype=np.int64))

    assert np.array_equal(poses, original)


def test_equal_headings_stored_differently_compare_equal() -> None:
    """The whole point of one stored form: 3.0 and 3.0 - 2*pi are the same rotation."""
    timestamps = _EPOCH_US + np.array([500_000], dtype=np.int64)
    left = TrajectorySE2(np.array([[0.0, 0.0, 3.0]]), timestamps)
    right = TrajectorySE2(np.array([[0.0, 0.0, 3.0 - 2.0 * math.pi]]), timestamps)

    assert np.allclose(left.pose_se2_array, right.pose_se2_array)


def test_polyline_se2_yaw_is_unwrapped_by_py123d() -> None:
    """The one carve-out: PolylineSE2 unwraps in its constructor, so it leaves the range."""
    trajectory = _turning_further_than_a_full_circle().to_se2()

    assert not _is_wrapped(trajectory.polyline_se2.array[:, PoseSE2Index.YAW])


def test_derived_yaw_on_a_constant_curvature_arc_has_constant_rate() -> None:
    """
    A steady turn must derive a steady yaw rate: the LQR tracks this heading, and any
    second-order wobble in it shows up as yaw acceleration in the comfort score.

    The positions follow the policy contract, i.e. the first one is a whole interval ahead
    of the ego rather than sitting on it.
    """
    interval_s, curvature, speed = 0.1, 0.03, 10.0
    heading_rad = curvature * speed * (np.arange(1, 41) * interval_s)
    positions = np.stack(
        [np.sin(heading_rad) / curvature, (1.0 - np.cos(heading_rad)) / curvature],
        axis=1,
    )

    derived = derive_yaw_from_positions(torch.from_numpy(positions)).numpy()
    yaw_rate = np.gradient(np.unwrap(derived), interval_s)
    yaw_acceleration = np.gradient(yaw_rate, interval_s)

    assert np.allclose(yaw_rate, curvature * speed, atol=1e-6)
    assert np.allclose(yaw_acceleration, 0.0, atol=1e-6)


def test_a_leading_ego_pose_corrupts_the_derived_yaw_rate() -> None:
    """
    Why the contract matters: a trajectory that opens on the ego pose has a zero-length
    first step, so the heading there is held rather than measured, and the discontinuity
    lands in the yaw acceleration.
    """
    interval_s, curvature, speed = 0.1, 0.03, 10.0
    heading_rad = curvature * speed * (np.arange(0, 41) * interval_s)
    positions = np.stack(
        [np.sin(heading_rad) / curvature, (1.0 - np.cos(heading_rad)) / curvature],
        axis=1,
    )

    derived = derive_yaw_from_positions(torch.from_numpy(positions)).numpy()
    yaw_acceleration = np.gradient(np.gradient(np.unwrap(derived), interval_s), interval_s)

    assert np.abs(yaw_acceleration).max() > 0.5
