"""Temporal sampling: past offset grids, ego SE2 interpolation and ego trajectory sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from py123d.api import SceneAPI
from py123d.datatypes import LidarID, Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    EgoStateSE3Metadata,
)
from py123d.geometry import PoseSE2
from py123d.geometry.pose import PoseSE3

from py123d_garage.py123d_help.scene_readers import (
    accumulate_lidar_in_anchor_frame,
    interpolate_ego_se2_at,
    past_offsets_us,
    sample_ego_se2,
    sample_ego_xy,
    sample_lidar_stack,
)

_EGO_METADATA = EgoStateSE3Metadata(
    vehicle_name="test_vehicle",
    width=2.0,
    length=4.5,
    height=1.6,
    wheel_base=2.8,
    center_to_imu_se3=PoseSE3.identity(),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)


def _ego_state(timestamp_us: int, x: float) -> EgoStateSE3:
    return EgoStateSE3.from_imu(
        imu_se3=PoseSE3.from_list([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        metadata=_EGO_METADATA,
        timestamp=Timestamp.from_us(timestamp_us),
    )


class _EgoScene:
    """SceneAPI stand-in serving ego states driving +x at 1 m per 100 ms."""

    def __init__(self, timestamps_us: list[int]) -> None:
        self._timestamps_us = timestamps_us

    @property
    def scene_uuid(self) -> str:
        return "test-scene"

    @property
    def log_name(self) -> str:
        return "test-log"

    def get_modality_between_timestamps(
        self,
        start_timestamp: int,
        end_timestamp: int,
        modality_type: str,
        inclusive: str = "left",
    ) -> Iterator[EgoStateSE3]:
        for timestamp_us in self._timestamps_us:
            if start_timestamp <= timestamp_us <= end_timestamp:
                yield _ego_state(timestamp_us, x=timestamp_us / 100_000)


SceneAPI.register(_EgoScene)


def _as_scene(stub: object) -> SceneAPI:
    """The stub is a virtual SceneAPI subclass (register); cast for the type checker."""
    return cast(SceneAPI, stub)


def test_past_offsets_cover_the_horizon_nearest_first() -> None:
    assert past_offsets_us(400_000, 100_000) == [-100_000, -200_000, -300_000, -400_000]
    assert past_offsets_us(0, 100_000) == []
    assert past_offsets_us(0, None) == []


def test_past_offsets_reject_bad_grids() -> None:
    with pytest.raises(ValueError, match="needs an interval"):
        past_offsets_us(400_000, None)
    with pytest.raises(ValueError, match="not a multiple"):
        past_offsets_us(250_000, 100_000)


def test_interpolation_hits_recorded_states_and_midpoints() -> None:
    scene = _EgoScene([1_000_000, 1_100_000, 1_200_000])
    poses = interpolate_ego_se2_at(
        _as_scene(scene),
        np.array([1_000_000, 1_050_000, 1_200_000], dtype=np.int64),
    )
    assert np.allclose(poses[:, 0], [10.0, 10.5, 12.0])


def test_interpolation_accepts_unordered_timestamps() -> None:
    scene = _EgoScene([1_000_000, 1_100_000, 1_200_000])
    poses = interpolate_ego_se2_at(
        _as_scene(scene),
        np.array([1_200_000, 1_000_000], dtype=np.int64),
    )
    assert np.allclose(poses[:, 0], [12.0, 10.0])


def test_interpolation_loads_states_from_the_margin() -> None:
    """Targets at the window edge interpolate from states outside [min, max] targets."""
    scene = _EgoScene([950_000, 1_050_000])
    poses = interpolate_ego_se2_at(
        _as_scene(scene),
        np.array([1_000_000], dtype=np.int64),
        interpolation_margin_us=100_000,
    )
    assert np.allclose(poses[:, 0], [10.0])


def test_boundary_tolerance_clamps_asynchronous_stream_edges() -> None:
    """A sensor timestamp slightly before the first ego state clamps to the boundary pose."""
    scene = _EgoScene([1_000_000, 1_100_000])
    poses = interpolate_ego_se2_at(
        _as_scene(scene),
        np.array([999_756, 1_100_200], dtype=np.int64),
        boundary_tolerance_us=1_000,
    )
    assert np.allclose(poses[:, 0], [10.0, 11.0])


def test_beyond_boundary_tolerance_raises() -> None:
    scene = _EgoScene([1_000_000, 1_100_000])
    with pytest.raises(ValueError, match="beyond the"):
        interpolate_ego_se2_at(
            _as_scene(scene),
            np.array([995_000, 1_100_000], dtype=np.int64),
            boundary_tolerance_us=1_000,
        )


def test_interpolation_needs_two_states() -> None:
    scene = _EgoScene([1_000_000])
    with pytest.raises(ValueError, match="needs at least two"):
        interpolate_ego_se2_at(_as_scene(scene), np.array([1_000_000], dtype=np.int64))


_EPOCH_US = 1_700_000_000_000_000


def _scene_api_with_log(
    pose_se2_array: npt.NDArray[np.float64],
    timestamps_us: npt.NDArray[np.int64],
) -> SceneAPI:
    """
    Builds a SceneAPI double serving the given logged rear-axle poses.

    Args:
        pose_se2_array: logged (x, y, yaw) rear-axle poses.
        timestamps_us: absolute timestamps in microseconds, one per pose.

    Returns:
        a mock passing the SceneAPI isinstance check.
    """
    states: list[EgoStateSE3] = []
    for pose, timestamp in zip(pose_se2_array, timestamps_us, strict=True):
        state = MagicMock(spec=EgoStateSE3)
        state.rear_axle_se2 = PoseSE2(
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
        )
        state.timestamp.time_us = int(timestamp)
        states.append(state)
    scene_api = MagicMock(spec=SceneAPI)
    scene_api.get_ego_state_se3_at_iteration.return_value = states[0]
    scene_api.get_modality_between_timestamps.return_value = states
    return scene_api


def test_sample_ego_trajectory_absolute_straight_line() -> None:
    """Resampling a 10 m/s straight log at 0.5 s steps lands at 5 m increments."""
    log_times_s = np.arange(0.0, 2.71, 0.3)
    poses = np.zeros((len(log_times_s), 3), dtype=np.float64)
    poses[:, 0] = 10.0 * log_times_s
    timestamps = _EPOCH_US + (log_times_s * 1e6).astype(np.int64)
    scene_api = _scene_api_with_log(poses, timestamps)

    result = sample_ego_se2(scene_api, num_steps=4, interval_us=500_000)

    expected_times = _EPOCH_US + np.array(
        [500_000, 1_000_000, 1_500_000, 2_000_000],
        dtype=np.int64,
    )
    assert np.array_equal(result.timestamps_us, expected_times)
    assert np.all(np.diff(result.timestamps_us) > 0)
    expected = np.array(
        [[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    assert np.allclose(result.pose_se2_array, expected, atol=1e-6)


def test_sample_ego_trajectory_relative_frame() -> None:
    """A log driving +y from a rotated origin becomes straight-ahead in the ego frame."""
    log_times_s = np.arange(0.0, 2.71, 0.3)
    poses = np.zeros((len(log_times_s), 3), dtype=np.float64)
    poses[:, 0] = 10.0
    poses[:, 1] = -5.0 + 4.0 * log_times_s
    poses[:, 2] = math.pi / 2.0
    timestamps = _EPOCH_US + (log_times_s * 1e6).astype(np.int64)
    scene_api = _scene_api_with_log(poses, timestamps)

    result = sample_ego_se2(scene_api, num_steps=4, interval_us=500_000, relative_to_anchor=True)

    expected = np.zeros((4, 3), dtype=np.float64)
    expected[:, 0] = 4.0 * np.array([0.5, 1.0, 1.5, 2.0])
    assert np.allclose(result.pose_se2_array, expected, atol=1e-6)


def test_sample_ego_xy_matches_the_se2_positions() -> None:
    """The xy resampler is the se2 one without the yaw column."""
    log_times_s = np.arange(0.0, 2.71, 0.3)
    poses = np.zeros((len(log_times_s), 3), dtype=np.float64)
    poses[:, 0] = 10.0 * log_times_s
    timestamps = _EPOCH_US + (log_times_s * 1e6).astype(np.int64)
    scene_api = _scene_api_with_log(poses, timestamps)

    se2 = sample_ego_se2(scene_api, num_steps=4, interval_us=500_000)
    xy = sample_ego_xy(scene_api, num_steps=4, interval_us=500_000)

    assert np.allclose(xy.position_xy_array, se2.pose_se2_array[:, :2], atol=1e-12)
    assert np.array_equal(xy.timestamps_us, se2.timestamps_us)


def _ego_state_at_pose(timestamp_us: int, pose: PoseSE2) -> EgoStateSE3:
    imu_se3 = PoseSE3.from_list(
        [
            pose.x,
            pose.y,
            0.0,
            float(np.cos(pose.yaw / 2.0)),
            0.0,
            0.0,
            float(np.sin(pose.yaw / 2.0)),
        ],
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        metadata=_EGO_METADATA,
        timestamp=Timestamp.from_us(timestamp_us),
    )


class _LidarStreamScene:
    """SceneAPI stand-in serving timestamped point clouds and the ego poses to place them."""

    def __init__(
        self,
        anchor_timestamp_us: int,
        sweeps: dict[int, npt.NDArray[np.float32]],
        poses: dict[int, PoseSE2] | None = None,
    ) -> None:
        self._anchor_timestamp_us = anchor_timestamp_us
        self._sweeps = sweeps
        self._poses = poses or {}

    def get_modality_between_timestamps(
        self,
        start_timestamp: int,
        end_timestamp: int,
        modality_type: str,
        inclusive: str = "left",
    ) -> Iterator[EgoStateSE3]:
        for timestamp_us in sorted(self._poses):
            if start_timestamp <= timestamp_us <= end_timestamp:
                yield _ego_state_at_pose(timestamp_us, self._poses[timestamp_us])

    @property
    def scene_uuid(self) -> str:
        return "test-scene"

    @property
    def log_name(self) -> str:
        return "test-log"

    def get_lidar_at_iteration(self, iteration: int, lidar_id: LidarID) -> SimpleNamespace | None:
        return self.get_lidar_at_timestamp(self._anchor_timestamp_us, lidar_id, criteria="nearest")

    def get_lidar_at_timestamp(
        self,
        timestamp: int,
        lidar_id: LidarID,
        criteria: str = "exact",
    ) -> SimpleNamespace | None:
        candidates_us = (
            [sweep_us for sweep_us in self._sweeps if sweep_us <= timestamp]
            if criteria == "backward"
            else list(self._sweeps)
        )
        if not candidates_us:
            return None
        matched_us = min(candidates_us, key=lambda sweep_us: abs(sweep_us - timestamp))
        return SimpleNamespace(xyz=self._sweeps[matched_us], timestamp=Timestamp.from_us(matched_us))


SceneAPI.register(_LidarStreamScene)

_SWEEP_INTERVAL_US = 100_000
_ANCHOR_SWEEP_US = 1_000_000
_ORIGIN_SE2 = PoseSE2(0.0, 0.0, 0.0)


def _sweep(marker: float) -> npt.NDArray[np.float32]:
    return np.array([[marker, 0.0, 0.0]], dtype=np.float32)


def test_lidar_stack_puts_the_anchor_first_then_the_requested_order() -> None:
    scene = _LidarStreamScene(
        _ANCHOR_SWEEP_US,
        {_ANCHOR_SWEEP_US - offset * _SWEEP_INTERVAL_US: _sweep(float(offset)) for offset in range(3)},
    )

    stack = sample_lidar_stack(
        cast(SceneAPI, scene),
        2 * _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )

    assert [sweep.xyz[0, 0] for sweep in stack] == [0.0, 1.0, 2.0]
    assert [sweep.timestamp.time_us for sweep in stack] == [
        _ANCHOR_SWEEP_US,
        _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US,
        _ANCHOR_SWEEP_US - 2 * _SWEEP_INTERVAL_US,
    ]


def test_lidar_stack_thins_where_the_stream_had_not_started() -> None:
    scene = _LidarStreamScene(_ANCHOR_SWEEP_US, {_ANCHOR_SWEEP_US: _sweep(0.0)})

    stack = sample_lidar_stack(
        cast(SceneAPI, scene),
        2 * _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )

    assert len(stack) == 1


def test_lidar_stack_rejects_a_scene_without_an_anchor_sweep() -> None:
    scene = _LidarStreamScene(_ANCHOR_SWEEP_US, {})

    with pytest.raises(ValueError, match="no LiDAR at its anchor frame"):
        sample_lidar_stack(cast(SceneAPI, scene), 0, None)


def test_sweep_accumulation_translates_past_points() -> None:
    """A point 5 m ahead of a past ego that sat 1 m behind the anchor is 4 m ahead now."""
    scene = _LidarStreamScene(
        _ANCHOR_SWEEP_US,
        {
            _ANCHOR_SWEEP_US: np.array([[1.0, 0.0, 0.5]], dtype=np.float32),
            _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US: np.array([[5.0, 0.0, 0.5]], dtype=np.float32),
        },
        {_ANCHOR_SWEEP_US: _ORIGIN_SE2, _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US: PoseSE2(-1.0, 0.0, 0.0)},
    )
    accumulated = accumulate_lidar_in_anchor_frame(
        _as_scene(scene),
        _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )
    assert np.allclose(
        accumulated,
        [[1.0, 0.0, 0.5], [4.0, 0.0, 0.5]],
        atol=1e-6,
    )


def test_sweep_accumulation_rotates_past_points() -> None:
    """A point ahead of a past ego facing +y lands at +y of an anchor facing +x."""
    scene = _LidarStreamScene(
        _ANCHOR_SWEEP_US,
        {
            _ANCHOR_SWEEP_US: np.array([[1.0, 0.0, 0.5]], dtype=np.float32),
            _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US: np.array([[1.0, 0.0, 0.2]], dtype=np.float32),
        },
        {_ANCHOR_SWEEP_US: _ORIGIN_SE2, _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US: PoseSE2(0.0, 0.0, np.pi / 2.0)},
    )
    accumulated = accumulate_lidar_in_anchor_frame(
        _as_scene(scene),
        _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )
    assert np.allclose(
        accumulated,
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.2]],
        atol=1e-6,
    )


def test_sweep_accumulation_uses_the_recorded_sweep_timestamp() -> None:
    """A sweep 20 ms off its target is compensated with the ego pose at its recorded time."""
    late_sweep_us = _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US + 20_000
    scene = _LidarStreamScene(
        _ANCHOR_SWEEP_US,
        {
            _ANCHOR_SWEEP_US: np.array([[1.0, 0.0, 0.5]], dtype=np.float32),
            late_sweep_us: np.array([[5.0, 0.0, 0.5]], dtype=np.float32),
        },
        {
            _ANCHOR_SWEEP_US: _ORIGIN_SE2,
            late_sweep_us: PoseSE2(-1.0, 0.0, 0.0),
        },
    )
    accumulated = accumulate_lidar_in_anchor_frame(
        _as_scene(scene),
        _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )
    assert np.allclose(
        accumulated,
        [[1.0, 0.0, 0.5], [4.0, 0.0, 0.5]],
        atol=1e-6,
    )


def test_sweep_accumulation_thins_at_the_stream_start() -> None:
    """Targets before the first recorded sweep thin the stack instead of failing."""
    scene = _LidarStreamScene(
        _ANCHOR_SWEEP_US,
        {
            _ANCHOR_SWEEP_US: np.array([[1.0, 0.0, 0.5]], dtype=np.float32),
            _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US: np.array([[5.0, 0.0, 0.5]], dtype=np.float32),
        },
        {_ANCHOR_SWEEP_US: _ORIGIN_SE2, _ANCHOR_SWEEP_US - _SWEEP_INTERVAL_US: PoseSE2(-1.0, 0.0, 0.0)},
    )
    accumulated = accumulate_lidar_in_anchor_frame(
        _as_scene(scene),
        3 * _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )
    assert np.allclose(
        accumulated,
        [[1.0, 0.0, 0.5], [4.0, 0.0, 0.5]],
        atol=1e-6,
    )


def test_sweep_accumulation_thins_to_the_anchor_alone() -> None:
    anchor = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    scene = _LidarStreamScene(_ANCHOR_SWEEP_US, {_ANCHOR_SWEEP_US: anchor}, {_ANCHOR_SWEEP_US: _ORIGIN_SE2})
    accumulated = accumulate_lidar_in_anchor_frame(
        _as_scene(scene),
        2 * _SWEEP_INTERVAL_US,
        _SWEEP_INTERVAL_US,
    )
    assert np.allclose(accumulated, anchor, atol=1e-6)


def test_sweep_accumulation_rejects_dropout_inside_coverage() -> None:
    """A missing sweep with older sweeps recorded is a dropout, not a stream start."""
    anchor = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    past = np.array([[0.0, 1.0, 0.2]], dtype=np.float32)
    dropout_scene = _LidarStreamScene(
        _ANCHOR_SWEEP_US,
        {
            _ANCHOR_SWEEP_US: anchor,
            _ANCHOR_SWEEP_US - 2 * _SWEEP_INTERVAL_US: past,
        },
        {_ANCHOR_SWEEP_US: _ORIGIN_SE2, _ANCHOR_SWEEP_US - 2 * _SWEEP_INTERVAL_US: _ORIGIN_SE2},
    )
    with pytest.raises(ValueError, match="already recording"):
        accumulate_lidar_in_anchor_frame(
            _as_scene(dropout_scene),
            2 * _SWEEP_INTERVAL_US,
            _SWEEP_INTERVAL_US,
        )
