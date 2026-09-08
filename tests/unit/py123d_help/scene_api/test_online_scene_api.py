"""OnlineSceneAPI: bundle merging, nearest-in-tolerance iteration reads, clamping, and snapshots."""

from __future__ import annotations

import numpy as np
import pytest
from py123d.datatypes import LogMetadata, ModalityType, Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    EgoStateSE3Metadata,
)
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D

from py123d_garage.py123d_help.scene_api.online_scene_api import OnlineSceneAPI

_INTERVAL_US = 100_000
_EGO_METADATA = EgoStateSE3Metadata(
    vehicle_name="test_vehicle",
    width=2.0,
    length=4.5,
    height=1.6,
    wheel_base=2.8,
    center_to_imu_se3=PoseSE3.identity(),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)


def _ego_state(timestamp_us: int, x: float = 0.0) -> EgoStateSE3:
    return EgoStateSE3.from_imu(
        imu_se3=PoseSE3.from_list([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        metadata=_EGO_METADATA,
        timestamp=Timestamp.from_us(timestamp_us),
        dynamic_state_se3=DynamicStateSE3(
            velocity=Vector3D(1.0, 0.0, 0.0),
            acceleration=Vector3D(0.0, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.0),
        ),
    )


def _scene(num_history_iterations: int = 4) -> OnlineSceneAPI:
    return OnlineSceneAPI(
        log_metadata=LogMetadata(dataset="test", split="", log_name="", location=None),
        modality_metadatas={},
        iteration_interval_us=_INTERVAL_US,
        num_history_iterations=num_history_iterations,
    )


def _ego_x_at_iteration(scene: OnlineSceneAPI, iteration: int) -> float | None:
    ego_state = scene.get_ego_state_se3_at_iteration(iteration)
    return None if ego_state is None else float(ego_state.rear_axle_se3.x)


def test_iterations_map_to_nearest_bundle_within_tolerance() -> None:
    scene = _scene()
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000, x=0.0)})
    scene.append(1_120_000, {"ego_state_se3": _ego_state(1_120_000, x=1.0)})
    scene.append(1_200_000, {"ego_state_se3": _ego_state(1_200_000, x=2.0)})
    snapshot = scene.snapshot()
    assert _ego_x_at_iteration(snapshot, 0) == 2.0
    assert _ego_x_at_iteration(snapshot, -1) == 1.0
    assert _ego_x_at_iteration(snapshot, -2) == 0.0


def test_iteration_misses_beyond_tolerance() -> None:
    scene = _scene()
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000)})
    scene.append(1_200_000, {"ego_state_se3": _ego_state(1_200_000)})
    snapshot = scene.snapshot()
    assert snapshot.get_ego_state_se3_at_iteration(-1) is None
    assert snapshot.get_ego_state_se3_at_iteration(-2) is not None


def test_append_merges_into_the_bundle_at_the_timestamp() -> None:
    scene = _scene()
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000)})
    scene.append(1_000_000, {"custom.extra": _ego_state(1_000_000, x=7.0)})
    snapshot = scene.snapshot()
    assert snapshot.get_ego_state_se3_at_iteration(0) is not None
    assert snapshot.get_modality_at_iteration(0, "custom", "extra") is not None


def test_snapshot_is_frozen_against_later_appends() -> None:
    scene = _scene()
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000, x=0.0)})
    snapshot = scene.snapshot()
    scene.append(1_100_000, {"ego_state_se3": _ego_state(1_100_000, x=1.0)})
    assert _ego_x_at_iteration(snapshot, 0) == 0.0
    assert _ego_x_at_iteration(scene.snapshot(), 0) == 1.0


def test_old_bundles_are_trimmed_by_time() -> None:
    scene = _scene(num_history_iterations=1)
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000)})
    scene.append(2_000_000, {"ego_state_se3": _ego_state(2_000_000)})
    timestamps = scene.snapshot().get_all_modality_timestamps(
        ModalityType.EGO_STATE_SE3,
        include_history=True,
    )
    assert [timestamp.time_us for timestamp in timestamps] == [2_000_000]


def test_between_timestamps_yields_time_ordered_modalities() -> None:
    scene = _scene()
    scene.append(1_200_000, {"ego_state_se3": _ego_state(1_200_000)})
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000)})
    scene.append(1_100_000, {"ego_state_se3": _ego_state(1_100_000)})
    states = list(
        scene.snapshot().get_modality_between_timestamps(
            1_000_000,
            1_200_000,
            "ego_state_se3",
            inclusive="both",
        ),
    )
    assert [state.timestamp.time_us for state in states] == [1_000_000, 1_100_000, 1_200_000]


def test_timestamp_criteria() -> None:
    scene = _scene()
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000, x=0.0)})
    scene.append(1_100_000, {"ego_state_se3": _ego_state(1_100_000, x=1.0)})
    snapshot = scene.snapshot()
    assert snapshot.get_ego_state_se3_at_timestamp(1_040_000, criteria="nearest") is not None
    assert snapshot.get_ego_state_se3_at_timestamp(1_040_000, criteria="exact") is None
    forward = snapshot.get_ego_state_se3_at_timestamp(1_040_000, criteria="forward")
    backward = snapshot.get_ego_state_se3_at_timestamp(1_040_000, criteria="backward")
    assert forward is not None
    assert forward.timestamp.time_us == 1_100_000
    assert backward is not None
    assert backward.timestamp.time_us == 1_000_000


def test_grid_timestamps_and_empty_scene() -> None:
    scene = _scene(num_history_iterations=2)
    with pytest.raises(AssertionError, match="empty scene"):
        scene.snapshot()
    assert scene.newest_timestamp_us is None
    scene.append(1_000_000, {"ego_state_se3": _ego_state(1_000_000)}, route_progress_m=3.0)
    snapshot = scene.snapshot()
    grid_us = [timestamp.time_us for timestamp in snapshot.get_all_iteration_timestamps(include_history=True)]
    assert grid_us == [800_000, 900_000, 1_000_000]
    assert snapshot.get_route_progress_at_iteration(0) == 3.0
    assert np.isclose(snapshot.get_scene_metadata().history_duration_s, 0.2)
