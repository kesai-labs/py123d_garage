"""Unit tests for target points read from the route polyline of a real written log."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.datatypes import Timestamp
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    EgoStateSE3Metadata,
)
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ModalitiesSync

from py123d_garage.py123d_help.scene_readers import (
    InsufficientRouteError,
    get_target_points,
)

_TIMESTEP_US = 100_000

# Identity extrinsics: the rear axle coincides with the IMU, so target points can be
# checked directly against the driven positions.
_EGO_METADATA = EgoStateSE3Metadata(
    vehicle_name="test_vehicle",
    width=2.0,
    length=4.5,
    height=1.6,
    wheel_base=2.8,
    center_to_imu_se3=PoseSE3.identity(),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)


def _ego_state(ts_us: int, x: float, y: float, yaw: float) -> EgoStateSE3:
    pose = PoseSE3.from_list(
        [
            x,
            y,
            0.0,
            float(np.cos(yaw / 2.0)),
            0.0,
            0.0,
            float(np.sin(yaw / 2.0)),
        ],
    )
    dynamic = DynamicStateSE3(
        velocity=Vector3D(1.0, 0.0, 0.0),
        acceleration=Vector3D(0.0, 0.0, 0.0),
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
    )
    return EgoStateSE3.from_imu(
        imu_se3=pose,
        metadata=_EGO_METADATA,
        timestamp=Timestamp.from_us(ts_us),
        dynamic_state_se3=dynamic,
    )


def _write_log(
    tmp_path: Path,
    xy_yaw: list[tuple[float, float, float]],
    write_route: bool = True,
) -> ArrowSceneAPI:
    """Writes a log driving through the given (x, y, yaw) poses and opens it."""
    log_metadata = LogMetadata(
        dataset="test-dataset",
        split="test-dataset_train",
        log_name="log_001",
        location=None,
        map_metadata=None,
    )
    writer = ArrowLogWriter(
        LogWriterConfig(write_route=write_route),
        logs_root=tmp_path,
        sensors_root=tmp_path,
    )
    writer.reset(log_metadata)
    for index, (x, y, yaw) in enumerate(xy_yaw):
        ts_us = index * _TIMESTEP_US
        writer.write_sync(
            ModalitiesSync(
                timestamp=Timestamp.from_us(ts_us),
                modalities=[_ego_state(ts_us, x, y, yaw)],
            ),
        )
    writer.close()
    return ArrowSceneAPI(tmp_path / log_metadata.split / log_metadata.log_name)


def test_straight_drive_lands_at_each_distance(tmp_path: Path) -> None:
    scene = _write_log(tmp_path, [(2.0 * i, 0.0, 0.0) for i in range(60)])
    target_points = get_target_points(scene, [40.0, 50.0])
    assert target_points.shape == (2, 2)
    np.testing.assert_allclose(
        target_points,
        [[40.0, 0.0], [50.0, 0.0]],
        atol=1e-3,
    )


def test_output_is_in_the_rear_axle_frame(tmp_path: Path) -> None:
    # Driving +y from an offset origin: 45 m ahead is straight ahead in the ego frame.
    scene = _write_log(
        tmp_path,
        [(10.0, -5.0 + 2.0 * i, np.pi / 2.0) for i in range(30)],
    )
    target_points = get_target_points(scene, [45.0])
    np.testing.assert_allclose(target_points[0], [45.0, 0.0], atol=1e-3)


def test_curved_drive_stays_on_the_route(tmp_path: Path) -> None:
    # Right-turning quarter circle of radius 60 m starting at the origin, tangent +y.
    theta = np.linspace(np.pi, np.pi / 2.0, 200)
    poses = [(60.0 + 60.0 * np.cos(t), 60.0 * np.sin(t), float(t) - np.pi / 2.0) for t in theta]
    scene = _write_log(tmp_path, poses)
    target_points = get_target_points(scene, [45.0])
    angle = 45.0 / 60.0  # arc distance / radius
    expected = [60.0 * np.sin(angle), -60.0 * (1.0 - np.cos(angle))]
    np.testing.assert_allclose(target_points[0], expected, atol=5e-2)


def test_standstill_prefix_keeps_the_anchor_progress(tmp_path: Path) -> None:
    poses = [(0.0, 0.0, 0.0)] * 10 + [(2.0 * i, 0.0, 0.0) for i in range(30)]
    scene = _write_log(tmp_path, poses)
    target_points = get_target_points(scene, [45.0])
    np.testing.assert_allclose(target_points[0], [45.0, 0.0], atol=1e-3)


def test_exactly_enough_route_passes(tmp_path: Path) -> None:
    scene = _write_log(
        tmp_path,
        [(1.0 * i, 0.0, 0.0) for i in range(46)],
    )  # 45 m
    target_points = get_target_points(scene, [45.0])
    np.testing.assert_allclose(target_points[0], [45.0, 0.0], atol=1e-3)


def test_short_route_raises(tmp_path: Path) -> None:
    scene = _write_log(
        tmp_path,
        [(1.0 * i, 0.0, 0.0) for i in range(11)],
    )  # 10 m
    with pytest.raises(InsufficientRouteError, match="min_remaining_route_m"):
        get_target_points(scene, [45.0])


def test_log_without_route_raises(tmp_path: Path) -> None:
    scene = _write_log(
        tmp_path,
        [(2.0 * i, 0.0, 0.0) for i in range(60)],
        write_route=False,
    )
    with pytest.raises(InsufficientRouteError, match="write_route"):
        get_target_points(scene, [45.0])
