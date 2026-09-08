"""TransFuser input features: the LiDAR BEV histogram, camera stitching and ego status."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from py123d.api import SceneAPI
from py123d.datatypes import CameraID, LidarID, Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    EgoStateSE3Metadata,
)
from py123d.geometry import PoseSE2
from py123d.geometry.pose import PoseSE3

from py123d_garage.config.schema.policy.transfuser_config import (
    CameraConfig,
    LidarConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import (
    build_camera_feature,
    build_lidar_feature,
    build_velocity,
)

# Identity extrinsics: the rear axle coincides with the IMU, so the stub's SE2
# poses come back unchanged from rear_axle_se2.
_EGO_METADATA = EgoStateSE3Metadata(
    vehicle_name="test_vehicle",
    width=2.0,
    length=4.5,
    height=1.6,
    wheel_base=2.8,
    center_to_imu_se3=PoseSE3.identity(),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)


def _ego_state(timestamp_us: int, pose: PoseSE2) -> EgoStateSE3:
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


class _LidarScene:
    """SceneAPI stand-in serving timestamped point clouds and ego poses."""

    def __init__(
        self,
        anchor_timestamp_us: int,
        sweeps: dict[int, npt.NDArray[np.float32]],
        poses: dict[int, PoseSE2],
    ) -> None:
        self._anchor_timestamp_us = anchor_timestamp_us
        self._sweeps = sweeps
        self._poses = poses

    @property
    def scene_uuid(self) -> str:
        return "test-scene"

    @property
    def log_name(self) -> str:
        return "test-log"

    def get_lidar_at_iteration(
        self,
        iteration: int,
        lidar_id: LidarID,
    ) -> SimpleNamespace | None:
        return self.get_lidar_at_timestamp(self._anchor_timestamp_us, lidar_id, criteria="nearest")

    def get_lidar_at_timestamp(
        self,
        timestamp: int,
        lidar_id: LidarID,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> SimpleNamespace | None:
        candidates_us = (
            [sweep_us for sweep_us in self._sweeps if sweep_us <= timestamp]
            if criteria == "backward"
            else list(self._sweeps)
        )
        if not candidates_us:
            return None
        matched_us = min(candidates_us, key=lambda sweep_us: abs(sweep_us - timestamp))
        return SimpleNamespace(
            xyz=self._sweeps[matched_us],
            timestamp=Timestamp.from_us(matched_us),
        )

    def get_modality_between_timestamps(
        self,
        start_timestamp: int,
        end_timestamp: int,
        modality_type: str,
        inclusive: str = "left",
    ) -> Iterator[EgoStateSE3]:
        for timestamp_us in sorted(self._poses):
            if start_timestamp <= timestamp_us <= end_timestamp:
                yield _ego_state(timestamp_us, self._poses[timestamp_us])


class _CameraScene:
    """SceneAPI stand-in serving one constant-color image per camera."""

    def __init__(self, images: dict[CameraID, npt.NDArray[np.uint8]]) -> None:
        self._images = images

    @property
    def scene_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(dataset="nuplan")

    @property
    def scene_uuid(self) -> str:
        return "test-scene"

    @property
    def log_name(self) -> str:
        return "test-log"

    def get_camera_at_iteration(
        self,
        iteration: int,
        camera_id: CameraID,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            image=self._images[camera_id],
            metadata=SimpleNamespace(camera_id=camera_id),
            timestamp=Timestamp.from_us(_ANCHOR_US),
        )


class _StatusScene:
    """SceneAPI stand-in serving one ego velocity."""

    def __init__(self, velocity_2d: npt.NDArray[np.float64]) -> None:
        self._velocity_2d = velocity_2d

    def get_ego_state_se3_at_iteration(
        self,
        iteration: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dynamic_state_se3=SimpleNamespace(
                velocity_2d=SimpleNamespace(array=self._velocity_2d),
            ),
        )


SceneAPI.register(_LidarScene)
SceneAPI.register(_CameraScene)
SceneAPI.register(_StatusScene)


def _as_scene(stub: object) -> SceneAPI:
    """The stubs are virtual SceneAPI subclasses (register); cast for the type checker."""
    return cast(SceneAPI, stub)


_ORIGIN = PoseSE2(0.0, 0.0, 0.0)
_ANCHOR_US = 1_000_000
_SWEEP_INTERVAL_US = 100_000


def _anchor_only_config() -> TransfuserConfig:
    return TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        lidar_config=LidarConfig(remove_lidar_ground_points=False),
    )


def test_lidar_raster_point_lands_in_expected_cell() -> None:
    """0.25 m cells from (bev_min_x, bev_min_y): (0.1, 0.1) is row 160, col 128; (10.4, -5.3) is row 138, col 169."""
    points = np.array(
        [[0.1, 0.1, 1.0], [10.4, -5.3, 1.0]],
        dtype=np.float32,
    )
    scene = _LidarScene(_ANCHOR_US, {_ANCHOR_US: points}, {_ANCHOR_US: _ORIGIN})
    raster = build_lidar_feature(_as_scene(scene), _anchor_only_config())
    assert raster.shape == (1, 320, 384)
    assert torch.isclose(raster[0, 160, 128], torch.tensor(0.2))
    assert torch.isclose(raster[0, 138, 169], torch.tensor(0.2))
    assert torch.isclose(raster.sum(), torch.tensor(0.4))
    assert int(torch.count_nonzero(raster)) == 2


def test_lidar_raster_clips_and_normalizes_counts() -> None:
    """Counts saturate at max_lidar_points_per_bev_pixel=5 and divide by it."""
    points = np.concatenate(
        [
            np.tile(np.array([[0.1, 0.1, 1.0]], dtype=np.float32), (7, 1)),
            np.tile(np.array([[5.1, 5.1, 1.0]], dtype=np.float32), (2, 1)),
        ],
        axis=0,
    )
    scene = _LidarScene(_ANCHOR_US, {_ANCHOR_US: points}, {_ANCHOR_US: _ORIGIN})
    raster = build_lidar_feature(_as_scene(scene), _anchor_only_config())
    assert raster[0, 160, 128] == 1.0
    assert torch.isclose(raster[0, 180, 148], torch.tensor(0.4))


def test_lidar_raster_drops_points_outside_extents_and_heights() -> None:
    points = np.array(
        [
            [70.0, 0.0, 1.0],  # beyond bev_max_x
            [0.0, 45.0, 1.0],  # beyond bev_max_y
            [0.1, 0.1, 11.0],  # above lidar_max_height
            [0.1, 0.1, -5.0],  # below lidar_min_height
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    scene = _LidarScene(_ANCHOR_US, {_ANCHOR_US: points}, {_ANCHOR_US: _ORIGIN})
    raster = build_lidar_feature(_as_scene(scene), _anchor_only_config())
    assert torch.isclose(raster.sum(), torch.tensor(0.2))
    assert torch.isclose(raster[0, 160, 128], torch.tensor(0.2))
    assert int(torch.count_nonzero(raster)) == 1


def test_camera_features_stitch_left_to_right() -> None:
    """input_cameras order is the stitch order: left, front, right thirds of the image."""
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(
            input_cameras={"nuplan": [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0]},
        ),
    )
    values = (10, 20, 30)
    images = {
        camera_id: np.full((64, 128, 3), value, dtype=np.uint8)
        for camera_id, value in zip(
            config.camera_config.input_cameras["nuplan"],
            values,
            strict=True,
        )
    }
    feature = build_camera_feature(_as_scene(_CameraScene(images)), config)
    assert feature.shape == (3, 256, 1024)
    assert feature.dtype == torch.float32
    for value, col in zip(values, (100, 512, 900), strict=True):
        assert torch.all(feature[:, :, col] == float(value))


def test_velocity_is_speed_norm() -> None:
    scene = _StatusScene(np.array([3.0, 4.0]))
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
    )
    velocity = build_velocity(_as_scene(scene), config)
    assert torch.equal(velocity, torch.tensor([5.0]))
