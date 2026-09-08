"""The CARLA rig of a trained policy, recovered from the sensor_rig yaml of its training run."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml
from py123d.api.utils.arrow_metadata_utils import resolve_metadata_class
from py123d.datatypes import (
    BaseModalityMetadata,
    CameraID,
    EgoStateSE3Metadata,
    LidarMergedMetadata,
    LidarMetadata,
    PinholeCameraMetadata,
)
from py123d.geometry import EulerAngles, PoseSE3
from py123d.parser.utils.sensor_utils.camera_conventions import (
    convert_camera_convention,
)

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveFloat, PositiveInt

LOG = logging.getLogger(__name__)

SensorSpec = dict[str, float | int | str]

CARLA_FPS = 20

_EGO_HALF_LENGTH_M = 2.4508416652679443
_EGO_HALF_WIDTH_M = 1.0641621351242065

_POINT_PRECISION_M = 0.1

_LIDAR_MOUNTS: list[dict[str, float]] = [
    {"x": 1.0, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -90.0},
    {"x": -1.0, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -270.0},
]


class SensorRig:
    """
    The sensors a policy was trained on, as CARLA mounts and as 123D metadata.

    A training run writes the modality metadata of its data source next to the
    checkpoint; that file is the only description of the rig the model expects,
    so evaluation mounts exactly what it names.
    """

    def __init__(self, sensor_rig_file: str) -> None:
        """
        Reads the rig, idealizing what CARLA cannot reproduce exactly.

        Lens distortion is dropped (CARLA renders ideal pinholes) and an
        off-center principal point is realized by rendering an overscanned frame
        that crop_to_rig cuts back to the metadata's geometry.

        Args:
            sensor_rig_file: sensor_rig_<source>.yaml of the training run.
        """
        rig: dict[str, Any] = yaml.safe_load(Path(sensor_rig_file).read_text())
        raw_modalities = cast("dict[str, dict[str, Any]]", rig["modalities"])
        for key, value in raw_modalities.items():
            if key.startswith("camera.") and (value.get("distortion") is not None or not value.get("is_undistorted")):
                LOG.warning(f"{key} of {sensor_rig_file} carries lens distortion; serving ideal pinhole images.")
                value["distortion"] = None
                value["is_undistorted"] = True
        # Only the sensors get mounted; the rig's other modalities, like the route of its log, stay unread.
        metadatas: dict[str, BaseModalityMetadata] = {
            key: cast("BaseModalityMetadata", resolve_metadata_class(key).from_dict(value))
            for key, value in raw_modalities.items()
            if key == "ego_state_se3" or key.startswith(("camera.", "lidar."))
        }

        ego_metadata = metadatas["ego_state_se3"]
        assert isinstance(ego_metadata, EgoStateSE3Metadata)
        assert np.allclose(
            ego_metadata.rear_axle_to_imu_se3.array,
            PoseSE3.identity().array,
        ), (
            f"{sensor_rig_file} places the IMU at {ego_metadata.rear_axle_to_imu_se3}, "
            f"not at the rear axle; the CARLA mount poses are derived in the rear-axle frame."
        )
        self._ego_metadata = ego_metadata

        self._camera_metadatas: dict[CameraID, PinholeCameraMetadata] = {}
        self._camera_focal_lengths: dict[CameraID, PositiveFloat] = {}
        self._camera_render_sizes: dict[CameraID, tuple[PositiveInt, PositiveInt]] = {}
        self._camera_crops: dict[CameraID, tuple[NonNegativeInt, NonNegativeInt]] = {}
        for key, metadata in metadatas.items():
            if not key.startswith("camera."):
                continue
            assert isinstance(metadata, PinholeCameraMetadata), (
                f"{key} of {sensor_rig_file} is a {type(metadata).__name__}; CARLA renders pinhole cameras only."
            )
            intrinsics = metadata.intrinsics
            assert intrinsics is not None, (
                f"{key} of {sensor_rig_file} carries no intrinsics, so its field of view is unknown."
            )
            assert np.isclose(intrinsics.fx, intrinsics.fy), (
                f"{key} of {sensor_rig_file} has fx {intrinsics.fx} != fy {intrinsics.fy}; "
                f"a CARLA camera has one focal length."
            )
            render_width = round(2 * max(intrinsics.cx, metadata.width - intrinsics.cx))
            render_height = round(2 * max(intrinsics.cy, metadata.height - intrinsics.cy))
            if (render_width, render_height) != (metadata.width, metadata.height):
                LOG.warning(
                    f"{key} of {sensor_rig_file} has its principal point ({intrinsics.cx}, {intrinsics.cy}) "
                    f"off-center; rendering {render_width}x{render_height} and cropping.",
                )
            self._camera_metadatas[metadata.camera_id] = metadata
            self._camera_focal_lengths[metadata.camera_id] = intrinsics.fx
            self._camera_render_sizes[metadata.camera_id] = (render_width, render_height)
            self._camera_crops[metadata.camera_id] = (
                round(render_width / 2 - intrinsics.cx),
                round(render_height / 2 - intrinsics.cy),
            )

        lidar_keys = [key for key in metadatas if key.startswith("lidar.")]
        assert len(lidar_keys) == 1, (
            f"{sensor_rig_file} declares the lidars {lidar_keys}; exactly one lidar modality is served."
        )
        self._lidar_modality_key = lidar_keys[0]
        lidar_metadata = metadatas[lidar_keys[0]]
        assert isinstance(lidar_metadata, LidarMetadata | LidarMergedMetadata), (
            f"{lidar_keys[0]} of {sensor_rig_file} is a {type(lidar_metadata).__name__}, not a lidar metadata."
        )
        self._lidar_metadata = lidar_metadata
        mount_matches = isinstance(lidar_metadata, LidarMetadata) and all(
            np.isclose(_pose_to_carla_mount(lidar_metadata.lidar_to_imu_se3, 0.0, 0.0)[axis], value)
            for axis, value in _LIDAR_MOUNTS[0].items()
        )
        if not mount_matches:
            LOG.warning(
                f"{lidar_keys[0]} of {sensor_rig_file} does not match the CARLA collector's rig; "
                f"mounting the CARLA lidar pair at {_LIDAR_MOUNTS} instead.",
            )
        self._lidar_mounts = _LIDAR_MOUNTS

    @property
    def ego_metadata(self) -> EgoStateSE3Metadata:
        """The recording vehicle, as the ego state modality of the logs describes it."""
        return self._ego_metadata

    @property
    def lidar_metadata(self) -> LidarMetadata | LidarMergedMetadata:
        """The lidar modality the accumulated sweep is served under."""
        return self._lidar_metadata

    @property
    def lidar_modality_key(self) -> str:
        """The 123D modality key the rig's lidar sweep is served under."""
        return self._lidar_modality_key

    @property
    def camera_metadatas(self) -> dict[CameraID, PinholeCameraMetadata]:
        """The rig's cameras by 123D camera id."""
        return self._camera_metadatas

    def crop_to_rig(
        self,
        camera_id: CameraID,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """
        Cuts a rendered frame back to the rig's geometry, landing the principal point where the metadata says.

        Args:
            camera_id: the 123D camera id.
            image: the frame CARLA rendered at the overscanned size.

        Returns:
            the metadata-sized frame.
        """
        x0, y0 = self._camera_crops[camera_id]
        metadata = self._camera_metadatas[camera_id]
        return image[y0 : y0 + metadata.height, x0 : x0 + metadata.width]

    def camera_sensor_id(self, camera_id: CameraID) -> str:
        """
        The leaderboard sensor id this rig gives a camera; keys input_data.

        Args:
            camera_id: the 123D camera id.

        Returns:
            the sensor id, e.g. "rgb_pcam_f0".
        """
        return f"rgb_{camera_id.name.lower()}"

    @property
    def lidar_sensor_ids(self) -> list[str]:
        """The leaderboard sensor ids of the rig's lidars, in the order their sweeps merge."""
        return [f"lidar_{index}" for index in range(len(self._lidar_mounts))]

    def leaderboard_sensors(self) -> list[SensorSpec]:
        """
        The full sensor suite to mount: the rig's cameras and lidar plus localization.

        Returns:
            the leaderboard sensor specifications.
        """
        floor_to_rear_axle_longitudinal_m = self._ego_metadata.rear_axle_to_center_longitudinal
        floor_to_rear_axle_vertical_m = self._ego_metadata.height / 2 - self._ego_metadata.rear_axle_to_center_vertical

        sensors: list[SensorSpec] = []
        for camera_id, metadata in self._camera_metadatas.items():
            unreal_pose = convert_camera_convention(
                metadata.camera_to_imu_se3,
                from_convention="pZmYpX",
                to_convention="pXpZmY",
            )
            mount = _pose_to_carla_mount(
                unreal_pose,
                floor_to_rear_axle_longitudinal_m,
                floor_to_rear_axle_vertical_m,
            )
            render_width, render_height = self._camera_render_sizes[camera_id]
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    **mount,
                    "width": render_width,
                    "height": render_height,
                    "fov": float(
                        np.rad2deg(
                            2
                            * np.arctan(
                                render_width / (2 * self._camera_focal_lengths[camera_id]),
                            ),
                        ),
                    ),
                    "id": self.camera_sensor_id(camera_id),
                },
            )
        for mount, sensor_id in zip(self._lidar_mounts, self.lidar_sensor_ids, strict=True):
            sensors.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    **mount,
                    "id": sensor_id,
                },
            )
        sensors.extend(
            [
                {
                    "type": "sensor.other.gnss",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "sensor_tick": 0.01,
                    "id": "gps",
                },
                {
                    "type": "sensor.other.imu",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 1.0 / CARLA_FPS,
                    "id": "imu",
                },
                {
                    "type": "sensor.speedometer",
                    "reading_frequency": CARLA_FPS,
                    "id": "speed",
                },
            ],
        )
        return sensors

    def merge_lidar_sweeps(
        self,
        sweeps: list[npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float32]:
        """
        Merges this tick's raw CARLA sweeps into the point cloud the logs store.

        Reproduces the collector's chain: each lidar's mount transform, the cut
        of the ego's own returns, the flip into the 123D rear-axle frame, and the
        storage quantization.

        Args:
            sweeps: each lidar's (x, y, z, intensity) returns in its own frame,
                in lidar_sensor_ids order.

        Returns:
            the merged sweep as (x, y, z) in the 123D IMU frame.
        """
        ego_frame_points = np.concatenate(
            [_mount_sweep(mount, sweep) for mount, sweep in zip(self._lidar_mounts, sweeps, strict=True)],
            axis=0,
        )
        outside_ego = (np.abs(ego_frame_points[:, 0]) > _EGO_HALF_LENGTH_M) & (
            np.abs(ego_frame_points[:, 1]) > _EGO_HALF_WIDTH_M
        )
        ego_frame_points = ego_frame_points[outside_ego]

        ego_frame_points[:, 0] += self._ego_metadata.rear_axle_to_center_longitudinal
        ego_frame_points[:, 1] *= -1
        ego_frame_points[:, 2] += self._lidar_mounts[0]["z"] / 2 - self._ego_metadata.rear_axle_to_center_vertical

        quantized = np.round(ego_frame_points / _POINT_PRECISION_M) * _POINT_PRECISION_M
        return quantized.astype(np.float32)


def _mount_sweep(
    mount: dict[str, float],
    sweep: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    """
    Places one lidar's returns in the CARLA ego frame the collector merged them in.

    Args:
        mount: the lidar's CARLA mount pose.
        sweep: its (x, y, z, intensity) returns in its own frame.

    Returns:
        the returns as (x, y, z) in the collector's ego frame.
    """
    rotation = EulerAngles(
        roll=float(np.deg2rad(mount["roll"])),
        pitch=float(np.deg2rad(mount["pitch"])),
        yaw=float(np.deg2rad(mount["yaw"])),
    ).rotation_matrix
    translation = np.array(
        [
            mount["x"],
            mount["y"],
            # Undone in merge_lidar_sweeps with the first mount's height alone.
            mount["z"] / 2,
        ],
    )
    return (rotation @ sweep[:, :3].astype(np.float64).T).T + translation


def _pose_to_carla_mount(
    pose: PoseSE3,
    floor_to_rear_axle_longitudinal_m: float,
    floor_to_rear_axle_vertical_m: float,
) -> dict[str, float]:
    """
    Expresses a 123D rear-axle-frame sensor pose as a CARLA mount pose.

    CARLA's ego frame sits on the floor under the vehicle centre and is
    left-handed, so the offsets undo the rear-axle anchoring and the y axis, the
    pitch and the yaw flip sign.

    Args:
        pose: the sensor pose in the 123D IMU frame.
        floor_to_rear_axle_longitudinal_m: rear-axle offset behind the CARLA origin.
        floor_to_rear_axle_vertical_m: rear-axle height above the CARLA origin.

    Returns:
        the mount as the leaderboard's x/y/z/roll/pitch/yaw entries.
    """
    euler = EulerAngles.from_rotation_matrix(pose.quaternion.rotation_matrix)
    return {
        "x": float(pose.x - floor_to_rear_axle_longitudinal_m),
        "y": float(-pose.y),
        "z": float(pose.z + floor_to_rear_axle_vertical_m),
        "roll": float(np.rad2deg(euler.roll)),
        "pitch": float(-np.rad2deg(euler.pitch)),
        "yaw": float(-np.rad2deg(euler.yaw)),
    }
