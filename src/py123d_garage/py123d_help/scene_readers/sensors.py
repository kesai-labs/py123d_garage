"""Camera frames and LiDAR sweeps: the recorded streams, read at the anchor or back through the past."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI
from py123d.datatypes import CameraID, LidarID
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata
from py123d.geometry import PoseSE2
from py123d.geometry.transform import (
    abs_to_rel_se2,
    rel_to_abs_points_2d_array,
)

from py123d_garage.py123d_help.scene_readers.ego_state import interpolate_ego_se2_at
from py123d_garage.py123d_help.scene_readers.synchronization import (
    TimestampedSample,
    _sample_past_stream,
    past_offsets_us,
)


@runtime_checkable
class CameraFrame(TimestampedSample, Protocol):
    """The part of a recorded camera frame the garage reads."""

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """The recorded image."""
        ...

    @property
    def metadata(self) -> BaseCameraMetadata:
        """The camera the image was recorded with."""
        ...


@runtime_checkable
class LidarSweep(TimestampedSample, Protocol):
    """The part of a recorded LiDAR sweep the garage reads."""

    @property
    def xyz(self) -> jt.Float32[npt.NDArray[np.float32], "num_points 3"]:
        """The sweep's points in the sensor frame."""
        ...


def lidar_at_timestamp(
    scene_api: SceneAPI,
    timestamp_us: int,
    criteria: Literal["nearest", "backward"] = "nearest",
) -> LidarSweep | None:
    """The merged LiDAR matching a timestamp, falling back to the top LiDAR."""
    lidar = scene_api.get_lidar_at_timestamp(timestamp_us, LidarID.LIDAR_MERGED, criteria=criteria)
    if lidar is None:
        lidar = scene_api.get_lidar_at_timestamp(timestamp_us, LidarID.LIDAR_TOP, criteria=criteria)
    return lidar


def lidar_at_anchor(scene_api: SceneAPI) -> LidarSweep | None:
    """The merged LiDAR of the anchor iteration, falling back to the top LiDAR."""
    lidar = scene_api.get_lidar_at_iteration(0, LidarID.LIDAR_MERGED)
    if lidar is None:
        lidar = scene_api.get_lidar_at_iteration(0, LidarID.LIDAR_TOP)
    return lidar


def sample_past_lidar(
    scene_api: SceneAPI,
    anchor_timestamp_us: int,
    horizon_us: int,
    interval_us: int | None,
) -> list[LidarSweep | None]:
    """
    The recorded LiDAR sweeps of one past window, nearest first.

    Args:
        scene_api: scene interface anchored at the current frame
        anchor_timestamp_us: recorded timestamp the window is measured back from
        horizon_us: how far back the oldest sweep reaches
        interval_us: spacing of the sweeps; a sweep may sit half an interval off

    Returns:
        one entry per sampled offset, None where the stream had not started yet

    Raises:
        ValueError: if the window is not a whole number of intervals, or the stream
            dropped out inside its recorded coverage
    """
    return _sample_past_stream(
        fetch=lambda timestamp_us, criteria: lidar_at_timestamp(scene_api, timestamp_us, criteria),
        anchor_timestamp_us=anchor_timestamp_us,
        past_offsets_us=past_offsets_us(horizon_us, interval_us),
        tolerance_us=(interval_us or 0) // 2,
        stream_description=f"LiDAR of scene {scene_api.scene_uuid} in log {scene_api.log_name}",
    )


def sample_lidar_stack(
    scene_api: SceneAPI,
    horizon_us: int,
    interval_us: int | None,
) -> list[LidarSweep]:
    """
    The anchor LiDAR sweep followed by the past sweeps the stream could serve, newest first.

    Offsets the stream had not started for are dropped, so the stack is shorter near the
    beginning of a log or an episode; a dropout inside recorded coverage still raises.

    Args:
        scene_api: scene interface anchored at the current frame
        horizon_us: how far back the oldest sweep reaches
        interval_us: spacing of the sweeps; a sweep may sit half an interval off

    Returns:
        the anchor sweep first, then the served past sweeps from newest to oldest

    Raises:
        ValueError: if the anchor frame records no LiDAR
    """
    anchor = lidar_at_anchor(scene_api)
    if anchor is None:
        raise ValueError(
            f"scene {scene_api.scene_uuid} of log {scene_api.log_name} has no LiDAR at its anchor frame.",
        )
    past_sweeps = sample_past_lidar(scene_api, anchor.timestamp.time_us, horizon_us, interval_us)
    return [anchor, *(sweep for sweep in past_sweeps if sweep is not None)]


def accumulate_lidar_in_anchor_frame(
    scene_api: SceneAPI,
    horizon_us: int,
    interval_us: int | None,
) -> jt.Float32[npt.NDArray[np.float32], "num_points 3"]:
    """
    The LiDAR stack concatenated into one cloud, past sweeps moved into the anchor frame.

    Every sweep is recorded from where the ego stood at the time, so stacking them raw
    smears the scene. Each past sweep is moved by the SE2 step between the ego pose
    interpolated at its own recorded timestamp and the pose at the anchor's; the ego's
    roll, pitch and height change, and the ego's motion within a single sweep, are not
    modelled. Callers that want the sweeps apart, or a different correction, take
    sample_lidar_stack instead.

    Args:
        scene_api: scene interface anchored at the current frame
        horizon_us: how far back the oldest stacked sweep reaches
        interval_us: spacing of the stacked sweeps

    Returns:
        the anchor sweep followed by the corrected past sweeps, in the anchor ego frame
    """
    sweeps = sample_lidar_stack(scene_api, horizon_us, interval_us)
    if len(sweeps) == 1:
        return sweeps[0].xyz

    timestamps_us = np.array([sweep.timestamp.time_us for sweep in sweeps], dtype=np.int64)
    poses_se2 = interpolate_ego_se2_at(
        scene_api,
        timestamps_us,
        boundary_tolerance_us=(interval_us or 0) // 2,
    )
    anchor_se2 = PoseSE2.from_array(poses_se2[0])

    points: list[jt.Float32[npt.NDArray[np.float32], "num_points 3"]] = [sweeps[0].xyz]
    for sweep, pose_se2 in zip(sweeps[1:], poses_se2[1:], strict=True):
        # Pose of the past ego frame expressed in the anchor frame.
        past_in_anchor = abs_to_rel_se2(anchor_se2, PoseSE2.from_array(pose_se2))
        corrected = sweep.xyz.copy()
        corrected[:, :2] = rel_to_abs_points_2d_array(
            past_in_anchor,
            corrected[:, :2].astype(np.float64),
        )
        points.append(corrected)
    return np.concatenate(points, axis=0)


def camera_at_anchor(scene_api: SceneAPI, camera_id: CameraID) -> CameraFrame:
    """
    The recorded frame of one camera at the anchor iteration.

    Args:
        scene_api: scene interface anchored at the current frame
        camera_id: the camera to read

    Returns:
        the recorded frame

    Raises:
        ValueError: if that camera recorded no frame at the anchor iteration
    """
    camera = scene_api.get_camera_at_iteration(0, camera_id)
    if camera is None:
        raise ValueError(
            f"camera {camera_id.name} of scene {scene_api.scene_uuid} in log {scene_api.log_name} "
            "has no frame at its anchor iteration.",
        )
    return camera


def sample_past_camera(
    scene_api: SceneAPI,
    camera_id: CameraID,
    anchor_timestamp_us: int,
    horizon_us: int,
    interval_us: int | None,
) -> list[CameraFrame | None]:
    """
    The recorded frames of one camera over a past window, nearest first.

    Args:
        scene_api: scene interface anchored at the current frame
        camera_id: the camera to read
        anchor_timestamp_us: recorded timestamp the window is measured back from
        horizon_us: how far back the oldest frame reaches
        interval_us: spacing of the frames; a frame may sit half an interval off

    Returns:
        one entry per sampled offset, None where the stream had not started yet

    Raises:
        ValueError: if the window is not a whole number of intervals, or the stream
            dropped out inside its recorded coverage
    """

    def fetch(timestamp_us: int, criteria: Literal["nearest", "backward"]) -> CameraFrame | None:
        return scene_api.get_camera_at_timestamp(timestamp_us, camera_id, criteria=criteria)

    return _sample_past_stream(
        fetch=fetch,
        anchor_timestamp_us=anchor_timestamp_us,
        past_offsets_us=past_offsets_us(horizon_us, interval_us),
        tolerance_us=(interval_us or 0) // 2,
        stream_description=(f"camera {camera_id.name} of scene {scene_api.scene_uuid} in log {scene_api.log_name}"),
    )
