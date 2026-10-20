# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import SceneAPI
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.geometry import Point3DIndex
from py123d.geometry.transform import abs_to_rel_se2_array

from py123d_garage.api.abstract_policy_tensors import AbstractFeatures
from py123d_garage.common.sensor.point_cloud_ground_removal import remove_ground
from py123d_garage.config.schema.policy.transfuser_config import (
    LidarConfig,
    TransfuserConfig,
)
from py123d_garage.py123d_help.scene_readers import (
    CameraFrame,
    accumulate_lidar_in_anchor_frame,
    camera_at_anchor,
    interpolate_ego_se2_at,
    sample_past_camera,
)


@dataclass(frozen=True)
class TransfuserFeatures(AbstractFeatures):
    """TransFuser's input tensors."""

    # Stitched camera image in [0, 255], past rows above the anchor row.
    camera_feature: jt.UInt8[
        torch.Tensor,
        "*batch 3 camera_height camera_width",
    ]
    # Ego speed at the anchor in m/s.
    velocity: jt.Float[torch.Tensor, "*batch 1"]
    # Ego velocity (v_x, v_y) at the anchor in m/s, in the ego frame.
    velocity_vector: jt.Float[torch.Tensor, "*batch 2"] | None = None
    # Ego acceleration (a_x, a_y) at the anchor in m/s^2, in the ego frame.
    acceleration_vector: jt.Float[torch.Tensor, "*batch 2"] | None = None
    # Past ego poses (x, y, yaw) relative to the anchor, oldest first; None without ego history.
    ego_history: jt.Float[torch.Tensor, "*batch steps 3"] | None = None
    # LiDAR BEV point-count raster in [0, 1]; None in latent mode.
    lidar_feature: jt.Float[torch.Tensor, "*batch 1 lidar_height lidar_width"] | None = None


def build_camera_feature(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.UInt8[torch.Tensor, "3 camera_height camera_width"]:
    """
    Stitched camera image of the anchor frame, below the past frames (oldest on top) when past_camera_offset_us is set.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        camera image with values in [0, 255]; the backbone normalizes it
    """
    frame_rows = camera_frame_rows(scene_api, config)
    row_height = config.camera_config.image_height // len(frame_rows)
    stitched_image = np.concatenate([_camera_row(frames, config, row_height) for frames in frame_rows], axis=0)
    return torch.as_tensor(stitched_image).permute(2, 0, 1).contiguous()


def camera_frame_rows(scene_api: SceneAPI, config: TransfuserConfig) -> list[list[CameraFrame]]:
    """
    The frames of every row of the stitched camera image, top to bottom: the past rows oldest first, then the anchor row.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        one frame per input camera in stitch order for each row; one row without past_camera_offset_us
    """
    camera = config.camera_config
    camera_ids = camera.input_cameras[scene_api.scene_metadata.dataset]
    frames = [camera_at_anchor(scene_api, camera_id) for camera_id in camera_ids]
    if not camera.past_camera_offset_us:
        return [frames]

    # Per camera, the past frames nearest first; a rollout younger than a row repeats the oldest frame it has.
    past_frames_by_camera: list[list[CameraFrame]] = []
    for camera_id, frame in zip(camera_ids, frames, strict=True):
        filled = [frame]
        for past in sample_past_camera(
            scene_api,
            camera_id,
            anchor_timestamp_us=frame.timestamp.time_us,
            horizon_us=camera.past_camera_offset_us * camera.num_past_camera_rows,
            interval_us=camera.past_camera_offset_us,
        ):
            filled.append(filled[-1] if past is None else past)
        past_frames_by_camera.append(filled[1:])
    return [
        *(
            [past_frames[row] for past_frames in past_frames_by_camera]
            for row in reversed(range(camera.num_past_camera_rows))
        ),
        frames,
    ]


def _camera_row(
    frames: list[CameraFrame],
    config: TransfuserConfig,
    height: int,
) -> jt.UInt8[npt.NDArray[np.uint8], "row_height image_width 3"]:
    """
    Stitches the input cameras left to right into one resized image row.

    Args:
        frames: one frame per input camera, in stitch order
        config: global config dataclass of TransFuser
        height: the row's height in pixels

    Returns:
        the row image
    """
    camera = config.camera_config
    stitched_image = np.concatenate([frame.image for frame in frames], axis=1)
    return cast("npt.NDArray[np.uint8]", cv2.resize(stitched_image, (camera.image_width, height)))


def build_lidar_feature(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, "1 lidar_height lidar_width"]:
    """
    Rasterized LiDAR BEV histogram, with past sweeps accumulated into the anchor frame.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        normalized point-count raster on the BEV grid
    """
    assert config.lidar_config.lidar_in_channels == 1, "Only single-channel LiDAR rasters are supported."

    lidar_xyz = accumulate_lidar_in_anchor_frame(
        scene_api,
        config.lidar_config.lidar_horizon_us,
        config.lidar_config.lidar_interval_us,
    )

    z = lidar_xyz[..., Point3DIndex.Z]
    lidar_xyz = lidar_xyz[(z > config.lidar_config.lidar_min_height_m) & (z < config.lidar_config.lidar_max_height_m)]
    if config.lidar_config.remove_lidar_ground_points:
        lidar_xyz = _drop_ground_points(lidar_xyz, config.lidar_config)

    return _rasterize_bev(lidar_xyz, config.lidar_config)


def build_velocity(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, " 1"]:
    """
    The ego speed of the anchor frame.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        the ego speed in m/s
    """
    del config
    speed = float(np.linalg.norm(_dynamic_state(scene_api).velocity_2d.array))

    return torch.tensor([speed], dtype=torch.float32)


def build_velocity_vector(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, " 2"]:
    """
    The ego velocity vector of the anchor frame.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        the ego velocity (v_x, v_y) in m/s, in the ego frame
    """
    del config
    return torch.tensor(_dynamic_state(scene_api).velocity_2d.array, dtype=torch.float32)


def build_acceleration_vector(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, " 2"]:
    """
    The ego acceleration vector of the anchor frame.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        the ego acceleration (a_x, a_y) in m/s^2, in the ego frame
    """
    del config
    return torch.tensor(_dynamic_state(scene_api).acceleration_2d.array, dtype=torch.float32)


def _dynamic_state(scene_api: SceneAPI) -> DynamicStateSE3:
    """
    Reads the ego dynamic state of the anchor frame.

    Args:
        scene_api: scene interface anchored at the current frame

    Returns:
        the anchor's ego velocity and acceleration state
    """
    ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for feature computation!"
    dynamic_state_se3 = ego_state_se3.dynamic_state_se3
    assert dynamic_state_se3 is not None, "Ego dynamic state should be available for feature computation!"
    return dynamic_state_se3


def build_ego_history(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, "steps 3"]:
    """
    The ego's past poses relative to the anchor, oldest step first, in meters and radians.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        the past ego poses as (x, y, yaw), relative to the anchor
    """
    conditioning = config.planning_conditioning_config
    num_steps = config.ego_history_num_steps
    ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for feature computation!"

    anchor_timestamp_us = ego_state_se3.timestamp.time_us
    lookback_us = np.arange(num_steps, 0, -1, dtype=np.int64) * np.int64(conditioning.ego_history_interval_us)
    poses_se2 = interpolate_ego_se2_at(
        scene_api,
        anchor_timestamp_us - lookback_us,
        boundary_tolerance_us=conditioning.ego_history_interval_us,
    )
    relative_se2 = abs_to_rel_se2_array(
        origin=ego_state_se3.rear_axle_se2,
        pose_se2_array=poses_se2,
    )
    return torch.tensor(relative_se2, dtype=torch.float32)


def _drop_ground_points(
    lidar_xyz: jt.Float32[npt.NDArray[np.float32], "n 3"],
    lidar_config: LidarConfig,
) -> jt.Float32[npt.NDArray[np.float32], "m 3"]:
    """
    Removes the ground returns from an anchor-frame point cloud.

    Args:
        lidar_xyz: the point cloud in the anchor ego frame
        lidar_config: the LiDAR section of the TransFuser config

    Returns:
        the points that are not ground

    Raises:
        ValueError: if lidar_config.ground_removal names no known method
    """
    method = lidar_config.ground_removal
    if len(lidar_xyz) == 0:
        return lidar_xyz
    if method == "fixed_z":
        return lidar_xyz[lidar_xyz[..., Point3DIndex.Z] > lidar_config.lidar_ground_z_m]
    if method == "fitted_plane":
        bounds = (
            lidar_config.bev_min_x_m,
            lidar_config.bev_max_x_m,
            lidar_config.bev_min_y_m,
            lidar_config.bev_max_y_m,
        )
        return lidar_xyz[~remove_ground(lidar_xyz, bounds)]
    raise ValueError(
        f"Unknown ground_removal {method!r}: use 'fixed_z' or 'fitted_plane'.",
    )


def _rasterize_bev(
    lidar_xyz: jt.Float32[npt.NDArray[np.float32], "n 3"],
    lidar_config: LidarConfig,
) -> jt.Float[torch.Tensor, "1 lidar_height lidar_width"]:
    """
    Counts the points falling into each cell of the BEV grid.

    Args:
        lidar_xyz: the point cloud in the anchor ego frame
        lidar_config: the LiDAR section of the TransFuser config

    Returns:
        point counts clipped at the configured maximum and scaled to [0, 1]
    """
    x_bins = np.linspace(
        lidar_config.bev_min_x_m,
        lidar_config.bev_max_x_m,
        lidar_config.bev_width_pixel + 1,
    )
    y_bins = np.linspace(
        lidar_config.bev_min_y_m,
        lidar_config.bev_max_y_m,
        lidar_config.bev_height_pixel + 1,
    )
    # rows index y, columns index x
    hist = np.histogramdd(
        lidar_xyz[:, [Point3DIndex.Y, Point3DIndex.X]],
        bins=(y_bins, x_bins),
    )[0]
    hist[hist > lidar_config.max_lidar_points_per_bev_pixel] = lidar_config.max_lidar_points_per_bev_pixel
    hist = hist / lidar_config.max_lidar_points_per_bev_pixel

    return torch.tensor(hist[None].astype(np.float32))
