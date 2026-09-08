from __future__ import annotations

from dataclasses import dataclass

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import SceneAPI
from py123d.geometry import Point3DIndex

from py123d_garage.api.abstract_policy_tensors import AbstractFeatures
from py123d_garage.common.sensor.point_cloud_ground_removal import remove_ground
from py123d_garage.config.schema.policy.transfuser_config import (
    LidarConfig,
    TransfuserConfig,
)
from py123d_garage.py123d_help.scene_readers import (
    accumulate_lidar_in_anchor_frame,
    camera_at_anchor,
)


@dataclass(frozen=True)
class TransfuserFeatures(AbstractFeatures):
    """TransFuser's input tensors; lidar_feature is None in latent mode."""

    camera_feature: jt.Float[
        torch.Tensor,
        "*batch 3 camera_height camera_width",
    ]
    velocity: jt.Float[torch.Tensor, "*batch 1"]
    lidar_feature: jt.Float[torch.Tensor, "*batch 1 lidar_height lidar_width"] | None = None


def build_camera_feature(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, "3 camera_height camera_width"]:
    """
    Stitched camera image of the anchor frame.

    Args:
        scene_api: scene interface anchored at the current frame
        config: global config dataclass of TransFuser

    Returns:
        camera image with values in [0, 255]
        (the backbone applies ImageNet normalization itself)
    """
    images: list[npt.NDArray[np.uint8]] = [
        camera_at_anchor(scene_api, camera_id).image
        for camera_id in config.camera_config.input_cameras[scene_api.scene_metadata.dataset]
    ]

    stitched_image = np.concatenate(images, axis=1)
    resized_image = cv2.resize(
        stitched_image,
        (config.camera_config.image_width, config.camera_config.image_height),
    )
    return torch.as_tensor(resized_image).permute(2, 0, 1).float()


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
    ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for feature computation!"
    dynamic_state_se3 = ego_state_se3.dynamic_state_se3
    assert dynamic_state_se3 is not None, "Ego dynamic state should be available for feature computation!"
    speed = float(np.linalg.norm(dynamic_state_se3.velocity_2d.array))

    return torch.tensor([speed], dtype=torch.float32)


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
