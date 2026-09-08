"""Low-level drawing primitives shared by the TransFuser visualization views."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast, overload

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig
from py123d_garage.datatypes.trajectory import derive_yaw_from_positions

TRAJECTORY_HEADING_ARROW_M = 3.0


@overload
def to_display(raster: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...
@overload
def to_display(
    raster: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]: ...
@overload
def to_display(raster: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]: ...


def to_display(raster: Any) -> Any:
    """Raster [row=y, col=x] → display image with x (forward) up and y (left) on the left."""
    return np.transpose(raster, (1, 0, 2))[::-1, ::-1] if raster.ndim == 3 else raster.T[::-1, ::-1]


def to_pixel(x: float, y: float, config: TransfuserConfig) -> tuple[int, int]:
    """Ego-frame (x, y) in meters → display (column, row)."""
    col = (
        config.lidar_config.bev_height_pixel
        - 1
        - int(
            (y - config.lidar_config.bev_min_y_m) * config.lidar_config.bev_pixels_per_meter,
        )
    )
    row = (
        config.lidar_config.bev_width_pixel
        - 1
        - int(
            (x - config.lidar_config.bev_min_x_m) * config.lidar_config.bev_pixels_per_meter,
        )
    )
    return col, row


def blend_semantic_overlay(
    base: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    overlay: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    mask: jt.Bool[npt.NDArray[np.bool_], "height width"],
    alpha: float,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """Alpha-blends the overlay onto the base image where the mask is set."""
    blended = base.astype(np.float32)
    blended[mask] = alpha * overlay[mask].astype(np.float32) + (1.0 - alpha) * blended[mask]
    return blended.astype(np.uint8)


def draw_polyline(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    points: jt.Float32[npt.NDArray[np.float32], "num_points dims"],
    color: Sequence[int],
    config: TransfuserConfig,
    thickness: int,
) -> None:
    """Draws an ego-frame polyline as a connected line; cv2 clips off-image parts."""
    if len(points) < 2:
        return
    pixels = np.array(
        [to_pixel(float(x), float(y), config) for x, y in points[:, :2]],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(
        image,
        [pixels],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_trajectory(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    poses: jt.Float32[npt.NDArray[np.float32], "num_poses dims"],
    trajectory_style: TrajectoryStyleConfig,
    config: TransfuserConfig,
) -> None:
    """
    Draws an ego-frame trajectory as a connected line, a dot per pose and a heading arrow.

    The arrow shows the yaw the benchmarks will use: the trajectory carries positions only,
    so the heading is derived from the path exactly as TrajectoryXY.to_se2 derives it.
    """
    draw_polyline(image, poses, trajectory_style.color_rgb, config, trajectory_style.line_thickness_pixel)
    for x, y in poses[:, :2]:
        cv2.circle(
            image,
            to_pixel(float(x), float(y), config),
            trajectory_style.pose_radius_pixel,
            trajectory_style.color_rgb,
            -1,
            lineType=cv2.LINE_AA,
        )
    draw_trajectory_headings(image, poses, trajectory_style.color_rgb, config)


def draw_trajectory_headings(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    poses: jt.Float32[npt.NDArray[np.float32], "num_poses dims"],
    color: Sequence[int],
    config: TransfuserConfig,
) -> None:
    """Draws one arrow per pose along its heading, derived from the path when absent."""
    if len(poses) == 0:
        return
    if poses.shape[-1] > 2:
        yaw_wrapped_rad = poses[:, 2].astype(np.float64)
    else:
        yaw_wrapped_rad = cast(
            npt.NDArray[np.float64],
            derive_yaw_from_positions(
                torch.from_numpy(poses[:, :2].astype(np.float64)),  # pyright: ignore[reportUnknownMemberType]
            )
            .numpy()  # pyright: ignore[reportUnknownMemberType]
            .astype(np.float64),
        )

    for (x, y), yaw in zip(poses[:, :2], yaw_wrapped_rad, strict=True):
        tip_x = float(x) + TRAJECTORY_HEADING_ARROW_M * math.cos(float(yaw))
        tip_y = float(y) + TRAJECTORY_HEADING_ARROW_M * math.sin(float(yaw))
        cv2.arrowedLine(
            image,
            to_pixel(float(x), float(y), config),
            to_pixel(tip_x, tip_y, config),
            color,
            1,
            line_type=cv2.LINE_AA,
            tipLength=0.35,
        )


def draw_target_point(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    x: float,
    y: float,
    config: TransfuserConfig,
) -> None:
    """
    Draws a target point as a filled circle; points outside the raster are clamped to its edge.

    An off-raster point is projected onto the BEV boundary along the ego→point
    ray, drawn as an outline and annotated with its true distance.
    """
    if np.isnan(x) or np.isnan(y):
        return
    lidar_config = config.lidar_config
    scale_candidates = [1.0]
    if x > lidar_config.bev_max_x_m:
        scale_candidates.append(lidar_config.bev_max_x_m / x)
    if x < lidar_config.bev_min_x_m:
        scale_candidates.append(lidar_config.bev_min_x_m / x)
    if y > lidar_config.bev_max_y_m:
        scale_candidates.append(lidar_config.bev_max_y_m / y)
    if y < lidar_config.bev_min_y_m:
        scale_candidates.append(lidar_config.bev_min_y_m / y)
    scale = min(scale_candidates)
    inside = scale == 1.0

    visualization_config = config.visualization_config
    col, row = to_pixel(x * scale, y * scale, config)
    cv2.circle(
        image,
        (col, row),
        visualization_config.target_point_radius_pixel,
        visualization_config.target_point_color_rgb,
        -1 if inside else 1,
        lineType=cv2.LINE_AA,
    )
    if not inside:
        distance = float(np.hypot(x, y))
        text_col = int(np.clip(col, 2, image.shape[1] - 45))
        text_row = int(np.clip(row, 14, image.shape[0] - 4))
        cv2.putText(
            image,
            f"{distance:.0f}m",
            (text_col, text_row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            visualization_config.target_point_color_rgb,
            1,
            lineType=cv2.LINE_AA,
        )
