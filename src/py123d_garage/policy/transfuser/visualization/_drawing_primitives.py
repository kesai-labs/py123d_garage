# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Low-level drawing primitives shared by the TransFuser visualization views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig


@overload
def to_display(raster: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...
@overload
def to_display(
    raster: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]: ...
@overload
def to_display(raster: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]: ...


def to_display(raster: Any) -> Any:
    """
    Turns a raster indexed [row=y, col=x] into a display image with x (forward) up and y (left) on the left.

    Args:
        raster: the raster, optionally with a trailing channel axis.

    Returns:
        the raster in display orientation.
    """
    return np.transpose(raster, (1, 0, 2))[::-1, ::-1] if raster.ndim == 3 else raster.T[::-1, ::-1]


def to_pixel(x: float, y: float, config: TransfuserConfig) -> tuple[int, int]:
    """
    Maps an ego-frame position to display pixel coordinates.

    Args:
        x: forward position in meters.
        y: leftward position in meters.
        config: TransFuser config whose BEV grid sets the mapping.

    Returns:
        the display (column, row).
    """
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
    """
    Alpha-blends the overlay onto the base image where the mask is set.

    Args:
        base: the image to blend onto.
        overlay: the image to blend in.
        mask: the pixels to blend.
        alpha: the overlay's weight.

    Returns:
        the blended image.
    """
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
    """
    Draws an ego-frame polyline as a connected line in place; cv2 clips off-image parts.

    Args:
        image: the display image to draw on.
        points: the polyline points, (x, y) in the first two columns.
        color: RGB line color.
        config: TransFuser config whose BEV grid sets the mapping.
        thickness: line thickness in pixels.
    """
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
    Draws an ego-frame trajectory as a dot per pose in place.

    Args:
        image: the display image to draw on.
        poses: the trajectory poses, (x, y) in the first two columns.
        trajectory_style: dot color and radius.
        config: TransFuser config whose BEV grid sets the mapping.
    """
    for x, y in poses[:, :2]:
        cv2.circle(
            image,
            to_pixel(float(x), float(y), config),
            trajectory_style.pose_radius_pixel,
            trajectory_style.color_rgb,
            -1,
            lineType=cv2.LINE_AA,
        )


def draw_path(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    points: jt.Float32[npt.NDArray[np.float32], "num_points dims"],
    path_style: TrajectoryStyleConfig,
    config: TransfuserConfig,
) -> None:
    """
    Draws ego-frame path points in place as a connected line with a dot per point.

    Args:
        image: the display image to draw on.
        points: the path points, (x, y) in the first two columns.
        path_style: line and dot color, thickness and radius.
        config: TransFuser config whose BEV grid sets the mapping.
    """
    draw_polyline(image, points, path_style.color_rgb, config, path_style.line_thickness_pixel)
    for x, y in points[:, :2]:
        cv2.circle(
            image,
            to_pixel(float(x), float(y), config),
            path_style.pose_radius_pixel,
            path_style.color_rgb,
            -1,
            lineType=cv2.LINE_AA,
        )


def draw_target_point(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    x: float,
    y: float,
    config: TransfuserConfig,
) -> None:
    """
    Draws a target point as a crossed circle, or off the raster as a circle on its edge with its distance.

    Args:
        image: the display image to draw on.
        x: forward position in meters; NaN draws nothing.
        y: leftward position in meters; NaN draws nothing.
        config: TransFuser config whose BEV grid and colors are used.
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
        1,
        lineType=cv2.LINE_AA,
    )
    if inside:
        cv2.drawMarker(
            image,
            (col, row),
            visualization_config.target_point_color_rgb,
            cv2.MARKER_CROSS,
            2 * visualization_config.target_point_radius_pixel,
            1,
            cv2.LINE_AA,
        )
    else:
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
