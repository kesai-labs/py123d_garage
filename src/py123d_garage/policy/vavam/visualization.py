"""VaVAM's rendered view: the context camera frame over a blank BEV sheet holding the trajectories."""

from __future__ import annotations

import math

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from py123d_garage.config.schema.policy.vavam_config import VavamVisualizationConfig
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig

_BORDER_PIXEL = 4
_LEGEND_ROW_PIXEL = 18


def render_view(
    camera_rgb: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    prediction_xy: jt.Float32[npt.NDArray[np.float32], "num_poses 2"],
    ground_truth_xy: jt.Float32[npt.NDArray[np.float32], "num_poses 2"] | None,
    target_points_xy: jt.Float32[npt.NDArray[np.float32], "num_points 2"],
    ego_box: jt.Float32[npt.NDArray[np.float32], " 5"],
    visualization_config: VavamVisualizationConfig,
) -> jt.UInt8[npt.NDArray[np.uint8], "canvas_height canvas_width 3"]:
    """
    Composes the camera frame above a BEV sheet with the ego box and the trajectories.

    Args:
        camera_rgb: the newest context frame the policy saw.
        prediction_xy: the predicted waypoints in the ego rear-axle frame.
        ground_truth_xy: the logged future on the same grid; None when the scene has no future.
        target_points_xy: the navigation target points in the ego frame.
        ego_box: the ego box (x, y, yaw, length, width) in the ego frame.
        visualization_config: the sheet geometry and the overlay styles.

    Returns:
        the composed RGB image.
    """
    config = visualization_config
    bev_height = round((config.bev_max_x_m - config.bev_min_x_m) * config.bev_pixels_per_meter)
    bev_width = round(2 * config.bev_half_width_m * config.bev_pixels_per_meter)
    bev = np.full((bev_height, bev_width, 3), 255, dtype=np.uint8)

    _draw_ego_box(bev, ego_box, config)
    legend: list[tuple[str, list[int]]] = [("prediction", config.prediction_trajectory_style.color_rgb)]
    if ground_truth_xy is not None:
        _draw_trajectory(bev, ground_truth_xy, config.ground_truth_trajectory_style, config)
        legend.append(("ground truth", config.ground_truth_trajectory_style.color_rgb))
    _draw_trajectory(bev, prediction_xy, config.prediction_trajectory_style, config)
    for x, y in target_points_xy:
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(
            bev,
            _to_pixel(float(x), float(y), config),
            config.target_point_radius_pixel,
            config.target_point_color_rgb,
            -1,
            lineType=cv2.LINE_AA,
        )
    legend.append(("target point", config.target_point_color_rgb))
    for row, (label, color_rgb) in enumerate(legend, start=1):
        cv2.putText(
            bev,
            label,
            (_BORDER_PIXEL, row * _LEGEND_ROW_PIXEL),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_rgb,
            1,
            lineType=cv2.LINE_AA,
        )

    width = camera_rgb.shape[1]
    scaled_bev = cv2.resize(bev, (width, round(width * bev_height / bev_width)))
    height = camera_rgb.shape[0] + scaled_bev.shape[0] + 3 * _BORDER_PIXEL
    canvas = np.full((height, width + 2 * _BORDER_PIXEL, 3), 255, dtype=np.uint8)
    canvas[_BORDER_PIXEL : _BORDER_PIXEL + camera_rgb.shape[0], _BORDER_PIXEL : _BORDER_PIXEL + width] = camera_rgb
    bev_row = 2 * _BORDER_PIXEL + camera_rgb.shape[0]
    canvas[bev_row : bev_row + scaled_bev.shape[0], _BORDER_PIXEL : _BORDER_PIXEL + width] = scaled_bev
    return canvas


def _to_pixel(x: float, y: float, config: VavamVisualizationConfig) -> tuple[int, int]:
    """Ego-frame (x, y) in meters → BEV (column, row), x (forward) up and y (left) on the left."""
    return (
        round((config.bev_half_width_m - y) * config.bev_pixels_per_meter),
        round((config.bev_max_x_m - x) * config.bev_pixels_per_meter),
    )


def _draw_trajectory(
    bev: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    poses_xy: jt.Float32[npt.NDArray[np.float32], "num_poses 2"],
    trajectory_style: TrajectoryStyleConfig,
    config: VavamVisualizationConfig,
) -> None:
    pixels = np.array([_to_pixel(float(x), float(y), config) for x, y in poses_xy], dtype=np.int32)
    cv2.polylines(
        bev,
        [pixels.reshape(-1, 1, 2)],
        isClosed=False,
        color=trajectory_style.color_rgb,
        thickness=trajectory_style.line_thickness_pixel,
        lineType=cv2.LINE_AA,
    )
    for column, row in pixels:
        cv2.circle(
            bev,
            (int(column), int(row)),
            trajectory_style.pose_radius_pixel,
            trajectory_style.color_rgb,
            -1,
            lineType=cv2.LINE_AA,
        )


def _draw_ego_box(
    bev: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    ego_box: jt.Float32[npt.NDArray[np.float32], " 5"],
    config: VavamVisualizationConfig,
) -> None:
    x, y, yaw, length, width = (float(value) for value in ego_box)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    corners = [
        _to_pixel(
            x + dx * cos_yaw - dy * sin_yaw,
            y + dx * sin_yaw + dy * cos_yaw,
            config,
        )
        for dx, dy in [
            (length / 2, width / 2),
            (length / 2, -width / 2),
            (-length / 2, -width / 2),
            (-length / 2, width / 2),
        ]
    ]
    cv2.polylines(
        bev,
        [np.array(corners, dtype=np.int32).reshape(-1, 1, 2)],
        isClosed=True,
        color=config.ego_box_color_rgb,
        thickness=config.ego_box_thickness_pixel,
        lineType=cv2.LINE_AA,
    )
