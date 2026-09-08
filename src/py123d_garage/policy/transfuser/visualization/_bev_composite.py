"""The BEV composite: LiDAR raster with semantic overlay, boxes, trajectories and target points."""

from __future__ import annotations

from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import MapLayer
from py123d.geometry.utils.bounding_box_utils import (
    bbse2_array_to_corners_array,
)
from py123d.visualization.color.default import (
    BOX_DETECTION_CONFIG,
    CENTERLINE_CONFIG,
    MAP_SURFACE_CONFIG,
)

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    BoundingBoxIndex,
)
from py123d_garage.policy.transfuser.visualization._drawing_primitives import (
    blend_semantic_overlay,
    draw_polyline,
    draw_target_point,
    draw_trajectory,
    to_display,
    to_pixel,
)

_SEMANTIC_OVERLAY_ALPHA = 0.4


def semantic_map_to_rgb(
    semantic_map: jt.Int64[npt.NDArray[np.int64], "height width"],
    config: TransfuserConfig,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Converts the BEV semantic raster to RGB using the py123d color palette.

    Args:
        semantic_map: semantic class raster
        config: global config dataclass of TransFuser

    Returns:
        RGB image (raster orientation)
    """
    height, width = semantic_map.shape[:2]
    rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    for label in range(1, config.bev_semantic_config.num_bev_semantic_classes):
        entity_type, layers = config.bev_semantic_config.selected_bev_semantic_classes[label]
        first_layer = layers[0]  # take color of first element

        if entity_type == "linestring":
            color = CENTERLINE_CONFIG.line_color
        elif first_layer in BOX_DETECTION_CONFIG:
            color = BOX_DETECTION_CONFIG[first_layer].fill_color
        else:
            color = MAP_SURFACE_CONFIG[cast(MapLayer, first_layer)].fill_color

        rgb_map[semantic_map == label] = color.rgb
    return rgb_map


def box_class_color(
    class_index: int,
    config: TransfuserConfig,
) -> tuple[int, int, int]:
    """Color of a detection class, from the py123d palette of its first box label."""
    box_labels = config.box_detection_config.detection_classes[class_index]
    return BOX_DETECTION_CONFIG[box_labels[0]].fill_color.rgb


def render_bev(
    lidar_map: jt.Float32[npt.NDArray[np.float32], "height width"] | None,
    semantic_map: jt.Int64[npt.NDArray[np.int64], "height width"] | None,
    boxes: jt.Float32[npt.NDArray[np.float32], "num_boxes 8"] | None,
    trajectories: list[tuple[npt.NDArray[np.float32], TrajectoryStyleConfig]],
    target_points: jt.Float32[npt.NDArray[np.float32], "num_points 2"],
    config: TransfuserConfig,
    route: jt.Float32[npt.NDArray[np.float32], "num_route_points 2"] | None = None,
    ego_box: jt.Float32[npt.NDArray[np.float32], " 5"] | None = None,
) -> jt.UInt8[npt.NDArray[np.uint8], "display_height display_width 3"]:
    """
    Composes the BEV view in display orientation (x forward up).

    LiDAR forms the base image, the semantic raster is alpha-blended on top,
    then the route, boxes, ego, trajectories and target points are drawn as
    overlays.

    Args:
        lidar_map: LiDAR histogram raster [H, W], or None for a blank base
        semantic_map: BEV semantic class raster [H, W], or None
        boxes: decoded boxes [K, 8] in the vehicle system, or None
        trajectories: (SE2 poses [N, >=2], style) pairs, drawn in order
        target_points: navigation target points [num_target_points, 2]
        config: global config dataclass of TransFuser
        route: ego-frame route polyline [N, >=2], or None to draw no route
        ego_box: ego bounding box (x, y, yaw, length, width) in the ego frame,
            or None to draw no ego

    Returns:
        RGB image of the composed BEV
    """
    display_shape = (
        config.lidar_config.bev_width_pixel,
        config.lidar_config.bev_height_pixel,
    )
    if lidar_map is None:
        image = np.full((*display_shape, 3), 255, dtype=np.uint8)
    else:
        lidar_display = (to_display(lidar_map) * 255).astype(np.uint8)
        image = 255 - lidar_display[..., None].repeat(3, axis=-1)
    image = np.ascontiguousarray(image)

    if semantic_map is not None:
        semantic_display = to_display(semantic_map_to_rgb(semantic_map, config))
        class_display = to_display(semantic_map)
        image = blend_semantic_overlay(
            image,
            np.ascontiguousarray(semantic_display),
            class_display > 0,
            _SEMANTIC_OVERLAY_ALPHA,
        )

    visualization_config = config.visualization_config
    if route is not None:
        draw_polyline(
            image,
            route,
            visualization_config.route_color_rgb,
            config,
            visualization_config.route_thickness_pixel,
        )

    if boxes is not None and len(boxes):
        # (x, y, yaw, length, width), the layout BoundingBoxSE2Index defines.
        boxes_se2 = boxes[
            :,
            [
                BoundingBoxIndex.X,
                BoundingBoxIndex.Y,
                BoundingBoxIndex.YAW,
                BoundingBoxIndex.W,
                BoundingBoxIndex.H,
            ],
        ].astype(np.float64)
        corners_per_box = bbse2_array_to_corners_array(boxes_se2)
        for box, corners in zip(boxes, corners_per_box, strict=True):
            pixels = np.array(
                [to_pixel(x, y, config) for x, y in corners],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(
                image,
                [pixels],
                isClosed=True,
                color=box_class_color(int(box[BoundingBoxIndex.CLASS]), config),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    if ego_box is not None:
        ego_corners = bbse2_array_to_corners_array(
            ego_box[None].astype(np.float64),
        )[0]
        pixels = np.array(
            [to_pixel(x, y, config) for x, y in ego_corners],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(
            image,
            [pixels],
            isClosed=True,
            color=visualization_config.ego_box_color_rgb,
            thickness=visualization_config.ego_box_thickness_pixel,
            lineType=cv2.LINE_AA,
        )

    for poses, trajectory_style in trajectories:
        draw_trajectory(image, poses, trajectory_style, config)

    for x, y in target_points:
        draw_target_point(image, float(x), float(y), config)

    return image
