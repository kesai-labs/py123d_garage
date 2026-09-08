"""Ego-frame overlays read from a scene: the ego box, the ground-truth boxes, the route."""

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.geometry_index import BoundingBoxSE2Index
from py123d.geometry.transform import abs_to_rel_se2_array

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.labels import (
    extract_relative_bounding_boxes,
)
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    BoundingBoxIndex,
)

_ROUTE_WINDOW_MARGIN_M = 5.0


def ego_box_of(
    ego_state: EgoStateSE3,
) -> jt.Float32[npt.NDArray[np.float32], " 5"]:
    """
    The ego bounding box (x, y, yaw, length, width) in its own rear-axle frame.

    Args:
        ego_state: the anchor's ego state

    Returns:
        the ego box array
    """
    ego_box = ego_state.bounding_box_se2.array.copy()
    se2_slice = BoundingBoxSE2Index.SE2
    ego_box[se2_slice] = abs_to_rel_se2_array(
        origin=ego_state.rear_axle_se2,
        pose_se2_array=ego_box[None, se2_slice],
    )[0]
    return ego_box.astype(np.float32)


def ground_truth_boxes_of(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float32[npt.NDArray[np.float32], "num_boxes 8"] | None:
    """
    The anchor's ground-truth boxes in the decoded BoundingBoxIndex layout.

    Args:
        scene_api: the anchor scene
        config: the policy config whose detection classes select and color boxes

    Returns:
        boxes, or None when the scene carries no box detections
    """
    if scene_api.get_box_detections_se3_metadata() is None:
        return None
    ego_state = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state is not None, "Ego state should be available at the anchor!"
    bbse2_array, bbse2_labels, bbse2_speeds = extract_relative_bounding_boxes(
        scene_api,
        ego_state,
    )
    class_by_label = {
        label: class_index
        for class_index, labels in config.box_detection_config.detection_classes.items()
        for label in labels
    }
    rows: list[npt.NDArray[np.float32]] = []
    for bbse2, label, speed in zip(
        bbse2_array,
        bbse2_labels,
        bbse2_speeds,
        strict=True,
    ):
        class_index = class_by_label.get(label)
        if class_index is None:
            continue
        row = np.zeros(len(BoundingBoxIndex), dtype=np.float32)
        row[BoundingBoxIndex.X] = bbse2[BoundingBoxSE2Index.X]
        row[BoundingBoxIndex.Y] = bbse2[BoundingBoxSE2Index.Y]
        row[BoundingBoxIndex.YAW] = bbse2[BoundingBoxSE2Index.YAW]
        row[BoundingBoxIndex.W] = bbse2[BoundingBoxSE2Index.LENGTH]
        row[BoundingBoxIndex.H] = bbse2[BoundingBoxSE2Index.WIDTH]
        row[BoundingBoxIndex.VELOCITY] = speed
        row[BoundingBoxIndex.CLASS] = class_index
        row[BoundingBoxIndex.SCORE] = 1.0
        rows.append(row)
    if not rows:
        return None
    return np.stack(rows)


def ego_frame_route_of(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float32[npt.NDArray[np.float32], "num_route_points 2"] | None:
    """
    The route polyline around the anchor, in its rear-axle frame.

    Args:
        scene_api: the anchor scene
        config: the policy config whose BEV extent bounds the window

    Returns:
        route points, or None when the log carries no route
    """
    route = scene_api.get_route()
    progress_m = scene_api.get_route_progress_at_iteration(0)
    if route is None or progress_m is None:
        return None
    _, polyline_arc_m, polyline_xyz = route

    lidar_config = config.lidar_config
    radius_m = (
        float(
            np.hypot(
                max(
                    abs(lidar_config.bev_min_x_m),
                    lidar_config.bev_max_x_m,
                ),
                max(
                    abs(lidar_config.bev_min_y_m),
                    lidar_config.bev_max_y_m,
                ),
            ),
        )
        + _ROUTE_WINDOW_MARGIN_M
    )
    low, high = np.searchsorted(
        polyline_arc_m,
        [progress_m - radius_m, progress_m + radius_m],
    )
    window_xyz = polyline_xyz[max(int(low) - 1, 0) : int(high) + 1]
    if len(window_xyz) < 2:
        return None

    ego_state = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state is not None, "Ego state should be available at the anchor!"
    pose_se2_array = np.concatenate(
        [window_xyz[:, :2], np.zeros((len(window_xyz), 1))],
        axis=1,
    )
    relative = abs_to_rel_se2_array(
        origin=ego_state.rear_axle_se2,
        pose_se2_array=pose_se2_array,
    )
    return relative[:, :2].astype(np.float32)
