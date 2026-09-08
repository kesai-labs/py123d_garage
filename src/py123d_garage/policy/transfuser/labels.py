from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import MapAPI, SceneAPI
from py123d.datatypes import (
    BaseMapSurfaceObject,
    DefaultBoxDetectionLabel,
    DepthCameraMetadata,
    EgoStateSE3,
    MapLayer,
    SegmentationCameraMetadata,
)
from py123d.datatypes.map_objects.map_objects import Lane
from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
)
from py123d.geometry import BoundingBoxSE2Index, PoseSE2
from py123d.geometry.transform import abs_to_rel_se2_array
from py123d.geometry.utils.bounding_box_utils import (
    bbse2_array_to_corners_array,
)
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from py123d_garage.api.abstract_policy_tensors import AbstractLabels
from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.datatypes.numerics import PositiveInt
from py123d_garage.policy.transfuser.network.bev_semantic_decoder import (
    BevSemanticLabels,
)
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    CenterNetLabels,
    gaussian_radius,
    gen_gaussian_target,
)
from py123d_garage.py123d_help.scene_readers import sample_ego_se2, sample_ego_xy

_CENTER_NET_GAUSSIAN_MIN_OVERLAP: float = 0.3


@dataclass(frozen=True)
class TransfuserLabels(AbstractLabels):
    """TransFuser's supervision tensors; fields of disabled heads are None."""

    trajectory: jt.Float[torch.Tensor, "*batch poses columns"] | None = None
    semantic: jt.Int[torch.Tensor, "*batch semantic_height semantic_width"] | None = None
    depth: jt.Float[torch.Tensor, "*batch semantic_height semantic_width"] | None = None
    bev_semantic: BevSemanticLabels | None = None
    center_net: CenterNetLabels | None = None


def build_trajectory_label(
    scene_api: SceneAPI,
    num_steps: PositiveInt,
    interval_us: PositiveInt,
    predict_yaw: bool,
) -> jt.Float[torch.Tensor, "poses columns"]:
    """
    The ego's future relative positions, with the logged yaw when predict_yaw.

    Args:
        scene_api: scene interface anchored at the current frame.
        num_steps: how many poses the label holds.
        interval_us: spacing between consecutive poses.
        predict_yaw: whether the yaw column is part of the label.

    Returns:
        the future ego trajectory as (x, y), plus yaw when predict_yaw.
    """
    if predict_yaw:
        trajectory_se2 = sample_ego_se2(
            scene_api=scene_api,
            num_steps=num_steps,
            interval_us=interval_us,
            relative_to_anchor=True,
        )
        return torch.tensor(trajectory_se2.pose_se2_array, dtype=torch.float32)
    trajectory = sample_ego_xy(
        scene_api=scene_api,
        num_steps=num_steps,
        interval_us=interval_us,
        relative_to_anchor=True,
    )
    return torch.tensor(trajectory.position_xy_array, dtype=torch.float32)


@functools.cache
def _semantic_to_default_lookup(
    label_class: type[CameraSegmentationLabel],
) -> jt.Int16[npt.NDArray[np.int16], " 256"]:
    """Native class id to default-taxonomy value; ids without a member hold -1."""
    lookup = np.full(256, -1, dtype=np.int16)
    for member in label_class:
        lookup[int(member.value)] = int(member.to_default().value)
    return lookup


def _map_semantic_to_default(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width"],
    label_class: type[CameraSegmentationLabel],
) -> jt.UInt8[npt.NDArray[np.uint8], "height width"]:
    """
    Maps a native semantic frame to the default taxonomy.

    Args:
        image: the segmentation camera's frame of native class ids.
        label_class: the dataset's segmentation label enum.

    Returns:
        the frame with every pixel mapped to its default-taxonomy value.

    Raises:
        ValueError: if the frame contains a class id label_class does not name.
    """
    mapped = _semantic_to_default_lookup(label_class)[image]
    unknown_ids = np.unique(image[mapped < 0])
    if unknown_ids.size:
        raise ValueError(
            f"semantic frame contains class ids {unknown_ids.tolist()} with no {label_class.__name__} member",
        )
    return mapped.astype(np.uint8)


def build_perspective_labels(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> tuple[
    jt.Int64[torch.Tensor, "semantic_height semantic_width"] | None,
    jt.Float32[torch.Tensor, "semantic_height semantic_width"] | None,
]:
    """
    The perspective semantic and depth labels, config-gated like the heads.

    Both mirror the camera feature's stitching, resized with nearest
    interpolation so class ids and quantized depth codes survive intact.
    Semantic classes are the dataset's converted to py123d's default taxonomy;
    depth is metric meters, like the predictions.
    """
    perspective = config.perspective_config
    width = config.camera_config.image_width // perspective.perspective_downsample_factor
    height = config.camera_config.image_height // perspective.perspective_downsample_factor
    input_cameras = config.camera_config.input_cameras[scene_api.scene_metadata.dataset]

    semantic: torch.Tensor | None = None
    if perspective.use_semantic:
        semantic_images: list[npt.NDArray[np.uint8]] = []
        for camera_id in input_cameras:
            camera = scene_api.get_camera_semantic_at_iteration(0, camera_id)
            assert camera is not None, f"Semantic camera {camera_id.name} should be available for label computation!"
            metadata = camera.metadata
            assert isinstance(metadata, SegmentationCameraMetadata)
            semantic_images.append(
                _map_semantic_to_default(
                    camera.image,
                    metadata.segmentation_label_class,
                ),
            )
        semantic_resized = cv2.resize(
            np.concatenate(semantic_images, axis=1),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        semantic = torch.as_tensor(semantic_resized, dtype=torch.int64)

    depth: torch.Tensor | None = None
    if perspective.use_depth:
        depth_images: list[npt.NDArray[np.float32]] = []
        for camera_id in input_cameras:
            camera = scene_api.get_camera_depth_at_iteration(0, camera_id)
            assert camera is not None, f"Depth camera {camera_id.name} should be available for label computation!"
            metadata = camera.metadata
            assert isinstance(metadata, DepthCameraMetadata)
            assert metadata.max_depth == perspective.depth_max_m, (
                f"Depth far plane {metadata.max_depth} of {camera_id.name} "
                f"differs from depth_max_m {perspective.depth_max_m}"
            )
            # The cache codec re-quantizes to 8-bit linear codes in [0, max_depth].
            assert metadata.depth_bits == 8, (
                f"Depth of {camera_id.name} is {metadata.depth_bits}-bit; the cache codec stores 8-bit codes"
            )
            assert metadata.depth_transform == "linear", (
                f"Depth transform {metadata.depth_transform!r} of {camera_id.name} is not linear"
            )
            assert metadata.min_depth == 0.0, f"Depth min_depth {metadata.min_depth} of {camera_id.name} must be 0"
            assert not metadata.has_invalid, (
                f"Depth of {camera_id.name} has an invalid sentinel; the cache codec cannot store it"
            )
            depth_images.append(metadata.decode_depth(camera.image))
        depth_resized = cv2.resize(
            np.concatenate(depth_images, axis=1),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = torch.as_tensor(depth_resized, dtype=torch.float32)

    return semantic, depth


def build_bev_labels(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> tuple[BevSemanticLabels | None, CenterNetLabels | None]:
    """The BEV-semantic and CenterNet labels of one scene, config-gated like the heads."""
    ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for label computation!"
    bbse2_array, bbse2_labels, bbse2_speeds = extract_relative_bounding_boxes(
        scene_api,
        ego_state_se3,
    )

    bev_semantic: BevSemanticLabels | None = None
    if config.bev_semantic_config.use_bev_semantic:
        map_api = scene_api.get_map_api()
        bev_semantic = BevSemanticLabels(
            bev_semantic=_compute_bev_semantic_map(
                map_api,
                ego_state_se3,
                bbse2_array,
                bbse2_labels,
                config,
            ),
            bev_semantic_has_map=torch.tensor(map_api is not None),
        )

    center_net: CenterNetLabels | None = None
    if config.box_detection_config.detect_boxes:
        center_net = _compute_center_net_labels(
            bbse2_array,
            bbse2_labels,
            bbse2_speeds,
            config,
        )

    return bev_semantic, center_net


def _compute_center_net_labels(
    bbse2_array: npt.NDArray[np.float64],
    bbse2_labels: list[DefaultBoxDetectionLabel],
    bbse2_speeds: npt.NDArray[np.float64],
    config: TransfuserConfig,
) -> CenterNetLabels:
    """Rasterizes the boxes into CenterNet label maps on the down-sampled BEV raster."""
    down = config.bev_semantic_config.bev_downsample_factor
    height = config.lidar_config.bev_height_pixel // down
    width = config.lidar_config.bev_width_pixel // down
    pixels_per_meter = config.lidar_config.bev_pixels_per_meter / down

    label_to_class = {
        label: class_idx
        for class_idx, labels in config.box_detection_config.detection_classes.items()
        for label in labels
    }

    heatmap = np.zeros(
        (config.box_detection_config.num_box_classes, height, width),
        dtype=np.float32,
    )
    wh = np.zeros((2, height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    yaw_class = np.zeros((height, width), dtype=np.int64)
    yaw_res = np.zeros((1, height, width), dtype=np.float32)
    velocity = np.zeros((1, height, width), dtype=np.float32)
    pixel_weight = np.zeros((2, height, width), dtype=np.float32)
    num_boxes = 0

    for box, label, speed in zip(
        bbse2_array,
        bbse2_labels,
        bbse2_speeds,
        strict=True,
    ):
        class_idx = label_to_class.get(label)
        if class_idx is None:
            continue
        col = (box[BoundingBoxSE2Index.X] - config.lidar_config.bev_min_x_m) * pixels_per_meter
        row = (box[BoundingBoxSE2Index.Y] - config.lidar_config.bev_min_y_m) * pixels_per_meter
        # int() truncates toward zero, mapping col/row in (-1, 0) onto index 0.
        col_idx, row_idx = math.floor(col), math.floor(row)
        if not (0 <= col_idx < width and 0 <= row_idx < height):
            continue

        length_px = box[BoundingBoxSE2Index.LENGTH] * pixels_per_meter
        width_px = box[BoundingBoxSE2Index.WIDTH] * pixels_per_meter
        radius = max(
            1,
            gaussian_radius(
                [width_px, length_px],
                _CENTER_NET_GAUSSIAN_MIN_OVERLAP,
            ),
        )
        gen_gaussian_target(heatmap[class_idx], [col_idx, row_idx], radius)

        wh[:, row_idx, col_idx] = (length_px, width_px)
        offset[:, row_idx, col_idx] = (col - col_idx, row - row_idx)
        bin_idx, bin_residual = _angle_to_class(
            float(box[BoundingBoxSE2Index.YAW]),
            config.box_detection_config.num_yaw_bins,
        )
        yaw_class[row_idx, col_idx] = bin_idx
        yaw_res[0, row_idx, col_idx] = bin_residual
        velocity[0, row_idx, col_idx] = speed
        pixel_weight[:, row_idx, col_idx] = 1.0
        num_boxes += 1

    return CenterNetLabels(
        center_net_heatmap=torch.as_tensor(heatmap),
        center_net_wh=torch.as_tensor(wh),
        center_net_offset=torch.as_tensor(offset),
        center_net_yaw_class=torch.as_tensor(yaw_class),
        center_net_yaw_res=torch.as_tensor(yaw_res),
        center_net_pixel_weight=torch.as_tensor(pixel_weight),
        center_net_avg_factor=torch.tensor(num_boxes, dtype=torch.float32),
        center_net_velocity=torch.as_tensor(velocity) if config.box_detection_config.predict_box_velocity else None,
    )


def _compute_bev_semantic_map(
    map_api: MapAPI | None,
    ego_state_se3: EgoStateSE3,
    bbse2_array: npt.NDArray[np.float64],
    bbse2_labels: list[DefaultBoxDetectionLabel],
    config: TransfuserConfig,
) -> jt.Int64[torch.Tensor, "height width"]:
    """Computes the BEV semantic raster; without a map only the box classes are drawn."""
    bev_semantic_map = np.zeros(
        (
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
        dtype=np.int64,
    )
    for label, (
        entity_type,
        layers,
    ) in config.bev_semantic_config.selected_bev_semantic_classes.items():
        if entity_type == "polygon":
            if map_api is None:
                continue
            entity_mask = _compute_map_polygon_mask(
                map_api,
                ego_state_se3,
                cast("list[MapLayer]", layers),
                config,
            )
        elif entity_type == "linestring":
            if map_api is None:
                continue
            entity_mask = _compute_map_linestring_mask(
                map_api,
                ego_state_se3,
                cast("list[MapLayer]", layers),
                config,
            )
        else:
            entity_mask = _compute_box_mask(
                bbse2_array,
                bbse2_labels,
                cast("list[DefaultBoxDetectionLabel]", layers),
                config,
            )
        bev_semantic_map[entity_mask] = label

    return torch.as_tensor(bev_semantic_map)


def _bev_query_radius(config: TransfuserConfig) -> float:
    """Radius covering the BEV grid, from the largest boundary extent."""
    return max(
        abs(config.lidar_config.bev_min_x_m),
        config.lidar_config.bev_max_x_m,
        abs(config.lidar_config.bev_min_y_m),
        config.lidar_config.bev_max_y_m,
    )


def _compute_map_polygon_mask(
    map_api: MapAPI,
    ego_state_se3: EgoStateSE3,
    layers: list[MapLayer],
    config: TransfuserConfig,
) -> npt.NDArray[np.bool_]:
    """Binary BEV mask of the given polygon map layers."""
    ego_pose_se2 = ego_state_se3.rear_axle_se2

    map_object_dict = map_api.get_map_objects_in_radius(
        point=ego_state_se3.rear_axle_2d,
        radius=_bev_query_radius(config),
        layers=layers,  # type: ignore[arg-type]
    )
    mask = np.zeros(
        (
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
        dtype=np.uint8,
    )
    for layer in layers:
        for map_object in map_object_dict[layer]:
            assert isinstance(map_object, BaseMapSurfaceObject), "Map object should be polygon type!"
            polygon = cast(
                Polygon,
                _geometry_local_coords(
                    map_object.shapely_polygon,
                    ego_pose_se2,
                ),
            )
            exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [_coords_to_pixel(exterior, config)], color=255)  # type: ignore[arg-type]
    return mask > 0


def _compute_map_linestring_mask(
    map_api: MapAPI,
    ego_state_se3: EgoStateSE3,
    layers: list[MapLayer],
    config: TransfuserConfig,
) -> npt.NDArray[np.bool_]:
    """Binary BEV mask of the given linestring map layers."""
    ego_pose_se2 = ego_state_se3.rear_axle_se2
    map_object_dict = map_api.get_map_objects_in_radius(
        point=ego_state_se3.rear_axle_2d,
        radius=_bev_query_radius(config),
        layers=layers,  # type: ignore[arg-type]
    )
    mask = np.zeros(
        (
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
        dtype=np.uint8,
    )
    for layer in layers:
        for map_object in map_object_dict[layer]:
            linestring: LineString = _geometry_local_coords(
                cast(Lane, map_object).centerline_2d.linestring,
                ego_pose_se2,
            )  # type: ignore[assignment]
            points = np.array(linestring.coords).reshape((-1, 1, 2))
            cv2.polylines(
                mask,
                [_coords_to_pixel(points, config)],
                isClosed=False,
                color=255,
                thickness=2,
            )  # type: ignore[arg-type]
    return mask > 0


def _compute_box_mask(
    bbse2_array: npt.NDArray[np.float64],
    bbse2_labels: list[DefaultBoxDetectionLabel],
    layers: list[DefaultBoxDetectionLabel],
    config: TransfuserConfig,
) -> npt.NDArray[np.bool_]:
    """Binary BEV mask of the boxes with the given labels; pedestrians are scaled up to stay visible."""
    mask = np.zeros(
        (
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
        dtype=np.uint8,
    )

    layer_set = set(layers)
    keep = np.fromiter(
        (label in layer_set for label in bbse2_labels),
        dtype=bool,
        count=len(bbse2_labels),
    )
    if keep.any():
        boxes = bbse2_array[keep].copy()
        kept_labels = [label for label in bbse2_labels if label in layer_set]
        for box, label in zip(boxes, kept_labels, strict=True):
            if label == DefaultBoxDetectionLabel.PERSON:
                box[BoundingBoxSE2Index.LENGTH] = max(
                    box[BoundingBoxSE2Index.LENGTH] * config.bev_semantic_config.pedestrian_bev_extent_scale,
                    config.bev_semantic_config.pedestrian_bev_min_extent_m,
                )
                box[BoundingBoxSE2Index.WIDTH] = max(
                    box[BoundingBoxSE2Index.WIDTH] * config.bev_semantic_config.pedestrian_bev_extent_scale,
                    config.bev_semantic_config.pedestrian_bev_min_extent_m,
                )
        for box_corners in bbse2_array_to_corners_array(boxes):
            cv2.fillPoly(
                mask,
                [_coords_to_pixel(box_corners.reshape((-1, 1, 2)), config)],
                color=255,
            )  # type: ignore[arg-type]
    return mask > 0


def extract_relative_bounding_boxes(
    scene_api: SceneAPI,
    ego_state_se3: EgoStateSE3,
) -> tuple[
    npt.NDArray[np.float64],
    list[DefaultBoxDetectionLabel],
    npt.NDArray[np.float64],
]:
    """The anchor tick's box detections as ego-relative SE2 arrays with labels and speeds."""
    box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(0)
    assert box_detections_se3 is not None, "Box detections should be available for label computation!"
    # The CARLA export contains boxes with non-positive extents; every consumer
    # downstream (gaussian radius, BEV rasterization) needs real areas.
    box_detections_se2 = [
        box_detection
        for box_detection in box_detections_se3.box_detections_se2
        if box_detection.bounding_box_se2.array[BoundingBoxSE2Index.LENGTH] > 0
        and box_detection.bounding_box_se2.array[BoundingBoxSE2Index.WIDTH] > 0
    ]

    bbse2_array = np.zeros(
        (len(box_detections_se2), len(BoundingBoxSE2Index)),
        dtype=np.float64,
    )
    bbse2_labels: list[DefaultBoxDetectionLabel] = []
    bbse2_speeds = np.zeros(len(box_detections_se2), dtype=np.float64)
    for box_idx, box_detection in enumerate(box_detections_se2):
        bbse2_array[box_idx] = box_detection.bounding_box_se2.array
        bbse2_labels.append(box_detection.attributes.default_label)
        velocity_2d = box_detection.velocity_2d
        if velocity_2d is not None:
            bbse2_speeds[box_idx] = float(np.linalg.norm(velocity_2d.array))

    se2_slice = BoundingBoxSE2Index.SE2
    bbse2_array[..., se2_slice] = abs_to_rel_se2_array(
        origin=ego_state_se3.rear_axle_se2,
        pose_se2_array=bbse2_array[..., se2_slice],
    )

    return bbse2_array, bbse2_labels, bbse2_speeds


def _geometry_local_coords(
    geometry: BaseGeometry,
    origin: PoseSE2,
) -> BaseGeometry:
    """Transforms a shapely geometry into the local coordinates of the origin pose."""
    cos, sin = np.cos(origin.yaw), np.sin(origin.yaw)
    translated_geometry = affinity.affine_transform(
        geometry,
        [1, 0, 0, 1, -origin.x, -origin.y],
    )
    return affinity.affine_transform(
        translated_geometry,
        [cos, sin, -sin, cos, 0, 0],
    )


def _coords_to_pixel(
    coords: npt.NDArray[np.float64],
    config: TransfuserConfig,
) -> npt.NDArray[np.int32]:
    """Transforms ego-frame (x, y) coordinates into (column, row) pixel indices."""
    pixel = np.empty_like(coords)
    pixel[..., 0] = (
        coords[..., 0] - config.lidar_config.bev_min_x_m
    ) * config.lidar_config.bev_pixels_per_meter  # column = x
    pixel[..., 1] = (
        coords[..., 1] - config.lidar_config.bev_min_y_m
    ) * config.lidar_config.bev_pixels_per_meter  # row = y
    return pixel.astype(np.int32)


def _angle_to_class(angle_rad: float, num_yaw_bins: int) -> tuple[int, float]:
    """Discretizes an angle into a direction bin and residual (inverse of class2angle)."""
    angle_per_class = 2.0 * np.pi / num_yaw_bins
    shifted = angle_rad % (2.0 * np.pi)
    bin_idx = int(np.floor((shifted + angle_per_class / 2.0) / angle_per_class)) % num_yaw_bins
    residual = shifted - bin_idx * angle_per_class
    residual = (residual + np.pi) % (2.0 * np.pi) - np.pi
    return bin_idx, float(residual)
