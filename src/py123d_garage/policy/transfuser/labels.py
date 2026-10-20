# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import MapAPI, SceneAPI
from py123d.datatypes import (
    BaseMapSurfaceObject,
    BoxDetectionsSE3,
    CameraID,
    DefaultBoxDetectionLabel,
    DepthCameraMetadata,
    EgoStateSE3,
    MapLayer,
    SegmentationCameraMetadata,
)
from py123d.datatypes.map_objects.map_objects import Lane
from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
    DefaultCameraSegmentationLabel,
)
from py123d.geometry import (
    BoundingBoxSE2Index,
    Point3D,
    Point3DIndex,
    PoseSE2,
)
from py123d.geometry.transform import abs_to_rel_se2_array
from py123d.geometry.utils.bounding_box_utils import (
    bbse2_array_to_corners_array,
)
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from py123d_garage.api.abstract_policy_tensors import AbstractLabels
from py123d_garage.config.schema.policy.transfuser_config import (
    PathConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import camera_frame_rows
from py123d_garage.policy.transfuser.network.decoder.bev_semantic_decoder import (
    BevSemanticLabels,
)
from py123d_garage.policy.transfuser.network.decoder.center_net_decoder import (
    CenterNetLabels,
    gaussian_radius,
    gen_gaussian_target,
)
from py123d_garage.py123d_help.scene_readers import sample_ego_se2
from py123d_garage.py123d_help.scene_readers.target_points import get_target_point_poses

# IoU a box shifted by the heatmap Gaussian's radius still keeps with its label box.
_CENTER_NET_GAUSSIAN_MIN_OVERLAP: float = 0.3


@dataclass(frozen=True)
class TransfuserLabels(AbstractLabels):
    """TransFuser's supervision tensors; fields of disabled heads are None."""

    # Future ego poses (x, y, yaw) relative to the anchor.
    trajectory: jt.Float[torch.Tensor, "*batch poses columns"] | None = None
    # Normalized unicycle (acceleration, curvature) per trajectory pose.
    action: jt.Float[torch.Tensor, "*batch poses 2"] | None = None
    # (x, y, route heading) every path_config.spacing_m along the route ahead.
    path: jt.Float[torch.Tensor, "*batch path_points 3"] | None = None
    # Perspective class id per pixel, before semantic_class_merges.
    semantic: jt.UInt8[torch.Tensor, "*batch semantic_height semantic_width"] | None = None
    # Perspective depth per pixel as 8-bit linear codes.
    depth: jt.UInt8[torch.Tensor, "*batch semantic_height semantic_width"] | None = None
    # BEV semantic class raster with its map and box-annotation flags.
    bev_semantic: BevSemanticLabels | None = None
    # CenterNet label maps of the annotated boxes.
    center_net: CenterNetLabels | None = None


def build_trajectory_label(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, "poses 3"]:
    """
    The ego's future poses relative to the anchor.

    Args:
        scene_api: scene anchored at the current frame.
        config: TransFuser config.

    Returns:
        the future trajectory as (x, y, yaw).
    """
    trajectory_se2 = sample_ego_se2(
        scene_api=scene_api,
        offsets_us=config.trajectory_offsets_us,
        relative_to_anchor=True,
    )
    return torch.tensor(trajectory_se2.pose_se2_array, dtype=torch.float32)


def build_path_label(
    scene_api: SceneAPI,
    path_config: PathConfig,
) -> jt.Float[torch.Tensor, "path_points 3"]:
    """
    The log's driven path ahead of the ego at evenly spaced distances.

    Args:
        scene_api: scene anchored at the current frame.
        path_config: number and spacing of the path points.

    Returns:
        (x, y, route heading) per path point in the anchor rear-axle frame.
    """
    distances_m = [path_config.spacing_m * (index + 1) for index in range(path_config.num_points)]
    return torch.tensor(get_target_point_poses(scene_api, distances_m), dtype=torch.float32)


def semantic_class_ids(config: TransfuserConfig) -> list[int]:
    """
    The stored class ids in order: semantic_class_ids, or the default taxonomy's values.

    Args:
        config: TransFuser config.

    Returns:
        the stored class ids.
    """
    class_ids = config.perspective_config.semantic_class_ids
    if class_ids is None:
        return [int(member.value) for member in DefaultCameraSegmentationLabel]
    return list(class_ids)


def semantic_classes(config: TransfuserConfig) -> list[int]:
    """
    The predicted class ids in order: the stored ids minus the merged ones.

    Args:
        config: TransFuser config.

    Returns:
        the predicted class ids.
    """
    class_ids = semantic_class_ids(config)
    merges = config.perspective_config.semantic_class_merges
    unknown = (merges.keys() | set(merges.values())) - set(class_ids)
    assert not unknown, f"semantic_class_merges names {sorted(unknown)}, not in semantic_class_ids {class_ids}"
    return [class_id for class_id in class_ids if class_id not in merges]


def depth_m_to_codes(
    depth_m: jt.Float[torch.Tensor, "*batch height width"],
    config: TransfuserConfig,
) -> jt.UInt8[torch.Tensor, "*batch height width"]:
    """
    Quantizes depth in meters to 8-bit linear codes.

    Args:
        depth_m: depth in meters, in [0, depth_max_m].
        config: TransFuser config.

    Returns:
        the linear code of every pixel.
    """
    scale = 255.0 / config.perspective_config.depth_max_m
    return depth_m.float().mul(scale).round().clamp_(0.0, 255.0).to(torch.uint8)


def depth_codes_to_m(
    depth_codes: jt.UInt8[torch.Tensor, "*batch height width"],
    config: TransfuserConfig,
) -> jt.Float[torch.Tensor, "*batch height width"]:
    """
    Restores meters from 8-bit linear depth codes.

    Args:
        depth_codes: the linear code of every pixel.
        config: TransFuser config.

    Returns:
        depth in meters, in [0, depth_max_m].
    """
    # Divide in float64 to match the cache codec's dequantization bit for bit.
    scale = 255.0 / config.perspective_config.depth_max_m
    return depth_codes.double().div(scale).float()


def merge_semantic_classes(
    semantic: jt.UInt8[torch.Tensor, "*batch height width"],
    config: TransfuserConfig,
) -> jt.UInt8[torch.Tensor, "*batch height width"]:
    """
    Folds merged classes into their targets and renumbers the pixels to indices into semantic_classes.

    Args:
        semantic: label as built or cached, in the unmerged taxonomy.
        config: TransFuser config.

    Returns:
        the label with every pixel's index into semantic_classes.
    """
    merges = config.perspective_config.semantic_class_merges
    if not merges:
        return semantic
    native = config.perspective_config.semantic_class_ids is not None
    classes = semantic_classes(config)
    lookup = torch.zeros(256, dtype=torch.uint8)
    for index, class_id in enumerate(semantic_class_ids(config)):
        stored = index if native else class_id
        lookup[stored] = classes.index(merges.get(class_id, class_id))
    return lookup.to(semantic.device)[semantic.long()]


@functools.cache
def _semantic_lookup(
    label_class: type[CameraSegmentationLabel],
    class_ids: tuple[int, ...] | None,
) -> jt.Int16[npt.NDArray[np.int16], " 256"]:
    """
    Lookup table from native class id to its position in class_ids or its default-taxonomy value.

    Args:
        label_class: the dataset's segmentation label enum.
        class_ids: native ids to keep, in order; None maps to the default taxonomy.

    Returns:
        the mapped value per native id, -1 for unmapped ids.
    """
    lookup = np.full(256, -1, dtype=np.int16)
    if class_ids is None:
        for member in label_class:
            lookup[int(member.value)] = int(member.to_default().value)
    else:
        for index, class_id in enumerate(class_ids):
            lookup[class_id] = index
    return lookup


def _map_semantic(
    image: jt.UInt8[npt.NDArray[np.uint8], "height width"],
    label_class: type[CameraSegmentationLabel],
    class_ids: list[int] | None,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width"]:
    """
    Maps native class ids to their positions in class_ids, or to the default taxonomy.

    Args:
        image: segmentation frame of native class ids.
        label_class: the dataset's segmentation label enum.
        class_ids: native ids to keep, in order; None maps to the default taxonomy.

    Returns:
        the frame with every pixel mapped.

    Raises:
        ValueError: if class_ids holds an id label_class does not name, or the frame has an unmapped id.
    """
    if class_ids is not None:
        foreign_ids = set(class_ids) - {int(member.value) for member in label_class}
        if foreign_ids:
            raise ValueError(f"semantic_class_ids {sorted(foreign_ids)} are no {label_class.__name__} members")
    mapped = _semantic_lookup(label_class, None if class_ids is None else tuple(class_ids))[image]
    unknown_ids = np.unique(image[mapped < 0])
    if unknown_ids.size:
        raise ValueError(f"semantic frame contains class ids {unknown_ids.tolist()} with no mapped class")
    return mapped.astype(np.uint8)


def build_perspective_labels(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> tuple[
    jt.UInt8[torch.Tensor, "semantic_height semantic_width"] | None,
    jt.UInt8[torch.Tensor, "semantic_height semantic_width"] | None,
]:
    """
    Perspective semantic and depth labels, stitched row by row like the camera feature and resized with nearest interpolation.

    Args:
        scene_api: scene anchored at the current frame.
        config: TransFuser config.

    Returns:
        the semantic label in unmerged class ids and the depth label as 8-bit codes; None for a disabled label.
    """
    perspective = config.perspective_config
    camera_config = config.camera_config
    width = camera_config.image_width // perspective.perspective_downsample_factor
    height = camera_config.image_height // perspective.perspective_downsample_factor
    input_cameras = camera_config.input_cameras[scene_api.scene_metadata.dataset]
    frame_rows = camera_frame_rows(scene_api, config)
    if height % len(frame_rows):
        raise ValueError(
            f"the {height} perspective label rows do not split into the {len(frame_rows)} rows of the stitched image; "
            "image_height / (1 + num_past_camera_rows) must be a multiple of perspective_downsample_factor.",
        )
    row_height = height // len(frame_rows)

    def label_canvas(
        at_iteration: Callable[..., Any],
        at_timestamp: Callable[..., Any],
        decode: Callable[[Any, CameraID], npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        """Stitches every row from the frames its camera row shows: the anchor row at iteration 0, a past row by timestamp."""
        rows: list[npt.NDArray[Any]] = []
        for row_index, frames in enumerate(frame_rows):
            images: list[npt.NDArray[Any]] = []
            for camera_id, frame in zip(input_cameras, frames, strict=True):
                camera = (
                    at_iteration(0, camera_id)
                    if row_index == len(frame_rows) - 1
                    else at_timestamp(frame.timestamp, camera_id, criteria="exact")
                )
                assert camera is not None, (
                    f"Camera {camera_id.name} has no label frame at {frame.timestamp.time_us} us for label computation!"
                )
                images.append(decode(camera, camera_id))
            rows.append(
                cv2.resize(np.concatenate(images, axis=1), (width, row_height), interpolation=cv2.INTER_NEAREST),
            )
        return np.concatenate(rows, axis=0)

    semantic: torch.Tensor | None = None
    if perspective.use_semantic:
        num_classes = len(semantic_classes(config))
        assert num_classes == perspective.num_semantic_classes, (
            f"num_semantic_classes {perspective.num_semantic_classes} is not the {num_classes} classes of semantic_classes"
        )

        def decode_semantic(camera: Any, camera_id: CameraID) -> npt.NDArray[np.uint8]:
            """Maps one semantic frame's native class ids."""
            del camera_id
            metadata = camera.metadata
            assert isinstance(metadata, SegmentationCameraMetadata)
            return _map_semantic(camera.image, metadata.segmentation_label_class, perspective.semantic_class_ids)

        semantic = torch.as_tensor(
            label_canvas(
                scene_api.get_camera_semantic_at_iteration,
                scene_api.get_camera_semantic_at_timestamp,
                decode_semantic,
            ),
            dtype=torch.uint8,
        )

    depth: torch.Tensor | None = None
    if perspective.use_depth:

        def decode_depth(camera: Any, camera_id: CameraID) -> npt.NDArray[np.float32]:
            """Decodes one depth frame to meters, checking it is stored as the cache codec expects."""
            metadata = camera.metadata
            assert isinstance(metadata, DepthCameraMetadata)
            assert metadata.max_depth == perspective.depth_max_m, (
                f"Depth far plane {metadata.max_depth} of {camera_id.name} "
                f"differs from depth_max_m {perspective.depth_max_m}"
            )
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
            return metadata.decode_depth(camera.image)

        depth = depth_m_to_codes(
            torch.as_tensor(
                label_canvas(
                    scene_api.get_camera_depth_at_iteration,
                    scene_api.get_camera_depth_at_timestamp,
                    decode_depth,
                ),
                dtype=torch.float32,
            ),
            config,
        )

    return semantic, depth


def build_bev_labels(
    scene_api: SceneAPI,
    config: TransfuserConfig,
) -> tuple[BevSemanticLabels | None, CenterNetLabels | None]:
    """
    The BEV-semantic and CenterNet labels of one scene, config-gated like the heads.

    Args:
        scene_api: scene anchored at the current frame.
        config: TransFuser config.

    Returns:
        the BEV-semantic and CenterNet labels; None for a disabled head.
    """
    ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for label computation!"
    # None when the log annotates no boxes at this frame, unlike an annotated frame without agents.
    box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(0)
    has_box_annotations = torch.tensor(box_detections_se3 is not None)
    bbse2_array, bbse2_labels, bbse2_speeds = _relative_bounding_boxes(box_detections_se3, ego_state_se3)

    bev_semantic: BevSemanticLabels | None = None
    if config.bev_semantic_config.use_bev_semantic:
        map_api = scene_api.get_map_api()
        map_metadata = scene_api.get_map_metadata() if map_api is not None else None
        z_gate_m = config.bev_semantic_config.map_z_gate_m if map_metadata and map_metadata.map_has_z else None
        bev_semantic = BevSemanticLabels(
            bev_semantic=_compute_bev_semantic_map(
                map_api,
                ego_state_se3,
                bbse2_array,
                bbse2_labels,
                config,
                z_gate_m,
            ),
            bev_semantic_has_map=torch.tensor(map_api is not None),
            bev_semantic_has_box_annotations=has_box_annotations,
        )

    center_net: CenterNetLabels | None = None
    if config.box_detection_config.detect_boxes:
        center_net = replace(
            _compute_center_net_labels(bbse2_array, bbse2_labels, bbse2_speeds, config),
            center_net_has_box_annotations=has_box_annotations,
        )

    return bev_semantic, center_net


def _compute_center_net_labels(
    bbse2_array: jt.Float64[npt.NDArray[np.float64], "num_boxes 5"],
    bbse2_labels: list[DefaultBoxDetectionLabel],
    bbse2_speeds: jt.Float64[npt.NDArray[np.float64], " num_boxes"],
    config: TransfuserConfig,
) -> CenterNetLabels:
    """
    Rasterizes the boxes into CenterNet label maps on the down-sampled BEV raster.

    Args:
        bbse2_array: ego-relative boxes (x, y, yaw, length, width).
        bbse2_labels: label of each box.
        bbse2_speeds: speed of each box in m/s.
        config: TransFuser config.

    Returns:
        the CenterNet labels; boxes of no detection class or off the raster are skipped.
    """
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
    yaw_class = np.zeros((height, width), dtype=np.uint8)
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
        # floor, not int(), so col/row in (-1, 0) do not land on index 0.
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
        center_net_has_box_annotations=torch.tensor(True),
        center_net_velocity=torch.as_tensor(velocity) if config.box_detection_config.predict_box_velocity else None,
    )


def _compute_bev_semantic_map(
    map_api: MapAPI | None,
    ego_state_se3: EgoStateSE3,
    bbse2_array: jt.Float64[npt.NDArray[np.float64], "num_boxes 5"],
    bbse2_labels: list[DefaultBoxDetectionLabel],
    config: TransfuserConfig,
    z_gate_m: float | None = None,
) -> jt.UInt8[torch.Tensor, "height width"]:
    """
    Computes the BEV semantic raster; without a map only the box classes are drawn.

    Args:
        map_api: the scene's map, or None.
        ego_state_se3: the anchor's ego state.
        bbse2_array: ego-relative boxes (x, y, yaw, length, width).
        bbse2_labels: label of each box.
        config: TransFuser config.
        z_gate_m: height window around the ego that map objects must lie in; None draws all.

    Returns:
        the class index of every BEV pixel, later classes drawn over earlier ones.
    """
    bev_semantic_map = np.zeros(
        (
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
        dtype=np.uint8,
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
                z_gate_m,
            )
        elif entity_type == "linestring":
            if map_api is None:
                continue
            entity_mask = _compute_map_linestring_mask(
                map_api,
                ego_state_se3,
                cast("list[MapLayer]", layers),
                config,
                z_gate_m,
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
    """
    Map query radius covering the BEV grid, from its largest boundary extent.

    Args:
        config: TransFuser config.

    Returns:
        the radius in meters.
    """
    return max(
        abs(config.lidar_config.bev_min_x_m),
        config.lidar_config.bev_max_x_m,
        abs(config.lidar_config.bev_min_y_m),
        config.lidar_config.bev_max_y_m,
    )


def _within_z_gate(
    vertices_xyz: jt.Float64[npt.NDArray[np.float64], "n 3"],
    ego_point_3d: Point3D,
    z_gate_m: float,
) -> bool:
    """
    Whether the vertex nearest to the ego in XY lies within z_gate_m of the ego height.

    Args:
        vertices_xyz: the map object's vertices.
        ego_point_3d: the ego rear-axle position.
        z_gate_m: the allowed height difference in meters.

    Returns:
        whether the map object is on the ego's level.
    """
    nearest = np.argmin(
        np.hypot(
            vertices_xyz[:, Point3DIndex.X] - ego_point_3d.x,
            vertices_xyz[:, Point3DIndex.Y] - ego_point_3d.y,
        ),
    )
    return bool(abs(float(vertices_xyz[nearest, Point3DIndex.Z]) - ego_point_3d.z) <= z_gate_m)


def _compute_map_polygon_mask(
    map_api: MapAPI,
    ego_state_se3: EgoStateSE3,
    layers: list[MapLayer],
    config: TransfuserConfig,
    z_gate_m: float | None = None,
) -> jt.Bool[npt.NDArray[np.bool_], "bev_height bev_width"]:
    """
    Binary BEV mask of the given polygon map layers, skipping objects outside the z gate.

    Args:
        map_api: the scene's map.
        ego_state_se3: the anchor's ego state.
        layers: the polygon map layers to draw.
        config: TransFuser config.
        z_gate_m: height window around the ego that map objects must lie in; None draws all.

    Returns:
        the mask of pixels covered by the layers.
    """
    ego_pose_se2 = ego_state_se3.rear_axle_se2
    ego_point_3d = ego_state_se3.rear_axle_3d

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
            if z_gate_m is not None and not _within_z_gate(map_object.outline_3d.array, ego_point_3d, z_gate_m):
                continue
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
    z_gate_m: float | None = None,
) -> jt.Bool[npt.NDArray[np.bool_], "bev_height bev_width"]:
    """
    Binary BEV mask of the given linestring map layers, skipping lanes outside the z gate.

    Args:
        map_api: the scene's map.
        ego_state_se3: the anchor's ego state.
        layers: the lane map layers whose centerlines to draw.
        config: TransFuser config.
        z_gate_m: height window around the ego that lanes must lie in; None draws all.

    Returns:
        the mask of pixels on the centerlines.
    """
    ego_pose_se2 = ego_state_se3.rear_axle_se2
    ego_point_3d = ego_state_se3.rear_axle_3d
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
            lane = cast(Lane, map_object)
            if z_gate_m is not None and not _within_z_gate(lane.centerline_3d.array, ego_point_3d, z_gate_m):
                continue
            linestring: LineString = _geometry_local_coords(
                lane.centerline_2d.linestring,
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
    bbse2_array: jt.Float64[npt.NDArray[np.float64], "num_boxes 5"],
    bbse2_labels: list[DefaultBoxDetectionLabel],
    layers: list[DefaultBoxDetectionLabel],
    config: TransfuserConfig,
) -> jt.Bool[npt.NDArray[np.bool_], "bev_height bev_width"]:
    """
    Binary BEV mask of the boxes with the given labels; pedestrians are scaled up to stay visible.

    Args:
        bbse2_array: ego-relative boxes (x, y, yaw, length, width).
        bbse2_labels: label of each box.
        layers: the box labels to draw.
        config: TransFuser config.

    Returns:
        the mask of pixels covered by the boxes.
    """
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
    jt.Float64[npt.NDArray[np.float64], "num_boxes 5"],
    list[DefaultBoxDetectionLabel],
    jt.Float64[npt.NDArray[np.float64], " num_boxes"],
]:
    """
    The anchor tick's box detections as ego-relative SE2 arrays with labels and speeds.

    Args:
        scene_api: scene anchored at the current frame.
        ego_state_se3: the anchor's ego state.

    Returns:
        the boxes (x, y, yaw, length, width), their labels and speeds; empty without annotations.
    """
    return _relative_bounding_boxes(scene_api.get_box_detections_se3_at_iteration(0), ego_state_se3)


def _relative_bounding_boxes(
    box_detections_se3: BoxDetectionsSE3 | None,
    ego_state_se3: EgoStateSE3,
) -> tuple[
    jt.Float64[npt.NDArray[np.float64], "num_boxes 5"],
    list[DefaultBoxDetectionLabel],
    jt.Float64[npt.NDArray[np.float64], " num_boxes"],
]:
    """
    Box detections as ego-relative SE2 arrays with labels and speeds.

    Args:
        box_detections_se3: the frame's box detections; None for a frame without annotations.
        ego_state_se3: the ego state whose rear axle is the origin.

    Returns:
        the boxes (x, y, yaw, length, width), their labels and speeds; empty for None.
    """
    if box_detections_se3 is None:
        return np.zeros((0, len(BoundingBoxSE2Index)), dtype=np.float64), [], np.zeros(0, dtype=np.float64)
    # CARLA exports some boxes with non-positive extents.
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
    """
    Transforms a shapely geometry into the local coordinates of the origin pose.

    Args:
        geometry: the geometry in global coordinates.
        origin: the pose of the local frame.

    Returns:
        the geometry in the local frame.
    """
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
    """
    Transforms ego-frame (x, y) coordinates into (column, row) pixel indices of the BEV raster.

    Args:
        coords: ego-frame coordinates, (x, y) in the last axis.
        config: TransFuser config.

    Returns:
        the pixel indices, same shape as coords.
    """
    pixel = np.empty_like(coords)
    pixel[..., 0] = (
        coords[..., 0] - config.lidar_config.bev_min_x_m
    ) * config.lidar_config.bev_pixels_per_meter  # column = x
    pixel[..., 1] = (
        coords[..., 1] - config.lidar_config.bev_min_y_m
    ) * config.lidar_config.bev_pixels_per_meter  # row = y
    return pixel.astype(np.int32)


def _angle_to_class(angle_rad: float, num_yaw_bins: int) -> tuple[int, float]:
    """
    Discretizes an angle into a direction bin and residual (inverse of class2angle).

    Args:
        angle_rad: the angle in radians.
        num_yaw_bins: the number of direction bins.

    Returns:
        the bin index and the residual in [-pi, pi).
    """
    angle_per_class = 2.0 * np.pi / num_yaw_bins
    shifted = angle_rad % (2.0 * np.pi)
    bin_idx = int(np.floor((shifted + angle_per_class / 2.0) / angle_per_class)) % num_yaw_bins
    residual = shifted - bin_idx * angle_per_class
    residual = (residual + np.pi) % (2.0 * np.pi) - np.pi
    return bin_idx, float(residual)
