"""The composed sample views: BEV composite on the left, perspective stack on the right."""

from __future__ import annotations

from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.labels import TransfuserLabels
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.visualization._bev_composite import (
    render_bev,
)
from py123d_garage.policy.transfuser.visualization._perspective_colorization import (
    depth_to_rgb,
    semantic_to_rgb,
)

_BORDER_PIXEL = 8


def render_ground_truth(
    features: TransfuserFeatures,
    labels: TransfuserLabels,
    target_points: jt.Float[torch.Tensor, "batch num_points 2"],
    config: TransfuserConfig,
    route: jt.Float32[npt.NDArray[np.float32], "num_route_points 2"] | None = None,
    boxes: jt.Float32[npt.NDArray[np.float32], "num_boxes 8"] | None = None,
    ego_box: jt.Float32[npt.NDArray[np.float32], " 5"] | None = None,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Renders the ground-truth view of the first sample of a collated batch.

    Args:
        features: batched feature bundle, as consumed by the policy's forward
        labels: batched label bundle
        target_points: batched navigation intent
        config: global config dataclass of TransFuser
        route: ego-frame route polyline of the first sample, or None
        boxes: ground-truth boxes of the first sample in the decoded
            BoundingBoxIndex layout, or None to draw no box outlines
        ego_box: ego bounding box (x, y, yaw, length, width) in the ego
            frame, or None to draw no ego

    Returns:
        composed RGB image
    """
    trajectories: list[tuple[npt.NDArray[np.float32], TrajectoryStyleConfig]] = []
    if labels.trajectory is not None:
        trajectories.append((_sample(labels.trajectory), config.visualization_config.ground_truth_trajectory_style))
    return _render_view(
        features=features,
        semantic_map=_sample_long(labels.bev_semantic.bev_semantic) if labels.bev_semantic is not None else None,
        perspective_semantic=_sample_long(labels.semantic) if labels.semantic is not None else None,
        perspective_depth=_sample(labels.depth) if labels.depth is not None else None,
        boxes=boxes,
        trajectories=trajectories,
        target_points=target_points,
        config=config,
        route=route,
        ego_box=ego_box,
    )


def render_prediction(
    features: TransfuserFeatures,
    target_points: jt.Float[torch.Tensor, "batch num_points 2"],
    predictions: TransfuserPredictions,
    config: TransfuserConfig,
    bev_semantic_classes: jt.Int[torch.Tensor, "batch bev_height bev_width"] | None = None,
    route: jt.Float32[npt.NDArray[np.float32], "num_route_points 2"] | None = None,
    ego_box: jt.Float32[npt.NDArray[np.float32], " 5"] | None = None,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Renders the prediction view of the first sample of a collated batch.

    Args:
        features: batched feature bundle, as consumed by the policy's forward
        target_points: batched navigation intent
        predictions: batched prediction bundle, as returned by forward
        config: global config dataclass of TransFuser
        bev_semantic_classes: predicted BEV class raster, as the label would draw it,
            or None to draw no BEV semantics
        route: ego-frame route polyline of the first sample, or None
        ego_box: ego bounding box (x, y, yaw, length, width) in the ego
            frame, or None to draw no ego

    Returns:
        composed RGB image
    """
    trajectories: list[tuple[npt.NDArray[np.float32], TrajectoryStyleConfig]] = []
    if predictions.trajectory is not None:
        trajectories.append((_sample(predictions.trajectory), config.visualization_config.prediction_trajectory_style))

    boxes: npt.NDArray[np.float32] | None = None
    if predictions.boxes is not None:
        boxes = predictions.boxes.apply(
            lambda tensor: tensor.detach().cpu(),
        ).pred_bounding_box_vehicle_system[0]

    return _render_view(
        features=features,
        semantic_map=_sample_long(bev_semantic_classes) if bev_semantic_classes is not None else None,
        perspective_semantic=_sample_long(predictions.semantic.argmax(1)) if predictions.semantic is not None else None,
        perspective_depth=_sample(predictions.depth) if predictions.depth is not None else None,
        boxes=boxes,
        trajectories=trajectories,
        target_points=target_points,
        config=config,
        route=route,
        ego_box=ego_box,
    )


def bev_semantic_classes_of(
    bev_semantic_logits: jt.Float[torch.Tensor, "batch classes height width"],
    config: TransfuserConfig,
) -> jt.Int64[torch.Tensor, "batch height width"]:
    """
    The predicted BEV class per pixel, map classes drawn as background when the config says so.

    Args:
        bev_semantic_logits: BEV semantic logits, as returned by forward
        config: global config dataclass of TransFuser

    Returns:
        class index raster
    """
    predicted = bev_semantic_logits.argmax(dim=1)
    if config.visualization_config.visualize_bev_map_classes:
        return predicted
    map_classes = torch.tensor(
        [
            label
            for label, (entity_type, _) in config.bev_semantic_config.selected_bev_semantic_classes.items()
            if entity_type != "box"
        ],
        device=predicted.device,
    )
    return torch.where(torch.isin(predicted, map_classes), torch.zeros_like(predicted), predicted)


def _sample(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """First sample of a batched tensor as a float32 numpy array."""
    return cast(
        npt.NDArray[np.float32],
        tensor[0].detach().cpu().float().numpy(),  # pyright: ignore[reportUnknownMemberType]
    )


def _sample_long(tensor: torch.Tensor) -> npt.NDArray[np.int64]:
    """First sample of a batched tensor as an int64 numpy array."""
    return cast(
        npt.NDArray[np.int64],
        tensor[0].detach().cpu().long().numpy(),  # pyright: ignore[reportUnknownMemberType]
    )


def _render_view(
    features: TransfuserFeatures,
    semantic_map: npt.NDArray[np.int64] | None,
    perspective_semantic: npt.NDArray[np.int64] | None,
    perspective_depth: npt.NDArray[np.float32] | None,
    boxes: npt.NDArray[np.float32] | None,
    trajectories: list[tuple[npt.NDArray[np.float32], TrajectoryStyleConfig]],
    target_points: jt.Float[torch.Tensor, "batch num_points 2"],
    config: TransfuserConfig,
    route: jt.Float32[npt.NDArray[np.float32], "num_route_points 2"] | None,
    ego_box: jt.Float32[npt.NDArray[np.float32], " 5"] | None,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """Composes one view from per-sample source arrays."""
    lidar_map = _sample(features.lidar_feature)[0] if features.lidar_feature is not None else None
    bev_image = render_bev(
        lidar_map,
        semantic_map,
        boxes,
        trajectories,
        _sample(target_points),
        config,
        route=route,
        ego_box=ego_box,
    )

    camera = _sample(features.camera_feature)
    perspectives: list[npt.NDArray[np.uint8]] = [
        np.transpose(camera, (1, 2, 0)).astype(np.uint8),
    ]
    if perspective_semantic is not None:
        perspectives.append(semantic_to_rgb(perspective_semantic))
    if perspective_depth is not None:
        perspectives.append(
            depth_to_rgb(
                perspective_depth,
                config.perspective_config.depth_max_m,
            ),
        )
    return _compose(bev_image, perspectives)


def _compose(
    bev_image: jt.UInt8[npt.NDArray[np.uint8], "bev_height bev_width 3"],
    perspectives: list[npt.NDArray[np.uint8]],
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Stacks the perspectives on top and the BEV below at one shared width.

    The shared width is the widest native perspective; the BEV scales up to it.
    """
    width = max(image.shape[1] for image in perspectives)
    scaled = [
        cv2.resize(
            image,
            (width, max(1, round(width * image.shape[0] / image.shape[1]))),
        )
        for image in perspectives
    ]
    scaled.append(
        cv2.resize(
            bev_image,
            (
                width,
                max(1, round(width * bev_image.shape[0] / bev_image.shape[1])),
            ),
        ),
    )

    height = sum(image.shape[0] for image in scaled) + _BORDER_PIXEL * (len(scaled) + 1)
    canvas = np.full(
        (height, width + 2 * _BORDER_PIXEL, 3),
        255,
        dtype=np.uint8,
    )
    row = _BORDER_PIXEL
    for image in scaled:
        canvas[
            row : row + image.shape[0],
            _BORDER_PIXEL : _BORDER_PIXEL + width,
        ] = image
        row += image.shape[0] + _BORDER_PIXEL
    return canvas
