# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from math import sqrt
from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from py123d_garage.api.abstract_policy_tensors import TensorBundle
from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)


class BoundingBoxIndex(IntEnum):
    """Index to access the decoded bounding box array."""

    X = 0
    Y = 1
    W = 2
    H = 3
    YAW = 4
    VELOCITY = 5
    CLASS = 6
    SCORE = 7


@dataclass(frozen=True)
class CenterNetLabels(TensorBundle):
    """Ground-truth rasters of the CenterNet head on the down-sampled BEV grid."""

    # Gaussian peak at every box center, one channel per box class.
    center_net_heatmap: jt.Float[torch.Tensor, "*batch classes height width"]
    # Box length and width in pixels at each box center pixel.
    center_net_wh: jt.Float[torch.Tensor, "*batch 2 height width"]
    # Sub-pixel offset of each box center from its pixel.
    center_net_offset: jt.Float[torch.Tensor, "*batch 2 height width"]
    # Yaw bin of the box at each box center pixel.
    center_net_yaw_class: jt.UInt8[torch.Tensor, "*batch height width"]
    # Yaw residual within the bin at each box center pixel.
    center_net_yaw_res: jt.Float[torch.Tensor, "*batch 1 height width"]
    # One at box center pixels and zero elsewhere, so only those pixels add regression loss.
    center_net_pixel_weight: jt.Float[torch.Tensor, "*batch 2 height width"]
    # Number of boxes in the sample.
    center_net_avg_factor: jt.Float[torch.Tensor, "*batch"]
    # False where the source annotates no boxes: unknown, not empty, so the sample adds no box loss.
    center_net_has_box_annotations: jt.Bool[torch.Tensor, "*batch"]
    # Speed of the box at each box center pixel; None unless predict_box_velocity.
    center_net_velocity: jt.Float[torch.Tensor, "*batch 1 height width"] | None = None


class CenterNetDecoder(nn.Module):
    """CenterNet box detection head on the BEV feature grid, adapted from MMDetection."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes the CenterNet decoder.

        Args:
            config: global config dataclass of TransFuser.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config

        self.heatmap_head = self._build_head(
            config.box_detection_config.box_head_input_channels,
            config.box_detection_config.num_box_classes,
        )
        self.wh_head = self._build_head(
            config.box_detection_config.box_head_input_channels,
            2,
        )
        self.offset_head = self._build_head(
            config.box_detection_config.box_head_input_channels,
            2,
        )
        self.yaw_class_head = self._build_head(
            config.box_detection_config.box_head_input_channels,
            config.box_detection_config.num_yaw_bins,
        )
        self.yaw_res_head = self._build_head(
            config.box_detection_config.box_head_input_channels,
            1,
        )
        if config.box_detection_config.predict_box_velocity:
            self.velocity_head = self._build_head(
                config.box_detection_config.box_head_input_channels,
                1,
            )

    def _build_head(self, in_channel: int, out_channel: int) -> nn.Sequential:
        """
        Builds the conv head of one output branch.

        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.

        Returns:
            the head network.
        """
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )

    @override
    def forward(
        self,
        bev_feature_grid: jt.Float[torch.Tensor, "batch channels height width"],
    ) -> CenterNetBoundingBoxPrediction:
        """
        Runs every CenterNet head on the BEV feature grid.

        Args:
            bev_feature_grid: BEV feature grid with box_head_input_channels channels.

        Returns:
            the predictions of every head.
        """
        center_heatmap_pred = self.heatmap_head(bev_feature_grid)
        wh_pred = self.wh_head(bev_feature_grid)
        offset_pred = self.offset_head(bev_feature_grid)
        yaw_class_pred = self.yaw_class_head(bev_feature_grid)
        yaw_res_pred = self.yaw_res_head(bev_feature_grid)
        velocity_pred: torch.Tensor | None = None
        if self._config.box_detection_config.predict_box_velocity:
            velocity_pred = self.velocity_head(bev_feature_grid)

        return CenterNetBoundingBoxPrediction(
            center_heatmap_logit_pred=center_heatmap_pred,
            center_heatmap_pred=center_heatmap_pred.float().sigmoid(),
            wh_pred=wh_pred,
            offset_pred=offset_pred,
            yaw_class_pred=yaw_class_pred,
            yaw_res_pred=yaw_res_pred,
            velocity_pred=velocity_pred,
            config=self._config,
        )

    def compute_loss(
        self,
        predictions: CenterNetBoundingBoxPrediction,
        labels: CenterNetLabels,
    ) -> dict[str, torch.Tensor]:
        """
        Computes bounding box prediction losses.

        Args:
            predictions: bounding box predictions from forward.
            labels: batched CenterNet ground-truth rasters.

        Returns:
            dictionary of unweighted losses.
        """
        center_heatmap_target = labels.center_net_heatmap.float()
        wh_target = labels.center_net_wh.float()
        yaw_class_target = labels.center_net_yaw_class.long()
        yaw_res_target = labels.center_net_yaw_res.float()
        offset_target = labels.center_net_offset.float()
        pixel_weight = labels.center_net_pixel_weight.float()  # [B, 2, H, W]

        # The floor of one keeps a box-free sample from amplifying the heatmap loss, whose focal term covers every pixel.
        avg_factor = labels.center_net_avg_factor.float()  # [B]
        avg_factor_clamped = avg_factor.clamp(min=1.0)
        annotated = labels.center_net_has_box_annotations.float()  # [B]

        def annotated_mean(per_sample: jt.Float[torch.Tensor, " batch"]) -> jt.Float[torch.Tensor, ""]:
            """
            Averages a per-sample loss over the samples with box annotations.

            Args:
                per_sample: loss of every sample.

            Returns:
                the mean loss, zero without annotated samples.
            """
            return (per_sample * annotated).sum() / annotated.sum().clamp(min=1.0)

        loss_center_heatmap_per_sample = gaussian_focal_loss(
            pred=predictions.center_heatmap_pred.float(),
            gaussian_target=center_heatmap_target,
            reduction="none",
        )  # (B, C, H, W)
        loss_center_heatmap_per_sample = loss_center_heatmap_per_sample.sum(
            dim=(1, 2, 3),
        )  # (B,)
        loss_center_heatmap = annotated_mean(loss_center_heatmap_per_sample / avg_factor_clamped)

        loss_wh_per_sample = (
            F.l1_loss(predictions.wh_pred.float(), wh_target, reduction="none") * pixel_weight
        )  # (B, 2, H, W)
        loss_wh_per_sample = loss_wh_per_sample.sum(dim=(1, 2, 3))  # (B,)
        loss_wh = annotated_mean(loss_wh_per_sample / (avg_factor_clamped * predictions.wh_pred.shape[1]))

        loss_offset_per_sample = (
            F.l1_loss(
                predictions.offset_pred.float(),
                offset_target,
                reduction="none",
            )
            * pixel_weight
        )  # (B, 2, H, W)
        loss_offset_per_sample = loss_offset_per_sample.sum(
            dim=(1, 2, 3),
        )  # (B,)
        loss_offset = annotated_mean(loss_offset_per_sample / (avg_factor_clamped * predictions.wh_pred.shape[1]))

        loss_yaw_class_per_sample = (
            F.cross_entropy(
                predictions.yaw_class_pred.float(),
                yaw_class_target,
                reduction="none",
            )
            * pixel_weight[:, 0]
        )  # (B, H, W)
        loss_yaw_class_per_sample = loss_yaw_class_per_sample.sum(
            dim=(1, 2),
        )  # (B,)
        loss_yaw_class = annotated_mean(loss_yaw_class_per_sample / avg_factor_clamped)

        loss_yaw_res_per_sample = (
            F.smooth_l1_loss(
                predictions.yaw_res_pred.float(),
                yaw_res_target,
                reduction="none",
            )
            * pixel_weight[:, 0:1]
        )  # (B, 1, H, W)
        loss_yaw_res_per_sample = loss_yaw_res_per_sample.sum(
            dim=(1, 2, 3),
        )  # (B,)
        loss_yaw_res = annotated_mean(loss_yaw_res_per_sample / avg_factor_clamped)

        losses = {
            "loss_center_net_heatmap": loss_center_heatmap,
            "loss_center_net_wh": loss_wh,
            "loss_center_net_offset": loss_offset,
            "loss_center_net_yaw_class": loss_yaw_class,
            "loss_center_net_yaw_res": loss_yaw_res,
        }

        if self._config.box_detection_config.predict_box_velocity:
            assert predictions.velocity_pred is not None
            assert labels.center_net_velocity is not None
            velocity_target = labels.center_net_velocity.float()
            loss_velocity_per_sample = (
                F.l1_loss(
                    predictions.velocity_pred.float(),
                    velocity_target,
                    reduction="none",
                )
                * pixel_weight[:, 0:1]
            )  # (B, 1, H, W)
            loss_velocity_per_sample = loss_velocity_per_sample.sum(
                dim=(1, 2, 3),
            )  # (B,)
            losses["loss_center_net_velocity"] = annotated_mean(loss_velocity_per_sample / avg_factor_clamped)

        return losses

    def compute_metrics(
        self,
        predictions: CenterNetBoundingBoxPrediction,
        labels: CenterNetLabels,
    ) -> dict[str, torch.Tensor]:
        """
        Returns no metrics, since the box head defines none.

        Args:
            predictions: bounding box predictions from forward.
            labels: batched CenterNet ground-truth rasters.

        Returns:
            an empty dictionary.
        """
        del predictions, labels
        return {}


@dataclass(frozen=True)
class CenterNetBoundingBoxPrediction(TensorBundle):
    """Output features of the CenterNet head."""

    # Heatmap logits, one channel per box class.
    center_heatmap_logit_pred: jt.Float[
        torch.Tensor,
        "*batch classes height width",
    ]
    # Heatmap probabilities, the sigmoid of the logits.
    center_heatmap_pred: jt.Float[torch.Tensor, "*batch classes height width"]
    # Box length and width in pixels.
    wh_pred: jt.Float[torch.Tensor, "*batch 2 height width"]
    # Sub-pixel offset of the box center.
    offset_pred: jt.Float[torch.Tensor, "*batch 2 height width"]
    # Yaw bin logits.
    yaw_class_pred: jt.Float[torch.Tensor, "*batch yaw_bins height width"]
    # Yaw residual within the bin.
    yaw_res_pred: jt.Float[torch.Tensor, "*batch 1 height width"]
    # Box speed; None unless predict_box_velocity.
    velocity_pred: jt.Float[torch.Tensor, "*batch 1 height width"] | None
    # Config that sets the decoding thresholds and raster geometry.
    config: TransfuserConfig

    @cached_property
    def pred_bounding_box_image_system(
        self,
    ) -> jt.Float32[npt.NDArray[np.float32], "batch k 8"]:
        """
        Decodes the top-k boxes from the heatmap into the BEV raster pixel system.

        Returns:
            (x, y, w, h, yaw, velocity, class, score) of every box in the full-resolution BEV raster.
        """
        config = self.config
        k = config.box_detection_config.max_center_net_detections
        kernel = config.box_detection_config.center_net_max_pooling_kernel_size

        center_heatmap_pred = get_local_maximum(
            self.center_heatmap_pred,
            kernel=kernel,
        )

        batch_scores, batch_index, batch_topk_classes, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred,
            k=k,
        )

        wh = transpose_and_gather_feat(self.wh_pred, batch_index)
        offset = transpose_and_gather_feat(self.offset_pred, batch_index)
        yaw_class = transpose_and_gather_feat(self.yaw_class_pred, batch_index)
        yaw_res = transpose_and_gather_feat(self.yaw_res_pred, batch_index)

        yaw_class = torch.argmax(yaw_class, -1)
        yaw = class2angle(
            yaw_class,
            yaw_res.squeeze(2),
            config.box_detection_config.num_yaw_bins,
        )

        if self.velocity_pred is None:
            velocity = torch.zeros_like(yaw)
        else:
            velocity = transpose_and_gather_feat(
                self.velocity_pred,
                batch_index,
            )
            velocity = velocity[..., 0]

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        batch_bboxes = torch.stack(
            [topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity],
            dim=2,
        )
        batch_bboxes = torch.cat(
            (
                batch_bboxes,
                batch_topk_classes[..., np.newaxis],
                batch_scores[..., np.newaxis],
            ),
            dim=-1,
        )
        # The heads run on the down-sampled BEV grid; scale coordinates and extents
        # up to the full-resolution raster pixel system.
        batch_bboxes[:, :, : BoundingBoxIndex.YAW] *= config.bev_semantic_config.bev_downsample_factor

        return cast(
            npt.NDArray[np.float32],
            batch_bboxes.detach().cpu().float().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )

    @cached_property
    def pred_bounding_box_vehicle_system(
        self,
    ) -> list[jt.Float32[npt.NDArray[np.float32], "_ 8"]]:
        """
        Decodes confident boxes into the ego vehicle system (meters).

        Returns:
            (x, y, w, h, yaw, velocity, class, score) of every box above the confidence threshold, one array per sample.
        """
        config = self.config
        bboxes_image_system = self.pred_bounding_box_image_system

        bounding_box_vehicle_system: list[npt.NDArray[np.float32]] = []
        for sample_boxes in bboxes_image_system:
            sample_boxes = sample_boxes[
                sample_boxes[:, BoundingBoxIndex.SCORE] > config.box_detection_config.box_confidence_threshold
            ]
            bounding_box_vehicle_system.append(
                bb_image_to_vehicle_system(
                    sample_boxes,
                    config.lidar_config.bev_pixels_per_meter,
                    config.lidar_config.bev_min_x_m,
                    config.lidar_config.bev_min_y_m,
                ),
            )
        return bounding_box_vehicle_system


def bb_image_to_vehicle_system(
    box: jt.Float32[npt.NDArray[np.float32], "n 8"],
    pixels_per_meter: float,
    min_x: float,
    min_y: float,
) -> jt.Float32[npt.NDArray[np.float32], "n 8"]:
    """
    Converts bounding boxes from the BEV raster pixel system to the vehicle system.

    Args:
        box: bounding box array in the image (BEV raster pixel) system.
        pixels_per_meter: scaling factor from meters to pixels.
        min_x: minimum x value of the raster in the vehicle coordinate system.
        min_y: minimum y value of the raster in the vehicle coordinate system.

    Returns:
        bounding box array in the vehicle coordinate system.
    """
    box = box.copy()
    box[:, :4] = box[:, :4] / pixels_per_meter
    box[:, :2] = box[:, :2] + np.array([min_x, min_y])
    return box


def gaussian_focal_loss(
    pred: jt.Float[torch.Tensor, "batch classes height width"],
    gaussian_target: jt.Float[torch.Tensor, "batch classes height width"],
    alpha: float = 2.0,
    gamma: float = 4.0,
    reduction: str = "mean",
) -> jt.Float[torch.Tensor, "*shape"]:
    """
    Gaussian focal loss for heatmap regression, adapted from MMDetection.

    Args:
        pred: the prediction (after sigmoid).
        gaussian_target: the learning target of the prediction in gaussian distribution.
        alpha: exponent of the focal modulating factor of the prediction.
        gamma: exponent of the (1 - target) weight of the negative pixels.
        reduction: the reduction method to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        the computed loss.
    """
    eps = 1e-12
    pred = pred.float()
    gaussian_target = gaussian_target.float()
    pos_weights = gaussian_target.eq(1)
    neg_weights = cast(torch.Tensor, 1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * cast(torch.Tensor, 1 - pred).pow(alpha) * pos_weights
    neg_loss = -cast(torch.Tensor, 1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    loss = pos_loss + neg_loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def gaussian2d(
    radius: int,
    sigma: float = 1,
    dtype: np.dtype[np.float32] | type[np.float32] = np.float32,
) -> jt.Float32[npt.NDArray[np.float32], "diameter diameter"]:
    """
    Generates a 2D gaussian kernel.

    Args:
        radius: radius of gaussian kernel.
        sigma: sigma of gaussian function, defaults to 1.
        dtype: dtype of gaussian array, defaults to np.float32.

    Returns:
        gaussian kernel with a (2 * radius + 1) * (2 * radius + 1) shape.
    """
    x = np.arange(-radius, radius + 1, dtype=dtype).reshape(1, -1)
    y = np.arange(-radius, radius + 1, dtype=dtype).reshape(-1, 1)

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def gen_gaussian_target(
    heatmap: jt.Float32[npt.NDArray[np.float32], "height width"],
    center: list[int],
    radius: int,
    k: int = 1,
) -> jt.Float32[npt.NDArray[np.float32], "height width"]:
    """
    Draws a 2D gaussian into a heatmap in place, keeping the maximum where they overlap.

    Args:
        heatmap: input heatmap; the gaussian kernel will cover it and maintain the max value.
        center: (column, row) of the gaussian kernel's center.
        radius: radius of gaussian kernel.
        k: coefficient of gaussian kernel, defaults to 1.

    Returns:
        updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2d(
        radius,
        sigma=diameter / 6,
        dtype=heatmap.dtype,
    )

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian_kernel[
        radius - top : radius + bottom,
        radius - left : radius + right,
    ]
    out_heatmap = heatmap
    np.maximum(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top : y + bottom, x - left : x + right],
    )

    return out_heatmap


def gaussian_radius(det_size: list[float], min_overlap: float) -> int:
    r"""
    Computes the gaussian kernel radius within which a shifted box keeps IoU min_overlap with the box of det_size.

    From CornerNet-Lite: https://github.com/princeton-vl/CornerNet-Lite/blob/6a54505d830a9d6afe26e99f0864b5d06d0bbbaf/core/sample/utils.py#L65

    Args:
        det_size: (height, width) of the box in heatmap pixels.
        min_overlap: minimum IoU the gaussian radius should keep.

    Returns:
        gaussian kernel radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return int(min(r1, r2, r3))


def get_local_maximum(
    heat: jt.Float[torch.Tensor, "batch classes height width"],
    kernel: int = 3,
) -> jt.Float[torch.Tensor, "batch classes height width"]:
    """
    Extracts local maximum pixels with a given kernel.

    Args:
        heat: target heatmap.
        kernel: kernel size of max pooling, defaults to 3.

    Returns:
        a heatmap where local maximum pixels maintain their own value and
        other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(
    scores: jt.Float[torch.Tensor, "batch classes height width"],
    k: int = 20,
) -> tuple[
    jt.Float[torch.Tensor, "batch k"],
    jt.Int[torch.Tensor, "batch k"],
    jt.Int[torch.Tensor, "batch k"],
    jt.Int[torch.Tensor, "batch k"],
    jt.Float[torch.Tensor, "batch k"],
]:
    """
    Gets top k positions from the heatmap.

    Args:
        scores: target heatmap.
        k: number of positions to keep.

    Returns:
        the scores, flat pixel indices, classes, rows and columns of the top-k positions.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, -1), k)
    topk_clses = torch.div(topk_inds, (height * width), rounding_mode="trunc")
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="trunc")
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(
    feat: jt.Float[torch.Tensor, "batch tokens dim"],
    ind: jt.Int[torch.Tensor, "batch k"],
    mask: jt.Bool[torch.Tensor, "batch k"] | None = None,
) -> jt.Float[torch.Tensor, "*shape"]:
    """
    Gathers features according to an index.

    Args:
        feat: target feature map.
        ind: target coordinate index.
        mask: True for the gathered entries to keep; None keeps all.

    Returns:
        gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(
    feat: jt.Float[torch.Tensor, "batch channels height width"],
    ind: jt.Int[torch.Tensor, "batch k"],
) -> jt.Float[torch.Tensor, "batch k channels"]:
    """
    Transposes and gathers features according to an index.

    Args:
        feat: target feature map.
        ind: target coordinate index.

    Returns:
        transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return gather_feat(feat, ind)


def class2angle(
    angle_cls: jt.Int[torch.Tensor, "batch k"],
    angle_res: jt.Float[torch.Tensor, "batch k"],
    num_dir_bins: int,
    limit_period: bool = True,
) -> jt.Float[torch.Tensor, "batch k"]:
    """
    Converts discrete angle class and residual back to a continuous angle.

    Args:
        angle_cls: discrete angle class tensor to decode.
        angle_res: angle residual tensor to decode.
        num_dir_bins: number of direction bins for object orientation.
        limit_period: whether to limit the angle to the [-pi, pi] range.

    Returns:
        decoded continuous angle tensor.
    """
    angle_per_class = 2 * np.pi / float(num_dir_bins)
    angle_center = angle_cls.float() * angle_per_class
    angle = angle_center + angle_res
    if limit_period:
        angle[angle > np.pi] -= 2 * np.pi
    return angle
