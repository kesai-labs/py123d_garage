# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

import dataclasses

import jaxtyping as jt
import torch
import torch.nn.functional as F
import torchmetrics.functional
from torch import nn
from typing_extensions import override

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.datatypes.tensor import TensorBundle
from py123d_garage.policy.transfuser.network.encoder.transfuser import BilinearMatmulUpsample


@dataclasses.dataclass(frozen=True)
class BevSemanticLabels(TensorBundle):
    """Ground-truth class raster of the BEV semantic head, and whether its scene had a map."""

    # Class index of every BEV raster pixel.
    bev_semantic: jt.UInt8[torch.Tensor, "*batch height width"]
    # Without a map, a background pixel may be any map class.
    bev_semantic_has_map: jt.Bool[torch.Tensor, "*batch"]
    # Without box annotations, any pixel may be any box class.
    bev_semantic_has_box_annotations: jt.Bool[torch.Tensor, "*batch"]


class BEVSemanticDecoder(nn.Module):
    """Dense BEV decoder predicting a semantic class per BEV raster pixel."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes the BEV semantic decoder.

        Args:
            config: global config dataclass of TransFuser.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config
        map_classes = [
            label
            for label, (entity_type, _) in config.bev_semantic_config.selected_bev_semantic_classes.items()
            if entity_type != "box"
        ]
        self._map_classes: torch.Tensor
        self.register_buffer("_map_classes", torch.tensor(map_classes, dtype=torch.long), persistent=False)
        self._non_box_classes: torch.Tensor
        self.register_buffer("_non_box_classes", torch.tensor([0, *map_classes], dtype=torch.long), persistent=False)
        box_classes = [
            label
            for label, (entity_type, _) in config.bev_semantic_config.selected_bev_semantic_classes.items()
            if entity_type == "box"
        ]
        self._box_classes: torch.Tensor
        self.register_buffer("_box_classes", torch.tensor(box_classes, dtype=torch.long), persistent=False)
        is_class = torch.arange(config.bev_semantic_config.num_bev_semantic_classes)
        self._is_map_class: torch.Tensor
        self.register_buffer("_is_map_class", torch.isin(is_class, self._map_classes), persistent=False)
        self._is_box_class: torch.Tensor
        self.register_buffer("_is_box_class", torch.isin(is_class, self._box_classes), persistent=False)

        self.net = nn.Sequential(
            nn.Conv2d(
                config.bev_semantic_config.bev_feature_channels,
                config.bev_semantic_config.bev_feature_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_semantic_config.bev_feature_channels,
                config.bev_semantic_config.num_bev_semantic_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            BilinearMatmulUpsample(
                size=(
                    config.lidar_config.bev_height_pixel,
                    config.lidar_config.bev_width_pixel,
                ),
            ),
        )

    @override
    def forward(
        self,
        bev_feature_grid: jt.Float[torch.Tensor, "batch channels height width"],
    ) -> jt.Float[torch.Tensor, "batch classes out_height out_width"]:
        """
        Forward pass for the BEV semantic decoder.

        Args:
            bev_feature_grid: BEV feature grid from the encoder.

        Returns:
            BEV semantic logits at the full LiDAR raster resolution.
        """
        return self.net(bev_feature_grid)

    def compute_loss(
        self,
        predictions: jt.Float[torch.Tensor, "batch classes height width"],
        label: BevSemanticLabels,
    ) -> dict[str, torch.Tensor]:
        """
        Computes the BEV semantic cross-entropy, where a missing map or missing boxes widen the classes a pixel allows.

        Args:
            predictions: BEV semantic logits from forward.
            label: ground-truth raster at the full LiDAR raster resolution, with the map and box flags.

        Returns:
            dictionary of unweighted losses.
        """
        target = label.bev_semantic.long()
        logits = predictions.float()

        # Without a map a background pixel also allows every map class; without boxes every pixel allows every box class.
        classes = torch.arange(logits.shape[1], device=target.device)
        allowed = classes[None, :, None, None] == target[:, None]
        no_map_background = ~label.bev_semantic_has_map[:, None, None] & (target == 0)
        allowed = allowed | (no_map_background[:, None] & self._is_map_class[None, :, None, None])
        no_boxes = ~label.bev_semantic_has_box_annotations
        allowed = allowed | (no_boxes[:, None, None, None] & self._is_box_class[None, :, None, None])
        if self._config.bev_semantic_config.two_logsumexp_loss:
            nll = torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
        else:
            nll = -torch.logsumexp(F.log_softmax(logits, dim=1).masked_fill(~allowed, -torch.inf), dim=1)

        # A sample with neither map nor boxes allows every class everywhere: no loss, and no share of the mean.
        informative = (label.bev_semantic_has_map | label.bev_semantic_has_box_annotations).float()
        per_sample = nll.mean(dim=(1, 2))
        return {"loss_bev_semantic": (per_sample * informative).sum() / informative.sum().clamp(min=1.0)}

    def compute_metrics(
        self,
        predictions: jt.Float[torch.Tensor, "batch classes height width"],
        label: BevSemanticLabels,
    ) -> dict[str, torch.Tensor]:
        """
        Computes BEV semantic mIoU and macro F1, not counting map or box classes the label could not show as errors.

        Args:
            predictions: BEV semantic logits from forward.
            label: ground-truth raster at the full LiDAR raster resolution, with the map and box flags.

        Returns:
            dictionary of scalar metric tensors.
        """
        num_classes = self._config.bev_semantic_config.num_bev_semantic_classes
        predicted = predictions.argmax(dim=1)
        if self._map_classes.numel() > 0:
            map_without_map = ~label.bev_semantic_has_map[:, None, None] & torch.isin(predicted, self._map_classes)
            predicted = torch.where(map_without_map, torch.zeros_like(predicted), predicted)
        target = label.bev_semantic.long()
        box_without_boxes = ~label.bev_semantic_has_box_annotations[:, None, None] & torch.isin(
            predicted,
            self._box_classes,
        )
        predicted = torch.where(box_without_boxes, target, predicted)
        return {
            "bev_semantic_miou": torchmetrics.functional.jaccard_index(
                predicted,
                target,
                task="multiclass",
                num_classes=num_classes,
            ),
            "bev_semantic_f1": torchmetrics.functional.f1_score(
                predicted,
                target,
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ),
        }
