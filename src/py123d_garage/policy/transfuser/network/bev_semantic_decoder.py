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


@dataclasses.dataclass(frozen=True)
class BevSemanticLabels(TensorBundle):
    """Ground-truth class raster of the BEV semantic head, and whether its scene had a map."""

    bev_semantic: jt.Int[torch.Tensor, "*batch height width"]
    # Without a map, a background pixel may be any map class.
    bev_semantic_has_map: jt.Bool[torch.Tensor, "*batch"]


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
            nn.Upsample(
                size=(
                    config.lidar_config.bev_height_pixel,
                    config.lidar_config.bev_width_pixel,
                ),
                mode="bilinear",
                align_corners=False,
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
        Cross-entropy over the raster with special handling for map-less samples.

        Args:
            predictions: BEV semantic logits from forward.
            label: ground-truth raster at the full LiDAR raster resolution, with the map flag.

        Returns:
            dictionary of unweighted losses.
        """
        target = label.bev_semantic.long()
        log_probs = F.log_softmax(predictions.float(), dim=1)

        nll = -log_probs.gather(1, target[:, None]).squeeze(1)
        if self._map_classes.numel() > 0:
            unknown_non_box = ~label.bev_semantic_has_map[:, None, None] & (target == 0)
            non_box_log_prob = torch.logsumexp(log_probs[:, self._non_box_classes], dim=1)

            # No map: background or any map class is correct.
            nll = torch.where(unknown_non_box, -non_box_log_prob, nll)
        return {"loss_bev_semantic": nll.mean()}

    def compute_metrics(
        self,
        predictions: jt.Float[torch.Tensor, "batch classes height width"],
        label: BevSemanticLabels,
    ) -> dict[str, torch.Tensor]:
        """
        BEV semantic segmentation quality: mIoU and macro F1. On a map-less sample a
        predicted map class counts as background, like the label draws it.

        Args:
            predictions: BEV semantic logits from forward.
            label: ground-truth raster at the full LiDAR raster resolution, with the map flag.

        Returns:
            dictionary of scalar metric tensors.
        """
        num_classes = self._config.bev_semantic_config.num_bev_semantic_classes
        predicted = predictions.argmax(dim=1)
        if self._map_classes.numel() > 0:
            map_without_map = ~label.bev_semantic_has_map[:, None, None] & torch.isin(predicted, self._map_classes)
            predicted = torch.where(map_without_map, torch.zeros_like(predicted), predicted)
        target = label.bev_semantic.long()
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
