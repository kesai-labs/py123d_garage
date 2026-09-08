# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
import torchmetrics.functional
from torch import nn
from typing_extensions import override

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)


class PerspectiveDecoder(nn.Module):
    """
    Decodes a low resolution perspective grid to a full resolution output.

    Used for the perspective semantic segmentation and depth auxiliary tasks.
    """

    def __init__(
        self,
        config: TransfuserConfig,
        in_channels: int,
        out_channels: int,
        perspective_upsample_factor: int,
        modality: str,
    ):
        """
        Initializes the perspective decoder.

        Args:
            config: global config dataclass of TransFuser.
            in_channels: feature channels of the input feature grid.
            out_channels: feature channels of the output feature grid.
            perspective_upsample_factor: upsampling factor from input feature grid to output.
            modality: "semantic" or "depth".
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert modality in ("semantic", "depth"), f"Unknown perspective modality: {modality}"
        self._config = config
        self._modality = modality
        self._scale_factor_0 = perspective_upsample_factor // config.perspective_config.deconv_scale_factor_0
        self._scale_factor_1 = perspective_upsample_factor // config.perspective_config.deconv_scale_factor_1

        self.deconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                config.perspective_config.deconv_channel_num_0,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.perspective_config.deconv_channel_num_0,
                config.perspective_config.deconv_channel_num_1,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(
                config.perspective_config.deconv_channel_num_1,
                config.perspective_config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.perspective_config.deconv_channel_num_2,
                config.perspective_config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(
                config.perspective_config.deconv_channel_num_2,
                config.perspective_config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.perspective_config.deconv_channel_num_2,
                out_channels,
                3,
                1,
                1,
                bias=True,
            ),
        )

    @override
    def forward(
        self,
        image_feature_grid: jt.Float[
            torch.Tensor,
            "batch channels height width",
        ],
    ) -> jt.Float[torch.Tensor, "batch ..."]:
        """
        Forward pass for the perspective decoder.

        Args:
            image_feature_grid: image feature grid from the encoder.

        Returns:
            prediction tensor of shape [B, out_channels, image_height, image_width]
            for "semantic", or [B, image_height, image_width] for "depth" (both divided
            by perspective_downsample_factor).
        """
        mode = self._config.backbone_config.upsample_mode
        x = self.deconv1(image_feature_grid)
        x = cast(
            torch.Tensor,
            F.interpolate(x, scale_factor=self._scale_factor_0, mode=mode),  # pyright: ignore[reportUnknownMemberType]
        )
        x = self.deconv2(x)
        x = cast(
            torch.Tensor,
            F.interpolate(x, scale_factor=self._scale_factor_1, mode=mode),  # pyright: ignore[reportUnknownMemberType]
        )
        x = self.deconv3(x)

        expected_h = (
            self._config.camera_config.image_height // self._config.perspective_config.perspective_downsample_factor
        )
        expected_w = (
            self._config.camera_config.image_width // self._config.perspective_config.perspective_downsample_factor
        )
        if x.shape[2] != expected_h or x.shape[3] != expected_w:
            height_error = abs(x.shape[2] - expected_h) / expected_h * 100
            width_error = abs(x.shape[3] - expected_w) / expected_w * 100
            if max(height_error, width_error) > 10:
                raise ValueError(
                    f"Output size mismatch too large: got ({x.shape[2]}, {x.shape[3]}), "
                    f"expected ({expected_h}, {expected_w})",
                )
            x = cast(
                torch.Tensor,
                F.interpolate(x, size=(expected_h, expected_w), mode=mode),  # pyright: ignore[reportUnknownMemberType]
            )

        if self._modality == "depth":
            x = x.squeeze(1)
        return x

    def compute_loss(
        self,
        predictions: jt.Float[torch.Tensor, "batch ..."],
        label: jt.Shaped[torch.Tensor, "batch height width"],
    ) -> dict[str, torch.Tensor]:
        """
        Computes the loss for the decoder's modality.

        Args:
            predictions: prediction tensor from forward.
            label: ground-truth image for the modality — class indices for
                "semantic", depth values for "depth".

        Returns:
            dictionary of unweighted losses.
        """
        if self._modality == "semantic":
            loss_value = F.cross_entropy(predictions.float(), label.long())
        else:
            loss_value = F.l1_loss(predictions.float(), label.float())

        return {f"loss_{self._modality}": loss_value}

    def compute_metrics(
        self,
        predictions: jt.Float[torch.Tensor, "batch ..."],
        label: jt.Shaped[torch.Tensor, "batch height width"],
    ) -> dict[str, torch.Tensor]:
        """
        Quality metrics of the decoder's modality: mIoU and macro F1, or MAE.

        Args:
            predictions: prediction tensor from forward.
            label: ground-truth image for the modality — class indices for
                "semantic", depth values for "depth".

        Returns:
            dictionary of scalar metric tensors.
        """
        if self._modality == "semantic":
            num_classes = self._config.perspective_config.num_semantic_classes
            return {
                "semantic_miou": torchmetrics.functional.jaccard_index(
                    predictions.float(),
                    label.long(),
                    task="multiclass",
                    num_classes=num_classes,
                ),
                "semantic_f1": torchmetrics.functional.f1_score(
                    predictions.float(),
                    label.long(),
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                ),
            }
        return {
            "depth_mae": torchmetrics.functional.mean_absolute_error(
                predictions.float(),
                label.float(),
            ),
        }
