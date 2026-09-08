# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

import math
from typing import cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
from py123d.geometry import Point2DIndex, PoseSE2Index
from torch import nn
from typing_extensions import override

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)


class PlanningDecoder(nn.Module):
    """Transformer decoder predicting the future ego trajectory from BEV features."""

    def __init__(self, input_bev_channels: int, config: TransfuserConfig):
        """
        Initializes the planning decoder.

        Args:
            input_bev_channels: number of channels of the input BEV feature map.
            config: global config dataclass of TransFuser.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config
        self.planning_context_encoder = PlanningContextEncoder(
            input_bev_channels=input_bev_channels,
            config=config,
        )

        self.query = nn.Parameter(
            torch.zeros(
                1,
                config.trajectory_num_steps,
                config.planning_config.transfuser_token_dim,
            ),
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                config.planning_config.transfuser_token_dim,
                config.planning_config.transfuser_num_bev_cross_attention_heads,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=config.planning_config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(config.planning_config.transfuser_token_dim),
        )

        self.wp_decoder = nn.Linear(
            config.planning_config.transfuser_token_dim,
            len(PoseSE2Index) if config.planning_config.predict_yaw else len(Point2DIndex),
        )

        nn.init.uniform_(self.query)

    @override
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "batch channels height width"],
        target_points: jt.Float[torch.Tensor, "batch num_points 2"],
        velocity: jt.Float[torch.Tensor, " batch"] | jt.Float[torch.Tensor, "batch 1"] | None = None,
    ) -> jt.Float[torch.Tensor, "batch poses columns"]:
        """
        Forward pass of the planning decoder.

        Args:
            bev_features: BEV features from the backbone.
            target_points: ego-frame target points.
            velocity: ego velocity; required if use_velocity.

        Returns:
            future trajectory, (x, y) per waypoint plus yaw when predict_yaw.
        """
        context_tokens = self.planning_context_encoder(
            bev_features=bev_features,
            target_points=target_points,
            velocity=velocity,
        )

        batch_size = context_tokens.shape[0]
        queries = self.transformer_decoder(
            self.query.repeat(batch_size, 1, 1),
            context_tokens,
        )
        # Per-step deltas, accumulated into ego-frame poses.
        return torch.cumsum(self.wp_decoder(queries), 1)

    def compute_loss(
        self,
        predictions: jt.Float[torch.Tensor, "batch poses columns"],
        label: jt.Float[torch.Tensor, "batch label_poses columns"],
    ) -> dict[str, torch.Tensor]:
        """
        Computes the trajectory loss.

        Args:
            predictions: predicted trajectory from forward.
            label: future ego poses, at least the predicted number of poses.

        Returns:
            dictionary of unweighted losses.
        """
        trajectory_label = label.float()[
            :,
            : self._config.trajectory_num_steps,
        ]
        trajectory_prediction = predictions.float()
        position_index = len(Point2DIndex)
        loss = F.l1_loss(
            trajectory_prediction[..., :position_index],
            trajectory_label[..., :position_index],
            reduction="none",
        ).mean()
        if self._config.planning_config.predict_yaw:
            # Both yaws are wrapped, so their raw difference jumps by 2pi across the
            # +-pi seam; wrapping the difference measures the angle between them.
            raw_yaw_difference_rad = (
                trajectory_prediction[..., PoseSE2Index.YAW] - trajectory_label[..., PoseSE2Index.YAW]
            )
            yaw_difference_rad = (raw_yaw_difference_rad + torch.pi) % (2 * torch.pi) - torch.pi
            loss = loss + yaw_difference_rad.abs().mean()
        return {"loss_trajectory": loss}

    def compute_metrics(
        self,
        predictions: jt.Float[torch.Tensor, "batch poses columns"],
        label: jt.Float[torch.Tensor, "batch label_poses columns"],
    ) -> dict[str, torch.Tensor]:
        """
        Trajectory quality: average and final displacement error in meters.

        Args:
            predictions: predicted trajectory from forward.
            label: future ego poses, at least the predicted number of poses.

        Returns:
            dictionary of scalar metric tensors.
        """
        trajectory_label = label.float()[
            :,
            : self._config.trajectory_num_steps,
        ]
        trajectory_prediction = predictions.float()
        position_index = len(Point2DIndex)
        displacements = (
            (trajectory_prediction[..., :position_index] - trajectory_label[..., :position_index])
            .pow(2)
            .sum(dim=-1)
            .sqrt()
        )
        metrics = {
            "waypoints_ade": displacements.mean(),
            "waypoints_fde": displacements[:, -1].mean(),
        }
        if self._config.planning_config.predict_yaw:
            raw_yaw_difference_rad = (
                trajectory_prediction[..., PoseSE2Index.YAW] - trajectory_label[..., PoseSE2Index.YAW]
            )
            yaw_difference_rad = (raw_yaw_difference_rad + torch.pi) % (2 * torch.pi) - torch.pi
            metrics["waypoints_aye"] = yaw_difference_rad.abs().mean()
        return metrics


class PlanningContextEncoder(nn.Module):
    """
    Builds the token sequence attended by the planning decoder.

    Tokens are the flattened BEV features with a cosine positional embedding, plus
    status tokens: one per target point and optionally one for the ego velocity.
    """

    def __init__(self, input_bev_channels: int, config: TransfuserConfig):
        """
        Initializes the planning context encoder.

        Args:
            input_bev_channels: number of channels of the input BEV feature map.
            config: global config dataclass of TransFuser.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config

        # One status token per target point (+ optionally the ego velocity).
        self.num_status_tokens = len(config.required_target_point_distances_m)
        self.tp_encoder = nn.Linear(
            2,
            config.planning_config.transfuser_token_dim,
        )

        if config.planning_conditioning_config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, config.planning_config.transfuser_token_dim),
            )

        self.cosine_pos_embedding = PositionEmbeddingSine(
            config.planning_config.transfuser_token_dim // 2,
            normalize=True,
        )
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_status_tokens,
                config.planning_config.transfuser_token_dim,
            ),
        )

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels,
            config.planning_config.transfuser_token_dim,
            kernel_size=1,
        )
        self.reset_parameters()

        self.register_buffer(
            "target_points_normalization_constants",
            torch.tensor(
                config.planning_conditioning_config.target_points_normalization_constants,
                dtype=torch.float32,
            ),
        )

    def reset_parameters(self) -> None:
        """Initializes the learnable status positional embedding."""
        nn.init.uniform_(self.status_pos_embedding)

    @override
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "batch channels height width"],
        target_points: jt.Float[torch.Tensor, "batch num_points 2"],
        velocity: jt.Float[torch.Tensor, " batch"] | jt.Float[torch.Tensor, "batch 1"] | None = None,
    ) -> jt.Float[torch.Tensor, "batch tokens token_dim"]:
        """
        Builds the context tokens for the planning transformer decoder.

        Args:
            bev_features: raw BEV features.
            target_points: ego-frame target points. NaN-padded entries
                (missing points) are treated as the origin.
            velocity: ego velocity; required if use_velocity.

        Returns:
            context tokens of shape [B, H * W + num_status_tokens, transfuser_token_dim].
        """
        status_tokens: list[torch.Tensor] = []

        if self._config.planning_conditioning_config.use_velocity:
            assert velocity is not None, "Velocity input is required when use_velocity is enabled!"
            velocity = velocity.reshape(-1, 1).float() / self._config.planning_conditioning_config.max_speed_mps
            velocity_token = self.velocity_encoder(velocity).reshape(
                -1,
                1,
                self._config.planning_config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(velocity_token)

        # Encode target points; NaN padding marks missing points and is mapped to the origin.
        num_target_points = len(self._config.required_target_point_distances_m)
        assert target_points.shape[1] == num_target_points, (
            f"Expected {num_target_points} target points, got {target_points.shape[1]}!"
        )
        target_points = torch.nan_to_num(target_points.float(), nan=0.0)
        target_points = target_points / cast(
            torch.Tensor,
            self.target_points_normalization_constants,
        )
        tp_tokens = self.tp_encoder(
            target_points,
        )  # (bs, num_target_points, transfuser_token_dim)
        status_tokens.append(tp_tokens)

        status_token_tensor = torch.cat(
            status_tokens,
            dim=1,
        )  # (bs, num_status_tokens, transfuser_token_dim)

        context_tokens = self.dimension_adapter(
            bev_features,
        )  # (bs, transfuser_token_dim, height, width)

        context_tokens = context_tokens + self.cosine_pos_embedding(
            context_tokens,
        )
        context_tokens = torch.flatten(
            context_tokens,
            start_dim=2,
        )  # (bs, transfuser_token_dim, height * width)
        context_tokens = torch.permute(
            context_tokens,
            (0, 2, 1),
        )  # (bs, height * width, transfuser_token_dim)

        status_token_tensor = status_token_tensor + self.status_pos_embedding
        return torch.cat([context_tokens, status_token_tensor], dim=1)


class PositionEmbeddingSine(nn.Module):
    """2D sine-cosine positional embedding for BEV feature maps (DETR-style)."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        """
        Initializes the sine positional embedding.

        Args:
            num_pos_feats: number of positional features per axis (half the embedding dim).
            temperature: frequency temperature of the sine embedding.
            normalize: whether to normalize coordinates to [0, scale].
            scale: coordinate scale; requires normalize, defaults to 2 * pi.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @override
    def forward(
        self,
        tensor: jt.Float[torch.Tensor, "batch channels height width"],
    ) -> jt.Float[torch.Tensor, "batch embed height width"]:
        """
        Computes the positional embedding for a feature map.

        Args:
            tensor: feature map of shape [B, C, H, W].

        Returns:
            positional embedding of shape [B, 2 * num_pos_feats, H, W].
        """
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device,
        )
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(tensor.dtype).contiguous()
