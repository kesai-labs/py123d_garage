# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

import math
from typing import Any, Final, cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
from py123d.geometry import Point2DIndex, PoseSE2Index
from torch import nn
from typing_extensions import override

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.datatypes.numerics import PositiveInt
from py123d_garage.policy.transfuser.network.decoder.transformer_decoder_layer import PlanningDecoderLayer

_TURNING_YAW_RAD: Final[float] = 0.5


_NORMALIZATION_CONSTANTS: Final[tuple[str, ...]] = (
    "ego_history_normalization_constants",
    "velocity_normalization_constants",
    "acceleration_normalization_constants",
    "target_points_normalization_constants",
)


def _drop_saved_normalization_constants(
    _module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    *_: object,
) -> None:
    """
    Drops the normalization constants a checkpoint saved while they were persistent buffers; the config defines them.

    Args:
        _module: the encoder loading the state dict.
        state_dict: the state dict about to load, edited in place.
        prefix: the encoder's key prefix in the state dict.
        *_: the rest of torch's pre-hook arguments.
    """
    for name in _NORMALIZATION_CONSTANTS:
        state_dict.pop(prefix + name, None)


class RegressionPlanningDecoder(nn.Module):
    """Transformer decoder predicting the future ego trajectory from BEV features."""

    def __init__(self, input_bev_channels: int, config: TransfuserConfig, input_register_channels: int = 0):
        """
        Initializes the planning decoder.

        Args:
            input_bev_channels: channels of the input BEV feature map.
            config: global config dataclass of TransFuser.
            input_register_channels: channels of the backbone's fusion registers; unused without registers.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config
        self.planning_context_encoder = PlanningContextEncoder(
            input_bev_channels=input_bev_channels,
            input_register_channels=input_register_channels,
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
            decoder_layer=PlanningDecoderLayer(
                config.planning_config.transfuser_token_dim,
                config.planning_config.transfuser_num_bev_cross_attention_heads,
                config.planning_config.gated_attention,
            ),
            num_layers=config.planning_config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(config.planning_config.transfuser_token_dim),
        )

        self.wp_decoder = nn.Linear(
            config.planning_config.transfuser_token_dim,
            len(PoseSE2Index),
        )

        nn.init.uniform_(self.query)

    @override
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "batch channels height width"],
        target_points: jt.Float[torch.Tensor, "batch num_points 2"],
        velocity: jt.Float[torch.Tensor, " batch"] | jt.Float[torch.Tensor, "batch 1"] | None = None,
        velocity_vector: jt.Float[torch.Tensor, "batch 2"] | None = None,
        acceleration_vector: jt.Float[torch.Tensor, "batch 2"] | None = None,
        ego_history: jt.Float[torch.Tensor, "batch steps 3"] | None = None,
        fusion_registers: jt.Float[torch.Tensor, "batch registers register_channels"] | None = None,
    ) -> jt.Float[torch.Tensor, "batch poses columns"]:
        """
        Predicts the future trajectory.

        Args:
            bev_features: BEV features from the backbone.
            target_points: ego-frame target points.
            velocity: ego speed; required if use_velocity.
            velocity_vector: ego velocity (v_x, v_y); required if use_velocity_vector.
            acceleration_vector: ego acceleration (a_x, a_y); required if use_acceleration_vector.
            ego_history: past ego poses; required if ego_history_duration_us is set.
            fusion_registers: the backbone's fusion register tokens; required with registers.

        Returns:
            future trajectory, (x, y, yaw) per waypoint.
        """
        context_tokens = self.planning_context_encoder(
            bev_features=bev_features,
            target_points=target_points,
            velocity=velocity,
            velocity_vector=velocity_vector,
            acceleration_vector=acceleration_vector,
            ego_history=ego_history,
            fusion_registers=fusion_registers,
        )

        batch_size = context_tokens.shape[0]
        queries = self.transformer_decoder(
            self.query.repeat(batch_size, 1, 1),
            context_tokens,
        )
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
            label: future ego poses, at least as many as predicted.

        Returns:
            dictionary of unweighted losses.
        """
        return {"loss_trajectory": pose_l1_loss(predictions, label[:, : self._config.trajectory_num_steps])}

    def compute_metrics(
        self,
        predictions: jt.Float[torch.Tensor, "batch poses columns"],
        label: jt.Float[torch.Tensor, "batch label_poses columns"],
    ) -> dict[str, torch.Tensor]:
        """
        Computes trajectory error metrics, overall and per speed and turning group; an empty group gives NaN.

        Args:
            predictions: predicted trajectory from forward.
            label: future ego poses, at least as many as predicted.

        Returns:
            dictionary of scalar metric tensors.
        """
        trajectory_label = label.float()[
            :,
            : self._config.trajectory_num_steps,
        ]
        trajectory_prediction = predictions.float()
        position_index = len(Point2DIndex)
        position_errors = trajectory_prediction[..., :position_index] - trajectory_label[..., :position_index]
        displacements = position_errors.pow(2).sum(dim=-1).sqrt()
        raw_yaw_difference_rad = trajectory_prediction[..., PoseSE2Index.YAW] - trajectory_label[..., PoseSE2Index.YAW]
        yaw_difference_rad = (raw_yaw_difference_rad + torch.pi) % (2 * torch.pi) - torch.pi
        label_yaw = trajectory_label[..., PoseSE2Index.YAW]
        along_track_errors = (
            position_errors[..., 0] * label_yaw.cos() + position_errors[..., 1] * label_yaw.sin()
        ).abs()
        cross_track_errors = (
            -position_errors[..., 0] * label_yaw.sin() + position_errors[..., 1] * label_yaw.cos()
        ).abs()
        first_step_s = self._config.trajectory_offsets_us[0] / 1e6
        label_speed_mps = trajectory_label[:, 0, :position_index].square().sum(dim=-1).sqrt() / first_step_s
        sample_groups = {
            "below_5mps": label_speed_mps < 5.0,
            "5_to_15mps": (label_speed_mps >= 5.0) & (label_speed_mps < 15.0),
            "above_15mps": label_speed_mps >= 15.0,
            "turning": label_yaw[:, -1].abs() > _TURNING_YAW_RAD,
        }
        metrics = {
            "waypoints_ade": displacements.mean(),
            "waypoints_fde": displacements[:, -1].mean(),
            "waypoints_aye": yaw_difference_rad.abs().mean(),
            "waypoints_along_track_error": along_track_errors.mean(),
            "waypoints_cross_track_error": cross_track_errors.mean(),
        }
        for group_name, in_group in sample_groups.items():
            group_weight = in_group.float()[:, None]
            num_group_values = group_weight.sum() * along_track_errors.shape[1]
            metrics[f"waypoints_along_track_error_{group_name}"] = (
                along_track_errors * group_weight
            ).sum() / num_group_values
            metrics[f"waypoints_cross_track_error_{group_name}"] = (
                cross_track_errors * group_weight
            ).sum() / num_group_values
        return metrics

    def planned_poses(
        self,
        predictions: jt.Float[torch.Tensor, "batch ..."],
    ) -> jt.Float[torch.Tensor, "batch poses 3"]:
        """
        Extracts the planned (x, y, yaw) per waypoint from a forward output.

        Args:
            predictions: output of forward.

        Returns:
            the planned poses.
        """
        return predictions[..., : len(PoseSE2Index)].float()


def pose_l1_loss(
    predictions: jt.Float[torch.Tensor, "batch rows 3"],
    label: jt.Float[torch.Tensor, "batch rows 3"],
) -> jt.Float[torch.Tensor, ""]:
    """
    Computes the L1 loss of the positions plus the L1 loss of the headings.

    Args:
        predictions: (x, y, heading) per row.
        label: labeled (x, y, heading) per row.

    Returns:
        the unweighted loss.
    """
    predictions = predictions.float()
    label = label.float()
    position_index = len(Point2DIndex)
    # The yaw is not wrapped on purpose: a wrapped loss lets the head settle full turns away from the label.
    return F.l1_loss(predictions[..., :position_index], label[..., :position_index]) + F.l1_loss(
        predictions[..., PoseSE2Index.YAW],
        label[..., PoseSE2Index.YAW],
    )


# A past pose reaches the encoder as x, y and the yaw's cosine and sine.
_EGO_HISTORY_ENCODER_COLUMNS: Final[int] = len(PoseSE2Index) + 1


class SineEmbedding(nn.Module):
    """DETR-style sine embedding of every input channel."""

    def __init__(self, embedding_dim: PositiveInt, base: float = 10_000.0):
        """
        Initializes the fixed wavelengths.

        Args:
            embedding_dim: number of features per input channel; must be even.
            base: ratio of the longest to the shortest wavelength.

        Raises:
            ValueError: if embedding_dim is odd.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if embedding_dim % 2:
            raise ValueError(f"embedding_dim pairs each sine with a cosine, so it must be even, got {embedding_dim}.")
        self.embedding_dim = embedding_dim
        index = torch.arange(embedding_dim, dtype=torch.float32)
        self.register_buffer("wavelengths", base ** (2 * (index // 2) / embedding_dim))

    @override
    def forward(
        self,
        features: jt.Float[torch.Tensor, "*batch channels"],
    ) -> jt.Float[torch.Tensor, "*batch embedded"]:
        """
        Embeds every channel as interleaved sine and cosine bands.

        Args:
            features: raw channels, expected in [-1, 1].

        Returns:
            the flattened sine and cosine bands of every channel.
        """
        scaled = features.unsqueeze(-1) * (2 * math.pi) / cast(torch.Tensor, self.wavelengths)
        embedded = torch.stack([scaled[..., 0::2].sin(), scaled[..., 1::2].cos()], dim=-1).flatten(start_dim=-2)
        return embedded.flatten(start_dim=-2)


class PlanningContextEncoder(nn.Module):
    """Builds the planning decoder's context: BEV tokens, optional registers, then ego-state and target point tokens."""

    def __init__(self, input_bev_channels: int, config: TransfuserConfig, input_register_channels: int = 0):
        """
        Initializes the planning context encoder.

        Args:
            input_bev_channels: channels of the input BEV feature map.
            config: global config dataclass of TransFuser.
            input_register_channels: channels of the backbone's fusion registers; unused without registers.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config

        # Only these parameters depend on the target point count, so checkpoints load across counts.
        num_target_points = len(config.required_target_point_distances_m)
        token_dim = config.planning_config.transfuser_token_dim
        self.tp_encoder = nn.Linear(2, token_dim) if num_target_points else None
        self.tp_pos_embedding = (
            nn.Parameter(torch.zeros(1, num_target_points, token_dim)) if num_target_points else None
        )
        self.tp_null_token = nn.Parameter(torch.zeros(1, 1, token_dim)) if num_target_points else None
        speed_dropout = config.planning_conditioning_config.speed_dropout_probability > 0.0
        assert not speed_dropout or config.planning_conditioning_config.use_velocity, "speed dropout needs use_velocity"
        self.speed_null_token = nn.Parameter(torch.zeros(1, 1, token_dim)) if speed_dropout else None
        self.num_status_tokens = 0

        if config.planning_conditioning_config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, config.planning_config.transfuser_token_dim),
            )

        if config.planning_conditioning_config.use_velocity_vector:
            self.num_status_tokens += 1
            self.velocity_vector_encoder = nn.Sequential(
                nn.Linear(2, config.planning_config.transfuser_token_dim),
            )

        if config.planning_conditioning_config.use_acceleration_vector:
            self.num_status_tokens += 1
            self.acceleration_vector_encoder = nn.Sequential(
                nn.Linear(2, config.planning_config.transfuser_token_dim),
            )

        num_history_steps = config.ego_history_num_steps
        single_history_token = config.planning_conditioning_config.ego_history_single_token
        self.num_status_tokens += min(num_history_steps, 1) if single_history_token else num_history_steps
        if num_history_steps:
            embedding_dim = config.planning_conditioning_config.ego_history_embedding_dim
            embedded_width = _EGO_HISTORY_ENCODER_COLUMNS * embedding_dim
            if single_history_token:
                # A hidden layer mixes the folded steps; a single linear layer could only sum them.
                self.ego_history_encoder = nn.Sequential(
                    SineEmbedding(embedding_dim),
                    nn.Flatten(start_dim=-2),
                    nn.Linear(embedded_width * num_history_steps, token_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(token_dim, token_dim),
                )
            else:
                self.ego_history_encoder = nn.Sequential(
                    SineEmbedding(embedding_dim),
                    nn.Linear(embedded_width, token_dim),
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
        self.num_status_tokens += num_target_points

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels,
            config.planning_config.transfuser_token_dim,
            kernel_size=1,
        )
        num_registers = config.backbone_config.num_fusion_registers
        self.register_adapter = (
            nn.Linear(input_register_channels, config.planning_config.transfuser_token_dim) if num_registers else None
        )
        self.register_pos_embedding = (
            nn.Parameter(torch.zeros(1, num_registers, config.planning_config.transfuser_token_dim))
            if num_registers
            else None
        )
        self.reset_parameters()

        # Divide each past position by max_speed_mps times its lookback, so all steps share one scale.
        lookback_s = torch.arange(config.ego_history_num_steps, 0, -1, dtype=torch.float32) * (
            config.planning_conditioning_config.ego_history_interval_us / 1e6
        )
        self.register_buffer(
            "ego_history_normalization_constants",
            (config.planning_conditioning_config.max_speed_mps * lookback_s)[:, None],
            persistent=False,
        )
        self.register_buffer(
            "velocity_normalization_constants",
            torch.tensor(
                config.planning_conditioning_config.velocity_normalization_constants,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "acceleration_normalization_constants",
            torch.tensor(
                config.planning_conditioning_config.acceleration_normalization_constants,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "target_points_normalization_constants",
            torch.tensor(
                config.planning_conditioning_config.target_points_normalization_constants,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_load_state_dict_pre_hook(_drop_saved_normalization_constants)  # pyright: ignore[reportUnknownMemberType]

    def reset_parameters(self) -> None:
        """Initializes the learnable embeddings; the target point parameters start near zero."""
        nn.init.uniform_(self.status_pos_embedding)
        if self.tp_encoder is not None:
            nn.init.zeros_(self.tp_encoder.weight)
            nn.init.zeros_(self.tp_encoder.bias)
        if self.tp_pos_embedding is not None:
            nn.init.normal_(self.tp_pos_embedding, std=0.02)
        if self.tp_null_token is not None:
            nn.init.normal_(self.tp_null_token, std=0.02)
        if self.speed_null_token is not None:
            nn.init.normal_(self.speed_null_token, std=0.02)
        if self.register_pos_embedding is not None:
            nn.init.uniform_(self.register_pos_embedding)

    @override
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "batch channels height width"],
        target_points: jt.Float[torch.Tensor, "batch num_points 2"],
        velocity: jt.Float[torch.Tensor, " batch"] | jt.Float[torch.Tensor, "batch 1"] | None = None,
        velocity_vector: jt.Float[torch.Tensor, "batch 2"] | None = None,
        acceleration_vector: jt.Float[torch.Tensor, "batch 2"] | None = None,
        ego_history: jt.Float[torch.Tensor, "batch steps 3"] | None = None,
        fusion_registers: jt.Float[torch.Tensor, "batch registers register_channels"] | None = None,
    ) -> jt.Float[torch.Tensor, "batch tokens token_dim"]:
        """
        Builds the context tokens for the planning decoder.

        Args:
            bev_features: raw BEV features.
            target_points: ego-frame target points; NaN marks a missing point, which reads as the null token.
            velocity: ego speed; required if use_velocity.
            velocity_vector: ego velocity (v_x, v_y); required if use_velocity_vector.
            acceleration_vector: ego acceleration (a_x, a_y); required if use_acceleration_vector.
            ego_history: past ego poses; required if ego_history_duration_us is set.
            fusion_registers: the backbone's fusion register tokens; required with registers.

        Returns:
            context tokens, BEV first, then the registers and the status tokens.
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
            speed_dropout = self._config.planning_conditioning_config.speed_dropout_probability
            if self.speed_null_token is not None and (self.training or speed_dropout >= 1.0):
                dropped = torch.rand(velocity_token.shape[0], 1, 1, device=velocity_token.device) < speed_dropout
                velocity_token = torch.where(dropped, self.speed_null_token.to(velocity_token.dtype), velocity_token)
            status_tokens.append(velocity_token)

        conditioning = self._config.planning_conditioning_config
        if conditioning.use_velocity_vector:
            assert velocity_vector is not None, "Velocity vector input is required when use_velocity_vector is enabled!"
            normalized = velocity_vector.float() / cast(torch.Tensor, self.velocity_normalization_constants)
            status_tokens.append(
                self.velocity_vector_encoder(normalized).reshape(
                    -1,
                    1,
                    self._config.planning_config.transfuser_token_dim,
                ),
            )

        if conditioning.use_acceleration_vector:
            assert acceleration_vector is not None, (
                "Acceleration vector input is required when use_acceleration_vector is enabled!"
            )
            normalized = acceleration_vector.float() / cast(torch.Tensor, self.acceleration_normalization_constants)
            status_tokens.append(
                self.acceleration_vector_encoder(normalized).reshape(
                    -1,
                    1,
                    self._config.planning_config.transfuser_token_dim,
                ),
            )

        num_history_steps = self._config.ego_history_num_steps
        if num_history_steps:
            assert ego_history is not None, "Ego history input is required when ego_history_duration_us is set!"
            assert ego_history.shape[1] == num_history_steps, (
                f"Expected {num_history_steps} past ego poses, got {ego_history.shape[1]}!"
            )
            # A missing past pose is NaN and reads as the anchor pose.
            history = torch.nan_to_num(ego_history.float(), nan=0.0)
            # Clamp so positions past the speed limit do not wrap around in the sine embedding.
            position = (
                history[..., PoseSE2Index.XY]
                / cast(
                    torch.Tensor,
                    self.ego_history_normalization_constants,
                )
            ).clamp(-1.0, 1.0)
            yaw = history[..., PoseSE2Index.YAW]
            heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
            encoded = self.ego_history_encoder(torch.cat([position, heading], dim=-1))
            status_tokens.append(encoded.unsqueeze(1) if conditioning.ego_history_single_token else encoded)

        num_target_points = len(self._config.required_target_point_distances_m)
        assert target_points.shape[1] == num_target_points, (
            f"Expected {num_target_points} target points, got {target_points.shape[1]}!"
        )
        status_token_tensor = torch.cat(status_tokens, dim=1) + self.status_pos_embedding if status_tokens else None
        if self.tp_encoder is not None and self.tp_pos_embedding is not None and self.tp_null_token is not None:
            missing = target_points.isnan().any(dim=-1, keepdim=True)  # (bs, num_target_points, 1)
            target_points = torch.nan_to_num(target_points.float(), nan=0.0)
            target_points = target_points / cast(
                torch.Tensor,
                self.target_points_normalization_constants,
            )
            tp_tokens = self.tp_encoder(
                target_points,
            )  # (bs, num_target_points, transfuser_token_dim)
            tp_tokens = torch.where(missing, self.tp_null_token, tp_tokens)
            tp_tokens = tp_tokens + self.tp_pos_embedding
            status_token_tensor = (
                tp_tokens if status_token_tensor is None else torch.cat([status_token_tensor, tp_tokens], dim=1)
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

        tokens = [context_tokens]
        if self.register_adapter is not None and self.register_pos_embedding is not None:
            assert fusion_registers is not None, "The backbone's registers are required with num_fusion_registers!"
            tokens.append(self.register_adapter(fusion_registers) + self.register_pos_embedding)
        if status_token_tensor is not None:
            tokens.append(status_token_tensor)
        return torch.cat(tokens, dim=1)


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
            num_pos_feats: positional features per axis, half the embedding dim.
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
        Computes the positional embedding of a feature map.

        Args:
            tensor: feature map, used for its shape, device and dtype.

        Returns:
            positional embedding with 2 * num_pos_feats channels.
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
