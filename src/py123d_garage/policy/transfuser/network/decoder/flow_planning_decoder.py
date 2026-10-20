# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Flow-matching planning decoder that denoises waypoints, unicycle actions and path points jointly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, NamedTuple, cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
from py123d.geometry import Point2DIndex, PoseSE2Index
from torch import nn
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.datatypes.numerics import PositiveFloat, PositiveInt
from py123d_garage.policy.transfuser.network.decoder.regression_planning_decoder import (
    RegressionPlanningDecoder,
    pose_l1_loss,
)
from py123d_garage.policy.transfuser.network.decoder.trajectory_normalization import (
    PATH_MEAN,
    PATH_STD,
    pose_statistics,
)
from py123d_garage.policy.transfuser.network.decoder.transformer_decoder_layer import PlanningDecoderLayer
from py123d_garage.policy.transfuser.network.decoder.unicycle import (
    STEP_S,
    actions_to_state,
    rollout,
    state_to_actions,
)

# The shortest lookahead the normalized time axis resolves.
_TIME_FLOOR_S: Final[float] = 0.1

# Scales normalized time to the DDPM step range [0, 1000] the embedding expects.
_TIME_SCALE: Final[float] = 1_000.0


class SinusoidalTimeEmbedding(nn.Module):
    """DDPM-style sinusoidal embedding of a scalar time, mixed by a two-layer MLP."""

    def __init__(self, embedding_dim: PositiveInt, max_period: PositiveFloat = 10_000.0) -> None:
        """
        Initializes the fixed frequencies and the mixing MLP.

        Args:
            embedding_dim: number of output features; must be even.
            max_period: ratio of the longest to the shortest wavelength.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        half_dim = embedding_dim // 2
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)),
        )
        self.projection_in = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.projection_out = nn.Linear(4 * embedding_dim, embedding_dim)

        # Zero init so the embedding starts as a no-op.
        nn.init.zeros_(self.projection_out.weight)
        nn.init.zeros_(self.projection_out.bias)

    @override
    def forward(
        self,
        time: jt.Float[torch.Tensor, " steps"],
    ) -> jt.Float[torch.Tensor, "steps embedding"]:
        """
        Embeds each time as sine and cosine bands, then mixes them.

        Args:
            time: scalar time of each pose.

        Returns:
            the embedding of every time.
        """
        scaled = time[:, None] * cast(torch.Tensor, self.frequencies)[None, :]
        embedded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return self.projection_out(F.mish(self.projection_in(embedded)))


# Scales the flow time in [0, 1] like the pose times.
_FLOW_TIME_SCALE: Final[float] = 1_000.0
# Shift, scale and gate for each of self-attention, cross-attention and feed-forward.
_NUM_LAYER_MODULATIONS: Final[int] = 9


class AdaptiveNormDecoderLayer(PlanningDecoderLayer):
    """
    PlanningDecoderLayer whose norms and residual branches are modulated by a conditioning vector.

    It loads the plain layer's weights, and its zero-init modulation starts as that layer.
    """

    def __init__(self, token_dim: PositiveInt, num_heads: PositiveInt, gated_attention: bool = False) -> None:
        """
        Initializes the layer and its zero-init modulation.

        Args:
            token_dim: width of the queries, the memory and the conditioning.
            num_heads: attention heads of both attention blocks.
            gated_attention: gate both attention outputs by a sigmoid of their queries.
        """
        super().__init__(token_dim, num_heads, gated_attention)
        self.modulation = nn.Linear(token_dim, _NUM_LAYER_MODULATIONS * token_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        tgt: jt.Float[torch.Tensor, "batch queries dim"],
        memory: jt.Float[torch.Tensor, "memory_batch tokens dim"],
        conditioning: jt.Float[torch.Tensor, "batch dim"],
        self_attention_mask: jt.Bool[torch.Tensor, "queries queries"] | None = None,
        cross_attention_mask: jt.Bool[torch.Tensor, "queries tokens"] | None = None,
    ) -> jt.Float[torch.Tensor, "batch queries dim"]:
        """
        Runs self-attention, cross-attention and feed-forward, each modulated by the conditioning.

        Args:
            tgt: the queries.
            memory: the tokens the queries attend to.
            conditioning: the vector the modulation is computed from.
            self_attention_mask: True where a query may attend to another query; None allows all.
            cross_attention_mask: True where a query may attend to a memory token; None allows all.

        Returns:
            the updated queries.
        """
        (
            self_shift,
            self_scale,
            self_gate,
            cross_shift,
            cross_scale,
            cross_gate,
            feed_forward_shift,
            feed_forward_scale,
            feed_forward_gate,
        ) = self.modulation(F.silu(conditioning))[:, None].chunk(_NUM_LAYER_MODULATIONS, dim=-1)
        x = tgt
        x = x + (1.0 + self_gate) * self._sa_block(  # pyright: ignore[reportPrivateUsage]
            self.norm1(x) * (1.0 + self_scale) + self_shift,
            self_attention_mask,
            None,
        )
        x = x + (1.0 + cross_gate) * self._mha_block(  # pyright: ignore[reportPrivateUsage]
            self.norm2(x) * (1.0 + cross_scale) + cross_shift,
            memory,
            cross_attention_mask,
            None,
        )
        return x + (1.0 + feed_forward_gate) * self._ff_block(  # pyright: ignore[reportPrivateUsage]
            self.norm3(x) * (1.0 + feed_forward_scale) + feed_forward_shift,
        )


# (x, (output, gate, dropout) or None, norm, shift or None, scale or None) to the residual stream after the residual
# and the normalized (and modulated) rows; a planning kernel patch's residual, gate, residual dropout, norm and
# modulation in one.
FusedAdaptiveNorm = Callable[
    [
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, nn.Dropout] | None,
        nn.LayerNorm,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


class AdaptiveNormTransformerDecoder(nn.Module):
    """A stack of AdaptiveNormDecoderLayer with a final norm, named like nn.TransformerDecoder."""

    fused_adaptive_norm: FusedAdaptiveNorm | None = None

    def __init__(
        self,
        token_dim: PositiveInt,
        num_heads: PositiveInt,
        num_layers: PositiveInt,
        gated_attention: bool = False,
    ) -> None:
        """
        Initializes the layers and the final norm.

        Args:
            token_dim: width of the queries, the memory and the conditioning.
            num_heads: attention heads per attention block.
            num_layers: number of decoder layers.
            gated_attention: gate every attention output by a sigmoid of its queries.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.layers = nn.ModuleList(
            AdaptiveNormDecoderLayer(token_dim, num_heads, gated_attention) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(token_dim)

    @override
    def forward(
        self,
        tgt: jt.Float[torch.Tensor, "batch queries dim"],
        memory: jt.Float[torch.Tensor, "memory_batch tokens dim"],
        conditioning: jt.Float[torch.Tensor, "batch dim"],
        self_attention_mask: jt.Bool[torch.Tensor, "queries queries"] | None = None,
        cross_attention_mask: jt.Bool[torch.Tensor, "queries tokens"] | None = None,
    ) -> jt.Float[torch.Tensor, "batch queries dim"]:
        """
        Runs every layer under the same conditioning and masks, then the final norm.

        Args:
            tgt: the queries.
            memory: the tokens the queries attend to.
            conditioning: the vector every layer's modulation is computed from.
            self_attention_mask: True where a query may attend to another query; None allows all.
            cross_attention_mask: True where a query may attend to a memory token; None allows all.

        Returns:
            the decoded queries.
        """
        if self.fused_adaptive_norm is not None:
            return self._fused_forward(
                self.fused_adaptive_norm,
                tgt,
                memory,
                conditioning,
                self_attention_mask,
                cross_attention_mask,
            )
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, conditioning, self_attention_mask, cross_attention_mask)
        return self.norm(x)

    def _fused_forward(
        self,
        adaptive_norm: FusedAdaptiveNorm,
        tgt: jt.Float[torch.Tensor, "batch queries dim"],
        memory: jt.Float[torch.Tensor, "memory_batch tokens dim"],
        conditioning: jt.Float[torch.Tensor, "batch dim"],
        self_attention_mask: jt.Bool[torch.Tensor, "queries queries"] | None,
        cross_attention_mask: jt.Bool[torch.Tensor, "queries tokens"] | None,
    ) -> jt.Float[torch.Tensor, "batch queries dim"]:
        """
        Runs the layers with each sublayer's residual, gate, residual dropout, norm and modulation in one adaptive_norm
        call, which adds a sublayer's output to the residual stream as it normalizes the next sublayer's input.

        Args:
            adaptive_norm: (x, (output, gate, dropout) or None, norm, shift or None, scale or None) to the residual
                stream and the normalized (and modulated) rows.
            tgt: the queries.
            memory: the tokens the queries attend to.
            conditioning: the vector every layer's modulation is computed from.
            self_attention_mask: True where a query may attend to another query; None allows all.
            cross_attention_mask: True where a query may attend to a memory token; None allows all.

        Returns:
            the decoded queries.
        """
        x, residual = tgt, None
        for layer in cast("list[AdaptiveNormDecoderLayer]", list(self.layers)):
            (
                self_shift,
                self_scale,
                self_gate,
                cross_shift,
                cross_scale,
                cross_gate,
                feed_forward_shift,
                feed_forward_scale,
                feed_forward_gate,
            ) = layer.modulation(F.silu(conditioning)).chunk(_NUM_LAYER_MODULATIONS, dim=-1)
            x, y = adaptive_norm(x, residual, layer.norm1, self_shift, self_scale)
            residual = (layer._sa_block(y, self_attention_mask, None, dropout=False), self_gate, layer.dropout1)  # pyright: ignore[reportPrivateUsage]
            x, y = adaptive_norm(x, residual, layer.norm2, cross_shift, cross_scale)
            residual = (
                layer._mha_block(y, memory, cross_attention_mask, None, dropout=False),  # pyright: ignore[reportPrivateUsage]
                cross_gate,
                layer.dropout2,
            )
            x, y = adaptive_norm(x, residual, layer.norm3, feed_forward_shift, feed_forward_scale)
            residual = (layer._ff_block(y, dropout=False), feed_forward_gate, layer.dropout3)  # pyright: ignore[reportPrivateUsage]
        _, decoded = adaptive_norm(x, residual, self.norm, None, None)
        return decoded


_NUM_ACTIONS: Final[int] = 2
_DRIVEN_OUTPUTS: Final[tuple[str, ...]] = ("waypoints", "actions", "waypoints_on_path", "actions_on_path")

# Column layout of a sampled step row; path rows hold (x, y, route heading) and then zeros.
_DRIVEN_COLUMNS: Final[slice] = slice(0, 3)
WAYPOINT_COLUMNS: Final[slice] = slice(3, 6)
ROLLOUT_COLUMNS: Final[slice] = slice(6, 9)
_ACTION_COLUMNS: Final[slice] = slice(9, 11)
_NUM_SAMPLE_COLUMNS: Final[int] = 11


class _Outputs(NamedTuple):
    """Every output of one sampled plan; None for a head the config leaves out."""

    # Predicted (x, y, yaw) per trajectory step.
    waypoints: jt.Float[torch.Tensor, "batch poses 3"] | None
    # Normalized (acceleration, curvature) per trajectory step.
    actions: jt.Float[torch.Tensor, "batch poses 2"] | None
    # (x, y, yaw) per trajectory step of the actions' unicycle rollout.
    rollout: jt.Float[torch.Tensor, "batch poses 3"] | None
    # (x, y, route heading) per path point.
    path: jt.Float[torch.Tensor, "batch path_points 3"] | None


class FlowPlanningDecoder(RegressionPlanningDecoder):
    """
    Flow matching that denoises waypoint, unicycle action and path point tokens together in one transformer.

    Under autograd forward returns the context tokens plus a speed token; without autograd it returns sampled rows.
    """

    def __init__(self, input_bev_channels: int, config: TransfuserConfig, input_register_channels: int = 0) -> None:
        """
        Initializes the flow-time conditioned transformer and the waypoint, action and path heads.

        Args:
            input_bev_channels: channels of the input BEV feature map.
            config: global config dataclass of TransFuser.
            input_register_channels: channels of the backbone's fusion registers; unused without registers.
        """
        super().__init__(input_bev_channels, config, input_register_channels)
        planning = config.planning_config
        token_dim = planning.transfuser_token_dim
        self.time_embedding = SinusoidalTimeEmbedding(token_dim)
        pose_times_s = torch.tensor(config.trajectory_offsets_us, dtype=torch.float32) * 1e-6
        horizon_s = config.trajectory_horizon_us / 1e6
        # A log axis gives near and far poses the same relative resolution.
        normalized_time = torch.log(pose_times_s / _TIME_FLOOR_S) / math.log(horizon_s / _TIME_FLOOR_S)
        self.register_buffer("query_times", normalized_time * _TIME_SCALE)
        if planning.predict_waypoints:
            self.state_encoder = nn.Linear(len(PoseSE2Index), token_dim)
        else:
            # Unused parameters would hold no gradient; the base class builds the waypoint head unconditionally.
            del self.query
            del self.wp_decoder
        self.flow_time_embedding = SinusoidalTimeEmbedding(token_dim)
        self.transformer_decoder = AdaptiveNormTransformerDecoder(
            token_dim,
            planning.transfuser_num_bev_cross_attention_heads,
            planning.transfuser_num_bev_cross_attention_layers,
            planning.gated_attention,
        )
        self.adaptive_norm_time_embedding = SinusoidalTimeEmbedding(token_dim)
        # The modulation is already zero-init; a zero embedding too would give its weights no gradient.
        self.adaptive_norm_time_embedding.projection_out.reset_parameters()

        pose_mean, pose_std = pose_statistics(list(config.trajectory_offsets_us))
        self.register_buffer("pose_scale", torch.tensor(pose_std, dtype=torch.float32), persistent=False)
        self.register_buffer("pose_mean", torch.tensor(pose_mean, dtype=torch.float32), persistent=False)

        path_config = planning.path_config
        action_config = planning.unicycle_action_config
        flow = planning.flow_config
        assert flow.driven_output in _DRIVEN_OUTPUTS, f"driven_output must be one of {_DRIVEN_OUTPUTS}"
        assert action_config is not None or not flow.driven_output.startswith("actions"), (
            f"driven_output={flow.driven_output} needs planning_config.unicycle_action_config"
        )
        assert planning.predict_waypoints or not flow.driven_output.startswith("waypoints"), (
            f"driven_output={flow.driven_output} needs planning_config.predict_waypoints"
        )
        assert path_config is not None or not flow.driven_output.endswith("_on_path"), (
            f"driven_output={flow.driven_output} needs planning_config.path_config"
        )
        assert config.planning_conditioning_config.use_velocity, "the action rollouts start from the speed input"
        if action_config is not None:
            assert action_config.action_loss or action_config.rollout_pose_loss, (
                "without action_loss the rollout pose loss is the only action loss term"
            )
            assert not action_config.bounded or action_config.rollout_pose_loss, (
                "bounded trains through the rollout pose loss"
            )
            assert not action_config.bounded or action_config.lateral_accel_actions, (
                "bounded squashes the lateral-acceleration actions"
            )
            step_us = round(STEP_S * 1e6)
            expected_offsets_us = uniform_offsets_us(step_us, step_us * config.trajectory_num_steps)
            assert list(config.trajectory_offsets_us) == expected_offsets_us, (
                f"the unicycle rollout needs trajectory_offsets_us {expected_offsets_us}"
            )
        self._driven_output = flow.driven_output
        self._waypoints_attend_to_actions = flow.waypoints_attend_to_actions
        self._actions_attend_to_waypoints = flow.actions_attend_to_waypoints
        self._waypoints_and_actions_attend_to_target_points = flow.waypoints_and_actions_attend_to_target_points
        self._action_config = action_config
        self._num_waypoint_steps = config.trajectory_num_steps if planning.predict_waypoints else 0
        self._num_action_steps = config.trajectory_num_steps if action_config is not None else 0
        self._num_path_points = path_config.num_points if path_config is not None else 0
        token_dim = planning.transfuser_token_dim

        if action_config is not None:
            self.action_query = nn.Parameter(torch.zeros(1, config.trajectory_num_steps, token_dim))
            nn.init.uniform_(self.action_query)
            self.action_state_encoder = nn.Linear(_NUM_ACTIONS, token_dim)
            self.action_decoder = nn.Linear(token_dim, _NUM_ACTIONS)

        if path_config is not None:
            num_points = path_config.num_points
            assert num_points <= len(PATH_MEAN), f"trajectory_normalization holds {len(PATH_MEAN)} path points"
            self._num_path_points_first_10m = math.ceil(10.0 / path_config.spacing_m)
            self.path_query = nn.Parameter(torch.zeros(1, num_points, token_dim))
            nn.init.uniform_(self.path_query)
            self.path_distance_embedding = SinusoidalTimeEmbedding(token_dim)
            distances_m = path_config.spacing_m * torch.arange(1, num_points + 1, dtype=torch.float32)
            normalized_distance = torch.log(distances_m / path_config.spacing_m) / math.log(num_points)
            self.register_buffer("path_query_distances", normalized_distance * _TIME_SCALE, persistent=False)
            self.path_state_encoder = nn.Linear(len(PoseSE2Index), token_dim)
            self.path_decoder = nn.Linear(token_dim, len(PoseSE2Index))
            self.register_buffer(
                "path_scale",
                torch.tensor(PATH_STD[:num_points], dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "path_mean",
                torch.tensor(PATH_MEAN[:num_points], dtype=torch.float32),
                persistent=False,
            )

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
    ) -> jt.Float[torch.Tensor, "batch ..."]:
        """
        Encodes the planning context and appends the speed token; samples the plan when autograd is off.

        Args:
            bev_features: BEV features from the backbone.
            target_points: ego-frame target points.
            velocity: ego speed, where the action rollouts start.
            velocity_vector: ego velocity (v_x, v_y); required if use_velocity_vector.
            acceleration_vector: ego acceleration (a_x, a_y); required if use_acceleration_vector.
            ego_history: past ego poses; required if ego_history_duration_us is set.
            fusion_registers: the backbone's fusion register tokens; required with registers.

        Returns:
            the context tokens and the speed token under autograd, otherwise the sampled rows.
        """
        assert velocity is not None, "the action rollouts start from the speed input"
        context_tokens = self.planning_context_encoder(
            bev_features=bev_features,
            target_points=target_points,
            velocity=velocity,
            velocity_vector=velocity_vector,
            acceleration_vector=acceleration_vector,
            ego_history=ego_history,
            fusion_registers=fusion_registers,
        )
        # Keep fp32, since a bf16 speed is about 0.1 m/s off at highway speed.
        speed_token = velocity.reshape(-1, 1, 1).float().expand(-1, 1, context_tokens.shape[-1])
        with_speed = torch.cat([context_tokens.float(), speed_token], dim=1)
        if torch.is_grad_enabled():
            return with_speed
        return self.sample(with_speed)

    def _velocity(
        self,
        context_tokens: jt.Float[torch.Tensor, "batch tokens dim"],
        state: jt.Float[torch.Tensor, "draws 1 state"],
        flow_time: jt.Float[torch.Tensor, " draws"],
    ) -> jt.Float[torch.Tensor, "draws 1 state"]:
        """
        Predicts the flow velocity of a noisy state.

        Args:
            context_tokens: the planning context, without the speed token.
            state: the noisy normalized flow state.
            flow_time: flow time per sample, 0 at noise and 1 at the clean state.

        Returns:
            the velocity of the flow state.
        """
        # Query order: waypoints, actions, path points; a head the config leaves out has no queries.
        num_waypoint_queries = self._num_waypoint_steps
        num_heads_queries = num_waypoint_queries + self._num_action_steps
        waypoint_state, action_state, path_state = self._unpack(state)
        step_times = self.time_embedding(cast(torch.Tensor, self.query_times))[None]
        query_groups: list[torch.Tensor] = []
        if waypoint_state is not None:
            query_groups.append(self.query + step_times + self.state_encoder(waypoint_state))
        if action_state is not None:
            query_groups.append(self.action_query + step_times + self.action_state_encoder(action_state))
        if path_state is not None:
            path_distances = self.path_distance_embedding(cast(torch.Tensor, self.path_query_distances))[None]
            query_groups.append(self.path_query + path_distances + self.path_state_encoder(path_state))
        queries = torch.cat(query_groups, dim=1) + self.flow_time_embedding(flow_time * _FLOW_TIME_SCALE)[:, None]
        self_attention_mask, cross_attention_mask = None, None
        if path_state is not None:
            self_attention_mask, cross_attention_mask = path_attention_masks(
                self._config,
                num_waypoint_queries,
                self._num_action_steps,
                context_tokens.shape[1],
                queries.device,
            )
        num_queries, num_context_tokens = queries.shape[1], context_tokens.shape[1]
        if (
            waypoint_state is not None
            and action_state is not None
            and (not self._waypoints_attend_to_actions or not self._actions_attend_to_waypoints)
        ):
            if self_attention_mask is None:
                self_attention_mask = torch.ones(num_queries, num_queries, dtype=torch.bool, device=queries.device)
            if not self._waypoints_attend_to_actions:
                self_attention_mask[:num_waypoint_queries, num_waypoint_queries:num_heads_queries] = False
            if not self._actions_attend_to_waypoints:
                self_attention_mask[num_waypoint_queries:num_heads_queries, :num_waypoint_queries] = False
        num_target_points = len(self._config.required_target_point_distances_m)
        # Without target points there is nothing to hide, and an all-True mask only slows the attention kernel.
        if not self._waypoints_and_actions_attend_to_target_points and num_target_points:
            if cross_attention_mask is None:
                cross_attention_mask = torch.ones(
                    num_queries,
                    num_context_tokens,
                    dtype=torch.bool,
                    device=queries.device,
                )
            # The target points are the last context tokens.
            first_target_point = num_context_tokens - num_target_points
            cross_attention_mask[:num_heads_queries, first_target_point:] = False
        decoded = self._decode(queries, context_tokens, flow_time, self_attention_mask, cross_attention_mask)
        return self._pack(
            self.wp_decoder(decoded[:, :num_waypoint_queries]) if waypoint_state is not None else None,
            self.action_decoder(decoded[:, num_waypoint_queries:num_heads_queries])
            if action_state is not None
            else None,
            self.path_decoder(decoded[:, num_heads_queries:]) if path_state is not None else None,
        )

    def _noise_draw_context(
        self,
        context_tokens: jt.Float[torch.Tensor, "batch tokens dim"],
    ) -> jt.Float[torch.Tensor, "batch_or_draws tokens dim"]:
        """
        Repeats the context for compute_loss's noise draws, unless shared_draw_context lets the draws share one copy.

        Args:
            context_tokens: the planning context of every sample.

        Returns:
            the context per draw, or per sample with shared_draw_context.
        """
        flow = self._config.planning_config.flow_config
        if flow.shared_draw_context:
            return context_tokens
        return context_tokens.repeat_interleave(flow.num_noise_draws, dim=0)

    def _pack(
        self,
        waypoint_state: jt.Float[torch.Tensor, "batch poses 3"] | None,
        action_state: jt.Float[torch.Tensor, "batch poses 2"] | None,
        path_state: jt.Float[torch.Tensor, "batch path_points 3"] | None,
    ) -> jt.Float[torch.Tensor, "batch 1 state"]:
        """
        Packs the normalized states of the configured heads into one flat flow state.

        Args:
            waypoint_state: normalized waypoints, or None without the waypoint head.
            action_state: normalized actions, or None without the action head.
            path_state: normalized path points, or None without the path head.

        Returns:
            the flow state.
        """
        states = [state.flatten(1) for state in (waypoint_state, action_state, path_state) if state is not None]
        return torch.cat(states, dim=1)[:, None]

    @property
    def _num_state_values(self) -> int:
        """Values in one flow state: the waypoints, the actions and the path points that are configured."""
        return (
            self._num_waypoint_steps * len(PoseSE2Index)
            + self._num_action_steps * _NUM_ACTIONS
            + self._num_path_points * len(PoseSE2Index)
        )

    def _unpack(
        self,
        state: jt.Float[torch.Tensor, "batch 1 state"],
    ) -> tuple[
        jt.Float[torch.Tensor, "batch poses 3"] | None,
        jt.Float[torch.Tensor, "batch poses 2"] | None,
        jt.Float[torch.Tensor, "batch path_points 3"] | None,
    ]:
        """
        Splits the flow state into the waypoint, action and path states.

        Args:
            state: the flow state.

        Returns:
            the waypoint, action and path states, None for a head the config leaves out.
        """
        state = state[:, 0]
        num_waypoint_values = self._num_waypoint_steps * len(PoseSE2Index)
        num_action_values = self._num_action_steps * _NUM_ACTIONS
        waypoint_state = None
        if self._num_waypoint_steps:
            waypoint_state = state[:, :num_waypoint_values].reshape(-1, self._num_waypoint_steps, len(PoseSE2Index))
        action_state = None
        if self._num_action_steps:
            action_state = state[:, num_waypoint_values : num_waypoint_values + num_action_values].reshape(
                -1,
                self._num_action_steps,
                _NUM_ACTIONS,
            )
        path_state = None
        if self._num_path_points:
            path_state = state[:, num_waypoint_values + num_action_values :].reshape(
                -1,
                self._num_path_points,
                len(PoseSE2Index),
            )
        return waypoint_state, action_state, path_state

    @override
    def compute_loss(
        self,
        predictions: jt.Float[torch.Tensor, "batch ..."],
        label: jt.Float[torch.Tensor, "batch rows columns"],
    ) -> dict[str, torch.Tensor]:
        """
        Computes L1 losses of the denoised waypoints, path and actions; sampled rows give waypoint and rollout L1 only.

        Args:
            predictions: context and speed tokens under autograd, else sampled rows.
            label: future ego poses and their fitted actions, then the path points.

        Returns:
            dictionary of unweighted losses.
        """
        num_steps = self._config.trajectory_num_steps
        action_config = self._action_config
        pose_label = label.float()[:, :num_steps, : len(PoseSE2Index)]
        if predictions.shape[-1] == _NUM_SAMPLE_COLUMNS:
            rows = predictions.float()[:, :num_steps]
            losses: dict[str, torch.Tensor] = {}
            if self._num_waypoint_steps:
                losses["loss_trajectory"] = pose_l1_loss(rows[..., WAYPOINT_COLUMNS], pose_label)
            if action_config is not None:
                losses["loss_action"] = pose_l1_loss(rows[..., ROLLOUT_COLUMNS], pose_label)
            return losses
        flow = self._config.planning_config.flow_config
        pose_scale = cast(torch.Tensor, self.pose_scale)
        pose_mean = cast(torch.Tensor, self.pose_mean)
        context_tokens, speed = _split_speed(predictions)
        action_label = None
        action_target = None
        if action_config is not None:
            action_label = label.float()[:, :num_steps, len(PoseSE2Index) : len(PoseSE2Index) + _NUM_ACTIONS]
            action_target = actions_to_state(action_label, speed, action_config)
        path_label = None
        path_target = None
        if self._num_path_points:
            path_label = label.float()[:, num_steps:, : len(PoseSE2Index)]
            path_target = (path_label - cast(torch.Tensor, self.path_mean)) / cast(torch.Tensor, self.path_scale)
        target_state = self._pack(
            (pose_label - pose_mean) / pose_scale if self._num_waypoint_steps else None,
            action_target,
            path_target,
        ).repeat_interleave(flow.num_noise_draws, dim=0)
        context_tokens = self._noise_draw_context(context_tokens)
        draw_speed = speed.repeat_interleave(flow.num_noise_draws)
        pose_label = pose_label.repeat_interleave(flow.num_noise_draws, dim=0)

        noise_level = torch.rand(target_state.shape[0], device=target_state.device) ** (
            1.0 / flow.noise_level_beta_alpha
        )
        flow_time = torch.ones_like(noise_level) - noise_level
        noise = torch.randn_like(target_state)
        noisy_state = flow_time[:, None, None] * target_state + noise_level[:, None, None] * noise
        velocity = self._velocity(context_tokens, noisy_state, flow_time)
        clean_waypoint_state, clean_action_state, clean_path_state = self._unpack(
            noisy_state + noise_level[:, None, None] * velocity,
        )

        losses = {}
        if clean_waypoint_state is not None:
            losses["loss_trajectory"] = pose_l1_loss(clean_waypoint_state * pose_scale + pose_mean, pose_label)
        if clean_path_state is not None and path_label is not None:
            path_label = path_label.repeat_interleave(flow.num_noise_draws, dim=0)
            path_scale = cast(torch.Tensor, self.path_scale)
            path_mean = cast(torch.Tensor, self.path_mean)
            losses["loss_path"] = pose_l1_loss(clean_path_state * path_scale + path_mean, path_label)
        if (
            clean_action_state is not None
            and action_config is not None
            and action_label is not None
            and action_target is not None
        ):
            action_label = action_label.repeat_interleave(flow.num_noise_draws, dim=0)
            clean_actions = state_to_actions(clean_action_state, draw_speed, action_config)
            action_loss = clean_action_state.new_zeros(())
            if action_config.action_loss:
                action_target = action_target.repeat_interleave(flow.num_noise_draws, dim=0)
                action_loss = action_loss + F.l1_loss(clean_action_state.float(), action_target)
            if action_config.rollout_pose_loss:
                action_loss = action_loss + pose_l1_loss(rollout(clean_actions, draw_speed, action_config), pose_label)
            if action_config.rollout_speed_loss:
                acceleration_errors = clean_actions[..., 0].float() - action_label[..., 0]
                speed_errors = torch.cumsum(acceleration_errors * action_config.accel_std_mps2 * STEP_S, dim=-1)
                action_loss = action_loss + speed_errors.abs().mean()
            losses["loss_action"] = action_loss
        return losses

    @override
    def compute_metrics(
        self,
        predictions: jt.Float[torch.Tensor, "batch ..."],
        label: jt.Float[torch.Tensor, "batch rows columns"],
    ) -> dict[str, torch.Tensor]:
        """
        Computes waypoint and rollout metrics, plus path metrics when given context tokens, which are sampled first.

        Args:
            predictions: context and speed tokens under autograd, else sampled rows.
            label: future ego poses and their fitted actions, then the path points.

        Returns:
            dictionary of scalar metric tensors.
        """
        num_steps = self._config.trajectory_num_steps
        pose_label = label.float()[:, :num_steps, : len(PoseSE2Index)]
        if predictions.shape[-1] == _NUM_SAMPLE_COLUMNS:
            rows = predictions.float()[:, :num_steps]
            waypoints = rows[..., WAYPOINT_COLUMNS] if self._num_waypoint_steps else None
            rollout_poses = rows[..., ROLLOUT_COLUMNS] if self._action_config is not None else None
            path = None
        else:
            outputs = self._sample_outputs(predictions)
            waypoints, rollout_poses, path = outputs.waypoints, outputs.rollout, outputs.path
        metrics: dict[str, torch.Tensor] = {}
        if waypoints is not None:
            metrics |= RegressionPlanningDecoder.compute_metrics(self, waypoints, pose_label)
        if rollout_poses is not None:
            metrics |= {
                name.replace("waypoints_", "rollout_", 1): value
                for name, value in RegressionPlanningDecoder.compute_metrics(self, rollout_poses, pose_label).items()
            }
        if path is None:
            return metrics
        path_label = label.float()[:, num_steps:, : len(PoseSE2Index)]
        position_errors = path[..., : len(Point2DIndex)] - path_label[..., : len(Point2DIndex)]
        displacements = position_errors.pow(2).sum(dim=-1).sqrt()
        label_yaw = path_label[..., PoseSE2Index.YAW]
        raw_yaw_difference_rad = path[..., PoseSE2Index.YAW] - label_yaw
        yaw_difference_rad = (raw_yaw_difference_rad + torch.pi) % (2 * torch.pi) - torch.pi
        cross_route_errors = (
            -position_errors[..., 0] * label_yaw.sin() + position_errors[..., 1] * label_yaw.cos()
        ).abs()
        return metrics | {
            "path_ade": displacements.mean(),
            "path_fde": displacements[:, -1].mean(),
            "path_aye": yaw_difference_rad.abs().mean(),
            "path_cross_route_error": cross_route_errors.mean(),
            "path_cross_route_error_first_10m": cross_route_errors[:, : self._num_path_points_first_10m].mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        context_tokens: jt.Float[torch.Tensor, "batch tokens dim"],
    ) -> jt.Float[torch.Tensor, "batch rows columns"]:
        """
        Integrates the flow of waypoints, actions and path points and rolls the actions out.

        Args:
            context_tokens: the planning context of every sample, the speed token last.

        Returns:
            step rows in the *_COLUMNS layout, then path rows of (x, y, route heading) and zeros.
        """
        outputs = self._sample_outputs(context_tokens)
        # A missing head's columns repeat the other head's poses; missing actions are zero.
        waypoints = outputs.waypoints if outputs.waypoints is not None else outputs.rollout
        rollout_poses = outputs.rollout if outputs.rollout is not None else outputs.waypoints
        assert waypoints is not None
        assert rollout_poses is not None
        actions = (
            outputs.actions
            if outputs.actions is not None
            else waypoints.new_zeros((*waypoints.shape[:2], _NUM_ACTIONS))
        )
        if self._driven_output == "waypoints":
            driven = waypoints
        elif self._driven_output == "actions":
            driven = rollout_poses
        else:
            assert outputs.path is not None, f"driven_output={self._driven_output} needs the path"
            driven = place_on_path(
                waypoints if self._driven_output == "waypoints_on_path" else rollout_poses,
                outputs.path,
            )
        step_rows = torch.cat([driven, waypoints, rollout_poses, actions], dim=-1)
        if outputs.path is None:
            return step_rows
        path_rows = F.pad(outputs.path, (0, _NUM_SAMPLE_COLUMNS - len(PoseSE2Index)))
        return torch.cat([step_rows, path_rows], dim=1)

    @torch.no_grad()
    def _sample_outputs(self, context_tokens: jt.Float[torch.Tensor, "batch tokens dim"]) -> _Outputs:
        """
        Samples the plan and decodes every configured output.

        Args:
            context_tokens: the planning context of every sample, the speed token last.

        Returns:
            the waypoints, actions, rollout and path, None for a head the config leaves out.
        """
        action_config = self._action_config
        context_tokens, speed = _split_speed(context_tokens)
        state = self._integrate(context_tokens, (1, self._num_state_values))
        waypoint_state, action_state, path_state = self._unpack(state)
        waypoints = None
        if waypoint_state is not None:
            waypoints = waypoint_state * cast(torch.Tensor, self.pose_scale) + cast(torch.Tensor, self.pose_mean)
        path = None
        if path_state is not None:
            path = path_state * cast(torch.Tensor, self.path_scale) + cast(torch.Tensor, self.path_mean)
        actions = None
        rollout_poses = None
        if action_state is not None and action_config is not None:
            actions = state_to_actions(action_state, speed, action_config)
            rollout_poses = rollout(actions, speed, action_config)
        return _Outputs(waypoints=waypoints, actions=actions, rollout=rollout_poses, path=path)

    @override
    def planned_poses(
        self,
        predictions: jt.Float[torch.Tensor, "batch ..."],
    ) -> jt.Float[torch.Tensor, "batch poses 3"]:
        """
        Extracts the driven (x, y, yaw) per waypoint, sampling context tokens first.

        Args:
            predictions: context and speed tokens under autograd, else sampled rows.

        Returns:
            the planned poses.
        """
        if predictions.shape[-1] != _NUM_SAMPLE_COLUMNS:
            predictions = self.sample(predictions)
        return predictions[:, : self._config.trajectory_num_steps, _DRIVEN_COLUMNS].float()

    def _decode(
        self,
        queries: jt.Float[torch.Tensor, "draws queries dim"],
        context_tokens: jt.Float[torch.Tensor, "batch tokens dim"],
        flow_time: jt.Float[torch.Tensor, " draws"],
        self_attention_mask: jt.Bool[torch.Tensor, "queries queries"] | None = None,
        cross_attention_mask: jt.Bool[torch.Tensor, "queries tokens"] | None = None,
    ) -> jt.Float[torch.Tensor, "draws queries dim"]:
        """
        Runs the transformer decoder over the queries, conditioned on the flow time.

        Args:
            queries: one query per state row.
            context_tokens: the planning context.
            flow_time: flow time per sample.
            self_attention_mask: True where a query may attend to another query; None allows all.
            cross_attention_mask: True where a query may attend to a context token; None allows all.

        Returns:
            the decoded queries.
        """
        conditioning = self.adaptive_norm_time_embedding(flow_time * _FLOW_TIME_SCALE)
        return self.transformer_decoder(
            queries,
            context_tokens,
            conditioning,
            self_attention_mask,
            cross_attention_mask,
        )

    @torch.no_grad()
    def _integrate(
        self,
        context_tokens: jt.Float[torch.Tensor, "batch tokens dim"],
        state_shape: tuple[int, ...],
    ) -> jt.Float[torch.Tensor, "batch steps columns"]:
        """
        Euler-integrates the flow from noise with dropout off.

        Args:
            context_tokens: the planning context of every sample.
            state_shape: shape of one sample's normalized state.

        Returns:
            the normalized end state, averaged over num_averaged_samples draws.
        """
        flow = self._config.planning_config.flow_config
        num_plans = context_tokens.shape[0]
        context_tokens = context_tokens.repeat_interleave(flow.num_averaged_samples, dim=0)
        batch_size = context_tokens.shape[0]
        generator = None if flow.resample_noise else torch.Generator().manual_seed(flow.noise_seed)
        # The state stays fp32 under autocast, so bf16 velocities are added in fp32.
        state = torch.randn((batch_size, *state_shape), generator=generator) * flow.noise_temperature
        state = state.to(context_tokens.device)
        was_training = self.training
        self.eval()
        try:
            for step in range(flow.num_inference_steps):
                flow_time = torch.full(
                    (batch_size,),
                    step / flow.num_inference_steps,
                    device=state.device,
                    dtype=state.dtype,
                )
                state = state + self._velocity(context_tokens, state, flow_time) / flow.num_inference_steps
        finally:
            self.train(was_training)
        return state.view(num_plans, flow.num_averaged_samples, *state_shape).mean(dim=1)


def place_on_path(
    waypoints: jt.Float[torch.Tensor, "batch poses 3"],
    path: jt.Float[torch.Tensor, "batch path_points 3"],
) -> jt.Float[torch.Tensor, "batch poses 3"]:
    """
    Moves each waypoint onto the path at the distance the waypoints have travelled, heading along the path.

    The path starts at the ego origin, and past its end the waypoints keep their own shape, shifted to continue from it.

    Args:
        waypoints: (x, y, yaw) per trajectory step.
        path: (x, y, route heading) per path point.

    Returns:
        the waypoints on the path, (x, y, yaw) per trajectory step.
    """
    num_position_values = len(Point2DIndex)
    origin = torch.zeros_like(path[:, :1, :num_position_values])
    polyline = torch.cat([origin, path[..., :num_position_values]], dim=1)
    segments = polyline.diff(dim=1)
    segment_lengths_m = segments.pow(2).sum(dim=-1).sqrt().clamp(min=1e-6)
    segment_ends_m = segment_lengths_m.cumsum(dim=1)
    segment_starts_m = F.pad(segment_ends_m, (1, 0))[:, :-1].contiguous()
    path_length_m = segment_ends_m[:, -1:].contiguous()
    waypoint_polyline = torch.cat([origin, waypoints[..., :num_position_values]], dim=1)
    steps = waypoint_polyline.diff(dim=1)
    step_lengths_m = steps.pow(2).sum(dim=-1).sqrt().clamp(min=1e-6)
    distances_m = step_lengths_m.cumsum(dim=1).contiguous()

    index = (torch.searchsorted(segment_starts_m, distances_m, right=True) - 1).clamp(0, segments.shape[1] - 1)
    segment = segments.gather(1, index[..., None].expand(-1, -1, num_position_values))
    start = polyline.gather(1, index[..., None].expand(-1, -1, num_position_values))
    fraction = (distances_m - segment_starts_m.gather(1, index)) / segment_lengths_m.gather(1, index)
    on_path = start + fraction.clamp(max=1.0)[..., None] * segment
    on_path_yaws = torch.atan2(segment[..., Point2DIndex.Y], segment[..., Point2DIndex.X])

    step_index = torch.searchsorted(distances_m, path_length_m).clamp(max=steps.shape[1] - 1)
    step_starts_m = F.pad(distances_m, (1, 0))[:, :-1]
    step_fraction = (path_length_m - step_starts_m.gather(1, step_index)) / step_lengths_m.gather(1, step_index)
    step_start = waypoint_polyline.gather(1, step_index[..., None].expand(-1, -1, num_position_values))
    step = steps.gather(1, step_index[..., None].expand(-1, -1, num_position_values))
    waypoints_at_path_end = step_start + step_fraction.clamp(0.0, 1.0)[..., None] * step
    shift = polyline[:, -1:] - waypoints_at_path_end

    beyond_path = distances_m > path_length_m
    positions = torch.where(beyond_path[..., None], waypoints[..., :num_position_values] + shift, on_path)
    yaws = torch.where(beyond_path, waypoints[..., PoseSE2Index.YAW], on_path_yaws)
    return torch.cat([positions, yaws[..., None]], dim=-1)


def _split_speed(
    tokens: jt.Float[torch.Tensor, "batch tokens dim"],
) -> tuple[jt.Float[torch.Tensor, "batch context_tokens dim"], jt.Float[torch.Tensor, " batch"]]:
    """
    Splits off the speed token that forward appends to the context.

    Args:
        tokens: the context tokens, the speed token last.

    Returns:
        the context tokens and the speed of every sample.
    """
    return tokens[:, :-1], tokens[:, -1, 0].float()


def path_attention_masks(
    config: TransfuserConfig,
    num_waypoint_queries: int,
    num_action_queries: int,
    num_context_tokens: int,
    device: torch.device,
) -> tuple[jt.Bool[torch.Tensor, "queries queries"] | None, jt.Bool[torch.Tensor, "queries tokens"] | None]:
    """
    Builds the self- and cross-attention masks of path_config, True where a query may attend.

    Queries are the waypoints, then the actions, then the path points; the target point tokens end the context.

    Args:
        config: global config dataclass of TransFuser.
        num_waypoint_queries: number of waypoint queries.
        num_action_queries: number of action queries.
        num_context_tokens: number of context tokens.
        device: device of the masks.

    Returns:
        the self-attention mask over the queries and the cross-attention mask over the context, None if unmasked.
    """
    path = config.planning_config.path_config
    assert path is not None, "path attention masks need planning_config.path_config"
    num_other_queries = num_waypoint_queries + num_action_queries
    num_queries = num_other_queries + path.num_points
    self_attention_mask = None
    if not path.path_attends_to_waypoints or not path.path_attends_to_actions:
        self_attention_mask = torch.ones(num_queries, num_queries, dtype=torch.bool, device=device)
        if not path.path_attends_to_waypoints:
            self_attention_mask[num_other_queries:, :num_waypoint_queries] = False
        if not path.path_attends_to_actions:
            self_attention_mask[num_other_queries:, num_waypoint_queries:num_other_queries] = False
    cross_attention_mask = None
    if path.visible_target_point_distances_m is not None:
        distances_m = config.required_target_point_distances_m
        first_target_point = num_context_tokens - len(distances_m)
        cross_attention_mask = torch.ones(num_queries, num_context_tokens, dtype=torch.bool, device=device)
        for index, distance_m in enumerate(distances_m):
            if distance_m not in path.visible_target_point_distances_m:
                cross_attention_mask[num_other_queries:, first_target_point + index] = False
    return self_attention_mask, cross_attention_mask
