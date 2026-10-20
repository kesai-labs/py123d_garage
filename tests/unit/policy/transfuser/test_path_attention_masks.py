# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The path queries see the waypoint queries, the action queries and the target points path_config lets them see, and
nothing else; the waypoint queries still see the whole path and every target point.
"""

from __future__ import annotations

import pytest
import torch

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    BevSemanticConfig,
    BoxDetectionConfig,
    FlowMatchingConfig,
    PathConfig,
    PlanningConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder import (
    FlowPlanningDecoder,
)

_NUM_PATH_POINTS = 8
_NUM_CONTEXT_TOKENS = 12
_TARGET_POINT_DISTANCES_M = [40.0, 60.0, 80.0]
_BATCH_SIZE = 2


def _config(path_config: PathConfig) -> TransfuserConfig:
    return TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 1_000_000),
        required_target_point_distances_m=_TARGET_POINT_DISTANCES_M,
        backbone_config=BackboneConfig(image_encoder_pretrained=False),
        bev_semantic_config=BevSemanticConfig(use_bev_semantic=False),
        box_detection_config=BoxDetectionConfig(detect_boxes=False),
        planning_config=PlanningConfig(
            path_config=path_config,
            flow_config=FlowMatchingConfig(
                waypoints_attend_to_actions=True,
                actions_attend_to_waypoints=True,
                waypoints_and_actions_attend_to_target_points=True,
            ),
        ),
    )


def _decoder(path_config: PathConfig) -> FlowPlanningDecoder:
    torch.manual_seed(0)
    decoder = FlowPlanningDecoder(64, _config(path_config)).eval()
    # Zero-initialized modulation and output layers would hide every dependence; random weights show it.
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.normal_(0.0, 0.2)
    return decoder


def _outputs(
    path_config: PathConfig,
    context_tokens: torch.Tensor,
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The waypoint, action and path velocities under the same weights, so two calls differ only by their inputs."""
    decoder = _decoder(path_config)
    with torch.no_grad():
        state = decoder._pack(*states)  # pyright: ignore[reportPrivateUsage]
        velocity = decoder._velocity(context_tokens, state, torch.full((_BATCH_SIZE,), 0.5))  # pyright: ignore[reportPrivateUsage]
        waypoints, actions, path = decoder._unpack(velocity)  # pyright: ignore[reportPrivateUsage]
        assert waypoints is not None
        assert actions is not None
        assert path is not None
        return waypoints, actions, path


def _inputs(config: TransfuserConfig) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    token_dim = config.planning_config.transfuser_token_dim
    num_steps = config.trajectory_num_steps
    context_tokens = torch.randn(_BATCH_SIZE, _NUM_CONTEXT_TOKENS, token_dim, generator=generator)
    states = (
        torch.randn(_BATCH_SIZE, num_steps, 3, generator=generator),
        torch.randn(_BATCH_SIZE, num_steps, 2, generator=generator),
        torch.randn(_BATCH_SIZE, _NUM_PATH_POINTS, 3, generator=generator),
    )
    return context_tokens, states


def _changed(
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    changed = list(states)
    changed[index] = changed[index] + 1.0
    return changed[0], changed[1], changed[2]


def _changed_target_point(context_tokens: torch.Tensor, distance_m: float) -> torch.Tensor:
    changed = context_tokens.clone()
    index = _NUM_CONTEXT_TOKENS - len(_TARGET_POINT_DISTANCES_M) + _TARGET_POINT_DISTANCES_M.index(distance_m)
    changed[:, index] += 1.0
    return changed


def test_attending_path_sees_the_waypoints_the_actions_and_every_target_point() -> None:
    path_config = PathConfig(num_points=_NUM_PATH_POINTS, path_attends_to_waypoints=True, path_attends_to_actions=True)
    context_tokens, states = _inputs(_config(path_config))
    _, _, path = _outputs(path_config, context_tokens, states)
    _, _, path_after_waypoints = _outputs(path_config, context_tokens, _changed(states, 0))
    _, _, path_after_actions = _outputs(path_config, context_tokens, _changed(states, 1))
    _, _, path_after_target_point = _outputs(path_config, _changed_target_point(context_tokens, 40.0), states)
    assert not torch.allclose(path, path_after_waypoints)
    assert not torch.allclose(path, path_after_actions)
    assert not torch.allclose(path, path_after_target_point)


def test_one_way_path_ignores_the_waypoints_and_the_actions() -> None:
    path_config = PathConfig(
        num_points=_NUM_PATH_POINTS,
        path_attends_to_waypoints=False,
        path_attends_to_actions=False,
    )
    context_tokens, states = _inputs(_config(path_config))
    waypoints, _, path = _outputs(path_config, context_tokens, states)
    _, _, path_after_waypoints = _outputs(path_config, context_tokens, _changed(states, 0))
    _, _, path_after_actions = _outputs(path_config, context_tokens, _changed(states, 1))
    waypoints_after_path, _, _ = _outputs(path_config, context_tokens, _changed(states, 2))
    torch.testing.assert_close(path, path_after_waypoints)
    torch.testing.assert_close(path, path_after_actions)
    assert not torch.allclose(waypoints, waypoints_after_path)


def test_path_sees_only_its_visible_target_points() -> None:
    path_config = PathConfig(
        num_points=_NUM_PATH_POINTS,
        path_attends_to_waypoints=False,
        path_attends_to_actions=False,
        visible_target_point_distances_m=[80.0],
    )
    context_tokens, states = _inputs(_config(path_config))
    waypoints, _, path = _outputs(path_config, context_tokens, states)
    for hidden_distance_m in (40.0, 60.0):
        waypoints_after, _, path_after = _outputs(
            path_config,
            _changed_target_point(context_tokens, hidden_distance_m),
            states,
        )
        torch.testing.assert_close(path, path_after)
        assert not torch.allclose(waypoints, waypoints_after)
    _, _, path_after_visible = _outputs(path_config, _changed_target_point(context_tokens, 80.0), states)
    assert not torch.allclose(path, path_after_visible)


@pytest.mark.parametrize(
    "path_config",
    [
        PathConfig(path_attends_to_actions=False, visible_target_point_distances_m=[80.0]),
        PathConfig(path_attends_to_waypoints=False, visible_target_point_distances_m=[80.0]),
        PathConfig(
            path_attends_to_waypoints=False,
            path_attends_to_actions=False,
            visible_target_point_distances_m=[70.0],
        ),
    ],
)
def test_config_rejects_leaking_or_unknown_target_points(path_config: PathConfig) -> None:
    with pytest.raises(ValueError, match="visible_target_point_distances_m"):
        _config(path_config)
