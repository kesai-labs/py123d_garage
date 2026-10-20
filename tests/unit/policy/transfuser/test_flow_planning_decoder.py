# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The waypoint, path and action flow decoder trains every output, scores the waypoints, the action rollout and the path,
and drives the output its config names.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from py123d.geometry import PoseSE2Index

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    ActionConfig,
    BackboneConfig,
    BevSemanticConfig,
    BoxDetectionConfig,
    FlowMatchingConfig,
    PathConfig,
    PlanningConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.labels import TransfuserLabels
from py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder import (
    FlowPlanningDecoder,
    place_on_path,
)
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

_NUM_PATH_POINTS = 16
_BATCH_SIZE = 2


def _config(driven_output: str = "waypoints") -> TransfuserConfig:
    # No pretrained download: CI runs offline. The planner alone, so the labels hold only its tensors.
    return TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        backbone_config=BackboneConfig(image_encoder_pretrained=False),
        bev_semantic_config=BevSemanticConfig(use_bev_semantic=False),
        box_detection_config=BoxDetectionConfig(detect_boxes=False),
        planning_config=PlanningConfig(
            planning_decoder_target=(
                "py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder:FlowPlanningDecoder"
            ),
            path_config=PathConfig(num_points=_NUM_PATH_POINTS),
            unicycle_action_config=ActionConfig(
                rollout_pose_loss=True,
                action_loss=False,
                rollout_speed_loss=True,
                lateral_accel_actions=True,
                bounded=True,
            ),
            flow_config=FlowMatchingConfig(driven_output=driven_output),
        ),
    )


def _policy(driven_output: str = "waypoints") -> TransfuserPolicy:
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=_config(driven_output)))
    policy.initialize()
    return policy


def _inputs(config: TransfuserConfig) -> tuple[TransfuserFeatures, NavigationConditioning]:
    features = TransfuserFeatures(
        camera_feature=torch.randint(
            0,
            256,
            (_BATCH_SIZE, 3, config.camera_config.image_height, config.camera_config.image_width),
            dtype=torch.uint8,
        ),
        velocity=torch.full((_BATCH_SIZE, 1), 8.0),
        velocity_vector=torch.zeros(_BATCH_SIZE, 2),
        acceleration_vector=torch.zeros(_BATCH_SIZE, 2),
        ego_history=None,
        lidar_feature=torch.rand(
            _BATCH_SIZE,
            1,
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
    )
    navigation = NavigationConditioning(
        target_points=torch.zeros(_BATCH_SIZE, 1, 2),
        target_point_headings=torch.zeros(_BATCH_SIZE, 1),
    )
    return features, navigation


def _labels(config: TransfuserConfig) -> TransfuserLabels:
    """Straight driving at 8 m/s along x: poses, zero actions and the path 1 m apart."""
    time_s = torch.tensor(config.trajectory_offsets_us, dtype=torch.float32) / 1e6
    trajectory = torch.zeros(_BATCH_SIZE, config.trajectory_num_steps, len(PoseSE2Index))
    trajectory[..., 0] = 8.0 * time_s
    path = torch.zeros(_BATCH_SIZE, _NUM_PATH_POINTS, len(PoseSE2Index))
    path[..., 0] = torch.arange(1.0, _NUM_PATH_POINTS + 1.0)
    return TransfuserLabels(
        trajectory=trajectory,
        action=torch.zeros(_BATCH_SIZE, config.trajectory_num_steps, 2),
        path=path,
    )


def test_every_output_trains() -> None:
    policy = _policy().train()
    config = policy.policy_config
    predictions: TransfuserPredictions = policy(*_inputs(config))
    labels = _labels(config)

    losses = policy.compute_loss(labels, predictions)
    assert set(losses) == {"loss_trajectory", "loss_path", "loss_action"}
    assert {name for name, weight in policy.loss_weights(0).items() if weight > 0.0} == set(losses)
    assert all(torch.isfinite(loss) for loss in losses.values())
    torch.stack(list(losses.values())).sum().backward()
    decoder = policy.planning_decoder
    assert isinstance(decoder, FlowPlanningDecoder)
    for output_layer in (decoder.wp_decoder, decoder.action_decoder, decoder.path_decoder):
        assert output_layer.weight.grad is not None
        assert output_layer.weight.grad.abs().sum() > 0


def test_metrics_score_the_waypoints_the_rollout_and_the_path() -> None:
    policy = _policy().train()
    config = policy.policy_config
    predictions: TransfuserPredictions = policy(*_inputs(config))
    with torch.no_grad():
        metrics = policy.compute_metrics(_labels(config), predictions)

    expected = {
        f"{output}_{name}"
        for output in ("waypoints", "rollout")
        for name in ("ade", "fde", "aye", "along_track_error", "cross_track_error")
    }
    expected |= {"path_ade", "path_fde", "path_aye", "path_cross_route_error", "path_cross_route_error_first_10m"}
    assert expected <= metrics.keys()
    assert all(torch.isfinite(metrics[name]) for name in expected)


@pytest.mark.parametrize(("driven_output", "columns"), [("waypoints", slice(3, 6)), ("actions", slice(6, 9))])
def test_evaluation_drives_the_configured_output(driven_output: str, columns: slice) -> None:
    trained = _policy()
    policy = _policy(driven_output)
    policy.load_state_dict(trained.state_dict())
    config = policy.policy_config
    with torch.no_grad():
        plan: TransfuserPredictions = policy.eval()(*_inputs(config))
        losses = policy.compute_loss(_labels(config), plan)
        metrics = policy.compute_metrics(_labels(config), plan)
    assert plan.trajectory is not None
    assert plan.path is not None
    assert plan.trajectory.shape == (_BATCH_SIZE, config.trajectory_num_steps, 11)
    assert plan.path.shape == (_BATCH_SIZE, _NUM_PATH_POINTS, len(PoseSE2Index))
    torch.testing.assert_close(plan.ego_trajectory_se2, plan.trajectory[..., columns])
    assert plan.waypoints is not None
    assert plan.rollout is not None
    torch.testing.assert_close(plan.waypoints, plan.trajectory[..., 3:6])
    torch.testing.assert_close(plan.rollout, plan.trajectory[..., 6:9])
    assert set(losses) == {"loss_trajectory", "loss_action"}
    assert {"waypoints_ade", "rollout_ade"} <= metrics.keys()


def _heads_config(with_waypoints: bool, with_actions: bool, with_path: bool) -> TransfuserConfig:
    """The test config with the given heads; the rollout drives when there are no waypoints."""
    config = _config("waypoints" if with_waypoints else "actions")
    planning = config.planning_config
    return replace(
        config,
        planning_config=replace(
            planning,
            predict_waypoints=with_waypoints,
            path_config=planning.path_config if with_path else None,
            unicycle_action_config=planning.unicycle_action_config if with_actions else None,
        ),
    )


@pytest.mark.parametrize(
    ("with_waypoints", "with_actions", "with_path"),
    [
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_decoder_runs_with_any_driving_head_combination(
    with_waypoints: bool,
    with_actions: bool,
    with_path: bool,
) -> None:
    config = _heads_config(with_waypoints, with_actions, with_path)
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    full_labels = _labels(config)
    labels = TransfuserLabels(
        trajectory=full_labels.trajectory,
        action=full_labels.action if with_actions else None,
        path=full_labels.path if with_path else None,
    )

    predictions: TransfuserPredictions = policy.train()(*_inputs(config))
    losses = policy.compute_loss(labels, predictions)
    expected_losses = (
        ({"loss_trajectory"} if with_waypoints else set())
        | ({"loss_action"} if with_actions else set())
        | ({"loss_path"} if with_path else set())
    )
    assert set(losses) == expected_losses
    assert {name for name, weight in policy.loss_weights(0).items() if weight > 0.0} == expected_losses
    torch.stack(list(losses.values())).sum().backward()
    decoder = policy.planning_decoder
    assert isinstance(decoder, FlowPlanningDecoder)
    # Every parameter of the decoder trains, so a left-out head leaves no dead weights behind.
    assert all(parameter.grad is not None for parameter in decoder.parameters() if parameter.requires_grad)

    with torch.no_grad():
        plan: TransfuserPredictions = policy.eval()(*_inputs(config))
        metrics = policy.compute_metrics(labels, plan)
    assert plan.trajectory is not None
    assert plan.trajectory.shape == (_BATCH_SIZE, config.trajectory_num_steps, 11)
    assert (plan.path is not None) == with_path
    assert (plan.waypoints is not None) == (with_waypoints and with_actions)
    driven_columns = slice(3, 6) if with_waypoints else slice(6, 9)
    torch.testing.assert_close(plan.ego_trajectory_se2, plan.trajectory[..., driven_columns])
    if not with_actions:
        # The rollout columns repeat the waypoints and the action columns are zero.
        torch.testing.assert_close(plan.trajectory[..., 6:9], plan.trajectory[..., 3:6])
        assert not plan.trajectory[..., 9:11].any()
    if not with_waypoints:
        # The waypoint columns repeat the rollout.
        torch.testing.assert_close(plan.trajectory[..., 3:6], plan.trajectory[..., 6:9])
    assert ("waypoints_ade" in metrics) == with_waypoints
    assert ("rollout_ade" in metrics) == with_actions
    # Path metrics need the context tokens of a training forward, which the decoder samples itself.
    with torch.no_grad():
        training_metrics = policy.compute_metrics(labels, predictions)
    assert ("path_ade" in training_metrics) == with_path


@pytest.mark.parametrize("with_path", [False, True])
def test_config_needs_waypoints_or_actions_to_drive(with_path: bool) -> None:
    with pytest.raises(ValueError, match="predict_waypoints=false needs unicycle_action_config"):
        _heads_config(with_waypoints=False, with_actions=False, with_path=with_path)


def test_place_on_path_keeps_the_waypoint_distances_on_a_straight_path() -> None:
    path = torch.zeros(1, 4, 3)
    path[0, :, 0] = torch.arange(1.0, 5.0)
    waypoints = torch.tensor([[[1.0, 0.3, 0.1], [2.0, -0.3, 0.0], [3.0, 0.0, 0.0]]])
    distances = torch.cat([torch.zeros(1, 1, 2), waypoints[..., :2]], dim=1).diff(dim=1).norm(dim=-1).cumsum(dim=1)
    placed = place_on_path(waypoints, path)
    torch.testing.assert_close(placed[..., 0], distances)
    torch.testing.assert_close(placed[..., 1:], torch.zeros(1, 3, 2))


def test_place_on_path_shifts_the_waypoints_past_the_path_end_onto_it() -> None:
    path = torch.zeros(1, 4, 3)
    path[0, :, 0] = torch.arange(1.0, 5.0)
    waypoints = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 1.0, 0.4], [6.0, 1.0, 0.0]]])
    placed = place_on_path(waypoints, path)
    # The waypoints have travelled the path's 4 m 1 m into their step from (3, 0) to (5, 1).
    at_path_length = torch.tensor([3.0, 0.0]) + torch.tensor([2.0, 1.0]) / 5**0.5
    torch.testing.assert_close(placed[0, :2, :2], torch.tensor([[1.0, 0.0], [3.0, 0.0]]))
    torch.testing.assert_close(placed[0, 2:, :2], waypoints[0, 2:, :2] + torch.tensor([4.0, 0.0]) - at_path_length)
    torch.testing.assert_close(placed[0, 2:, 2], waypoints[0, 2:, 2])


def test_place_on_path_follows_a_curve_at_the_travelled_distance() -> None:
    radius_m = 10.0
    arc_m = torch.arange(1.0, 13.0)
    path = torch.stack(
        [radius_m * torch.sin(arc_m / radius_m), radius_m * (1 - torch.cos(arc_m / radius_m)), arc_m / radius_m],
        -1,
    )[None]
    waypoints = torch.tensor([[[2.5, 0.0, 0.0], [4.0, 0.0, 0.0], [5.5, 0.0, 0.0]]])
    placed = place_on_path(waypoints, path)
    polyline = torch.cat([torch.zeros(1, 2), path[0, :, :2]])
    segment_lengths = polyline.diff(dim=0).norm(dim=-1)
    for (x, y, yaw), distance_m in zip(placed[0].tolist(), (2.5, 4.0, 5.5), strict=True):
        index = int(distance_m // 1.0)
        start, end = polyline[index], polyline[index + 1]
        fraction = (distance_m - segment_lengths[:index].sum()) / segment_lengths[index]
        torch.testing.assert_close(torch.tensor([x, y]), start + fraction * (end - start))
        torch.testing.assert_close(torch.tensor(yaw), torch.atan2(*(end - start).flip(0)))


@pytest.mark.parametrize("output", ["waypoints", "actions"])
def test_evaluation_drives_the_output_placed_on_the_path(output: str) -> None:
    trained = _policy()
    policy = _policy(f"{output}_on_path")
    policy.load_state_dict(trained.state_dict())
    with torch.no_grad():
        plan: TransfuserPredictions = policy.eval()(*_inputs(policy.policy_config))
    poses = plan.waypoints if output == "waypoints" else plan.rollout
    assert poses is not None
    assert plan.path is not None
    torch.testing.assert_close(plan.ego_trajectory_se2, place_on_path(poses, plan.path))


def test_driven_output_is_checked() -> None:
    config = _config()
    planning = replace(
        config.planning_config,
        flow_config=replace(config.planning_config.flow_config, driven_output="path"),
    )
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=replace(config, planning_config=planning)))
    with pytest.raises(AssertionError, match="driven_output"):
        policy.initialize()


class _Velocities:
    """The decoder's velocities under random weights for one random state, as functions of perturbed inputs."""

    def __init__(self, path_config: PathConfig, **attention: bool) -> None:
        config = _config()
        planning = replace(
            config.planning_config,
            path_config=path_config,
            flow_config=replace(config.planning_config.flow_config, **attention),
        )
        torch.manual_seed(0)
        policy = TransfuserPolicy(PolicyConfig(transfuser_config=replace(config, planning_config=planning)))
        policy.initialize()
        decoder = policy.planning_decoder.eval()
        assert isinstance(decoder, FlowPlanningDecoder)
        # Zero-initialized modulation and output layers would hide every dependence; random weights show it.
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.normal_(0.0, 0.2)
        self._decoder = decoder
        generator = torch.Generator().manual_seed(1)
        num_steps = config.trajectory_num_steps
        # The last context token is the one target point.
        self.context_tokens = torch.randn(
            _BATCH_SIZE,
            12,
            config.planning_config.transfuser_token_dim,
            generator=generator,
        )
        self.waypoint_state = torch.randn(_BATCH_SIZE, num_steps, len(PoseSE2Index), generator=generator)
        self.action_state = torch.randn(_BATCH_SIZE, num_steps, 2, generator=generator)
        self.path_state = torch.randn(_BATCH_SIZE, _NUM_PATH_POINTS, len(PoseSE2Index), generator=generator)

    def __call__(
        self,
        waypoint_offset: float = 0.0,
        action_offset: float = 0.0,
        target_point_offset: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_tokens = self.context_tokens.clone()
        context_tokens[:, -1] += target_point_offset
        with torch.no_grad():
            state = self._decoder._pack(  # pyright: ignore[reportPrivateUsage]
                self.waypoint_state + waypoint_offset,
                self.action_state + action_offset,
                self.path_state,
            )
            velocity = self._decoder._velocity(  # pyright: ignore[reportPrivateUsage]
                context_tokens,
                state,
                torch.full((_BATCH_SIZE,), 0.5),
            )
            waypoints, actions, path = self._decoder._unpack(velocity)  # pyright: ignore[reportPrivateUsage]
            assert waypoints is not None
            assert actions is not None
            assert path is not None
            return waypoints, actions, path


def test_waypoints_ignore_the_actions_that_still_read_them() -> None:
    velocities = _Velocities(
        PathConfig(num_points=_NUM_PATH_POINTS, path_attends_to_waypoints=False, path_attends_to_actions=False),
        waypoints_attend_to_actions=False,
        actions_attend_to_waypoints=True,
    )
    waypoints, actions, _ = velocities()
    waypoints_after_actions, _, _ = velocities(action_offset=1.0)
    _, actions_after_waypoints, _ = velocities(waypoint_offset=1.0)
    torch.testing.assert_close(waypoints, waypoints_after_actions)
    assert not torch.allclose(actions, actions_after_waypoints)


def test_independent_waypoints_and_actions_read_only_the_path() -> None:
    velocities = _Velocities(
        PathConfig(num_points=_NUM_PATH_POINTS, path_attends_to_waypoints=False, path_attends_to_actions=False),
        waypoints_attend_to_actions=False,
        actions_attend_to_waypoints=False,
    )
    waypoints, actions, path = velocities()
    waypoints_after_actions, _, path_after_actions = velocities(action_offset=1.0)
    _, actions_after_waypoints, path_after_waypoints = velocities(waypoint_offset=1.0)
    torch.testing.assert_close(waypoints, waypoints_after_actions)
    torch.testing.assert_close(actions, actions_after_waypoints)
    torch.testing.assert_close(path, path_after_actions)
    torch.testing.assert_close(path, path_after_waypoints)


@pytest.mark.parametrize("path_sees_the_target_point", [True, False])
def test_waypoints_and_actions_see_the_target_points_only_through_the_path(path_sees_the_target_point: bool) -> None:
    velocities = _Velocities(
        PathConfig(
            num_points=_NUM_PATH_POINTS,
            path_attends_to_waypoints=False,
            path_attends_to_actions=False,
            visible_target_point_distances_m=None if path_sees_the_target_point else [],
        ),
        waypoints_attend_to_actions=False,
        actions_attend_to_waypoints=False,
        waypoints_and_actions_attend_to_target_points=False,
    )
    before = velocities()
    after = velocities(target_point_offset=10.0)
    for output_before, output_after in zip(before, after, strict=True):
        assert torch.equal(output_before, output_after) != path_sees_the_target_point


def test_config_rejects_actions_reaching_the_waypoints_through_the_path() -> None:
    config = _config()
    planning = replace(
        config.planning_config,
        path_config=PathConfig(num_points=_NUM_PATH_POINTS, path_attends_to_actions=True),
        flow_config=replace(config.planning_config.flow_config, waypoints_attend_to_actions=False),
    )
    with pytest.raises(ValueError, match="path_attends_to_actions"):
        replace(config, planning_config=planning)


def test_config_rejects_waypoints_reaching_the_actions_through_the_path() -> None:
    config = _config()
    planning = replace(
        config.planning_config,
        path_config=PathConfig(num_points=_NUM_PATH_POINTS, path_attends_to_waypoints=True),
        flow_config=replace(config.planning_config.flow_config, actions_attend_to_waypoints=False),
    )
    with pytest.raises(ValueError, match="path_attends_to_waypoints"):
        replace(config, planning_config=planning)
