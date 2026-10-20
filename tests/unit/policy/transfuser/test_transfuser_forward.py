# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Forward-pass smoke test: the asymmetric BEV geometry must propagate through every head."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from py123d.geometry import PoseSE2Index

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    ActionConfig,
    BackboneConfig,
    FlowMatchingConfig,
    PathConfig,
    PlanningConditioningConfig,
    PlanningConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.labels import TransfuserLabels
from py123d_garage.policy.transfuser.network.decoder.bev_semantic_decoder import (
    BevSemanticLabels,
)
from py123d_garage.policy.transfuser.network.decoder.center_net_decoder import (
    CenterNetLabels,
)
from py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder import (
    FlowPlanningDecoder,
)
from py123d_garage.policy.transfuser.network.decoder.regression_planning_decoder import RegressionPlanningDecoder
from py123d_garage.policy.transfuser.network.decoder.trajectory_normalization import POSE_MEAN, POSE_STD
from py123d_garage.policy.transfuser.network.decoder.unicycle import physical_actions, rollout
from py123d_garage.policy.transfuser.network.encoder.transfuser import (
    GPT,
    Block,
    TransfuserBackbone,
    _resample_to,
)
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.policy.transfuser.visualization import render_ground_truth, render_prediction

_NUM_PATH_POINTS = 16
# A sampled trajectory step: the driven poses, the waypoints, the action rollout and the normalized actions.
_NUM_SAMPLE_COLUMNS = 11
_WAYPOINT_COLUMNS = slice(3, 6)
_ROLLOUT_COLUMNS = slice(6, 9)
_ACTION_COLUMNS = slice(9, 11)


@pytest.fixture(scope="module")
def config() -> TransfuserConfig:
    # No pretrained download: CI runs offline; the architecture is what is under test.
    return TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        backbone_config=BackboneConfig(
            image_encoder_pretrained=False,
        ),
        planning_config=PlanningConfig(path_config=PathConfig(num_points=_NUM_PATH_POINTS)),
    )


@pytest.fixture(scope="module")
def policy(config: TransfuserConfig) -> TransfuserPolicy:
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    return policy.eval()


def _features(
    config: TransfuserConfig,
    batch_size: int = 1,
) -> TransfuserFeatures:
    """The feature bundle of one batch, at the config's geometry."""
    return TransfuserFeatures(
        camera_feature=torch.randint(
            0,
            256,
            (batch_size, 3, config.camera_config.image_height, config.camera_config.image_width),
            dtype=torch.uint8,
        ),
        velocity=torch.zeros(batch_size, 1),
        velocity_vector=torch.zeros(batch_size, 2),
        acceleration_vector=torch.zeros(batch_size, 2),
        ego_history=torch.zeros(batch_size, config.ego_history_num_steps, len(PoseSE2Index))
        if config.ego_history_num_steps
        else None,
        lidar_feature=torch.rand(
            batch_size,
            1,
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
    )


def _small_geometry(config: TransfuserConfig) -> TransfuserConfig:
    """The config on a 64 x 128 image and a 32 m x 32 m BEV raster, for the statistical tests over large batches."""
    return replace(
        config,
        camera_config=replace(config.camera_config, image_width=128, image_height=64),
        lidar_config=replace(
            config.lidar_config,
            bev_min_x_m=0.0,
            bev_max_x_m=32.0,
            bev_min_y_m=-16.0,
            bev_max_y_m=16.0,
            bev_pixels_per_meter=2.0,
        ),
    )


def _navigation(
    config: TransfuserConfig,
    batch_size: int = 1,
) -> NavigationConditioning:
    return NavigationConditioning(
        target_points=torch.zeros(
            batch_size,
            len(config.required_target_point_distances_m),
            2,
        ),
        target_point_headings=torch.zeros(batch_size, len(config.required_target_point_distances_m)),
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_forward_shapes_at_asymmetric_geometry(
    config: TransfuserConfig,
    policy: TransfuserPolicy,
    batch_size: int,
) -> None:
    with torch.no_grad():
        predictions: TransfuserPredictions = policy(
            _features(config, batch_size),
            _navigation(config, batch_size),
        )

    assert predictions.trajectory is not None
    assert predictions.trajectory.shape == (batch_size, config.trajectory_num_steps, _NUM_SAMPLE_COLUMNS)
    assert predictions.path is not None
    assert predictions.path.shape == (batch_size, _NUM_PATH_POINTS, len(PoseSE2Index))
    assert predictions.bev_semantic is not None
    assert predictions.bev_semantic.shape == (
        batch_size,
        config.bev_semantic_config.num_bev_semantic_classes,
        config.lidar_config.bev_height_pixel,
        config.lidar_config.bev_width_pixel,
    )
    down = config.bev_semantic_config.bev_downsample_factor
    assert predictions.boxes is not None
    assert predictions.boxes.center_heatmap_pred.shape == (
        batch_size,
        config.box_detection_config.num_box_classes,
        config.lidar_config.bev_height_pixel // down,
        config.lidar_config.bev_width_pixel // down,
    )


@pytest.mark.parametrize(
    ("conditioning", "expected_steps"),
    [
        (PlanningConditioningConfig(use_velocity_vector=True), 0),
        (PlanningConditioningConfig(use_acceleration_vector=True), 0),
        (PlanningConditioningConfig(ego_history_duration_us=200_000), 2),
        (PlanningConditioningConfig(ego_history_duration_us=500_000), 5),
        (PlanningConditioningConfig(ego_history_duration_us=1_000_000), 10),
        (PlanningConditioningConfig(ego_history_duration_us=1_000_000, ego_history_single_token=True), 1),
    ],
    ids=[
        "velocity_vector",
        "acceleration_vector",
        "history_0_2s",
        "history_0_5s",
        "history_1s",
        "history_1s_single_token",
    ],
)
def test_ego_conditioning_adds_its_status_tokens(
    config: TransfuserConfig,
    conditioning: PlanningConditioningConfig,
    expected_steps: int,
) -> None:
    conditioned = replace(config, planning_conditioning_config=conditioning)
    assert conditioned.ego_history_num_steps == (
        conditioning.ego_history_duration_us // conditioning.ego_history_interval_us
    )
    assert conditioned.required_history_duration_us >= conditioning.ego_history_duration_us
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=conditioned))
    policy.initialize()
    encoder = policy.planning_decoder.planning_context_encoder
    enabled = (conditioning.use_velocity, conditioning.use_velocity_vector, conditioning.use_acceleration_vector)
    expected = len(conditioned.required_target_point_distances_m) + expected_steps + sum(enabled)
    assert encoder.num_status_tokens == expected

    with torch.no_grad():
        predictions: TransfuserPredictions = policy.eval()(
            _features(conditioned),
            _navigation(conditioned),
        )
    assert predictions.trajectory is not None
    assert predictions.trajectory.shape == (1, conditioned.trajectory_num_steps, _NUM_SAMPLE_COLUMNS)


def _labels(config: TransfuserConfig, batch_size: int = 1) -> TransfuserLabels:
    """A zeroed label bundle covering every enabled head."""
    num_steps = config.trajectory_num_steps
    path_config = config.planning_config.path_config
    down = config.bev_semantic_config.bev_downsample_factor
    height, width = (
        config.lidar_config.bev_height_pixel // down,
        config.lidar_config.bev_width_pixel // down,
    )
    return TransfuserLabels(
        trajectory=torch.zeros(batch_size, num_steps, len(PoseSE2Index)),
        action=torch.zeros(batch_size, num_steps, 2) if config.planning_config.unicycle_action_config else None,
        path=torch.zeros(batch_size, path_config.num_points, len(PoseSE2Index)) if path_config else None,
        bev_semantic=BevSemanticLabels(
            bev_semantic=torch.zeros(
                batch_size,
                config.lidar_config.bev_height_pixel,
                config.lidar_config.bev_width_pixel,
                dtype=torch.uint8,
            ),
            bev_semantic_has_map=torch.ones(batch_size, dtype=torch.bool),
            bev_semantic_has_box_annotations=torch.ones(batch_size, dtype=torch.bool),
        ),
        center_net=CenterNetLabels(
            center_net_heatmap=torch.zeros(
                batch_size,
                config.box_detection_config.num_box_classes,
                height,
                width,
            ),
            center_net_wh=torch.zeros(batch_size, 2, height, width),
            center_net_offset=torch.zeros(batch_size, 2, height, width),
            center_net_yaw_class=torch.zeros(
                batch_size,
                height,
                width,
                dtype=torch.uint8,
            ),
            center_net_yaw_res=torch.zeros(batch_size, 1, height, width),
            center_net_pixel_weight=torch.zeros(batch_size, 2, height, width),
            center_net_avg_factor=torch.zeros(batch_size),
            center_net_has_box_annotations=torch.ones(batch_size, dtype=torch.bool),
            center_net_velocity=torch.zeros(batch_size, 1, height, width)
            if config.box_detection_config.predict_box_velocity
            else None,
        ),
    )


def test_loss_covers_every_enabled_head(
    config: TransfuserConfig,
    policy: TransfuserPolicy,
) -> None:
    batch_size = 1
    labels = _labels(config, batch_size)

    predictions = policy(
        _features(config, batch_size),
        _navigation(config, batch_size),
    )
    losses = policy.compute_loss(labels, predictions)

    enabled_weights = {name for name, weight in policy.loss_weights(0).items() if weight > 0.0}
    assert set(losses) == enabled_weights
    assert all(torch.isfinite(value) for value in losses.values())


def test_metrics_cover_every_enabled_head(
    config: TransfuserConfig,
    policy: TransfuserPolicy,
) -> None:
    labels = _labels(config)
    with torch.no_grad():
        predictions = policy(_features(config), _navigation(config))
        metrics = policy.compute_metrics(labels, predictions)

    expected_names: set[str] = set()
    if config.perspective_config.use_semantic:
        expected_names |= {"semantic_miou", "semantic_f1"}
    if config.perspective_config.use_depth:
        expected_names |= {"depth_mae"}
    if config.bev_semantic_config.use_bev_semantic:
        expected_names |= {"bev_semantic_miou", "bev_semantic_f1"}
    grouped_names: set[str] = set()
    if config.planning_config.use_planning_decoder:
        expected_names |= {
            "waypoints_ade",
            "waypoints_fde",
            "waypoints_aye",
            "waypoints_along_track_error",
            "waypoints_cross_track_error",
        }
        expected_names |= {name.replace("waypoints_", "rollout_") for name in expected_names if "waypoints_" in name}
        grouped_names = {
            f"{output}_{direction}_track_error_{group}"
            for output in ("waypoints", "rollout")
            for direction in ("along", "cross")
            for group in ("below_5mps", "5_to_15mps", "above_15mps", "turning")
        }
    assert set(metrics) == expected_names | grouped_names
    assert all(torch.isfinite(metrics[name]) for name in expected_names)


def test_forward_traces_under_torch_compile() -> None:
    """
    The bundles must survive dynamo tracing; eager backend keeps the test fast.

    Runs in a subprocess with the jaxtyping import hook off: tracing the hook's
    beartype wrapper trips dynamo on CPU for bundle and dict inputs alike, so it
    would mask what this test is about.
    """
    script = """
import torch

from py123d.geometry import Point2DIndex, PoseSE2Index

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

config = TransfuserConfig(
    trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
    required_target_point_distances_m=[45.0],
    backbone_config=BackboneConfig(image_encoder_pretrained=False),
)
policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
policy.initialize()
compiled_policy = torch.compile(policy.eval(), backend="eager")
features = TransfuserFeatures(
    camera_feature=torch.rand(
        1,
        3,
        config.camera_config.image_height,
        config.camera_config.image_width,
    ),
    velocity=torch.zeros(1, 1),
    lidar_feature=torch.rand(
        1,
        1,
        config.lidar_config.bev_height_pixel,
        config.lidar_config.bev_width_pixel,
    ),
)
navigation = NavigationConditioning(
    target_points=torch.zeros(1, len(config.required_target_point_distances_m), 2),
    target_point_headings=torch.zeros(1, len(config.required_target_point_distances_m)),
)
with torch.no_grad():
    predictions = compiled_policy(features, navigation)
assert predictions.trajectory is not None
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=os.environ | {"PY123D_GARAGE_RUNTIME_TYPE_CHECKING": "false"},
    )


def _flow_policy(config: TransfuserConfig) -> tuple[TransfuserConfig, TransfuserPolicy]:
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    return config, policy


def test_flow_decoder_samples_the_same_poses_without_autograd(config: TransfuserConfig) -> None:
    flow, policy = _flow_policy(config)
    features, navigation = _features(flow), _navigation(flow)
    expected_shape = (1, flow.trajectory_num_steps, _NUM_SAMPLE_COLUMNS)
    with torch.no_grad():
        first: TransfuserPredictions = policy.eval()(features, navigation)
        second: TransfuserPredictions = policy.eval()(features, navigation)
        # The training views run the forward in train mode without autograd.
        views: TransfuserPredictions = policy.train()(features, navigation)
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert views.trajectory is not None
    assert first.trajectory.shape == expected_shape
    assert views.trajectory.shape == expected_shape
    assert torch.equal(first.trajectory, second.trajectory)

    # The sampler switches the velocity net's dropout off and puts the mode back afterwards; the train-mode
    # views call above moved the backbone's BatchNorm statistics, so the eval reference is drawn again.
    with torch.no_grad():
        reference: TransfuserPredictions = policy.eval()(features, navigation)
        policy.planning_decoder.train().planning_context_encoder.eval()
        head_in_train: TransfuserPredictions = policy(features, navigation)
    assert policy.planning_decoder.training
    assert reference.trajectory is not None
    assert head_in_train.trajectory is not None
    assert torch.equal(reference.trajectory, head_in_train.trajectory)


def test_flow_decoder_sampling_knobs(config: TransfuserConfig) -> None:
    """Temperature 0 ignores the seed, fresh noise differs per call, averaged draws keep one plan per sample."""
    flow, policy = _flow_policy(config)
    decoder = cast(FlowPlanningDecoder, policy.planning_decoder)
    token_dim = flow.planning_config.transfuser_token_dim
    # The speed token closes the context.
    context_tokens = torch.cat([torch.randn(2, 8, token_dim), torch.full((2, 1, token_dim), 5.0)], dim=1)

    def sample(**knobs: Any) -> torch.Tensor:
        decoder._config = replace(  # pyright: ignore[reportPrivateUsage]
            flow,
            planning_config=replace(flow.planning_config, flow_config=FlowMatchingConfig(**knobs)),
        )
        return decoder.sample(context_tokens)

    assert torch.equal(sample(noise_temperature=0.0, noise_seed=0), sample(noise_temperature=0.0, noise_seed=1))
    assert not torch.equal(sample(noise_temperature=1.0, noise_seed=0), sample(noise_temperature=1.0, noise_seed=1))
    assert not torch.equal(
        sample(noise_temperature=1.0, resample_noise=True),
        sample(noise_temperature=1.0, resample_noise=True),
    )
    assert torch.equal(sample(num_averaged_samples=1), sample())
    assert sample(num_averaged_samples=4).shape == sample().shape


def test_flow_decoder_trains_on_the_velocity_field(config: TransfuserConfig) -> None:
    flow, policy = _flow_policy(config)
    predictions: TransfuserPredictions = policy.train()(_features(flow), _navigation(flow))
    assert predictions.trajectory is not None
    assert predictions.trajectory.shape[-1] == flow.planning_config.transfuser_token_dim

    labels = _labels(flow)
    loss = policy.compute_loss(labels, predictions)["loss_trajectory"]
    assert torch.isfinite(loss)
    assert loss.requires_grad
    metrics = policy.compute_metrics(labels, predictions)
    assert {"waypoints_ade", "waypoints_fde", "waypoints_aye"} <= metrics.keys()


def test_gated_attention_planner_trains_at_any_width(config: TransfuserConfig) -> None:
    plain = config
    gated = replace(
        plain,
        planning_config=replace(
            plain.planning_config,
            transfuser_num_bev_cross_attention_layers=8,
            transfuser_token_dim=384,
            gated_attention=True,
        ),
    )
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=gated))
    policy.initialize()
    features, navigation = _features(gated), _navigation(gated)
    predictions: TransfuserPredictions = policy.train()(features, navigation)
    policy.compute_loss(_labels(gated), predictions)["loss_trajectory"].backward()

    parameters = dict(policy.named_parameters())
    for layer in range(8):
        for gate in ["self_attn_gate", "cross_attn_gate"]:
            gradient = parameters[f"planning_decoder.transformer_decoder.layers.{layer}.{gate}.weight"].grad
            assert gradient is not None
            assert gradient.abs().sum() > 0
    with torch.no_grad():
        sampled: TransfuserPredictions = policy.eval()(features, navigation)
    assert sampled.trajectory is not None
    assert torch.isfinite(sampled.trajectory).all()


def test_flow_decoder_scales_poses_by_their_statistics(config: TransfuserConfig) -> None:
    """The leading rows of POSE_STD and POSE_MEAN scale and shift the poses; another trajectory grid is refused."""
    mismatched = replace(config, trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000))
    with pytest.raises(ValueError, match="offsets"):
        TransfuserPolicy(PolicyConfig(transfuser_config=mismatched)).initialize()

    flow, policy = _flow_policy(config)
    decoder = cast(FlowPlanningDecoder, policy.planning_decoder)
    num_steps = flow.trajectory_num_steps
    assert torch.equal(cast(torch.Tensor, decoder.pose_scale), torch.tensor(POSE_STD[:num_steps]))

    # The same noise and weights under a zero mean differ from the shifted waypoints by the mean alone.
    token_dim = flow.planning_config.transfuser_token_dim
    context_tokens = torch.cat([torch.randn(1, 8, token_dim), torch.full((1, 1, token_dim), 5.0)], dim=1)
    with torch.no_grad():
        shifted = decoder.sample(context_tokens)[:, :num_steps, _WAYPOINT_COLUMNS]
        cast(torch.Tensor, decoder.pose_mean).zero_()
        centered = decoder.sample(context_tokens)[:, :num_steps, _WAYPOINT_COLUMNS]
    assert torch.allclose(shifted - centered, torch.tensor(POSE_MEAN[:num_steps])[None], atol=1e-5)


def test_lateral_target_point_noise_moves_along_the_route_normal(config: TransfuserConfig) -> None:
    noisy_config = replace(
        _small_geometry(config),
        planning_conditioning_config=PlanningConditioningConfig(target_point_lateral_noise_std_m=4.0),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=noisy_config))
    policy.initialize()
    seen_target_points: list[torch.Tensor] = []
    policy.planning_decoder.register_forward_pre_hook(lambda _, args: seen_target_points.append(args[1].clone()))
    batch_size = 64
    navigation = NavigationConditioning(
        target_points=torch.tensor([[[45.0, 10.0]]]).expand(batch_size, 1, 2).clone(),
        target_point_headings=torch.full((batch_size, 1), torch.pi / 2.0),
    )

    with torch.no_grad():
        policy.train()(_features(noisy_config, batch_size), navigation)
        policy.eval()(_features(noisy_config, batch_size), navigation)

    train_target_points, eval_target_points = seen_target_points
    # A route heading +y has its left normal along -x, so only x moves.
    torch.testing.assert_close(train_target_points[..., 1], navigation.target_points[..., 1])
    assert 2.0 < float((train_target_points[..., 0] - 45.0).std()) < 6.0
    torch.testing.assert_close(eval_target_points, navigation.target_points)


@pytest.mark.parametrize("correlated", [False, True])
def test_correlated_lateral_noise_shifts_every_target_point_alike(config: TransfuserConfig, correlated: bool) -> None:
    noisy_config = replace(
        _small_geometry(config),
        required_target_point_distances_m=[40.0, 60.0, 80.0],
        planning_conditioning_config=PlanningConditioningConfig(
            target_point_lateral_noise_std_m=1.0,
            target_point_lateral_noise_correlated=correlated,
        ),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=noisy_config))
    policy.initialize()
    seen_target_points: list[torch.Tensor] = []
    policy.planning_decoder.register_forward_pre_hook(lambda _, args: seen_target_points.append(args[1].clone()))
    batch_size = 64
    navigation = NavigationConditioning(
        target_points=torch.tensor([[[40.0, 0.0], [60.0, 0.0], [80.0, 0.0]]]).expand(batch_size, 3, 2).clone(),
        target_point_headings=torch.zeros(batch_size, 3),
    )

    with torch.no_grad():
        policy.train()(_features(noisy_config, batch_size), navigation)

    # A route heading +x has its left normal along +y, so only y moves.
    lateral_offsets_m = seen_target_points[0][..., 1]
    torch.testing.assert_close(seen_target_points[0][..., 0], navigation.target_points[..., 0])
    assert 0.5 < float(lateral_offsets_m.std()) < 1.5
    spread_within_sample_m = float((lateral_offsets_m - lateral_offsets_m[:, :1]).abs().max())
    assert (spread_within_sample_m < 1e-6) == correlated


@pytest.mark.slow
def test_target_point_dropout_flips_a_coin_per_sample_and_point(config: TransfuserConfig) -> None:
    three_points = replace(
        _small_geometry(config),
        required_target_point_distances_m=[40.0, 60.0, 80.0],
        planning_conditioning_config=PlanningConditioningConfig(target_point_dropout_probability=0.5),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=three_points))
    policy.initialize()
    seen_target_points: list[torch.Tensor] = []
    policy.planning_decoder.register_forward_pre_hook(lambda _, args: seen_target_points.append(args[1].clone()))
    batch_size = 256
    navigation = NavigationConditioning(
        target_points=torch.tensor([[[40.0, 0.0], [60.0, 0.0], [80.0, 0.0]]]).expand(batch_size, 3, 2).clone(),
        target_point_headings=torch.zeros(batch_size, 3),
    )

    with torch.no_grad():
        policy.train()(_features(three_points, batch_size), navigation)
        policy.eval()(_features(three_points, batch_size), navigation)

    train_target_points, eval_target_points = seen_target_points
    dropped = train_target_points.isnan().all(dim=-1)
    assert 0.44 < float(dropped.float().mean()) < 0.56
    assert 0.06 < float(dropped.all(dim=1).float().mean()) < 0.2
    assert bool((dropped.any(dim=1) & ~dropped.all(dim=1)).any())
    torch.testing.assert_close(eval_target_points, navigation.target_points)


def _seen_target_points(
    config: TransfuserConfig,
    conditioning: PlanningConditioningConfig,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The target points the planning decoder receives in one training and one inference forward, 3 points."""
    three_points = replace(
        _small_geometry(config),
        required_target_point_distances_m=[40.0, 60.0, 80.0],
        planning_conditioning_config=conditioning,
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=three_points))
    policy.initialize()
    seen_target_points: list[torch.Tensor] = []
    policy.planning_decoder.register_forward_pre_hook(lambda _, args: seen_target_points.append(args[1].clone()))
    navigation = NavigationConditioning(
        target_points=torch.tensor([[[40.0, 0.0], [60.0, 0.0], [80.0, 0.0]]]).expand(batch_size, 3, 2).clone(),
        target_point_headings=torch.zeros(batch_size, 3),
    )
    with torch.no_grad():
        torch.manual_seed(1)
        policy.train()(_features(three_points, batch_size), navigation)
        policy.eval()(_features(three_points, batch_size), navigation)
    return seen_target_points[0], seen_target_points[1]


@pytest.mark.slow
def test_target_point_dropout_per_point_drops_each_at_its_own_rate(config: TransfuserConfig) -> None:
    train_target_points, eval_target_points = _seen_target_points(
        config,
        PlanningConditioningConfig(target_point_dropout_probabilities=[0.9, 0.5, 0.0]),
    )

    dropped_share_per_point = train_target_points.isnan().all(dim=-1).float().mean(dim=0)
    assert 0.84 < float(dropped_share_per_point[0]) < 0.96
    assert 0.42 < float(dropped_share_per_point[1]) < 0.58
    assert float(dropped_share_per_point[2]) == 0.0
    assert not bool(eval_target_points.isnan().any())


@pytest.mark.slow
def test_equal_per_point_dropout_matches_the_single_probability(config: TransfuserConfig) -> None:
    per_point_train, _ = _seen_target_points(
        config,
        PlanningConditioningConfig(target_point_dropout_probabilities=[0.5, 0.5, 0.5]),
    )
    single_train, _ = _seen_target_points(config, PlanningConditioningConfig(target_point_dropout_probability=0.5))

    torch.testing.assert_close(per_point_train, single_train, equal_nan=True)


@pytest.mark.parametrize(
    ("conditioning", "message"),
    [
        (PlanningConditioningConfig(target_point_dropout_probabilities=[0.9, 0.5]), "2 values for 3 target points"),
        (PlanningConditioningConfig(target_point_dropout_probabilities=[0.9, 0.5, 1.0]), "must lie in"),
        (
            PlanningConditioningConfig(
                target_point_dropout_probability=0.5,
                target_point_dropout_probabilities=[0.9] * 3,
            ),
            "not both",
        ),
    ],
    ids=["wrong_length", "one_is_unconditional", "both_set"],
)
def test_per_point_dropout_rejects_invalid_settings(
    config: TransfuserConfig,
    conditioning: PlanningConditioningConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(config, required_target_point_distances_m=[40.0, 60.0, 80.0], planning_conditioning_config=conditioning)


def _target_point_tokens(
    config: TransfuserConfig,
    target_points: torch.Tensor,
) -> tuple[TransfuserPolicy, torch.Tensor]:
    """A fresh policy and its target point tokens, without their positional embedding, for one sample."""
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    encoder = policy.planning_decoder.planning_context_encoder
    assert encoder.tp_pos_embedding is not None
    seen_context: list[torch.Tensor] = []
    encoder.register_forward_hook(lambda _, __, output: seen_context.append(cast(torch.Tensor, output)))
    navigation = NavigationConditioning(
        target_points=target_points[None],
        target_point_headings=torch.zeros(1, target_points.shape[0]),
    )
    with torch.no_grad():
        policy.eval()(_features(config), navigation)
    # The target point tokens close the context.
    num_target_points = target_points.shape[0]
    return policy, seen_context[0][0, -num_target_points:] - encoder.tp_pos_embedding[0]


def test_missing_target_points_read_as_the_null_token(config: TransfuserConfig) -> None:
    two_points = replace(config, required_target_point_distances_m=[45.0, 80.0])
    policy, (present, missing) = _target_point_tokens(two_points, torch.tensor([[45.0, 10.0], [torch.nan, torch.nan]]))
    null_token = policy.planning_decoder.planning_context_encoder.tp_null_token
    assert null_token is not None
    torch.testing.assert_close(missing, null_token[0, 0])
    assert not torch.allclose(present, null_token[0, 0])


def test_target_points_start_as_one_token(config: TransfuserConfig) -> None:
    two_points = replace(config, required_target_point_distances_m=[45.0, 80.0])
    _, (near, far) = _target_point_tokens(two_points, torch.tensor([[45.0, 10.0], [80.0, -20.0]]))
    torch.testing.assert_close(near, far)


def test_no_target_points_build_no_target_point_parameters(config: TransfuserConfig) -> None:
    unconditional = replace(config, required_target_point_distances_m=[])
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=unconditional))
    policy.initialize()
    assert not [name for name, _ in policy.named_parameters() if ".tp_" in name]

    with torch.no_grad():
        predictions: TransfuserPredictions = policy.eval()(_features(unconditional), _navigation(unconditional))
    assert predictions.trajectory is not None


def test_checkpoints_load_under_any_number_of_target_points(config: TransfuserConfig) -> None:
    unconditional = TransfuserPolicy(
        PolicyConfig(transfuser_config=replace(config, required_target_point_distances_m=[])),
    )
    three_points = TransfuserPolicy(
        PolicyConfig(transfuser_config=replace(config, required_target_point_distances_m=[40.0, 60.0, 80.0])),
    )
    unconditional.initialize()
    three_points.initialize()

    missing, unexpected = three_points.load_state_dict(unconditional.state_dict(), strict=False)

    prefix = "planning_decoder.planning_context_encoder."
    assert sorted(missing) == [
        f"{prefix}tp_encoder.bias",
        f"{prefix}tp_encoder.weight",
        f"{prefix}tp_null_token",
        f"{prefix}tp_pos_embedding",
    ]
    assert not unexpected


def test_fusion_stages_take_their_own_heads_and_layers(config: TransfuserConfig) -> None:
    per_stage = replace(
        config,
        backbone_config=replace(
            config.backbone_config,
            n_head_per_stage=[4, 4, 4, 8],
            n_layer_per_stage=[2, 2, 2, 4],
        ),
    )
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=per_stage))
    policy.initialize()
    backbone = cast(TransfuserBackbone, policy.backbone).train()
    features = _features(per_stage, batch_size=2)

    bev_features, image_features, _ = backbone(features.camera_feature.float(), features.lidar_feature)

    fusion_transformers = [cast(GPT, gpt) for gpt in backbone.transformers]
    assert [cast(Block, gpt.blocks[0]).attn.n_head for gpt in fusion_transformers] == [4, 4, 4, 8]
    assert [len(gpt.blocks) for gpt in fusion_transformers] == [2, 2, 2, 4]
    assert torch.isfinite(bev_features).all()
    assert torch.isfinite(image_features).all()


@pytest.mark.slow
def test_relative_speed_noise_scales_with_speed(config: TransfuserConfig) -> None:
    noisy_config = replace(
        _small_geometry(config),
        planning_conditioning_config=PlanningConditioningConfig(speed_relative_noise_std=0.2),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=noisy_config))
    policy.initialize()
    seen_speeds: list[torch.Tensor] = []
    policy.planning_decoder.register_forward_pre_hook(lambda _, args: seen_speeds.append(args[2].clone()))
    batch_size = 256
    features = _features(noisy_config, batch_size)
    moving_features = replace(features, velocity=torch.full_like(features.velocity, 10.0))
    stopped_features = replace(features, velocity=torch.zeros_like(features.velocity))
    navigation = _navigation(noisy_config, batch_size)

    with torch.no_grad():
        policy.train()(moving_features, navigation)
        policy.train()(stopped_features, navigation)
        policy.eval()(moving_features, navigation)

    train_moving_speeds, train_stopped_speeds, eval_moving_speeds = seen_speeds
    assert 1.5 < float(train_moving_speeds.std()) < 2.5
    assert abs(float(train_moving_speeds.mean()) - 10.0) < 0.5
    assert torch.all(train_stopped_speeds == 0.0)
    torch.testing.assert_close(eval_moving_speeds, moving_features.velocity)


def test_regression_decoder_plans_poses_on_its_own_grid(config: TransfuserConfig) -> None:
    """The CARLA presets' regression head: (x, y, yaw) per 0.25 s step, no path, the trajectory loss alone."""
    regression = replace(
        config,
        trajectory_offsets_us=uniform_offsets_us(250_000, 2_000_000),
        planning_config=PlanningConfig(
            planning_decoder_target=(
                "py123d_garage.policy.transfuser.network.decoder.regression_planning_decoder:RegressionPlanningDecoder"
            ),
            unicycle_action_config=None,
            path_config=None,
        ),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=regression))
    policy.initialize()
    with torch.no_grad():
        plan: TransfuserPredictions = policy.eval()(_features(regression), _navigation(regression))
    assert plan.trajectory is not None
    assert plan.trajectory.shape == (1, regression.trajectory_num_steps, len(PoseSE2Index))
    assert plan.path is None

    predictions: TransfuserPredictions = policy.train()(_features(regression), _navigation(regression))
    losses = policy.compute_loss(_labels(regression), predictions)
    assert set(losses) == {name for name, weight in policy.loss_weights(0).items() if weight > 0.0}
    assert "loss_path" not in losses
    assert "loss_action" not in losses
    losses["loss_trajectory"].backward()
    assert policy.planning_decoder.wp_decoder.weight.grad is not None


def _action_policy(
    config: TransfuserConfig,
    action_config: ActionConfig,
) -> tuple[TransfuserConfig, TransfuserPolicy]:
    actions = replace(config, planning_config=replace(config.planning_config, unicycle_action_config=action_config))
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=actions))
    policy.initialize()
    return actions, policy


def test_planning_decoder_needs_an_action_loss_term(config: TransfuserConfig) -> None:
    with pytest.raises(AssertionError, match="only action loss term"):
        _action_policy(config, ActionConfig(rollout_pose_loss=False, action_loss=False, bounded=False))
    with pytest.raises(AssertionError, match="trains through the rollout pose loss"):
        _action_policy(config, ActionConfig(rollout_pose_loss=False, action_loss=True))
    with pytest.raises(AssertionError, match="squashes the lateral-acceleration actions"):
        _action_policy(config, ActionConfig(lateral_accel_actions=False))


def test_sampled_actions_stay_within_bounds_and_roll_out_to_the_rollout(config: TransfuserConfig) -> None:
    actions_config, policy = _action_policy(config, ActionConfig())
    action_config = actions_config.planning_config.unicycle_action_config
    assert action_config is not None
    features = replace(_features(actions_config), velocity=torch.full((1, 1), 20.0))
    with torch.no_grad():
        sampled: TransfuserPredictions = policy.eval()(features, _navigation(actions_config))
    assert sampled.trajectory is not None
    actions = sampled.trajectory[..., _ACTION_COLUMNS]
    speed = torch.tensor([20.0])
    acceleration, lateral_acceleration = physical_actions(actions, speed, action_config).unbind(-1)
    assert bool((acceleration >= -action_config.max_brake_mps2).all())
    assert bool((acceleration <= action_config.max_accel_mps2).all())
    assert bool((lateral_acceleration.abs() <= action_config.max_lateral_accel_mps2 + 1e-3).all())
    expected_poses = rollout(actions, speed, action_config)
    assert torch.allclose(sampled.trajectory[..., _ROLLOUT_COLUMNS], expected_poses, atol=1e-5)


def test_rollout_speed_loss_adds_the_speed_error(config: TransfuserConfig) -> None:
    losses: list[torch.Tensor] = []
    for rollout_speed_loss in (False, True):
        actions_config, policy = _action_policy(config, ActionConfig(rollout_speed_loss=rollout_speed_loss))
        features = replace(_features(actions_config), velocity=torch.full((1, 1), 8.0))
        predictions: TransfuserPredictions = policy.train()(features, _navigation(actions_config))
        labels = replace(_labels(actions_config), action=torch.ones(1, actions_config.trajectory_num_steps, 2))
        torch.manual_seed(1)
        losses.append(policy.compute_loss(labels, predictions)["loss_action"])
    pose_only_loss, with_speed_loss = losses
    assert with_speed_loss > pose_only_loss
    with_speed_loss.backward()
    action_decoder_grad = cast(
        FlowPlanningDecoder,
        policy.planning_decoder,
    ).action_decoder.weight.grad
    assert action_decoder_grad is not None
    assert action_decoder_grad.abs().sum() > 0


def test_frozen_backbone_stays_in_eval_mode_while_the_heads_train(config: TransfuserConfig) -> None:
    frozen = replace(config, backbone_config=replace(config.backbone_config, freeze_backbone=True))
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=frozen))
    policy.initialize()
    policy.train()
    assert not any(parameter.requires_grad for parameter in policy.backbone.parameters())
    assert not any(module.training for module in policy.backbone.modules())
    assert all(parameter.requires_grad for parameter in policy.planning_decoder.parameters())
    assert policy.planning_decoder.training


def test_planning_metrics_split_the_error_along_and_across_the_logged_heading(
    config: TransfuserConfig,
    policy: TransfuserPolicy,
) -> None:
    time_s = torch.tensor(config.trajectory_offsets_us, dtype=torch.float32) / 1e6
    label = torch.zeros(1, config.trajectory_num_steps, len(PoseSE2Index))
    label[..., 0] = 10.0 * time_s
    predictions = label.clone()
    predictions[..., 1] += 1.0
    metrics = RegressionPlanningDecoder.compute_metrics(policy.planning_decoder, predictions, label)
    torch.testing.assert_close(metrics["waypoints_cross_track_error"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["waypoints_along_track_error"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["waypoints_cross_track_error_5_to_15mps"], torch.tensor(1.0))
    assert torch.isnan(metrics["waypoints_cross_track_error_above_15mps"])
    assert torch.isnan(metrics["waypoints_cross_track_error_turning"])


@pytest.mark.parametrize("channels_last", [False, True], ids=["nchw", "channels_last"])
@pytest.mark.parametrize(
    ("in_size", "out_size"),
    [((8, 32), (64, 256)), ((8, 8), (64, 64)), ((8, 32), (11, 44)), ((8, 11), (16, 22)), ((5, 7), (3, 4))],
)
def test_bilinear_matmul_matches_interpolate(
    in_size: tuple[int, int],
    out_size: tuple[int, int],
    channels_last: bool,
) -> None:
    """
    The matrix-product up-sample computes bilinear's values and gradients, including uneven ratios and edges, and
    keeps a channels-last input's layout.
    """
    features = torch.randn(2, 3, *in_size, dtype=torch.float64)
    if channels_last:
        features = features.to(memory_format=torch.channels_last)
    features.requires_grad_()
    reference = _resample_to(features, torch.Size(out_size), "bilinear")
    matmul = _resample_to(features, torch.Size(out_size), "bilinear_matmul")
    assert matmul.is_contiguous(memory_format=torch.channels_last) == channels_last
    torch.testing.assert_close(matmul, reference, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(reference)
    (reference_grad,) = torch.autograd.grad(reference, features, upstream)
    (matmul_grad,) = torch.autograd.grad(matmul, features, upstream)
    torch.testing.assert_close(matmul_grad, reference_grad, atol=1e-6, rtol=1e-6)


def test_head_upsamples_match_interpolate(config: TransfuserConfig, policy: TransfuserPolicy) -> None:
    """The top-down path and the BEV semantic head up-sample as matrix products with bilinear's values."""
    backbone = policy.backbone
    lidar = config.lidar_config
    down = config.bev_semantic_config.bev_downsample_factor
    torch.manual_seed(0)
    bev_features = torch.randn(2, backbone.num_lidar_features, lidar.lidar_bev_grid_rows, lidar.lidar_bev_grid_cols)

    def bilinear(features: torch.Tensor, **size: Any) -> torch.Tensor:
        return F.interpolate(features, mode="bilinear", align_corners=False, **size)

    with torch.no_grad():
        p5 = F.relu(backbone.c5_conv(bev_features))
        p4 = F.relu(backbone.up_conv5(bilinear(p5, scale_factor=config.bev_semantic_config.bev_upsample_factor)))
        reference_grid = F.relu(
            backbone.up_conv4(bilinear(p4, size=(lidar.bev_height_pixel // down, lidar.bev_width_pixel // down))),
        )
        grid = backbone.top_down(bev_features)
        torch.testing.assert_close(grid, reference_grid, atol=1e-5, rtol=1e-5)

        decoder = policy.bev_semantic_decoder
        reference_logits = bilinear(decoder.net[:3](grid), size=(lidar.bev_height_pixel, lidar.bev_width_pixel))
        torch.testing.assert_close(decoder(grid), reference_logits, atol=1e-5, rtol=1e-5)


def test_predicted_and_labelled_paths_are_drawn(config: TransfuserConfig) -> None:
    flow_path, policy = _flow_policy(config)
    features, navigation = _features(flow_path), _navigation(flow_path)
    with torch.no_grad():
        plan: TransfuserPredictions = policy.eval()(features, navigation)
    # A straight path 2 to 32 m ahead, inside the BEV.
    path = torch.zeros(1, _NUM_PATH_POINTS, len(PoseSE2Index))
    path[..., 0] = torch.linspace(2.0, 32.0, _NUM_PATH_POINTS)
    plan = replace(plan, path=path)
    target_points = navigation.target_points
    assert (
        render_prediction(features, target_points, plan, flow_path)
        != render_prediction(features, target_points, replace(plan, path=None), flow_path)
    ).any()
    labels = _labels(flow_path)
    assert (
        render_ground_truth(features, replace(labels, path=path), target_points, flow_path)
        != render_ground_truth(features, replace(labels, path=None), target_points, flow_path)
    ).any()


def test_test_time_target_points_drop_the_rest(config: TransfuserConfig) -> None:
    three_points = replace(config, required_target_point_distances_m=[40.0, 60.0, 80.0])
    served = replace(
        three_points,
        planning_conditioning_config=replace(
            three_points.planning_conditioning_config,
            test_time_target_point_distances_m=[80.0],
        ),
    )
    torch.manual_seed(0)
    all_points = TransfuserPolicy(PolicyConfig(transfuser_config=three_points))
    all_points.initialize()
    only_far = TransfuserPolicy(PolicyConfig(transfuser_config=served))
    only_far.initialize()
    only_far.load_state_dict(all_points.state_dict())
    features = _features(three_points)
    navigation = _navigation(three_points)
    navigation = replace(navigation, target_points=torch.tensor([[[40.0, 1.0], [60.0, 3.0], [80.0, 6.0]]]))
    near_dropped = replace(navigation, target_points=navigation.target_points.clone())
    near_dropped.target_points[:, :2] = torch.nan
    with torch.no_grad():
        served_far = only_far.eval()(features, navigation).trajectory
        dropped_near = all_points.eval()(features, near_dropped).trajectory
        fed_all = all_points.eval()(features, navigation).trajectory
    assert served_far is not None
    assert dropped_near is not None
    assert fed_all is not None
    torch.testing.assert_close(served_far, dropped_near)
    assert not torch.allclose(served_far, fed_all)


@pytest.mark.parametrize("speed_dropout", [0.999, 1.0])
def test_speed_dropout_swaps_the_speed_token(config: TransfuserConfig, speed_dropout: float) -> None:
    """Training reads the null token; inference reads the speed, unless the speed is always dropped."""
    dropped = replace(
        config,
        planning_conditioning_config=replace(
            config.planning_conditioning_config,
            speed_dropout_probability=speed_dropout,
        ),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=dropped))
    policy.initialize()
    policy.eval()
    encoder = policy.planning_decoder.planning_context_encoder
    assert encoder.speed_null_token is not None
    speed_tokens: list[torch.Tensor] = []
    # The speed token is the first status token.
    encoder.register_forward_hook(
        lambda _, __, output: speed_tokens.append(cast(torch.Tensor, output)[:, -encoder.num_status_tokens]),
    )

    def speed_token(speed_mps: float, training: bool) -> torch.Tensor:
        encoder.train(training)
        with torch.no_grad():
            policy(replace(_features(dropped), velocity=torch.full((1, 1), speed_mps)), _navigation(dropped))
        return speed_tokens[-1]

    null_token = (encoder.speed_null_token + encoder.status_pos_embedding[:, :1]).detach()[0]
    torch.testing.assert_close(speed_token(3.0, training=True), null_token)
    torch.testing.assert_close(speed_token(20.0, training=True), null_token)
    if speed_dropout >= 1.0:
        torch.testing.assert_close(speed_token(3.0, training=False), null_token)
        torch.testing.assert_close(speed_token(20.0, training=False), null_token)
    else:
        assert not torch.allclose(speed_token(3.0, training=False), speed_token(20.0, training=False))
