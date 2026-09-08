"""Forward-pass smoke test: the asymmetric BEV geometry must propagate through every head."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
from py123d.geometry import Point2DIndex, PoseSE2Index

from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.labels import TransfuserLabels
from py123d_garage.policy.transfuser.network.bev_semantic_decoder import (
    BevSemanticLabels,
)
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    CenterNetLabels,
)
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy


def _trajectory_columns(config: TransfuserConfig) -> int:
    """Width of the trajectory tensors: (x, y), plus yaw when the head predicts it."""
    return len(PoseSE2Index) if config.planning_config.predict_yaw else len(Point2DIndex)


@pytest.fixture(scope="module")
def config() -> TransfuserConfig:
    # No pretrained download: CI runs offline; the architecture is what is under test.
    return TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        backbone_config=BackboneConfig(
            image_encoder_pretrained=False,
        ),
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
        camera_feature=torch.rand(
            batch_size,
            3,
            config.camera_config.image_height,
            config.camera_config.image_width,
        )
        * 255,
        velocity=torch.zeros(batch_size, 1),
        lidar_feature=torch.rand(
            batch_size,
            1,
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
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
    assert predictions.trajectory.shape == (
        batch_size,
        config.trajectory_num_steps,
        _trajectory_columns(config),
    )
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


def _labels(config: TransfuserConfig, batch_size: int = 1) -> TransfuserLabels:
    """A zeroed label bundle covering every enabled head."""
    down = config.bev_semantic_config.bev_downsample_factor
    height, width = (
        config.lidar_config.bev_height_pixel // down,
        config.lidar_config.bev_width_pixel // down,
    )
    return TransfuserLabels(
        trajectory=torch.zeros(
            batch_size,
            config.trajectory_num_steps,
            _trajectory_columns(config),
        ),
        bev_semantic=BevSemanticLabels(
            bev_semantic=torch.zeros(
                batch_size,
                config.lidar_config.bev_height_pixel,
                config.lidar_config.bev_width_pixel,
                dtype=torch.int64,
            ),
            bev_semantic_has_map=torch.ones(batch_size, dtype=torch.bool),
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
                dtype=torch.int64,
            ),
            center_net_yaw_res=torch.zeros(batch_size, 1, height, width),
            center_net_pixel_weight=torch.zeros(batch_size, 2, height, width),
            center_net_avg_factor=torch.zeros(batch_size),
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

    with torch.no_grad():
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
    if config.planning_config.use_planning_decoder:
        expected_names |= {"waypoints_ade", "waypoints_fde"}
        if config.planning_config.predict_yaw:
            expected_names |= {"waypoints_aye"}
    assert set(metrics) == expected_names
    assert all(torch.isfinite(value) for value in metrics.values())


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

from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

config = TransfuserConfig(
    trajectory_horizon_us=4_000_000,
    trajectory_interval_us=500_000,
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
