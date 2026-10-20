# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Unconditional epochs: every target point dropped at first, the configured dropout from then on."""

from __future__ import annotations

from typing import cast

import torch

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    PlanningConditioningConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

_BATCH_SIZE = 2


def _policy() -> TransfuserPolicy:
    # No pretrained download: CI runs offline.
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
        required_target_point_distances_m=[40.0, 60.0, 80.0],
        backbone_config=BackboneConfig(image_encoder_pretrained=False),
        planning_conditioning_config=PlanningConditioningConfig(
            target_point_dropout_probability=0.0,
            unconditional_epochs=2,
        ),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    return policy.train()


def _dropped_share(policy: TransfuserPolicy) -> float:
    """The share of target points the planning decoder receives as NaN in one training forward."""
    config = policy.policy_config
    features = TransfuserFeatures(
        camera_feature=torch.zeros(
            _BATCH_SIZE,
            3,
            config.camera_config.image_height,
            config.camera_config.image_width,
            dtype=torch.uint8,
        ),
        velocity=torch.full((_BATCH_SIZE, 1), 10.0),
        velocity_vector=torch.zeros(_BATCH_SIZE, 2),
        acceleration_vector=torch.zeros(_BATCH_SIZE, 2),
        ego_history=None,
        lidar_feature=torch.zeros(
            _BATCH_SIZE,
            1,
            config.lidar_config.bev_height_pixel,
            config.lidar_config.bev_width_pixel,
        ),
    )
    navigation = NavigationConditioning(
        target_points=torch.ones(_BATCH_SIZE, 3, 2),
        target_point_headings=torch.zeros(_BATCH_SIZE, 3),
    )
    received: list[torch.Tensor] = []
    hook = policy.planning_decoder.register_forward_pre_hook(lambda _, args: received.append(args[1]))
    try:
        with torch.no_grad():
            policy(features, navigation)
    finally:
        hook.remove()
    return float(received[0].isnan().any(dim=-1).float().mean())


def test_dropout_is_one_in_the_unconditional_epochs_and_the_configured_one_after() -> None:
    policy = _policy()
    dropouts = []
    for epoch in range(4):
        policy.on_train_epoch_start(epoch)
        dropouts.append(cast(torch.Tensor, policy.target_point_dropout).tolist())
    assert dropouts == [[1.0] * 3, [1.0] * 3, [0.0] * 3, [0.0] * 3]


def test_planner_receives_no_target_points_in_the_unconditional_epochs_and_all_after() -> None:
    policy = _policy()
    policy.on_train_epoch_start(1)
    assert _dropped_share(policy) == 1.0
    policy.on_train_epoch_start(2)
    assert _dropped_share(policy) == 0.0
    # Inference keeps the configured dropout, 0.0 here.
    policy.eval()
    assert _dropped_share(policy) == 0.0
