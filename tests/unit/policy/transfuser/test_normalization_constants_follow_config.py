# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The planning context's normalization constants come from the config, never from loaded weights."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, TransfuserConfig
from py123d_garage.policy.transfuser.network.decoder.regression_planning_decoder import PlanningContextEncoder

_CONSTANTS = (
    "ego_history_normalization_constants",
    "velocity_normalization_constants",
    "acceleration_normalization_constants",
    "target_points_normalization_constants",
)


def _constants(encoder: PlanningContextEncoder) -> dict[str, torch.Tensor]:
    return {name: cast(torch.Tensor, getattr(encoder, name)).clone() for name in _CONSTANTS}


def _encoder(velocity_scale: float) -> PlanningContextEncoder:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        backbone_config=BackboneConfig(image_encoder_pretrained=False),
    )
    config = replace(
        config,
        planning_conditioning_config=replace(
            config.planning_conditioning_config,
            velocity_normalization_constants=[
                velocity_scale * constant
                for constant in config.planning_conditioning_config.velocity_normalization_constants
            ],
        ),
    )
    torch.manual_seed(0)
    return PlanningContextEncoder(64, config)


def test_constants_are_not_saved_and_saved_copies_do_not_override_the_config() -> None:
    trained = _encoder(1.0)
    finetuned = _encoder(2.0)
    expected = _constants(finetuned)
    assert not torch.equal(
        expected["velocity_normalization_constants"],
        _constants(trained)["velocity_normalization_constants"],
    )

    state_dict = trained.state_dict()
    assert not any(key.endswith("_normalization_constants") for key in state_dict)
    finetuned.load_state_dict(state_dict, strict=True)
    assert all(torch.equal(value, expected[name]) for name, value in _constants(finetuned).items())

    legacy = dict(state_dict)
    for name in _CONSTANTS:
        legacy[name] = torch.full_like(expected[name], -1.0)
    finetuned.load_state_dict(legacy, strict=True)
    assert all(torch.equal(value, expected[name]) for name, value in _constants(finetuned).items())
