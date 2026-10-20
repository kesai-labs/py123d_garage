# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The camera layout TransFuser hands to the training's camera augmentation."""

from __future__ import annotations

import pytest
import torch
from py123d.datatypes import CameraID

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, CameraConfig, TransfuserConfig
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

_THREE_CAMERAS = {"physical-ai-av": [CameraID.FTCAM_L0, CameraID.FTCAM_F0, CameraID.FTCAM_R0]}


@pytest.mark.parametrize(
    ("past_camera_offset_us", "num_past_camera_rows", "layout"),
    [
        (0, 1, (3, 1, 1)),
        (1_000_000, 1, (3, 2, 1)),
        (500_000, 2, (3, 3, 1)),
    ],
)
def test_policy_hands_over_its_camera_layout(
    past_camera_offset_us: int,
    num_past_camera_rows: int,
    layout: tuple[int, int, int],
) -> None:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[],
        backbone_config=BackboneConfig(image_encoder_pretrained=False),
        camera_config=CameraConfig(
            input_cameras=_THREE_CAMERAS,
            past_camera_offset_us=past_camera_offset_us,
            num_past_camera_rows=num_past_camera_rows,
        ),
    )
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    features = TransfuserFeatures(
        camera_feature=torch.zeros(2, 3, 256, 1024, dtype=torch.uint8),
        velocity=torch.ones(2, 1),
    )
    calls: list[tuple[int, int, int]] = []

    def augment_camera(rgb: torch.Tensor, num_tiles: int, num_rows: int, front_tile: int) -> torch.Tensor:
        calls.append((num_tiles, num_rows, front_tile))
        return rgb + 1

    augmented = policy.augment_features(features, augment_camera)
    assert calls == [layout]
    assert torch.equal(augmented.camera_feature, features.camera_feature + 1)
    assert torch.equal(augmented.velocity, features.velocity)


def test_front_camera_is_found_by_its_id_not_its_position() -> None:
    camera = CameraConfig(input_cameras={"nuplan": [CameraID.PCAM_F0, CameraID.PCAM_R0]})
    assert camera.front_camera_index == 0


def test_front_camera_must_sit_at_the_same_position_everywhere() -> None:
    camera = CameraConfig(
        input_cameras={
            "nuplan": [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0],
            "physical-ai-av": [CameraID.FTCAM_F0],
        },
    )
    with pytest.raises(ValueError, match="same position"):
        _ = camera.front_camera_index
