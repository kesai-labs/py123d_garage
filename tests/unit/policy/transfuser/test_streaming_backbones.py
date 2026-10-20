# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Streaming a rollout frame by frame must match the canvas forward pass the model trains on."""

from __future__ import annotations

import pytest
import torch
from py123d.geometry import PoseSE2Index

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, CameraConfig, TransfuserConfig
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.network.encoder.streaming_transfuser import (
    FrameTokens,
    StreamingTransfuserBackbone,
)
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

_ENCODER = "py123d_garage.policy.transfuser.network.encoder"


def _policy(backbone_target: str, num_past_camera_rows: int) -> TransfuserPolicy:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(
            image_width=256,
            image_height=128 * (1 + num_past_camera_rows),
            past_camera_offset_us=500_000,
            num_past_camera_rows=num_past_camera_rows,
        ),
        backbone_config=BackboneConfig(
            image_encoder_pretrained=False,
            latent=True,
            backbone_target=backbone_target,
        ),
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    return policy.eval()


def _features(camera_feature: torch.Tensor) -> TransfuserFeatures:
    return TransfuserFeatures(
        camera_feature=camera_feature,
        velocity=torch.zeros(1, 1),
        velocity_vector=torch.zeros(1, 2),
        acceleration_vector=torch.zeros(1, 2),
        ego_history=None,
        lidar_feature=None,
    )


def _navigation() -> NavigationConditioning:
    return NavigationConditioning(
        target_points=torch.zeros(1, 1, 2),
        target_point_headings=torch.zeros(1, 1),
    )


@pytest.mark.parametrize("num_past_camera_rows", [1, 2])
def test_streaming_matches_the_canvas(num_past_camera_rows: int) -> None:
    policy = _policy(f"{_ENCODER}.streaming_transfuser:StreamingTransfuserBackbone", num_past_camera_rows)
    frames = [torch.randint(0, 256, (1, 3, 128, 256), dtype=torch.uint8) for _ in range(4)]
    past_frame_tokens: list[FrameTokens] = []
    with torch.no_grad():
        for step, frame in enumerate(frames):
            # The canvas repeats the oldest frame a rollout has, oldest row on top.
            rows = [frames[max(step - offset, 0)] for offset in reversed(range(1 + num_past_camera_rows))]
            canvas_predictions = policy.forward(_features(torch.cat(rows, dim=2)), _navigation())
            streamed_predictions, frame_tokens = policy.forward_streaming(
                _features(frame),
                _navigation(),
                past_frame_tokens,
            )
            past_frame_tokens = [frame_tokens, *past_frame_tokens][:num_past_camera_rows]
            assert canvas_predictions.trajectory is not None
            assert streamed_predictions.trajectory is not None
            torch.testing.assert_close(
                streamed_predictions.trajectory[..., : len(PoseSE2Index)],
                canvas_predictions.trajectory[..., : len(PoseSE2Index)],
                atol=1e-4,
                rtol=1e-4,
            )


def test_streaming_image_tokens_ignore_the_past() -> None:
    policy = _policy(f"{_ENCODER}.streaming_transfuser:StreamingTransfuserBackbone", 1)
    backbone = policy.backbone
    assert isinstance(backbone, StreamingTransfuserBackbone)
    frame = torch.rand(1, 3, 128, 256) * 255
    with torch.no_grad():
        alone = backbone.stream(frame, [], None)[3]
        after_past = backbone.stream(frame, [backbone.stream(torch.rand_like(frame) * 255, [], None)[3]], None)[3]
    for alone_tokens, after_past_tokens in zip(alone, after_past, strict=True):
        torch.testing.assert_close(alone_tokens, after_past_tokens)
