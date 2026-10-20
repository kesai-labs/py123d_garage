# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""CenterNet loss: a sample of a source without box annotations adds no loss and no gradient."""

from __future__ import annotations

import pytest
import torch

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.policy.transfuser.network.decoder.center_net_decoder import CenterNetDecoder, CenterNetLabels

_HEIGHT, _WIDTH = 8, 8


def _config() -> TransfuserConfig:
    return TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
    )


def _labels(config: TransfuserConfig, annotated: list[bool]) -> CenterNetLabels:
    """One box per sample, at its own pixel, so every sample has a loss of its own."""
    batch_size = len(annotated)
    heatmap = torch.zeros(batch_size, config.box_detection_config.num_box_classes, _HEIGHT, _WIDTH)
    pixel_weight = torch.zeros(batch_size, 2, _HEIGHT, _WIDTH)
    for sample in range(batch_size):
        heatmap[sample, 0, sample, sample] = 1.0
        pixel_weight[sample, :, sample, sample] = 1.0
    return CenterNetLabels(
        center_net_heatmap=heatmap,
        center_net_wh=torch.ones(batch_size, 2, _HEIGHT, _WIDTH),
        center_net_offset=torch.full((batch_size, 2, _HEIGHT, _WIDTH), 0.5),
        center_net_yaw_class=torch.ones(batch_size, _HEIGHT, _WIDTH, dtype=torch.uint8),
        center_net_yaw_res=torch.full((batch_size, 1, _HEIGHT, _WIDTH), 0.1),
        center_net_pixel_weight=pixel_weight,
        center_net_avg_factor=torch.ones(batch_size),
        center_net_has_box_annotations=torch.tensor(annotated),
    )


def test_unannotated_sample_adds_no_loss_and_no_gradient() -> None:
    config = _config()
    torch.manual_seed(0)
    decoder = CenterNetDecoder(config)
    grid = torch.randn(2, config.box_detection_config.box_head_input_channels, _HEIGHT, _WIDTH, requires_grad=True)

    losses = decoder.compute_loss(decoder(grid), _labels(config, [True, False]))
    alone = decoder.compute_loss(decoder(grid[:1]), _labels(config, [True]))
    torch.stack(list(losses.values())).sum().backward()  # pyright: ignore[reportUnknownMemberType]

    for name, loss in losses.items():
        assert loss.item() == pytest.approx(alone[name].item(), rel=1e-5), name
    assert grid.grad is not None
    assert torch.equal(grid.grad[1], torch.zeros_like(grid.grad[1]))


def test_fully_annotated_batch_keeps_the_plain_mean() -> None:
    config = _config()
    torch.manual_seed(0)
    decoder = CenterNetDecoder(config)
    grid = torch.randn(2, config.box_detection_config.box_head_input_channels, _HEIGHT, _WIDTH)

    both = decoder.compute_loss(decoder(grid), _labels(config, [True, True]))
    first = decoder.compute_loss(decoder(grid[:1]), _labels(config, [True]))
    second_labels = _labels(config, [True, True])
    second = decoder.compute_loss(
        decoder(grid[1:]),
        CenterNetLabels(**{name: tensor[1:] for name, tensor in second_labels.named_tensors()}),
    )

    for name, loss in both.items():
        assert loss.item() == pytest.approx((first[name].item() + second[name].item()) / 2, rel=1e-5), name
