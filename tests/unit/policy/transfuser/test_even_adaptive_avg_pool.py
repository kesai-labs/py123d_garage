# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""EvenAdaptiveAvgPool2d computes adaptive_avg_pool2d's means and gradients, with avg_pool2d where the map tiles evenly."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from py123d_garage.policy.transfuser.network.encoder.transfuser import EvenAdaptiveAvgPool2d


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [((128, 88), (16, 11)), ((64, 64), (8, 8)), ((16, 11), (16, 11)), ((15, 11), (4, 11)), ((32, 32), (8, None))],
)
def test_matches_adaptive_avg_pool2d(input_size: tuple[int, int], output_size: tuple[int | None, int]) -> None:
    features = torch.randn(2, 3, *input_size, dtype=torch.float64, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    pooled = EvenAdaptiveAvgPool2d(output_size)(features)
    reference = F.adaptive_avg_pool2d(reference_features, output_size)
    torch.testing.assert_close(pooled, reference)
    weights = torch.randn_like(reference)
    (pooled * weights).sum().backward()
    (reference * weights).sum().backward()
    torch.testing.assert_close(features.grad, reference_features.grad)
