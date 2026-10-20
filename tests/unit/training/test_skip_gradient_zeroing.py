# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The batched gradient copy without zeroing leaves the flat buffer as the zeroed copy does, stale values included."""

from __future__ import annotations

import torch

from py123d_garage.training.concurrent_seeds_training import _copy_gradients  # pyright: ignore[reportPrivateUsage]


def _views_and_grads(zero_missing: bool) -> torch.Tensor:
    torch.manual_seed(0)
    parameters = [torch.nn.Parameter(torch.randn(shape)) for shape in ((3, 4), (5,), (2, 2))]
    flat = torch.zeros(sum(parameter.numel() for parameter in parameters) + 7)
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    offset = 0
    for parameter in parameters:
        views.append((parameter, flat[offset : offset + parameter.numel()].view(parameter.shape)))
        offset += parameter.numel()
    for step in range(2):
        if not zero_missing:
            flat.zero_()
        # The second parameter gets a gradient in the first step only, so its view holds a stale one after it.
        for index, (parameter, _) in enumerate(views):
            parameter.grad = None if (index == 1 and step == 1) else torch.randn(parameter.shape)
        _copy_gradients(views, zero_missing)
    return flat


def test_matches_the_zeroed_copy() -> None:
    torch.testing.assert_close(
        _views_and_grads(zero_missing=True),
        _views_and_grads(zero_missing=False),
        rtol=0,
        atol=0,
    )
