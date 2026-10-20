# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The fused fusion residual, x + bilinear_upsample(grid), matches torch's bilinear interpolation in fp64."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason

if (_unsupported := unsupported_reason("fusion_upsample_add")) is not None:
    pytest.skip(_unsupported, allow_module_level=True)

from py123d_garage_cuda.transfuser.fusion import fusion_upsample_add, upsample_add
from py123d_garage_cuda.transfuser.registry import recording_fallbacks

from py123d_garage.policy.transfuser.network.encoder.transfuser import (
    _resample_to,  # pyright: ignore[reportPrivateUsage]
)

# The 200 recipe's image (16 x 11 anchors) and BEV (8 x 8) stages, and odd factors and sizes.
SHAPES = [
    (2, 64, 16, 11, 128, 88),
    (2, 128, 16, 11, 64, 44),
    (2, 256, 16, 11, 32, 22),
    (2, 64, 8, 8, 64, 64),
    (3, 16, 5, 7, 15, 21),
    (1, 8, 3, 4, 10, 9),
]


@pytest.mark.parametrize(("batch", "channels", "grid_height", "grid_width", "height", "width"), SHAPES)
def test_matches_bilinear_interpolation(
    batch: int,
    channels: int,
    grid_height: int,
    grid_width: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch, channels, height, width, device="cuda").to(torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last).requires_grad_(True)
    grid = torch.randn(batch, channels, grid_height, grid_width, device="cuda").to(torch.bfloat16)
    grid = grid.contiguous(memory_format=torch.channels_last).requires_grad_(True)
    out = fusion_upsample_add(x, grid)
    assert out.is_contiguous(memory_format=torch.channels_last)

    x64 = x.detach().double().requires_grad_(True)
    grid64 = grid.detach().double().requires_grad_(True)
    reference = x64 + F.interpolate(grid64, size=(height, width), mode="bilinear", align_corners=False)
    # The matmul form the recipe trains with is the same interpolation.
    torch.testing.assert_close(
        x64 + _resample_to(grid64, torch.Size((height, width)), "bilinear_matmul"),
        reference,
        rtol=1e-12,
        atol=1e-12,
    )
    # One bf16 rounding of the fp32 sum.
    torch.testing.assert_close(out.double(), reference, rtol=2**-8, atol=1e-5)

    grad = torch.randn(batch, channels, height, width, device="cuda").to(torch.bfloat16)
    grad = grad.contiguous(memory_format=torch.channels_last)
    out.backward(grad)
    reference.backward(grad.double())
    assert x.grad is not None
    assert grid.grad is not None
    assert grid64.grad is not None
    torch.testing.assert_close(x.grad.double(), grad.double(), rtol=0, atol=0)
    scale = grid64.grad.abs().max().item()
    torch.testing.assert_close(grid.grad.double(), grid64.grad, rtol=2**-8, atol=1e-3 * scale)


def test_a_shared_map_of_batch_one_runs_on_the_kernel_with_the_broadcast_sum() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 64, 64, 64, device="cuda").to(torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last).requires_grad_(True)
    grid = torch.randn(4, 64, 8, 8, device="cuda").to(torch.bfloat16)
    grid = grid.contiguous(memory_format=torch.channels_last).requires_grad_(True)
    with recording_fallbacks() as fallbacks, torch.autocast("cuda", dtype=torch.bfloat16):
        out = upsample_add(x, grid)
    assert fallbacks == {}
    assert out is not None

    x64 = x.detach().double().requires_grad_(True)
    grid64 = grid.detach().double().requires_grad_(True)
    reference = x64 + F.interpolate(grid64, size=(64, 64), mode="bilinear", align_corners=False)
    torch.testing.assert_close(out.double(), reference, rtol=2**-8, atol=1e-5)

    grad = torch.randn(4, 64, 64, 64, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    out.backward(grad)
    reference.backward(grad.double())
    assert x.grad is not None
    assert x64.grad is not None
    assert grid.grad is not None
    assert grid64.grad is not None
    torch.testing.assert_close(x.grad.double(), x64.grad, rtol=2**-7, atol=1e-2)
    scale = grid64.grad.abs().max().item()
    torch.testing.assert_close(grid.grad.double(), grid64.grad, rtol=2**-8, atol=1e-3 * scale)
