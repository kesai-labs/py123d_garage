# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The CUDA grouped convolution of RegNet matches torch's conv2d in values and gradients and composes with compile."""

from __future__ import annotations

import pytest
import timm
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
py123d_garage_cuda = pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason  # noqa: E402
from py123d_garage_cuda.transfuser.regnet import CudaGroupedConv2d, patch_grouped_convs  # noqa: E402

_UNSUPPORTED = unsupported_reason("regnet_grouped_conv")
_kernel = pytest.mark.skipif(_UNSUPPORTED is not None, reason=_UNSUPPORTED or "")

# (N, C, H, W): one and two 72-channel chunks, odd and even sizes, tiles cut by the image border.
SHAPES = [(2, 72, 19, 37), (3, 144, 16, 8), (1, 216, 5, 3), (2, 72, 34, 21)]


def _inputs(shape: tuple[int, int, int, int], weight_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    weight = (torch.randn(shape[1], 24, 3, 3, device="cuda") / 8).to(weight_dtype)
    return x, weight


@_kernel
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_format", [torch.contiguous_format, torch.channels_last])
def test_matches_conv2d(
    shape: tuple[int, int, int, int],
    stride: int,
    weight_dtype: torch.dtype,
    weight_format: torch.memory_format,
) -> None:
    x, weight = _inputs(shape, weight_dtype)
    weight = weight.contiguous(memory_format=weight_format)
    x.requires_grad_()
    weight.requires_grad_()
    reference_x = x.detach().double().requires_grad_()
    reference_weight = weight.detach().to(torch.bfloat16).double().requires_grad_()
    y = py123d_garage_cuda.grouped_conv(x, weight, stride)
    reference = F.conv2d(reference_x, reference_weight, stride=stride, padding=1, groups=shape[1] // 24)
    assert y.dtype == torch.bfloat16
    assert y.is_contiguous(memory_format=torch.channels_last)
    # fp32 accumulation, then one bf16 rounding.
    torch.testing.assert_close(y.double(), reference, atol=1e-5, rtol=2**-8)
    grad = torch.randn_like(y)
    y.backward(grad)
    reference.backward(grad.double())
    assert x.grad is not None
    assert weight.grad is not None
    assert reference_x.grad is not None
    assert reference_weight.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert x.grad.is_contiguous(memory_format=torch.channels_last)
    assert weight.grad.dtype == weight_dtype
    assert weight.grad.stride() == weight.stride()
    torch.testing.assert_close(x.grad.double(), reference_x.grad, atol=1e-5, rtol=2**-8)
    weight_tolerance = 1e-5 if weight_dtype == torch.float32 else 2**-8
    scale = reference_weight.grad.abs().max().item()
    torch.testing.assert_close(
        weight.grad.double(),
        reference_weight.grad,
        atol=1e-5 * scale,
        rtol=weight_tolerance,
    )


@_kernel
@pytest.mark.parametrize("stride", [1, 2])
def test_backward_is_deterministic(stride: int) -> None:
    x, weight = _inputs((4, 144, 40, 24), torch.float32)
    grad = torch.randn(4, 144, 40 // stride, 24 // stride, device="cuda").to(torch.bfloat16)
    grad = grad.contiguous(memory_format=torch.channels_last)
    first = torch.ops.py123d_garage_cuda.regnet_grouped_conv_backward(grad, x, weight, stride)
    second = torch.ops.py123d_garage_cuda.regnet_grouped_conv_backward(grad, x, weight, stride)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


@_kernel
@pytest.mark.parametrize("stride", [1, 2])
def test_opcheck(stride: int) -> None:
    x, weight = _inputs(SHAPES[0], torch.float32)
    args = (x.requires_grad_(), weight.requires_grad_(), stride)
    torch.library.opcheck(torch.ops.py123d_garage_cuda.regnet_grouped_conv.default, args)


def test_patch_keeps_the_state_and_swaps_every_grouped_conv() -> None:
    torch.manual_seed(0)
    encoder = timm.create_model("regnety_032", pretrained=False, features_only=True).cuda()
    keys = list(encoder.state_dict())
    patch_grouped_convs(encoder)
    assert list(encoder.state_dict()) == keys
    assert sum(type(m) is CudaGroupedConv2d for m in encoder.modules()) == 21


@_kernel
def test_compiled_matches_eager() -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(144, 144, 3, stride=2, padding=1, groups=6, bias=False).cuda()
    patch_grouped_convs(conv)
    assert type(conv) is CudaGroupedConv2d
    x = torch.randn(2, 144, 20, 28, device="cuda").contiguous(memory_format=torch.channels_last)
    compiled = torch.compile(conv, fullgraph=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager = conv(x)
        traced = compiled(x)
    assert torch.equal(eager, traced)
