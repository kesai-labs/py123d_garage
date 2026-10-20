# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The CUDA depthwise convolution matches torch's conv2d in values and gradients, and composes with compile."""

from __future__ import annotations

import copy

import pytest
import timm
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
py123d_garage_cuda = pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason  # noqa: E402

if (_unsupported := unsupported_reason("depthwise_conv")) is not None:
    pytest.skip(_unsupported, allow_module_level=True)

from py123d_garage_cuda.transfuser.convnext import CudaDepthwiseConv2d, patch_depthwise_convs  # noqa: E402

# Shapes for each path of the bf16 kernels: tensor cores on sm_90+ for 7 x 7 with C % 16 == 0 (else pipelined for
# C % 32 == 0), tiled (even C), narrow (W <= 16); for the tensor cores also one 16-channel chunk, an image smaller than a
# tile, a width just past the narrow tiles, and a ConvNeXt-T stage-1 slice.
SHAPES = [(2, 64, 19, 37), (3, 40, 17, 23), (2, 96, 11, 16), (2, 16, 33, 65), (3, 48, 5, 7), (2, 32, 16, 17)]
STAGE_1_SLICE = (4, 96, 88, 128)


def _inputs(
    dtype: torch.dtype,
    kernel: int,
    bias: bool,
    shape: tuple[int, int, int, int] = SHAPES[0],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(0)
    channels = shape[1]
    x = torch.randn(*shape, device="cuda").to(dtype).contiguous(memory_format=torch.channels_last)
    weight = torch.randn(channels, 1, kernel, kernel, device="cuda").to(dtype)
    b = torch.randn(channels, device="cuda").to(dtype) if bias else None
    return x, weight, b


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kernel", [3, 5, 7])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
def test_matches_conv2d(
    shape: tuple[int, int, int, int],
    kernel: int,
    bias: bool,
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    x, weight, b = _inputs(dtype, kernel, bias, shape)
    leaves = [t.detach().requires_grad_() for t in (x, weight, b) if t is not None]
    reference_leaves = [t.detach().double().requires_grad_() for t in leaves]
    y = py123d_garage_cuda.depthwise_conv(leaves[0], leaves[1], leaves[2] if bias else None)
    reference = F.conv2d(
        reference_leaves[0],
        reference_leaves[1],
        reference_leaves[2] if bias else None,
        padding=kernel // 2,
        groups=x.shape[1],
    )
    assert y.is_contiguous(memory_format=torch.channels_last)
    scale = reference.abs().max().item()
    torch.testing.assert_close(y.double(), reference, atol=tolerance * scale, rtol=tolerance)
    grad = torch.randn_like(y)
    y.backward(grad)
    reference.backward(grad.double())
    for leaf, reference_leaf in zip(leaves, reference_leaves, strict=True):
        assert leaf.grad is not None
        assert reference_leaf.grad is not None
        scale = reference_leaf.grad.abs().max().item()
        torch.testing.assert_close(leaf.grad.double(), reference_leaf.grad, atol=tolerance * scale, rtol=tolerance)


def test_stage_1_slice_matches_conv2d() -> None:
    test_matches_conv2d(STAGE_1_SLICE, 7, True, torch.bfloat16, 3e-2)


def test_backward_is_deterministic() -> None:
    x, weight, _ = _inputs(torch.bfloat16, 7, False, STAGE_1_SLICE)
    grad = torch.randn_like(x)
    first = torch.ops.py123d_garage_cuda.depthwise_conv_backward(grad, x, weight)
    second = torch.ops.py123d_garage_cuda.depthwise_conv_backward(grad, x, weight)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_opcheck() -> None:
    x, weight, b = _inputs(torch.float32, 7, True)
    args = (x.requires_grad_(), weight.requires_grad_(), b.requires_grad_())  # pyright: ignore[reportOptionalMemberAccess]
    torch.library.opcheck(torch.ops.py123d_garage_cuda.depthwise_conv.default, args)


def test_patched_convnext_trains_like_cudnn() -> None:
    torch.manual_seed(0)
    encoder = timm.create_model("convnext_tiny", pretrained=False, features_only=True).cuda()
    encoder = encoder.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    reference = copy.deepcopy(encoder)
    keys = list(encoder.state_dict())
    patch_depthwise_convs(encoder)
    assert list(encoder.state_dict()) == keys
    assert sum(type(m) is CudaDepthwiseConv2d for m in encoder.modules()) == 18
    x = torch.randn(2, 3, 128, 160, device="cuda").contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features = encoder(x)[-1].float()
        reference_features = reference(x)[-1].float()
    scale = reference_features.abs().max().item()
    torch.testing.assert_close(features, reference_features, atol=3e-2 * scale, rtol=3e-2)
    features.square().mean().backward()
    reference_features.square().mean().backward()
    for (name, parameter), reference_parameter in zip(encoder.named_parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        scale = reference_parameter.grad.abs().max().item()
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=5e-2 * scale, rtol=5e-2, msg=name)
