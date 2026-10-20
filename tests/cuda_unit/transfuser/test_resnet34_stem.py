# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The CUDA ResNet stem (convolution, batch statistics, batch norm -> ReLU -> max-pool) matches torch."""

from __future__ import annotations

import copy
from typing import cast

import pytest
import timm
import torch
import torch.nn.functional as F
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
py123d_garage_cuda = pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason  # noqa: E402

if (_unsupported := unsupported_reason("resnet_stem_conv")) is not None:
    pytest.skip(_unsupported, allow_module_level=True)

from py123d_garage_cuda.transfuser.resnet import CudaStemConv, CudaStemNorm, fuse_stem  # noqa: E402

# Full 16 x 16 output tiles, and edge tiles in both directions.
SHAPES = [(2, 3, 64, 96), (3, 3, 70, 46), (2, 2, 38, 30), (1, 2, 33, 21)]


def _stem_inputs(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    padded = F.pad(x, (0, 0, 0, 0, 0, 4 - shape[1])).contiguous(memory_format=torch.channels_last)
    weight = (torch.randn(64, shape[1], 7, 7, device="cuda") * 0.1).to(torch.bfloat16)
    return x, padded, weight


@pytest.mark.parametrize("shape", SHAPES)
def test_stem_conv_matches_conv2d_as_closely_as_cudnn(shape: tuple[int, int, int, int]) -> None:
    x, padded, weight = _stem_inputs(shape)
    y, partial = py123d_garage_cuda.resnet_stem_conv(padded, weight)
    reference = F.conv2d(x.double(), weight.double(), stride=2, padding=3)
    cudnn_error = (F.conv2d(x, weight, stride=2, padding=3).double() - reference).abs().max()
    assert y.is_contiguous(memory_format=torch.channels_last)
    assert (y.double() - reference).abs().max() <= cudnn_error + 1e-6

    norm_weight = torch.rand(64, device="cuda") + 0.5
    norm_bias = torch.randn(64, device="cuda")
    values = py123d_garage_cuda.batch_norm_finalize(partial, norm_weight, norm_bias, 1e-5)
    mean = y.double().mean((0, 2, 3))
    variance = y.double().var((0, 2, 3), unbiased=False)
    rstd = (variance + 1e-5).rsqrt()
    expected = torch.stack([mean, variance, rstd, norm_weight * rstd, norm_bias - mean * norm_weight * rstd])
    torch.testing.assert_close(values.double(), expected, rtol=1e-5, atol=1e-5)

    grad = torch.randn_like(y)
    grad_weight = torch.ops.py123d_garage_cuda.resnet_stem_conv_weight_grad(grad, padded, shape[1])
    arguments = (None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1, (False, True, False))
    reference_grad = torch.ops.aten.convolution_backward(grad.double(), x.double(), weight.double(), *arguments)[1]
    cudnn_grad = torch.ops.aten.convolution_backward(grad, x, weight, *arguments)[1]
    cudnn_grad_error = (cudnn_grad.double() - reference_grad).abs().max()
    assert (grad_weight.double() - reference_grad).abs().max() <= 1.5 * cudnn_grad_error + 1e-6


def test_stem_conv_is_deterministic() -> None:
    _, padded, weight = _stem_inputs((4, 3, 96, 64))
    y, partial = py123d_garage_cuda.resnet_stem_conv(padded, weight)
    grad = torch.randn_like(y)
    grad_weight = torch.ops.py123d_garage_cuda.resnet_stem_conv_weight_grad(grad, padded, 3)
    for _ in range(3):
        again, again_partial = py123d_garage_cuda.resnet_stem_conv(padded, weight)
        assert torch.equal(again, y)
        assert torch.equal(again_partial, partial)
        assert torch.equal(torch.ops.py123d_garage_cuda.resnet_stem_conv_weight_grad(grad, padded, 3), grad_weight)


def _resnet(in_chans: int) -> nn.Module:
    torch.manual_seed(0)
    return timm.create_model("resnet34", pretrained=False, in_chans=in_chans, num_classes=0).cuda()


def _stem(model: nn.Module) -> tuple[nn.Conv2d, nn.BatchNorm2d]:
    return cast(nn.Conv2d, model.conv1), cast(nn.BatchNorm2d, model.bn1)


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return ((value.float() - reference.float()).abs().max() / reference.float().abs().max()).item()


def test_cuda_fuse_stem_keeps_the_state_and_only_patches_stems() -> None:
    model = _resnet(3)
    keys = list(model.state_dict())
    fuse_stem(model)
    assert list(model.state_dict()) == keys
    conv, norm = _stem(model)
    assert type(conv) is CudaStemConv
    assert type(norm) is CudaStemNorm
    assert sum(type(module) in (CudaStemConv, CudaStemNorm) for module in model.modules()) == 2


@pytest.mark.parametrize("in_chans", [3, 2])
def test_cuda_fuse_stem_trains_like_the_unpatched_model(in_chans: int) -> None:
    reference = _resnet(in_chans).train()
    patched = copy.deepcopy(reference)
    fuse_stem(patched)
    full = copy.deepcopy(reference)
    x = torch.randn(4, in_chans, 128, 96, device="cuda")
    results = []
    for model in (reference, patched):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out.float().square().mean().backward()
        conv, norm = _stem(model)
        results.append((out, conv.weight.grad, norm.weight.grad, norm.bias.grad))
    out = full(x)
    out.square().mean().backward()
    conv, norm = _stem(full)
    fp32 = (out, conv.weight.grad, norm.weight.grad, norm.bias.grad)
    # bf16 noise reaches ~20 % of the stem's gradients; the patched model must sit as close to fp32 as torch's own bf16.
    for value, reference_value, fp32_value in zip(results[1], results[0], fp32, strict=True):
        assert value is not None
        assert reference_value is not None
        assert fp32_value is not None
        assert _relative_error(value, fp32_value) <= 1.25 * _relative_error(reference_value, fp32_value) + 1e-3
    patched_norm, reference_norm = _stem(patched)[1], _stem(reference)[1]
    for name in ("running_mean", "running_var"):
        torch.testing.assert_close(getattr(patched_norm, name), getattr(reference_norm, name), rtol=1e-4, atol=1e-5)
    assert patched_norm.num_batches_tracked is not None
    assert reference_norm.num_batches_tracked is not None
    assert torch.equal(patched_norm.num_batches_tracked, reference_norm.num_batches_tracked)


def test_cuda_fuse_stem_falls_back_in_eval_and_in_fp32() -> None:
    reference = _resnet(3)
    patched = copy.deepcopy(reference)
    fuse_stem(patched)
    x = torch.randn(2, 3, 64, 48, device="cuda")
    reference.eval()
    patched.eval()
    with torch.no_grad():
        torch.testing.assert_close(patched(x), reference(x), rtol=0, atol=0)
    reference.train()
    patched.train()
    torch.testing.assert_close(patched(x), reference(x), rtol=0, atol=0)


def test_cuda_fuse_stem_runs_a_frozen_stem_in_eval_like_torch() -> None:
    reference = _resnet(3)
    reference.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference(torch.randn(4, 3, 64, 48, device="cuda") * 2 + 1)
    reference.eval().requires_grad_(False)
    patched = copy.deepcopy(reference)
    fuse_stem(patched)
    x = torch.randn(2, 3, 128, 96, device="cuda")
    stems = []
    for model in (reference, patched):
        conv, norm = _stem(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            stems.append(cast(nn.Module, model.maxpool)(cast(nn.Module, model.act1)(norm(conv(x)))))
    assert stems[1].shape == stems[0].shape
    # One bf16 rounding of the stem output apart at most.
    torch.testing.assert_close(stems[1], stems[0], rtol=1e-2, atol=1e-2)


def test_cuda_fuse_stem_compiles() -> None:
    patched = _resnet(3).train()
    fuse_stem(patched)
    compiled = cast(nn.Module, torch.compile(copy.deepcopy(patched)))
    x = torch.randn(2, 3, 64, 96, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager = patched(x)
        out = compiled(x)
    out.float().sum().backward()
    torch.testing.assert_close(out, eager, rtol=2e-2, atol=2e-2)
    assert _stem(compiled)[0].weight.grad is not None
