# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The ResNet convolutions on pinned engines (cuDNN engine configurations and the halo kernels) match torch's conv2d in
values and gradients and compose with compile.
"""

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

from py123d_garage_cuda.transfuser import resnet  # noqa: E402
from py123d_garage_cuda.transfuser.registry import unsupported_reason  # noqa: E402
from py123d_garage_cuda.transfuser.resnet import (  # noqa: E402
    HALO_ENGINE,
    PinnedConv2d,
    pin_conv_engines,
    pinned_conv,
)

_ops = torch.ops.py123d_garage_cuda
_UNSUPPORTED = unsupported_reason("pinned_conv")
_halo = pytest.mark.skipif(_UNSUPPORTED is not None, reason=_UNSUPPORTED or "")
_PINNED = _UNSUPPORTED is None and torch.backends.cudnn.version() == resnet._PINNED_CUDNN_VERSION
_ENTRIES = sorted(resnet._PINNED_ENGINES.items())


def _error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return ((value.double() - reference).abs().max() / reference.abs().max()).item()


@pytest.mark.skipif(not _PINNED, reason="the engine table belongs to another cuDNN or GPU")
@pytest.mark.parametrize(("key", "config"), _ENTRIES, ids=[str(key) for key, _ in _ENTRIES])
def test_pinned_engine_matches_fp64_as_closely_as_torch(
    key: tuple[int, ...],
    config: tuple[int, tuple[int, ...]],
) -> None:
    pass_index, n, c, h, w, k, r, s, stride, padding = key
    engine, knobs = config
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    weight = (torch.randn(k, c, r, s, device="cuda") / (c * r * s) ** 0.5).to(torch.bfloat16)
    weight = weight.contiguous(memory_format=torch.channels_last)
    out_size = (n, k, (h + 2 * padding - r) // stride + 1, (w + 2 * padding - s) // stride + 1)
    grad = torch.randn(out_size, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    geometry = ([stride, stride], [padding, padding])
    arguments = (None, [stride, stride], [padding, padding], [1, 1], False, [0, 0], 1, [True, True, False])

    def pinned() -> torch.Tensor:
        if pass_index == 0:
            return _ops.pinned_conv(x, weight, *geometry, [engine, -1, -1], list(knobs), [], [])
        if pass_index == 1:
            return _ops.pinned_conv_data_grad(grad, weight, [n, c, h, w], *geometry, engine, list(knobs))
        return _ops.pinned_conv_weight_grad(grad, x, [k, c, r, s], *geometry, engine, list(knobs))

    value = pinned()
    if pass_index == 0:
        torch_value = F.conv2d(x, weight, stride=stride, padding=padding)
        reference = F.conv2d(x.double(), weight.double(), stride=stride, padding=padding)
    else:
        torch_value = torch.ops.aten.convolution_backward(grad, x, weight, *arguments)[pass_index - 1]
        reference = torch.ops.aten.convolution_backward(grad.double(), x.double(), weight.double(), *arguments)[
            pass_index - 1
        ]
    assert value.is_contiguous(memory_format=torch.channels_last)
    assert _error(value, reference) <= 1.05 * _error(torch_value, reference) + 1e-6
    assert torch.equal(pinned(), value)


# (N, H, W): whole 16 x 8 tiles, tiles cut by the image border in both directions, one image.
HALO_SHAPES = [(4, 32, 16), (3, 20, 12), (2, 17, 9), (1, 7, 30)]


@_halo
@pytest.mark.parametrize("shape", HALO_SHAPES)
def test_halo_kernels_match_fp64_as_closely_as_cudnn(shape: tuple[int, int, int]) -> None:
    n, h, w = shape
    torch.manual_seed(0)
    x = torch.randn(n, 64, h, w, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    weight = (
        (torch.randn(64, 64, 3, 3, device="cuda") / 24).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    )
    grad = torch.randn(n, 64, h, w, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    x.requires_grad_()
    weight.requires_grad_()
    out = pinned_conv(x, weight, [1, 1], [1, 1], [HALO_ENGINE] * 3, ([], [], []))
    out.backward(grad)
    reference_x = x.detach().double().requires_grad_()
    reference_weight = weight.detach().double().requires_grad_()
    reference = F.conv2d(reference_x, reference_weight, padding=1)
    reference.backward(grad.double())
    cudnn_x = x.detach().clone().requires_grad_()
    cudnn_weight = weight.detach().clone().requires_grad_()
    cudnn = F.conv2d(cudnn_x, cudnn_weight, padding=1)
    cudnn.backward(grad)
    assert x.grad is not None
    assert weight.grad is not None
    assert cudnn_x.grad is not None
    assert cudnn_weight.grad is not None
    assert reference_x.grad is not None
    assert reference_weight.grad is not None
    for value, cudnn_value, fp64 in (
        (out, cudnn, reference.detach()),
        (x.grad, cudnn_x.grad, reference_x.grad),
        (weight.grad, cudnn_weight.grad, reference_weight.grad),
    ):
        assert value.is_contiguous(memory_format=torch.channels_last)
        assert _error(value, fp64) <= 1.05 * _error(cudnn_value, fp64) + 1e-6


@_halo
def test_halo_kernels_are_deterministic() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64, 40, 24, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    weight = (
        (torch.randn(64, 64, 3, 3, device="cuda") / 24).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    )
    geometry = ([1, 1], [1, 1])
    outputs = [
        (
            _ops.pinned_conv(x, weight, *geometry, [HALO_ENGINE, -1, -1], [], [], []),
            _ops.pinned_conv_data_grad(x, weight, list(x.shape), *geometry, HALO_ENGINE, []),
            _ops.pinned_conv_weight_grad(x, x, [64, 64, 3, 3], *geometry, HALO_ENGINE, []),
        )
        for _ in range(3)
    ]
    for again in outputs[1:]:
        assert all(torch.equal(value, first) for value, first in zip(again, outputs[0], strict=True))


def _encoder() -> nn.Module:
    torch.manual_seed(0)
    # A zero-initialized last batch norm per block would leave every convolution inside the blocks without gradient.
    return timm.create_model("resnet34", pretrained=False, features_only=True, zero_init_last=False).cuda()


def test_pin_conv_engines_keeps_the_state_and_skips_the_stem_and_odd_channels() -> None:
    model = nn.Sequential(_encoder(), nn.Conv2d(64, 64, 3, padding=1).cuda(), nn.Conv2d(64, 12, 1).cuda())
    keys = list(model.state_dict())
    pin_conv_engines(model)
    assert list(model.state_dict()) == keys
    if torch.backends.cudnn.version() != resnet._PINNED_CUDNN_VERSION:
        assert not any(type(module) is PinnedConv2d for module in model.modules())
        return
    encoder = cast(nn.Module, model[0])
    assert type(encoder.conv1) is nn.Conv2d
    assert type(model[1]) is PinnedConv2d
    assert type(model[2]) is nn.Conv2d
    assert sum(type(module) is PinnedConv2d for module in model.modules()) == 36


@pytest.mark.skipif(not _PINNED, reason="the engine table belongs to another cuDNN or GPU")
def test_pinned_encoder_trains_like_the_unpatched_one() -> None:
    reference = _encoder().train()
    patched = copy.deepcopy(reference)
    pin_conv_engines(patched)
    full = copy.deepcopy(reference)
    # The image of the latent TransFuser, so its layers hit the table.
    x = torch.randn(32, 3, 512, 352, device="cuda").contiguous(memory_format=torch.channels_last)
    gradients = []
    for model in (reference, patched):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model(x)
        torch.stack([feature.float().square().mean() for feature in features]).sum().backward()
        gradients.append([parameter.grad for parameter in model.parameters()])
    features = full(x)
    torch.stack([feature.square().mean() for feature in features]).sum().backward()
    fp32 = [parameter.grad for parameter in full.parameters()]
    # Through the whole encoder bf16 noise compounds; compare the gradients' distance from fp32 as a norm.
    for patched_grad, reference_grad, fp32_grad in zip(gradients[1], gradients[0], fp32, strict=True):
        assert patched_grad is not None
        assert reference_grad is not None
        assert fp32_grad is not None
        fp64 = fp32_grad.double()
        patched_error = (patched_grad.double() - fp64).norm() / fp64.norm()
        reference_error = (reference_grad.double() - fp64).norm() / fp64.norm()
        assert patched_error <= 1.5 * reference_error + 1e-3


@pytest.mark.skipif(not _PINNED, reason="the engine table belongs to another cuDNN or GPU")
def test_pinned_conv_with_bias_trains_like_conv2d() -> None:
    torch.manual_seed(0)
    conv = nn.Conv2d(64, 64, 3, padding=1).cuda()
    patched = copy.deepcopy(conv)
    patched.__class__ = PinnedConv2d
    # The 64 x 64 BEV grid of the decoder heads at batch 32, which the table runs on the halo kernels.
    x = torch.randn(32, 64, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
    grads = []
    for module in (conv, patched):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = module(x)
        out.float().square().mean().backward()
        assert module.weight.grad is not None
        assert module.bias is not None
        assert module.bias.grad is not None
        grads.append((out, module.weight.grad, module.bias.grad))
    for value, reference in zip(grads[1], grads[0], strict=True):
        torch.testing.assert_close(value.float(), reference.float(), rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(not _PINNED, reason="the engine table belongs to another cuDNN or GPU")
def test_frozen_pinned_conv_runs_the_table_in_eval() -> None:
    torch.manual_seed(0)
    conv = nn.Conv2d(64, 64, 3, padding=1, bias=False).cuda().eval().requires_grad_(False)
    patched = copy.deepcopy(conv)
    patched.__class__ = PinnedConv2d
    # Image layer1 at batch 32, on the halo kernel in the table.
    x = torch.randn(32, 64, 128, 88, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = patched(x)
        reference = conv(x)
    weight = conv.weight.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    halo = _ops.pinned_conv(x, weight, [1, 1], [1, 1], [HALO_ENGINE, -1, -1], [], [], [])
    assert torch.equal(out, halo)
    fp64 = F.conv2d(x.double(), weight.double(), padding=1)
    assert _error(out, fp64) <= 1.05 * _error(reference, fp64) + 1e-6


def test_pinned_conv_falls_back_in_eval_in_fp32_and_off_the_table() -> None:
    conv = nn.Conv2d(64, 64, 3, padding=1, bias=False).cuda()
    patched = copy.deepcopy(conv)
    patched.__class__ = PinnedConv2d
    x = torch.randn(2, 64, 20, 12, device="cuda")
    torch.testing.assert_close(patched(x), conv(x), rtol=0, atol=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(patched(x), conv(x), rtol=0, atol=0)
    patched.eval()
    conv.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(patched(x), conv(x), rtol=0, atol=0)


@pytest.mark.skipif(not _PINNED, reason="the engine table belongs to another cuDNN or GPU")
def test_pinned_conv_compiles() -> None:
    key = next(key for key in resnet._PINNED_ENGINES if key[0] == 2 and key[6] == 3 and key[8] == 1)
    _, n, c, h, w, k, _, _, _, _ = key
    conv = nn.Conv2d(c, k, 3, padding=1, bias=False).cuda()
    patched = copy.deepcopy(conv)
    patched.__class__ = PinnedConv2d
    compiled = cast(nn.Module, torch.compile(copy.deepcopy(patched)))
    x = torch.randn(n, c, h, w, device="cuda").contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager = patched(x)
        out = compiled(x)
    out.float().sum().backward()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    weight_grad = cast(nn.Conv2d, compiled).weight.grad
    assert weight_grad is not None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = conv(x)
    reference.float().sum().backward()
    assert conv.weight.grad is not None
    torch.testing.assert_close(weight_grad, conv.weight.grad, rtol=2e-2, atol=2e-2)
