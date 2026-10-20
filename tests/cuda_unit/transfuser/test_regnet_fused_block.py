# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The fused batch norm + ReLU + grouped convolution of RegNet matches a float64 reference in values and gradients, and a
patched RegNetY trains like the unfused CUDA kernels: outputs, gradients and running statistics.
"""

from __future__ import annotations

import copy
from typing import cast

import pytest
import timm
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
py123d_garage_cuda = pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason  # noqa: E402

if (_unsupported := unsupported_reason("regnet_bn_relu_grouped_conv")) is not None:
    pytest.skip(_unsupported, allow_module_level=True)

from py123d_garage_cuda.transfuser.regnet import (  # noqa: E402
    fused_bottleneck_class,
    patch_grouped_convs,
    patch_regnet_blocks,
)

FusedBottleneck = fused_bottleneck_class()

# (N, C, H, W): one and two 72-channel chunks, odd and even sizes, tiles cut by the image border.
SHAPES = [(2, 72, 19, 37), (3, 144, 16, 8), (1, 216, 5, 3), (2, 72, 34, 21)]


def _inputs(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    channels = shape[1]
    y = (torch.randn(*shape, device="cuda") + 0.5).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    scale = torch.rand(channels, device="cuda") + 0.5
    shift = torch.randn(channels, device="cuda") * 0.5
    center = y.float().mean((0, 2, 3))
    weight = torch.randn(channels, 24, 3, 3, device="cuda") / 8
    return y, scale, shift, center, weight


def _relative_error(actual: torch.Tensor | None, expected: torch.Tensor | None) -> float:
    assert actual is not None
    assert expected is not None
    return ((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("stride", [1, 2])
def test_matches_float64_reference(shape: tuple[int, int, int, int], stride: int) -> None:
    y, scale, shift, center, weight = _inputs(shape)
    inputs = [y, scale, shift, weight]
    for tensor in inputs:
        tensor.requires_grad_()
    references = [tensor.detach().double().requires_grad_() for tensor in inputs]
    reference_y, reference_scale, reference_shift, reference_weight = references

    out, sums = py123d_garage_cuda.bn_relu_grouped_conv(y, scale, shift, center, weight, stride)
    activation = torch.relu(reference_y * reference_scale[:, None, None] + reference_shift[:, None, None])
    reference_out = F.conv2d(
        activation,
        reference_weight.bfloat16().double(),
        stride=stride,
        padding=1,
        groups=shape[1] // 24,
    )
    assert out.dtype == torch.bfloat16
    assert out.is_contiguous(memory_format=torch.channels_last)
    # The activation is rounded to bf16 before the convolution, the output once more.
    torch.testing.assert_close(out.double(), reference_out, atol=2e-2, rtol=2**-7)
    # The sums are the output's own, in fp32.
    own = torch.stack([out.double().sum((0, 2, 3)), out.double().square().sum((0, 2, 3))])
    torch.testing.assert_close(sums.double(), own, atol=1e-3, rtol=1e-5)

    grad = torch.randn_like(out)
    grad_sums = torch.randn_like(sums) / out[0, 0].numel()
    (out.float() * grad.float()).sum().add((sums * grad_sums).sum()).backward()
    reference_sums = torch.stack([reference_out.sum((0, 2, 3)), reference_out.square().sum((0, 2, 3))])
    (reference_out * grad.double()).sum().add((reference_sums * grad_sums.double()).sum()).backward()
    for tensor, reference in zip(inputs, references, strict=True):
        assert tensor.grad is not None
        assert reference.grad is not None
        assert _relative_error(tensor.grad, reference.grad) < 1e-2
    assert y.grad is not None
    assert y.grad.dtype == torch.bfloat16
    assert y.grad.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("stride", [1, 2])
def test_backward_is_deterministic(stride: int) -> None:
    y, scale, shift, center, weight = _inputs((2, 144, 33, 29))
    out, _ = py123d_garage_cuda.bn_relu_grouped_conv(y, scale, shift, center, weight, stride)
    grad = torch.randn_like(out)
    op = torch.ops.py123d_garage_cuda.regnet_bn_relu_grouped_conv_backward
    first = op(grad, y, scale, shift, center, weight, stride)
    second = op(grad, y, scale, shift, center, weight, stride)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
    again, sums = py123d_garage_cuda.bn_relu_grouped_conv(y, scale, shift, center, weight, stride)
    assert torch.equal(out, again)
    assert torch.equal(
        sums,
        py123d_garage_cuda.bn_relu_grouped_conv(y, scale, shift, center, weight, stride)[1],
    )


@pytest.mark.parametrize("stride", [1, 2])
def test_opcheck(stride: int) -> None:
    y, scale, shift, center, weight = _inputs((2, 72, 12, 10))
    for tensor in (y, scale, shift, weight):
        tensor.requires_grad_()
    torch.library.opcheck(
        torch.ops.py123d_garage_cuda.regnet_bn_relu_grouped_conv.default,
        (y, scale, shift, center, weight, stride),
    )


def _regnet(pretrained: bool = True) -> torch.nn.Module:
    """
    RegNetY-032 on its ImageNet weights when cached (a trained network: at random init the deep bf16 network is
    chaotic, bf16 and fp32 disagree by tens of percent at stage 3 already), else random weights and affines.
    """
    torch.manual_seed(0)
    try:
        model = timm.create_model("regnety_032.ra_in1k", pretrained=pretrained, features_only=True)
    except Exception:
        pytest.skip("needs the cached regnety_032.ra_in1k weights")
    if not pretrained:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.uniform_(module.weight, 0.5, 1.0)
                torch.nn.init.normal_(module.bias, 0.0, 0.1)
    return model.cuda().to(memory_format=torch.channels_last).train()  # pyright: ignore[reportCallIssue]


def _train_step(model: torch.nn.Module, x: torch.Tensor, autocast: bool = True) -> list[torch.Tensor]:
    """One training step's features; without autocast in float64 (fp32 would run cuDNN's convolutions in TF32)."""
    if not autocast:
        model.double()
        x = x.double()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        features = model(x)
    # A fixed random upstream gradient per feature map, varying over the pixels like a real loss's: a spatially constant
    # one is almost all batch-mean component, which the batch norms' backward cancels down to rounding noise.
    generator = torch.Generator(device="cuda").manual_seed(2)
    grads = [torch.randn(feature.shape, device="cuda", generator=generator) for feature in features]
    torch.autograd.backward([feature.double() for feature in features], [grad.double() for grad in grads])
    return [feature.detach().float() for feature in features]


@pytest.mark.parametrize("stage", ["s1", "s2", "s4"])
@pytest.mark.parametrize("index", [0, 1])
def test_block_matches_float64(stage: str, index: int) -> None:
    """A single block (the first: stride 2 with the downsample shortcut) against float64, like the unfused kernels."""
    blocks = list(getattr(_regnet(pretrained=False), stage).children())
    if index >= len(blocks):
        pytest.skip(f"{stage} has one block")
    reference = blocks[index]
    unfused = copy.deepcopy(reference)
    patch_grouped_convs(unfused)
    fused = copy.deepcopy(reference)
    patch_regnet_blocks(fused)
    assert isinstance(fused, FusedBottleneck)
    in_channels = reference.conv1.conv.in_channels
    torch.manual_seed(1)
    x = torch.randn(4, in_channels, 26, 19, device="cuda").relu().contiguous(memory_format=torch.channels_last)
    grad = None

    def step(block: torch.nn.Module, autocast: bool) -> torch.Tensor:
        nonlocal grad
        inputs = (x if autocast else x.double()).detach().requires_grad_()
        if not autocast:
            block.double()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = block(inputs)
        if grad is None:
            grad = torch.randn(out.shape, device="cuda", dtype=torch.float64)
        out.double().backward(grad)
        assert inputs.grad is not None
        return torch.cat([out.detach().double().flatten(), inputs.grad.double().flatten()])

    expected = step(reference, autocast=False)
    unfused_result = step(unfused, autocast=True)
    fused_result = step(fused, autocast=True)
    assert _relative_error(fused_result, expected) < 1.2 * _relative_error(unfused_result, expected)
    parameters = dict(reference.named_parameters())
    unfused_parameters = dict(unfused.named_parameters())
    for name, parameter in fused.named_parameters():
        assert parameter.grad is not None, name
        f_error = _relative_error(parameter.grad, parameters[name].grad)
        u_error = _relative_error(unfused_parameters[name].grad, parameters[name].grad)
        assert f_error < 1.5 * u_error + 2e-3, (name, f_error, u_error)
    buffers = dict(reference.named_buffers())
    for name, buffer in fused.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert torch.equal(buffer, buffers[name].to(buffer.device))
        else:
            assert _relative_error(buffer, buffers[name]) < 1e-2, name


def test_patched_regnet_trains_like_unfused() -> None:
    """
    The whole network against float64: on synthetic inputs, a training-mode RegNetY amplifies rounding (unfused bf16 is
    tens of percent off at stage 4), so fused and unfused are compared statistically, tensor by tensor.
    """
    reference = _regnet()
    unfused = copy.deepcopy(reference)
    patch_grouped_convs(unfused)
    fused = copy.deepcopy(reference)
    patch_regnet_blocks(fused)
    assert sum(isinstance(module, FusedBottleneck) for module in fused.modules()) == 21
    torch.manual_seed(1)
    x = torch.randn(8, 3, 160, 224, device="cuda").contiguous(memory_format=torch.channels_last)

    expected = _train_step(copy.deepcopy(reference), x, autocast=False)
    unfused_features = _train_step(unfused, x)
    fused_features = _train_step(fused, x)
    for e, u, f in zip(expected, unfused_features, fused_features, strict=True):
        assert _relative_error(f, e) < 1.1 * _relative_error(u, e) + 1e-3

    reference.double()
    _train_step(reference, x, autocast=False)
    parameters = dict(reference.named_parameters())
    unfused_parameters = dict(unfused.named_parameters())
    ratios = []
    for name, parameter in fused.named_parameters():
        assert parameter.grad is not None, name
        u_error = _relative_error(unfused_parameters[name].grad, parameters[name].grad)
        ratios.append(_relative_error(parameter.grad, parameters[name].grad) / max(u_error, 1e-9))
    ratios.sort()
    assert ratios[len(ratios) // 2] < 1.1
    assert ratios[int(0.9 * len(ratios))] < 2.0

    unfused_buffers = dict(unfused.named_buffers())
    for name, buffer in fused.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert torch.equal(buffer, unfused_buffers[name])
        else:
            assert _relative_error(buffer, unfused_buffers[name]) < 2e-2, name

    # Eval runs the unfused modules on the running statistics.
    fused.eval()
    unfused.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for f, u in zip(fused(x), unfused(x), strict=True):
            assert _relative_error(f.float(), u.float()) < 5e-2


def test_compiled_matches_eager() -> None:
    """Compiled like the trainings, as close to eager as compiling the unfused kernels is."""
    torch.manual_seed(1)
    x = torch.randn(4, 3, 96, 128, device="cuda").contiguous(memory_format=torch.channels_last)
    errors = {}
    for name, patch in (
        ("fused", patch_regnet_blocks),
        ("unfused", patch_grouped_convs),
    ):
        eager = _regnet()
        patch(eager)
        compiled_model = copy.deepcopy(eager)
        eager_features = _train_step(eager, x)
        compiled_features = _train_step(cast(torch.nn.Module, torch.compile(compiled_model, dynamic=False)), x)
        compiled_parameters = dict(compiled_model.named_parameters())
        grad_errors = sorted(
            _relative_error(compiled_parameters[key].grad, parameter.grad)
            for key, parameter in eager.named_parameters()
        )
        feature_error = max(_relative_error(c, e) for e, c in zip(eager_features, compiled_features, strict=True))
        errors[name] = (feature_error, grad_errors[len(grad_errors) // 2])
    assert errors["fused"][0] < 2 * errors["unfused"][0] + 1e-2, errors
    assert errors["fused"][1] < 2 * errors["unfused"][1] + 1e-2, errors
