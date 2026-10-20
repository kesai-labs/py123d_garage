# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The patchify stem and the pointwise convolutions give conv2d's outputs and gradients, and patch only their convolutions.
"""

from __future__ import annotations

import copy

import pytest
import timm
import torch
from torch import nn

pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.linear_convs import (
    PatchifyConv2d,
    patch_patchify_stems,
    patch_pointwise_convs,
)


@pytest.mark.parametrize("bias", [True, False])
def test_matches_conv2d(bias: bool) -> None:
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, kernel_size=4, stride=4, bias=bias).double()
    patched = copy.deepcopy(conv)
    patch_patchify_stems(patched)
    assert type(patched) is PatchifyConv2d
    x = torch.randn(2, 3, 24, 32, dtype=torch.float64).contiguous(memory_format=torch.channels_last)
    out, reference = patched(x), conv(x)
    torch.testing.assert_close(out, reference, rtol=1e-12, atol=1e-12)
    grad = torch.randn_like(reference)
    out.backward(grad)
    reference.backward(grad)
    for parameter, reference_parameter in zip(patched.parameters(), conv.parameters(), strict=True):
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-12, atol=1e-12)


def test_patches_only_the_image_stem_of_convnext() -> None:
    encoder = timm.create_model("convnext_tiny", pretrained=False, features_only=True)
    keys = list(encoder.state_dict())
    patch_patchify_stems(encoder)
    assert list(encoder.state_dict()) == keys
    patched = [m for m in encoder.modules() if type(m) is PatchifyConv2d]
    assert len(patched) == 1
    assert patched[0].in_channels == 3


def test_uneven_input_falls_back_to_conv2d() -> None:
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, kernel_size=4, stride=4).double()
    patched = copy.deepcopy(conv)
    patch_patchify_stems(patched)
    x = torch.randn(1, 3, 10, 13, dtype=torch.float64)
    torch.testing.assert_close(patched(x), conv(x))


@pytest.mark.parametrize("bias", [True, False])
def test_pointwise_conv_matches_conv2d(bias: bool) -> None:
    torch.manual_seed(0)
    conv = nn.Conv2d(64, 12, kernel_size=1, bias=bias).double()
    patched = copy.deepcopy(conv)
    patch_pointwise_convs(patched)
    assert type(patched) is PatchifyConv2d
    x = torch.randn(2, 64, 9, 7, dtype=torch.float64).contiguous(memory_format=torch.channels_last)
    out, reference = patched(x), conv(x)
    torch.testing.assert_close(out, reference, rtol=1e-12, atol=1e-12)
    grad = torch.randn_like(reference)
    out.backward(grad)
    reference.backward(grad)
    for parameter, reference_parameter in zip(patched.parameters(), conv.parameters(), strict=True):
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-12, atol=1e-12)


def test_patches_only_stride_one_pointwise_convs() -> None:
    model = nn.Sequential(
        nn.Conv2d(64, 64, 1),
        nn.Conv2d(64, 64, 1, stride=2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.Conv2d(64, 64, 1, groups=2),
    )
    keys = list(model.state_dict())
    patch_pointwise_convs(model)
    assert list(model.state_dict()) == keys
    assert [type(module) for module in model] == [PatchifyConv2d, nn.Conv2d, nn.Conv2d, nn.Conv2d]
