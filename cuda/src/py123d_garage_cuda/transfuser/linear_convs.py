# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Model-agnostic patches, as kernel_patches entries: patchify stems (kernel == stride, no padding) and 1 x 1 convolutions
run as one matrix product over the image's non-overlapping patches.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override


class PatchifyConv2d(nn.Conv2d):
    """A Conv2d with kernel_size == stride, no padding and dilation 1, computed as a product over patches."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the convolution as a product over patches, or as a plain Conv2d when the size does not divide.

        Args:
            input: the [N, C, H, W] image.

        Returns:
            the [N, out_channels, H / kh, W / kw] output, channels-last.
        """
        batch, channels, height, width = input.shape
        kh, kw = self.kernel_size
        if height % kh or width % kw:
            return super().forward(input)
        rows, cols = height // kh, width // kw
        # [B, C, rows, kh, cols, kw] -> [B, rows, cols, C, kh, kw]: the weight's (C, kh, kw) order per patch.
        patches = input.view(batch, channels, rows, kh, cols, kw).permute(0, 2, 4, 1, 3, 5)
        out = F.linear(patches.reshape(batch, rows, cols, channels * kh * kw), self.weight.flatten(1), self.bias)
        # [B, rows, cols, out] is the channels-last layout of [B, out, rows, cols].
        return out.permute(0, 3, 1, 2)


def patch_patchify_stems(module: nn.Module, max_in_channels: int = 4) -> None:
    """
    Turns every patchify Conv2d inside module into a PatchifyConv2d, in place; parameters and state_dict keys stay.

    A Conv2d qualifies with kernel equal to stride (not 1 x 1), no padding, dilation 1, one group and zero padding mode.

    Args:
        module: the module to patch.
        max_in_channels: largest input channel count to patch; the default takes image stems only.
    """
    for child in module.modules():
        if type(child) is not nn.Conv2d or child.groups != 1 or child.padding_mode != "zeros":
            continue
        if child.in_channels > max_in_channels:
            continue
        if child.kernel_size != child.stride or child.padding != (0, 0) or child.dilation != (1, 1):
            continue
        if child.kernel_size == (1, 1):
            continue
        child.__class__ = PatchifyConv2d


def patch_pointwise_convs(module: nn.Module) -> None:
    """
    Turns every 1 x 1, stride-1 Conv2d inside module without padding into a PatchifyConv2d, in place, so it trains as
    a linear layer over the channels whose weight gradient compiles as a matrix product; parameters and state_dict
    keys stay.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        if type(child) is not nn.Conv2d or child.groups != 1 or child.padding_mode != "zeros":
            continue
        if child.kernel_size == (1, 1) and child.stride == (1, 1) and child.padding == (0, 0):
            child.__class__ = PatchifyConv2d
