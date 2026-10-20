# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
ConvNeXt's depthwise convolution ops with their fake implementations and autograd, and the patch that
swaps them into a timm ConvNeXt: py123d_garage_cuda.transfuser.convnext:patch_depthwise_convs.
"""

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor, nn
from typing_extensions import override

from py123d_garage_cuda.transfuser.registry import note_fallback, supports, unsupported_reason

_ops = torch.ops.py123d_garage_cuda


def depthwise_conv(
    x: jt.Float[Tensor, "batch channels height width"],
    weight: jt.Float[Tensor, "channels 1 kernel kernel"],
    bias: jt.Float[Tensor, " channels"] | None,
) -> jt.Float[Tensor, "batch channels height width"]:
    """
    Computes conv2d(x, weight, bias, padding=k // 2, groups=C) for a stride-1 depthwise kernel, accumulated in fp32.

    Args:
        x: the channels-last input.
        weight: the weight in x's dtype, k odd.
        bias: the bias in x's dtype, or None.

    Returns:
        the channels-last output in x's dtype.
    """
    return _ops.depthwise_conv(x, weight, bias)


@torch.library.register_fake("py123d_garage_cuda::depthwise_conv")
def _(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Fake depthwise_conv: the channels-last output."""
    return torch.empty_like(x, memory_format=torch.channels_last)


@torch.library.register_fake("py123d_garage_cuda::depthwise_conv_backward")
def _(grad: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fake depthwise_conv_backward: the gradients of x, weight and bias."""
    grad_x = torch.empty_like(x, memory_format=torch.channels_last)
    return grad_x, torch.empty_like(weight), weight.new_empty(weight.shape[0])


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple[object, ...], output: Tensor) -> None:
    """Saves x and weight, and whether there is a bias, for the backward."""
    x, weight, bias = cast("tuple[Tensor, Tensor, Tensor | None]", inputs)
    ctx.save_for_backward(x, weight)  # pyright: ignore[reportAttributeAccessIssue]
    ctx.has_bias = bias is not None  # pyright: ignore[reportAttributeAccessIssue]


def _backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor | None, ...]:
    """Gradients of x, weight and bias through depthwise_conv_backward."""
    x, weight = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    grad_x, grad_weight, grad_bias = _ops.depthwise_conv_backward(grad, x, weight)
    return grad_x, grad_weight, grad_bias if ctx.has_bias else None  # pyright: ignore[reportAttributeAccessIssue]


torch.library.register_autograd("py123d_garage_cuda::depthwise_conv", _backward, setup_context=_setup_context)


class CudaDepthwiseConv2d(nn.Conv2d):
    """A stride-1 depthwise k x k Conv2d with padding k // 2 that trains through the CUDA kernel under bf16 autocast."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the convolution, on the CUDA kernel in training under bf16 autocast on a GPU that runs it, else as a
        plain Conv2d.

        Args:
            input: the [N, C, H, W] input.

        Returns:
            the [N, C, H, W] output.
        """
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else input.dtype
        if not (self.training and input.is_cuda and dtype == torch.bfloat16):
            note_fallback(
                "depthwise_conv",
                lambda: f"needs bf16 on CUDA in training; got {dtype} on {input.device.type}",
            )
            return super().forward(input)
        if not supports("depthwise_conv", input.device):
            note_fallback("depthwise_conv", lambda: str(unsupported_reason("depthwise_conv", input.device)))
            return super().forward(input)
        x = input.to(dtype).contiguous(memory_format=torch.channels_last)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return depthwise_conv(x, self.weight.to(dtype), bias)


def patch_depthwise_convs(module: nn.Module) -> None:
    """
    Turns every depthwise Conv2d inside module with an odd square kernel of 3, 5 or 7, stride 1, padding k // 2 and
    zero padding into a CudaDepthwiseConv2d, in place; parameters and state_dict keys stay the same.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        if type(child) is not nn.Conv2d or child.groups != child.in_channels or child.in_channels != child.out_channels:
            continue
        kernel = child.kernel_size[0]
        if child.kernel_size != (kernel, kernel) or kernel not in (3, 5, 7):
            continue
        if child.stride != (1, 1) or child.padding != (kernel // 2, kernel // 2) or child.dilation != (1, 1):
            continue
        if child.padding_mode != "zeros":
            continue
        child.__class__ = CudaDepthwiseConv2d
