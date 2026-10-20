# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
RegNet's grouped 3 x 3 convolution ops (24 channels per group) with their fake implementations and
autograd, and the patches that swap them into a timm RegNet.
"""

from __future__ import annotations

import functools
from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor, nn
from typing_extensions import override

from py123d_garage_cuda.transfuser.registry import note_fallback, supports, unsupported_reason

_ops = torch.ops.py123d_garage_cuda

GROUP = 24
BLOCK_CHANNELS = 72


def _output_size(size: int, stride: int) -> int:
    """
    Computes the output size of a 3 x 3, padding-1 convolution.

    Args:
        size: the input size.
        stride: the convolution stride.

    Returns:
        the output size.
    """
    return (size - 1) // stride + 1


def _output(x: Tensor, stride: int) -> Tensor:
    """
    Allocates the channels-last output of a 3 x 3, padding-1 convolution of x.

    Args:
        x: the [N, C, H, W] input.
        stride: the convolution stride.

    Returns:
        the uninitialized output.
    """
    batch, channels, height, width = x.shape
    size = (batch, channels, _output_size(height, stride), _output_size(width, stride))
    return x.new_empty(size).contiguous(memory_format=torch.channels_last)


def grouped_conv(
    x: jt.Float[Tensor, "batch channels height width"],
    weight: jt.Float[Tensor, "channels 24 3 3"],
    stride: int,
) -> jt.Float[Tensor, "batch channels out_height out_width"]:
    """
    Computes conv2d(x, weight, stride=stride, padding=1, groups=C // 24), accumulated in fp32.

    Args:
        x: the channels-last bf16 input, C a multiple of 72.
        weight: the weight, fp32 or bf16, any strides; rounded to bf16 like autocast does.
        stride: 1 or 2.

    Returns:
        the channels-last bf16 output.
    """
    return _ops.regnet_grouped_conv(x, weight, stride)


@torch.library.register_fake("py123d_garage_cuda::regnet_grouped_conv")
def _(x: Tensor, weight: Tensor, stride: int) -> Tensor:
    """Fake regnet_grouped_conv: the channels-last output."""
    return _output(x, stride)


@torch.library.register_fake("py123d_garage_cuda::regnet_grouped_conv_backward")
def _(grad: Tensor, x: Tensor, weight: Tensor, stride: int) -> tuple[Tensor, Tensor]:
    """Fake regnet_grouped_conv_backward: the gradients of x and weight."""
    return torch.empty_like(x, memory_format=torch.channels_last), torch.empty_like(weight)


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple[object, ...], output: Tensor) -> None:
    """Saves x, weight and the stride for the backward."""
    x, weight, stride = cast("tuple[Tensor, Tensor, int]", inputs)
    ctx.save_for_backward(x, weight)  # pyright: ignore[reportAttributeAccessIssue]
    ctx.stride = stride  # pyright: ignore[reportAttributeAccessIssue]


def _backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor | None, ...]:
    """Gradients of x and weight through regnet_grouped_conv_backward."""
    x, weight = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    stride = cast(int, ctx.stride)  # pyright: ignore[reportAttributeAccessIssue]
    grad_x, grad_weight = _ops.regnet_grouped_conv_backward(grad, x, weight, stride)
    return grad_x, grad_weight, None


torch.library.register_autograd("py123d_garage_cuda::regnet_grouped_conv", _backward, setup_context=_setup_context)


def bn_relu_grouped_conv(
    y: jt.Float[Tensor, "batch channels height width"],
    scale: jt.Float[Tensor, " channels"],
    shift: jt.Float[Tensor, " channels"],
    center: jt.Float[Tensor, " channels"],
    weight: jt.Float[Tensor, "channels 24 3 3"],
    stride: int,
) -> tuple[jt.Float[Tensor, "batch channels out_height out_width"], jt.Float[Tensor, "2 channels"]]:
    """
    Computes grouped_conv(relu(y * scale + shift), weight, stride) and the per-channel sums the next batch norm needs.

    Args:
        y: the channels-last bf16 batch norm input, C a multiple of 72.
        scale: the batch norm's contiguous fp32 scale, weight / sqrt(var + eps).
        shift: the fp32 shift, bias - mean * scale.
        center: an fp32 value close to y's channel means (the batch mean, no gradient); only the backward reads it, to
            sum scale's gradient around it.
        weight: the weight, fp32 or bf16, any strides.
        stride: 1 or 2.

    Returns:
        the channels-last bf16 output and the fp32 per-channel sums of the output and of its square.
    """
    return _ops.regnet_bn_relu_grouped_conv(y, scale, shift, center, weight, stride)


@torch.library.register_fake("py123d_garage_cuda::regnet_bn_relu_grouped_conv")
def _(y: Tensor, scale: Tensor, shift: Tensor, center: Tensor, weight: Tensor, stride: int) -> tuple[Tensor, Tensor]:
    """Fake regnet_bn_relu_grouped_conv: the channels-last output and the [2, C] sums."""
    return _output(y, stride), scale.new_empty(2, y.shape[1])


@torch.library.register_fake("py123d_garage_cuda::regnet_bn_relu_grouped_conv_backward")
def _(
    grad: Tensor,
    y: Tensor,
    scale: Tensor,
    shift: Tensor,
    center: Tensor,
    weight: Tensor,
    stride: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fake regnet_bn_relu_grouped_conv_backward: the gradients of y and weight and the [2, C] gradient sums."""
    return (
        torch.empty_like(y, memory_format=torch.channels_last),
        torch.empty_like(weight),
        scale.new_empty(2, y.shape[1]),
    )


def _fused_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[object, ...],
    output: tuple[Tensor, Tensor],
) -> None:
    """Saves the inputs, the output and the stride for the backward."""
    y, scale, shift, center, weight, stride = cast("tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]", inputs)
    ctx.save_for_backward(y, scale, shift, center, weight, output[0])  # pyright: ignore[reportAttributeAccessIssue]
    ctx.stride = stride  # pyright: ignore[reportAttributeAccessIssue]


def _fused_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad: Tensor | None,
    grad_sums: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Gradients of y, scale, shift and weight through regnet_bn_relu_grouped_conv_backward."""
    y, scale, shift, center, weight, out = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    stride = cast(int, ctx.stride)  # pyright: ignore[reportAttributeAccessIssue]
    # The sums' gradients join the output's: d sum(out) / d out = 1, d sum(out^2) / d out = 2 out.
    total = torch.zeros_like(out, dtype=torch.float32) if grad is None else grad.float()
    if grad_sums is not None:
        total = total + grad_sums[0][:, None, None] + 2 * out.float() * grad_sums[1][:, None, None]
    total = total.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    grad_y, grad_weight, sums = _ops.regnet_bn_relu_grouped_conv_backward(
        total,
        y,
        scale,
        shift,
        center,
        weight,
        stride,
    )
    # sums[1] is scale's gradient around center: sum g (y - center) = sum g y - center sum g.
    return grad_y, sums[1] + center * sums[0], sums[0], None, grad_weight, None


torch.library.register_autograd(
    "py123d_garage_cuda::regnet_bn_relu_grouped_conv",
    _fused_backward,
    setup_context=_fused_setup_context,
)


class CudaGroupedConv2d(nn.Conv2d):
    """A grouped 3 x 3, padding-1, bias-free Conv2d with 24 channels per group that trains through the CUDA kernels."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the convolution, on the CUDA kernels in training under bf16 autocast on a GPU that runs them, else as
        a plain Conv2d.

        Args:
            input: the [N, C, H, W] input.

        Returns:
            the convolution output.
        """
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else input.dtype
        if not (self.training and input.is_cuda and dtype == torch.bfloat16):
            note_fallback(
                "regnet_grouped_conv",
                lambda: f"needs bf16 on CUDA in training; got {dtype} on {input.device.type}",
            )
            return super().forward(input)
        if not supports("regnet_grouped_conv", input.device):
            note_fallback("regnet_grouped_conv", lambda: str(unsupported_reason("regnet_grouped_conv", input.device)))
            return super().forward(input)
        x = input.to(dtype).contiguous(memory_format=torch.channels_last)
        return grouped_conv(x, self.weight, self.stride[0])


def _runs_on_grouped_kernels(conv: nn.Module) -> bool:
    """
    Checks whether the grouped convolution kernels run a Conv2d: grouped 3 x 3 with padding 1, stride 1 or 2, no bias,
    equal input and output channels, 24 channels per group and a multiple of 72 channels.

    Args:
        conv: the module to check.

    Returns:
        whether the kernels run it.
    """
    return (
        isinstance(conv, nn.Conv2d)
        and conv.groups != 1
        and conv.bias is None
        and conv.in_channels == conv.out_channels
        and conv.in_channels // conv.groups == GROUP
        and conv.in_channels % BLOCK_CHANNELS == 0
        and conv.padding_mode == "zeros"
        and conv.kernel_size == (3, 3)
        and conv.padding == (1, 1)
        and conv.dilation == (1, 1)
        and conv.stride in ((1, 1), (2, 2))
    )


def patch_grouped_convs(module: nn.Module) -> None:
    """
    Turns every plain Conv2d inside module that the grouped convolution kernels run into a CudaGroupedConv2d, in place;
    parameters and state_dict keys stay the same. Eval and inference keep cuDNN: the kernels run in training under bf16
    autocast.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        if type(child) is nn.Conv2d and _runs_on_grouped_kernels(child):
            child.__class__ = CudaGroupedConv2d


def _update_running_stats(bn: nn.BatchNorm2d, mean: Tensor, var: Tensor, count: int) -> None:
    """
    Applies nn.BatchNorm2d's running statistics update in training.

    Args:
        bn: the batch norm whose running statistics to update.
        mean: the batch mean.
        var: the biased batch variance.
        count: the number of values per channel.
    """
    if not bn.track_running_stats or bn.running_mean is None or bn.running_var is None:
        return
    with torch.no_grad():
        assert bn.num_batches_tracked is not None
        bn.num_batches_tracked.add_(1)
        factor = 1.0 / bn.num_batches_tracked.float() if bn.momentum is None else bn.momentum
        # Not lerp_: inductor's Triton 3.4 fails to compile the fused reduction with it.
        bn.running_mean.copy_(bn.running_mean * (1 - factor) + mean.detach() * factor)
        bn.running_var.copy_(bn.running_var * (1 - factor) + var.detach() * (factor * count / max(count - 1, 1)))


@functools.cache
def fused_bottleneck_class() -> type[nn.Module]:
    """
    Builds FusedBottleneck on first use, so importing this module needs no timm.

    Returns:
        the FusedBottleneck class, a subclass of timm's RegNet Bottleneck.
    """
    from timm.models.regnet import Bottleneck

    class FusedBottleneck(Bottleneck):
        """
        A timm RegNet Bottleneck whose conv1 batch norm + ReLU, conv2 and conv2 batch norm statistics train as one op
        (bn_relu_grouped_conv): relu(bn1(conv1(x))) is never written, and conv2's output is not read again for bn2's
        statistics. Batch statistics, running statistics and the rest of the block stay in torch, for inductor.
        """

        @override
        def forward(self, x: Tensor) -> Tensor:
            """
            Applies the block, fused in training under bf16 autocast on a GPU that runs the kernels, else unfused.

            Args:
                x: the [N, C, H, W] block input.

            Returns:
                the block output.
            """
            autocast = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
            if not (self.training and x.is_cuda and autocast):
                note_fallback("regnet_bn_relu_grouped_conv", "needs bf16 autocast on CUDA in training")
                return super().forward(x)
            if not supports("regnet_bn_relu_grouped_conv", x.device):
                note_fallback(
                    "regnet_bn_relu_grouped_conv",
                    lambda: str(unsupported_reason("regnet_bn_relu_grouped_conv", x.device)),
                )
                return super().forward(x)
            shortcut = x
            bn1 = cast(nn.BatchNorm2d, self.conv1.bn)
            conv2 = cast(nn.Conv2d, self.conv2.conv)
            bn2 = cast(nn.BatchNorm2d, self.conv2.bn)

            y1 = self.conv1.conv(x).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
            # Sums, not var_mean: inductor's Triton 3.4 fails to compile var_mean's fused Welford combine with the scale
            # and shift below.
            count1 = y1.numel() // y1.shape[1]
            y1_float = y1.float()
            mean1 = y1_float.sum((0, 2, 3)) / count1
            var1 = (y1_float.square().sum((0, 2, 3)) / count1 - mean1.square()).clamp_min(0.0)
            _update_running_stats(bn1, mean1, var1, count1)
            scale1 = bn1.weight * torch.rsqrt(var1 + bn1.eps)
            shift1 = bn1.bias - mean1 * scale1
            y2, sums = bn_relu_grouped_conv(y1, scale1, shift1, mean1.detach(), conv2.weight, conv2.stride[0])

            count2 = y2.numel() // y2.shape[1]
            mean2 = sums[0] / count2
            var2 = (sums[1] / count2 - mean2.square()).clamp_min(0.0)
            _update_running_stats(bn2, mean2, var2, count2)
            scale2 = bn2.weight * torch.rsqrt(var2 + bn2.eps)
            shift2 = bn2.bias - mean2 * scale2
            x = torch.relu(y2.float() * scale2[:, None, None] + shift2[:, None, None]).to(torch.bfloat16)

            x = self.se(x)
            x = self.conv3(x)
            if self.downsample is not None:
                x = self.drop_path(x) + self.downsample(shortcut)
            return self.act3(x)

    return FusedBottleneck


def _is_bn_relu(module: nn.Module) -> bool:
    """
    Checks whether module is an affine timm BatchNormAct2d with a ReLU and no drop block.

    Args:
        module: the module to check.

    Returns:
        whether the module qualifies.
    """
    from timm.layers.norm_act import BatchNormAct2d

    return (
        type(module) is BatchNormAct2d
        and bool(module.affine)
        and isinstance(getattr(module, "drop", nn.Identity()), nn.Identity)
        and isinstance(getattr(module, "act", None), nn.ReLU)
    )


def patch_regnet_blocks(module: nn.Module) -> None:
    """
    Turns every timm RegNet Bottleneck whose conv2 the grouped convolution kernels run and whose conv1 and conv2 end in
    an affine batch norm + ReLU (no drop block, no anti-aliasing) into a FusedBottleneck, in place; parameters and
    state_dict keys stay the same. Eval and inference run the unfused block; patch_grouped_convs puts that block's
    grouped convolution on the kernels too.

    Args:
        module: the module to patch.
    """
    from timm.models.regnet import Bottleneck

    fused = fused_bottleneck_class()
    for child in module.modules():
        if type(child) is not Bottleneck or not _runs_on_grouped_kernels(child.conv2.conv):
            continue
        if not (_is_bn_relu(child.conv1.bn) and _is_bn_relu(child.conv2.bn)):
            continue
        if getattr(child.conv1, "aa", None) is not None or getattr(child.conv2, "aa", None) is not None:
            continue
        child.__class__ = fused  # pyright: ignore[reportAttributeAccessIssue]
