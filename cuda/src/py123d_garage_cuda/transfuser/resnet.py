# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
ResNet's ops with their fake implementations, autograd and patch: the stem, max_pool(relu(bn1(conv1(x)))) as a
convolution with batch statistics and, through the max-pool of stem_pool, a fused batch norm -> ReLU -> max-pool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from py123d_garage_cuda.transfuser.registry import note_fallback, supports, unsupported_reason
from py123d_garage_cuda.transfuser.stem_pool import resnet_stem_pool

_ops = torch.ops.py123d_garage_cuda

# The convolution's output tile edge and persistent blocks per SM; must match csrc/sm90/resnet_stem_conv.cu.
_STEM_TILE = 16
_STEM_BLOCKS_PER_SM = 2


def _stem_blocks(x: Tensor) -> int:
    """
    Computes the number of persistent blocks resnet_stem_conv launches, one statistics piece each.

    Args:
        x: the [N, 4, H, W] stem input.

    Returns:
        the number of blocks.
    """
    out_height, out_width = (x.shape[2] - 1) // 2 + 1, (x.shape[3] - 1) // 2 + 1
    tiles = x.shape[0] * -(-out_height // _STEM_TILE) * -(-out_width // _STEM_TILE)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    return max(1, min(sms * _STEM_BLOCKS_PER_SM, tiles))


def resnet_stem_conv(
    x: jt.Float[Tensor, "batch 4 height width"],
    weight: jt.Float[Tensor, "64 in_channels 7 7"],
) -> tuple[jt.Float[Tensor, "batch 64 out_height out_width"], jt.Float[Tensor, "blocks 3 64"]]:
    """
    Computes conv2d(x, weight, stride=2, padding=3) for ResNet's 7 x 7 stem with 64 outputs, accumulated in fp32, plus
    the batch statistics of its bf16 output in pieces.

    Args:
        x: the channels-last bf16 input, its channels past the weight's zero.
        weight: the bf16 weight, in_channels 2 or 3.

    Returns:
        the channels-last bf16 output, and per kernel block the count, mean and centred sum of squares of its share of
        the output, for batch_norm_finalize. Only the output has a gradient.
    """
    return _ops.resnet_stem_conv(x, weight)


def batch_norm_finalize(
    partial: jt.Float[Tensor, "blocks 3 channels"],
    weight: jt.Float[Tensor, " channels"],
    bias: jt.Float[Tensor, " channels"],
    eps: float,
) -> jt.Float[Tensor, "5 channels"]:
    """
    Merges the batch statistics from per-block pieces, in a fixed order.

    Args:
        partial: fp32 count, mean and centred sum of squares per block and channel.
        weight: the batch norm's weight.
        bias: the batch norm's bias.
        eps: added to the variance before the square root.

    Returns:
        fp32 mean, biased variance, rstd, scale = weight * rstd and shift = bias - mean * scale.
    """
    return _ops.batch_norm_finalize(partial, weight, bias, eps)


@torch.library.register_fake("py123d_garage_cuda::resnet_stem_conv")
def _(x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """Fake resnet_stem_conv: the channels-last output and the per-block statistics."""
    batch, _, height, width = x.shape
    y = x.new_empty((batch, 64, (height - 1) // 2 + 1, (width - 1) // 2 + 1)).contiguous(
        memory_format=torch.channels_last,
    )
    return y, x.new_empty((_stem_blocks(x), 3, 64), dtype=torch.float32)


@torch.library.register_fake("py123d_garage_cuda::resnet_stem_conv_weight_grad")
def _(grad: Tensor, x: Tensor, channels: int) -> Tensor:
    """Fake resnet_stem_conv_weight_grad: the weight gradient."""
    return x.new_empty((64, channels, 7, 7))


@torch.library.register_fake("py123d_garage_cuda::batch_norm_finalize")
def _(partial: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """Fake batch_norm_finalize: the [5, C] statistics."""
    return partial.new_empty((5, partial.shape[2]))


def _conv_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[object, ...],
    output: tuple[Tensor, Tensor],
) -> None:
    """Saves x and the input channel count for the backward; the statistics get no gradient."""
    x, weight = cast("tuple[Tensor, Tensor]", inputs)
    ctx.save_for_backward(x)  # pyright: ignore[reportAttributeAccessIssue]
    ctx.channels = weight.shape[1]  # pyright: ignore[reportAttributeAccessIssue]
    ctx.mark_non_differentiable(output[1])


def _conv_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad: Tensor,
    _: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Gradient of the weight through resnet_stem_conv_weight_grad; the input gets none."""
    (x,) = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    return None, _ops.resnet_stem_conv_weight_grad(grad, x, ctx.channels)  # pyright: ignore[reportAttributeAccessIssue]


torch.library.register_autograd(
    "py123d_garage_cuda::resnet_stem_conv",
    _conv_backward,
    setup_context=_conv_setup_context,
)


def _runs_fused(conv: nn.Conv2d, x: Tensor) -> bool:
    """
    Checks whether the stem runs on the kernels: bf16 on a GPU that runs them, in training or (a frozen stem) without
    gradients.

    Args:
        conv: the stem convolution.
        x: the stem input.

    Returns:
        whether the kernels run.
    """
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if not (x.is_cuda and dtype == torch.bfloat16):
        _note_stem_fallback(lambda: f"needs bf16 on CUDA, got {dtype} on {x.device.type}")
        return False
    if not (supports("resnet_stem_conv", x.device) and supports("resnet_stem_pool", x.device)):
        _note_stem_fallback(
            lambda: "; ".join(
                str(unsupported_reason(op, x.device))
                for op in ("resnet_stem_conv", "resnet_stem_pool")
                if not supports(op, x.device)
            ),
        )
        return False
    if conv.training or not (torch.is_grad_enabled() and (conv.weight.requires_grad or x.requires_grad)):
        return True
    _note_stem_fallback(lambda: "eval with gradients; the kernels run in training or in a frozen stem")
    return False


def _note_stem_fallback(reason: Callable[[], str]) -> None:
    """
    Records a stem call that runs torch for both stem kernel families.

    Args:
        reason: returns why.
    """
    note_fallback("resnet_stem_conv", reason)
    note_fallback("resnet_stem_pool", reason)


class CudaStemConv(nn.Conv2d):
    """conv1 of a fused stem: where the kernels run, it hands its input on to bn1, which runs the convolution."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """
        Returns the input unchanged where the kernels run, else applies the convolution.

        Args:
            input: the [N, C, H, W] image.

        Returns:
            the input or the convolution output.
        """
        if _runs_fused(self, input):
            return input
        return super().forward(input)


class CudaStemNorm(nn.BatchNorm2d):
    """
    bn1 of a fused stem, which where the kernels run also runs conv1 before it and the ReLU and max-pool after it: with
    batch statistics in training, with the running statistics (a frozen backbone) in eval.
    """

    _stem_conv: tuple[nn.Conv2d]

    @override
    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the fused stem to conv1's input, or batch norm, ReLU and max-pool to conv1's output.

        Args:
            input: conv1's input where the kernels run, else conv1's output.

        Returns:
            the pooled stem output.
        """
        # conv1 hands its input on unchanged exactly when the input still has conv1's channels.
        conv = self._stem_conv[0]
        if input.shape[1] != conv.in_channels:
            return F.max_pool2d(F.relu(super().forward(input)), 3, 2, 1)
        # The kernel reads pixels of 4 channels.
        x = F.pad(input.to(torch.bfloat16), (0, 0, 0, 0, 0, 4 - input.shape[1])).contiguous(
            memory_format=torch.channels_last,
        )
        y, partial = resnet_stem_conv(x, conv.weight.to(torch.bfloat16))
        assert self.running_mean is not None
        assert self.running_var is not None
        if not self.training:
            rstd = torch.rsqrt(self.running_var + self.eps)
            scale = self.weight * rstd
            values = torch.stack(
                [self.running_mean, self.running_var, rstd, scale, self.bias - self.running_mean * scale],
            )
            return resnet_stem_pool(y, values.float().contiguous(), self.weight, self.bias)[0]
        values = batch_norm_finalize(partial, self.weight.detach(), self.bias.detach(), self.eps)
        out, _ = resnet_stem_pool(y, values, self.weight, self.bias)
        assert self.num_batches_tracked is not None
        assert self.momentum is not None
        rows = y.numel() // y.shape[1]
        with torch.no_grad():
            self.running_mean.lerp_(values[0], self.momentum)
            self.running_var.lerp_(values[1] * (rows / (rows - 1)), self.momentum)
            self.num_batches_tracked.add_(1)
        return out


def fuse_stem(module: nn.Module) -> None:
    """
    Fuses conv1, bn1, act1 and maxpool of every timm ResNet stem inside module into its bn1, in place, onto the stem
    kernels.

    A stem qualifies with a 7 x 7, stride-2, padding-3, bias-free conv1 from 2 or 3 to 64 channels, a plain
    BatchNorm2d bn1, a ReLU act1 and a 3x3, stride-2, padding-1 maxpool. act1 and maxpool become identities; the
    parameters, buffers and state_dict keys stay the same. bf16 training and a frozen stem in eval run on the kernels;
    other dtypes and eval with gradients run the unfused modules.

    Args:
        module: the module to patch.
    """
    for child in module.children():
        fuse_stem(child)
    conv = getattr(module, "conv1", None)
    norm = getattr(module, "bn1", None)
    act = getattr(module, "act1", None)
    pool = getattr(module, "maxpool", None)
    if not (
        type(conv) is nn.Conv2d
        and type(norm) is nn.BatchNorm2d
        and isinstance(act, nn.ReLU)
        and isinstance(pool, nn.MaxPool2d)
    ):
        return
    if not (
        conv.kernel_size == (7, 7)
        and conv.stride == (2, 2)
        and conv.padding == (3, 3)
        and conv.dilation == (1, 1)
        and conv.groups == 1
        and conv.bias is None
        and conv.padding_mode == "zeros"
        and conv.in_channels in (2, 3)
        and conv.out_channels == 64
    ):
        return
    if not (pool.kernel_size in (3, (3, 3)) and pool.stride in (2, (2, 2)) and pool.padding in (1, (1, 1))):
        return
    if pool.dilation not in (1, (1, 1)) or pool.ceil_mode or norm.momentum is None or not norm.track_running_stats:
        return
    conv.__class__ = CudaStemConv
    norm.__class__ = CudaStemNorm
    # A tuple, not a submodule: conv1 keeps its place in the module tree and the state_dict.
    norm.__dict__["_stem_conv"] = (conv,)
    module.act1 = nn.Identity()
    module.maxpool = nn.Identity()
