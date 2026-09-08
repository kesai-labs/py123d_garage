# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

import math
from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import v2

# Ranges of the equivalent CPU pipeline.
_NOISE_STD_RANGE = (0.0, 0.05)
_PIXEL_DROPOUT_RATE = 0.05
_GAIN_RANGE = (1.0 / 1.2, 1.2)
_CONTRAST_RANGE = (1.0 / 1.2, 1.2)
_BLUR_SIGMA_RANGE = (0.01, 1.0)
# Elastic displacement magnitude and smoothing, in pixels.
_ELASTIC_ALPHA = 1.0
_ELASTIC_SIGMA = 0.25


def _elastic_deform(
    x: jt.Float[torch.Tensor, "n 3 height width"],
) -> jt.Float[torch.Tensor, "n 3 height width"]:
    """
    Warps each image by its own smoothed random displacement field.

    Args:
        x: the samples selected for deformation, on device.

    Returns:
        the warped samples, same dtype and device.
    """
    count, _, height, width = x.shape
    device, dtype = x.device, x.dtype

    kernel_size = int(8 * _ELASTIC_SIGMA + 1)
    kernel_size += kernel_size % 2 == 0
    field = torch.rand(2 * count, 1, height, width, device=device, dtype=dtype) * 2 - 1
    if _ELASTIC_SIGMA > 0.0:
        field = v2.functional.gaussian_blur(
            field,
            kernel_size=[kernel_size, kernel_size],
            sigma=[_ELASTIC_SIGMA, _ELASTIC_SIGMA],
        )
    displacement = torch.cat(
        [
            field[:count] * _ELASTIC_ALPHA / width,
            field[count:] * _ELASTIC_ALPHA / height,
        ],
        dim=1,
    ).permute(0, 2, 3, 1)

    grid = torch.empty(1, height, width, 2, device=device, dtype=dtype)
    grid[..., 0] = torch.linspace(
        (-width + 1) / width,
        (width - 1) / width,
        width,
        device=device,
        dtype=dtype,
    )
    grid[..., 1] = torch.linspace(
        (-height + 1) / height,
        (height - 1) / height,
        height,
        device=device,
        dtype=dtype,
    ).unsqueeze(-1)
    return F.grid_sample(
        x,
        grid + displacement,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def augment_rgb_batch(
    rgb: jt.Shaped[torch.Tensor, "batch 3 height width"],
    probability: float,
) -> jt.Shaped[torch.Tensor, "batch 3 height width"]:
    """
    Applies the colour augmentation pipeline to a collated [0, 255] image batch.

    Args:
        rgb: batched channel-first camera images, on any device.
        probability: chance of each augmentation op applying, per sample.

    Returns:
        the augmented batch, same dtype, device and layout.
    """
    batch_size = rgb.shape[0]
    device = rgb.device
    x = rgb.to(torch.float32)

    def gate() -> jt.Bool[torch.Tensor, "batch 1 1 1"]:
        # CPU-decided mask, so no device sync
        return (torch.rand(batch_size) < probability).view(-1, 1, 1, 1).to(device)

    # Gates fold into the ops' coefficients: where(g, x * a, x) == x * (g ? a : 1).

    # Additive Gaussian noise, std drawn per sample.
    std = torch.rand(batch_size, 1, 1, 1, device=device)
    std = _NOISE_STD_RANGE[0] + (_NOISE_STD_RANGE[1] - _NOISE_STD_RANGE[0]) * std
    std = torch.where(gate(), std * 255.0, torch.zeros_like(std))
    x = x.addcmul_(torch.randn_like(x), std)

    # Per-pixel, per-channel dropout to black.
    dropped = torch.rand_like(x) < _PIXEL_DROPOUT_RATE
    x = x.masked_fill_(gate() & dropped, 0.0)

    # Gain and contrast composed into one affine.
    gain = torch.empty(batch_size, 3, 1, 1, device=device).uniform_(
        *_GAIN_RANGE,
    )
    gain = torch.where(gate(), gain, torch.ones_like(gain))
    contrast = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        *_CONTRAST_RANGE,
    )
    contrast = torch.where(gate(), contrast, torch.ones_like(contrast))
    x = x.mul_(gain * contrast).add_(cast(Tensor, 127.5 * (1.0 - contrast)))

    # Per-sample blur; torchvision v2 has no batched form with per-sample sigma.
    blur_gate = torch.rand(batch_size) < probability
    blur_sigmas = torch.empty(batch_size).uniform_(*_BLUR_SIGMA_RANGE)
    for index in range(batch_size):
        if blur_gate[index]:
            sigma = float(blur_sigmas[index])
            kernel_size = 2 * math.ceil(3.0 * sigma) + 1
            x[index] = v2.functional.gaussian_blur(
                x[index],
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma, sigma],
            )

    elastic_gate = torch.nonzero(torch.rand(batch_size) < probability).flatten()
    if elastic_gate.numel() > 0:
        elastic_index = elastic_gate.to(device)
        x[elastic_index] = _elastic_deform(x[elastic_index])

    return x.round_().clamp_(0.0, 255.0).to(rgb.dtype)
