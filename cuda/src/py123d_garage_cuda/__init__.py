# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Opt-in CUDA kernels of py123d_garage, compiled at install time against the installed torch.

The modules under transfuser/ hold the ops of one network part each, with their fake implementations, autograd
formulas and the patches that swap them into models; transfuser/registry.py holds what every kernel family needs
from the GPU. The sources under transfuser/csrc/ are grouped by the oldest architecture they are built for: sm80
runs on any sm_80 or newer GPU, sm90 on sm_90 and newer, sm90a only on sm_90.
"""

from __future__ import annotations

import torch

try:
    from py123d_garage_cuda import _C  # pyright: ignore[reportAttributeAccessIssue]
except ImportError as error:
    raise ImportError(
        f"py123d_garage_cuda._C does not load against torch {torch.__version__}; it was probably built against "
        "another torch. Rebuild: pip install --no-build-isolation --force-reinstall ./cuda",
    ) from error

from py123d_garage_cuda.transfuser.adaptive_norm import planner_adaptive_norm
from py123d_garage_cuda.transfuser.convnext import depthwise_conv
from py123d_garage_cuda.transfuser.fusion import fusion_upsample_add
from py123d_garage_cuda.transfuser.planner import planner_attention, planner_swiglu_feed_forward, planner_swiglu_linear
from py123d_garage_cuda.transfuser.regnet import bn_relu_grouped_conv, grouped_conv
from py123d_garage_cuda.transfuser.resnet import batch_norm_finalize, pinned_conv, resnet_stem_conv
from py123d_garage_cuda.transfuser.stem_pool import resnet_stem_pool

__all__ = [
    "batch_norm_finalize",
    "bn_relu_grouped_conv",
    "depthwise_conv",
    "fusion_upsample_add",
    "grouped_conv",
    "pinned_conv",
    "planner_adaptive_norm",
    "planner_attention",
    "planner_swiglu_feed_forward",
    "planner_swiglu_linear",
    "resnet_stem_conv",
    "resnet_stem_pool",
]
