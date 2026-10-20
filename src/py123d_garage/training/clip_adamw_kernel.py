# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Triton kernel of FlatAdamW that applies the clip scale, AdamW moments, weight decay and update in one pass.

The hyperparameters come from a device tensor, so a recorded launch replays with the current schedule.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import functools
from typing import Any


@functools.cache
def clip_adamw_kernel() -> Any:
    """
    Build the fused clip and AdamW kernel on first use, so the package imports without triton.

    Returns:
        the Triton kernel
    """
    import triton
    import triton.language as tl

    @triton.jit
    def clip_adamw(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        scale_ptr,
        hyperparameters_ptr,
        n_elements,
        beta1,
        beta2,
        eps,
        weight_decay,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        lr = tl.load(hyperparameters_ptr)
        step_size = tl.load(hyperparameters_ptr + 1)
        bias2_sqrt = tl.load(hyperparameters_ptr + 2)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0) * tl.load(scale_ptr)
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        p = p * (1.0 - lr * weight_decay) - step_size * m / (tl.sqrt(v) / bias2_sqrt + eps)
        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)

    return clip_adamw
