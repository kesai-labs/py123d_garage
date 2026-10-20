# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
What every TransFuser kernel needs from the GPU and its inputs, and the supports() check the Python wrappers run before
launching a kernel, falling back to torch where it fails.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

try:
    from py123d_garage_cuda._build import ARCHITECTURES as _BUILT_ARCHITECTURES
except ImportError:  # a source tree setup.py has not built yet
    _BUILT_ARCHITECTURES: tuple[str, ...] = ()
# Read once at import, like the architectures: torch.compile cannot trace the call in a patched module's forward.
_CUDNN_VERSION = torch.backends.cudnn.version()


@dataclass(frozen=True)
class OpSupport:
    """The requirements of one kernel family: a forward op with its backward and helper ops."""

    ops: tuple[str, ...]
    capability: tuple[int, int]
    sm90a_only: bool
    shared_memory_bytes: int
    dtypes: frozenset[torch.dtype]
    channels_last: bool
    # The cuDNN version the family was tuned for; None runs on any.
    cudnn_version: int | None = None


OPS: dict[str, OpSupport] = {
    "fusion_upsample_add": OpSupport(
        ops=("fusion_upsample_add", "fusion_upsample_add_backward"),
        capability=(8, 0),
        sm90a_only=False,
        shared_memory_bytes=0,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=True,
    ),
    "resnet_stem_pool": OpSupport(
        ops=("resnet_stem_pool", "resnet_stem_pool_backward"),
        capability=(8, 0),
        sm90a_only=False,
        shared_memory_bytes=0,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=True,
    ),
    "planner_adaptive_norm": OpSupport(
        ops=("planner_adaptive_norm", "planner_adaptive_norm_backward"),
        capability=(8, 0),
        sm90a_only=False,
        shared_memory_bytes=61440,
        dtypes=frozenset({torch.float32, torch.bfloat16}),
        channels_last=False,
    ),
    "depthwise_conv": OpSupport(
        ops=("depthwise_conv", "depthwise_conv_backward"),
        capability=(8, 0),
        sm90a_only=False,
        shared_memory_bytes=93200,
        dtypes=frozenset({torch.float32, torch.bfloat16}),
        channels_last=True,
    ),
    "resnet_stem_conv": OpSupport(
        ops=("resnet_stem_conv", "resnet_stem_conv_weight_grad", "batch_norm_finalize"),
        capability=(9, 0),
        sm90a_only=False,
        shared_memory_bytes=96224,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=True,
    ),
    "pinned_conv": OpSupport(
        ops=("pinned_conv", "pinned_conv_data_grad", "pinned_conv_weight_grad"),
        capability=(9, 0),
        sm90a_only=True,
        shared_memory_bytes=231472,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=True,
        cudnn_version=91002,
    ),
    "regnet_grouped_conv": OpSupport(
        ops=("regnet_grouped_conv", "regnet_grouped_conv_backward"),
        capability=(9, 0),
        sm90a_only=False,
        shared_memory_bytes=117656,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=True,
    ),
    "regnet_bn_relu_grouped_conv": OpSupport(
        ops=("regnet_bn_relu_grouped_conv", "regnet_bn_relu_grouped_conv_backward"),
        capability=(9, 0),
        sm90a_only=False,
        shared_memory_bytes=109584,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=True,
    ),
    "planner_attention": OpSupport(
        ops=(
            "planner_attention",
            "planner_attention_backward",
            "planner_attention_backward_shared_bytes",
            "planner_dropout_keep",
        ),
        capability=(8, 0),
        sm90a_only=False,
        shared_memory_bytes=46080,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=False,
    ),
    "planner_swiglu_linear": OpSupport(
        ops=("planner_swiglu_linear", "planner_swiglu_linear_backward"),
        capability=(9, 0),
        sm90a_only=True,
        shared_memory_bytes=197744,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=False,
    ),
    "planner_swiglu_feed_forward": OpSupport(
        ops=("planner_swiglu_feed_forward", "planner_swiglu_feed_forward_backward"),
        capability=(9, 0),
        sm90a_only=True,
        shared_memory_bytes=203408,
        dtypes=frozenset({torch.bfloat16}),
        channels_last=False,
    ),
}

PATCHES: dict[str, tuple[str, ...]] = {
    "py123d_garage_cuda.transfuser.fusion:patch_fusion_upsample_add": ("fusion_upsample_add",),
    "py123d_garage_cuda.transfuser.linear_convs:patch_patchify_stems": (),
    "py123d_garage_cuda.transfuser.linear_convs:patch_pointwise_convs": (),
    "py123d_garage_cuda.transfuser.adaptive_norm:patch_planner_adaptive_norm": ("planner_adaptive_norm",),
    "py123d_garage_cuda.transfuser.convnext:patch_depthwise_convs": ("depthwise_conv",),
    "py123d_garage_cuda.transfuser.planner:patch_planner_attention": ("planner_attention",),
    "py123d_garage_cuda.transfuser.planner:patch_planner_swiglu": ("planner_swiglu_linear",),
    "py123d_garage_cuda.transfuser.planner:patch_planner_feed_forward": ("planner_swiglu_feed_forward",),
    "py123d_garage_cuda.transfuser.regnet:patch_grouped_convs": ("regnet_grouped_conv",),
    "py123d_garage_cuda.transfuser.regnet:patch_regnet_blocks": ("regnet_bn_relu_grouped_conv",),
    "py123d_garage_cuda.transfuser.resnet:fuse_stem": ("resnet_stem_conv", "resnet_stem_pool"),
    "py123d_garage_cuda.transfuser.resnet:pin_conv_engines": ("pinned_conv",),
}

# Patches whose kernels take over the modules of another patch when both apply, so the other's kernels never run.
REPLACES: dict[str, str] = {
    "py123d_garage_cuda.transfuser.planner:patch_planner_feed_forward": (
        "py123d_garage_cuda.transfuser.planner:patch_planner_swiglu"
    ),
}

_recorded_fallbacks: dict[str, dict[str, int]] | None = None


def built_architectures() -> tuple[str, ...]:
    """
    Reads the TORCH_CUDA_ARCH_LIST entries setup.py built the package for.

    Returns:
        the entries, such as "9.0a" or "8.0+PTX".
    """
    # Read once at import: an import inside a call breaks the torch.compile graph of every patched module's forward.
    return tuple(_BUILT_ARCHITECTURES)


def _built_for(architecture: str, capability: tuple[int, int]) -> bool:
    """
    Decides whether the code built for one TORCH_CUDA_ARCH_LIST entry runs on a compute capability.

    Args:
        architecture: the entry, such as "9.0a", "12.0" or "8.0+PTX".
        capability: the device's (major, minor).

    Returns:
        whether its cubin runs there, or its PTX compiles there.
    """
    base, plus, _ = architecture.partition("+")
    specific = base.endswith("a")
    major, minor = (int(part) for part in base.rstrip("a").split("."))
    if specific:
        return (major, minor) == capability
    if major == capability[0] and minor <= capability[1]:
        return True
    return bool(plus) and (major, minor) <= capability


def family_of(op: str) -> str:
    """
    Names the registry entry an op belongs to.

    Args:
        op: an op name of torch.ops.py123d_garage_cuda.

    Returns:
        the key of OPS whose ops hold op.
    """
    for name, entry in OPS.items():
        if op in entry.ops:
            return name
    raise KeyError(f"{op} is not in the kernel registry")


def unsupported_reason(
    op: str,
    device: torch.device | int | None = None,
    shared_memory_bytes: int = 0,
) -> str | None:
    """
    Explains why a device cannot run an op, or returns None where it can.

    Args:
        op: an op name of torch.ops.py123d_garage_cuda.
        device: the CUDA device; None takes the current one.
        shared_memory_bytes: the dynamic shared memory this call needs when it depends on the shape; the registry's
            entry covers the rest.

    Returns:
        the reason, naming the op and the device, or None.
    """
    entry = OPS[family_of(op)]
    if isinstance(device, torch.device) and device.type != "cuda":
        return f"{op} runs on CUDA devices only; the input is on {device}"
    properties = torch.cuda.get_device_properties(device)
    capability = (properties.major, properties.minor)
    architectures = built_architectures()
    if not any(_built_for(architecture, capability) for architecture in architectures):
        return (
            f"{op}: py123d_garage_cuda was built for {', '.join(architectures)}, not for {properties.name} "
            f"(sm_{properties.major}{properties.minor}); rebuild with TORCH_CUDA_ARCH_LIST={properties.major}.{properties.minor}"
        )
    if entry.sm90a_only and capability != (9, 0):
        return f"{op} is built on sm_90a instructions; {properties.name} is sm_{properties.major}{properties.minor}"
    if capability < entry.capability:
        return (
            f"{op} needs compute capability {entry.capability[0]}.{entry.capability[1]}; "
            f"{properties.name} has {properties.major}.{properties.minor}"
        )
    if entry.cudnn_version is not None and entry.cudnn_version != _CUDNN_VERSION:
        return f"{op} is tuned for cuDNN {entry.cudnn_version}; torch loads cuDNN {_CUDNN_VERSION}"
    needed = max(entry.shared_memory_bytes, shared_memory_bytes)
    if needed > properties.shared_memory_per_block_optin:
        return (
            f"{op} needs {needed} bytes of shared memory per block; "
            f"{properties.name} allows {properties.shared_memory_per_block_optin}"
        )
    return None


def supports(op: str, device: torch.device | int | None = None, shared_memory_bytes: int = 0) -> bool:
    """
    Checks whether a device runs an op.

    Args:
        op: an op name of torch.ops.py123d_garage_cuda.
        device: the CUDA device; None takes the current one.
        shared_memory_bytes: the dynamic shared memory this call needs when it depends on the shape.

    Returns:
        whether the device runs the op.
    """
    return unsupported_reason(op, device, shared_memory_bytes) is None


@contextmanager
def recording_fallbacks() -> Iterator[dict[str, dict[str, int]]]:
    """
    Records, while active, every call of a patched module that ran torch instead of its kernels, and why.

    Yields:
        per kernel family, the number of such calls for each reason.
    """
    global _recorded_fallbacks
    previous = _recorded_fallbacks
    _recorded_fallbacks = {}
    try:
        yield _recorded_fallbacks
    finally:
        _recorded_fallbacks = previous


def note_fallback(family: str, reason: str | Callable[[], str]) -> None:
    """
    Records one call of a patched module that runs torch instead of the kernels of a family; a no-op unless
    recording_fallbacks is active, so compiled training traces nothing here.

    Args:
        family: the key of OPS.
        reason: why, or a function returning why, called only while recording.
    """
    if _recorded_fallbacks is None:
        return
    text = reason() if callable(reason) else reason
    counts = _recorded_fallbacks.setdefault(family, {})
    counts[text] = counts.get(text, 0) + 1
