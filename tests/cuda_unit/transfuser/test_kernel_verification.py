# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The startup kernel check passes a TransFuser policy with the kernel patches of the training scripts that run on this GPU
and batch, refuses a requested patch whose kernels do not run and says why, and leaves the policy as it found it; the
patches add no torch.compile graph breaks.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import PATCHES, unsupported_reason

from py123d_garage.common import runtime_typing
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    BevSemanticConfig,
    BoxDetectionConfig,
    TransfuserConfig,
)
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training.concurrent_seeds_training import SeedInputs, policy_forward_and_loss
from py123d_garage.training.kernel_verification import kernel_patches_of, verify_kernel_patches, with_kernel_patches
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _labels, _navigation, config  # noqa: F401

_CUDA = "py123d_garage_cuda.transfuser"
_ENCODER = "py123d_garage.policy.transfuser.network.encoder.transfuser"
_PINNED = f"{_CUDA}.resnet:pin_conv_engines"
_SWIGLU = f"{_CUDA}.planner:patch_planner_swiglu"
_FEED_FORWARD = f"{_CUDA}.planner:patch_planner_feed_forward"
BACKBONE_PATCHES = [
    f"{_CUDA}.resnet:fuse_stem",
    _PINNED,
    f"{_CUDA}.linear_convs:patch_pointwise_convs",
    f"{_ENCODER}:fuse_fusion_qkv",
    f"{_CUDA}.fusion:patch_fusion_upsample_add",
]
HEAD_PATCHES = [_PINNED, f"{_CUDA}.linear_convs:patch_pointwise_convs"]
PLANNER_PATCHES = [
    f"{_CUDA}.planner:patch_planner_attention",
    _FEED_FORWARD,
    f"{_CUDA}.adaptive_norm:patch_planner_adaptive_norm",
]


def _autocast() -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], torch.autocast("cuda", dtype=torch.bfloat16))


def _runs_here(patch: str) -> bool:
    """Whether this GPU and build run every kernel family of a patch; a torch-only patch always runs."""
    return all(unsupported_reason(family) is None for family in PATCHES.get(patch, ()))


def test_every_training_script_patch_is_registered() -> None:
    cuda_patches = [p for p in BACKBONE_PATCHES + HEAD_PATCHES + PLANNER_PATCHES + [_SWIGLU] if p.startswith(_CUDA)]
    assert set(cuda_patches) <= set(PATCHES)


def _patched(
    geometry: TransfuserConfig,
    backbone: list[str],
    heads: list[str],
    planner: list[str],
    require_patches: bool = True,
) -> tuple[TrainingConfig, TransfuserPolicy, SeedInputs]:
    """
    Builds policy 0 with the given kernel patches and one batch of 2 for it.

    Args:
        geometry: the small test geometry.
        backbone: the backbone's patches.
        heads: the BEV semantic and box detection heads' patches.
        planner: the planning decoder's patches.
        require_patches: refuse requested patches whose kernels do not run.

    Returns:
        the training config, the patched policy on the GPU and its inputs.
    """
    transfuser = TransfuserConfig(
        trajectory_offsets_us=geometry.trajectory_offsets_us,
        required_target_point_distances_m=geometry.required_target_point_distances_m,
        backbone_config=BackboneConfig(image_encoder_pretrained=False, kernel_patches=backbone),
        bev_semantic_config=BevSemanticConfig(kernel_patches=heads),
        box_detection_config=BoxDetectionConfig(kernel_patches=heads),
        planning_config=geometry.planning_config,
    )
    transfuser.planning_config.kernel_patches = planner
    training_config = TrainingConfig(
        policy_config=PolicyConfig(transfuser_config=transfuser),
        offline_data_sources=[],
        compile_mode=None,
        kernel_check_require_patches=require_patches,
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(training_config.policy_config)
    policy.initialize()
    policy.prepare_training()
    policy = policy.to(memory_format=torch.channels_last).cuda().train()  # pyright: ignore[reportCallIssue]
    torch.manual_seed(1)
    inputs = (
        _features(transfuser, 2).to("cuda"),
        _labels(transfuser, batch_size=2).to("cuda"),
        _navigation(transfuser, 2).to("cuda"),
    )
    return training_config, policy, inputs


def _verify(
    training_config: TrainingConfig,
    policy: TransfuserPolicy,
    inputs: SeedInputs,
    report_file: Path,
) -> dict[str, Any]:
    return verify_kernel_patches(
        policy,
        training_config,
        inputs,
        policy.loss_weights(0),
        _autocast,
        torch.device("cuda", 0),
        report_file,
    )


def test_the_startup_check_passes_with_every_patch_that_runs_here(config: TransfuserConfig, tmp_path: Path) -> None:  # noqa: F811
    # The pinned engine table holds the training geometry at batch 32, so the test's batch runs plain convolutions.
    backbone, heads, planner = (
        [patch for patch in patches if patch != _PINNED]
        for patches in (BACKBONE_PATCHES, HEAD_PATCHES, PLANNER_PATCHES)
    )
    # A warn-only check finds the patches this GPU rules out, including by the shared memory their kernels need for
    # this geometry, which the registry knows only per call: the planner attention's backward fits an H100, not an
    # RTX 5090. Every reason names the GPU; a patch idle on the batch for any other reason fails here.
    training_config, policy, inputs = _patched(config, backbone, heads, planner, require_patches=False)
    ruled_out = _verify(training_config, policy, inputs, tmp_path / "survey.json")["not_running"]
    assert all(torch.cuda.get_device_properties(0).name in reason for reason in ruled_out), ruled_out
    ruled_out_patches = {reason.split(": ")[0] for reason in ruled_out}
    backbone, heads, planner = (
        [patch for patch in patches if patch not in ruled_out_patches] for patches in (backbone, heads, planner)
    )
    del policy
    training_config, policy, inputs = _patched(config, backbone, heads, planner)
    buffers = {name: buffer.clone() for name, buffer in policy.named_buffers()}
    report_file = tmp_path / "kernel_report.json"

    report = _verify(training_config, policy, inputs, report_file)

    assert report["status"] == "verified", report["failures"]
    assert report["not_running"] == []
    assert [entry["patch"] for entry in report["patches"]] == kernel_patches_of(training_config.policy_config)
    for entry in json.loads(report_file.read_text())["patches"]:
        assert entry["status"] in ("runs", "torch only", "unknown to the kernel registry"), entry
    for name, buffer in policy.named_buffers():
        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_a_requested_patch_whose_kernels_do_not_run_is_refused_with_why(
    config: TransfuserConfig,  # noqa: F811
    tmp_path: Path,
) -> None:
    if not _runs_here(_PINNED):
        pytest.skip("the pinned convolutions do not run on this GPU")
    training_config, policy, inputs = _patched(config, [_PINNED], [], [])

    with pytest.raises(RuntimeError, match="fall back to torch") as refused:
        _verify(training_config, policy, inputs, tmp_path / "kernel_report.json")

    assert f"{_PINNED}: pinned_conv: shape (N, C, H, W, K, R, S, stride, padding) (2, " in str(refused.value)
    assert "is not in the engine table" in str(refused.value)


def test_a_replaced_patch_is_refused_and_names_the_patch_that_replaces_it(
    config: TransfuserConfig,  # noqa: F811
    tmp_path: Path,
) -> None:
    if not (_runs_here(_SWIGLU) and _runs_here(_FEED_FORWARD)):
        pytest.skip("the planner feed-forward kernels do not run on this GPU")
    training_config, policy, inputs = _patched(config, [], [], [_FEED_FORWARD, _SWIGLU])

    with pytest.raises(RuntimeError, match="fall back to torch") as refused:
        _verify(training_config, policy, inputs, tmp_path / "kernel_report.json")

    assert f"{_SWIGLU}: planner_swiglu_linear: never called, {_FEED_FORWARD} replaces it" in str(refused.value)


def test_without_require_patches_the_check_warns_and_trains_on(
    config: TransfuserConfig,  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    training_config, policy, inputs = _patched(config, [_PINNED], [], [], require_patches=False)

    with caplog.at_level(logging.WARNING):
        report = _verify(training_config, policy, inputs, tmp_path / "kernel_report.json")

    assert [entry.split(": ")[0] for entry in report["not_running"]] == [_PINNED]
    assert "kernel_check_require_patches is off" in caplog.text


def test_the_patches_add_no_graph_breaks(config: TransfuserConfig) -> None:  # noqa: F811
    # Dynamo cannot trace the beartype wrappers of the jaxtyping import hook, so under the hook the test reruns itself
    # in a subprocess with the hook off, as test_forward_traces_under_torch_compile does.
    if runtime_typing.hook_manager is not None:
        subprocess.run(
            [sys.executable, "-m", "pytest", f"{__file__}::{test_the_patches_add_no_graph_breaks.__name__}", "-q"],
            check=True,
            env=os.environ | {"PY123D_GARAGE_RUNTIME_TYPE_CHECKING": "false"},
        )
        return
    training_config, patched, inputs = _patched(config, BACKBONE_PATCHES, HEAD_PATCHES, [*PLANNER_PATCHES, _SWIGLU])
    plain = TransfuserPolicy(with_kernel_patches(training_config.policy_config, lambda _: False))
    plain.initialize()
    plain.prepare_training()
    plain = plain.to(memory_format=torch.channels_last).cuda().train()  # pyright: ignore[reportCallIssue]
    features, labels, navigation = inputs

    def graph_breaks(policy: TransfuserPolicy) -> tuple[int, list[str]]:
        def step() -> torch.Tensor:
            with _autocast():
                loss_weights = policy.loss_weights(0)
                return policy_forward_and_loss(policy, features, labels, navigation, loss_weights, True)[0]

        torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]
        # torch.compile's explain(fn)(*args) form, which the stubs still type as the older explain(fn, *args).
        explain = cast("Callable[[], Any]", torch._dynamo.explain(step))  # pyright: ignore[reportPrivateUsage]
        explanation = explain()
        return explanation.graph_break_count, [str(reason.reason) for reason in explanation.break_reasons]

    patched_count, patched_reasons = graph_breaks(patched)
    plain_count, _ = graph_breaks(plain)
    assert patched_count == plain_count, patched_reasons
