# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The kernel check finds every kernel_patches list of a config, zeroes and restores dropouts, compares a patched step
with the plain one within the tolerances, names the patch at fault and leaves the policy as it found it.
"""

from __future__ import annotations

import json
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, PlanningConfig, TransfuserConfig
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training.concurrent_seeds_training import SeedInputs
from py123d_garage.training.kernel_verification import (
    StepOutcome,
    compare,
    kernel_patches_of,
    patches_not_running,
    verify_kernel_patches,
    with_kernel_patches,
    zero_dropouts,
)
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _labels, _navigation, config  # noqa: F401

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel check runs on the GPU")
_FUSE_QKV = "py123d_garage.policy.transfuser.network.encoder.transfuser:fuse_fusion_qkv"
_PERTURB = "tests.unit.training.test_kernel_verification:perturb_backbone"


@dataclass
class _Leaf:
    kernel_patches: list[str] = field(default_factory=list)
    name: str = ""


@dataclass
class _Tree:
    leaves: list[_Leaf] = field(default_factory=list)
    optional: _Leaf | None = None
    kernel_patches: list[str] = field(default_factory=list)


def test_kernel_patches_of_walks_nested_dataclasses_lists_and_optionals() -> None:
    tree = _Tree(
        leaves=[_Leaf(["a:b", "c:d"]), _Leaf([])],
        optional=_Leaf(["c:d", "e:f"]),
        kernel_patches=["g:h"],
    )
    assert kernel_patches_of(tree) == ["a:b", "c:d", "e:f", "g:h"]
    assert kernel_patches_of(_Tree(optional=None)) == []


def test_with_kernel_patches_filters_every_list_on_a_copy() -> None:
    tree = _Tree(leaves=[_Leaf(["a:b", "c:d"])], optional=_Leaf(["c:d"]), kernel_patches=["g:h"])
    cleared = with_kernel_patches(tree, lambda _: False)
    only_cd = with_kernel_patches(tree, lambda patch: patch == "c:d")
    assert kernel_patches_of(cleared) == []
    assert only_cd.leaves[0].kernel_patches == ["c:d"]
    assert only_cd.optional is not None
    assert only_cd.optional.kernel_patches == ["c:d"]
    assert only_cd.kernel_patches == []
    assert kernel_patches_of(tree) == ["a:b", "c:d", "g:h"]


def test_kernel_patches_of_reads_the_training_config() -> None:
    transfuser = TransfuserConfig(
        backbone_config=BackboneConfig(kernel_patches=["x:backbone"]),
        planning_config=PlanningConfig(kernel_patches=["x:planner", "x:backbone"]),
    )
    training_config = TrainingConfig(policy_config=PolicyConfig(transfuser_config=transfuser))
    assert kernel_patches_of(training_config.policy_config) == ["x:backbone", "x:planner"]


class _FloatDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = 0.1
        self.drop_prob = 0.2
        self.p = 0.3


def test_patches_not_running_names_every_requested_patch_whose_kernels_did_not_run() -> None:
    entries = [
        {"patch": "a:runs", "status": "runs"},
        {"patch": "b:torch", "status": "torch only"},
        {"patch": "c:unknown", "status": "unknown to the kernel registry"},
        {
            "patch": "d:shape",
            "status": "does not run: pinned_conv: shape (8, 64) is not in the engine table (12 calls)",
        },
        {"patch": "e:replaced", "status": "does not run: planner_swiglu_linear: never called, f:ffn replaces it"},
        {"patch": "f:hopper", "status": "falls back: swiglu is built on sm_90a instructions; GPU is sm_120"},
        {"patch": "g:partly", "status": "partly runs: fusion_upsample_add: map and grid differ (1 calls)"},
    ]
    assert patches_not_running(entries) == [
        "d:shape: pinned_conv: shape (8, 64) is not in the engine table (12 calls)",
        "e:replaced: planner_swiglu_linear: never called, f:ffn replaces it",
        "f:hopper: swiglu is built on sm_90a instructions; GPU is sm_120",
        "g:partly: fusion_upsample_add: map and grid differ (1 calls)",
    ]


def test_zero_dropouts_zeroes_modules_and_float_attributes_and_restores() -> None:
    module = nn.Sequential(nn.Dropout(0.3), nn.MultiheadAttention(8, 2, dropout=0.2), _FloatDropout())
    dropout = module[0]
    attention = module[1]
    floats = module[2]
    assert isinstance(dropout, nn.Dropout)
    assert isinstance(attention, nn.MultiheadAttention)
    assert isinstance(floats, _FloatDropout)
    restore = zero_dropouts(module)
    assert dropout.p == 0.0
    assert attention.dropout == 0.0
    assert floats.dropout == 0.0
    assert floats.drop_prob == 0.0
    assert floats.p == 0.3
    restore()
    assert dropout.p == 0.3
    assert attention.dropout == 0.2
    assert floats.dropout == 0.1
    assert floats.drop_prob == 0.2


def _outcome(loss: float, gradients: dict[str, list[float]]) -> StepOutcome:
    return StepOutcome(loss, {name: torch.tensor(values) for name, values in gradients.items()}, set())


def test_compare_flags_the_loss_and_the_gradients_outside_the_tolerances() -> None:
    reference = _outcome(
        1.0,
        {"a": [1.0, 0.0], "b": [0.0, 2.0], "c": [1.0, 1.0], "tiny": [1e-4, 0.0], "unused": [0.0]},
    )
    plain = _outcome(
        1.001,
        {"a": [1.01, 0.0], "b": [0.0, 2.02], "c": [1.0, 1.0], "tiny": [1.5e-4, 0.0], "unused": [0.0]},
    )
    close = {"a": [0.985, 0.0], "b": [0.0, 1.98], "c": [1.0, 1.0], "tiny": [1.6e-4, 0.0], "unused": [0.0]}
    same = compare(_outcome(1.005, close), plain, reference, 1e-2, 2.0, 1e-3)
    assert same.failures == []
    assert same.worst_parameter == "a"
    assert same.worst_margin == pytest.approx(0.015 / (2.0 * 0.01 + 1e-3 * 1.0))
    hidden = compare(_outcome(1.005, {**close, "tiny": [3e-4, 0.0]}), plain, reference, 1e-2, 2.0, 1e-3)
    assert [failure.split(":")[0] for failure in hidden.failures] == ["tiny"]
    moved = {"a": [1.0, 0.0], "b": [0.0, 2.2], "d": [1.0], "tiny": [1e-4, 0.0], "unused": [0.0]}
    off = compare(_outcome(1.1, moved), plain, reference, 1e-2, 2.0, 1e-3)
    assert [failure.split(":")[0] for failure in off.failures] == [
        "loss 1.1 with the patches, 1.001 without",
        "b",
        "c",
        "d",
    ]
    assert off.worst_parameter == "b"
    assert "only with no patches and fp32" in off.failures[2]
    without_floor = compare(_outcome(1.0, close), plain, reference, 1e-2, 1.0, 0.0)
    assert [failure.split(":")[0] for failure in without_floor.failures] == ["a", "tiny"]


def test_compare_allows_more_noise_on_a_gradient_below_the_bf16_error() -> None:
    # A key norm's bias: softmax cancels it, so its fp32 gradient is ~0 and both bf16 gradients are rounding noise.
    reference = _outcome(1.0, {"a": [1.0, 0.0], "key_bias": [1e-9, 0.0]})
    plain = _outcome(1.0, {"a": [1.01, 0.0], "key_bias": [3e-6, 0.0]})
    noise = compare(_outcome(1.0, {"a": [1.01, 0.0], "key_bias": [6.3e-6, 0.0]}), plain, reference, 1e-2, 2.0, 2e-2)
    assert noise.failures == []
    broken = compare(_outcome(1.0, {"a": [1.01, 0.0], "key_bias": [5e-5, 0.0]}), plain, reference, 1e-2, 2.0, 2e-2)
    assert [failure.split(":")[0] for failure in broken.failures] == ["key_bias"]
    assert "an unresolved gradient 10" in broken.failures[0]
    resolved = compare(_outcome(1.0, {"a": [1.05, 0.0], "key_bias": [3e-6, 0.0]}), plain, reference, 1e-2, 2.0, 2e-2)
    assert [failure.split(":")[0] for failure in resolved.failures] == ["a"]
    assert "kernel_check_grad_factor 2" in resolved.failures[0]


def perturb_backbone(backbone: nn.Module) -> None:
    """A patch at fault: scales every output of the backbone by five percent."""

    def scale(module: nn.Module, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(value * 1.05 if isinstance(value, torch.Tensor) else value for value in output)

    backbone.register_forward_hook(scale)


def _policy_and_config(
    base_config: TransfuserConfig,
    patches: list[str],
    **training_fields: Any,
) -> tuple[TransfuserPolicy, TrainingConfig]:
    transfuser = TransfuserConfig(
        trajectory_offsets_us=base_config.trajectory_offsets_us,
        required_target_point_distances_m=base_config.required_target_point_distances_m,
        backbone_config=BackboneConfig(image_encoder_pretrained=False, kernel_patches=patches),
        planning_config=base_config.planning_config,
    )
    training_config = TrainingConfig(
        policy_config=PolicyConfig(transfuser_config=transfuser),
        offline_data_sources=[],
        compile_mode=None,
        **training_fields,
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(training_config.policy_config)
    policy.initialize()
    policy.prepare_training()
    policy = policy.to(memory_format=torch.channels_last).cuda().train()  # pyright: ignore[reportCallIssue]
    return policy, training_config


def _autocast() -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], torch.autocast("cuda", dtype=torch.bfloat16))


def _inputs(training_config: TrainingConfig) -> SeedInputs:
    transfuser = training_config.policy_config.transfuser_config
    assert transfuser is not None
    torch.manual_seed(1)
    return (
        _features(transfuser, 2).to("cuda"),
        _labels(transfuser, batch_size=2).to("cuda"),
        _navigation(transfuser, 2).to("cuda"),
    )


def _check(
    policy: TransfuserPolicy,
    training_config: TrainingConfig,
    report_file: Path,
    inputs: SeedInputs | None = None,
) -> dict[str, Any]:
    if inputs is None:
        inputs = _inputs(training_config)
    return verify_kernel_patches(
        policy,
        training_config,
        inputs,
        policy.loss_weights(0),
        _autocast,
        torch.device("cuda", 0),
        report_file,
    )


@_needs_gpu
def test_a_torch_only_patch_verifies_and_leaves_the_policy_as_found(config: TransfuserConfig, tmp_path: Path) -> None:  # noqa: F811
    policy, training_config = _policy_and_config(config, [_FUSE_QKV])
    inputs = _inputs(training_config)
    buffers = {name: buffer.clone() for name, buffer in policy.named_buffers()}
    cuda_rng, cpu_rng = torch.cuda.get_rng_state(), torch.random.get_rng_state()
    report_file = tmp_path / "kernel_report.json"

    report = _check(policy, training_config, report_file, inputs)

    assert report["status"] == "verified"
    assert report["failures"] == []
    assert json.loads(report_file.read_text())["patches"][0]["patch"] == _FUSE_QKV
    assert report["gradients"]["worst_margin"] < 1.0
    for name, buffer in policy.named_buffers():
        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(module.p == 0.1 for module in policy.modules() if isinstance(module, nn.Dropout) and module.p)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)


@_needs_gpu
def test_a_patch_at_fault_is_named_and_refused(config: TransfuserConfig, tmp_path: Path) -> None:  # noqa: F811
    policy, training_config = _policy_and_config(config, [_FUSE_QKV, _PERTURB])
    report_file = tmp_path / "kernel_report.json"

    with pytest.raises(
        RuntimeError,
        match="patches at fault on their own: \\['" + _PERTURB.replace(".", "\\.") + "'\\]",
    ):
        _check(policy, training_config, report_file)

    report = json.loads(report_file.read_text())
    assert report["status"] == "mismatch"
    assert report["failures"]
    assert {entry["patch"]: bool(entry["failures"]) for entry in report["attribution"]} == {
        _FUSE_QKV: False,
        _PERTURB: True,
    }


@_needs_gpu
def test_a_mismatch_only_warns_with_refuse_off(
    config: TransfuserConfig,  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy, training_config = _policy_and_config(config, [_PERTURB], kernel_check_refuse=False)
    with caplog.at_level(logging.WARNING):
        report = _check(policy, training_config, tmp_path / "kernel_report.json")
    assert report["status"] == "mismatch"
    assert "training anyway" in caplog.text
